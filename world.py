"""Read-only world scenes share local geometry, material graphs, images and placement prototypes."""

import json
import math
import os
from pathlib import Path
import struct
import time

import bpy
import numpy as np
from bpy.props import EnumProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion

from . import bundle, effects, world_lights


MANIFEST = 'world.sunrise-scene.json'
MAX_COUNT = 2_000_000
STRIDE = 128
# One placement row; it fills STRIDE bytes.
RECORD = '<II16fIIiIiIQIIiiII'
# EEVEE shadow pool for a map; the 512 MB default overflows and drops pages as hard light bands.
SHADOW_POOL_MB = 1024


def _count(value, maximum=MAX_COUNT):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise ValueError('World scene count is invalid.')
    return value


def _rows(document, name):
    result = document.get(name)
    if not isinstance(result, list) or len(result) > MAX_COUNT:
        raise ValueError('World scene table is invalid: ' + name)
    return result


class _Buffers:
    """Views read directly into typed arrays; a shard is hashed once without retaining its bytes."""
    def __init__(self, root, descriptors):
        self.root, self.descriptors = root, descriptors
        self.files = {}

    def close(self):
        for stream in self.files.values():
            stream.close()
        self.files.clear()

    def view(self, view, kind, width, count=None):
        if view.get('ComponentType') != kind or view.get('Components') != width:
            raise ValueError('World buffer component layout differs.')
        rows = _count(view.get('Count'), bundle.MAX_BYTES)
        if count is not None and rows != count:
            raise ValueError('World attribute count differs from its vertices.')
        index = _count(view.get('Buffer'), len(self.descriptors) - 1)
        item = self.descriptors[index]
        length = _count(item.get('Length'), bundle.MAX_BYTES)
        offset = _count(view.get('Offset'), length)
        size = 1 if kind == 'UInt8' else 4
        amount = rows * width * size
        if amount > length - offset or offset % size:
            raise ValueError('World buffer view leaves its shard.')
        if index not in self.files:
            path = bundle._path(self.root, item['Path'])
            if path.stat().st_size != length or bundle._sha(path) != item['Sha256']:
                raise ValueError('World binary shard differs from its manifest.')
            self.files[index] = path.open('rb')
        stream = self.files[index]
        stream.seek(offset)
        values = np.fromfile(stream, dtype={'Float32': '<f4', 'UInt32': '<u4', 'UInt8': 'u1'}[kind], count=rows * width)
        if len(values) != rows * width or kind == 'Float32' and not np.isfinite(values).all():
            raise ValueError('World buffer is truncated or contains nonfinite values.')
        return values


def _matrix(values):
    value = bundle._matrix(values)
    if any(abs(value[3][i]) > 1e-6 for i in range(3)) or abs(value[3][3] - 1) > 1e-6:
        raise ValueError('World placement must use an affine matrix.')
    return value


def _place(obj, parent, matrix):
    """A parent-inverse matrix preserves shear that Blender's location/rotation/scale cannot hold."""
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = matrix
    if not bundle._close(bundle._flat(obj.matrix_basis), bundle._flat(matrix)):
        obj.matrix_basis = Matrix.Identity(4)
        obj.matrix_parent_inverse = matrix


def _mesh(info, buffers, needs_basis):
    count = _count(info['Positions']['Count'])
    positions = buffers.view(info['Positions'], 'Float32', 3)
    normals = buffers.view(info['Normals'], 'Float32', 3, count).reshape((-1, 3))
    uvs = buffers.view(info['TexCoords'], 'Float32', 2, count).reshape((-1, 2))
    indices = buffers.view(info['Indices'], 'UInt32', 1)
    if count == 0 or len(indices) == 0 or len(indices) % 3 or int(indices.max()) >= count:
        raise ValueError('World triangle indices are invalid.')
    triangles = indices.reshape((-1, 3))
    valid = (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2]) & (triangles[:, 2] != triangles[:, 0])
    degenerate = int(len(triangles) - np.count_nonzero(valid))
    if degenerate:
        indices = triangles[valid].reshape(-1)
    mesh = bpy.data.meshes.new(f'World geometry {info["Id"]}')
    mesh.vertices.add(count)
    mesh.loops.add(len(indices))
    mesh.polygons.add(len(indices) // 3)
    mesh.vertices.foreach_set('co', positions)
    mesh.loops.foreach_set('vertex_index', indices)
    mesh.polygons.foreach_set('loop_start', np.arange(0, len(indices), 3, dtype=np.int32))
    mesh.polygons.foreach_set('loop_total', np.full(len(indices) // 3, 3, dtype=np.int32))
    mesh.polygons.foreach_set('use_smooth', np.ones(len(indices) // 3, dtype=np.bool_))
    mesh.update(calc_edges=True)
    uv = mesh.uv_layers.new(name='UVMap')
    uv.data.foreach_set('uv', uvs[indices].reshape(-1))
    colors = buffers.view(info['Colors'], 'UInt8', 4, count) / 255 if info.get('Colors') else None
    zw = buffers.view(info['TexCoordsZw'], 'Float32', 2, count) if info.get('TexCoordsZw') else None
    bundle._vertex_streams(mesh, colors, zw)
    if len(indices):
        mesh.normals_split_custom_set_from_vertices(normals)
    bundle._bitangent_sign(mesh)
    if needs_basis:
        tangents = buffers.view(info['Tangents'], 'Float32', 4, count).reshape((-1, 4)) if info.get('Tangents') else None
        if tangents is not None:
            for name, values in (('normal', normals), ('tangent', tangents[:, :3]),
                                 ('bitangent', np.cross(normals, tangents[:, :3]) * tangents[:, 3:4])):
                attribute = mesh.attributes.new('sunrise_native_' + name, 'FLOAT_VECTOR', 'POINT')
                attribute.data.foreach_set('vector', np.ascontiguousarray(values).reshape(-1))
    mesh['sunrise_world_geometry'] = info['Id']
    mesh['sunrise_degenerate_triangles'] = degenerate
    return mesh


def _reuse(built, key, info, slots):
    """A copy of a graph built for other slot textures, with each slot's texture swapped; None when none fits."""
    source, source_slots, textured = built.get(key, (None, None, None))
    if source is None:
        return None
    if textured is None:
        textured = [node.name for node in source.node_tree.nodes if node.bl_idname == 'ShaderNodeTexImage' and node.image]
        built[key] = (source, source_slots, textured)
    by_source = {}
    for slot, image in source_slots.items():
        if image is None:
            continue
        # Two slots on one image cannot be told apart in the copy unless they still share it.
        first = by_source.setdefault(image.get('sunrise_source_path'), slot)
        if slots[first] != slots[slot]:
            return None
    material = source.copy()
    nodes = material.node_tree.nodes
    for name in textured:
        node = nodes[name]
        slot = by_source.get(node.image.get('sunrise_source_path'))
        if slot is None:
            continue
        image, space = slots[slot], node.image.get('sunrise_preview_space')
        node.image = bundle._preview_image(image, space == 'sRGB') if space and image is not None else image
        if 'sunrise_texture_tag' in node and image is not None:
            node['sunrise_texture_tag'] = image['sunrise_texture_tag']
    material.name = info['Name']
    return material


def _new_material(info):
    material = bpy.data.materials.new(info['Name'])
    material['sunrise_world_material'] = info['Id']
    material['sunrise_material_tag'] = f'{info["TagHash"]:08X}'
    material['sunrise_preview_only'] = True
    material['sunrise_world_read_only'] = True
    return material


def _material(root, info, images, sizes, built):
    """Materials that differ only in slot textures copy the first built graph."""
    if info.get('Path') is None:
        material = _new_material(info)
        material['sunrise_preview_notes'] = 'Native material preview unavailable.'
        return material
    path = bundle._path(root, info['Path'])
    if bundle._sha(path) != info['Sha256']:
        raise ValueError('World material differs from its manifest.')
    data = bundle._read(path)
    native, profile = data['Native'], data['Preview']
    if native.get('Format') != 'sunrise-material' or native.get('Version') != 1 or native.get('TagHash') != info['TagHash']:
        raise ValueError('World material identity differs.')
    if _needs_basis(profile) != info['NeedsNativeBasis']:
        raise ValueError('World material basis requirements differ.')
    pixel = [slot for slot in native['Slots'] if slot['Stage'] == 'Pixel']
    slots = {slot['Slot']: images.get(slot['TextureTag']) for slot in pixel}
    # Texture sizes are part of the graph: point sampling snaps to texels.
    key = json.dumps([{name: value for name, value in profile.items() if name != 'SlotTextures'}, info['TagHash'],
                      bool(info.get('BackfaceCulling')),
                      sorted((slot['Slot'], sizes.get(slot['TextureTag']) if slots[slot['Slot']] else None) for slot in pixel)],
                     sort_keys=True)
    material = _reuse(built, key, info, slots)
    if material is not None:
        material['sunrise_world_material'] = info['Id']
        return material
    material = _new_material(info)
    material['sunrise_preview_root'] = str(root)
    for slot in pixel:
        node = material.node_tree.nodes.new('ShaderNodeTexImage')
        node.name = f'Pixel slot {slot["Slot"]}'
        node['sunrise_preview_slot'] = slot['Slot']
        node['sunrise_preview_stage'] = 'Pixel'
        node['sunrise_texture_tag'] = f'{slot["TextureTag"]:08X}'
        node.image = images.get(slot['TextureTag'])
    bundle._build_material_preview(material, profile)
    # EEVEE and the viewport hide the faces the game culls; Cycles ignores this setting.
    material.use_backface_culling = bool(info.get('BackfaceCulling'))
    built[key] = (material, slots, None)
    return material


def _needs_basis(profile):
    program = profile.get('Program') or {}
    nodes = program.get('Nodes') or []
    if not isinstance(nodes, list) or len(nodes) > 16384:
        raise ValueError('World material program exceeds its node limit.')
    for index, node in enumerate(nodes):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= index for value in node['Inputs']):
            raise ValueError('World material program is not an ordered expression graph.')
    if profile.get('Recipe') or profile.get('Palette') or profile.get('BlendMode') == 'NoColor':
        return False
    pending = list((program.get('Outputs') or {}).get('Normal', []))
    visited = set()
    while pending:
        index = pending.pop()
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(nodes):
            raise ValueError('World material output references an unknown node.')
        if index in visited:
            continue
        visited.add(index)
        if nodes[index]['Operation'] in ('basis', 'frontface'):
            return True
        pending.extend(nodes[index]['Inputs'])
    return False


def _visible(part, material, levels):
    """The parts Studio draws: colour stages only, and sky parts only after lighting."""
    key = (part['ModelTag'], part['ConfigTag'])
    sky_gbuffer = part.get('Layer') == 'Skybox' and part['RenderStage'] in (0, 1, 2)
    return part.get('InitiallyVisible') is not False and part['Lod'] == levels[key] and not sky_gbuffer \
        and part['RenderStage'] in (-1, 0, 1, 2, 6, 7) and material.get('BlendMode') != 'NoColor'


def _set_source(obj, row, names):
    obj['sunrise_world_prototype'] = row[0]
    obj['sunrise_source_name'] = names[row[1]]
    obj['sunrise_bubble'] = f'{row[18]:08X}'
    obj['sunrise_instance_source'] = f'{row[19]:08X}'
    obj['sunrise_instance_index'] = row[20]
    obj['sunrise_placement_source'] = f'{row[21]:08X}'
    obj['sunrise_placement_offset'] = row[22]
    obj['sunrise_class_tag'] = f'{row[23]:08X}'
    obj['sunrise_placement_identity'] = f'{row[24]:016X}'
    obj['sunrise_config_tag'] = f'{row[25]:08X}'
    obj['sunrise_activity_tag'] = f'{row[26]:08X}'
    obj['sunrise_state_ordinal'] = row[27]
    obj['sunrise_state_name'] = names[row[29]]


def import_world(manifest_path, instance_mode='AUTO'):
    """One object owns each placement; unsafe world-basis prototypes use linked mesh parts."""
    path = Path(os.path.abspath(bpy.path.abspath(str(manifest_path))))
    bundle._reject_links(path)
    if path.name != MANIFEST:
        raise ValueError('Choose world.sunrise-scene.json.')
    data = bundle._read(path)
    if data.get('Format') != 'sunrise-world-scene' or data.get('Version') != 1 \
            or data.get('Coordinates') != 'NativeZUp' or data.get('MatrixConvention') != 'RowMajorRowVector':
        raise ValueError('Unsupported Sunrise world scene.')
    if instance_mode not in ('AUTO', 'LINKED'):
        raise ValueError('Unsupported world instancing mode.')
    if any(collection.get('sunrise_world_manifest') == str(path) for collection in bpy.data.collections):
        raise ValueError('This world scene is already imported.')
    root = path.parent
    geometry, materials, textures = (_rows(data, key) for key in ('Geometry', 'Materials', 'Textures'))
    prototypes, names = (_rows(data, key) for key in ('Prototypes', 'Names'))
    for table in (geometry, materials, prototypes):
        if any(row.get('Id') != index for index, row in enumerate(table)):
            raise ValueError('World resource IDs are not consecutive.')
    if any(not isinstance(name, str) or len(name) > 16384 for name in names):
        raise ValueError('World name table is invalid.')
    buffers = _Buffers(root, _rows(data, 'Buffers'))
    previous = {kind: set(getattr(bpy.data, kind)) for kind in ('objects', 'collections', 'meshes', 'materials', 'images', 'node_groups')}
    started = time.perf_counter()
    phases = {}
    try:
        rows = data['Instances']
        if rows['Stride'] != STRIDE:
            raise ValueError('World instance stride differs.')
        records = buffers.view(rows['Rows'], 'UInt8', STRIDE)
        record_count = len(records) // STRIDE
        for row in struct.iter_unpack(RECORD, records):
            if row[0] >= len(prototypes) or row[1] >= len(names) or row[29] >= len(names) or row[30] != 0:
                raise ValueError('World placement references an unknown prototype or name.')
            _matrix(row[2:18])
        inputs = buffers.view(rows['Inputs'], 'Float32', 8, record_count).reshape((-1, 8)) if rows.get('Inputs') else None
        images = {}
        for texture in textures:
            if texture['TagHash'] in images:
                raise ValueError('World texture identity is repeated.')
            image_path = bundle._path(root, texture['Path'])
            if bundle._sha(image_path) != texture['Sha256']:
                raise ValueError('World texture differs from its manifest.')
            image = bpy.data.images.load(str(image_path), check_existing=False)
            # Native alpha is a data channel; premultiplying would erase RGB where alpha is 0.
            image.alpha_mode = 'CHANNEL_PACKED'
            image['sunrise_source_path'] = str(image_path)
            image['sunrise_source_sha256'] = texture['Sha256']
            image['sunrise_texture_tag'] = f'{texture["TagHash"]:08X}'
            images[texture['TagHash']] = image
        sizes = {texture['TagHash']: [texture['Width'], texture['Height']] for texture in textures}
        built = {}
        native_materials = [_material(root, info, images, sizes, built) for info in materials]
        phases['materials_images'] = time.perf_counter() - started
        began = time.perf_counter()
        needs_basis = set()
        for prototype in prototypes:
            for part in prototype['Parts']:
                if part['Geometry'] < 0 or part['Geometry'] >= len(geometry) or part['Material'] < 0 or part['Material'] >= len(materials):
                    raise ValueError('World part references an unknown resource.')
                _matrix(part['Transform'])
                if materials[part['Material']]['NeedsNativeBasis']:
                    needs_basis.add(part['Geometry'])
        meshes = [_mesh(info, buffers, info['Id'] in needs_basis) for info in geometry]
        phases['geometry'] = time.perf_counter() - began
        collection = bpy.data.collections.new(data['Name'])
        bpy.context.scene.collection.children.link(collection)
        collection['sunrise_world_manifest'] = str(path)
        collection['sunrise_world_source'] = f'{data["SourceTag"]:08X}'
        collection['sunrise_world_read_only'] = True
        origin = bpy.data.objects.new(data['Name'] + ' origin', None)
        collection.objects.link(origin)
        basis_group = None
        motion_groups = {}

        def part_object(part, owner, target, visible):
            nonlocal basis_group
            mesh, material = meshes[part['Geometry']], native_materials[part['Material']]
            obj = bpy.data.objects.new(part['Name'] or f'Part {part["PartIndex"]}', mesh)
            target.objects.link(obj)
            if not mesh.materials:
                mesh.materials.append(material)
            obj.material_slots[0].link = 'OBJECT'
            obj.material_slots[0].material = material
            _place(obj, owner, _matrix(part['Transform']))
            obj['sunrise_model_tag'] = f'{part["ModelTag"]:08X}'
            obj['sunrise_part_index'] = part['PartIndex']
            obj['sunrise_world_geometry'] = part['Geometry']
            obj['sunrise_world_material'] = part['Material']
            obj['sunrise_source_name_inherits_placement'] = not part['Name']
            obj['sunrise_initially_visible'] = part.get('InitiallyVisible') is not False
            obj['sunrise_initial_visibility_reason'] = part.get('VisibilityReason') or ''
            obj['sunrise_lod'] = part['Lod']
            obj['sunrise_render_stage'] = part['RenderStage']
            obj.hide_render = obj.hide_viewport = not visible
            info = geometry[part['Geometry']]
            if info.get('Motion'):
                if part['Geometry'] not in motion_groups:
                    motion = dict(info['Motion'])
                    motion['Colors'] = buffers.view(motion['Colors'], 'Float32', 4, len(mesh.vertices)).reshape((-1, 4)).tolist()
                    bundle._native_vertex_motion(obj, motion, reorder=False)
                    motion_groups[part['Geometry']] = obj.modifiers[-1].node_group
                else:
                    modifier = obj.modifiers.new('Passive motion', 'NODES')
                    modifier.node_group = motion_groups[part['Geometry']]
            if materials[part['Material']]['NeedsNativeBasis'] and info.get('Tangents') is not None:
                if basis_group is None:
                    bundle._native_basis_modifier(obj)
                    basis_group = obj.modifiers[-1].node_group
                else:
                    modifier = obj.modifiers.new('Native shader basis', 'NODES')
                    modifier.node_group = basis_group
            return obj

        began = time.perf_counter()
        collections = {}
        visibility = {}
        for prototype in prototypes:
            levels = {}
            for part in prototype['Parts']:
                key = (part['ModelTag'], part['ConfigTag'])
                levels[key] = min(levels.get(key, part['Lod']), part['Lod'])
            visibility[prototype['Id']] = [_visible(part, materials[part['Material']], levels) for part in prototype['Parts']]
            safe = prototype['CollectionInstanceSafe'] and all(not materials[p['Material']]['NeedsNativeBasis'] for p in prototype['Parts'])
            if instance_mode == 'AUTO' and safe:
                prototype_collection = bpy.data.collections.new(f'{data["Name"]} prototype {prototype["Id"]}')
                anchor = bpy.data.objects.new('Prototype origin', None)
                prototype_collection.objects.link(anchor)
                for part, visible in zip(prototype['Parts'], visibility[prototype['Id']]):
                    part_object(part, anchor, prototype_collection, visible)
                collections[prototype['Id']] = prototype_collection
        for index, row in enumerate(struct.iter_unpack(RECORD, records)):
            prototype = prototypes[row[0]]
            obj = bpy.data.objects.new(names[row[1]] or f'Placement {index}', None)
            collection.objects.link(obj)
            _place(obj, origin, _matrix(row[2:18]))
            obj['sunrise_world_instance'] = index
            _set_source(obj, row, names)
            owners = [obj]
            if row[0] in collections:
                obj.instance_type = 'COLLECTION'
                obj.instance_collection = collections[row[0]]
            else:
                owners += [part_object(part, obj, collection, visible) for part, visible in zip(prototype['Parts'], visibility[row[0]])]
            # Shaders read these through the instancer, which is the part itself when it is not instanced.
            if inputs is not None and inputs[index].any():
                for owner in owners:
                    owner['sunrise_instance_record'] = inputs[index, :4].tolist()
                    owner['sunrise_instance_occlusion'] = inputs[index, 4:].tolist()
        marker_collection = bpy.data.collections.new(data['Name'] + ' markers')
        collection.children.link(marker_collection)
        marker_collection.hide_viewport = marker_collection.hide_render = True
        for marker in _rows(data, 'Markers'):
            values = marker['Position'] + marker['Rotation'] + marker['Extents']
            if len(values) != 10 or not all(math.isfinite(value) for value in values):
                raise ValueError('World marker transform is invalid.')
            obj = bpy.data.objects.new(marker['Name'] or 'Marker', None)
            marker_collection.objects.link(obj)
            obj.parent = origin
            obj.empty_display_type = 'CUBE'
            obj.location = marker['Position']
            rotation = marker['Rotation']
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = Quaternion((rotation[3], *rotation[:3]))
            obj.scale = marker['Extents']
            obj['sunrise_marker'] = json.dumps(marker)
        world_lights.import_lights(path, collection, origin)
        effects.import_world_effects(path, collection, origin)
        phases['placements'] = time.perf_counter() - began
        began = time.perf_counter()
        bpy.context.view_layer.update()
        phases['dependency_update'] = time.perf_counter() - began
        phases['total'] = time.perf_counter() - started
        collection['sunrise_world_import_seconds'] = json.dumps(phases)
        collection['sunrise_world_counts'] = json.dumps({'placements': record_count, 'geometry': len(meshes),
                                                       'prototypes': len(prototypes), 'collection_prototypes': len(collections),
                                                       'materials': len(native_materials), 'images': len(images),
                                                       'degenerate_triangles_per_resource': sum(mesh['sunrise_degenerate_triangles'] for mesh in meshes)})
        # Raised only; a larger pool the user chose stays.
        eevee = bpy.context.scene.eevee
        if int(eevee.shadow_pool_size) < SHADOW_POOL_MB:
            collection['sunrise_world_shadow_pool'] = f'EEVEE shadow pool raised from {eevee.shadow_pool_size} MB to {SHADOW_POOL_MB} MB'
            eevee.shadow_pool_size = str(SHADOW_POOL_MB)
        return collection
    except BaseException:
        for kind in ('objects', 'collections', 'meshes', 'materials', 'node_groups', 'images'):
            table = getattr(bpy.data, kind)
            for value in set(table) - previous[kind]:
                table.remove(value, do_unlink=True)
        raise
    finally:
        buffers.close()


class SUNRISE_OT_import_world(bpy.types.Operator, ImportHelper):
    bl_idname = 'sunrise.import_world'
    bl_label = 'Import Sunrise world scene'
    bl_description = 'Import a shared read-only map scene from Sunrise Studio'
    filename_ext = '.json'
    # Blender truncates each glob to 15 characters, so the manifest name cannot be the filter;
    # import_world refuses any other file.
    filter_glob: StringProperty(default='*.json', options={'HIDDEN'})
    instance_mode: EnumProperty(name='Instances', items=(('AUTO', 'Automatic', 'Use shared prototypes when the material permits it'),
                                                        ('LINKED', 'Linked objects', 'Create mesh objects sharing the source data')), default='AUTO')

    def execute(self, context):
        try:
            collection = import_world(self.filepath, self.instance_mode)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError, IndexError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, '; '.join(['Imported ' + collection.name]
                                        + ([collection['sunrise_world_shadow_pool']] if 'sunrise_world_shadow_pool' in collection else [])))
        return {'FINISHED'}


def _import_world_menu(self, context):
    self.layout.operator(SUNRISE_OT_import_world.bl_idname, text='Sunrise world scene')


def register():
    bpy.utils.register_class(SUNRISE_OT_import_world)
    bpy.types.TOPBAR_MT_file_import.append(_import_world_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_import_world_menu)
    bpy.utils.unregister_class(SUNRISE_OT_import_world)
