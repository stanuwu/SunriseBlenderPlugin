"""Read-only world scenes share local geometry, material graphs, images and placement prototypes."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import bpy
import numpy as np
from bpy.props import EnumProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion

from . import bundle, effects, progress, world_lights


MANIFEST = 'world.sunrise-scene.json'
MAX_COUNT = 2_000_000
STRIDE = 128
# One placement row; it fills STRIDE bytes.
RECORD = '<II16fIIiIiIQIIiiII'
ROW = np.dtype([('prototype', '<u4'), ('name', '<u4'), ('matrix', '<f4', 16), ('fields', 'V48'), ('state_name', '<u4'),
                ('reserved', '<u4')])
# EEVEE shadow pool for a map; the 512 MB default overflows and drops pages as hard light bands.
SHADOW_POOL_MB = 1024
PHASES = ('Reading the manifest', 'Textures', 'Lights', 'Effects', 'Geometry', 'Materials', 'Placements', 'Sky and probes',
          'Finishing')
# Placements per step; one step takes a few milliseconds.
CHUNK = 200
# Material work, in squared document bytes, from which other Blender processes build the materials.
# Below it, starting the processes costs more than it saves.
PROCESS_WORK = 200_000_000_000
# Most material processes at once; each holds a Blender and the textures it samples.
MAX_PROCESSES = 16
# Materials per process: a graph edit costs time in proportion to every graph already in the file,
# so a fresh process for each small batch keeps every build cheap.
MATERIALS_PER_PROCESS = 24
KINDS = ('objects', 'collections', 'meshes', 'materials', 'images', 'node_groups', 'lights', 'lightprobes', 'worlds', 'libraries')
SCENE_KEYS = ('sunrise_view_transform', 'sunrise_view_look', 'sunrise_exposure')


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
    def __init__(self, root, descriptors, verified=()):
        self.root, self.descriptors = root, descriptors
        self.files = {}
        # Shards whose hash the caller already checked.
        self.verified = set(verified)

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
            if path.stat().st_size != length or index not in self.verified and bundle._sha(path) != item['Sha256']:
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


def _reuse_key(info, native, profile, images, sizes):
    """Materials with one key differ only in slot textures; images needs only membership of texture tags."""
    pixel = [slot for slot in native['Slots'] if slot['Stage'] == 'Pixel']
    present = {slot['Slot']: slot['TextureTag'] in images for slot in pixel}
    # Texture sizes are part of the graph: point sampling snaps to texels.
    return json.dumps([{name: value for name, value in profile.items() if name != 'SlotTextures'}, info['TagHash'],
                       bool(info.get('BackfaceCulling')),
                       sorted((slot['Slot'], sizes.get(slot['TextureTag']) if present[slot['Slot']] else None) for slot in pixel)],
                      sort_keys=True)


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
    key = _reuse_key(info, native, profile, images, sizes)
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


def _set_inputs(obj, inputs, index):
    """A placement's decorator inputs, which shaders read through the instancer; skipped when all are zero."""
    if index is not None and inputs is not None and inputs[index].any():
        obj['sunrise_instance_record'] = inputs[index, :4].tolist()
        obj['sunrise_instance_occlusion'] = inputs[index, 4:].tolist()


def _set_up_marker(obj, marker, collection, origin):
    collection.objects.link(obj)
    obj.parent = origin.obj
    obj.empty_display_type = 'CUBE'
    obj.location = marker['Position']
    rotation = marker['Rotation']
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = Quaternion((rotation[3], *rotation[:3]))
    obj.scale = marker['Extents']
    obj['sunrise_marker'] = json.dumps(marker)


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
    return progress.run(import_steps(manifest_path, instance_mode))


def _hash_all(pool, paths):
    """Hashes files on threads; hashlib releases the interpreter lock while it reads."""
    return {path: pool.submit(bundle._sha, path) for path in paths}


def _placements(buffers, data, prototypes, names):
    """The placement rows, checked in one pass, and their decorator inputs."""
    rows = data['Instances']
    if rows['Stride'] != STRIDE:
        raise ValueError('World instance stride differs.')
    records = buffers.view(rows['Rows'], 'UInt8', STRIDE)
    table = records.view(ROW)
    if ((table['prototype'] >= len(prototypes)) | (table['name'] >= len(names)) | (table['state_name'] >= len(names))
            | (table['reserved'] != 0)).any():
        raise ValueError('World placement references an unknown prototype or name.')
    matrices = table['matrix'].astype(np.float64)
    if not np.isfinite(matrices).all():
        raise ValueError('Invalid source transform.')
    if (np.abs(matrices[:, [3, 7, 11]]) > 1e-6).any() or (np.abs(matrices[:, 15] - 1) > 1e-6).any():
        raise ValueError('World placement must use an affine matrix.')
    inputs = buffers.view(rows['Inputs'], 'Float32', 8, len(table)).reshape((-1, 8)) if rows.get('Inputs') else None
    return records, table['matrix'], inputs


def _material_work(materials, root):
    """Build time per material, estimated as its document size squared; a graph edit costs time in proportion to the graph."""
    return [bundle._path(root, info['Path']).stat().st_size ** 2 if info.get('Path') else 0 for info in materials]


def _material_groups(materials, root, textures, work, processes):
    """
    Material ids per process, each list in id order. A material copies only an earlier graph with its reuse key,
    so all materials of one key go to one process; the largest groups are spread first.
    """
    tags = {texture['TagHash'] for texture in textures}
    sizes = {texture['TagHash']: [texture['Width'], texture['Height']] for texture in textures}
    groups = {}
    for info in materials:
        key = info['Id']
        if info.get('Path') is not None:
            data = bundle._read(bundle._path(root, info['Path']))
            key = _reuse_key(info, data['Native'], data['Preview'], tags, sizes)
        groups.setdefault(key, []).append(info['Id'])
    weight = {tag: sum(work[index] for index in ids) for tag, ids in groups.items()}
    shares = [[] for _ in range(min(processes, len(groups)))]
    loads = [0] * len(shares)
    for tag in sorted(groups, key=lambda key: -weight[key]):
        lightest = loads.index(min(loads))
        shares[lightest] += groups[tag]
        loads[lightest] += weight[tag]
    return [sorted(share) for share in shares if share]


def _processes(work):
    """How many Blender processes build the materials; 1 builds them in this one."""
    if sum(work) < PROCESS_WORK or not bpy.app.binary_path or MAX_PROCESSES < 2:
        return 1
    return max(1, min(MAX_PROCESSES, (os.cpu_count() or 2) // 2))


class _MaterialProcesses:
    """Blender processes that build shares of a world's materials and save them to a temporary folder."""

    def __init__(self):
        self.folder = Path(tempfile.mkdtemp(prefix='sunrise-materials-'))
        self.jobs, self.launches, self.libraries = [], [], []
        # Starting a Blender process takes most of a second, so they start on threads.
        self.pool = ThreadPoolExecutor(max_workers=MAX_PROCESSES)

    def start(self, path, shares):
        for index, ids in enumerate(shares):
            job = {'manifest': str(path), 'ids': ids, 'out': str(self.folder / f'materials-{index}.blend'),
                   'progress': str(self.folder / f'progress-{index}.txt'), 'log': str(self.folder / f'process-{index}.log')}
            job_path = self.folder / f'job-{index}.json'
            job_path.write_text(json.dumps(job), encoding='utf-8')
            self.jobs.append(job)
            self.launches.append(self.pool.submit(self._launch, job, job_path))

    @staticmethod
    def _launch(job, job_path):
        """Runs one batch to its end; the pool's size caps how many run at once."""
        package = Path(__file__).resolve().parent
        code = ('import sys, importlib; sys.path.insert(0, sys.argv[-2]); '
                'importlib.import_module(sys.argv[-3] + ".world")._build_materials(sys.argv[-1])')
        with open(job['log'], 'wb') as log:
            job['process'] = subprocess.Popen(
                [bpy.app.binary_path, '--background', '--factory-startup', '--python-exit-code', '1', '--python-expr', code,
                 '--', package.name, str(package.parent), str(job_path)],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            job['process'].wait()
        return job['process']

    def done(self):
        """Materials built so far."""
        total = 0
        for job in self.jobs:
            try:
                total += int(Path(job['progress']).read_text() or 0)
            except (OSError, ValueError):
                pass
        return total

    def collect(self):
        """Appends the materials of each process that has finished; returns them."""
        result = []
        for job, launch in zip(self.jobs, self.launches):
            if job.get('collected') or not launch.done():
                continue
            code = launch.result().poll()
            if code is None:
                continue
            if code != 0:
                lines = Path(job['log']).read_text(errors='replace').strip().splitlines()
                raise RuntimeError('A material process failed: ' + (lines[-1] if lines else f'exit code {code}'))
            # Appending makes each material local in a pass over every data block, which grows with the file;
            # linking and copying does not. The copies use the linked images until release.
            libraries = set(bpy.data.libraries)
            with bpy.data.libraries.load(job['out'], link=True) as (source, target):
                target.materials = list(source.materials)
            result += [material.copy() for material in target.materials]
            self.libraries += [library for library in bpy.data.libraries if library not in libraries]
            job['collected'] = True
        return result

    def release(self):
        """Removes the linked materials and images, once the copies use this file's images."""
        owned = set(self.libraries)
        linked = [item for kind in (bpy.data.materials, bpy.data.images, bpy.data.node_groups)
                  for item in kind if item.library in owned]
        bpy.data.batch_remove(linked + self.libraries)
        self.libraries = []

    def running(self):
        return not all(job.get('collected') for job in self.jobs)

    def close(self):
        for launch in self.launches:
            launch.cancel()
        for job in self.jobs:
            process = job.get('process')
            if process is not None and process.poll() is None:
                process.kill()
        self.pool.shutdown(wait=True)
        self.launches = []
        shutil.rmtree(self.folder, ignore_errors=True)


def _image_record(image):
    """What the importing process needs to find the same image: source, preview colour space and properties."""
    return {'path': image.get('sunrise_source_path') or bpy.path.abspath(image.filepath),
            'space': image.get('sunrise_preview_space'),
            'properties': {key: image[key] for key in image.keys()
                           if key != 'sunrise_preview_space' and isinstance(image[key], (str, int, float, bool))}}


def _build_materials(job_path):
    """Runs in a material process: builds its share of materials and saves them, images replaced by records."""
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    path = Path(job['manifest'])
    data = bundle._read(path)
    root = path.parent
    images = {}
    for texture in data['Textures']:
        images[texture['TagHash']] = _load_texture(bundle._path(root, texture['Path']), texture)
    sizes = {texture['TagHash']: [texture['Width'], texture['Height']] for texture in data['Textures']}
    built, done = {}, []
    for count, index in enumerate(job['ids'], 1):
        done.append(_material(root, data['Materials'][index], images, sizes, built))
        Path(job['progress']).write_text(str(count))
    # The importing process swaps each image for its own copy in one remap; setting a node's image
    # there would update the material's whole graph once per node.
    for image in bpy.data.images:
        image['sunrise_image'] = json.dumps(_image_record(image))
    bpy.data.libraries.write(job['out'], set(done), path_remap='NONE', fake_user=False, compress=True)


def _load_texture(image_path, texture):
    image = bpy.data.images.load(str(image_path), check_existing=False)
    # Native alpha is a data channel; premultiplying would erase RGB where alpha is 0.
    image.alpha_mode = 'CHANNEL_PACKED'
    image['sunrise_source_path'] = str(image_path)
    image['sunrise_source_sha256'] = texture['Sha256']
    image['sunrise_texture_tag'] = f'{texture["TagHash"]:08X}'
    return image


def _adopt_materials(appended, images, materials, libraries):
    """
    Names materials built in other processes in id order, so names match a build in this process,
    then swaps each linked image for this file's image in one remap per image.
    """
    count = len(materials)
    result = [None] * count
    for material in appended:
        result[material['sunrise_world_material']] = material
    if any(material is None for material in result):
        raise RuntimeError('A material process left out a material.')
    for material in result:
        material.name = f'Sunrise material {material["sunrise_world_material"]}'
    for material, info in zip(result, materials):
        material.name = info['Name']
    # First use in id order, so the first colour space asked of an image matches a build in this process.
    owned = set(libraries)
    linked = list(dict.fromkeys(node.image for material in result for node in material.node_tree.nodes
                                if node.bl_idname == 'ShaderNodeTexImage' and node.image is not None
                                and node.image.library in owned and 'sunrise_image' in node.image))
    seen = set(linked)
    linked += [image for image in bpy.data.images
               if image.library in owned and 'sunrise_image' in image and image not in seen]
    bases = {image['sunrise_source_path']: image for image in images.values()}
    found = {}
    for index, image in enumerate(linked):
        record = json.loads(image['sunrise_image'])
        key = (record['path'], record['space'])
        if key not in found:
            base = bases.get(record['path'])
            if base is None:
                # A texture outside the manifest's table, such as a volume texture's slices.
                base = bpy.data.images.load(record['path'], check_existing=True)
                base.use_fake_user = True
                base.alpha_mode = 'CHANNEL_PACKED'
                for name, value in record['properties'].items():
                    base[name] = value
                bases[record['path']] = base
            found[key] = bundle._preview_image(base, record['space'] == 'sRGB') if record['space'] else base
        image.user_remap(found[key])
        yield 'Materials', index + 1, len(linked)
    return result


def _scene_state(scene):
    return {'world': scene.world, 'view': (scene.view_settings.view_transform, scene.view_settings.look,
                                           scene.view_settings.exposure),
            'pools': (scene.eevee.shadow_pool_size, scene.eevee.gi_irradiance_pool_size),
            'keys': {key: scene[key] for key in SCENE_KEYS if key in scene}}


def _restore_scene(scene, state):
    scene.world = state['world']
    scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure = state['view']
    scene.eevee.shadow_pool_size, scene.eevee.gi_irradiance_pool_size = state['pools']
    for key in SCENE_KEYS:
        if key in state['keys']:
            scene[key] = state['keys'][key]
        elif key in scene:
            del scene[key]


def import_steps(manifest_path, instance_mode='AUTO'):
    """
    The world import as steps that yield (phase, done, total) and return the world collection.
    Closing the steps early removes everything they added and restores the scene's world and view.
    """
    path = Path(os.path.abspath(bpy.path.abspath(str(manifest_path))))
    bundle._reject_links(path)
    if path.name != MANIFEST:
        raise ValueError('Choose world.sunrise-scene.json.')
    if instance_mode not in ('AUTO', 'LINKED'):
        raise ValueError('Unsupported world instancing mode.')
    yield PHASES[0], 0, 0
    data = bundle._read(path)
    if data.get('Format') != 'sunrise-world-scene' or data.get('Version') != 1 \
            or data.get('Coordinates') != 'NativeZUp' or data.get('MatrixConvention') != 'RowMajorRowVector':
        raise ValueError('Unsupported Sunrise world scene.')
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
    descriptors = _rows(data, 'Buffers')
    scene = bpy.context.scene
    before, scene_state = progress.snapshot(KINDS), _scene_state(scene)
    started = time.perf_counter()
    phases = {}
    clock = [started]

    def lap(name):
        now = time.perf_counter()
        phases[name] = now - clock[0]
        clock[0] = now

    pool = builders = None
    buffers = _Buffers(root, descriptors)
    try:
        shard_paths = [bundle._path(root, item['Path']) for item in descriptors]
        texture_paths = [bundle._path(root, texture['Path']) for texture in textures]
        pool = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))
        hashes = _hash_all(pool, shard_paths + texture_paths)
        for index, (item, shard) in enumerate(zip(descriptors, shard_paths)):
            if hashes[shard].result() != item['Sha256']:
                raise ValueError('World binary shard differs from its manifest.')
            buffers.verified.add(index)
            yield PHASES[0], index + 1, len(descriptors)
        records, matrices, inputs = _placements(buffers, data, prototypes, names)
        record_count = len(matrices)
        work = _material_work(materials, root)
        processes = _processes(work)
        if processes > 1:
            builders = _MaterialProcesses()
            batches = max(processes, math.ceil(len(materials) / MATERIALS_PER_PROCESS))
            builders.start(path, _material_groups(materials, root, textures, work, batches))
        lap('manifest')
        images = {}
        for index, (texture, image_path) in enumerate(zip(textures, texture_paths)):
            if texture['TagHash'] in images:
                raise ValueError('World texture identity is repeated.')
            if hashes[image_path].result() != texture['Sha256']:
                raise ValueError('World texture differs from its manifest.')
            images[texture['TagHash']] = _load_texture(image_path, texture)
            yield 'Textures', index + 1, len(textures)
        pool.shutdown()
        sizes = {texture['TagHash']: [texture['Width'], texture['Height']] for texture in textures}
        lap('textures')
        objects = bundle._Objects()
        collection = bpy.data.collections.new(data['Name'])
        collection['sunrise_world_manifest'] = str(path)
        collection['sunrise_world_source'] = f'{data["SourceTag"]:08X}'
        collection['sunrise_world_read_only'] = True
        # The collection joins the scene once it is complete; until then no view layer update covers it.
        world_layout = bundle._Layout(collection)
        origin = objects.add(data['Name'] + ' origin', None, lambda obj: world_layout.link(obj, bundle._Layout.IDENTITY))
        # Lights and effects build their small node graphs before the materials arrive: every graph edit
        # costs time in proportion to all the graphs in the file.
        lights = yield from world_lights.light_steps(path, collection, origin, objects)
        lap('lights')
        effect_state = yield from effects.world_steps(path, collection, origin, objects)
        lap('effects')
        needs_basis = set()
        part_matrices = {}
        for prototype in prototypes:
            for part in prototype['Parts']:
                if part['Geometry'] < 0 or part['Geometry'] >= len(geometry) or part['Material'] < 0 or part['Material'] >= len(materials):
                    raise ValueError('World part references an unknown resource.')
                _matrix(part['Transform'])
                part_matrices[id(part)] = np.asarray(part['Transform'], dtype=np.float32)
                if materials[part['Material']]['NeedsNativeBasis']:
                    needs_basis.add(part['Geometry'])
        meshes, appended = [], []
        for info in geometry:
            meshes.append(_mesh(info, buffers, info['Id'] in needs_basis))
            if builders is not None:
                appended += builders.collect()
            yield 'Geometry', len(meshes), len(geometry)
        lap('geometry')
        if builders is None:
            built, native_materials = {}, []
            for info in materials:
                native_materials.append(_material(root, info, images, sizes, built))
                yield 'Materials', len(native_materials), len(materials)
        else:
            while builders.running():
                appended += builders.collect()
                yield 'Materials', builders.done(), len(materials)
                time.sleep(0.02)
            native_materials = yield from _adopt_materials(appended, images, materials, builders.libraries)
            builders.release()
            builders.close()
        lap('materials')
        shared = {'basis': None, 'motion': {}}

        def set_up_part(obj, part, owner, layout, visible, index):
            mesh, material = meshes[part['Geometry']], native_materials[part['Material']]
            layout.link(obj, part_matrices[id(part)], not visible)
            if not mesh.materials:
                mesh.materials.append(material)
            obj.material_slots[0].link = 'OBJECT'
            obj.material_slots[0].material = material
            obj.parent = owner.obj
            obj['sunrise_model_tag'] = f'{part["ModelTag"]:08X}'
            obj['sunrise_part_index'] = part['PartIndex']
            obj['sunrise_world_geometry'] = part['Geometry']
            obj['sunrise_world_material'] = part['Material']
            obj['sunrise_source_name_inherits_placement'] = not part['Name']
            obj['sunrise_initially_visible'] = part.get('InitiallyVisible') is not False
            obj['sunrise_initial_visibility_reason'] = part.get('VisibilityReason') or ''
            obj['sunrise_lod'] = part['Lod']
            obj['sunrise_render_stage'] = part['RenderStage']
            info = geometry[part['Geometry']]
            if info.get('Motion'):
                if part['Geometry'] not in shared['motion']:
                    motion = dict(info['Motion'])
                    motion['Colors'] = buffers.view(motion['Colors'], 'Float32', 4, len(mesh.vertices)).reshape((-1, 4)).tolist()
                    bundle._native_vertex_motion(obj, motion, reorder=False)
                    shared['motion'][part['Geometry']] = obj.modifiers[-1].node_group
                else:
                    modifier = obj.modifiers.new('Passive motion', 'NODES')
                    modifier.node_group = shared['motion'][part['Geometry']]
            if materials[part['Material']]['NeedsNativeBasis'] and info.get('Tangents') is not None:
                if shared['basis'] is None:
                    bundle._native_basis_modifier(obj)
                    shared['basis'] = obj.modifiers[-1].node_group
                else:
                    modifier = obj.modifiers.new('Native shader basis', 'NODES')
                    modifier.node_group = shared['basis']
            _set_inputs(obj, inputs, index)

        def part_object(part, owner, layout, visible, index=None):
            return objects.add(part['Name'] or f'Part {part["PartIndex"]}', meshes[part['Geometry']],
                               lambda obj: set_up_part(obj, part, owner, layout, visible, index))

        layouts = [world_layout]
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
                layout = bundle._Layout(prototype_collection)
                layouts.append(layout)
                anchor = objects.add('Prototype origin', None, layout.link)
                for part, visible in zip(prototype['Parts'], visibility[prototype['Id']]):
                    part_object(part, anchor, layout, visible)
                collections[prototype['Id']] = prototype_collection
            if prototype['Id'] % CHUNK == 0:
                yield 'Placements', 0, record_count

        def set_up_placement(obj, index):
            row = struct.unpack_from(RECORD, records, index * STRIDE)
            world_layout.link(obj, matrices[index])
            obj.parent = origin.obj
            obj['sunrise_world_instance'] = index
            _set_source(obj, row, names)
            if row[0] in collections:
                obj.instance_type = 'COLLECTION'
                obj.instance_collection = collections[row[0]]
            _set_inputs(obj, inputs, index)

        for start in range(0, record_count, CHUNK):
            rows = struct.iter_unpack(RECORD, records[start * STRIDE:min(start + CHUNK, record_count) * STRIDE])
            for index, row in enumerate(rows, start):
                placement = objects.add(names[row[1]] or f'Placement {index}', None,
                                        lambda obj, index=index: set_up_placement(obj, index))
                if row[0] not in collections:
                    # Shaders read the placement inputs through the instancer, which is the part itself when it is not instanced.
                    for part, visible in zip(prototypes[row[0]]['Parts'], visibility[row[0]]):
                        part_object(part, placement, world_layout, visible, index)
            yield 'Placements', 0, record_count
        marker_collection = bpy.data.collections.new(data['Name'] + ' markers')
        collection.children.link(marker_collection)
        marker_collection.hide_viewport = marker_collection.hide_render = True
        for marker in _rows(data, 'Markers'):
            values = marker['Position'] + marker['Rotation'] + marker['Extents']
            if len(values) != 10 or not all(math.isfinite(value) for value in values):
                raise ValueError('World marker transform is invalid.')
            objects.add(marker['Name'] or 'Marker', None, lambda obj, marker=marker: _set_up_marker(obj, marker, marker_collection, origin))
        yield from objects.create('Placements')
        yield from objects.set_up(lambda index: 'Placements')
        for layout in layouts:
            layout.write()
        lap('placements')
        if effect_state is not None:
            effects.finish(effect_state)
        probes = []
        if lights:
            probes = world_lights.finish_lights(lights)
            world_lights.set_sky(lights, collection.name)
        lap('sky')
        yield 'Sky and probes', len(probes), len(probes)
        scene.collection.children.link(collection)
        bpy.context.view_layer.update()
        lap('dependency_update')
        if lights:
            lights['target'].collection['sunrise_light_volume_bake'] = world_lights._bake(scene, probes)
        lap('bake')
        yield 'Finishing', 0, 0
        phases['total'] = time.perf_counter() - started
        collection['sunrise_world_import_seconds'] = json.dumps(phases)
        collection['sunrise_world_counts'] = json.dumps({'placements': record_count, 'geometry': len(meshes),
                                                       'prototypes': len(prototypes), 'collection_prototypes': len(collections),
                                                       'materials': len(native_materials), 'images': len(images),
                                                       'degenerate_triangles_per_resource': sum(mesh['sunrise_degenerate_triangles'] for mesh in meshes)})
        # Raised only; a larger pool the user chose stays.
        eevee = scene.eevee
        if int(eevee.shadow_pool_size) < SHADOW_POOL_MB:
            collection['sunrise_world_shadow_pool'] = f'EEVEE shadow pool raised from {eevee.shadow_pool_size} MB to {SHADOW_POOL_MB} MB'
            eevee.shadow_pool_size = str(SHADOW_POOL_MB)
        return collection
    except BaseException:
        _restore_scene(scene, scene_state)
        progress.rollback(before)
        raise
    finally:
        buffers.close()
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        if builders is not None:
            builders.close()


class SUNRISE_OT_import_world(progress.ImportSteps, bpy.types.Operator, ImportHelper):
    bl_idname = 'sunrise.import_world'
    bl_label = 'Import Sunrise world scene'
    bl_description = 'Import a shared read-only map scene from Sunrise Studio'
    filename_ext = '.json'
    # Blender truncates each glob to 15 characters, so the manifest name cannot be the filter;
    # import_world refuses any other file.
    filter_glob: StringProperty(default='*.json', options={'HIDDEN'})
    instance_mode: EnumProperty(name='Instances', items=(('AUTO', 'Automatic', 'Use shared prototypes when the material permits it'),
                                                        ('LINKED', 'Linked objects', 'Create mesh objects sharing the source data')), default='AUTO')
    phases = PHASES

    def steps(self, context):
        return import_steps(self.filepath, self.instance_mode)

    def finished(self, context, collection):
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
