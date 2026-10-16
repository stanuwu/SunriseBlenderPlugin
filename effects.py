"""Plays the particle loops Sunrise Studio baked as camera-facing sprites."""

import bpy
import numpy as np

from . import bundle, progress


FORMAT = 'sunrise-effects'
WORLD_EFFECTS = 'world.effects.json'
BUNDLE_EFFECTS = 'particles/effects.json'
GROUP = 'Sunrise effect sprites'
# Bump when the node group changes, so an old file rebuilds it.
GROUP_VERSION = 1
MAX_EMITTERS = 100_000
MAX_SAMPLES = 50_000_000
SAMPLE = np.dtype([('position', '<f4', 3), ('size', '<f2'), ('alpha', '<f2'), ('spin', '<f2'), ('frame', '<u2')])
# Blend factors the material builds; others fall back to One and are noted on the material.
SOURCE_FACTORS = {'Zero', 'One', 'SourceAlpha', 'InverseSourceAlpha', 'SourceColor', 'InverseSourceColor',
                  'DestinationAlpha', 'InverseDestinationAlpha', 'DestinationColor'}
DESTINATION_FACTORS = {'Zero', 'One', 'SourceAlpha', 'InverseSourceAlpha', 'SourceColor', 'InverseSourceColor',
                       'DestinationAlpha', 'InverseDestinationAlpha'}
# Ray visibility a sprite turns off.
RAY_FLAGS = ('visible_shadow', 'visible_diffuse', 'visible_glossy', 'visible_transmission', 'visible_volume_scatter')


def _count(value, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise ValueError('Effect count is invalid.')
    return value


def _rows(document, name, maximum=MAX_EMITTERS):
    rows = document.get(name)
    if not isinstance(rows, list) or len(rows) > maximum:
        raise ValueError('Effect table is invalid: ' + name)
    return rows


KINDS = ('objects', 'collections', 'meshes', 'materials', 'images', 'node_groups')


def world_steps(manifest_path, collection, origin, objects):
    """Requests a world scene's baked effects in a child collection; returns the state for finish, or None."""
    path = manifest_path.parent / WORLD_EFFECTS
    if not path.exists():
        return None
    return (yield from _steps(path, manifest_path.parent, collection, lambda info: origin, objects))


def import_world_effects(manifest_path, collection, origin):
    """Adds a world scene's baked effects to a child collection; None when it has none."""
    return _import(lambda objects: world_steps(manifest_path, collection, origin, objects))


def import_bundle_effects(root, collection):
    """Adds a bundle's baked effects as children of their emitter markers; None when it has none."""
    path = root / BUNDLE_EFFECTS
    if not path.exists():
        return None
    markers = {obj.get('sunrise_emitter_tag'): obj for obj in collection.all_objects if obj.get('sunrise_emitter_tag')}
    return _import(lambda objects: _steps(path, root, collection, lambda info: markers.get(f'{info["EmitterTag"]:08X}'), objects))


def _import(plan):
    """Plans, creates and finishes the effects; a failure removes what they added."""
    before = progress.snapshot(KINDS)
    try:
        objects = bundle._Objects()
        state = progress.run(plan(objects))
        if state is None:
            return None
        progress.run(objects.create('Effects'))
        progress.run(objects.set_up(lambda index: 'Effects'))
        return finish(state)
    except BaseException:
        progress.rollback(before)
        raise


def _steps(path, root, collection, parent_of, objects):
    data = bundle._read(path)
    if data.get('Format') != FORMAT or data.get('Version') != 1 or data.get('Coordinates') != 'NativeZUp':
        raise ValueError('Unsupported Sunrise effects.')
    if data.get('SampleStride') != SAMPLE.itemsize:
        raise ValueError('Effect sample layout differs.')
    rate = data.get('Rate')
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 1 <= rate <= 1000:
        raise ValueError('Effect rate is invalid.')
    emitters = _rows(data, 'Emitters')
    if not emitters:
        return None
    samples = _samples(root, data)
    images = [_image(root, info, index) for index, info in enumerate(_rows(data, 'Sprites'))]
    materials = _rows(data, 'Materials')
    if any(info.get('Id') != index for index, info in enumerate(materials)):
        raise ValueError('Effect material IDs are not consecutive.')
    built = {}
    group = _node_group()
    holder = bpy.data.collections.new(collection.name + ' effects')
    collection.children.link(holder)
    holder['sunrise_effects'] = str(path)
    target = bundle._Layout(holder)
    meshes = bundle._Names(bpy.data.meshes)
    for index, info in enumerate(emitters):
        first, count = _count(info.get('First'), len(samples)), _count(info.get('Count'), len(samples))
        frames = _count(info.get('Frames'), 65535)
        if first + count > len(samples) or frames == 0:
            raise ValueError('Effect emitter leaves its samples.')
        rows = samples[first:first + count]
        if len(rows) and int(rows['frame'].max()) >= frames:
            raise ValueError('Effect sample frame leaves its loop.')
        material = _count(info.get('Material'), len(materials) - 1)
        if material not in built:
            built[material] = _material(materials[material], images)
        _emitter(info, rows, frames, rate, built[material], group, target, parent_of(info), meshes, objects)
        yield 'Effects', index + 1, len(emitters)
    return {'holder': holder, 'target': target, 'emitters': len(emitters), 'samples': len(samples),
            'skipped': len(_rows(data, 'Skipped'))}


def finish(state):
    """Turns off the sprites' ray visibility and writes their hiding, once their objects are set up."""
    holder = state['holder']
    # A sprite is seen, never lights or shadows anything, as in Studio.
    off = np.zeros(len(holder.objects), dtype=bool)
    for flag in RAY_FLAGS:
        if flag in bpy.types.Object.bl_rna.properties:
            holder.objects.foreach_set(flag, off)
    state['target'].write(refresh=True)
    holder['sunrise_effect_counts'] = str({'emitters': state['emitters'], 'samples': state['samples'], 'skipped': state['skipped']})
    return holder

def _samples(root, data):
    path = bundle._path(root, data['DataPath'])
    length = _count(data.get('DataLength'), MAX_SAMPLES * SAMPLE.itemsize)
    if length % SAMPLE.itemsize or path.stat().st_size != length or bundle._sha(path) != data.get('DataSha256'):
        raise ValueError('Effect samples differ from their document.')
    samples = np.fromfile(path, dtype=SAMPLE)
    values = np.concatenate([samples['position'].reshape(-1), samples['size'].astype(np.float32),
                             samples['alpha'].astype(np.float32), samples['spin'].astype(np.float32)])
    if not np.isfinite(values).all():
        raise ValueError('Effect samples contain nonfinite values.')
    return samples


def _image(root, info, index):
    if info.get('Id') != index:
        raise ValueError('Effect sprite IDs are not consecutive.')
    path = bundle._path(root, info['Path'])
    if bundle._sha(path) != info['Sha256']:
        raise ValueError('Effect sprite differs from its document.')
    image = bpy.data.images.load(str(path), check_existing=True)
    # The viewer reads raw colour and alpha; premultiplying would change the blend.
    image.alpha_mode = 'CHANNEL_PACKED'
    image.colorspace_settings.name = 'sRGB' if info.get('Srgb') and not info.get('SingleChannel') else 'Non-Color'
    image['sunrise_texture_tag'] = f'{info["TextureTag"]:08X}'
    image['sunrise_single_channel'] = bool(info.get('SingleChannel'))
    return image


def _emitter(info, rows, frames, rate, material, group, target, parent, meshes, objects):
    """One point per sample; the node group draws the points of the current loop frame."""
    name = str(info.get('Name') or 'Effect')
    mesh = bpy.data.meshes.new(meshes(name))
    mesh.vertices.add(len(rows))
    mesh.vertices.foreach_set('co', np.ascontiguousarray(rows['position'], dtype=np.float32).reshape(-1))
    for key, kind, values in (('sunrise_frame', 'INT', rows['frame'].astype(np.int32)),
                              ('sunrise_size', 'FLOAT', rows['size'].astype(np.float32)),
                              ('sunrise_alpha', 'FLOAT', rows['alpha'].astype(np.float32)),
                              ('sunrise_spin', 'FLOAT', rows['spin'].astype(np.float32))):
        attribute = mesh.attributes.new(key, kind, 'POINT')
        attribute.data.foreach_set('value', values)
    mesh.update()

    def setup(obj):
        target.link(obj, hidden=not info.get('Default', True))
        owner = bundle._resolve(parent)
        if owner is not None:
            # Samples are in scene space; the parent inverse keeps them there.
            obj.parent = owner
            obj.matrix_parent_inverse = owner.matrix_world.inverted()
        modifier = obj.modifiers.new('Effect sprites', 'NODES')
        modifier.node_group = group
        for key, value in (('Frames', frames), ('Rate', float(rate)), ('Material', material)):
            getattr(modifier.properties.inputs, group.interface.items_tree[key].identifier).value = value
        obj['sunrise_effect_preview'] = True
        obj['sunrise_effect_index'] = info['Id']
        obj['sunrise_preview_only'] = True
        obj['sunrise_effect_tag'] = f'{info["EffectTag"]:08X}'
        obj['sunrise_effect_system'] = f'{info["SystemTag"]:08X}'
        obj['sunrise_effect_emitter'] = f'{info["EmitterTag"]:08X}'
        obj['sunrise_placement_source'] = f'{info["SourceTag"]:08X}'
        obj['sunrise_placement_offset'] = info['SourceOffset']
        obj['sunrise_effect_trigger'] = str(info.get('Trigger', ''))
        obj['sunrise_effect_loop'] = f'{info.get("Mode", "")}, {frames / rate:.2f} s'
        if info.get('Gpu'):
            obj['sunrise_effect_note'] = 'The game simulates this emitter on the GPU; Studio runs its CPU steps.'

    return objects.add(name, mesh, setup)


def _node_group():
    """Draws the points of frame floor(seconds * Rate) mod Frames as quads that face the active camera."""
    existing = bpy.data.node_groups.get(GROUP)
    if existing is not None and existing.get('sunrise_version') == GROUP_VERSION:
        return existing
    tree = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
    tree['sunrise_version'] = GROUP_VERSION
    tree.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    frames = tree.interface.new_socket('Frames', in_out='INPUT', socket_type='NodeSocketInt')
    frames.default_value, frames.min_value = 1, 1
    rate = tree.interface.new_socket('Rate', in_out='INPUT', socket_type='NodeSocketFloat')
    rate.default_value = 30.0
    tree.interface.new_socket('Material', in_out='INPUT', socket_type='NodeSocketMaterial')
    tree.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, link = tree.nodes, tree.links.new
    inputs, outputs = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')

    def math(operation, a, b=None):
        node = nodes.new('ShaderNodeMath')
        node.operation = operation
        link(a, node.inputs[0])
        if b is not None:
            link(b, node.inputs[1])
        return node

    def attribute(name, kind):
        node = nodes.new('GeometryNodeInputNamedAttribute')
        node.data_type = kind
        node.inputs['Name'].default_value = name
        return node.outputs['Attribute']

    time = nodes.new('GeometryNodeInputSceneTime')
    step = math('FLOOR', math('MULTIPLY', time.outputs['Seconds'], inputs.outputs['Rate']).outputs[0])
    current = math('FLOORED_MODULO', step.outputs[0], inputs.outputs['Frames'])
    match = math('COMPARE', attribute('sunrise_frame', 'INT'), current.outputs[0])
    match.inputs[2].default_value = 0.5
    points = nodes.new('GeometryNodeMeshToPoints')
    link(inputs.outputs['Geometry'], points.inputs['Mesh'])
    # A 2 by 2 quad, so the size is the half-width the viewer uses.
    grid = nodes.new('GeometryNodeMeshGrid')
    for key, value in (('Size X', 2.0), ('Size Y', 2.0), ('Vertices X', 2), ('Vertices Y', 2)):
        grid.inputs[key].default_value = value
    uv = nodes.new('GeometryNodeStoreNamedAttribute')
    uv.data_type, uv.domain = 'FLOAT2', 'CORNER'
    uv.inputs['Name'].default_value = 'UVMap'
    link(grid.outputs['Mesh'], uv.inputs['Geometry'])
    link(grid.outputs['UV Map'], uv.inputs['Value'])
    camera = nodes.new('GeometryNodeInputActiveCamera')
    info = nodes.new('GeometryNodeObjectInfo')
    info.transform_space = 'RELATIVE'
    link(camera.outputs[0], info.inputs['Object'])
    spin = nodes.new('FunctionNodeAxisAngleToRotation')
    spin.inputs['Axis'].default_value = (0.0, 0.0, 1.0)
    link(attribute('sunrise_spin', 'FLOAT'), spin.inputs['Angle'])
    # Screen-aligned like the viewer's sprites: the camera's turn, then the spin about the view axis.
    turn = nodes.new('FunctionNodeRotateRotation')
    turn.rotation_space = 'LOCAL'
    link(info.outputs['Rotation'], turn.inputs['Rotation'])
    link(spin.outputs['Rotation'], turn.inputs['Rotate By'])
    place = nodes.new('GeometryNodeInstanceOnPoints')
    link(points.outputs['Points'], place.inputs['Points'])
    link(match.outputs[0], place.inputs['Selection'])
    link(uv.outputs['Geometry'], place.inputs['Instance'])
    link(turn.outputs['Rotation'], place.inputs['Rotation'])
    link(attribute('sunrise_size', 'FLOAT'), place.inputs['Scale'])
    realize = nodes.new('GeometryNodeRealizeInstances')
    link(place.outputs['Instances'], realize.inputs['Geometry'])
    paint = nodes.new('GeometryNodeSetMaterial')
    link(realize.outputs['Geometry'], paint.inputs['Geometry'])
    link(inputs.outputs['Material'], paint.inputs['Material'])
    link(paint.outputs['Geometry'], outputs.inputs['Geometry'])
    return tree


def _material(info, images):
    """
    The viewer's sprite shading: texel times fade times a radial falloff, blended as
    Source * colour + Destination * background, built from emission plus a transparent BSDF.
    """
    blend = info.get('Blend') or {}
    sprite = info.get('Sprite')
    material = bpy.data.materials.new(f'Effect {info["TagHash"]:08X}')
    material['sunrise_material_tag'] = f'{info["TagHash"]:08X}'
    material['sunrise_preview_only'] = True
    if hasattr(material, 'use_nodes') and not material.use_nodes:
        material.use_nodes = True
    if hasattr(material, 'surface_render_method'):
        material.surface_render_method = 'BLENDED'
    material.use_backface_culling = False
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    notes = []

    def math(operation, a, b=None, clamp=False):
        node = nodes.new('ShaderNodeMath')
        node.operation, node.use_clamp = operation, clamp
        for index, item in enumerate((a, b)):
            if item is None:
                continue
            if isinstance(item, (int, float)):
                node.inputs[index].default_value = item
            else:
                links.new(item, node.inputs[index])
        return node.outputs[0]

    def vector(operation, a, b=None, scale=None):
        node = nodes.new('ShaderNodeVectorMath')
        node.operation = operation
        links.new(a, node.inputs[0])
        if b is not None:
            if isinstance(b, (tuple, list)):
                node.inputs[1].default_value = b
            else:
                links.new(b, node.inputs[1])
        if scale is not None:
            links.new(scale, node.inputs['Scale'])
        return node.outputs['Vector']

    def attribute(name, socket):
        node = nodes.new('ShaderNodeAttribute')
        node.attribute_type, node.attribute_name = 'GEOMETRY', name
        return node.outputs[socket]

    uv = nodes.new('ShaderNodeSeparateXYZ')
    links.new(attribute('UVMap', 'Vector'), uv.inputs[0])
    cx = math('MULTIPLY_ADD', uv.outputs['X'], 2.0)
    cy = math('MULTIPLY_ADD', uv.outputs['Y'], 2.0)
    for socket in (cx, cy):
        socket.node.inputs[2].default_value = -1.0
    radius2 = math('ADD', math('MULTIPLY', cx, cx), math('MULTIPLY', cy, cy))
    falloff = math('SUBTRACT', 1.0, radius2, clamp=True)
    image = images[sprite] if isinstance(sprite, int) and 0 <= sprite < len(images) else None
    if image is None:
        # The viewer's soft dot: white, alpha (1 - r)^2.
        dot = math('SUBTRACT', 1.0, math('SQRT', radius2), clamp=True)
        texel_alpha = math('MULTIPLY', dot, dot)
        rgb = None
    else:
        texture = nodes.new('ShaderNodeTexImage')
        texture.image, texture.extension, texture.interpolation = image, 'CLIP', 'Linear'
        links.new(attribute('UVMap', 'Vector'), texture.inputs['Vector'])
        if image.get('sunrise_single_channel'):
            red = nodes.new('ShaderNodeSeparateColor')
            links.new(texture.outputs['Color'], red.inputs[0])
            texel_alpha, rgb = red.outputs[0], None
        else:
            texel_alpha, rgb = texture.outputs['Alpha'], texture.outputs['Color']
    alpha = math('MULTIPLY', math('MULTIPLY', texel_alpha, attribute('sunrise_alpha', 'Fac')), falloff)
    if blend.get('Premultiply', True):
        color = vector('SCALE', rgb, scale=alpha) if rgb is not None else _constant(nodes, links, alpha)
    else:
        color = rgb if rgb is not None else _constant(nodes, links, 1.0)
    enabled = blend.get('Enabled', True)
    source = blend.get('Source', 'One') if enabled else 'One'
    destination = blend.get('Destination', 'One') if enabled else 'Zero'
    if blend.get('Operation', 'Add') != 'Add':
        notes.append(f'blend operation {blend.get("Operation")} drawn as Add')
    if source not in SOURCE_FACTORS:
        notes.append(f'source factor {source} drawn as One')
        source = 'One'
    if destination not in DESTINATION_FACTORS:
        notes.append(f'destination factor {destination} drawn as One')
        destination = 'One'

    def factor(name):
        """A factor as a colour socket or a number; the background alpha reads 1."""
        if name == 'Zero' or name == 'InverseDestinationAlpha':
            return 0.0
        if name == 'One' or name == 'DestinationAlpha':
            return 1.0
        if name == 'SourceAlpha':
            return alpha
        if name == 'InverseSourceAlpha':
            return math('SUBTRACT', 1.0, alpha)
        if name == 'SourceColor':
            return color
        return vector('SUBTRACT', _constant(nodes, links, 1.0), color)

    transparent = nodes.new('ShaderNodeBsdfTransparent')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 1.0
    if source == 'DestinationColor':
        # Colour times background: all of it goes through the transparent colour.
        destination_factor = factor(destination)
        if isinstance(destination_factor, float):
            transmit = vector('ADD', color, (destination_factor,) * 3)
        else:
            transmit = vector('ADD', color, destination_factor if destination_factor.type == 'VECTOR' else
                              _constant(nodes, links, destination_factor))
        links.new(transmit, transparent.inputs['Color'])
        emission.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    else:
        source_factor = factor(source)
        if isinstance(source_factor, float):
            if source_factor == 0:
                emission.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            else:
                links.new(color, emission.inputs['Color'])
        elif source_factor.type == 'VALUE':
            links.new(vector('SCALE', color, scale=source_factor), emission.inputs['Color'])
        else:
            links.new(vector('MULTIPLY', color, source_factor), emission.inputs['Color'])
        destination_factor = factor(destination)
        if isinstance(destination_factor, float):
            transparent.inputs['Color'].default_value = (destination_factor,) * 3 + (1.0,)
        else:
            links.new(destination_factor, transparent.inputs['Color'])
    add = nodes.new('ShaderNodeAddShader')
    links.new(emission.outputs[0], add.inputs[0])
    links.new(transparent.outputs[0], add.inputs[1])
    links.new(add.outputs[0], output.inputs['Surface'])
    material['sunrise_effect_blend'] = f'{source} + {destination}' + (' premultiplied' if blend.get('Premultiply', True) else '')
    if notes:
        material['sunrise_effect_note'] = '; '.join(notes)
    return material


def _constant(nodes, links, value):
    """A grey colour from a number or a value socket."""
    node = nodes.new('ShaderNodeCombineXYZ')
    for index in range(3):
        if isinstance(value, float):
            node.inputs[index].default_value = value
        else:
            links.new(value, node.inputs[index])
    return node.outputs[0]
