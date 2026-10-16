"""Places a world scene's lights: the game's point, spot and line lights, the time-of-day sun, the sky and the view."""

import math

import bpy
from mathutils import Matrix, Vector

from . import bundle, progress


LIGHTS = 'world.lights.json'
# Blender point and spot power P gives radiant intensity P / (4 pi) W/sr; sun strength is irradiance.
WATTS_PER_INTENSITY = 4 * math.pi
# Lights past this count stay hidden, the weakest first; EEVEE slows with many lights in view.
MAX_VISIBLE = 1024
# A line light becomes this many points at most.
MAX_LINE_POINTS = 16
MAX_LIGHTS = 200_000
# The view transform and look closest to the game's film curve.
VIEW_TRANSFORM, VIEW_LOOK = 'AgX', 'AgX - High Contrast'
# The game's auto exposure starts at 0.
EXPOSURE = 0.0
WINDOW_GROUP = 'Sunrise light window'
# A light volume gets a probe sample about this far apart, within these counts per axis.
VOLUME_SPACING, VOLUME_MIN, VOLUME_MAX = 2.0, 4, 32
# EEVEE's irradiance pool in MB; the 16 MB default cannot hold a map's volumes.
IRRADIANCE_POOL_MB = 256


def _numbers(values, count, name):
    if not isinstance(values, list) or len(values) != count \
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('World light field is invalid: ' + name)
    return [float(v) for v in values]


def _scalar(info, name, low=0.0):
    value = info.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < low:
        raise ValueError('World light field is invalid: ' + name)
    return float(value)


def _aim(obj, direction):
    """Lights shine along their local -Z."""
    vector = Vector(direction)
    if vector.length < 1e-6:
        vector = Vector((0, 0, -1))
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = (-vector).to_track_quat('Z', 'Y')


def _window_group():
    """Node group (1 - min(d / Range, 1)^4)^2 of the light ray's length d; one per file."""
    group = bpy.data.node_groups.get(WINDOW_GROUP)
    if group is not None and group.get('sunrise_light_window') == 1:
        return group
    group = bpy.data.node_groups.new(WINDOW_GROUP, 'ShaderNodeTree')
    group['sunrise_light_window'] = 1
    group.interface.new_socket('Range', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Window', in_out='OUTPUT', socket_type='NodeSocketFloat')
    nodes, links = group.nodes, group.links
    inputs, outputs, path = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput'), nodes.new('ShaderNodeLightPath')

    def math(operation, first, second, clamp=False):
        node = nodes.new('ShaderNodeMath')
        node.operation, node.use_clamp = operation, clamp
        for socket, value in zip(node.inputs, (first, second)):
            if isinstance(value, float):
                socket.default_value = value
            else:
                links.new(value, socket)
        return node.outputs[0]

    ratio = math('DIVIDE', path.outputs['Ray Length'], inputs.outputs['Range'])
    rest = math('SUBTRACT', 1.0, math('POWER', ratio, 4.0), clamp=True)
    links.new(math('POWER', rest, 2.0), outputs.inputs['Window'])
    return group


def _window(data):
    """Cycles ignores Custom Distance, so the light's own shader applies the same window."""
    data.use_nodes = True
    nodes, links = data.node_tree.nodes, data.node_tree.links
    window = nodes.new('ShaderNodeGroup')
    window.node_tree = _window_group()
    window.inputs['Range'].default_value = data.cutoff_distance
    links.new(window.outputs['Window'], nodes['Emission'].inputs['Strength'])


def _light(name, kind, info, energy, target, origin, location, objects, direction=None):
    """Creates the light data now and requests its object; returns the request."""
    data = bpy.data.lights.new(name, kind)
    data.color = _numbers(info['Color'], 3, 'Color')
    data.energy = energy
    data.use_shadow = bool(info.get('Shadows'))
    # The fit assumes soft falloff: irradiance I / (d^2 + Radius^2).
    data.use_soft_falloff = True
    data.shadow_soft_size = _scalar(info, 'Radius')
    # The fit assumes the custom distance window (1 - (d / Range)^4)^2.
    data.use_custom_distance = True
    data.cutoff_distance = max(_scalar(info, 'Range'), 0.01)
    _window(data)
    aim = direction if direction is not None else _numbers(info['Direction'], 3, 'Direction')
    request = objects.add(name, data)

    def setup(obj):
        target.link(obj, hidden=request.hidden)
        obj.parent = bundle._resolve(origin)
        obj.location = location
        _aim(obj, aim)
        obj['sunrise_light_id'] = info['Id']
        obj['sunrise_light_table'] = f'{info["TableTag"]:08X}'
        obj['sunrise_light_row'] = info['Row']
        obj['sunrise_material_tag'] = f'{info["MaterialTag"]:08X}'
        obj['sunrise_pixel_shader'] = f'{info["PixelShaderTag"]:08X}'
        obj['sunrise_bubble'] = f'{info["BubbleHash"]:08X}'
        obj['sunrise_light_fit'] = str(info.get('Fit', ''))
        obj['sunrise_light_copies'] = int(info.get('Copies', 1))
        obj['sunrise_light_animated'] = bool(info.get('Animated'))

    request.setup = setup
    return request


def _place(info, target, origin, objects):
    """One game light as Blender lights; a line becomes evenly spaced points."""
    kind = info.get('Kind')
    name = f'Light {info["Id"]} ({kind})'
    position = Vector(_numbers(info['Position'], 3, 'Position'))
    intensity = _scalar(info, 'Intensity')
    if kind == 'Point':
        return [_light(name, 'POINT', info, intensity * WATTS_PER_INTENSITY, target, origin, position, objects)]
    if kind == 'Spot':
        return _cone(name, info, intensity, target, origin, position, None, objects)
    if kind == 'Line':
        length = _scalar(info, 'Length')
        axis = Vector(_numbers(info['Direction'], 3, 'Direction')).normalized()
        count = info.get('Points', 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('World light field is invalid: Points')
        if count == 0:
            # Older exports carry no point count.
            spacing = max(_scalar(info, 'Range') / 2, _scalar(info, 'Radius') * 2, 0.25)
            count = math.ceil(length / spacing)
        count = min(max(count, 2), MAX_LINE_POINTS)
        # A line whose light favours one side carries the axis its points shine along.
        aim = _numbers(info['Aim'], 3, 'Aim') if info.get('Aim') is not None else None
        # Intensity is per metre; each point carries its share of the length.
        share = length / count
        result = []
        for index in range(count):
            offset = ((index + 0.5) / count - 0.5) * length
            at = position + axis * offset
            if aim is None:
                requests = [_light(f'{name} {index + 1}', 'POINT', info, intensity * share * WATTS_PER_INTENSITY, target, origin, at,
                                   objects)]
            else:
                requests = _cone(f'{name} {index + 1}', info, intensity * share, target, origin, at, aim, objects, share)
            for request in requests:
                request.props['sunrise_line_length'] = length
            result += requests
        return result
    raise ValueError('World light kind is unsupported: ' + str(kind))


def _cone(name, info, intensity, target, origin, location, direction, objects, share=1.0):
    """
    A spot, a point for the light the game sends outside the cone, since Blender spots end at 90 degrees,
    and a negative spot facing back that takes light away behind the light.
    """
    if direction is None:
        direction = _numbers(info['Direction'], 3, 'Direction')
    spot = _light(name, 'SPOT', info, intensity * WATTS_PER_INTENSITY, target, origin, location, objects, direction)
    spot.data.spot_size = math.radians(min(max(_scalar(info, 'SpotAngle'), 1.0), 180.0))
    spot.data.spot_blend = min(_scalar(info, 'SpotBlend'), 1.0)
    result = [spot]
    omni = _scalar(info, 'Omni') if 'Omni' in info else 0.0
    if omni > 0:
        result.append(_light(name + ' omni', 'POINT', info, omni * share * WATTS_PER_INTENSITY, target, origin, location, objects))
    back = _scalar(info, 'Back') if 'Back' in info else 0.0
    if back > 0:
        # The export keeps the back spot at or below the omni light, so no surface goes below black.
        dark = _light(name + ' back', 'SPOT', info, -back * share * WATTS_PER_INTENSITY, target, origin, location, objects,
                      [-v for v in direction])
        dark.data.spot_size = math.radians(min(max(_scalar(info, 'BackAngle'), 1.0), 180.0))
        dark.data.spot_blend = min(_scalar(info, 'BackBlend'), 1.0)
        dark.props['sunrise_negative'] = True
        result.append(dark)
    return result


def _sun(info, target, origin, name, objects):
    data = bpy.data.lights.new(name + ' sun', 'SUN')
    data.color = _numbers(info['Color'], 3, 'Color')
    data.energy = _scalar(info, 'Irradiance')
    data.use_shadow = True
    # Direction points at the sun; the light shines the other way.
    aim = [-v for v in _numbers(info['Direction'], 3, 'Direction')]
    time_of_day, curve_set = _scalar(info, 'TimeOfDay'), f'{info["CurveSetTag"]:08X}'
    # The game casts sun shadows along this direction; one Blender sun shades and shadows along Direction.
    shadow = _numbers(info['ShadowDirection'], 3, 'ShadowDirection') if info.get('ShadowDirection') is not None else None

    def setup(obj):
        target.link(obj)
        obj.parent = bundle._resolve(origin)
        _aim(obj, aim)
        obj['sunrise_time_of_day'] = time_of_day
        obj['sunrise_curve_set'] = curve_set
        if shadow is not None:
            obj['sunrise_shadow_direction'] = shadow

    return objects.add(name + ' sun', data, setup)

def _image(root, relative):
    image = bpy.data.images.load(str(bundle._path(root, relative)), check_existing=False)
    image.colorspace_settings.name = 'Linear Rec.709'
    return image


def _world(info, root, name):
    """
    A world that shows the game's sky to the camera and lights surfaces with the game's diffuse sky light.
    Every other ray, glossy ones too, sees the light image.
    """
    world = bpy.data.worlds.new(name + ' sky')
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputWorld')
    mix = nodes.new('ShaderNodeMixShader')
    path = nodes.new('ShaderNodeLightPath')
    shaders = []
    for key in ('LightPath', 'Path'):
        texture = nodes.new('ShaderNodeTexEnvironment')
        texture.image = _image(root, info[key])
        texture.name = texture.label = 'Sky light' if key == 'LightPath' else 'Sky'
        background = nodes.new('ShaderNodeBackground')
        links.new(texture.outputs['Color'], background.inputs['Color'])
        shaders.append(background)
    links.new(path.outputs['Is Camera Ray'], mix.inputs['Fac'])
    links.new(shaders[0].outputs['Background'], mix.inputs[1])
    links.new(shaders[1].outputs['Background'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    world['sunrise_time_of_day'] = _scalar(info, 'TimeOfDay')
    world['sunrise_atmosphere'] = f'{info["PlacementListTag"]:08X}'
    world['sunrise_sky_lookup'] = f'{info["LookupTag"]:08X}'
    return world


def _volumes(volumes, target, origin, name, objects):
    """
    One EEVEE volume light probe per game cubemap box. Baked, they shade the sky light by what blocks it,
    as the game's indoor cubemaps do; Cycles traces it instead and ignores them.
    """
    result = []
    for index, info in enumerate(volumes):
        extents = _numbers(info['Extents'], 3, 'Extents')
        if min(extents) <= 0:
            raise ValueError('World light field is invalid: Extents')
        data = bpy.data.lightprobes.new(f'{name} light volume {index}', 'VOLUME')
        data.resolution_x, data.resolution_y, data.resolution_z = (
            min(max(round(2 * e / VOLUME_SPACING), VOLUME_MIN), VOLUME_MAX) for e in extents)
        matrix = bundle._matrix(_numbers(info['Transform'], 16, 'Transform')) @ Matrix.Diagonal((*extents, 1))
        props = {'sunrise_bubble': f'{info["BubbleHash"]:08X}', 'sunrise_placement_source': f'{info["PlacementSourceTag"]:08X}'}

        def setup(obj, matrix=matrix):
            target.link(obj)
            obj.parent = bundle._resolve(origin)
            obj.matrix_basis = matrix

        request = objects.add(data.name, data, setup)
        request.props.update(props)
        result.append(request)
    return result

def _bake(scene, volumes):
    """Bakes the volumes before returning, in the interface too."""
    if not volumes:
        return 'no light volumes'
    if int(scene.eevee.gi_irradiance_pool_size) < IRRADIANCE_POOL_MB:
        scene.eevee.gi_irradiance_pool_size = str(IRRADIANCE_POOL_MB)
    try:
        with bpy.context.temp_override(scene=scene, active_object=volumes[0], object=volumes[0]):
            result = bpy.ops.object.lightprobe_cache_bake(subset='ALL')
        return f'{len(volumes)} light volumes, bake {sorted(result)[0].lower()}'
    except RuntimeError as error:
        # A failed bake leaves unbaked volumes, which EEVEE treats as absent.
        return f'light volume bake failed: {error}'


def _view(scene):
    """The view transform closest to the game's film curve, at the game's starting exposure."""
    scene.view_settings.view_transform = VIEW_TRANSFORM
    scene.view_settings.look = VIEW_LOOK
    scene.view_settings.exposure = EXPOSURE
    scene['sunrise_view_transform'] = VIEW_TRANSFORM
    scene['sunrise_view_look'] = VIEW_LOOK
    scene['sunrise_exposure'] = EXPOSURE


def light_steps(manifest_path, collection, origin, objects):
    """
    Requests the lights, the sun and the light volumes beside a world manifest, in a child collection of the world.
    Returns the state finish and sky need, or None when the export has no lights file or no lights.
    """
    path = manifest_path.parent / LIGHTS
    if not path.exists():
        return None
    data = bundle._read(path)
    if data.get('Format') != 'sunrise-world-lights' or data.get('Version') != 1:
        raise ValueError('Unsupported Sunrise world lights.')
    lights = data.get('Lights')
    if not isinstance(lights, list) or len(lights) > MAX_LIGHTS:
        raise ValueError('World light table is invalid.')
    if not lights and not isinstance(data.get('Sun'), dict) and not isinstance(data.get('Sky'), dict):
        return None
    volumes = data.get('Volumes', [])
    if not isinstance(volumes, list) or len(volumes) > MAX_LIGHTS:
        raise ValueError('World light volume table is invalid.')
    holder = bpy.data.collections.new(collection.name + ' lights')
    collection.children.link(holder)
    holder['sunrise_world_lights'] = str(path)
    target = bundle._Layout(holder)
    placed = []
    for index, info in enumerate(lights):
        requests = _place(info, target, origin, objects)
        for request in requests:
            request.hidden = not info.get('Enabled', True)
        placed += requests
        yield 'Lights', index + 1, len(lights)
    visible = [(request.data.energy * request.data.cutoff_distance, request) for request in placed if not request.hidden]
    # Negative lights sort last, so the cap hides every one of them before any positive light.
    visible.sort(key=lambda pair: pair[0], reverse=True)
    for _, request in visible[MAX_VISIBLE:]:
        request.hidden = True
        request.props['sunrise_hidden_over_cap'] = True
    if isinstance(data.get('Sun'), dict):
        _sun(data['Sun'], target, origin, collection.name, objects)
    probes = _volumes(volumes, target, origin, collection.name, objects)
    return {'path': path, 'data': data, 'target': target, 'probes': probes, 'blender_lights': len(placed),
            'hidden_over_cap': max(len(visible) - MAX_VISIBLE, 0)}


def finish_lights(state):
    """Writes the hiding of the light objects once they are set up; returns the volume objects for _bake."""
    target = state['target']
    target.collection['sunrise_light_counts'] = str({'lights': len(state['data']['Lights']), 'blender_lights': state['blender_lights'],
                                                     'hidden_over_cap': state['hidden_over_cap'], 'light_volumes': len(state['probes'])})
    target.write(refresh=True)
    return [request.obj for request in state['probes']]


def set_sky(state, name):
    """Sets the scene's world to the game's sky and the view to the game's film curve; older exports carry no sky."""
    sky = state['data'].get('Sky')
    if isinstance(sky, dict):
        scene = bpy.context.scene
        scene.world = _world(sky, state['path'].parent, name)
        _view(scene)
        state['target'].collection['sunrise_world_sky'] = scene.world.name


def import_lights(manifest_path, collection, origin):
    """
    Adds the lights beside a world manifest to a child collection of the world and bakes its light volumes.
    Returns None when the export has no lights file or no lights.
    """
    kinds = ('objects', 'collections', 'lights', 'worlds', 'images', 'lightprobes', 'node_groups')
    before = progress.snapshot(kinds)
    scene = bpy.context.scene
    view = (scene.world, scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure)
    try:
        objects = bundle._Objects()
        state = progress.run(light_steps(manifest_path, collection, origin, objects))
        if state is None:
            return None
        progress.run(objects.create('Lights'))
        progress.run(objects.set_up(lambda index: 'Lights'))
        probes = finish_lights(state)
        set_sky(state, collection.name)
        bpy.context.view_layer.update()
        state['target'].collection['sunrise_light_volume_bake'] = _bake(scene, probes)
        return state['target'].collection
    except BaseException:
        scene.world = view[0]
        scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure = view[1:]
        progress.rollback(before)
        raise