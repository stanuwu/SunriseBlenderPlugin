"""Bundle edits retain native identity; preview shaders and bone poses stay in Blender."""

import array
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
from urllib.parse import unquote
import uuid

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector

if __package__:
    from . import progress
else:
    # Run as a single file, the module loads the progress module beside it.
    import importlib.util
    _spec = importlib.util.spec_from_file_location('sunrise_progress', Path(__file__).with_name('progress.py'))
    progress = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(progress)


MANIFEST = "asset.sunrise-bundle.json"
EDITS = "asset.edited.sunrise-bundle.json"
MAX_BYTES = 256 * 1024 * 1024
# DXGI formats stored as sRGB.
SRGB_FORMATS = (29, 72, 75, 78, 91, 93, 99)
_ui_documents = {}


def _reject_links(path):
    for current in (path, *path.parents):
        if current.exists() or current.is_symlink():
            if current.is_symlink() or getattr(current.lstat(), "st_file_attributes", 0) & 0x400:
                raise ValueError("Bundle paths cannot use links or junctions.")
        if current.is_dir() and (current / "sunrise.project.json").exists():
            raise ValueError("Working bundles must stay outside delta projects.")


def _path(root, relative):
    parts = str(relative).replace("\\", "/").split("/")
    if not relative or any(p in ("", ".", "..") or ":" in p for p in parts):
        raise ValueError("Bundle path leaves its directory.")
    result = Path(os.path.abspath(root.joinpath(*parts)))
    result.relative_to(root)
    _reject_links(result)
    return result


def _read(path):
    _reject_links(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Bundle document exceeds its size limit.")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path):
    _reject_links(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write(root, relative, document):
    path = _path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
        _reject_links(path)
        os.replace(temporary, path)
    finally:
        _reject_links(temporary)
        temporary.unlink(missing_ok=True)
    return path


def _matrix(values):
    if len(values) != 16 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid source transform.")
    return Matrix([values[i:i + 4] for i in range(0, 16, 4)]).transposed()


def _flat(matrix):
    return [float(v) for row in matrix for v in row]


def _close(a, b, tolerance=0.00001):
    return len(a) == len(b) and all(math.isfinite(float(x)) and abs(x - y) <= tolerance
                                  for x, y in zip(a, b))


def _same_vectors(a, b):
    return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))


# Longest data block name, in UTF-8 bytes.
NAME_BYTES = 255


def _clip(name, size=NAME_BYTES):
    return name.encode('utf-8')[:size].decode('utf-8', 'ignore')


class _Names:
    """Unused names for one kind of data block; Blender's own search slows as a name repeats."""
    def __init__(self, table):
        self.used = {value.name for value in table}
        self.counts = {}

    def __call__(self, name):
        name = _clip(name)
        if name not in self.used:
            self.used.add(name)
            return name
        number = self.counts.get(name, 0)
        while True:
            number += 1
            suffix = f'.{number:03d}'
            candidate = _clip(name, NAME_BYTES - len(suffix)) + suffix
            if candidate not in self.used:
                self.counts[name] = number
                self.used.add(candidate)
                return candidate


class _Request:
    """An object an import will create; setup runs once every requested object exists, then props are set."""
    __slots__ = ('name', 'data', 'setup', 'obj', 'props', 'hidden')

    def __init__(self, name, data, setup):
        self.name, self.data, self.setup, self.obj = name, data, setup, None
        self.props, self.hidden = {}, False


def _resolve(value):
    """The object of a request, or the value itself when it is already an object."""
    return value.obj if isinstance(value, _Request) else value


class _Objects:
    """
    The objects of one import, created together in name order and then set up in request order.
    Blender keeps objects sorted by name and finds a new object's place from the end of the list,
    so an object created out of name order costs time in proportion to every object after it.
    """
    STEP = 2000

    def __init__(self):
        self.names = _Names(bpy.data.objects)
        self.requests = []

    def add(self, name, data=None, setup=None):
        request = _Request(self.names(name), data, setup)
        self.requests.append(request)
        return request

    def create(self, phase):
        """Steps that create every requested object; the order matches Blender's case-blind byte order."""
        order = sorted(self.requests, key=lambda request: request.name.encode('utf-8').lower())
        for count, request in enumerate(order, 1):
            request.obj = bpy.data.objects.new(request.name, request.data)
            if count % self.STEP == 0:
                yield phase, count, len(order)

    def set_up(self, phase_of):
        """Steps that run each request's setup in request order; phase_of(request index) names the phase."""
        for index, request in enumerate(self.requests):
            if request.setup is not None:
                request.setup(request.obj)
            for key, value in request.props.items():
                request.obj[key] = value
            if index % self.STEP == 0:
                yield phase_of(index), index, len(self.requests)


class _Layout:
    """
    Transforms and hiding for a collection's objects, written in one pass when the collection is done.
    A single hide write updates every collection, and a single matrix check is slow in Python.
    """
    IDENTITY = np.eye(4, dtype=np.float32).reshape(-1)

    def __init__(self, collection):
        self.collection = collection
        self.matrices, self.hidden, self.rows = [], [], {}

    def link(self, obj, matrix=None, hidden=False):
        """
        matrix is a placement's 16 values in row-vector order, which is Blender's memory order.
        When any object has one, an object without one gets the identity; otherwise transforms stay as set.
        """
        self.collection.objects.link(obj)
        self.rows[obj.as_pointer()] = len(self.matrices)
        self.matrices.append(matrix)
        self.hidden.append(hidden)

    def hide(self, obj, hidden=True):
        self.hidden[self.rows[obj.as_pointer()]] = hidden

    def is_hidden(self, obj):
        return self.hidden[self.rows[obj.as_pointer()]]

    def write(self, refresh=False):
        """A matrix the object cannot hold as location, rotation and scale goes to its parent inverse."""
        objects = self.collection.objects
        if len(objects) != len(self.matrices):
            raise RuntimeError('Collection objects changed during the import.')
        if any(matrix is not None for matrix in self.matrices):
            self._transforms(objects)
        hidden = np.array(self.hidden, dtype=bool)
        if hidden.any():
            objects.foreach_set('hide_render', hidden)
            objects.foreach_set('hide_viewport', hidden)
            if refresh:
                # A bulk write skips the view layer update; one ordinary write runs it.
                first = objects[int(np.argmax(hidden))]
                first.hide_viewport = True

    def _transforms(self, objects):
        basis = np.tile(self.IDENTITY, (len(self.matrices), 1))
        for index, matrix in enumerate(self.matrices):
            if matrix is not None:
                basis[index] = matrix
        objects.foreach_set('matrix_basis', basis.reshape(-1))
        held = np.empty(basis.size, dtype=np.float32)
        objects.foreach_get('matrix_basis', held)
        held = held.reshape(basis.shape).astype(np.float64)
        moved = ~(np.isfinite(held).all(axis=1) & (np.abs(held - basis) <= 0.00001).all(axis=1))
        if moved.any():
            inverse = np.tile(self.IDENTITY, (len(basis), 1))
            inverse[moved], basis[moved] = basis[moved], self.IDENTITY
            objects.foreach_set('matrix_parent_inverse', inverse.reshape(-1))
            objects.foreach_set('matrix_basis', basis.reshape(-1))


def _uid(block):
    if not block.get("sunrise_id"):
        block["sunrise_id"] = uuid.uuid4().hex
    return block["sunrise_id"]


def _curves(action):
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    yield bag, curve


def _action_state(action):
    """Preview actions are compared without changing their curve data."""
    digest = hashlib.sha256(b'v2')
    for bag, curve in _curves(action):
        points = curve.keyframe_points
        count = len(points)
        digest.update(f'{bag.slot_handle}|{curve.data_path}|{curve.array_index}|{count}|'
                      f'{",".join(m.type for m in curve.modifiers)}\n'.encode())
        for field, width, code in (('co', 2, 'f'), ('handle_left', 2, 'f'), ('handle_right', 2, 'f'), ('interpolation', 1, 'i')):
            values = array.array(code, bytes(4 * width * count))
            points.foreach_get(field, values)
            digest.update(values)
    return 'v2:' + digest.hexdigest()


def _action_unchanged(action, stored):
    """A stored fingerprint without the v2: prefix was saved by an older add-on."""
    if isinstance(stored, str) and stored.startswith('v2:'):
        return _action_state(action) == stored
    return _action_state_v1(action) == stored


def _action_state_v1(action):
    values = []
    names = {item.value: item.identifier for item in bpy.types.Keyframe.bl_rna.properties['interpolation'].enum_items}
    for bag, curve in _curves(action):
        count = len(curve.keyframe_points)
        co, left, right = (array.array('f', [0]) * (count * 2) for _ in range(3))
        interpolation = array.array('i', [0]) * count
        curve.keyframe_points.foreach_get("co", co)
        curve.keyframe_points.foreach_get("handle_left", left)
        curve.keyframe_points.foreach_get("handle_right", right)
        curve.keyframe_points.foreach_get("interpolation", interpolation)
        keys = [(list(co[i * 2:i * 2 + 2]), list(left[i * 2:i * 2 + 2]),
                 list(right[i * 2:i * 2 + 2]), names[interpolation[i]]) for i in range(count)]
        values.append((bag.slot_handle, curve.data_path, curve.array_index,
                       keys, [m.type for m in curve.modifiers]))
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _rig_state(obj):
    return {"world": _flat(obj.matrix_world), "bones": [
        [b.name, b.parent.name if b.parent else None, _flat(b.matrix_local)] for b in obj.data.bones],
        "constraints": [c.type for b in obj.pose.bones for c in b.constraints]}


def _normal_state(mesh):
    """Derived normals may rotate after position edits without changing authored normal storage."""
    digest = hashlib.sha256()
    for name in ("custom_normal", "sharp_edge", "sharp_face"):
        value = mesh.attributes.get(name)
        digest.update(name.encode())
        if value is None:
            continue
        width = 2 if value.data_type == "INT16_2D" else 1
        numbers = array.array('i', [0]) * (len(value.data) * width)
        value.data.foreach_get("value", numbers)
        digest.update(numbers.tobytes())
    return digest.hexdigest()


def _manifest_path(path):
    result = Path(os.path.abspath(bpy.path.abspath(str(path))))
    _reject_links(result)
    if result.name != MANIFEST:
        raise ValueError("Choose asset.sunrise-bundle.json.")
    return result


def _bundle(path=None, verify=True):
    if path is None:
        path = bpy.context.scene.get("sunrise_active_bundle")
    if not path:
        raise ValueError("Import a Sunrise bundle first.")
    path = _manifest_path(path)
    matches = [c for c in bpy.data.collections if c.get("sunrise_manifest") == str(path)]
    if len(matches) != 1:
        raise ValueError("The bundle collection is missing or duplicated.")
    collection = matches[0]
    if verify and _sha(path) != collection["sunrise_manifest_sha256"]:
        raise ValueError("The source manifest changed after import.")
    if verify:
        document = _read(path)
    else:
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        if str(path) not in _ui_documents or _ui_documents[str(path)][0] != stamp:
            _ui_documents[str(path)] = (stamp, _read(path))
        document = _ui_documents[str(path)][1]
    return path, collection, document


def _validate_gltf(root, path):
    document = _read(path)
    if document.get("asset", {}).get("version") != "2.0" or document.get("extensionsRequired"):
        raise ValueError("The bundle needs core glTF 2.0.")
    for entry in document.get("buffers", []) + document.get("images", []):
        uri = entry.get("uri", "")
        if uri and not uri.startswith("data:"):
            resolved = _path(path.parent, unquote(uri))
            resolved.relative_to(root)
        if entry.get("byteLength", 0) > MAX_BYTES:
            raise ValueError("A glTF buffer exceeds its size limit.")
    return document


def _motion_state(group, basis=False):
    """Generated deformation graphs are preview data and cannot become native geometry edits."""
    nodes = []
    def plain(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, bpy.types.ID):
            return (value.bl_rna.identifier, value.name)
        return [plain(item) for item in value]
    for item in group.nodes:
        defaults = []
        for socket in item.inputs:
            if hasattr(socket, 'default_value'):
                value = socket.default_value
                defaults.append(plain(value) if basis else value if isinstance(value, (str, int, float, bool)) else list(value))
        nodes.append((item.name, item.bl_idname, item.mute, getattr(item, 'operation', None),
                      getattr(item, 'data_type', None), defaults))
        if basis:
            nodes.append((getattr(item, 'domain', None), getattr(item, 'transform_space', None),
                          getattr(item, 'use_clamp', None)))
    links = [(link.from_node.name, list(link.from_node.outputs).index(link.from_socket),
              link.to_node.name, list(link.to_node.inputs).index(link.to_socket)) for link in group.links]
    return hashlib.sha256(json.dumps((nodes, links), sort_keys=True).encode()).hexdigest()


def _motion_image(texture):
    """A loaded bundle image of a vertex-stage texture, in the colour space the program reads it in."""
    tag = f'{int(texture["TextureTag"]):08X}'
    image = next((item for item in bpy.data.images if item.get('sunrise_texture_tag') == tag and not item.get('sunrise_volume')
                  and item.get('sunrise_mip', 0) == 0 and item.get('sunrise_slice', 0) == 0 and item.get('sunrise_depth', 0) == 0), None)
    return _preview_image(image, bool(texture.get('Srgb'))) if image is not None else None


def _node_math(group):
    """connect(value, socket) and math_node(operation, *inputs) that build plain math nodes in group."""
    def connect(value, socket):
        if isinstance(value, (int, float)):
            socket.default_value = value
        else:
            group.links.new(value, socket)

    def math_node(operation, *inputs):
        value = group.nodes.new('ShaderNodeMath')
        value.operation = operation
        for item, socket in zip(inputs, value.inputs):
            connect(item, socket)
        return value.outputs[0]

    return connect, math_node


def _offset_group(name):
    """A geometry node group that moves each point by its Set Position node's Offset."""
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    source = group.nodes.new('NodeGroupInput')
    target = group.nodes.new('NodeGroupOutput')
    position = group.nodes.new('GeometryNodeSetPosition')
    group.links.new(source.outputs['Geometry'], position.inputs['Geometry'])
    group.links.new(position.outputs['Geometry'], target.inputs['Geometry'])
    return group, position


def _attach_motion(obj, group, name, reorder):
    """Adds the motion modifier; reorder puts it first, before the armature."""
    modifier = obj.modifiers.new(name, 'NODES')
    modifier.node_group = group
    if reorder:
        with bpy.context.temp_override(object=obj, active_object=obj):
            bpy.ops.object.modifier_move_to_index(modifier=modifier.name, index=0)
    group['sunrise_motion_state'] = _motion_state(group)
    obj['sunrise_motion_group'] = group.name


def _vertex_program_group(motion):
    """The "Offset" of a vertex program as geometry nodes; each stream reads the mesh's own attribute."""
    group, position = _offset_group('Sunrise vertex program')
    streams, values = {}, {}
    connect, math_node = _node_math(group)

    def stream(operation, channel):
        if operation not in streams:
            if operation == 'time':
                streams[operation] = [group.nodes.new('GeometryNodeInputSceneTime').outputs['Seconds']]
            elif operation == 'color':
                attribute = group.nodes.new('GeometryNodeInputNamedAttribute')
                attribute.data_type = 'FLOAT_COLOR'
                attribute.inputs['Name'].default_value = 'sunrise_color'
                split = group.nodes.new('FunctionNodeSeparateColor')
                group.links.new(attribute.outputs['Attribute'], split.inputs['Color'])
                streams[operation] = list(split.outputs)
            else:
                if operation == 'normal':
                    vector = group.nodes.new('GeometryNodeInputNormal').outputs['Normal']
                else:
                    attribute = group.nodes.new('GeometryNodeInputNamedAttribute')
                    attribute.data_type = 'FLOAT_VECTOR'
                    attribute.inputs['Name'].default_value = {'rawposition': 'sunrise_rawposition', 'uv': 'UVMap',
                                                              'uvzw': 'sunrise_uvzw'}[operation]
                    vector = attribute.outputs['Attribute']
                split = group.nodes.new('ShaderNodeSeparateXYZ')
                group.links.new(vector, split.inputs[0])
                streams[operation] = list(split.outputs)
        return streams[operation][channel]

    try:
        _vertex_program_nodes(group, motion, position, connect, math_node, stream, values)
    except ValueError:
        bpy.data.node_groups.remove(group)
        raise
    return group


def _vertex_program_nodes(group, motion, position, connect, math_node, stream, values):
    """The program's nodes in order; an input always precedes its node."""
    program, fetches = motion['Program'], {}
    nodes = program['Nodes']
    for index, item in enumerate(nodes):
        operation = item['Operation']
        if any(value < 0 or value >= index for value in item['Inputs']):
            raise ValueError('Vertex program is not an ordered expression graph.')
        if operation == 'constant':
            values[index] = float(item['Value'])
        elif operation in ('time', 'color', 'normal', 'rawposition', 'uv', 'uvzw'):
            values[index] = stream(operation, int(item['Channel']))
        elif operation == 'sample':
            key = ('sample', int(item['Slot']), *item['Inputs'])
            if key not in fetches:
                texture = (motion.get('Textures') or {}).get(str(item['Slot']))
                image = _motion_image(texture) if texture else None
                if image is None:
                    raise ValueError('Vertex program needs vertex texture slot ' + str(item['Slot']))
                fetch = group.nodes.new('GeometryNodeImageTexture')
                fetch.inputs['Image'].default_value = image
                fetch.extension = 'EXTEND' if texture.get('Clamp') else 'REPEAT'
                point = group.nodes.new('ShaderNodeCombineXYZ')
                connect(values[item['Inputs'][0]], point.inputs['X'])
                connect(math_node('SUBTRACT', 1, values[item['Inputs'][1]]), point.inputs['Y'])
                group.links.new(point.outputs[0], fetch.inputs['Vector'])
                split = group.nodes.new('FunctionNodeSeparateColor')
                group.links.new(fetch.outputs['Color'], split.inputs['Color'])
                fetches[key] = [*list(split.outputs)[:3], fetch.outputs['Alpha']]
            values[index] = fetches[key][int(item['Channel'])]
        else:
            values[index] = _program_math(program, item, [values[value] for value in item['Inputs']], math_node)
            if values[index] is None:
                raise ValueError('Unsupported vertex program operation: ' + operation)
    offset = group.nodes.new('ShaderNodeCombineXYZ')
    for axis, index in enumerate(program['Outputs']['Offset']):
        connect(values[index], offset.inputs[axis])
    group.links.new(offset.outputs[0], position.inputs['Offset'])


def _native_vertex_motion(obj, motion, reorder=True):
    """Vertex-shader displacement precedes the armature, matching the native vertex shader."""
    if motion.get('Profile') == 'VertexProgram':
        try:
            group = _vertex_program_group(motion)
        except ValueError as error:
            # A newer export or a missing image costs this part its motion, not the import.
            print(f'Sunrise vertex motion: {obj.name} keeps its rest shape: {error}')
            return
        _attach_motion(obj, group, 'Shader motion', reorder)
        return
    if motion.get('Profile') != 'VertexWave5':
        raise ValueError('Unsupported native vertex motion profile.')
    colors = motion['Colors']
    if len(colors) != len(obj.data.vertices):
        raise ValueError('Native motion colours do not match source vertices.')
    attribute = obj.data.attributes.new('sunrise_motion_color', 'FLOAT_COLOR', 'POINT')
    attribute.data.foreach_set('color', [c for value in colors for c in value])
    group, position = _offset_group('Sunrise passive motion')
    clock = group.nodes.new('GeometryNodeInputSceneTime')
    native_color = group.nodes.new('GeometryNodeInputNamedAttribute')
    native_color.data_type = 'FLOAT_COLOR'
    native_color.inputs['Name'].default_value = attribute.name
    channels = group.nodes.new('FunctionNodeSeparateColor')
    group.links.new(native_color.outputs['Attribute'], channels.inputs['Color'])
    connect, math_node = _node_math(group)

    def constant(row, lane=0):
        value, rate = motion['Constants'][row][lane], motion['Rates'][row][lane]
        return math_node('MULTIPLY_ADD', clock.outputs['Seconds'], rate, value) if rate else value

    phase = math_node('MULTIPLY_ADD', math_node('ADD', channels.outputs['Green'], constant(2)), constant(3, 3), constant(3, 2))
    phase = math_node('SUBTRACT', phase, math_node('ROUND', phase))
    wave = math_node('MULTIPLY', phase, math_node('MULTIPLY_ADD', math_node('ABSOLUTE', phase), -16, 8))
    wave = math_node('MULTIPLY', wave, math_node('MULTIPLY_ADD', math_node('ABSOLUTE', wave), .225, .775))
    wave = math_node('MULTIPLY_ADD', wave, constant(3), constant(3, 1))
    weight = math_node('MULTIPLY_ADD', channels.outputs['Blue'], constant(1), channels.outputs['Red'])
    offsets = [constant(0), constant(0), math_node('MULTIPLY', wave, weight)]
    combine = group.nodes.new('ShaderNodeCombineXYZ')
    for axis, offset in enumerate(offsets):
        bounded = math_node('MINIMUM', math_node('MAXIMUM', offset, math_node('MULTIPLY', constant(4, axis), -1)), constant(4, axis))
        connect(bounded, combine.inputs[axis])
    group.links.new(combine.outputs[0], position.inputs['Offset'])
    _attach_motion(obj, group, 'Passive motion', reorder)


def _native_basis_modifier(obj):
    """Native N/T normalization and bitangent construction run per point before interpolation."""
    group = bpy.data.node_groups.new('Sunrise native basis', 'GeometryNodeTree')
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    source = group.nodes.new('NodeGroupInput')
    target = group.nodes.new('NodeGroupOutput')
    self_object = group.nodes.new('GeometryNodeSelfObject')
    info = group.nodes.new('GeometryNodeObjectInfo')
    info.transform_space = 'ORIGINAL'
    group.links.new(self_object.outputs['Self Object'], info.inputs['Object'])
    connect, math_node = _node_math(group)

    def vector(operation, *inputs):
        value = group.nodes.new('ShaderNodeVectorMath')
        value.operation = operation
        for index, item in enumerate(inputs):
            connect(item, value.inputs['Scale'] if operation == 'SCALE' and index == 1 else value.inputs[index])
        return value.outputs['Value'] if operation in ('LENGTH', 'DOT_PRODUCT') else value.outputs['Vector']

    original, transformed, present = [], [], []
    for name in ('normal', 'tangent', 'bitangent'):
        attribute = group.nodes.new('GeometryNodeInputNamedAttribute')
        attribute.data_type = 'FLOAT_VECTOR'
        attribute.inputs['Name'].default_value = 'sunrise_native_' + name
        original.append(attribute.outputs['Attribute'])
        present.append(attribute.outputs['Exists'])
        transform = group.nodes.new('FunctionNodeTransformDirection')
        connect(attribute.outputs['Attribute'], transform.inputs['Direction'])
        connect(info.outputs['Transform'], transform.inputs['Transform'])
        transformed.append(transform.outputs['Direction'])
    length = vector('LENGTH', transformed[0])
    reciprocal = math_node('DIVIDE', 1, math_node('MAXIMUM', length, .000001))
    normal, tangent = [vector('SCALE', value, reciprocal) for value in transformed[:2]]
    cross = vector('CROSS_PRODUCT', original[0], original[1])
    denominator = vector('DOT_PRODUCT', cross, cross)
    orientation = vector('DOT_PRODUCT', cross, original[2])
    handedness = math_node('DIVIDE', orientation, math_node('MAXIMUM', denominator, 1e-12))
    bitangent = vector('SCALE', vector('CROSS_PRODUCT', normal, tangent), handedness)
    determinant_sign = math_node('MULTIPLY', orientation,
        vector('DOT_PRODUCT', vector('CROSS_PRODUCT', transformed[0], transformed[1]), transformed[2]))
    reflected = math_node('LESS_THAN', determinant_sign, 0)
    available = math_node('MULTIPLY', math_node('GREATER_THAN', length, .000001),
                          math_node('GREATER_THAN', denominator, 1e-12))
    for value in present:
        available = math_node('MULTIPLY', available, value)
    geometry = source.outputs['Geometry']
    for name, value, kind in (('normal', normal, 'FLOAT_VECTOR'), ('tangent', tangent, 'FLOAT_VECTOR'),
                              ('bitangent', bitangent, 'FLOAT_VECTOR'), ('basis', available, 'FLOAT'),
                              ('reflected', reflected, 'FLOAT')):
        store = group.nodes.new('GeometryNodeStoreNamedAttribute')
        store.domain, store.data_type = 'POINT', kind
        store.inputs['Name'].default_value = 'sunrise_shader_' + name
        connect(geometry, store.inputs['Geometry']); connect(value, store.inputs['Value'])
        geometry = store.outputs['Geometry']
    connect(geometry, target.inputs['Geometry'])
    modifier = obj.modifiers.new('Native shader basis', 'NODES')
    modifier.node_group = group
    group['sunrise_basis_state'] = _motion_state(group, basis=True)
    obj['sunrise_basis_group'] = group.name


def _faces(indices):
    """Triangles Blender can hold; one that repeats a vertex stays only in the source."""
    return [indices[i:i + 3] for i in range(0, len(indices), 3) if len(set(indices[i:i + 3])) == 3]


def _vertex_streams(mesh, colors, zw, raw=None):
    """COLOR0, the raw UV z and w and raw POSITION0 the shader reads; point attributes, so they are not UV maps."""
    for name, values, kind, field, width in (('sunrise_color', colors, 'FLOAT_COLOR', 'color', 4),
                                             ('sunrise_uvzw', zw, 'FLOAT_VECTOR', 'vector', 2),
                                             ('sunrise_rawposition', raw, 'FLOAT_VECTOR', 'vector', 3)):
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32).reshape((-1, width))
        if len(values) != len(mesh.vertices) or not np.isfinite(values).all():
            raise ValueError('Native vertex stream does not match its source vertices.')
        if width == 2:
            values = np.hstack((values, np.zeros((len(values), 1), dtype=np.float32)))
        attribute = mesh.attributes.new(name, kind, 'POINT')
        attribute.data.foreach_set(field, values.reshape(-1))


# Studio's water preview profile; its program is the whole surface.
WATER_RECIPE = 'Water'
# Refractive index of water.
WATER_IOR = 1.333


def _bitangent_sign(mesh):
    """Per-corner MikkTSpace bitangent sign; authored frames share it, so shaders build B as cross(N, T) * sign."""
    if not mesh.polygons or 'UVMap' not in mesh.uv_layers:
        return
    try:
        mesh.calc_tangents(uvmap='UVMap')
    except RuntimeError:
        return
    signs = np.empty(len(mesh.loops), dtype=np.float32)
    mesh.loops.foreach_get('bitangent_sign', signs)
    mesh.free_tangents()
    attribute = mesh.attributes.new('sunrise_bitangent_sign', 'FLOAT', 'CORNER')
    attribute.data.foreach_set('value', signs)


def _mesh_baseline(obj, info, geometry, bones, basis_preview=False):
    """Source vertex identities do not depend on the glTF importer's vertex ordering."""
    indices = geometry["Indices"]
    if len(indices) % 3 or any(i < 0 or i >= len(geometry["Positions"]) for i in indices):
        raise ValueError("Geometry has invalid triangle indices.")
    replaced = obj.data
    mesh = bpy.data.meshes.new(obj.name)
    mesh.from_pydata(geometry["Positions"], [], _faces(indices))
    obj.data = mesh
    # The glTF mesh this replaces holds the object's name until it is removed.
    if replaced is not None and replaced.users == 0:
        bpy.data.meshes.remove(replaced)
    mesh.name = obj.name
    uv = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        uv.data[loop.index].uv = geometry["TexCoords"][loop.vertex_index]
    if geometry.get("TexCoords1") is not None:
        # Detail UVs the vertex shader derives; preview only, never exported as an edit.
        detail = mesh.uv_layers.new(name="UVMap1")
        for loop in mesh.loops:
            detail.data[loop.index].uv = geometry["TexCoords1"][loop.vertex_index]
    _vertex_streams(mesh, geometry.get("Colors"), geometry.get("TexCoordsZw"), geometry.get("RawPositions"))
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.normals_split_custom_set_from_vertices(geometry["Normals"])
    _bitangent_sign(mesh)
    if geometry.get('Tangents') is not None and geometry.get('Joints') is None:
        tangents = geometry['Tangents']
        if len(tangents) != len(mesh.vertices) or any(len(value) != 4 for value in tangents):
            raise ValueError('Native tangent basis does not match its source vertices.')
        bases = {'sunrise_native_normal': geometry['Normals'],
                 'sunrise_native_tangent': [value[:3] for value in tangents],
                 'sunrise_native_bitangent': [list(Vector(normal).cross(Vector(tangent[:3])) * tangent[3])
                                             for normal, tangent in zip(geometry['Normals'], tangents)]}
        for name, values in bases.items():
            attribute = mesh.attributes.new(name, 'FLOAT_VECTOR', 'POINT')
            attribute.data.foreach_set('vector', [component for value in values for component in value])
        if basis_preview:
            _native_basis_modifier(obj)
    identity = mesh.attributes.new("sunrise_vertex_id", "INT", "POINT")
    identity.data.foreach_set("value", list(range(len(mesh.vertices))))
    obj.matrix_world = _matrix(info["Transform"])
    obj["sunrise_mesh_id"] = info["Id"]
    obj["sunrise_normal_state"] = _normal_state(mesh)
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            modifier.use_deform_preserve_volume = info.get('Skinning') == 'DualQuaternion'
    if geometry.get('VertexMotion'):
        _native_vertex_motion(obj, geometry['VertexMotion'])
    obj["sunrise_modifiers"] = json.dumps([(m.type, _uid(m.object) if m.type == 'ARMATURE' and m.object else None)
                                         for m in obj.modifiers])
    obj.vertex_groups.clear()
    if geometry.get("Joints") is not None:
        used = {joint for js, ws in zip(geometry["Joints"], geometry["Weights"])
                for joint, weight in zip(js, ws) if weight != 0}
        if not used.issubset(bones):
            raise ValueError("Skin refers to a missing native bone index.")
        groups = {i: obj.vertex_groups.new(name=bones[i]) for i in sorted(used)}
        for vertex, (joints, weights) in enumerate(zip(geometry["Joints"], geometry["Weights"])):
            combined = {}
            for joint, weight in zip(joints, weights):
                if weight:
                    combined[joint] = combined.get(joint, 0) + weight
            for joint, weight in combined.items():
                groups[joint].add([vertex], weight, 'REPLACE')
    obj["sunrise_bones"] = json.dumps(bones)
    return obj


def _load_image(root, relative, tag, mip=0, slice_index=0, depth=0, expected=None):
    path = _path(root, relative)
    if not path.exists():
        return None
    image = bpy.data.images.load(str(path), check_existing=True)
    image.use_fake_user = True
    # Native alpha is a data channel; premultiplying would erase RGB where alpha is 0.
    image.alpha_mode = 'CHANNEL_PACKED'
    image["sunrise_texture_tag"] = f"{tag:08X}"
    image["sunrise_mip"] = mip
    image["sunrise_slice"] = slice_index
    image["sunrise_depth"] = depth
    image["sunrise_source_path"] = str(path)
    image["sunrise_source_sha256"] = expected or _sha(path)
    return image


def _shader_preview(text, textures, constants=None, inverse_alpha=False):
    """Legacy bundles derive texture roles from shader lanes rather than fixed slot numbers."""
    empty = (frozenset(), None, None)
    registers, branches, normals, normal_pairs, smoothness = {}, [], [], [], []
    deferred = bool(re.search(r'\bdcl_output o2\.', text))

    def combine(values):
        return (frozenset(source for value in values for source in value[0]), None, None)

    def read(operand, lane):
        operand = operand.strip()
        negative = operand.startswith('-')
        if negative:
            operand = operand[1:]
        absolute = operand.startswith('|') and operand.endswith('|')
        if absolute:
            operand = operand[1:-1]
        if operand.startswith('l('):
            numbers = operand[2:-1].split(',')
            raw = numbers[0 if len(numbers) == 1 else min(lane, len(numbers) - 1)].strip()
            number = struct.unpack('<f', struct.pack('<I', int(raw, 16)))[0] if raw.startswith('0x') else float(raw)
            value = (frozenset(), None, number if math.isfinite(number) else None)
        else:
            match = re.fullmatch(r'([rvo]\d+|cb\d+\[\d+\])(?:\.([xyzw]+))?', operand)
            if not match:
                return empty
            register, swizzle = match.groups()
            component = lane if not swizzle else 'xyzw'.index(swizzle[0 if len(swizzle) == 1 else min(lane, len(swizzle) - 1)])
            if register.startswith('v'):
                value = (frozenset(), register + '.' + 'xyzw'[component], None)
            elif register.startswith('cb0[') and constants is not None and int(register[4:-1]) < len(constants):
                value = (frozenset(), None, constants[int(register[4:-1])][component])
            else:
                value = registers.get(register, [empty] * 4)[component]
        if negative or absolute:
            number = value[2]
            value = (value[0], None, None if number is None else (-1 if negative else 1) * (abs(number) if absolute else number))
        return value

    def evaluate(op, values):
        if any(value[2] is None for value in values):
            return None
        a = values[0][2]; b = values[1][2] if len(values) > 1 else 0
        try:
            result = {'mov': lambda: a, 'add': lambda: a + b, 'mul': lambda: a * b,
                      'mad': lambda: a * b + values[2][2], 'div': lambda: a / b,
                      'min': lambda: min(a, b), 'max': lambda: max(a, b),
                      'sqrt': lambda: math.sqrt(a), 'rsq': lambda: 1 / math.sqrt(a),
                      'exp': lambda: 2 ** a, 'log': lambda: math.log2(a)}.get(op, lambda: None)()
            return result if result is not None and math.isfinite(result) else None
        except (ValueError, OverflowError, ZeroDivisionError):
            return None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(('//', 'dcl_')):
            continue
        parts = line.split(None, 1); opcode = parts[0]
        operands = re.split(r',\s*(?![^()]*\))', parts[1]) if len(parts) > 1 else []
        if opcode.startswith('if'):
            branches.append((dict(registers), None)); continue
        if opcode == 'else' and branches:
            start, _ = branches[-1]; branches[-1] = (start, dict(registers)); registers = dict(start); continue
        if opcode == 'endif' and branches:
            start, first = branches.pop(); other = first if first is not None else start
            registers = {key: [combine([registers.get(key, [empty] * 4)[i], other.get(key, [empty] * 4)[i]])
                               for i in range(4)] for key in registers.keys() | other.keys()}
            continue
        if len(operands) < 2:
            continue
        match = re.fullmatch(r'([ro]\d+)(?:\.([xyzw]+))?', operands[0])
        if not match:
            continue
        name, mask = match.groups(); mask = mask or 'xyzw'
        output = list(registers.get(name, [empty] * 4))
        if opcode.startswith(('sample', 'ld_')):
            match = next((re.fullmatch(r't(\d+)(?:\.([xyzw]+))?', item) for item in operands[2:]
                          if re.fullmatch(r't(\d+)(?:\.([xyzw]+))?', item)), None)
            if match is None:
                continue
            slot = int(match[1]); swizzle = match[2] or 'xyzw'
            u, v = read(operands[1], 0)[1], read(operands[1], 1)[1]
            direct = bool(u and u.endswith('.x') and v == u[:-1] + 'y')
            for channel in mask:
                lane = 'xyzw'.index(channel)
                output[lane] = (frozenset([(slot, swizzle[0 if len(swizzle) == 1 else lane], direct)]), None, None)
        else:
            op = opcode.split('_')[0]; reduced = None
            if op in ('dp2', 'dp3', 'dp4'):
                reduced = combine([read(item, i) for i in range(int(op[-1])) for item in operands[1:]])
                if op == 'dp2':
                    normals.append(reduced)
                    normal_pairs.append([combine([read(item, lane) for item in operands[1:]])[0] for lane in range(2)])
            if op == 'mad' and len(operands) == 4:
                literals = [read(item, 'xyzw'.index(mask[0]))[2] for item in operands[2:]]
                if literals == [.125, .375]:
                    smoothness.append(read(operands[1], 'xyzw'.index(mask[0])))
            for channel in mask:
                lane = 'xyzw'.index(channel); values = [read(item, lane) for item in operands[1:]]
                value = reduced or (values[0] if op == 'mov' else combine(values))
                number = evaluate(op, values) if reduced is None else None
                if number is not None:
                    value = (value[0], value[1], min(1, max(0, number)) if opcode.endswith('_sat') else number)
                output[lane] = value
        registers[name] = output
    color, third = registers.get('o0', [empty] * 4), registers.get('o2', [empty] * 4)
    bindings = []

    def color_score(slot):
        if not all(any(source[0] == slot and source[1] == 'xyz'[i] for source in color[i][0]) for i in range(3)):
            return -1
        channels = [{source[1] for source in value[0] if source[0] == slot} for value in color[:3]]
        if not textures[slot]['Srgb'] and all(value >= set('xyz') for value in channels):
            return -1
        cross = sum(source[0] == slot and source[1] != 'xyz'[i] for i in range(3) for source in color[i][0])
        direct = any(source[0] == slot and source[2] for value in color[:3] for source in value[0])
        return 30 - cross + 30 * textures[slot]['Srgb'] + 10 * direct

    base = next(iter(sorted((slot for slot in textures if color_score(slot) >= 0), key=lambda slot: (-color_score(slot), slot))), None)

    def binding(role, slot, channels, **values):
        bindings.append(dict(Role=role, Slot=slot, TextureTag=textures[slot]['Tag'], Channels=channels,
                             Scale=1, Bias=0, Srgb=False, ReconstructNormalZ=False) | values)

    if base is not None:
        binding('BaseColor', base, 'rgb', Srgb=textures[base]['Srgb'])
    sources = combine(normals)[0]
    normal = next(iter(sorted((slot for slot in textures if any(
                               any(s[0] == slot and s[1] == 'x' for s in pair[0])
                               and any(s[0] == slot and s[1] == 'y' for s in pair[1])
                               and not any(s[0] == slot and s[1] == 'y' for s in pair[0])
                               and not any(s[0] == slot and s[1] == 'x' for s in pair[1]) for pair in normal_pairs)),
                             key=lambda slot: (-any(s[0] == slot and s[2] for s in sources), textures[slot]['Srgb'], slot))), None)
    if deferred and normal is not None and not inverse_alpha:
        binding('Normal', normal, 'rg', ReconstructNormalZ=True)

    def scalar(value, preferred=None, channel='x', exclude=True):
        return next(iter(sorted((s for s in value[0] if s[0] in textures and (not exclude or s[0] not in (normal, base))),
                                key=lambda s: (s[0] != preferred, not s[2], s[1] != channel, s[0], s[1]))), None)

    if deferred and not inverse_alpha:
        metal = scalar(third[0])
        if metal:
            binding('Metallic', metal[0], 'rgba'['xyzw'.index(metal[1])])
        smooth = scalar(combine(smoothness), metal[0] if metal else None, 'y')
        if smooth:
            binding('Roughness', smooth[0], 'rgba'['xyzw'.index(smooth[1])], Scale=-1, Bias=1)
    else:
        alpha = scalar(color[3], base, 'w', exclude=False)
        if alpha and alpha[1] == 'w':
            binding('Alpha', alpha[0], 'a', Scale=-1 if inverse_alpha else 1, Bias=1 if inverse_alpha else 0)
    rgb = [value[2] for value in color[:3]]
    resolved = all(value is not None and 0 <= value <= 1 for value in rgb)
    return {'Mode': 'Shader dataflow approximation', 'Bindings': bindings,
            'BaseColor': [*rgb, 1] if base is None and resolved else [0, 0, 0, 1] if inverse_alpha and base is None else [1, 1, 1, 1],
            'Metallic': min(1, max(0, third[0][2] or 0)), 'Roughness': .5,
            'Alpha': 1 - (color[3][2] if color[3][2] is not None else 1) if inverse_alpha else 1,
            'Emission': [0, 0, 0], 'BlendMode': 'InverseAlpha' if inverse_alpha else 'Opaque',
            'Notes': ['Legacy bundle shader analysis; runtime dye and detail terms are approximated.'] +
                     ([] if base is not None or resolved else ['Shader palette is unavailable; export this asset again.'])}


def _legacy_material_preview(root, info, material, bundle):
    """Older exports use their stored shader and DDS metadata without editing the manifest."""
    unavailable = {'Mode': 'Re-export for material preview', 'Bindings': [], 'BaseColor': [.5, .5, .5, 1],
                   'Metallic': 0, 'Roughness': .5, 'Alpha': 1, 'Emission': [0, 0, 0], 'Notes': ['Shader metadata is missing.']}
    native = next((value for value in bundle.get('NativeAssets', []) if value['TagHash'] == info['TagHash']), None)
    if native is None:
        return unavailable
    shader_tag = next((value['TargetTag'] for value in native['References'] if value['Role'] == 'Pixel shader'), None)
    shader = next((value for value in bundle.get('Shaders', []) if value['HeaderTag'] == shader_tag), None)
    if shader is None:
        return unavailable
    textures = {}
    for slot in material['Slots']:
        if slot['Stage'] != 'Pixel':
            continue
        texture = next((value for value in bundle.get('Textures', []) if value['TagHash'] == slot['TextureTag']), None)
        srgb = False
        if texture and texture.get('DdsPath'):
            with _path(root, texture['DdsPath']).open('rb') as stream:
                header = stream.read(148)
            if len(header) >= 148 and header[84:88] == b'DX10':
                if struct.unpack_from('<I', header, 132)[0] != 3 or struct.unpack_from('<I', header, 136)[0] & 4 \
                        or struct.unpack_from('<I', header, 140)[0] != 1:
                    continue
                srgb = struct.unpack_from('<I', header, 128)[0] in SRGB_FORMATS
            elif len(header) >= 116 and struct.unpack_from('<I', header, 112)[0] & (0x200 | 0x200000):
                continue
        textures[slot['Slot']] = {'Tag': slot['TextureTag'], 'Srgb': srgb}
    constants = None
    fields = [field for field in material['Constants'] if field['Stage'] == 'Pixel' and field['Bank'] == 'Runtime']
    if fields:
        constants = [[0.] * 4 for _ in range(max(field['Vector'] for field in fields) + 1)]
        for field in fields:
            constants[field['Vector']]['xyzw'.index(field['Component'].lower())] = float(field['Value'])
    native_path = _path(root, native['Path'])
    with native_path.open('rb') as stream:
        header = stream.read(36)
    inverse = len(header) >= 36 and (header[32] & 127) in (22, 27)
    result = _shader_preview(_path(root, shader['DisassemblyPath']).read_text(), textures, constants, inverse)
    result['PixelShaderTag'] = shader_tag
    result['ShaderSha256'] = shader['Sha256']
    return result


def _preview_image(image, srgb):
    """Color and data uses of one native texture keep separate image color spaces."""
    if image is None:
        return None
    space = 'sRGB' if srgb else 'Non-Color'
    current = image.get('sunrise_preview_space')
    if current and current != space:
        source = image.get('sunrise_source_path')
        match = next((item for item in bpy.data.images if item.get('sunrise_source_path') == source
                      and item.get('sunrise_preview_space') == space), None)
        if match is not None:
            return match
        image = image.copy()
        image.name += ' ' + space
        image.use_fake_user = True
    # Setting the color space reloads the image, even to the same value.
    if image.colorspace_settings.name != space:
        image.colorspace_settings.name = space
    image['sunrise_preview_space'] = space
    return image


def _add_preview_role(texture, role):
    """Adds role to a native slot node's preview roles; returns the roles."""
    roles = json.loads(texture.get('sunrise_preview_roles', '[]'))
    if role not in roles:
        roles.append(role)
    texture['sunrise_preview_roles'] = json.dumps(roles)
    return roles


# Program operations that one Blender math node computes.
PROGRAM_OPERATIONS = {'add': 'ADD', 'mul': 'MULTIPLY', 'div': 'DIVIDE', 'mad': 'MULTIPLY_ADD',
                      'min': 'MINIMUM', 'max': 'MAXIMUM', 'abs': 'ABSOLUTE', 'sqrt': 'SQRT',
                      'round_ni': 'FLOOR', 'round_ne': 'ROUND', 'round_pi': 'CEIL', 'round_z': 'TRUNC',
                      'sin': 'SINE', 'cos': 'COSINE', 'lt': 'LESS_THAN'}
# Every program operation _program_math builds.
PROGRAM_MATH = set(PROGRAM_OPERATIONS) | {'neg', 'rsq', 'exp', 'log', 'frc', 'eq', 'ne', 'ge', 'movc'}


def _program_math(program, item, inputs, math_node):
    """One program math operation as math nodes, in a shader or a geometry tree; None for any other operation."""
    operation = item['Operation']
    if operation in PROGRAM_OPERATIONS:
        return math_node(PROGRAM_OPERATIONS[operation], *inputs)
    if operation == 'neg':
        return math_node('MULTIPLY', inputs[0], -1)
    if operation == 'rsq':
        return math_node('INVERSE_SQRT', inputs[0])
    if operation == 'exp':
        return math_node('POWER', 2, inputs[0])
    if operation == 'log':
        # Blender's log of zero is 0 but Direct3D's is -inf, so a pow of zero built from log must stay 0.
        return math_node('MULTIPLY_ADD', math_node('LESS_THAN', inputs[0], 1.1754944e-38), -1e30,
                         math_node('LOGARITHM', inputs[0], 2))
    if operation == 'frc':
        return math_node('FRACT', inputs[0])
    if operation in ('eq', 'ne'):
        same = math_node('COMPARE', *inputs, 0)
        return math_node('SUBTRACT', 1, same) if operation == 'ne' else same
    if operation == 'ge':
        return math_node('SUBTRACT', 1, math_node('LESS_THAN', *inputs))
    if operation == 'movc':
        # A comparison already reads 0 or 1.
        condition = inputs[0] if program['Nodes'][item['Inputs'][0]]['Operation'] in ('lt', 'ge', 'eq', 'ne') \
            else math_node('GREATER_THAN', math_node('ABSOLUTE', inputs[0]), 0)
        return math_node('MULTIPLY_ADD', condition, math_node('SUBTRACT', inputs[1], inputs[2]), inputs[2])
    return None


def _volume_image(root, volume):
    """The written slices of a 3D texture side by side; None when the export did not write them."""
    if not root or not volume.get('Path'):
        return None
    path = _path(Path(root), volume['Path'])
    if not path.exists() or _sha(path) != volume.get('Sha256'):
        return None
    image = bpy.data.images.load(str(path), check_existing=True)
    image.use_fake_user = True
    image.alpha_mode = 'CHANNEL_PACKED'
    image['sunrise_texture_tag'] = f'{int(volume["TextureTag"]):08X}'
    image['sunrise_source_path'] = str(path)
    image['sunrise_volume'] = True
    return image


def _apply_material_program(profile, slots, node, connect, math_node, component, uv, signals, root=None):
    """Known shader expressions keep texture channels, constants, UVs and cutouts together."""
    program = profile.get('Program')
    if not program:
        return
    # A dye recipe or palette owns the surface; the program then adds only its cutout. Any other program is the surface.
    owned = str(profile.get('Recipe') or '').startswith('VexDye') or profile.get('Palette')
    roles = [role for role in program['Outputs'] if not owned or role == 'Visibility']
    if not roles:
        return
    values, samples, basis_values = {}, {}, {}
    basis_available = basis_reflected = None
    separate = node('ShaderNodeSeparateXYZ', 'Shader UV components')
    connect(uv, separate.inputs[0])
    shared = {}

    def scene_time():
        """Seconds at the current frame, from the Scene Time node that shaders have since Blender 5.2."""
        if 'time' not in shared:
            shared['time'] = node('GeometryNodeInputSceneTime', 'Shader time in seconds').outputs['Seconds']
        return shared['time']

    def detail_uv():
        """UV map 1, the detail UVs; 'uv' channels 2 and 3 read it."""
        if 'detail' not in shared:
            source = node('ShaderNodeUVMap', 'Shader UV map 1')
            source.uv_map = 'UVMap1'
            split = node('ShaderNodeSeparateXYZ', 'Shader UV map 1 components')
            connect(source.outputs['UV'], split.inputs[0])
            shared['detail'] = split.outputs
        return shared['detail']

    def vertex_stream(name):
        """COLOR0 ('color'), the raw UV z and w ('uvzw') or raw POSITION0 ('rawposition'); a mesh without it reads zero."""
        if name not in shared:
            attribute = node('ShaderNodeAttribute', 'Shader vertex ' + name)
            attribute.attribute_name = 'sunrise_' + name
            split = node('ShaderNodeSeparateXYZ', 'Shader vertex ' + name + ' components')
            connect(attribute.outputs['Color' if name == 'color' else 'Vector'], split.inputs[0])
            shared[name] = [*split.outputs, attribute.outputs['Alpha']]
        return shared[name]

    def instance_inputs(channel):
        """Decorator placement inputs: channels 0-3 the unpacked record, 4-7 the AO; absent ones read zero."""
        name = 'record' if channel < 4 else 'occlusion'
        if name not in shared:
            # The instancer is the placement; a part that is not instanced reads its own copy.
            attribute = node('ShaderNodeAttribute', 'Shader instance ' + name)
            attribute.attribute_type = 'INSTANCER'
            attribute.attribute_name = 'sunrise_instance_' + name
            split = node('ShaderNodeSeparateXYZ', 'Shader instance ' + name + ' components')
            connect(attribute.outputs['Vector'], split.inputs[0])
            shared[name] = [*split.outputs, attribute.outputs['Alpha']]
        return shared[name][channel % 4]

    def screen():
        """Window coordinates from 0 to 1; Direct3D screen y runs down, Blender's runs up."""
        if 'screen' not in shared:
            coordinates = node('ShaderNodeTexCoord', 'Shader screen position')
            split = node('ShaderNodeSeparateXYZ', 'Shader screen components')
            connect(coordinates.outputs['Window'], split.inputs[0])
            shared['screen'] = [split.outputs[0], math_node('SUBTRACT', 1, split.outputs[1])]
        return shared['screen']

    def world_vectors():
        """World position, camera position and the surface basis, split into components."""
        if 'vectors' not in shared:
            geometry = node('ShaderNodeNewGeometry', 'Shader world geometry')
            camera = node('ShaderNodeCameraData', 'Shader camera data')
            tangent = node('ShaderNodeTangent', 'Shader UV tangent')
            tangent.direction_type = 'UV_MAP'
            tangent.uv_map = 'UVMap'

            def vector(operation, *values):
                value = node('ShaderNodeVectorMath', 'Shader ' + operation.lower())
                value.operation = operation
                for index, source in enumerate(values):
                    connect(source, value.inputs['Scale'] if operation == 'SCALE' and index == 1 else value.inputs[index])
                return value.outputs['Vector']

            # Incoming points at the viewer, so the camera sits that far along it.
            eye = vector('ADD', geometry.outputs['Position'],
                         vector('SCALE', geometry.outputs['Incoming'], camera.outputs['View Distance']))
            # A mesh without the sign attribute reads 0 and keeps the unsigned cross product.
            sign = node('ShaderNodeAttribute', 'Shader bitangent sign')
            sign.attribute_name = 'sunrise_bitangent_sign'
            sign = math_node('SUBTRACT', math_node('ADD', sign.outputs['Fac'], 1), math_node('ABSOLUTE', sign.outputs['Fac']))
            bitangent = vector('SCALE', vector('CROSS_PRODUCT', geometry.outputs['Normal'], tangent.outputs['Tangent']), sign)
            # The view +Z axis in world space, pointing back at the camera.
            view = node('ShaderNodeVectorTransform', 'Shader view axis')
            view.vector_type, view.convert_from, view.convert_to = 'VECTOR', 'CAMERA', 'WORLD'
            view.inputs['Vector'].default_value = (0, 0, 1)
            shared['vectors'] = {}
            for name, socket in (('position', geometry.outputs['Position']), ('camera', eye),
                                 ('normal', geometry.outputs['Normal']), ('tangent', tangent.outputs['Tangent']),
                                 ('bitangent', bitangent), ('view', view.outputs['Vector'])):
                split = node('ShaderNodeSeparateXYZ', 'Shader world ' + name)
                connect(socket, split.inputs[0])
                shared['vectors'][name] = split.outputs
        return shared['vectors']

    def object_normal():
        """The NORMAL0 stream: the mesh normal in object space, split into components."""
        if 'normal' not in shared:
            coordinates = node('ShaderNodeTexCoord', 'Shader object normal')
            split = node('ShaderNodeSeparateXYZ', 'Shader object normal components')
            connect(coordinates.outputs['Normal'], split.inputs[0])
            shared['normal'] = split.outputs
        return shared['normal']

    def world_basis():
        """Point-domain shader attributes already contain the native vertex-stage basis."""
        nonlocal basis_available, basis_reflected
        if basis_values:
            return
        available = node('ShaderNodeAttribute', 'Shader basis available')
        available.attribute_name = 'sunrise_shader_basis'
        basis_available = math_node('GREATER_THAN', available.outputs['Fac'], .999999)
        # A missing basis reads zero, and a normal normalized from zero is NaN, which survives a mix factor of 0.
        # Blender's own basis stands in, so the branch the mix drops stays finite.
        missing = math_node('SUBTRACT', 1, basis_available)
        geometry = node('ShaderNodeNewGeometry', 'Basis stand-in normal')
        tangent = node('ShaderNodeTangent', 'Basis stand-in tangent')
        tangent.direction_type = 'UV_MAP'
        tangent.uv_map = 'UVMap'
        bitangent = node('ShaderNodeVectorMath', 'Basis stand-in bitangent')
        bitangent.operation = 'CROSS_PRODUCT'
        connect(geometry.outputs['Normal'], bitangent.inputs[0])
        connect(tangent.outputs['Tangent'], bitangent.inputs[1])
        stand_ins = (geometry.outputs['Normal'], tangent.outputs['Tangent'], bitangent.outputs['Vector'])
        for index, (name, stand_in) in enumerate(zip(('normal', 'tangent', 'bitangent'), stand_ins)):
            attribute = node('ShaderNodeAttribute', 'Shader world ' + name)
            attribute.attribute_name = 'sunrise_shader_' + name
            guarded = node('ShaderNodeVectorMath', 'Shader world ' + name + ' or stand-in')
            guarded.operation = 'MULTIPLY_ADD'
            connect(stand_in, guarded.inputs[0])
            connect(missing, guarded.inputs[1])
            connect(attribute.outputs['Vector'], guarded.inputs[2])
            separate_basis = node('ShaderNodeSeparateXYZ', 'Shader world basis ' + str(index))
            connect(guarded.outputs['Vector'], separate_basis.inputs[0])
            basis_values[index] = separate_basis.outputs
        reflected = node('ShaderNodeAttribute', 'Shader instance reflected')
        reflected.attribute_name = 'sunrise_shader_reflected'
        basis_reflected = math_node('GREATER_THAN', reflected.outputs['Fac'], .5)

    def volume(item, coordinates):
        """One channel of a trilinear 3D texture read; point reads at texel centres keep the atlas tiles apart."""
        slot, channel = int(item['Slot']), item['Channel']
        key = ('volume', slot, item.get('SamplerSlot'), *item['Inputs'])
        info = (profile.get('VolumeTextures') or {}).get(str(slot), {})
        if key not in samples:
            image = _volume_image(root, info)
            if image is None:
                samples[key] = None
            else:
                formats = profile.get('TextureFormats') or {}
                fmt = formats.get(str(slot), formats.get(slot))
                image = _preview_image(image, fmt is not None and int(fmt) in SRGB_FORMATS)
                samplers = profile.get('Samplers') or {}
                sampler = samplers.get(str(item.get('SamplerSlot')), samplers.get(item.get('SamplerSlot'))) or {}
                sizes = (int(info['Width']), int(info['Height']), int(info['Depth']))
                columns = int(info['Columns'])

                def texel(value, size, mode):
                    position = math_node('SUBTRACT', math_node('MULTIPLY', value, size), .5)
                    first = math_node('FLOOR', position)
                    ends = [first, math_node('ADD', first, 1)]
                    if mode == 'Clamp':
                        ends = [math_node('MINIMUM', math_node('MAXIMUM', end, 0), size - 1) for end in ends]
                    else:
                        ends = [math_node('FLOORED_MODULO', end, size) for end in ends]
                    return ends, math_node('SUBTRACT', position, first)

                axes = [texel(value, size, sampler.get('Address' + name)) for value, size, name in zip(coordinates, sizes, 'UVW')]
                corners = {}
                for dz, z in enumerate(axes[2][0]):
                    left = math_node('MULTIPLY', math_node('FLOORED_MODULO', z, columns), sizes[0])
                    top = math_node('MULTIPLY', math_node('FLOOR', math_node('DIVIDE', math_node('ADD', z, .5), columns)), sizes[1])
                    for dy, y in enumerate(axes[1][0]):
                        # Atlas row 0 is the top of the image; Blender's v runs up.
                        v = math_node('SUBTRACT', 1, math_node('DIVIDE', math_node('ADD', math_node('ADD', top, y), .5), image.size[1]))
                        for dx, x in enumerate(axes[0][0]):
                            u = math_node('DIVIDE', math_node('ADD', math_node('ADD', left, x), .5), image.size[0])
                            texture = node('ShaderNodeTexImage', 'Shader volume t' + str(slot))
                            texture.image, texture.interpolation, texture.extension = image, 'Closest', 'EXTEND'
                            point = node('ShaderNodeCombineXYZ', 'Shader volume texel')
                            connect(u, point.inputs['X'])
                            connect(v, point.inputs['Y'])
                            connect(point.outputs[0], texture.inputs['Vector'])
                            corners[dx, dy, dz] = texture
                samples[key] = (corners, [weight for _, weight in axes], {})
        if samples[key] is None:
            mean = info.get('Mean')
            if mean is None:
                raise ValueError('Shader expression needs volume texture slot ' + str(slot))
            return float(mean[channel])
        corners, weights, channels = samples[key]
        if channel not in channels:
            def lerp(a, b, weight):
                return math_node('MULTIPLY_ADD', math_node('SUBTRACT', b, a), weight, a)
            lanes = {corner: component(texture, 'rgba'[channel]) for corner, texture in corners.items()}
            rows = {(dy, dz): lerp(lanes[0, dy, dz], lanes[1, dy, dz], weights[0]) for dy in (0, 1) for dz in (0, 1)}
            planes = [lerp(rows[0, dz], rows[1, dz], weights[1]) for dz in (0, 1)]
            channels[channel] = lerp(planes[0], planes[1], weights[2])
        return channels[channel]

    def expression(index):
        if index in values:
            return values[index]
        item = program['Nodes'][index]
        operation = item['Operation']
        negated = [i for i in item['Inputs'] if program['Nodes'][i]['Operation'] == 'neg']
        if operation == 'add' and negated:
            # a + -b is one subtraction; the negation is built only where another node reads it.
            kept = item['Inputs'][0] if negated[-1] == item['Inputs'][1] else item['Inputs'][1]
            values[index] = math_node('SUBTRACT', expression(kept), expression(program['Nodes'][negated[-1]]['Inputs'][0]))
            return values[index]
        inputs = [expression(i) for i in item['Inputs']]
        if operation == 'constant':
            result = item['Value']
        elif operation == 'uv':
            result = separate.outputs[item['Channel']] if item['Channel'] < 2 else detail_uv()[item['Channel'] - 2]
        elif operation == 'screen':
            result = screen()[item['Channel']]
        elif operation in ('color', 'uvzw', 'rawposition'):
            result = vertex_stream(operation)[item['Channel']]
        elif operation == 'instance':
            result = instance_inputs(item['Channel'])
        elif operation == 'time':
            result = scene_time()
        elif operation == 'normal':
            result = object_normal()[item['Channel']]
        elif operation in ('position', 'camera', 'view'):
            result = world_vectors()[operation][item['Channel']]
        elif operation == 'surface':
            result = world_vectors()[('normal', 'tangent', 'bitangent')[int(item['Slot'])]][item['Channel']]
        elif operation == 'basis':
            world_basis()
            result = basis_values[int(item['Slot'])][item['Channel']]
        elif operation == 'frontface':
            world_basis()
            geometry = node('ShaderNodeNewGeometry', 'Shader facing')
            # Blender corrects Backfacing for reflected objects; native raster facing keeps the winding.
            result = math_node('ABSOLUTE', math_node('SUBTRACT',
                               math_node('SUBTRACT', 1, geometry.outputs['Backfacing']), basis_reflected))
        elif operation == 'sample':
            slot = int(item['Slot'])
            sampler_slot = item.get('SamplerSlot')
            key = (slot, sampler_slot, *item['Inputs'])
            if key not in samples:
                original = slots.get(slot)
                if original is None or original.image is None:
                    raise ValueError('Shader expression needs native texture slot ' + str(slot))
                texture = node('ShaderNodeTexImage', 'Shader sample t' + str(slot))
                formats = profile.get('TextureFormats') or {}
                fmt = formats.get(str(slot), formats.get(slot))
                srgb = int(fmt) in SRGB_FORMATS if fmt is not None else any(
                    b.get('Srgb') for b in profile.get('Bindings', []) if int(b['Slot']) == slot)
                texture.image = _preview_image(original.image, srgb)
                samplers = profile.get('Samplers') or {}
                sampler = samplers.get(str(sampler_slot), samplers.get(sampler_slot)) or {}
                if sampler.get('AddressU') == sampler.get('AddressV') == 'Clamp':
                    texture.extension = 'EXTEND'
                elif sampler.get('AddressU') == sampler.get('AddressV') == 'Wrap':
                    texture.extension = 'REPEAT'
                if sampler.get('Filter') == 'MinMagMipPoint':
                    texture.interpolation = 'Closest'
                    # Snap before V reversal so both APIs select the same boundary texel.
                    inputs = [math_node('DIVIDE', math_node('ADD',
                        math_node('FLOOR', math_node('MULTIPLY', value, size)), .5), size)
                        for value, size in zip(inputs, texture.image.size)]
                coordinates = node('ShaderNodeCombineXYZ', 'Shader sample UV')
                connect(inputs[0], coordinates.inputs['X'])
                connect(math_node('SUBTRACT', 1, inputs[1]), coordinates.inputs['Y'])
                connect(coordinates.outputs[0], texture.inputs['Vector'])
                samples[key] = texture
            result = component(samples[key], 'rgba'[item['Channel']])
        elif operation == 'volume':
            result = volume(item, inputs)
        else:
            result = _program_math(program, item, inputs, math_node)
            if result is None:
                raise ValueError('Unsupported shader expression: ' + operation)
        values[index] = result
        return result

    available = {}
    known = PROGRAM_MATH | {'constant', 'uv', 'screen', 'time', 'position', 'camera', 'view', 'surface', 'basis', 'frontface',
                            'sample', 'color', 'uvzw', 'instance', 'normal', 'rawposition', 'volume'}
    unknown = set()

    def can_read(index):
        if index not in available:
            item = program['Nodes'][index]
            texture = slots.get(int(item['Slot'])) if item['Operation'] == 'sample' else None
            if item['Operation'] not in known:
                unknown.add(item['Operation'])
            volume_info = (profile.get('VolumeTextures') or {}).get(str(item['Slot']), {}) if item['Operation'] == 'volume' else None
            available[index] = item['Operation'] in known and all(can_read(i) for i in item['Inputs']) and (
                item['Operation'] != 'sample' or texture is not None and texture.image is not None) and (
                volume_info is None or volume_info.get('Mean') is not None or _volume_image(root, volume_info) is not None)
        return available[index]

    def reads(indices, operations):
        pending, seen = list(indices), set()
        while pending:
            index = pending.pop()
            if index not in seen:
                seen.add(index)
                if program['Nodes'][index]['Operation'] in operations:
                    return True
                pending.extend(program['Nodes'][index]['Inputs'])
        return False

    for role in roles:
        indices = program['Outputs'][role]
        if not all(can_read(index) for index in indices):
            # A newer export's operation drops only this output, not the whole import.
            if unknown:
                print(f'Sunrise material preview: {role} needs a newer add-on for {", ".join(sorted(unknown))}.')
            continue
        if role in ('BaseColor', 'Emission'):
            color = node('ShaderNodeCombineColor', 'Authored base color' if role == 'BaseColor' else 'Authored emission')
            for name, index in zip(('Red', 'Green', 'Blue'), indices):
                connect(expression(index), color.inputs[name])
            signals[role] = color.outputs['Color']
        elif role == 'Normal':
            normal = node('ShaderNodeCombineXYZ', 'Authored world normal')
            for lane, index in enumerate(indices):
                connect(expression(index), normal.inputs[lane])
            if not reads(indices, ('basis', 'frontface')):
                # Built on NORMAL0 alone, it needs no native basis attributes.
                signals[role] = normal.outputs['Vector']
                continue
            world_basis()
            fallback = signals.get('Normal')
            if fallback is None:
                fallback = node('ShaderNodeNewGeometry', 'Fallback surface normal').outputs['Normal']
            choose = node('ShaderNodeMixRGB', 'Use shader normal when basis is available')
            connect(basis_available, choose.inputs[0])
            connect(fallback, choose.inputs[1])
            connect(normal.outputs['Vector'], choose.inputs[2])
            signals[role] = choose.outputs['Color']
        else:
            signals[role] = expression(indices[0])


def _apply_material_recipe(profile, slots, node, connect, math_node, component, uv, signals, native_normal):
    """Pinned dye recipes retain authored constants and UV transforms; other recipe names are left to the program."""
    recipe = profile.get('Recipe')
    if not recipe:
        return
    constants = profile.get('PixelConstants') or []
    masks = profile.get('PixelConstantMasks')

    def scalar(row, lane=0):
        if row >= len(constants) or (masks is not None and not masks[row] & (1 << lane)):
            raise ValueError('Shader recipe needs an unavailable constant lane.')
        return float(constants[row][lane])

    def vector(row):
        return tuple(scalar(row, lane) for lane in range(3))

    def vec(operation, *values):
        value = node('ShaderNodeVectorMath', 'Authored ' + operation.lower())
        value.operation = operation
        for index, source in enumerate(values):
            connect(source, value.inputs['Scale'] if operation == 'SCALE' and index == 1 else value.inputs[index])
        return value.outputs['Value'] if operation in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else value.outputs['Vector']

    def sat(value):
        return math_node('MINIMUM', math_node('MAXIMUM', value, 0), 1)

    def clamp_color(value):
        return vec('MINIMUM', vec('MAXIMUM', value, (0, 0, 0)), (1, 1, 1))

    def mix(a, b, factor):
        return vec('ADD', a, vec('SCALE', vec('SUBTRACT', b, a), factor))

    def mix_scalar(a, b, factor):
        return math_node('MULTIPLY_ADD', math_node('SUBTRACT', b, a), factor, a)

    def transform(row, source=None):
        return vec('MULTIPLY_ADD', source or uv, (scalar(row), scalar(row, 1), 1),
                   (scalar(row, 2), scalar(row, 3), 0))

    sampled = {}
    used = set()

    def sample(slot, coordinates=None, srgb=None):
        coordinates = coordinates or uv
        formats = profile.get('TextureFormats') or {}
        native_format = formats.get(str(slot), formats.get(slot))
        if native_format is not None:
            srgb = int(native_format) in SRGB_FORMATS
        elif srgb is None:
            srgb = next((bool(binding.get('Srgb')) for binding in profile.get('Bindings', []) if binding['Slot'] == slot), False)
        key = (slot, coordinates, srgb)
        if key in sampled:
            return sampled[key]
        original = slots.get(slot)
        if original is None or original.image is None:
            raise ValueError('Shader recipe needs native pixel texture slot ' + str(slot))
        texture = original if slot not in used else node('ShaderNodeTexImage', 'Repeated native t' + str(slot))
        texture.image = _preview_image(original.image, srgb)
        used.add(slot)
        _add_preview_role(original, 'ShaderRecipe')
        connect(vec('MULTIPLY_ADD', coordinates, (1, -1, 1), (0, 1, 0)), texture.inputs['Vector'])
        sampled[key] = texture
        return texture

    if recipe.startswith('VexDye'):
        family = int(recipe[6:])
        parameter = 5 if family == 47 else 0
        # Family 52 is family 46 with a patina mask at t0: every other slot moves up one, most rows by six.
        slot = 1 if family == 52 else 0
        detail_row, gain_row, palette_row = (15, 16, 17) if family == 49 else (12, 13, 14) if family == 46 else \
            (18, 19, 20) if family == 52 else (17, 18, 19)
        metal_row, vertex_row, vertex_limit = (11, 15, 16) if family == 52 else (parameter + 7, parameter + 9, parameter + 10)
        base = sample(slot)
        packed = sample(slot + 1)
        detail = sample(slot + 2, transform(detail_row))
        gain_texture = sample(slot + 3, transform(gain_row))
        color = base.outputs['Color']
        mask = component(packed, 'a')
        coordinates = node('ShaderNodeSeparateXYZ', 'Native dye region')
        connect(uv, coordinates.inputs[0])
        upper = math_node('GREATER_THAN', coordinates.outputs['Y'], 1)
        normal_row, detail_uv, detail_unpack, coat_uv, coat_mask_uv, coat_mask, coat_adjust, coat_gain, coat_unpack = \
            (20, 21, 22, 23, 24, 25, 26, 27, 28) if family == 49 else \
            (17, 18, 19, 20, 21, 22, 23, 24, 25) if family == 46 else \
            (23, 24, 25, 26, 27, 28, 29, 30, 31) if family == 52 else \
            (22, 18, 23, 24, 25, 26, 27, 28, 29)
        main_normal = sample(slot + 4)
        detail_normal = sample(slot + 5, transform(detail_uv))
        coat_normal = sample(slot + 6, transform(coat_uv))
        coat_texture = sample(slot + 7, transform(coat_mask_uv))
        enabled = math_node('MULTIPLY', mask, math_node('SUBTRACT', 1, upper))
        amount = min(1, max(0, scalar(coat_adjust) + scalar(coat_adjust, 1) * min(1, max(0, -scalar(coat_mask)))))
        coating = component(coat_texture, 'a')
        weight = math_node('DIVIDE', .5 * (amount - 1) ** 2, coating)
        weight = math_node('ADD', math_node('MULTIPLY', math_node('MULTIPLY', coating, coating), 10),
                           math_node('SUBTRACT', 3.3, weight))
        weight = math_node('MULTIPLY', weight, math_node('DIVIDE', amount,
                           math_node('MULTIPLY', 4, math_node('SUBTRACT', 1, coating))))
        weight = sat(math_node('MULTIPLY_ADD', weight, scalar(coat_gain, 1), scalar(coat_gain)))
        xy = []
        for channel in 'rg':
            primary = math_node('MULTIPLY_ADD', component(main_normal, channel), scalar(normal_row), scalar(normal_row, 1))
            detail_xy = math_node('MULTIPLY', enabled,
                                  math_node('MULTIPLY_ADD', component(detail_normal, channel), scalar(detail_unpack), scalar(detail_unpack, 1)))
            coat_xy = math_node('MULTIPLY', weight,
                                math_node('MULTIPLY_ADD', component(coat_normal, channel), scalar(coat_unpack), scalar(coat_unpack, 1)))
            xy.append(math_node('ADD', math_node('ADD', primary, detail_xy), coat_xy))
        nz = math_node('SQRT', math_node('MAXIMUM', math_node('SUBTRACT', 1,
                      math_node('ADD', math_node('MULTIPLY', xy[0], xy[0]), math_node('MULTIPLY', xy[1], xy[1]))), 0))
        encoded = node('ShaderNodeCombineColor', 'Authored layered tangent normal')
        for name, value in zip(('Red', 'Green', 'Blue'), [*xy, nz]):
            connect(math_node('MULTIPLY_ADD', value, .5, .5), encoded.inputs[name])
        mapped = node('ShaderNodeNormalMap', 'Authored layered normal')
        mapped.space = 'TANGENT'; mapped.uv_map = 'UVMap'
        connect(encoded.outputs['Color'], mapped.inputs['Color'])
        signals['Normal'] = native_normal(xy[0], xy[1], nz, mapped.outputs['Normal'])
        smooth_limit = math_node('MINIMUM', sat(math_node('ADD', component(main_normal, 'b'), scalar(normal_row, 2))),
                                sat(math_node('ADD', component(detail_normal, 'b'), scalar(detail_unpack, 2))))
        smooth_limit = math_node('MINIMUM', smooth_limit,
                                sat(math_node('ADD', component(coat_normal, 'b'), scalar(coat_unpack, 2))))
        gain = math_node('MULTIPLY_ADD', component(gain_texture, 'r'), scalar(parameter, 1), scalar(parameter))
        scaled = vec('SCALE', color, gain)
        low = clamp_color(vec('SCALE', scaled, 4))
        high = clamp_color(vec('ADD', scaled, (-.25, -.25, -.25)))
        dyed = vec('ADD', vec('MULTIPLY', vector(palette_row + 1), low), high)
        dyed = mix(color, dyed, mask)
        gray = vec('DOT_PRODUCT', dyed, (.3, .59, .11))
        dyed = mix(gray, dyed, scalar(parameter + 1))
        geometry = node('ShaderNodeNewGeometry', 'Native dye view direction')
        facing = vec('DOT_PRODUCT', signals.get('Normal', geometry.outputs['Normal']), geometry.outputs['Incoming'])
        facing = sat(math_node('MULTIPLY_ADD', math_node('MULTIPLY', facing, facing),
                               scalar(parameter + 2), scalar(parameter + 2, 1)))
        layer = sat(math_node('MULTIPLY_ADD', component(detail, 'a'), scalar(parameter + 3), facing))
        shade = vec('ADD', vector(parameter + 4), vec('SCALE', vector(parameter + 5), layer))
        shade = mix(dyed, shade, scalar(parameter + 6))
        if family == 52:
            # Rows 7 and 8 tint by base green; the authored tint row takes over where base alpha is set.
            green_tint = vec('ADD', vector(7), vec('SCALE', vector(8), sat(component(base, 'g'))))
            surface_tint = mix(green_tint, vector(palette_row), component(base, 'a'))
        else:
            surface_tint = mix(color, vector(palette_row), component(base, 'a'))
        shaded = vec('SCALE', vec('MULTIPLY', color, vec('MULTIPLY', surface_tint, shade)), 21.112095)
        shaded = mix(shaded, dyed, upper)
        shaded = mix(color, shaded, mask)
        metal_mask = sat(math_node('MULTIPLY_ADD', component(packed, 'r'), scalar(metal_row, 1), scalar(metal_row)))
        nonmetal = math_node('SUBTRACT', 1, metal_mask)
        if family == 52:
            # The patina mask joins the non-metal mask: p + q - p * q.
            patina = math_node('MULTIPLY_ADD', component(sample(0, transform(9)), 'r'), scalar(10, 1), scalar(10))
            nonmetal = math_node('SUBTRACT', math_node('ADD', patina, nonmetal), math_node('MULTIPLY', patina, nonmetal))
        nonmetal_tint = clamp_color(mix(shaded, vector(palette_row + 2), nonmetal)) if family == 52 else \
            mix(shaded, vector(palette_row + 2), nonmetal)
        shaded = mix(nonmetal_tint, shaded, upper)
        raw_vertex_factor = scalar(vertex_row)
        vertex_factor = math_node('MINIMUM', math_node('MAXIMUM', raw_vertex_factor, scalar(vertex_limit)), scalar(vertex_limit + 1))
        if family == 52:
            # The vertex blend is scaled by a detail-alpha ramp, row 13 below the dye line and row 14 above it.
            ramp = mix_scalar(math_node('MULTIPLY_ADD', component(detail, 'a'), scalar(13, 1), scalar(13)),
                              math_node('MULTIPLY_ADD', component(detail, 'a'), scalar(14, 1), scalar(14)), upper)
            vertex_factor = math_node('MULTIPLY', vertex_factor, ramp)
        shaded = clamp_color(mix(shaded, vector(12 if family == 52 else parameter + 8), vertex_factor))
        signals['BaseColor'] = mix(color, shaded, mask)

        metal_start = 41 if family == 49 else 38 if family == 46 else 44 if family == 52 else 41
        metal_low = math_node('MULTIPLY', component(detail, 'g'),
                              math_node('MULTIPLY_ADD', component(packed, 'r'), scalar(metal_start, 1), scalar(metal_start)))
        metal_low = math_node('MULTIPLY', metal_low, 4.594790)
        metal_high = math_node('MULTIPLY_ADD', component(packed, 'r'), scalar(metal_start + 1, 1), scalar(metal_start + 1))
        metal = sat(mix_scalar(component(packed, 'r'), mix_scalar(metal_low, metal_high, upper), mask))
        metal_facing = math_node('MULTIPLY_ADD', facing, scalar(metal_start + 2, 1), scalar(metal_start + 2))
        lower_metal = mix_scalar(metal, metal_facing, scalar(metal_start + 3))
        lower_metal = mix_scalar(component(packed, 'r'), lower_metal, mask)
        signals['Metallic'] = sat(mix_scalar(lower_metal, metal, upper))
        if family in (46, 49, 52):
            smooth_start = 45 if family == 49 else 48 if family == 52 else 42
            smooth_low = math_node('MULTIPLY_ADD', component(packed, 'g'), scalar(smooth_start, 1), scalar(smooth_start))
            smooth_high = math_node('MULTIPLY_ADD', component(packed, 'g'), scalar(smooth_start + 1, 1), scalar(smooth_start + 1))
            smooth = sat(mix_scalar(component(packed, 'g'), mix_scalar(smooth_low, smooth_high, upper), mask))
            low_smooth = mix_scalar(smooth, math_node('MULTIPLY_ADD', smooth, scalar(smooth_start + 2, 1), scalar(smooth_start + 2)), component(base, 'a'))
            smooth = mix_scalar(low_smooth, smooth, upper)
            smooth = sat(mix_scalar(smooth, scalar(smooth_start + 3), raw_vertex_factor))
            smooth = math_node('MINIMUM', smooth, smooth_limit)
            signals['Roughness'] = math_node('SUBTRACT', 1, smooth)
        else:
            smooth = sat(mix_scalar(component(packed, 'g'),
                math_node('MULTIPLY_ADD', component(packed, 'g'), scalar(45, 1), scalar(45)), mask))
            inside = mix_scalar(smooth, math_node('MULTIPLY_ADD', smooth, scalar(46, 1), scalar(46)), component(base, 'a'))
            smooth = sat(mix_scalar(mix_scalar(inside, smooth, upper), scalar(31), raw_vertex_factor))
            signals['Roughness'] = math_node('SUBTRACT', 1, math_node('MINIMUM', smooth, smooth_limit))


def _build_material_preview(material, profile):
    """Preview wiring never changes native slot, constant, or UV identities."""
    tree = material.node_tree
    for node in list(tree.nodes):
        if node.get('sunrise_preview_node') or node.name in ('Principled BSDF', 'Material Output'):
            if node.bl_idname == 'ShaderNodeValue':
                # Add-ons before 1.10 drove time with a driver; it outlives its node otherwise.
                node.outputs[0].driver_remove('default_value')
            tree.nodes.remove(node)
    slots = {}
    for item in tree.nodes:
        if 'sunrise_preview_slot' in item and item.get('sunrise_preview_stage') == 'Pixel':
            # A texture the part's draw binds over the material's own slot wins.
            if item.get('sunrise_preview_override') or int(item['sunrise_preview_slot']) not in slots:
                slots[int(item['sunrise_preview_slot'])] = item
    for node in slots.values():
        node['sunrise_preview_roles'] = '[]'
    serial = 0
    math_serial = 0
    # Each node edit updates the whole tree, so locations are set together before pruning.
    placed = []

    def node(kind, label, x=0, y=0):
        nonlocal serial
        value = tree.nodes.new(kind)
        value.label = label
        value['sunrise_preview_node'] = True
        placed.append((value, x, y - serial * 12))
        serial += 1
        return value

    def connect(value, socket):
        if isinstance(value, (int, float, tuple, list)):
            # socket.links scans every link in the tree, so only a linked socket reads it.
            if socket.is_linked:
                for link in list(socket.links):
                    tree.links.remove(link)
            elif _holds(socket, value):
                return
            socket.default_value = value
        else:
            tree.links.new(value, socket)

    def math_node(operation, *values):
        nonlocal math_serial
        value = node('ShaderNodeMath', operation.replace('_', ' ').title(),
                     -350 + math_serial % 4 * 190, -700 - math_serial // 4 * 180)
        # Freeing the default name keeps the next node's unique-name search short.
        value.name = f'Math {math_serial}'
        math_serial += 1
        value.operation = operation
        for index, source in enumerate(values):
            connect(source, value.inputs[index])
        return value.outputs[0]

    uv = node('ShaderNodeUVMap', 'Native UVs', -1250, 280)
    uv.uv_map = 'UVMap'
    mapping = node('ShaderNodeVectorMath', 'Texture V orientation', -1060, 280)
    mapping.operation = 'MULTIPLY_ADD'
    connect(uv.outputs['UV'], mapping.inputs[0])
    mapping.inputs[1].default_value = (1, -1, 1)
    mapping.inputs[2].default_value = (0, 1, 0)
    output = node('ShaderNodeOutputMaterial', 'Material Output', 820, 180)
    output.is_active_output = True
    surface = node('ShaderNodeBsdfPrincipled', 'Material surface', 520, 180)
    surface.inputs['Base Color'].default_value = profile.get('BaseColor', [1, 1, 1, 1])
    surface.inputs['Metallic'].default_value = profile.get('Metallic', 0)
    surface.inputs['Roughness'].default_value = profile.get('Roughness', .5)
    surface.inputs['Alpha'].default_value = profile.get('Alpha', 1)
    surface.inputs['Emission Color'].default_value = [*profile.get('Emission', [0, 0, 0]), 1]
    surface.inputs['Emission Strength'].default_value = 1
    tree.links.new(surface.outputs['BSDF'], output.inputs['Surface'])
    separated, signals = {}, {}

    def component(texture, channel):
        if channel == 'a':
            return texture.outputs['Alpha']
        if texture not in separated:
            split = node('ShaderNodeSeparateColor', texture.label + ' channels', -450, -40)
            split.mode = 'RGB'
            tree.links.new(texture.outputs['Color'], split.inputs['Color'])
            separated[texture] = split
        return separated[texture].outputs['rgb'.index(channel)]

    def native_normal(nx, ny, nz, fallback):
        vectors = []
        for name, label, weight in (('normal', 'Native normal', nz), ('tangent', 'Native tangent', nx),
                                    ('bitangent', 'Native bitangent', ny)):
            attribute = node('ShaderNodeAttribute', label, -120, -210)
            attribute.attribute_name = 'sunrise_native_' + name
            weighted = node('ShaderNodeVectorMath', label + ' contribution', 70, -210)
            weighted.operation = 'SCALE'
            connect(attribute.outputs['Vector'], weighted.inputs[0])
            connect(weight, weighted.inputs['Scale'])
            vectors.append(weighted.outputs['Vector'])
            if name == 'normal':
                length = node('ShaderNodeVectorMath', 'Native basis available', -120, -380)
                length.operation = 'LENGTH'
                connect(attribute.outputs['Vector'], length.inputs[0])
        combined = node('ShaderNodeVectorMath', 'Native normal plus tangent', 250, -210)
        combined.operation = 'ADD'
        connect(vectors[0], combined.inputs[0]); connect(vectors[1], combined.inputs[1])
        bitangent = node('ShaderNodeVectorMath', 'Interpolated native basis', 430, -210)
        bitangent.operation = 'ADD'
        connect(combined.outputs['Vector'], bitangent.inputs[0]); connect(vectors[2], bitangent.inputs[1])
        world = node('ShaderNodeVectorTransform', 'Native normal to world', 610, -210)
        world.vector_type = 'NORMAL'; world.convert_from = 'OBJECT'; world.convert_to = 'WORLD'
        connect(bitangent.outputs['Vector'], world.inputs['Vector'])
        normalized = node('ShaderNodeVectorMath', 'Normalize mapped normal', 800, -210)
        normalized.operation = 'NORMALIZE'
        connect(world.outputs['Vector'], normalized.inputs[0])
        choose = node('ShaderNodeMixRGB', 'Use authored basis when available', 990, -210)
        choose.blend_type = 'MIX'
        connect(math_node('GREATER_THAN', length.outputs['Value'], .000001), choose.inputs[0])
        connect(fallback, choose.inputs[1]); connect(normalized.outputs['Vector'], choose.inputs[2])
        return choose.outputs['Color']

    for binding in profile.get('Bindings', []):
        role = binding['Role']
        texture = slots.get(int(binding['Slot']))
        if texture is None or texture.image is None:
            continue
        formats = profile.get('TextureFormats') or {}
        native_format = formats.get(str(binding['Slot']), formats.get(binding['Slot']))
        srgb = int(native_format) in SRGB_FORMATS if native_format is not None else any(
            bool(value.get('Srgb')) for value in profile.get('Bindings', []) if value['Slot'] == binding['Slot'])
        texture.image = _preview_image(texture.image, srgb)
        roles = _add_preview_role(texture, role)
        texture.label = ' / '.join(roles) + ' | t' + str(binding['Slot'])
        tree.links.new(mapping.outputs['Vector'], texture.inputs['Vector'])
        channels = binding.get('Channels', 'rgb')
        if role == 'Normal':
            if binding.get('ReconstructNormalZ'):
                x, y = component(texture, channels[0]), component(texture, channels[1])
                scale, bias = binding.get('UnpackScale', 2), binding.get('UnpackBias', -1)
                nx, ny = math_node('MULTIPLY_ADD', x, scale, bias), math_node('MULTIPLY_ADD', y, scale, bias)
                length = math_node('ADD', math_node('MULTIPLY', nx, nx), math_node('MULTIPLY', ny, ny))
                z = math_node('SQRT', math_node('MAXIMUM', math_node('SUBTRACT', 1, length), 0))
                encoded = node('ShaderNodeCombineColor', 'Reconstructed normal', 10, -180)
                encoded.mode = 'RGB'
                connect(math_node('MULTIPLY_ADD', nx, .5, .5), encoded.inputs['Red'])
                connect(math_node('MULTIPLY_ADD', ny, .5, .5), encoded.inputs['Green'])
                connect(math_node('MULTIPLY_ADD', z, .5, .5), encoded.inputs['Blue'])
                value = encoded.outputs['Color']
            else:
                value = texture.outputs['Color']
            normal = node('ShaderNodeNormalMap', 'Tangent normal', 260, -130)
            normal.space = 'TANGENT'; normal.uv_map = 'UVMap'
            connect(value, normal.inputs['Color'])
            signals[role] = native_normal(nx, ny, z, normal.outputs['Normal']) if binding.get('ReconstructNormalZ') else normal.outputs['Normal']
        else:
            value = texture.outputs['Color'] if channels == 'rgb' else component(texture, channels[0])
            scale, bias = binding.get('Scale', 1), binding.get('Bias', 0)
            if scale != 1 or bias != 0:
                if channels == 'rgb':
                    adjusted = node('ShaderNodeVectorMath', role + ' scale', 10, 120)
                    adjusted.operation = 'MULTIPLY_ADD'
                    connect(value, adjusted.inputs[0])
                    adjusted.inputs[1].default_value = (scale,) * 3
                    adjusted.inputs[2].default_value = (bias,) * 3
                    value = adjusted.outputs['Vector']
                else:
                    value = math_node('MULTIPLY_ADD', value, scale, bias)
            signals[role] = value
    if 'RoughnessFloor' in signals:
        floor = math_node('MINIMUM', math_node('MAXIMUM', signals['RoughnessFloor'], 0), 1)
        signals['Roughness'] = math_node('MAXIMUM', signals.get('Roughness', profile.get('Roughness', .5)), floor)
    palette = profile.get('Palette')
    if palette and palette.get('Profile') == 'StaticPalette815B8187':
        mask, detail = slots.get(palette['MaskSlot']), slots.get(palette['DetailSlot'])
        if mask and mask.image and detail and detail.image:
            for texture, role in ((mask, 'PaletteMask'), (detail, 'PaletteDetail')):
                texture.image = _preview_image(texture.image, False)
                roles = _add_preview_role(texture, role)
                texture.label = ' / '.join(roles) + ' | t' + str(texture['sunrise_preview_slot'])
            connect(mapping.outputs['Vector'], mask.inputs['Vector'])
            detail_uv = node('ShaderNodeVectorMath', 'Native detail UV transform', -1080, -160)
            detail_uv.operation = 'MULTIPLY_ADD'
            connect(uv.outputs['UV'], detail_uv.inputs[0])
            values = palette['DetailUv']
            detail_uv.inputs[1].default_value = (values[0], values[1], 1)
            detail_uv.inputs[2].default_value = (values[2], values[3], 0)
            detail_flip = node('ShaderNodeVectorMath', 'Detail texture V orientation', -900, -160)
            detail_flip.operation = 'MULTIPLY_ADD'
            connect(detail_uv.outputs[0], detail_flip.inputs[0])
            detail_flip.inputs[1].default_value = (1, -1, 1)
            detail_flip.inputs[2].default_value = (0, 1, 0)
            connect(detail_flip.outputs[0], detail.inputs['Vector'])

            def color_math(operation, a, b, factor=1):
                value = node('ShaderNodeMixRGB', 'Palette ' + operation.lower(), 10, -420)
                value.blend_type = operation
                connect(factor, value.inputs[0]); connect(a, value.inputs[1]); connect(b, value.inputs[2])
                return value.outputs[0]

            colors = palette['Colors']
            a = color_math('ADD', colors[0], color_math('MULTIPLY', colors[1], component(detail, 'a')))
            b = color_math('ADD', colors[2], color_math('MULTIPLY', colors[3], component(detail, 'a')))
            c = color_math('ADD', colors[4], color_math('MULTIPLY', colors[5], component(detail, 'g')))
            mixed = color_math('MIX', color_math('MIX', a, b, component(mask, 'b')), c, component(mask, 'r'))
            blend = palette['Blend']
            factor = math_node('MINIMUM', math_node('MAXIMUM',
                               math_node('MULTIPLY_ADD', component(mask, 'g'), blend[1], blend[0]), 0), 1)
            signals['BaseColor'] = color_math('MIX', mixed, colors[6], factor)
    _apply_material_program(profile, slots, node, connect, math_node, component, uv.outputs['UV'], signals,
                            material.get('sunrise_preview_root') or material.get('sunrise_bundle_root'))
    _apply_material_recipe(profile, slots, node, connect, math_node, component, uv.outputs['UV'], signals, native_normal)
    if 'Visibility' in signals:
        # The native discard is a cutout; alpha carries it with the dithered render method.
        signals['Alpha'] = math_node('MULTIPLY', signals.get('Alpha', profile.get('Alpha', 1)), signals['Visibility'])
    for role in ('Metallic', 'Roughness'):
        if role in signals:
            signals[role] = math_node('MINIMUM', math_node('MAXIMUM', signals[role], 0), 1)
    # A blended G-buffer draw changes albedo before lighting, so it is a lit surface with straight colour.
    decal = profile.get('Deferred') and profile.get('BlendMode') in ('InverseAlpha', 'PremultipliedAlpha') \
        and 'BaseColor' in signals and 'BaseColor' in ((profile.get('Program') or {}).get('Outputs') or {})
    if decal:
        alpha = signals.get('Alpha', profile.get('Alpha', 0))
        straight = node('ShaderNodeVectorMath', 'Straight decal colour', 250, 120)
        straight.operation = 'SCALE'
        connect(signals['BaseColor'], straight.inputs[0])
        connect(math_node('DIVIDE', 1, math_node('MAXIMUM', alpha, .0001)), straight.inputs['Scale'])
        limited = node('ShaderNodeVectorMath', 'Decal colour limit', 380, 120)
        limited.operation = 'MINIMUM'
        connect(straight.outputs['Vector'], limited.inputs[0])
        limited.inputs[1].default_value = (1, 1, 1)
        signals['BaseColor'] = limited.outputs['Vector']
        signals['Alpha'] = alpha
    for role, socket in {'BaseColor': 'Base Color', 'Normal': 'Normal', 'Roughness': 'Roughness',
                         'Metallic': 'Metallic', 'Alpha': 'Alpha', 'Emission': 'Emission Color'}.items():
        if role in signals:
            connect(signals[role], surface.inputs[socket])
    water = profile.get('Recipe') == WATER_RECIPE
    if water:
        # The game's water covers what is behind it past a few centimetres, so it is an opaque water surface.
        surface.inputs['IOR'].default_value = WATER_IOR
    # Additive (ONE, ONE) is a clear surface plus emission: the colour adds to what is behind it.
    if profile.get('BlendMode') in ('InverseAlpha', 'PremultipliedAlpha', 'Additive') and not decal and not water:
        transparent = node('ShaderNodeBsdfTransparent', 'Background transmission', 250, -260)
        black = node('ShaderNodeBsdfDiffuse', 'Surface absorption', 250, -390)
        black.inputs['Color'].default_value = (0, 0, 0, 1)
        mix = node('ShaderNodeMixShader', 'Native opacity', 490, -260)
        connect(0 if profile.get('BlendMode') == 'Additive' else signals.get('Alpha', profile.get('Alpha', 0)), mix.inputs[0])
        connect(transparent.outputs[0], mix.inputs[1]); connect(black.outputs[0], mix.inputs[2])
        emission = node('ShaderNodeEmission', 'Premultiplied color', 490, -470)
        connect(signals.get('BaseColor', profile.get('BaseColor', [0, 0, 0, 1])), emission.inputs['Color'])
        combined = node('ShaderNodeAddShader', 'Native ' + profile['BlendMode'] + ' blend', 700, -150)
        connect(mix.outputs[0], combined.inputs[0]); connect(emission.outputs[0], combined.inputs[1])
        connect(combined.outputs[0], output.inputs['Surface'])
    # Native blending sorts back to front; a cutout or an opaque surface uses the dithered default.
    blended = profile.get('BlendMode') in ('InverseAlpha', 'PremultipliedAlpha', 'Additive', 'Alpha') and not water
    material.surface_render_method = 'BLENDED' if blended else 'DITHERED'
    material['sunrise_blend_mode'] = profile.get('BlendMode', 'Opaque')
    material['sunrise_preview_mode'] = profile.get('Mode', 'Material preview')
    material['sunrise_preview_notes'] = '\n'.join(profile.get('Notes', []))
    material['sunrise_preview_profile'] = json.dumps(profile)
    _place_nodes(tree, placed)
    _prune_preview(tree, output)


def _holds(socket, value):
    """True when an unlinked socket already holds value; writing it again would update the whole tree."""
    current = getattr(socket, 'default_value', None)
    if current is None or isinstance(current, (str, bpy.types.ID)):
        return False
    if isinstance(value, (int, float)):
        return isinstance(current, (int, float)) and np.float32(value) == np.float32(current)
    try:
        return len(current) == len(value) and bool((np.asarray(current[:], np.float32) == np.asarray(value, np.float32)).all())
    except TypeError:
        return False


def _place_nodes(tree, placed):
    """Sets the locations of new nodes in one write; nodes must not have a frame yet."""
    wanted = {node.name: (x, y) for node, x, y in placed}
    if not wanted:
        return
    locations = np.empty(len(tree.nodes) * 2, dtype=np.float32)
    tree.nodes.foreach_get('location', locations)
    for index, item in enumerate(tree.nodes):
        if item.name in wanted:
            locations[index * 2:index * 2 + 2] = wanted[item.name]
    tree.nodes.foreach_set('location', locations)


def _prune_preview(tree, output):
    """Preview nodes that do not reach the output are removed; unwired native fields go in one frame."""
    # socket.links scans the whole tree, so the sources of each node are read in one pass.
    sources = {}
    for link in tree.links:
        sources.setdefault(link.to_node.name, []).append(link.from_node)
    reached, pending = set(), [output]
    while pending:
        item = pending.pop()
        if item.name not in reached:
            reached.add(item.name)
            pending.extend(sources.get(item.name, ()))
    for item in list(tree.nodes):
        if item.name not in reached and item.get('sunrise_preview_node'):
            tree.nodes.remove(item)
    frame = tree.nodes.get('Sunrise native fields')
    for item in tree.nodes:
        if item.bl_idname == 'NodeFrame':
            continue
        if item.name in reached:
            if frame is not None and item.parent == frame:
                item.parent = None
            continue
        if frame is None:
            frame = tree.nodes.new('NodeFrame')
            frame.name = 'Sunrise native fields'
            frame.label = 'Native fields (edit only)'
        item.parent = frame


def _material(root, info, images, bundle=None):
    """Native fields remain editable while shader-derived roles drive the visible surface."""
    source = _path(root, info["Path"])
    document = _read(source)
    if document.get("Format") != "sunrise-material" or document.get("Version") != 1:
        raise ValueError("Unsupported native material document.")
    material = bpy.data.materials.new(info.get("Name") or document["Name"])
    material.use_fake_user = True
    if info.get("Variant"):
        # Only the plain material carries the native tag, so edits and the fields panel stay on one material.
        material["sunrise_part_variant"] = info["Variant"]
    else:
        material["sunrise_material_tag"] = f'{info["TagHash"]:08X}'
    material["sunrise_material_path"] = info["Path"]
    material["sunrise_bundle_root"] = str(root)
    material["sunrise_material_sha256"] = _sha(source)
    material["sunrise_preview_only"] = True
    for index, slot in enumerate(document["Slots"]):
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.name = f'{slot["Stage"]} slot {slot["Slot"]}'
        node.label = node.name
        node.location = (-700, -index * 260)
        node["sunrise_slot_offset"] = slot["Offset"]
        node['sunrise_preview_slot'] = slot['Slot']
        node['sunrise_preview_stage'] = slot['Stage']
        node["sunrise_texture_tag"] = f'{slot["TextureTag"]:08X}'
        node.image = images.get((slot["TextureTag"], 0, 0, 0))
        if node.image is None:
            file = next((f for f in document["Textures"] if f["TagHash"] == slot["TextureTag"]), None)
            if file and (file.get("PreviewPath") or file.get("DdsPath")):
                relative = file.get("PreviewPath") or file["DdsPath"]
                node.image = _load_image(source.parent, relative, file["TagHash"])
    for index, field in enumerate(document["Constants"]):
        if not math.isfinite(float(field["Value"])):
            continue
        node = material.node_tree.nodes.new("ShaderNodeValue")
        node.name = f'Native {field["Offset"]:X}'
        node.label = f'{field["Stage"]} {field["Bank"]} {field["Vector"]}.{field["Component"]}'
        node["sunrise_constant_offset"] = field["Offset"]
        node.outputs[0].default_value = float(field["Value"])
        node.location = (-2050 + index % 4 * 180, -350 - index // 4 * 120)
    profile = info.get('Preview') or _legacy_material_preview(root, info, document, bundle or {})
    # Textures the part's draw binds, such as plate atlases; preview only, so they carry no slot offset.
    for index, (slot, tag) in enumerate(sorted((int(k), v) for k, v in (profile.get('SlotTextures') or {}).items())):
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.name = f'Pixel slot {slot} draw texture'
        node.label = node.name
        node.location = (-950, -index * 260)
        node['sunrise_preview_slot'] = slot
        node['sunrise_preview_stage'] = 'Pixel'
        node['sunrise_preview_override'] = True
        node["sunrise_texture_tag"] = f'{tag:08X}'
        node.image = images.get((tag, 0, 0, 0))
    _build_material_preview(material, profile)
    return material


def _material_key(tag, variant=None):
    """Loaded materials are keyed by tag, plus the variant for a part's preview-only copy."""
    return f'{tag}:{variant}' if variant else str(tag)


def _mesh_material(document, info):
    """The manifest material a part draws with: its variant when it names one, else the plain material."""
    materials = document.get('Materials', [])
    variant = info.get('MaterialVariant')
    return next((m for m in materials if m['TagHash'] == info['MaterialTag'] and m.get('Variant') == variant), None) \
        or next((m for m in materials if m['TagHash'] == info['MaterialTag'] and not m.get('Variant')), None)


def _stage_blend(profile, stage):
    """A stage the export did not resolve keeps the material's own blend."""
    profile = profile or {}
    return (profile.get('StageBlendModes') or {}).get(str(stage), profile.get('BlendMode', 'Opaque'))


def _apply_stage_blends(collection, document):
    """Each part uses its render stage's blend; a variant is a read-only copy of the base."""
    loaded = json.loads(collection.get('sunrise_loaded_materials', '{}'))
    previous = set(json.loads(collection.get('sunrise_blend_variants', '{}')).values())
    for material in [m for m in bpy.data.materials if m.get('sunrise_id') in previous]:
        bpy.data.materials.remove(material)
    bases = {m['sunrise_id']: m for m in bpy.data.materials if m.get('sunrise_id') in set(loaded.values())}
    objects = {obj.get('sunrise_mesh_id'): obj for obj in collection.all_objects if obj.get('sunrise_mesh_id')}
    variants = {}
    for info in document['Meshes']:
        obj = objects.get(info['Id'])
        material_info = _mesh_material(document, info) or {}
        material_key = _material_key(info['MaterialTag'], material_info.get('Variant'))
        base = bases.get(loaded.get(material_key))
        if obj is None or base is None:
            continue
        profile = material_info.get('Preview') or {}
        blend = _stage_blend(profile, info.get('RenderStage', -1))
        obj['sunrise_blend_mode'] = blend
        material = base
        variant = (profile.get('BlendVariants') or {}).get(blend)
        if variant and blend != profile.get('BlendMode', 'Opaque'):
            key = f'{material_key}:{blend}'
            if key not in variants:
                copy = base.copy()
                # Only the base carries the native tag, so edits and the fields panel stay on one material.
                for name in ('sunrise_id', 'sunrise_material_tag'):
                    if name in copy:
                        del copy[name]
                copy.name = f'{base.name} [{blend}]'
                copy['sunrise_variant_of'] = _uid(base)
                _build_material_preview(copy, variant)
                variants[key] = copy
            material = variants[key]
        if len(obj.data.materials) == 0:
            obj.data.materials.append(material)
        else:
            obj.data.materials[0] = material
        obj['sunrise_material_slots'] = json.dumps([_uid(m) for m in obj.data.materials])
    collection['sunrise_blend_variants'] = json.dumps({key: _uid(value) for key, value in variants.items()})


def _material_shading():
    """Every 3D view shows material preview, where the preview shaders draw."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.type = 'MATERIAL'


BUNDLE_PHASES = ('Reading the bundle', 'Textures', 'Materials', 'glTF scene', 'Meshes', 'Effects and collision', 'Finishing')
# Data the glTF importer and the bundle import can add.
BUNDLE_KINDS = ('objects', 'collections', 'meshes', 'materials', 'images', 'actions', 'armatures', 'node_groups', 'texts',
                'cameras', 'lights', 'curves')


def import_bundle(manifest_path):
    """Import original geometry identities; glTF supplies the rig and preview actions."""
    return progress.run(import_bundle_steps(manifest_path))


def import_bundle_steps(manifest_path):
    """The bundle import as steps that yield (phase, done, total); closing them early removes what they added."""
    path = _manifest_path(manifest_path)
    yield BUNDLE_PHASES[0], 0, 0
    document = _read(path)
    if document.get("Format") != "sunrise-asset-bundle" or document.get("Version") != 1:
        raise ValueError("Unsupported Sunrise asset bundle.")
    if any(c.get("sunrise_manifest") == str(path) for c in bpy.data.collections):
        raise ValueError("This bundle is already imported.")
    root = path.parent
    scene = bpy.context.scene
    before = progress.snapshot(BUNDLE_KINDS)
    scene_state = (scene.render.fps, scene.render.fps_base, scene.get("sunrise_active_bundle"))
    previous = before['objects']
    previous_actions = before['actions']
    collection = bpy.data.collections.new(document["Name"])
    scene.collection.children.link(collection)
    collection["sunrise_manifest"] = str(path)
    collection["sunrise_manifest_sha256"] = _sha(path)
    collection["sunrise_external_objects"] = json.dumps([_uid(o) for o in previous])
    try:
        material_warnings = [notice for notice in document.get("Notices", [])
                             if notice.get("Surface") == "Materials" and notice.get("Status") == "Initial state fallback"]
        collection["sunrise_material_warnings"] = json.dumps(material_warnings)
        for notice in material_warnings:
            print("Sunrise material fallback: " + notice["Message"])
        # Build materials before the glTF import; a node rename is slow once actions exist.
        images = {}
        items = [(texture, item) for texture in document.get("Textures", []) for item in texture["Images"]
                 if (item["Mip"], item["Slice"], item["DepthSlice"]) == (0, 0, 0)]
        for index, (texture, item) in enumerate(items):
            key = (texture["TagHash"], item["Mip"], item["Slice"], item["DepthSlice"])
            images[key] = _load_image(root, item["Path"], *key, expected=item["Sha256"])
            yield 'Textures', index + 1, len(items)
        used = {_material_key(m["TagHash"], m.get("Variant"))
                for m in (_mesh_material(document, info) for info in document["Meshes"]) if m}
        used |= {str(document["RootTag"]), str(document["SelectedTag"])}
        wanted = [info for info in document.get("Materials", []) if _material_key(info["TagHash"], info.get("Variant")) in used]
        materials = {}
        for info in wanted:
            materials[_material_key(info["TagHash"], info.get("Variant"))] = _material(root, info, images, document)
            yield 'Materials', len(materials), len(wanted)
        collection["sunrise_loaded_materials"] = json.dumps({key: _uid(value) for key, value in materials.items()})
        scene_document = None
        if document.get("ScenePath"):
            scene_path = _path(root, document["ScenePath"])
            scene_document = _validate_gltf(root, scene_path)
            yield 'glTF scene', 0, 0
            scene.render.fps = 30
            scene.render.fps_base = 1
            bpy.ops.import_scene.gltf(filepath=str(scene_path), merge_vertices=False, import_pack_images=False,
                                      disable_bone_shape=True)
        imported = list(set(bpy.data.objects) - previous)
        for obj in imported:
            for owner in list(obj.users_collection):
                owner.objects.unlink(obj)
            collection.objects.link(obj)
            _uid(obj)
        source_bones = {n.get("name"): n["extras"]["sunriseBoneIndex"]
                        for n in (scene_document or {}).get("nodes", [])
                        if "sunriseBoneIndex" in n.get("extras", {})}
        bones = {}
        for obj in imported:
            if obj.type == 'ARMATURE':
                for bone in obj.data.bones:
                    index = bone.get("sunriseBoneIndex", source_bones.get(bone.name))
                    if index is not None:
                        bones[int(index)] = bone.name
        by_id = {}
        for obj in imported:
            if obj.type == 'MESH' and obj.get("sunriseMeshId"):
                mesh_id = obj["sunriseMeshId"]
                if mesh_id in by_id:
                    raise ValueError("Duplicate source mesh identity.")
                by_id[mesh_id] = obj
        if set(by_id) != {info["Id"] for info in document["Meshes"]}:
            raise ValueError("glTF mesh identities differ from the manifest.")
        materialless = 0
        guardian = bool((document.get('Context') or {}).get('Guardian'))
        for mesh_index, info in enumerate(document["Meshes"]):
            yield 'Meshes', mesh_index, len(document["Meshes"])
            source = _path(root, info["Path"])
            if _sha(source) != info["Sha256"]:
                raise ValueError("Original geometry changed on disk.")
            material_info = _mesh_material(document, info) or {}
            profile = material_info.get('Preview') or {}
            blend = _stage_blend(profile, info.get('RenderStage', -1))
            profile = (profile.get('BlendVariants') or {}).get(blend, profile)
            program = profile.get('Program') or {}
            pending = list((program.get('Outputs') or {}).get('Normal', []))
            visited, needs_basis = set(), False
            while pending:
                index = pending.pop()
                if index in visited:
                    continue
                visited.add(index)
                value = program['Nodes'][index]
                needs_basis |= value['Operation'] in ('basis', 'frontface')
                pending.extend(value['Inputs'])
            needs_basis &= not (profile.get('Recipe') or profile.get('Palette') or blend == 'NoColor')
            obj = _mesh_baseline(by_id[info["Id"]], info, _read(source), bones, needs_basis)
            material_key = _material_key(info["MaterialTag"], material_info.get("Variant"))
            if material_key in materials:
                obj.data.materials.append(materials[material_key])
            obj["sunrise_material_slots"] = json.dumps([_uid(m) for m in obj.data.materials])
            obj['sunrise_render_stage'] = info.get('RenderStage', -1)
            # A decal that reads no frame texture draws on its own mesh; an older bundle does not say, so it stays hidden.
            obj['sunrise_surface_decal'] = guardian or bool(profile.get('Program')) and profile.get('ReadsFrameTextures') is False
            if info['MaterialTag'] in (0, 0xFFFFFFFF):
                materialless += 1
            if _decal_hidden(obj, bpy.context.scene):
                obj.hide_render = True
                obj.hide_set(True)
        collection['sunrise_materialless_parts'] = materialless
        _apply_stage_blends(collection, document)
        if materialless:
            print(f'Sunrise bundle: {materialless} mesh parts have no resolved material.')
        yield 'Effects and collision', 0, 0
        _import_emitters(root, collection, document)
        # Run as a single file, without its package, the bundle draws no baked effects.
        if __package__:
            from . import effects
            effects.import_bundle_effects(root, collection)
        _import_collision(root, collection, document)
        for shader in document.get("Shaders", []):
            source = _path(root, shader["DisassemblyPath"])
            if source.stat().st_size > MAX_BYTES:
                raise ValueError("Shader source exceeds its size limit.")
            text = bpy.data.texts.load(str(source))
            text.use_fake_user = True
            text["sunrise_shader_tag"] = f'{shader["HeaderTag"]:08X}'
            text["sunrise_bytecode_path"] = str(_path(root, shader["Path"]))
            text["sunrise_bundle_root"] = str(root)
            text["sunrise_source_sha256"] = hashlib.sha256(text.as_string().encode()).hexdigest()
        clip_names = {f'{a["ClipTag"]:08X}': a["Name"] for a in document.get("Animations", [])}
        for action in set(bpy.data.actions) - previous_actions:
            _uid(action)
            action["sunrise_pose_preview"] = True
            action["sunrise_original_action"] = _action_state(action)
            tag = action.get('sunriseClipTag') or next((t for t in clip_names if 'Pose_' + t in action.name), None)
            if tag in clip_names:
                # The tag stays in the name and a property; pose lookup matches either.
                action['sunriseClipTag'] = tag
                action.name = f'{clip_names[tag]} [Pose_{tag}]'
        yield 'Finishing', 0, 0
        bpy.context.view_layer.update()
        collection["sunrise_objects"] = json.dumps([_uid(o) for o in collection.all_objects])
        collection["sunrise_rigs"] = json.dumps({_uid(o): _rig_state(o) for o in imported if o.type == 'ARMATURE'})
        collection["sunrise_actions"] = json.dumps([_uid(a) for a in set(bpy.data.actions) - previous_actions])
        bpy.context.scene["sunrise_active_bundle"] = str(path)
        _prepare_pose_actions(collection, initialize_ranges=True)
        _bind_pose(collection)
        _apply_lod_visibility(collection, document, bpy.context.scene)
        _material_shading()
        return collection
    except BaseException:
        scene.render.fps, scene.render.fps_base = scene_state[:2]
        if scene_state[2] is None:
            scene.pop("sunrise_active_bundle", None)
        else:
            scene["sunrise_active_bundle"] = scene_state[2]
        progress.rollback(before)
        raise


def _import_emitters(root, collection, document):
    if not document.get("ParticleMetadataPath"):
        return
    particles = _read(_path(root, document["ParticleMetadataPath"]))
    for emitter in particles["Emitters"]:
        obj = bpy.data.objects.new(f'Emitter {emitter["TagHash"]:08X}', None)
        collection.objects.link(obj)
        obj.empty_display_type = 'SPHERE'
        obj.empty_display_size = 0.5
        placement = next((effect for effect in particles.get("Effects", [])
                          if any(system["EmitterTag"] == emitter["TagHash"] for system in effect.get("Systems", []))), None)
        if placement and isinstance(placement.get("Transform"), dict):
            values = placement["Transform"]
            obj.matrix_world = _matrix([values[f"M{row}{column}"]
                                       for row in range(1, 5) for column in range(1, 5)])
        obj["sunrise_emitter_tag"] = f'{emitter["TagHash"]:08X}'
        obj["sunrise_preview_only"] = True
        obj["sunrise_preview_transform"] = json.dumps(_flat(obj.matrix_world))
        for field in emitter["Constants"]:
            key = f'native_{field["Channel"]}_{field["Offset"]:X}'
            obj[key] = [float(v) for v in field["Value"]]
            obj.id_properties_ui(key).update(description=field["Name"])
        _uid(obj)


def _collision_document(root, document):
    """The collision section must be the one Studio wrote; its hash is in the manifest."""
    path = _path(root, document["CollisionPath"])
    if _sha(path) != document.get("CollisionSha256"):
        raise ValueError("The collision section changed on disk.")
    data = _read(path)
    if data.get("Format") != "sunrise-collision" or data.get("Version") != 1:
        raise ValueError("Unsupported collision section.")
    return data


def _import_collision(root, collection, document):
    """Each placed shape is a wire object that never renders; placements of one shape share its mesh."""
    if not document.get("CollisionPath"):
        return
    data = _collision_document(root, document)
    group = bpy.data.collections.new("Collision")
    collection.children.link(group)
    group["sunrise_collision"] = True
    # A new object may name one of its compound's material tags in sunrise_collision_material.
    group["sunrise_collision_materials"] = json.dumps(data.get("Materials") or {})
    pieces = {piece["Id"]: piece for piece in data["Pieces"]}
    meshes = {}
    for piece in data["Pieces"]:
        indices = piece["Indices"]
        if len(indices) % 3 or any(i < 0 or i >= len(piece["Positions"]) for i in indices):
            raise ValueError("A collision shape has invalid triangles.")
        if not indices:
            meshes[piece["Id"]] = None
            continue
        mesh = bpy.data.meshes.new(f'{piece["Kind"]} {piece["Tag"]:08X}')
        mesh.from_pydata(piece["Positions"], [], [indices[i:i + 3] for i in range(0, len(indices), 3)])
        mesh["sunrise_collision_piece"] = piece["Id"]
        meshes[piece["Id"]] = mesh
    for placement in data["Placements"]:
        piece = pieces[placement["Piece"]]
        mesh = meshes[placement["Piece"]]
        obj = bpy.data.objects.new(placement["Name"], mesh)
        group.objects.link(obj)
        obj.matrix_world = _matrix(placement["Transform"])
        obj["sunrise_collision_baseline_transform"] = json.dumps(_flat(obj.matrix_basis.transposed()))
        if mesh is None:
            # Spheres and capsules have no triangles; an empty keeps their placement visible.
            obj.empty_display_type = 'SPHERE'
            obj.empty_display_size = 0.5
        else:
            obj.display_type = 'WIRE'
        obj.hide_render = True
        obj["sunrise_collision_id"] = placement["Id"]
        obj["sunrise_collision_piece"] = placement["Piece"]
        obj["sunrise_collision_source"] = placement["Source"]
        obj["sunrise_collision_entity"] = f'{placement["Entity"]:08X}'
        obj["sunrise_collision_compound"] = f'{placement["CompoundTag"]:08X}'
        obj["sunrise_collision_instance"] = placement["Instance"]
        obj["sunrise_collision_shape"] = f'{piece["Tag"]:08X}'
        obj["sunrise_collision_shape_offset"] = piece["Offset"]
        obj["sunrise_collision_kind"] = piece["Kind"]
        obj["sunrise_collision_editable"] = placement["Editable"]
        _uid(obj)
    collection["sunrise_collision_count"] = len(data["Placements"])


def _same_transform(values, expected):
    """Translation uses absolute metres, so distant objects retain small edits."""
    return len(values) == 16 and all(math.isfinite(v) and abs(v - e) <=
                                     (0.001 if 12 <= i <= 14 else 0.00001 * max(1, abs(e)))
                                     for i, (v, e) in enumerate(zip(values, expected)))


def _collision_transform(obj, placement):
    """Blender's canonical import matrix is the edit baseline, not a rewrite of native bind data."""
    source = _matrix(placement['Transform'])
    temporary = bpy.data.objects.new('Sunrise collision baseline', None)
    try:
        temporary.matrix_world = source
        canonical = temporary.matrix_basis.copy()
    finally:
        bpy.data.objects.remove(temporary)
    stored = obj.get('sunrise_collision_baseline_transform')
    baseline = _matrix(json.loads(stored)) if stored else canonical
    if not _same_transform(_flat(baseline.transposed()), _flat(canonical.transposed())):
        raise ValueError(f'{obj.name}: the collision import baseline changed.')
    if _same_transform(_flat(obj.matrix_world.transposed()), _flat(baseline.transposed())):
        return False, placement['Transform']
    return True, _flat((obj.matrix_world @ baseline.inverted() @ source).transposed())


def _in_collision_group(obj):
    return any(c.get("sunrise_collision") for c in obj.users_collection)


def _triangles(mesh):
    """Positions and triangle indices of a mesh in its own space."""
    if all(len(face.vertices) == 3 for face in mesh.polygons):
        indices = [v for face in mesh.polygons for v in face.vertices]
    else:
        mesh.calc_loop_triangles()
        indices = [v for triangle in mesh.loop_triangles for v in triangle.vertices]
    return [list(v.co) for v in mesh.vertices], indices


def _collision_addition(obj, source, values):
    """A new shape from an object's own mesh; its kind, compound and material come from custom properties."""
    if obj.type != 'MESH' or not obj.data.polygons:
        raise ValueError(f"{obj.name}: a new collision object needs a mesh with faces.")
    if obj.mode == 'EDIT':
        obj.update_from_editmode()
    positions, indices = _triangles(obj.data)
    return {"Source": source, "Name": obj.name, "Transform": values, "Positions": positions, "Indices": indices,
            "Kind": str(obj.get("sunrise_collision_kind", "Mesh")), "Compound": obj.get("sunrise_collision_compound"),
            "Material": obj.get("sunrise_collision_material") or None}


def _collision_edits(root, document, objects):
    """Moved, edited, copied and new collision objects; removing an exported one is refused."""
    if not document.get("CollisionPath"):
        return None
    data = _collision_document(root, document)
    pieces = {piece["Id"]: piece for piece in data["Pieces"]}
    placements = {placement["Id"]: placement for placement in data["Placements"]}
    found, new = {}, []
    for obj in objects:
        identity = obj.get("sunrise_collision_id")
        if identity is not None:
            found.setdefault(identity, []).append(obj)
        elif _in_collision_group(obj):
            new.append(obj)
    if set(found) - set(placements):
        raise ValueError("A collision object names a placement that was not exported.")
    if set(placements) - set(found):
        raise ValueError("Collision objects were removed. Removing an exported collision instance is not supported, "
                         "because it renumbers the instances after it.")
    moved, shapes, meshes, added = [], [], {}, []
    for identity, copies in found.items():
        placement = placements[identity]
        # A copy keeps the original's properties; the original keeps the exported name.
        named = [o for o in copies if o.name == placement["Name"]]
        if len(copies) > 1 and len(named) != 1:
            raise ValueError(f'Several objects copy {placement["Name"]}; keep that name on the original only.')
        obj = named[0] if len(copies) > 1 else copies[0]
        for copy in copies:
            if copy.mode == 'EDIT':
                copy.update_from_editmode()
        for copy in copies:
            if copy is obj:
                continue
            if copy.modifiers or copy.constraints:
                raise ValueError(f"{copy.name}: apply modifiers and constraints before exporting collision.")
            _, values = _collision_transform(copy, placement)
            if copy.data is not None and copy.data is not obj.data and (copy.type != 'MESH' or obj.data is None
                                                                        or _triangles(copy.data) != _triangles(obj.data)):
                added.append(_collision_addition(copy, identity, values))
            elif not placement["Editable"]:
                raise ValueError(f"{copy.name} copies a shape with no stored transform, so it cannot be placed again.")
            else:
                added.append({"Source": identity, "Name": copy.name, "Transform": values})
        if obj.modifiers or obj.constraints:
            raise ValueError(f"{obj.name}: apply modifiers and constraints before exporting collision.")
        changed, values = _collision_transform(obj, placement)
        if changed:
            if not placement["Editable"]:
                raise ValueError(f"{obj.name} has no stored transform, so it cannot move.")
            moved.append({"Id": identity, "Transform": values})
        if placement["Piece"] not in pieces or (obj.data is None) != (not pieces[placement["Piece"]]["Indices"]):
            raise ValueError(f"{obj.name} changed its object type.")
        if obj.data is None:
            continue
        if obj.data.get("sunrise_collision_piece") != placement["Piece"]:
            raise ValueError(f"{obj.name} uses another mesh; collision objects keep their shape's mesh.")
        previous = meshes.get(placement["Piece"])
        if previous is not None and previous is not obj.data and _triangles(previous) != _triangles(obj.data):
            raise ValueError(f"{obj.name}: placements of one native shape have different meshes. "
                             "Keep their geometry linked, or add a new collision object instead.")
        meshes[placement["Piece"]] = obj.data
    for piece_id, mesh in meshes.items():
        piece = pieces[piece_id]
        positions, indices = _triangles(mesh)
        if indices == piece["Indices"] and _same_vectors(positions, piece["Positions"]):
            continue
        shapes.append({"Piece": piece_id, "Positions": positions, "Indices": indices})
    for obj in new:
        if obj.modifiers or obj.constraints:
            raise ValueError(f"{obj.name}: apply modifiers and constraints before exporting collision.")
        added.append(_collision_addition(obj, None, [float(v) for row in obj.matrix_world.transposed() for v in row]))
    return {"Placements": moved, "Shapes": shapes, "Added": added}


def _check_objects(collection):
    """Removing every mesh is still a deletion, never an unchanged bundle."""
    expected = json.loads(collection["sunrise_objects"])
    objects = list(collection.all_objects)
    # Collision objects may be copied or added; _collision_edits checks them.
    collision = [o for o in objects if _in_collision_group(o)]
    actual = [o.get("sunrise_id") for o in objects if o not in collision]
    if len(set(actual)) != len(actual) or not set(actual) <= set(expected) \
            or set(expected) - set(actual) - {o.get("sunrise_id") for o in collision}:
        raise ValueError("Objects were added, removed, or duplicated; fixed-layout bundles cannot import that change.")
    external = set(json.loads(collection["sunrise_external_objects"]))
    for obj in bpy.context.scene.objects:
        if obj not in objects and obj.get("sunrise_id") not in external:
            raise ValueError("A new object is outside the source bundle; import new topology separately in Studio.")
    rigs = json.loads(collection["sunrise_rigs"])
    preview_ids = set(json.loads(collection["sunrise_actions"]))
    for obj in objects:
        if obj.get("sunrise_id") in rigs:
            if obj.type != 'ARMATURE' or _rig_state(obj) != rigs[obj["sunrise_id"]]:
                raise ValueError("Rig rest layout or armature transform changed.")
            animation = obj.animation_data
            actions = ([animation.action] if animation and animation.action else [])
            if animation:
                actions += [s.action for t in animation.nla_tracks for s in t.strips if s.action]
            if any(a.get("sunrise_id") not in preview_ids for a in actions):
                raise ValueError("New bone actions cannot be mapped back to native controls.")
    for identity in preview_ids:
        actions = [a for a in bpy.data.actions if a.get("sunrise_id") == identity]
        if len(actions) != 1 or not _action_unchanged(actions[0], actions[0].get("sunrise_original_action")):
            raise ValueError("Sampled bone actions are preview only; edit the native control actions.")
    return objects


def _vertex_values(mesh, values, width):
    output = [None] * len(mesh.vertices)
    for loop in mesh.loops:
        value = list(values[loop.index])[:width]
        previous = output[loop.vertex_index]
        if previous is not None and not _close(previous, value):
            raise ValueError("Split corner attributes would change native vertex identity.")
        output[loop.vertex_index] = value
    return output


def _mesh_edit(obj, info, original):
    """Native geometry edits retain topology, joints, and source-space vertex identities."""
    if obj.type != 'MESH':
        raise ValueError("A native mesh was replaced by a different object type.")
    mesh = obj.data
    if obj.mode == 'EDIT':
        obj.update_from_editmode()
    identity = mesh.attributes.get("sunrise_vertex_id")
    if identity is None or identity.domain != 'POINT' or identity.data_type != 'INT' or \
            [v.value for v in identity.data] != list(range(len(original["Positions"]))):
        raise ValueError("Source vertices were removed, reordered, or duplicated.")
    indices = [v for face in mesh.polygons for v in face.vertices]
    if any(len(face.vertices) != 3 for face in mesh.polygons) or \
            indices != [v for face in _faces(original["Indices"]) for v in face]:
        raise ValueError("Native topology or face order changed.")
    modifiers = [(m.type, _uid(m.object) if m.type == 'ARMATURE' and m.object else None) for m in obj.modifiers]
    if modifiers != [tuple(x) for x in json.loads(obj["sunrise_modifiers"])] or mesh.shape_keys:
        raise ValueError("Modifiers or shape keys would add unsupported geometry changes.")
    if obj.get('sunrise_basis_group'):
        group = bpy.data.node_groups.get(obj['sunrise_basis_group'])
        if group is None or not any(m.type == 'NODES' and m.node_group == group for m in obj.modifiers) \
                or _motion_state(group, basis=True) != group.get('sunrise_basis_state'):
            raise ValueError('Native shader basis is a read-only preview.')
        expected_basis = {'normal': original['Normals'], 'tangent': [value[:3] for value in original['Tangents']],
                          'bitangent': [list(Vector(n).cross(Vector(t[:3])) * t[3])
                                        for n, t in zip(original['Normals'], original['Tangents'])]}
        for name, values in expected_basis.items():
            attribute = mesh.attributes.get('sunrise_native_' + name)
            if attribute is None or attribute.data_type != 'FLOAT_VECTOR' or attribute.domain != 'POINT' \
                    or len(attribute.data) != len(values) or any(not math.isfinite(a) or abs(a - b) > .000001
                        for item, expected in zip(attribute.data, values) for a, b in zip(item.vector, expected)):
                raise ValueError('Native tangent basis attributes are read-only preview data.')
    if original.get('VertexMotion'):
        group = bpy.data.node_groups.get(obj.get('sunrise_motion_group', ''))
        if group is None or not any(m.type == 'NODES' and m.node_group == group for m in obj.modifiers) \
                or _motion_state(group) != group.get('sunrise_motion_state'):
            raise ValueError('Native shader motion is a read-only preview.')
        attribute = mesh.attributes.get('sunrise_motion_color')
        if original['VertexMotion'].get('Profile') == 'VertexWave5' and (attribute is None or any(abs(a - b) > .000001
                for item, expected in zip(attribute.data, original['VertexMotion']['Colors']) for a, b in zip(item.color, expected))):
            raise ValueError('Native motion colours cannot be edited as mesh geometry.')
    if obj.constraints or obj.animation_data and (obj.animation_data.action or obj.animation_data.drivers):
        raise ValueError("Mesh animation and constraints cannot be baked into fixed bundle geometry.")
    if any(m is None for m in mesh.materials) or \
            [_uid(m) for m in mesh.materials] != json.loads(obj["sunrise_material_slots"]) or \
            any(p.material_index != 0 for p in mesh.polygons):
        raise ValueError("Mesh material assignments changed; edit native material texture slots instead.")
    bones = {int(k): v for k, v in json.loads(obj["sunrise_bones"]).items()}
    if original.get("Joints") is not None:
        names = {g.index: g.name for g in obj.vertex_groups}
        for vertex, joints, weights in zip(mesh.vertices, original["Joints"], original["Weights"]):
            expected = {}
            for joint, weight in zip(joints, weights):
                if weight:
                    expected[bones[joint]] = expected.get(bones[joint], 0) + weight
            actual = {names[g.group]: g.weight for g in vertex.groups if g.weight}
            if set(actual) != set(expected) or any(abs(actual[k] - expected[k]) > 0.000002 for k in expected):
                raise ValueError("Skin weights or native joint assignments changed.")
    elif obj.vertex_groups:
        raise ValueError("New skin groups cannot be imported into this native mesh.")
    source = _matrix(info["Transform"])
    delta = Matrix.Identity(4) if _close(_flat(source), _flat(obj.matrix_world)) else source.inverted() @ obj.matrix_world
    positions = [list(delta @ v.co) for v in mesh.vertices]
    normals = None
    normal_transform = delta.to_3x3().inverted().transposed()
    authored_normals = _normal_state(mesh) != obj["sunrise_normal_state"]
    if authored_normals or not _close(_flat(normal_transform), _flat(Matrix.Identity(3))):
        values = _vertex_values(mesh, [n.vector for n in mesh.corner_normals], 3) if authored_normals else original["Normals"]
        normals = [list((normal_transform @ Vector(value or original["Normals"][i])).normalized())
                   for i, value in enumerate(values)]
        if _same_vectors(normals, original['Normals']):
            normals = None
    # The detail layer is derived from the native streams, so only the first layer is an edit.
    detail = original.get("TexCoords1") is not None
    if len(mesh.uv_layers) != (2 if detail else 1) or detail and mesh.uv_layers[1].name != "UVMap1":
        raise ValueError("Keep one native UV layer.")
    uvs = _vertex_values(mesh, [v.uv for v in mesh.uv_layers[0].data], 2)
    uvs = [value if value is not None else original["TexCoords"][i] for i, value in enumerate(uvs)]
    uv_changed = not _same_vectors(uvs, original["TexCoords"])
    if _same_vectors(positions, original["Positions"]) and normals is None and not uv_changed:
        return None
    return {"Id": info["Id"], "Positions": positions, "Indices": original["Indices"],
            "Joints": original.get("Joints"), "Weights": original.get("Weights"),
            "Normals": normals, "TexCoords": uvs if uv_changed else None}


def _material_edit(root, info, transaction, known_textures):
    matches = [m for m in bpy.data.materials if m.get("sunrise_material_tag") == f'{info["TagHash"]:08X}'
               and m.get("sunrise_material_path") == info["Path"] and m.get("sunrise_bundle_root") == str(root)]
    if len(matches) != 1:
        raise ValueError("A native material is missing or duplicated.")
    material = matches[0]
    source = _path(root, info["Path"])
    if _sha(source) != material["sunrise_material_sha256"]:
        raise ValueError("The source material document changed.")
    document = _read(source)
    changed = False
    slots = {f["Offset"]: f for f in document["Slots"]}
    fields = {f["Offset"]: f for f in document["Constants"] if math.isfinite(float(f["Value"]))}
    slot_nodes = [n for n in material.node_tree.nodes if "sunrise_slot_offset" in n]
    field_nodes = [n for n in material.node_tree.nodes if "sunrise_constant_offset" in n]
    if sorted(n["sunrise_slot_offset"] for n in slot_nodes) != sorted(slots) or \
            sorted(n["sunrise_constant_offset"] for n in field_nodes) != sorted(fields):
        raise ValueError("Native material fields were removed or duplicated.")
    for node in slot_nodes:
        field = slots[node["sunrise_slot_offset"]]
        value = int(str(node["sunrise_texture_tag"]), 16)
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("Native texture tag is outside its range.")
        if value != field["TextureTag"] and value not in known_textures and value not in (0, 0xFFFFFFFF):
            raise ValueError("Choose a native texture from this bundle.")
        changed |= value != field["TextureTag"]
        field["TextureTag"] = value
        if node.image and node.image.get("sunrise_texture_tag") not in \
                (f'{value:08X}', f'{field["SourceTextureTag"]:08X}'):
            raise ValueError("Native slot image and texture tag disagree; choose a known bundle texture.")
    for node in field_nodes:
        field = fields[node["sunrise_constant_offset"]]
        value = float(node.outputs[0].default_value)
        if not math.isfinite(value):
            raise ValueError("Material constants must be finite.")
        changed |= struct.pack("<f", value) != struct.pack("<f", float(field["Value"]))
        field["Value"] = format(value, ".9g")
    if not changed:
        return None
    relative = str(Path(info["Path"]).parent /
                   f"material.edited.{transaction}.sunrise-material.json").replace("\\", "/")
    _write(root, relative, document)
    return {"TagHash": info["TagHash"], "Path": relative}


def export_bundle_edits(manifest_path=None):
    """Validate the complete bundle before publishing one edit manifest."""
    path, collection, document = _bundle(manifest_path)
    objects = _check_objects(collection)
    bpy.context.view_layer.update()
    root = path.parent
    transaction = uuid.uuid4().hex
    for text in bpy.data.texts:
        if text.get("sunrise_bundle_root") == str(root) and "sunrise_shader_tag" in text and \
                hashlib.sha256(text.as_string().encode()).hexdigest() != text["sunrise_source_sha256"]:
            raise ValueError("Shader text is read-only source metadata; Blender does not compile native DXBC.")
    by_id = {o.get("sunrise_mesh_id"): o for o in objects if o.get("sunrise_mesh_id")}
    edits = {"Format": "sunrise-asset-bundle-edits", "Version": 1,
             "SourceManifestSha256": collection["sunrise_manifest_sha256"],
             "Meshes": [], "Textures": [], "Animations": [], "Materials": [], "Emitters": []}
    all_meshes = []
    any_mesh_changed = False
    for info in document["Meshes"]:
        source = _path(root, info["Path"])
        if _sha(source) != info["Sha256"]:
            raise ValueError("Original geometry changed on disk.")
        if info["Id"] not in by_id:
            raise ValueError("A source mesh identity is missing.")
        original = _read(source)
        result = _mesh_edit(by_id[info["Id"]], info, original)
        any_mesh_changed |= result is not None
        all_meshes.append(result or {"Id": info["Id"], "Positions": original["Positions"],
            "Indices": original["Indices"], "Joints": original.get("Joints"),
            "Weights": original.get("Weights"), "Normals": None, "TexCoords": None})
    if any_mesh_changed:
        edits["Meshes"] = all_meshes
    loaded_materials = json.loads(collection["sunrise_loaded_materials"])
    known_textures = {value["TagHash"] for value in document.get("Textures", [])}
    for info in document.get("Materials", []):
        if info.get("Variant") or str(info["TagHash"]) not in loaded_materials:
            continue
        result = _material_edit(root, info, transaction, known_textures)
        if result:
            edits["Materials"].append(result)
    edits["Textures"] = _texture_edits(root, document, transaction)
    edits["Animations"] = _control_edits(root, collection, document, transaction)
    collision = _collision_edits(root, document, objects)
    if collision is not None:
        edits["Collision"] = collision
    if document.get("ParticleMetadataPath"):
        particles = _read(_path(root, document["ParticleMetadataPath"]))
        for emitter in particles["Emitters"]:
            obj = next(o for o in objects if o.get("sunrise_emitter_tag") == f'{emitter["TagHash"]:08X}')
            if not _close(_flat(obj.matrix_world), json.loads(obj["sunrise_preview_transform"])):
                raise ValueError("Emitter placement edits need the scene editor; bundle edits use native constants.")
            if {k for k in obj.keys() if k.startswith("native_")} != \
                    {f'native_{f["Channel"]}_{f["Offset"]:X}' for f in emitter["Constants"]}:
                raise ValueError("Native emitter fields were added or removed.")
            for field in emitter["Constants"]:
                key = f'native_{field["Channel"]}_{field["Offset"]:X}'
                value = list(obj.get(key, []))
                if len(value) != len(field["Value"]) or not all(math.isfinite(v) for v in value):
                    raise ValueError("Emitter constants need their original finite vector size.")
                if not _close(value, field["Value"]):
                    edits["Emitters"].append({"TagHash": emitter["TagHash"], "Channel": field["Channel"],
                                              "Offset": field["Offset"], "Value": value})
    return str(_write(root, EDITS, edits))


def import_material(tag, manifest_path=None):
    """The loaded material for tag; a material no part uses loads on demand."""
    path, collection, document = _bundle(manifest_path)
    tag = int(tag, 16) if isinstance(tag, str) else int(tag)
    loaded = json.loads(collection["sunrise_loaded_materials"])
    if str(tag) in loaded:
        matches = [m for m in bpy.data.materials if m.get("sunrise_id") == loaded[str(tag)]]
        if len(matches) != 1:
            raise ValueError("A loaded native material is missing or duplicated.")
        return matches[0]
    info = next((m for m in document.get("Materials", []) if m["TagHash"] == tag), None)
    if info is None:
        raise ValueError("Choose a material from this bundle.")
    images = {(int(i["sunrise_texture_tag"], 16), i.get("sunrise_mip", 0),
               i.get("sunrise_slice", 0), i.get("sunrise_depth", 0)): i
              for i in bpy.data.images if "sunrise_texture_tag" in i and
              Path(os.path.abspath(bpy.path.abspath(i.filepath))).is_relative_to(path.parent)}
    material = _material(path.parent, info, images, document)
    loaded[str(tag)] = _uid(material)
    collection["sunrise_loaded_materials"] = json.dumps(loaded)
    return material


def rebuild_materials(manifest_path=None):
    """Saved bundles can refresh preview links without resetting native field edits."""
    path, collection, document = _bundle(manifest_path)
    _apply_lod_visibility(collection, document, bpy.context.scene)
    for obj in collection.all_objects:
        if obj.type == 'ARMATURE':
            for bone in obj.pose.bones:
                shape = bone.custom_shape
                if shape and shape.name in collection.all_objects and not shape.get('sunrise_mesh_id'):
                    shape.hide_render = True
                    shape.hide_set(True)
    loaded = json.loads(collection['sunrise_loaded_materials'])
    count = 0
    for info in document.get('Materials', []):
        identity = loaded.get(_material_key(info['TagHash'], info.get('Variant')))
        if identity is None:
            continue
        matches = [material for material in bpy.data.materials if material.get('sunrise_id') == identity]
        if len(matches) != 1:
            raise ValueError('A native material is missing or duplicated.')
        material = matches[0]
        source = _path(path.parent, info['Path'])
        if _sha(source) != material['sunrise_material_sha256']:
            raise ValueError('The source material document changed.')
        fields = _read(source)
        slots = {slot['Offset']: slot for slot in fields['Slots']}
        for node in material.node_tree.nodes:
            if 'sunrise_slot_offset' in node:
                slot = slots.get(node['sunrise_slot_offset'])
                if slot is None:
                    raise ValueError('Native slot identity changed.')
                node['sunrise_preview_slot'] = slot['Slot']
                node['sunrise_preview_stage'] = slot['Stage']
        profile = info.get('Preview') or _legacy_material_preview(path.parent, info, fields, document)
        _build_material_preview(material, profile)
        count += 1
    _apply_stage_blends(collection, document)
    _material_shading()
    return count


def import_texture(tag, mip=0, slice_index=0, depth=0, manifest_path=None):
    path, _, document = _bundle(manifest_path)
    tag = int(tag, 16) if isinstance(tag, str) else int(tag)
    texture = next((t for t in document.get("Textures", []) if t["TagHash"] == tag), None)
    image = None if texture is None else next((i for i in texture["Images"]
        if (i["Mip"], i["Slice"], i["DepthSlice"]) == (mip, slice_index, depth)), None)
    if image is None:
        raise ValueError("That texture subresource has no exported image.")
    return _load_image(path.parent, image["Path"], tag, mip, slice_index, depth, image["Sha256"])


def _texture_edits(root, document, transaction):
    """Painted images are saved as new working copies before the edit manifest is published."""
    known = {(t["TagHash"], i["Mip"], i["Slice"], i["DepthSlice"]): i
             for t in document.get("Textures", []) for i in t["Images"]}
    changes = {}
    for image in bpy.data.images:
        if "sunrise_texture_tag" not in image:
            continue
        path = Path(os.path.abspath(bpy.path.abspath(image.filepath)))
        if not path.is_relative_to(root):
            continue
        _reject_links(path)
        key = (int(image["sunrise_texture_tag"], 16), image.get("sunrise_mip", 0),
               image.get("sunrise_slice", 0), image.get("sunrise_depth", 0))
        source = known.get(key)
        if source is None:
            continue
        edited = image.get("sunrise_edited_path")
        if not image.is_dirty and not edited and _sha(path) == source["Sha256"]:
            continue
        relative = f"edits/{transaction}/textures/{key[0]:08X}_{key[1]}_{key[2]}_{key[3]}.png"
        output = _path(root, relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        if image.is_dirty:
            previous, previous_format = image.filepath_raw, image.file_format
            try:
                image.filepath_raw = str(output)
                image.file_format = 'PNG'
                image.save()
            finally:
                image.filepath_raw, image.file_format = previous, previous_format
        else:
            saved = _path(root, edited) if edited else path
            if saved.suffix.lower() != ".png":
                raise ValueError("Edited bundle images must be saved as PNG.")
            shutil.copyfile(saved, output)
        digest = _sha(output)
        if key in changes and changes[key][0] != digest:
            raise ValueError("Images sharing a native texture have conflicting edits.")
        changes[key] = (digest, {"TagHash": key[0], "Path": relative,
                               "Mip": key[1], "Slice": key[2], "DepthSlice": key[3]})
        image["sunrise_edited_path"] = relative
    return [value[1] for value in changes.values()]


def _gltf_values(root, document, path, accessor_index):
    item = document["accessors"][accessor_index]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}.get(item["type"])
    if width is None or item["componentType"] != 5126 or "sparse" in item:
        raise ValueError("Native controls need contiguous float accessors.")
    view = document["bufferViews"][item["bufferView"]]
    buffer = document["buffers"][view.get("buffer", 0)]
    uri = buffer["uri"]
    if uri.startswith("data:application/octet-stream;base64,"):
        data = base64.b64decode(uri.split(",", 1)[1], validate=True)
    else:
        source = _path(path.parent, unquote(uri))
        source.relative_to(root)
        if source.stat().st_size > MAX_BYTES:
            raise ValueError("Animation data exceeds its size limit.")
        data = source.read_bytes()
    offset = view.get("byteOffset", 0) + item.get("byteOffset", 0)
    count = item["count"]
    if count < 1 or count * width * 4 > MAX_BYTES or len(data) > MAX_BYTES or \
            offset < 0 or offset + count * width * 4 > len(data) or view.get("byteStride", width * 4) != width * 4:
        raise ValueError("Animation accessor leaves its buffer.")
    values = struct.unpack_from("<" + "f" * count * width, data, offset)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Animation values must be finite.")
    return [list(values[i:i + width]) for i in range(0, len(values), width)]


def import_controls(clip_tag, manifest_path=None):
    """Native controls remain independent from the sampled bone preview."""
    path, collection, document = _bundle(manifest_path)
    _check_objects(collection)
    clip_tag = int(clip_tag, 16) if isinstance(clip_tag, str) else int(clip_tag)
    if any(o.get("sunrise_control_clip") == f"{clip_tag:08X}" for o in collection.all_objects):
        raise ValueError("These controls are already imported.")
    info = next((a for a in document["Animations"] if a["ClipTag"] == clip_tag), None)
    if info is None or not info.get("ControlsPath"):
        raise ValueError("This clip has no writable native controls.")
    source = _path(path.parent, info["ControlsPath"])
    if _sha(source) != info["ControlsSha256"]:
        raise ValueError("Original native controls changed on disk.")
    gltf = _validate_gltf(path.parent, source)
    metadata = gltf.get("extras", {}).get("sunriseAnimationControls")
    if not metadata or metadata["clipTag"] != f"{clip_tag:08X}" or len(gltf["animations"]) != 1:
        raise ValueError("The clip lacks an explicit native control mapping.")
    label = f'{info.get("Name") or "Clip"} [{clip_tag:08X}]'
    root = bpy.data.objects.new(f"Controls {label}", None)
    root["sunrise_control_root"] = f"{clip_tag:08X}"
    root["sunrise_control_source"] = info["ControlsPath"]
    root["sunrise_control_identity"] = json.dumps(metadata)
    root.empty_display_type = 'PLAIN_AXES'
    collection.objects.link(root)
    _uid(root)
    action = bpy.data.actions.new(f"Native controls {label}")
    action["sunrise_native_controls"] = f"{clip_tag:08X}"
    _uid(action)
    layer = action.layers.new("Native controls")
    strip = layer.strips.new(type='KEYFRAME')
    nodes = {}
    bags = {}
    for index, node in enumerate(gltf["nodes"]):
        if "sunriseControlIndex" not in node.get("extras", {}):
            continue
        obj = bpy.data.objects.new(node["name"], None)
        collection.objects.link(obj)
        obj.parent = root
        obj.empty_display_type = 'PLAIN_AXES'
        obj.empty_display_size = 0.15
        obj.rotation_mode = 'QUATERNION'
        obj.location = node.get("translation", [0, 0, 0])
        q = node.get("rotation", [0, 0, 0, 1])
        obj.rotation_quaternion = (q[3], *q[:3])
        obj.scale = node.get("scale", [1, 1, 1])
        obj["sunrise_control_index"] = node["extras"]["sunriseControlIndex"]
        obj["sunrise_control_clip"] = f"{clip_tag:08X}"
        obj["sunrise_control_node"] = index
        obj["sunrise_control_channels"] = "[]"
        obj["sunrise_control_defaults"] = json.dumps({"location": list(obj.location),
            "rotation_quaternion": list(obj.rotation_quaternion), "scale": list(obj.scale)})
        _uid(obj)
        slot = action.slots.new('OBJECT', obj.name)
        bag = strip.channelbags.new(slot)
        obj.animation_data_create()
        obj.animation_data.action = action
        obj.animation_data.action_slot = slot
        nodes[index] = obj
        bags[index] = bag
    animation = gltf["animations"][0]
    for channel in animation["channels"]:
        target = channel["target"]
        index, component = target["node"], target["path"]
        obj = nodes[index]
        sampler = animation["samplers"][channel["sampler"]]
        times = _gltf_values(path.parent, gltf, source, sampler["input"])
        values = _gltf_values(path.parent, gltf, source, sampler["output"])
        prop = {"translation": "location", "rotation": "rotation_quaternion", "scale": "scale"}[component]
        width = 4 if component == "rotation" else 3
        properties = json.loads(obj["sunrise_control_channels"])
        properties.append(prop)
        obj["sunrise_control_channels"] = json.dumps(properties)
        for axis in range(width):
            curve = bags[index].fcurves.new(data_path=prop, index=axis)
            curve.keyframe_points.add(len(times))
            original_axis = (axis + 3) % 4 if width == 4 else axis
            coordinates = [number for t, value in zip(times, values)
                           for number in (t[0] * 30, value[original_axis])]
            curve.keyframe_points.foreach_set("co", coordinates)
            for key in curve.keyframe_points:
                key.interpolation = 'CONSTANT' if sampler.get("interpolation") == "STEP" else 'LINEAR'
            curve.update()
    root["sunrise_control_action"] = action["sunrise_id"]
    root["sunrise_control_action_state"] = _action_state(action)
    root["sunrise_control_objects"] = json.dumps({i: _uid(o) for i, o in nodes.items()})
    collection["sunrise_objects"] = json.dumps([_uid(o) for o in collection.all_objects])
    bpy.context.scene.render.fps = 30
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max(0, metadata["frameCount"] - 1)
    bpy.context.scene.frame_set(0)
    return root


def _control_edits(root, collection, document, transaction):
    """Curve samples target the original native control indices at 30 Hz."""
    output = []
    for control_root in collection.all_objects:
        if "sunrise_control_root" not in control_root:
            continue
        if control_root.parent or not _close(_flat(control_root.matrix_local), _flat(Matrix.Identity(4))):
            raise ValueError("Keep native control coordinates unchanged.")
        tag = int(control_root["sunrise_control_root"], 16)
        source = _path(root, control_root["sunrise_control_source"])
        info = next(a for a in document["Animations"] if a["ClipTag"] == tag)
        if _sha(source) != info["ControlsSha256"]:
            raise ValueError("Original native controls changed on disk.")
        gltf = _read(source)
        actions = [a for a in bpy.data.actions if a.get("sunrise_id") == control_root["sunrise_control_action"]]
        if len(actions) != 1:
            raise ValueError("Native control action is missing or duplicated.")
        action = actions[0]
        if len(action.layers) != 1 or len(action.layers[0].strips) != 1:
            raise ValueError("Native controls need one action layer and strip.")
        changed = not _action_unchanged(action, control_root["sunrise_control_action_state"])
        metadata = json.loads(control_root["sunrise_control_identity"])
        buffers = bytearray()
        accessors, views, samplers, channels = [], [], [], []
        def add(values, width):
            view = len(views)
            offset = len(buffers)
            flat = [float(x) for value in values for x in value]
            if not all(math.isfinite(v) for v in flat):
                raise ValueError("Native controls must be finite.")
            buffers.extend(struct.pack("<" + "f" * len(flat), *flat))
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(buffers) - offset})
            index = len(accessors)
            accessors.append({"bufferView": view, "componentType": 5126, "count": len(values),
                              "type": {1: "SCALAR", 3: "VEC3", 4: "VEC4"}[width]})
            return index
        objects = {o["sunrise_control_node"]: o for o in control_root.children if "sunrise_control_node" in o}
        source_nodes = {i for i, node in enumerate(gltf["nodes"])
                        if "sunriseControlIndex" in node.get("extras", {})}
        if set(objects) != source_nodes or len(objects) != len(control_root.children):
            raise ValueError("Native control hierarchy changed.")
        if {str(i): o.get("sunrise_id") for i, o in objects.items()} != json.loads(control_root["sunrise_control_objects"]):
            raise ValueError("Native control object identities changed.")
        for node_index, obj in objects.items():
            if obj.get("sunrise_control_index") != gltf["nodes"][node_index]["extras"]["sunriseControlIndex"] or \
                    obj.get("sunrise_control_clip") != control_root["sunrise_control_root"]:
                raise ValueError("Native control indices changed.")
            if obj.animation_data is None or obj.animation_data.action != action or obj.constraints or \
                    obj.animation_data.nla_tracks or obj.animation_data.drivers or obj.rotation_mode != 'QUATERNION':
                raise ValueError("Keep native controls on their original action without constraints.")
            bag = action.layers[0].strips[0].channelbag(obj.animation_data.action_slot)
            expected = set(json.loads(obj["sunrise_control_channels"]))
            keys = {(prop, i) for prop in expected for i in range(4 if prop == 'rotation_quaternion' else 3)}
            if bag is None or len(bag.fcurves) != len(keys) or \
                    {(c.data_path, c.array_index) for c in bag.fcurves} != keys or any(c.modifiers for c in bag.fcurves):
                raise ValueError("Control channels or curve modifiers are unsupported.")
            if not changed:
                continue
            for prop in sorted(expected):
                width = 4 if prop == "rotation_quaternion" else 3
                curves = [next((c for c in bag.fcurves if c.data_path == prop and c.array_index == axis), None)
                          for axis in range(width)]
                if any(c is None for c in curves):
                    raise ValueError("A native control component was removed.")
                if any(k.co.x < 0 or k.co.x > metadata["frameCount"] - 1 for c in curves for k in c.keyframe_points):
                    raise ValueError("Control keys must stay within the original clip duration.")
                values = [[c.evaluate(frame) for c in curves] for frame in range(metadata["frameCount"])]
                if prop == "rotation_quaternion":
                    values = [[*value[1:], value[0]] for value in values]
                if prop == "scale" and any(abs(v[0] - v[1]) > 0.00001 or abs(v[0] - v[2]) > 0.00001 or v[0] == 0 for v in values):
                    raise ValueError("Native control scale must be uniform and nonzero.")
                sampler = len(samplers)
                samplers.append({"input": add([[f / 30] for f in range(metadata["frameCount"])], 1),
                                 "output": add(values, width), "interpolation": "LINEAR"})
                channels.append({"sampler": sampler, "target": {"node": node_index,
                    "path": {"location": "translation", "rotation_quaternion": "rotation", "scale": "scale"}[prop]}})
        if not changed:
            continue
        gltf["bufferViews"], gltf["accessors"] = views, accessors
        gltf["buffers"] = [{"byteLength": len(buffers), "uri": "data:application/octet-stream;base64," + base64.b64encode(buffers).decode()}]
        gltf["animations"] = [{"name": f"Native controls {tag:08X}", "samplers": samplers, "channels": channels}]
        relative = f"edits/{transaction}/animations/{tag:08X}.controls.gltf"
        _write(root, relative, gltf)
        output.append({"ClipTag": tag, "Path": relative})
    return output


class SUNRISE_OT_import_bundle(progress.ImportSteps, bpy.types.Operator, ImportHelper):
    bl_idname = "sunrise.import_bundle"
    bl_label = "Import Sunrise bundle"
    bl_description = "Open asset.sunrise-bundle.json from an exported bundle"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'}, maxlen=255)
    phases = BUNDLE_PHASES

    def invoke(self, context, event):
        active = context.scene.get("sunrise_active_bundle")
        if not self.filepath and active and Path(active).is_file():
            self.filepath = str(active)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def steps(self, context):
        return import_bundle_steps(self.filepath)

    def finished(self, context, collection):
        missing = collection.get('sunrise_materialless_parts', 0)
        if missing:
            self.report({'WARNING'}, f'{missing} mesh parts have no resolved material.')
        warnings = json.loads(collection.get('sunrise_material_warnings', '[]'))
        if warnings:
            self.report({'WARNING'}, f'Material fallback: {warnings[0]["Message"]}')
        return {'FINISHED'}


class SUNRISE_OT_export_bundle(bpy.types.Operator):
    bl_idname = "sunrise.export_bundle_edits"
    bl_label = "Export bundle edits"

    def execute(self, context):
        try:
            export_bundle_edits()
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, "Bundle edits saved; preview shading stays in Blender")
        return {'FINISHED'}


class SUNRISE_OT_import_controls(bpy.types.Operator):
    bl_idname = "sunrise.import_controls"
    bl_label = "Edit native controls"
    clip: StringProperty()

    def execute(self, context):
        try:
            import_controls(self.clip)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class SUNRISE_OT_load_material(bpy.types.Operator):
    bl_idname = "sunrise.load_bundle_material"
    bl_label = "Load native material"

    def execute(self, context):
        try:
            import_material(context.scene.sunrise_bundle_material)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class SUNRISE_OT_load_texture(bpy.types.Operator):
    bl_idname = "sunrise.load_bundle_texture"
    bl_label = "Load texture image"
    mip: IntProperty(name="Mip", min=0, default=0)
    slice_index: IntProperty(name="Slice", min=0, default=0)
    depth: IntProperty(name="Depth", min=0, default=0)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        try:
            image = import_texture(context.scene.sunrise_bundle_texture, self.mip, self.slice_index, self.depth)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        for area in context.screen.areas if context.screen else []:
            if area.type == 'IMAGE_EDITOR':
                area.spaces.active.image = image
        self.report({'INFO'}, image.name)
        return {'FINISHED'}


def _native_choices(self, context, surface):
    try:
        _, _, document = _bundle(verify=False)
    except (ValueError, OSError):
        return [('NONE', "No bundle", "")]
    cache_key = "_ui_" + surface
    if cache_key not in document:
        names = {a["TagHash"]: a["Name"] for a in document["NativeAssets"]}
        document[cache_key] = [(f'{value["TagHash"]:08X}', names.get(value["TagHash"], f'{value["TagHash"]:08X}'), "")
                               for value in document[surface]] or [('NONE', "None", "")]
    return document[cache_key]


def _material_fields(layout, material):
    layout.label(text=material.get('sunrise_preview_mode', 'Material preview'))
    layout.label(text="Preview links stay in Blender")
    for node in material.node_tree.nodes:
        if "sunrise_texture_tag" in node:
            layout.prop(node, '["sunrise_texture_tag"]', text=node.name)
        elif "sunrise_constant_offset" in node:
            layout.prop(node.outputs[0], "default_value", text=node.label)


def _pose_actions(collection):
    identities = set(json.loads(collection.get('sunrise_actions', '[]')))
    return [action for action in bpy.data.actions if action.get('sunrise_id') in identities]


def _native_frame_endpoint(value):
    """Native endpoints are integral at 30 Hz; float32 seconds and frames can round twice."""
    value = float(value)
    tolerance = max(.0001, math.ldexp(1.0, math.frexp(abs(value))[1] - 23))
    return round(value) if abs(value - round(value)) <= tolerance else value


def _prepare_pose_actions(collection, initialize_ranges=False):
    """Action and strip ranges come from stored key times, never a default timeline length."""
    rigs = [obj for obj in collection.all_objects if obj.type == 'ARMATURE']
    for action in _pose_actions(collection):
        start, end = action.curve_frame_range
        start = _native_frame_endpoint(start)
        end = _native_frame_endpoint(end)
        if initialize_ranges or 'sunrise_frame_start' not in action:
            action.use_frame_range = True
            action.frame_start = start
            action.frame_end = end
        action.use_fake_user = True
        action['sunrise_frame_start'] = start
        action['sunrise_frame_end'] = end
        action['sunrise_frames_per_second'] = 30
        targets = {}
        for rig in rigs:
            data = rig.animation_data
            if data is None:
                continue
            for track in data.nla_tracks:
                for strip in track.strips:
                    if strip.action != action:
                        continue
                    if strip.action_slot:
                        targets[_uid(rig)] = strip.action_slot.identifier
                    if initialize_ranges:
                        strip.action_frame_start = start
                        strip.action_frame_end = max(start + .001, end)
                        strip.scale = 1
                        strip.repeat = 1
                        strip.frame_start = start
                        strip.frame_end = max(start + .001, end)
                        strip.extrapolation = 'NOTHING'
            if _uid(rig) not in targets:
                matches = []
                for slot in action.slots:
                    paths = [curve.data_path for bag, curve in _curves(action)
                             if bag.slot_handle == slot.handle]
                    if not paths or not any(path.startswith('pose.bones[') for path in paths):
                        continue
                    try:
                        for path in paths:
                            rig.path_resolve(path)
                        matches.append(slot)
                    except (ValueError, KeyError):
                        continue
                if len(matches) == 1:
                    targets[_uid(rig)] = matches[0].identifier
        action['sunrise_pose_targets'] = json.dumps(targets)
    return rigs


def _bind_pose(collection):
    """Bind preview keeps Pose Position enabled and preserves the stored animation actions."""
    for rig in collection.all_objects:
        if rig.type != 'ARMATURE':
            continue
        rig.data.pose_position = 'POSE'
        if rig.animation_data:
            rig.animation_data.action = None
            for track in rig.animation_data.nla_tracks:
                track.mute = True
                track.is_solo = False
        for bone in rig.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
    collection['sunrise_active_pose'] = ''
    bpy.context.view_layer.update()


def repair_animation_ranges(manifest_path=None):
    _, collection, _ = _bundle(manifest_path)
    _prepare_pose_actions(collection, initialize_ranges=True)
    return len(_pose_actions(collection))


def preview_pose(clip=None, manifest_path=None, loop=None):
    """Loop playback excludes the terminal sample; transitions retain it."""
    _, collection, _ = _bundle(manifest_path)
    rigs = _prepare_pose_actions(collection)
    if clip is None or clip == 'BIND':
        _bind_pose(collection)
        return None
    token = f'{clip:08X}' if isinstance(clip, int) else str(clip)
    matches = [action for action in _pose_actions(collection) if token in
               (action.get('sunrise_id'), action.name, action.get('sunriseClipTag')) or 'Pose_' + token in action.name]
    if len(matches) != 1:
        raise ValueError('Choose one pose animation from this bundle.')
    action = matches[0]
    targets = json.loads(action.get('sunrise_pose_targets', '{}'))
    if not targets:
        raise ValueError('The pose action has no matching armature slot.')
    _bind_pose(collection)
    selected_rigs = []
    for rig in rigs:
        identifier = targets.get(_uid(rig))
        if identifier is None:
            continue
        slot = next((slot for slot in action.slots if slot.identifier == identifier), None)
        if slot is None:
            raise ValueError('The pose action slot is missing.')
        rig.animation_data_create()
        rig.animation_data.action = action
        rig.animation_data.action_slot = slot
        selected_rigs.append(rig)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.render.fps_base = 1
    start, end = float(action['sunrise_frame_start']), float(action['sunrise_frame_end'])
    loop = getattr(scene, 'sunrise_pose_loop', False) if loop is None else loop
    scene.frame_start = math.floor(start)
    scene.frame_end = max(scene.frame_start, math.ceil(end) - (1 if loop and end > start else 0))
    scene.use_preview_range = False
    scene.frame_set(scene.frame_start)
    action.use_cyclic = bool(loop)
    collection['sunrise_active_pose'] = _uid(action)
    if selected_rigs:
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for rig in selected_rigs:
            rig.select_set(True)
        bpy.context.view_layer.objects.active = selected_rigs[0]
    return action


def _pose_choices(self, context):
    try:
        _, collection, _ = _bundle(verify=False)
    except (ValueError, OSError):
        return [('BIND', 'Bind pose', '')]
    actions = _pose_actions(collection)
    values = [('BIND', 'Bind pose', '')] + [(_uid(action), action.name, '') for action in actions]
    # Blender needs Python to keep dynamic enum items alive.
    _ui_documents['_pose_choices'] = values
    return values


class SUNRISE_OT_preview_pose(bpy.types.Operator):
    bl_idname = 'sunrise.preview_pose'
    bl_label = 'Preview animation'
    bind_pose: BoolProperty(default=False)

    def execute(self, context):
        try:
            preview_pose(None if self.bind_pose else context.scene.sunrise_pose_action)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


def _lod_groups(document):
    groups = {}
    for info in document.get('Meshes', []):
        key = f'{info["ModelTag"]:08X}:{info.get("ConfigTag", 0):08X}'
        groups.setdefault(key, []).append(info)
    return groups


def _apply_lod_visibility(collection, document, scene):
    """Each model/configuration draws one level while every source object remains editable."""
    choices = json.loads(collection.get('sunrise_lod_choices', '{}'))
    objects = {obj.get('sunrise_mesh_id'): obj for obj in collection.all_objects
               if obj.get('sunrise_mesh_id')}
    selected = {}
    for key, rows in _lod_groups(document).items():
        levels = sorted({int(info.get('Lod', 0)) for info in rows})
        requested = choices.get(key)
        level = levels[0] if requested is None else min(levels, key=lambda value: (abs(value - requested), value))
        selected[key] = level
        for info in rows:
            obj = objects.get(info['Id'])
            if obj is None:
                continue
            obj['sunrise_lod'] = int(info.get('Lod', 0))
            obj['sunrise_lod_group'] = key
            obj['sunrise_lod_visible'] = obj['sunrise_lod'] == level
            obj['sunrise_render_stage'] = info.get('RenderStage', obj.get('sunrise_render_stage', -1))
            obj['sunrise_materialless'] = info['MaterialTag'] in (0, 0xFFFFFFFF)
            obj['sunrise_initially_visible'] = info.get('InitiallyVisible') is not False
            obj['sunrise_initial_visibility_reason'] = info.get('InitialVisibilityReason') or ''
            obj['sunrise_legacy_materialless'] = 'InitiallyVisible' not in info and obj['sunrise_materialless']
            profile = (_mesh_material(document, info) or {}).get('Preview')
            obj['sunrise_noncolor_pass'] = obj['sunrise_render_stage'] not in (-1, 0, 1, 2, 6, 7) or bool(
                profile is not None and (profile.get('Mode') == 'No pixel shader'
                                         or _stage_blend(profile, obj['sunrise_render_stage']) == 'NoColor'))
            _set_mesh_visibility(obj, scene)
    return selected


def repair_lod_visibility(manifest_path=None):
    _, collection, document = _bundle(manifest_path)
    return _apply_lod_visibility(collection, document, bpy.context.scene)


def set_lod(level=None, manifest_path=None, model_tag=None, config_tag=None):
    """Missing levels use the nearest available level, with higher detail winning a tie."""
    if level is not None and (not isinstance(level, int) or level < 0):
        raise ValueError('Choose a nonnegative LOD or Best.')
    _, collection, document = _bundle(manifest_path)
    choices = json.loads(collection.get('sunrise_lod_choices', '{}'))
    matched = False
    for key, rows in _lod_groups(document).items():
        info = rows[0]
        if model_tag is not None and info['ModelTag'] != model_tag or \
                config_tag is not None and info.get('ConfigTag', 0) != config_tag:
            continue
        matched = True
        if level is None:
            choices.pop(key, None)
        else:
            choices[key] = level
    if not matched:
        raise ValueError('The bundle has no matching model/configuration group.')
    collection['sunrise_lod_choices'] = json.dumps(choices)
    selected = _apply_lod_visibility(collection, document, bpy.context.scene)
    active = bpy.context.view_layer.objects.active
    if active and active.name in collection.all_objects and not active.get('sunrise_lod_visible', True):
        editing = active.mode == 'EDIT'
        if active.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        active.select_set(False)
        candidates = [obj for obj in collection.all_objects if obj.get('sunrise_lod_group') == active.get('sunrise_lod_group')
                      and not obj.hide_get() and obj.name in bpy.context.view_layer.objects]
        replacement = min(candidates, key=lambda obj: (obj.get('sunrise_materialless', False), obj.name)) if candidates else None
        bpy.context.view_layer.objects.active = replacement
        if replacement:
            replacement.select_set(True)
            if editing:
                bpy.ops.object.mode_set(mode='EDIT')
    return selected


class SUNRISE_OT_set_lod(bpy.types.Operator):
    bl_idname = 'sunrise.set_lod'
    bl_label = 'Show LOD'
    bl_description = 'Choose the visible detail level without removing source mesh parts'
    level: IntProperty(default=-1)
    group: StringProperty(default='')

    def execute(self, context):
        try:
            model, config = (int(value, 16) for value in self.group.split(':')) if self.group else (None, None)
            set_lod(None if self.level < 0 else self.level, model_tag=model, config_tag=config)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


def _draw_lod_choices(layout, collection, document):
    groups = _lod_groups(document)
    if not groups:
        return
    box = layout.box()
    box.label(text='Detail level')
    if any(info.get('InitiallyVisible') is False for info in document.get('Meshes', [])):
        box.prop(bpy.context.scene, 'sunrise_show_inactive_parts', text='Inactive parts')
    choices = json.loads(collection.get('sunrise_lod_choices', '{}'))
    names = {value['TagHash']: value['Name'] for value in document.get('NativeAssets', [])}
    for key, rows in sorted(groups.items()):
        levels = sorted({int(info.get('Lod', 0)) for info in rows})
        if len(groups) > 1:
            info = rows[0]
            label = names.get(info['ModelTag'], f'Model {info["ModelTag"]:08X}')
            if info.get('ConfigTag', 0):
                label += ' / ' + names.get(info['ConfigTag'], f'{info["ConfigTag"]:08X}')
            box.label(text=label)
        requested = choices.get(key)
        current = levels[0] if requested is None else min(levels, key=lambda value: (abs(value - requested), value))
        buttons = box.grid_flow(row_major=True, columns=0, even_columns=True, align=True)
        best = buttons.operator('sunrise.set_lod', text='Best', depress=requested is None)
        best.level = -1
        best.group = key
        for level in levels:
            option = buttons.operator('sunrise.set_lod', text=f'LOD {level}', depress=requested is not None and current == level)
            option.level = level
            option.group = key


def _decal_hidden(obj, scene):
    """Stage 1 decals that project onto the scene stay hidden; surface decals and a guardian's stage 1 parts show."""
    return obj.get('sunrise_render_stage') == 1 and not obj.get('sunrise_surface_decal', False) \
        and not getattr(scene, 'sunrise_show_decals', False)


def _set_mesh_visibility(obj, scene):
    """Initial state, detail level, and native pass visibility are independent filters."""
    hidden = not obj.get('sunrise_lod_visible', True) or obj.get('sunrise_noncolor_pass', False) or \
        _decal_hidden(obj, scene) or \
        (not obj.get('sunrise_initially_visible', True) and not getattr(scene, 'sunrise_show_inactive_parts', False))
    obj.hide_set(hidden)
    obj.hide_render = hidden or obj.get('sunrise_legacy_materialless', False)


def _show_decals(scene, context):
    """Visibility toggles retain all source objects and their editable data."""
    for obj in bpy.data.objects:
        if 'sunrise_lod_group' in obj:
            _set_mesh_visibility(obj, scene)


class SUNRISE_PT_bundle(bpy.types.Panel):
    bl_label = "Sunrise bundle"
    bl_idname = "SUNRISE_PT_bundle"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Sunrise"

    def draw(self, context):
        layout = self.layout
        layout.operator("sunrise.import_bundle", icon='IMPORT')
        if not context.scene.get("sunrise_active_bundle"):
            return
        layout.operator("sunrise.export_bundle_edits", icon='EXPORT')
        layout.operator('sunrise.rebuild_materials', icon='MATERIAL')
        layout.label(text="Fixed topology and skin")
        layout.label(text="Bone actions: preview only")
        try:
            _, collection, document = _bundle(verify=False)
        except (ValueError, OSError) as error:
            layout.label(text=str(error), icon='ERROR')
            return
        _draw_lod_choices(layout, collection, document)
        material_warnings = json.loads(collection.get('sunrise_material_warnings', '[]'))
        if material_warnings:
            layout.label(text=f'Material fallbacks: {len(material_warnings)}', icon='ERROR')
        if document.get('CollisionPath'):
            layout.label(text=f'Collision: {collection.get("sunrise_collision_count", 0)} shapes, wire, not rendered')
            layout.label(text='New objects in Collision become new shapes')
            obj = context.active_object
            if obj is not None and _in_collision_group(obj) and obj.get("sunrise_collision_id") is None:
                group = next(c for c in obj.users_collection if c.get("sunrise_collision"))
                materials = json.loads(group.get("sunrise_collision_materials", "{}"))
                compound = obj.get("sunrise_collision_compound")
                box = layout.box()
                box.label(text=f'Compound: {compound or "nearest"}')
                box.label(text=f'Kind: {obj.get("sunrise_collision_kind", "Mesh")}')
                box.label(text=f'Material: {obj.get("sunrise_collision_material") or "nearest triangle"}')
                listed = materials.get(compound) or (next(iter(materials.values())) if len(materials) == 1 else None)
                if listed:
                    box.label(text='Materials: ' + ", ".join(listed[:6]) + (" ..." if len(listed) > 6 else ""))
        if json.loads(collection.get('sunrise_actions', '[]')):
            pose = layout.box()
            pose.prop(context.scene, 'sunrise_pose_action', text='Pose')
            pose.prop(context.scene, 'sunrise_pose_loop', text='Loop range')
            row = pose.row(align=True)
            row.operator('sunrise.preview_pose', text='Preview', icon='ACTION').bind_pose = False
            row.operator('sunrise.preview_pose', text='Bind pose').bind_pose = True
        if any(info.get('RenderStage') == 1 for info in document.get('Meshes', [])):
            layout.prop(context.scene, 'sunrise_show_decals', text='Native decal volumes')
        if document.get("Materials"):
            box = layout.box()
            box.prop(context.scene, "sunrise_bundle_material", text="Material")
            box.operator("sunrise.load_bundle_material", icon='MATERIAL')
            loaded = json.loads(collection["sunrise_loaded_materials"])
            tag = context.scene.sunrise_bundle_material
            identity = loaded.get(str(int(tag, 16))) if tag != 'NONE' else None
            material = next((m for m in bpy.data.materials if m.get("sunrise_id") == identity), None) if identity else None
            if material is not None:
                _material_fields(box, material)
        if document.get("Textures"):
            box = layout.box()
            box.prop(context.scene, "sunrise_bundle_texture", text="Texture")
            box.operator("sunrise.load_bundle_texture", icon='IMAGE_DATA')
        for animation in document.get("Animations", []):
            if animation.get("ControlsPath"):
                op = layout.operator("sunrise.import_controls", text=animation["Name"], icon='ACTION')
                op.clip = f'{animation["ClipTag"]:08X}'
        obj = context.object
        if obj and "sunrise_emitter_tag" in obj:
            layout.label(text="Emitter constants")
            layout.label(text="GPU program is not simulated")
            for key in obj.keys():
                if key.startswith("native_"):
                    label = obj.id_properties_ui(key).as_dict().get("description", key)
                    layout.prop(obj, f'["{key}"]', text=label)


class SUNRISE_PT_bundle_material(bpy.types.Panel):
    bl_label = "Sunrise native fields"
    bl_idname = "SUNRISE_PT_bundle_material"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.material is not None and "sunrise_material_tag" in context.material

    def draw(self, context):
        layout = self.layout
        _material_fields(layout, context.material)
        layout.operator("sunrise.export_bundle_edits", icon='EXPORT')


class SUNRISE_OT_rebuild_materials(bpy.types.Operator):
    bl_idname = 'sunrise.rebuild_materials'
    bl_label = 'Rebuild material preview'
    bl_description = 'Refresh automatic shader links; native slot and constant edits are kept'

    def execute(self, context):
        try:
            count = rebuild_materials()
        except (ValueError, OSError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Rebuilt {count} materials')
        return {'FINISHED'}


CLASSES = (SUNRISE_OT_import_bundle, SUNRISE_OT_export_bundle, SUNRISE_OT_import_controls, SUNRISE_OT_rebuild_materials, SUNRISE_OT_set_lod, SUNRISE_OT_preview_pose,
           SUNRISE_OT_load_material, SUNRISE_OT_load_texture, SUNRISE_PT_bundle, SUNRISE_PT_bundle_material)


def _import_menu(self, context):
    self.layout.operator("sunrise.import_bundle", text="Sunrise bundle")


def _export_menu(self, context):
    self.layout.operator("sunrise.export_bundle_edits", text="Sunrise bundle edits")


def _remove_bundle_menus():
    """Reloading the add-on, or replacing an older copy, must not leave its previous menu callbacks."""
    for menu, name in ((bpy.types.TOPBAR_MT_file_import, "_import_menu"),
                       (bpy.types.TOPBAR_MT_file_export, "_export_menu")):
        for callback in list(getattr(menu.draw, "_draw_funcs", [])):
            scope = getattr(callback, "__globals__", {})
            if getattr(callback, "__name__", "") == name and scope.get("MANIFEST") == MANIFEST and \
                    any(getattr(cls, "bl_idname", None) == "sunrise.import_bundle"
                        for cls in scope.get("CLASSES", ())):
                menu.remove(callback)


def _registered_bundle_class(cls):
    """Operator RNA names follow bl_idname rather than the Python class name."""
    if issubclass(cls, bpy.types.Operator):
        category, name = cls.bl_idname.split(".", 1)
        return bpy.types.Operator.bl_rna_get_subclass_py(f"{category.upper()}_OT_{name}")
    return getattr(bpy.types, cls.__name__, None)


def register():
    _remove_bundle_menus()
    for cls in CLASSES:
        old = _registered_bundle_class(cls)
        if old is not None:
            bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)
    bpy.types.Scene.sunrise_bundle_material = EnumProperty(name="Material",
        items=lambda self, context: _native_choices(self, context, "Materials"))
    bpy.types.Scene.sunrise_bundle_texture = EnumProperty(name="Texture",
        items=lambda self, context: _native_choices(self, context, "Textures"))
    bpy.types.Scene.sunrise_show_decals = BoolProperty(name='Native decal volumes', default=False,
        description='Show preserved decal geometry; native depth compositing is unavailable', update=_show_decals)
    bpy.types.Scene.sunrise_show_inactive_parts = BoolProperty(name='Inactive parts', default=False,
        description='Inspect source parts disabled by supported initial state or with no active material', update=_show_decals)
    bpy.types.Scene.sunrise_pose_action = EnumProperty(name='Pose animation', items=_pose_choices)
    bpy.types.Scene.sunrise_pose_loop = BoolProperty(name='Loop range', default=False,
        description='Exclude the terminal sample from the looping timeline range')
    bpy.types.TOPBAR_MT_file_import.append(_import_menu)
    bpy.types.TOPBAR_MT_file_export.append(_export_menu)


def unregister():
    _remove_bundle_menus()
    for name in ("sunrise_bundle_material", "sunrise_bundle_texture", "sunrise_show_decals", "sunrise_show_inactive_parts",
                 'sunrise_pose_action', 'sunrise_pose_loop'):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
    for cls in reversed(CLASSES):
        current = _registered_bundle_class(cls)
        if current is not None:
            bpy.utils.unregister_class(current)
