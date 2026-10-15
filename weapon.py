"""Exports weapon part assignments, controls, clips and particle settings beside the original scene."""

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import struct
import uuid

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import original


SIDECAR = ".sunrise-weapon.json"
# Studio reads the first-person arms from this file beside the weapon scene.
ARMS_SCENE = ".sunrise-arms.glb"
ARMS_PREFIX = "Arms "
PARTS = (
    ('NONE', "Unassigned", "Choose the assembly this mesh belongs to"),
    ('0', "Main assembly", "Main weapon geometry"),
    ('1', "Moving assembly", "The separate moving assembly"),
    ('21', "Detail assembly", "Separate detail geometry"),
    ('3', "Helper geometry", "The frame's auxiliary art part"),
)
EVENTS = (
    ('idle', "Idle", "Extra named clip to preview or bind in Studio"),
    ('fire', "Fire", "When the weapon fires"),
    ('reload', "Reload", "When the weapon reloads"),
    ('inspect', "Inspect", "Extra named clip to preview or bind in Studio"),
)
ACTION_EVENTS = EVENTS + (('CUSTOM', "Named clip", "Add a named clip without a default input binding"),)
# Muzzle and eject bursts play on every shot from the weapon's firing row, not from a clip.
PARTICLE_EVENTS = EVENTS + (
    ('muzzle', "Muzzle (every shot)", "Plays at the emitter on every shot"),
    ('eject', "Eject (every shot)", "Plays at the emitter on every shot, in place of a shell"),
)


def _scene_json(path):
    """The published part names must match the actual exported glTF nodes."""
    with path.open('rb') as stream:
        magic, version, length = struct.unpack('<III', stream.read(12))
        size, kind = struct.unpack('<II', stream.read(8))
        if magic != 0x46546C67 or version != 2 or kind != 0x4E4F534A or size > length - 20:
            raise ValueError("Blender did not write a valid GLB scene.")
        return json.loads(stream.read(size))


@contextmanager
def _staging(parent):
    """The private staging folder inherits the export folder's Windows access rules."""
    folder = parent / ('.sunrise-weapon-' + uuid.uuid4().hex)
    folder.mkdir()
    if folder.resolve().parent != parent.resolve():
        raise ValueError("Weapon staging folder leaves the export folder.")
    try:
        yield folder
    finally:
        for child in folder.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink()
        folder.rmdir()


@contextmanager
def _selection(context, chosen=None):
    """Exports see only the chosen objects, or the whole scene, never the user's last selection."""
    selected = list(context.selected_objects)
    for obj in selected:
        obj.select_set(False)
    try:
        for obj in chosen or ():
            obj.select_set(True)
        yield
    finally:
        for obj in chosen or ():
            obj.select_set(False)
        for obj in selected:
            obj.select_set(True)


def _arms_objects(objects):
    """The imported first-person arms rig and everything parented under it."""
    result = set()
    for obj in objects:
        if obj.sunrise_fp_arms:
            result.add(obj)
            result.update(obj.children_recursive)
    return result


def _arms_event(action):
    return action.sunrise_arms_event.strip()


def _fcurves(action):
    """Layered actions keep their curves in channel bags."""
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


def import_arms(filepath, context=None):
    """Imports the Studio arms rig and tags its actions with their weapon events."""
    context = context or bpy.context
    path = Path(os.path.abspath(bpy.path.abspath(str(filepath))))
    objects, actions = set(bpy.data.objects), set(bpy.data.actions)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if 'FINISHED' not in result:
        raise RuntimeError("Blender's glTF import did not finish.")
    added = [obj for obj in bpy.data.objects if obj not in objects]
    rigs = [obj for obj in added if obj.type == 'ARMATURE']
    if len(rigs) != 1:
        raise ValueError("The arms file must hold one armature.")
    for obj in added:
        obj.sunrise_fp_arms = True
    rig = rigs[0]
    rig.name = 'First-person arms'
    rig.show_in_front = True
    rig.animation_data_create()
    tracked = {strip.action for track in rig.animation_data.nla_tracks for strip in track.strips}
    for action in [value for value in bpy.data.actions if value not in actions]:
        if not action.name.startswith(ARMS_PREFIX):
            continue
        event = action.name[len(ARMS_PREFIX):]
        # The glTF importer may add the armature's node name after the animation name.
        for obj in added:
            if event.endswith('_' + obj.name):
                event = event[:-len(obj.name) - 1]
        action.name = ARMS_PREFIX + event
        action.sunrise_arms_event = event
        # glTF time zero imports on frame 0; weapon actions start on frame 1.
        shift = 1 - round(action.frame_range[0])
        for curve in _fcurves(action):
            for point in curve.keyframe_points:
                point.co.x += shift
                point.handle_left.x += shift
                point.handle_right.x += shift
        action.use_fake_user = True
        if action not in tracked:
            track = rig.animation_data.nla_tracks.new()
            track.name = action.name
            track.strips.new(action.name, int(action.frame_range[0]), action)
            track.mute = True
    return rig


def _particle_settings(objects):
    """Particle settings travel with their authored clip event."""
    values = []
    for obj in objects:
        for system in getattr(obj, 'particle_systems', ()):
            settings = system.settings
            values.append({
                "object": obj.name, "system": system.name,
                "event": settings.sunrise_weapon_event,
                "normalSpeed": float(settings.normal_factor),
                "randomSpeed": float(settings.factor_random),
                "particleSize": float(settings.particle_size),
                "sizeRandom": float(settings.size_random),
                "lifetimeRandom": float(settings.lifetime_random),
                "objectVelocity": list(settings.object_align_factor),
                "gravity": float(settings.effector_weights.gravity),
                "instanceObject": settings.instance_object.name if settings.instance_object else None,
            })
    return values


def _rig(objects):
    """Every exported armature bone needs an explicit control slot."""
    rigs = {obj for obj in objects if obj.type == 'ARMATURE'}
    for obj in objects:
        rigs.update(modifier.object for modifier in obj.modifiers
                    if modifier.type == 'ARMATURE' and modifier.object is not None)
    if not rigs:
        return None
    if len(rigs) != 1:
        raise ValueError("Export one weapon armature at a time.")
    rig = next(iter(rigs))
    bones = [{"bone": bone.name, "control": int(bone.sunrise_weapon_control)}
             for bone in rig.data.bones]
    if any(bone['control'] < 0 for bone in bones):
        raise ValueError("Assign every bone a control slot in the Sunrise Weapon panel.")
    if len(bones) != 5 or {bone['control'] for bone in bones} != set(range(5)):
        raise ValueError("The supported weapon frame needs five bones with different control slots from 0 to 4.")
    return {"object": rig.name, "bones": bones}


def export_weapon(filepath, context=None):
    """The GLB and both sidecars publish together only after all source assignments pass."""
    context = context or bpy.context
    path = Path(os.path.abspath(bpy.path.abspath(str(filepath))))
    if path.suffix.lower() != '.glb':
        path = path.with_name(path.name + '.glb')
    for parent in path.parents:
        if (parent / 'sunrise.project.json').exists():
            raise ValueError("Export working assets outside the Studio project, then import them.")
    scene_objects = list(context.scene.objects)
    arms = _arms_objects(scene_objects)
    objects = [obj for obj in scene_objects if obj not in arms]
    effects = _particle_settings(objects)
    emitters = {effect['object'] for effect in effects}
    emitters.update(effect['instanceObject'] for effect in effects if effect['instanceObject'])
    assignments = {}
    for obj in objects:
        if obj.type != 'MESH' or obj.name in emitters:
            continue
        part = obj.sunrise_weapon_part
        if part == 'NONE':
            raise ValueError("Assign an art part to " + obj.name + " in the Sunrise Weapon panel.")
        assignments[obj.name] = int(part)
    if set(assignments.values()) != {0, 1, 21, 3}:
        raise ValueError("The supported frame needs main, moving, detail and helper geometry.")
    rig = _rig(objects)
    units = float(context.scene.sunrise_weapon_metres_per_unit)
    if not math.isfinite(units) or units <= 0:
        raise ValueError("Metres per unit must be a positive number.")
    fps = float(context.scene.render.fps) / float(context.scene.render.fps_base)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("The scene frame rate must be positive.")
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = [path, path.with_name(path.stem + original.SIDECAR),
               path.with_name(path.stem + SIDECAR)]
    if arms:
        outputs.append(path.with_name(path.stem + ARMS_SCENE))
    for output in outputs:
        original._regular_output(output)
    identifier = uuid.uuid4().hex
    with _staging(path.parent) as staging:
        with _selection(context, objects if arms else None):
            exported = Path(original.export_original(staging / path.name, context, nla_actions_only=bool(arms)))
        document = _scene_json(exported)
        meshes = document.get('meshes', [])
        names = []
        for node in document.get('nodes', []):
            if 'mesh' not in node or node.get('name') in emitters:
                continue
            if not meshes[node['mesh']].get('primitives'):
                continue
            names.append(node.get('name', ''))
        if len(names) != len(set(names)) or set(names) != set(assignments):
            raise ValueError("Exported mesh names differ from the assigned objects.")
        actions = {action.name: action for action in bpy.data.actions if not _arms_event(action)}
        clips = []
        for animation in document.get('animations', []):
            name = animation.get('name', '')
            action = actions.get(name)
            if action is None:
                candidates = [value for key, value in actions.items() if name.endswith('|' + key)]
                action = candidates[0] if len(candidates) == 1 else None
            if action is None:
                raise ValueError("Name each exported action and assign its event: " + name)
            event = action.sunrise_weapon_event
            if event == 'CUSTOM':
                event = action.sunrise_weapon_clip_name.strip() or action.name
                if event in {value[0] for value in EVENTS}:
                    raise ValueError("Use the matching event instead of a reserved named clip: " + event)
            clips.append({"action": name, "event": event,
                          "sourceStartFrame": float(action.frame_range[0]),
                          "sourceEndFrame": float(action.frame_range[1])})
        if len({clip['event'] for clip in clips}) != len(clips):
            raise ValueError("Assign a different event to each exported action.")
        sidecar = {
            "format": "sunrise-weapon-source", "version": 1, "scene": path.name,
            "forward": "+Y", "up": "+Z", "metresPerUnit": units,
            "framesPerSecond": fps,
            "parts": [{"selector": selector,
                       "meshes": sorted(name for name in names if assignments[name] == selector)}
                      for selector in (0, 1, 21, 3)],
            "clips": clips, "effects": effects,
        }
        if rig is not None:
            sidecar['rig'] = rig
        sources = [exported, staging / (path.stem + original.SIDECAR)]
        if arms:
            with _selection(context, list(arms)):
                arms_path = Path(original.export_original(staging / (path.stem + ARMS_SCENE), context,
                                                          nla_actions_only=True))
            named = {action.name: action for action in bpy.data.actions if _arms_event(action)}
            arms_clips = []
            for animation in _scene_json(arms_path).get('animations', []):
                name = animation.get('name', '')
                action = named.get(name)
                if action is None:
                    raise ValueError("Give each arms action a weapon event: " + name)
                arms_clips.append({"action": name, "event": _arms_event(action)})
            events = {clip['event'] for clip in clips}
            if not arms_clips or len({clip['event'] for clip in arms_clips}) != len(arms_clips) \
                    or any(clip['event'] not in events for clip in arms_clips):
                raise ValueError("Each arms action needs a weapon action with the same event.")
            sidecar['arms'] = {"scene": path.stem + ARMS_SCENE, "clips": arms_clips}
        written = staging / (path.stem + SIDECAR)
        written.write_text(json.dumps(sidecar, indent=2, allow_nan=False), encoding='utf-8')
        sources.append(written)
        if arms:
            sources.append(arms_path)
        original._publish_pair(tuple(zip(sources, outputs)), identifier)
    return str(path)


class SUNRISE_OT_export_weapon(bpy.types.Operator, ExportHelper):
    bl_idname = 'sunrise.export_weapon'
    bl_label = 'Export Sunrise weapon'
    bl_description = 'Export the weapon with its part, control, animation and particle assignments'
    filename_ext = '.glb'
    filter_glob: StringProperty(default='*.glb', options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        try:
            path = export_weapon(self.filepath, context)
        except (ValueError, OSError, RuntimeError, TypeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, path)
        return {'FINISHED'}


class SUNRISE_OT_import_fp_arms(bpy.types.Operator, ImportHelper):
    bl_idname = 'sunrise.import_fp_arms'
    bl_label = 'Import first-person arms'
    bl_description = 'Import the arms rig that Studio exports from a weapon page'
    filename_ext = '.glb'
    filter_glob: StringProperty(default='*.glb', options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        try:
            rig = import_arms(self.filepath, context)
        except (ValueError, OSError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, rig.name)
        return {'FINISHED'}


class SUNRISE_OT_preview_weapon_event(bpy.types.Operator):
    bl_idname = 'sunrise.preview_weapon_event'
    bl_label = 'Preview event'
    bl_description = 'Play the weapon action and the arms action of one event together'
    event: StringProperty()

    def execute(self, context):
        for obj in context.scene.objects:
            if obj.type != 'ARMATURE' or obj.animation_data is None:
                continue
            for action in bpy.data.actions:
                arms = _arms_event(action)
                if (obj.sunrise_fp_arms and arms == self.event) or \
                        (not obj.sunrise_fp_arms and not arms and action.sunrise_weapon_event == self.event):
                    obj.animation_data.action = action
                    context.scene.frame_start, context.scene.frame_end = (int(value) for value in action.frame_range)
        return {'FINISHED'}


class SUNRISE_PT_weapon(bpy.types.Panel):
    bl_label = 'Weapon'
    bl_idname = 'SUNRISE_PT_weapon'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Sunrise'

    def draw(self, context):
        layout = self.layout
        layout.label(text='1. Assign the selected mesh to an art part.')
        obj = context.object
        if obj is not None and obj.type == 'MESH':
            if obj.particle_systems:
                layout.label(text='Particle emitter: no art part needed.')
            else:
                layout.prop(obj, 'sunrise_weapon_part', text='Art part')
            for system in obj.particle_systems:
                layout.prop(system.settings, 'sunrise_weapon_event', text=system.name)
        layout.label(text='2. Assign bone controls and action events.')
        if obj is not None and obj.type == 'ARMATURE':
            layout.label(text='Use five bones, with slots 0 through 4.')
            layout.label(text='Weight each vertex to one bone.')
            for bone in obj.data.bones:
                layout.prop(bone, 'sunrise_weapon_control', text=bone.name)
        for action in bpy.data.actions:
            if _arms_event(action):
                continue
            layout.prop(action, 'sunrise_weapon_event', text=action.name)
            if action.sunrise_weapon_event == 'CUSTOM':
                layout.prop(action, 'sunrise_weapon_clip_name', text='Clip name')
        layout.label(text='Optional: first-person arms from Studio.')
        layout.operator('sunrise.import_fp_arms', text='Import first-person arms')
        for action in bpy.data.actions:
            if _arms_event(action):
                layout.prop(action, 'sunrise_arms_event', text=action.name)
                layout.operator('sunrise.preview_weapon_event', text='Preview ' + _arms_event(action)).event = \
                    _arms_event(action)
        layout.label(text='3. Export, then choose the GLB in Studio.')
        layout.prop(context.scene, 'sunrise_weapon_metres_per_unit', text='Metres per unit')
        layout.operator('sunrise.export_weapon', text='Export weapon')


CLASSES = (SUNRISE_OT_export_weapon, SUNRISE_OT_import_fp_arms, SUNRISE_OT_preview_weapon_event, SUNRISE_PT_weapon)


def _export_menu(self, context):
    self.layout.operator('sunrise.export_weapon', text='Sunrise weapon')


def _remove_menu():
    """Reloading removes only this module's earlier export callback."""
    for callback in list(getattr(bpy.types.TOPBAR_MT_file_export.draw, '_draw_funcs', [])):
        if getattr(callback, '__name__', '') == '_export_menu' \
                and getattr(callback, '__globals__', {}).get('SIDECAR') == SIDECAR:
            bpy.types.TOPBAR_MT_file_export.remove(callback)


def register():
    _remove_menu()
    bpy.types.Object.sunrise_weapon_part = EnumProperty(name='Art part', items=PARTS, default='NONE')
    bpy.types.Bone.sunrise_weapon_control = IntProperty(name='Control slot', default=-1, min=-1, max=4,
                                                      description='Unique slot from 0 to 4 in the supported weapon frame')
    bpy.types.Action.sunrise_weapon_event = EnumProperty(name='Weapon event', items=ACTION_EVENTS)
    bpy.types.Action.sunrise_weapon_clip_name = StringProperty(name='Clip name',
                                                              description='Blank uses the action name')
    bpy.types.Action.sunrise_arms_event = StringProperty(name='Arms event',
                                                         description='Weapon event whose first-person arms this action replaces')
    bpy.types.Object.sunrise_fp_arms = BoolProperty(name='First-person arms', default=False)
    bpy.types.ParticleSettings.sunrise_weapon_event = EnumProperty(name='Weapon event', items=PARTICLE_EVENTS)
    bpy.types.Scene.sunrise_weapon_metres_per_unit = FloatProperty(name='Metres per unit', default=1.0,
                                                                  min=0.0001, max=1000.0)
    for cls in CLASSES:
        previous = getattr(bpy.types, cls.__name__, None)
        if previous is not None:
            bpy.utils.unregister_class(previous)
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(_export_menu)


def unregister():
    _remove_menu()
    for cls in reversed(CLASSES):
        current = getattr(bpy.types, cls.__name__, None)
        if current is not None:
            bpy.utils.unregister_class(current)
    for owner, name in ((bpy.types.Object, 'sunrise_weapon_part'),
                        (bpy.types.Bone, 'sunrise_weapon_control'),
                        (bpy.types.Action, 'sunrise_weapon_event'),
                        (bpy.types.Action, 'sunrise_weapon_clip_name'),
                        (bpy.types.Action, 'sunrise_arms_event'),
                        (bpy.types.Object, 'sunrise_fp_arms'),
                        (bpy.types.ParticleSettings, 'sunrise_weapon_event'),
                        (bpy.types.Scene, 'sunrise_weapon_metres_per_unit')):
        if hasattr(owner, name):
            delattr(owner, name)
