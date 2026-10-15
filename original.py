"""Original assets: a glTF scene from Blender's exporter plus a sidecar for what glTF cannot carry."""

import json
import os
from pathlib import Path
import sys
import uuid

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper


FORMAT = "sunrise-original-asset"
SIDECAR = ".sunrise-original.json"


def _generator():
    info = getattr(sys.modules.get(__package__), "bl_info", {})
    return "Sunrise Studio " + ".".join(str(part) for part in info.get("version", ()))


def _objects(context):
    """Nothing selected exports the whole scene."""
    objects = list(context.scene.objects)
    selected = [obj for obj in objects if obj.select_get()]
    return (selected, True) if selected else (objects, False)


def _emitters(objects):
    """glTF has no particles, so the sidecar lists every system for Studio to refuse by name."""
    result = []
    for obj in objects:
        for system in getattr(obj, "particle_systems", ()):
            settings = system.settings
            result.append({"object": obj.name, "system": system.name, "type": settings.type,
                           "count": int(settings.count), "frameStart": float(settings.frame_start),
                           "frameEnd": float(settings.frame_end), "lifetime": float(settings.lifetime),
                           "emitFrom": settings.emit_from})
    return result


def _regular_output(path):
    """Exports replace regular files, not links or directories."""
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)() \
            or path.exists() and not path.is_file():
        raise ValueError("Export outputs must be regular files.")


def _publish_pair(staged, identifier):
    """Both outputs replace their prior files or restore them when publication fails."""
    backups, published = [], []
    try:
        for source, target in staged:
            _regular_output(target)
            if target.exists():
                backup = target.with_name(".sunrise-" + identifier + "-" + str(len(backups)) + ".bak")
                os.replace(target, backup)
                backups.append((target, backup))
        for source, target in staged:
            os.replace(source, target)
            published.append(target)
    except BaseException as error:
        failures = []
        restored = {target for target, backup in backups}
        for target in reversed(published):
            if target not in restored:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    failures.append(str(target))
        for target, backup in reversed(backups):
            try:
                os.replace(backup, target)
            except OSError:
                failures.append(str(backup))
        if failures:
            raise RuntimeError("Export rollback needs recovery: " + ", ".join(failures)) from error
        raise
    for target, backup in backups:
        backup.unlink(missing_ok=True)


def export_original(filepath, context=None, nla_actions_only=False):
    """Writes <name>.glb and <name>.sunrise-original.json; needs no imported Sunrise bundle.
    nla_actions_only keeps another armature's actions out when the scene holds two."""
    context = context or bpy.context
    path = Path(os.path.abspath(bpy.path.abspath(str(filepath))))
    if path.suffix.lower() != ".glb":
        path = path.with_name(path.name + ".glb")
    objects, selection = _objects(context)
    if not any(obj.type == 'MESH' for obj in objects):
        raise ValueError("Include a mesh object.")
    emitters = _emitters(objects)
    sidecar = path.with_name(path.stem + SIDECAR)
    _regular_output(path)
    _regular_output(sidecar)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"format": FORMAT, "version": 1, "scene": path.name, "name": path.stem,
                "generator": _generator(), "emitters": emitters}
    text = json.dumps(document, indent=2, allow_nan=False)
    identifier = uuid.uuid4().hex
    scene_staging = path.with_name(".sunrise-" + identifier + ".glb")
    sidecar_staging = sidecar.with_name(".sunrise-" + identifier + ".json")
    try:
        sidecar_staging.write_text(text, encoding="utf-8")
        # Y-up output; Studio converts it back to Z-up. Studio refuses Draco compression.
        result = bpy.ops.export_scene.gltf(filepath=str(scene_staging), export_format='GLB',
                                           use_selection=selection, use_active_scene=True,
                                           export_yup=True, export_texcoords=True, export_normals=True,
                                           export_materials='EXPORT', export_skins=True,
                                           export_animations=True, export_extras=True,
                                           export_morph=True, export_draco_mesh_compression_enable=False,
                                           export_anim_single_armature=not nla_actions_only)
        if 'FINISHED' not in result or not scene_staging.is_file():
            raise RuntimeError("Blender's glTF export did not finish.")
        _publish_pair(((scene_staging, path), (sidecar_staging, sidecar)), identifier)
    finally:
        scene_staging.unlink(missing_ok=True)
        sidecar_staging.unlink(missing_ok=True)
    return str(path)


class SUNRISE_OT_export_original_asset(bpy.types.Operator, ExportHelper):
    bl_idname = "sunrise.export_original_asset"
    bl_label = "Export Sunrise original asset"
    bl_description = "Export the selection, or the whole scene, as a GLB and a Sunrise sidecar"
    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.glb", options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        try:
            path = export_original(self.filepath, context)
        except (ValueError, OSError, RuntimeError, TypeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, path)
        return {'FINISHED'}


CLASSES = (SUNRISE_OT_export_original_asset,)


def _export_menu(self, context):
    self.layout.operator("sunrise.export_original_asset", text="Sunrise original asset")


def _remove_menu():
    """Reloading must not leave the previous module copy's menu callback."""
    for callback in list(getattr(bpy.types.TOPBAR_MT_file_export.draw, "_draw_funcs", [])):
        if getattr(callback, "__name__", "") == "_export_menu" \
                and getattr(callback, "__globals__", {}).get("SIDECAR") == SIDECAR:
            bpy.types.TOPBAR_MT_file_export.remove(callback)


def _registered(cls):
    """Operator RNA names follow bl_idname rather than the Python class name."""
    category, name = cls.bl_idname.split(".", 1)
    return bpy.types.Operator.bl_rna_get_subclass_py(f"{category.upper()}_OT_{name}")


def register():
    _remove_menu()
    for cls in CLASSES:
        old = _registered(cls)
        if old is not None:
            bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(_export_menu)


def unregister():
    _remove_menu()
    for cls in reversed(CLASSES):
        current = _registered(cls)
        if current is not None:
            bpy.utils.unregister_class(current)
