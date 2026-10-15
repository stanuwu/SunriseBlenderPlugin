"""Material folders keep native slot and constant identities; node wiring is a Blender preview."""

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper


MANIFEST = "material.sunrise-material.json"
EDITS = "material.edited.sunrise-material.json"


def _bundle_path(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _write_edits(output, document):
    """A failed save leaves the prior edits intact and never follows an output link."""
    if output.is_symlink() or getattr(output, "is_junction", lambda: False)() \
            or output.exists() and not output.is_file():
        raise ValueError("Material edits need a regular output file.")
    text = json.dumps(document, indent=2, allow_nan=False)
    temporary = output.with_name(".sunrise-material-" + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def import_material(manifest_path, target=None):
    """Adds the material to target, or to the active object, and selects its slot."""
    manifest = Path(manifest_path).resolve()
    root = manifest.parent
    document = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if document.get("Format") != "sunrise-material" or document.get("Version") != 1:
        raise ValueError("Unsupported Sunrise material bundle.")
    material = bpy.data.materials.new(document["Name"])
    material["sunrise_document"] = json.dumps(document)
    material["sunrise_root"] = str(root)
    material["sunrise_tag"] = f'{document["TagHash"]:08X}'
    nodes = material.node_tree.nodes
    files = {item["TagHash"]: item for item in document["Textures"]}
    for index, slot in enumerate(document["Slots"]):
        node = nodes.new("ShaderNodeTexImage")
        node.name = f'{slot["Stage"]}_slot_{slot["Slot"]}'
        node.label = f'{slot["Stage"]} slot {slot["Slot"]} / {slot["TextureTag"]:08X}'
        node.location = (-650, -index * 260)
        node["sunrise_slot_offset"] = slot["Offset"]
        node["sunrise_texture_tag"] = f'{slot["TextureTag"]:08X}'
        file = files.get(slot["SourceTextureTag"])
        if file:
            relative = file.get("PreviewPath") or file.get("DdsPath")
            if relative:
                try:
                    node.image = bpy.data.images.load(str(_bundle_path(root, relative)), check_existing=True)
                except RuntimeError as error:
                    node["sunrise_image_error"] = str(error)
    for index, scalar in enumerate(document["Constants"]):
        value = float(scalar["Value"])
        if not math.isfinite(value):
            continue
        node = nodes.new("ShaderNodeValue")
        node.name = f'constant_{scalar["Offset"]:X}'
        node.label = f'{scalar["Stage"]} {scalar["Bank"]} {scalar["Vector"]}.{scalar["Component"]}'
        node.location = (-300 + (index % 4) * 180, -350 - (index // 4) * 130)
        node.outputs[0].default_value = value
        node["sunrise_constant_offset"] = scalar["Offset"]
    selected = target if target is not None else getattr(bpy.context, "object", None)
    if selected is not None and hasattr(selected.data, "materials"):
        selected.data.materials.append(material)
        selected.active_material_index = len(selected.data.materials) - 1
    return material


def export_material_edits(material=None):
    if material is None:
        selected = getattr(bpy.context, "object", None)
        material = getattr(selected, "active_material", None)
    if material is None or "sunrise_document" not in material:
        raise ValueError("Select a Sunrise material.")
    root = Path(material["sunrise_root"]).resolve()
    document = json.loads(material["sunrise_document"])
    slots = {item["Offset"]: item for item in document["Slots"]}
    constants = {item["Offset"]: item for item in document["Constants"]}
    files = {item["TagHash"]: item for item in document["Textures"]}
    edited_images = {}
    for node in material.node_tree.nodes:
        if "sunrise_constant_offset" in node:
            field = constants[node["sunrise_constant_offset"]]
            value = float(node.outputs[0].default_value)
            if not math.isfinite(value):
                raise ValueError("Material constants must be finite.")
            field["Value"] = format(value, ".9g")
        if "sunrise_slot_offset" in node:
            slot = slots[node["sunrise_slot_offset"]]
            slot["TextureTag"] = int(str(node["sunrise_texture_tag"]), 16)
            file = files.get(slot["SourceTextureTag"])
            if file and node.image is not None and slot["TextureTag"] == slot["SourceTextureTag"]:
                image_path = Path(bpy.path.abspath(node.image.filepath)).resolve()
                relative = image_path.relative_to(root).as_posix()
                previous = edited_images.setdefault(file["TagHash"], relative)
                if previous != relative:
                    raise ValueError("Slots sharing a native texture must use the same edited image.")
                if image_path.suffix.lower() not in (".png", ".dds"):
                    raise ValueError("Save texture edits as PNG or DDS inside the material folder.")
                if node.image.is_dirty:
                    node.image.save()
                original_hash = file.get("PreviewSha256") if relative == file.get("PreviewPath") else file.get("BaseSha256")
                current_hash = hashlib.sha256(image_path.read_bytes()).hexdigest().upper()
                if file.get("EditedPath") or current_hash != original_hash:
                    file["EditedPath"] = relative
    output = root / EDITS
    _write_edits(output, document)
    material["sunrise_document"] = json.dumps(document)
    return str(output)


class SUNRISE_OT_import_material(bpy.types.Operator, ImportHelper):
    bl_idname = "sunrise.import_material"
    bl_label = "Import Sunrise material"
    bl_description = "Add material.sunrise-material.json to the active mesh"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        try:
            import_material(self.filepath, context.object)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class SUNRISE_OT_export_material(bpy.types.Operator):
    bl_idname = "sunrise.export_material_edits"
    bl_label = "Export Sunrise edits"

    def execute(self, context):
        try:
            path = export_material_edits(getattr(context, "material", None))
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, path)
        return {'FINISHED'}


class SUNRISE_PT_material(bpy.types.Panel):
    bl_label = "Sunrise material"
    bl_idname = "SUNRISE_PT_material"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        material = getattr(context, "material", None)
        return material is not None and "sunrise_document" in material

    def draw(self, context):
        self.layout.label(text="Native slots; wire roles explicitly")
        for node in context.material.node_tree.nodes:
            if "sunrise_texture_tag" in node:
                self.layout.prop(node, '["sunrise_texture_tag"]', text=node.name)
        self.layout.operator("sunrise.export_material_edits")


CLASSES = (SUNRISE_OT_import_material, SUNRISE_OT_export_material, SUNRISE_PT_material)


def _import_menu(self, context):
    self.layout.operator("sunrise.import_material", text="Sunrise material")


def _remove_menu():
    """Reloading must not leave the previous module copy's menu callback."""
    for callback in list(getattr(bpy.types.TOPBAR_MT_file_import.draw, "_draw_funcs", [])):
        if getattr(callback, "__name__", "") == "_import_menu" \
                and getattr(callback, "__globals__", {}).get("MANIFEST") == MANIFEST:
            bpy.types.TOPBAR_MT_file_import.remove(callback)


def _registered(cls):
    """Operator RNA names follow bl_idname rather than the Python class name."""
    if issubclass(cls, bpy.types.Operator):
        category, name = cls.bl_idname.split(".", 1)
        return bpy.types.Operator.bl_rna_get_subclass_py(f"{category.upper()}_OT_{name}")
    return getattr(bpy.types, cls.__name__, None)


def register():
    _remove_menu()
    for cls in CLASSES:
        old = _registered(cls)
        if old is not None:
            bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(_import_menu)


def unregister():
    _remove_menu()
    for cls in reversed(CLASSES):
        current = _registered(cls)
        if current is not None:
            bpy.utils.unregister_class(current)
