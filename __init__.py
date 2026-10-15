"""Imports Sunrise world scenes, asset bundles and materials, and exports supported asset edits."""

bl_info = {"name": "Sunrise Studio", "blender": (5, 2, 0),
           "category": "Import-Export", "version": (1, 13, 19),
           "description": "Sunrise world scenes, asset bundles, materials and original asset export"}

# Blender re-runs this file when the add-on is re-enabled; reload keeps the submodules current.
# A submodule that an older copy of this file never imported is imported fresh below.
if "bundle" in globals():
    import importlib
    for _name in ("bundle", "material", "original", "world_lights", "effects", "world", "weapon"):
        if _name in globals():
            importlib.reload(globals()[_name])

from . import bundle, material, original, world_lights, effects, world, weapon


def register():
    bundle.register()
    material.register()
    original.register()
    world.register()
    weapon.register()


def unregister():
    weapon.unregister()
    world.unregister()
    original.unregister()
    material.unregister()
    bundle.unregister()
