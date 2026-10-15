# Sunrise Blender Add-on

Blender Add-on for Sunrise Studio

> Add-on for [Sunrise Studio](https://github.com/stanuwu/SunriseStudio) that handles imports, exports and edits of Sunrise Studio generated assets.

- [Sunrise Studio](https://github.com/stanuwu/SunriseStudio)
- [Sunrise](https://github.com/stanuwu/Sunrise)
- [Documentation](https://projectsunrise.dev/docs/sunrise-studio/blender-round-trip/)
- [Discord](https://discord.gg/22JS6et5k9)

## Requirements

- Blender 5.2 or newer
- [Sunrise Studio](https://github.com/stanuwu/SunriseStudio) for the exports and imports

## Install

1. Get `sunrise_blender_plugin-<version>.zip` from releases, or build it (see Building).
2. In Blender, open **Edit > Preferences > Add-ons**, choose **Install from Disk...** in the menu at the
   top right, and pick the zip.
3. Enable **Sunrise Studio**.

## Building

1. Clone the repository
```powershell
git clone https://github.com/stanuwu/SunriseBlenderPlugin
cd SunriseBlenderPlugin
```

2. Build the zip with Blender's extension builder
```powershell
blender --command extension build --source-dir . --output-dir build
```

This writes `build/sunrise_blender_plugin-<version>.zip` from `blender_manifest.toml`.

## Usage

| Task | Where |
|---|---|
| Open a map export | File > Import > Sunrise world scene |
| Open an asset bundle | File > Import > Sunrise bundle |
| Export bundle edits | 3D View sidebar > Sunrise > Export bundle edits |
| Open a material | File > Import > Sunrise material |
| Export material edits | Material Properties > Export Sunrise edits |
| Export your own asset | File > Export > Sunrise original asset |
| Author a weapon | 3D View sidebar > Sunrise > Weapon |

Keep working folders outside Sunrise projects and the game installation.

The guides have the full steps and limits:

- [Edit an asset in Blender](https://projectsunrise.dev/docs/sunrise-studio/blender-round-trip/)
- [Export a map to Blender](https://projectsunrise.dev/docs/sunrise-studio/export-world/)
- [Edit materials](https://projectsunrise.dev/docs/sunrise-studio/materials/)
- [Edit collision](https://projectsunrise.dev/docs/sunrise-studio/collision/)
- [Edit animation controls](https://projectsunrise.dev/docs/sunrise-studio/animation/)
- [Import an original Blender asset](https://projectsunrise.dev/docs/sunrise-studio/original-asset/)
- [Create an exotic weapon](https://projectsunrise.dev/docs/sunrise-studio/exotic-weapon/)

## Contributing

Pull Requests are welcome. Keep changes small and readable, and test them with a Sunrise Studio
export and import. Do not put multiple features into one PR.

## License

GPL-3.0-or-later. The full text is in [LICENSE](LICENSE).

## Disclaimers

This project is not affiliated with Bungie, Sony or the Blender Foundation in any way.

AI was used in the creation of this project. All AI work that is publicly released is reviewed by a
human.
