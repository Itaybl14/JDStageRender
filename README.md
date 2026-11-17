# JD Stage Render

> Built with [Claude Code](https://claude.com/claude-code): the code and docs in this repository were written by Claude (Anthropic).

Renders a Just Dance mashup map's **mainsequence** over the game's generated
**Mashup stage**: the generic stage a song gets when its songdesc `backgroundType`
is Mashup (6). You get a video (or PNG sequence) with the stage, the coach video
composited on top, and the song audio.

https://github.com/user-attachments/assets/REPLACE-WITH-UPLOADED-SHOWCASE

*30-second sample ([docs/showcase.mp4](docs/showcase.mp4)): Give Me Everything (Mashup), 720p.
Full showcase: https://youtu.be/Y-I2EZa-z88*

The stage is not a recording. It is rebuilt from the game's own data: the stage scene,
actor templates, material shaders, textures, meshes, particle FX and tapes. The engine's
behaviour (tape clocks, clip players, binds, the multi-texture shader, the 25° camera) is
re-implemented in Python.

Only the Mashup stage is supported. Its output has been checked against in-game footage.

## Requirements

- Windows, with Python 3.10 or newer (developed on 3.12)
- `pip install -r requirements.txt` (numpy, opencv-python, texture2ddecoder)
- Optional: `pip install moderngl` for the much faster GPU renderer. Without it,
  the CPU rasterizer is used.
- **ffmpeg and ffprobe** (not included in the repository, they're too big for GitHub):
  download a Windows build (e.g. from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or
  [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases)) and put `ffmpeg.exe` and
  `ffprobe.exe` in a `bin/` folder next to `stage_render.py`, or anywhere on your `PATH`.
  NVIDIA NVENC is used automatically when it's available.

## Quick start (GUI)

Double-click **`JD Stage Render.bat`**.

1. Pick the map's `<map>/cinematics/<map>_mainsequence.tape.ckd`.
2. Choose the resolution, FPS and quality, and optionally a starting mashup colour.
   Tick **Song preview only** to render just the 30-second song preview.
3. Click **Preview frame** to check a single frame.
4. Click **Render**. The output can be `.mp4`, `.mkv`, `.webm` or a PNG sequence.

The coach video, audio and musictrack are found automatically inside the map folder.

## Command line

```
python stage_render.py <map>/cinematics/<map>_mainsequence.tape.ckd
python stage_render.py seq.tape.ckd --mashup-color purple -o out.mp4
python stage_render.py seq.tape.ckd --start 30 --duration 10 --size 1280x720
python stage_render.py seq.tape.ckd --still 10 --still 30        # PNG previews only
python stage_render.py seq.tape.ckd -o frames/stage.png          # PNG sequence
python stage_render.py seq.tape.ckd --preview --fps 25           # 30 s song preview
python stage_render.py seq.tape.ckd --info                       # show the plan, render nothing
```

Useful options (`--help` lists them all):

| Option | What it does |
|--------|--------------|
| `-o` | Output file. `.mp4`/`.mkv`/`.webm`, or `.png` for a sequence (see below). Default: `<map>_mashup.mp4` |
| `--mashup-color` | Force the starting colour set (`blue`, `orange`, `green`, `purple`). By default the mainsequence drives it |
| `--start`, `--duration` | Render only part of the song (seconds) |
| `--preview` | Render the song preview: starts at the musictrack's `previewLoopStart`, 30 s unless `--duration` is given. Default output `<map>_preview.mp4`. In the GUI, tick "Song preview only" |
| `--size`, `--fps` | Output resolution (default 1920x1080) and frame rate (default 30) |
| `--coach`, `--no-coach`, `--coach-mode` | Override or disable the coach video, or choose how its alpha is read |
| `--audio`, `--no-audio` | Override or disable the muxed audio |
| `--renderer auto\|gpu\|cpu` | `auto` uses the GPU when moderngl works |
| `--encoder auto\|nvenc\|x264`, `--crf`, `--preset` | Video encoding |
| `--hide ACTOR` | Hide a stage actor by name (repeatable) |
| `--bundle file.ipk` | Read the stage assets from a game bundle instead of `data/` |
| `--export-assets file.ipk` | Rebuild the local `data/` folder from a game bundle |

### PNG sequence

With `-o out/stage.png` you get `out/stage_000001.png`, `stage_000002.png`, …
(frames only, no audio). For custom numbering, put a pattern in the name:
`-o out/frame_%04d.png`. Frames are opaque RGB.

## Mainsequence editor (mashup maps)

**`JD Mainsequence Editor.bat`** builds a mashup mainsequence for a custom MU map:

- Each section is a colour, a start and an end.
- Scroll direction and the duo layout are filled in the way official MU maps do it.
- A live preview shows the coach cut out over the section colour.
  **Enter** (or Space outside the text fields) plays and pauses it.
- Sections are placed to 1/24 of a beat, the finest step a tape can store:
  - Type a time (`1:23.45`) or a beat (`b12`, or `b12:5` for beat 12 + 5/24).
  - ◀ ▶ move a section edge by 1/24 beat and ◀◀ ▶▶ by a whole beat.
  - Dragging snaps to whole beats. Hold Shift while dragging to snap to 1/24 beat.
  - Zoom in to see the 1/24-beat marks on the ruler.

Saving writes only `<map>/cinematics/<map>_mainsequence.tape.ckd` and backs up the old one.
The colour, scroll and pulse tapes it refers to are the same for every mashup map, so they
aren't copied into the map. The renderer uses the tool's own copies when a map doesn't have them.

## Layout

```
stage_render.py        CLI entry point
stage_render_gui.py    GUI (runs the CLI as a subprocess)
mainseq_editor.py      Mashup mainsequence editor
jdstage/               Renderer: scene/tape evaluation, materials, CPU + GPU rasterizers, video I/O
data/                  Mashup stage assets extracted from the game (world/maps/_mashup, gameconfig)
templates/             Editor templates and UI images
bin/                   ffmpeg.exe / ffprobe.exe (download separately, see Requirements)
docs/                  Showcase clip
```

## Known limitations

- The CentralShape (star) pieces of the mashup stage use scale factors calibrated
  from in-game footage, because nothing in the cooked data explains their in-game
  size (see `jdstage/stages.py`).
- Only the Mashup stage is supported. The CMU and Party Master stages are not.

## Credits

Built by Claude (Anthropic) using [Claude Code](https://claude.com/claude-code).
