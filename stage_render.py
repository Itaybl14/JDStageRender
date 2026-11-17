#!/usr/bin/env python3
"""
stage_render.py

Reads a map's mainsequence and renders a video of it over one of the engine's
in-game generated Mashup background stage -- the "generic stage" a song gets when
its songdesc backgroundType is Mashup (6)
(JD_GSS_LoadMap::addSongLayers, gameconfig.isg "genericstages").

Everything visual comes out of the game bundle (default
output/bin/Release/bundle_pc.ipk): the stage .isc scene, actor templates,
.msh material shaders, cooked textures (.tga/.png.ckd -> DDS), .m3d meshes,
particle FX templates and the stage's own tapes. The engine behaviour is
re-implemented from this repo's source (see the jdstage/ modules):
tape clocks & clip players, binds, the multi-texture shader (transcribed from
the shipped MultiTexture.fx bytecode), camera (25 deg vertical FOV at z=10).

Usage:
  python stage_render.py <map>/cinematics/<map>_mainsequence.tape.ckd
  python stage_render.py seq.tape.ckd --mashup-color purple -o out.mp4
  python stage_render.py seq.tape.ckd --coach coach.webm --audio song.ogg
  python stage_render.py seq.tape.ckd --still 10 --still 30   # PNG previews
  python stage_render.py seq.tape.ckd --info                                     # show plan only

Requires: numpy, opencv-python, texture2ddecoder, ffmpeg (+ffprobe).
"""

import sys

from jdstage.cli import main

if __name__ == "__main__":
    sys.exit(main())
