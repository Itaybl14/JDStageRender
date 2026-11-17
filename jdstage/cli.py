"""
stage_render.py: render a map's mainsequence over an in-engine generated
Mashup background stage, using the textures, scenes,
materials and tapes straight out of the game's .ipk bundle.
"""

import argparse
import collections
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from . import video as V
from .assets import AssetStore, Bundle
from .render import RenderConfig, StageRenderer, worker_init, worker_render
from .stages import MASHUP_COLORS
from .tapes import ENGINE_CLOCKS
from .timeline import (MainSequence, Timeline, find_audio, find_coach_video, find_musictrack,
                       load_musictrack, map_dir_of)

TOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TOOL_ROOT, "data")  # stage assets exported from the game bundle


def log(msg):
    print(msg, flush=True)


def _parse_size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _parse_rect(s):
    vals = [float(v) for v in s.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("--coach-rect needs x,y,w,h (normalized 0..1)")
    return tuple(vals)


def build_parser():
    p = argparse.ArgumentParser(
        prog="stage_render.py",
        description="Render a Just Dance mashup mainsequence over the in-game generated Mashup stage.")
    p.add_argument("mainsequence", nargs="?", help="Path to <map>_mainsequence.tape(.ckd) (cooked JSON or Lua)")
    p.add_argument("-o", "--output", help="Output video (.mp4/.mkv/.webm), or .png for a PNG sequence "
                        "(name.png -> name_000001.png..., no audio). Default: <map>_mashup.mp4")
    p.add_argument("--map-dir", help="Map folder (musictrack, audio, coach). Default: from the mainsequence path")
    p.add_argument("--asset-root", action="append", default=[],
                   help="Extra folder searched first for map files, laid out as <root>/<map>/... (editor previews)")
    p.add_argument("--bundle", action="append",
                   help="Read stage assets from a game bundle (.ipk) instead of the local data folder")
    p.add_argument("--export-assets", metavar="BUNDLE_IPK",
                   help="(Re)build the local data folder from a game bundle, then exit")
    p.add_argument("--musictrack", help="musictrack .tpl(.ckd) (auto-detected in <map>/audio)")
    p.add_argument("--bpm", type=float, default=120.0, help="Tempo when no musictrack is found (default 120)")
    p.add_argument("--audio", help="Audio to mux (auto-detected in <map>/audio)")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--coach", help="Coach video composited on top (auto-detected in <map>/videoscoach)")
    p.add_argument("--no-coach", action="store_true")
    p.add_argument("--coach-mode", default="auto", choices=["auto", "packed", "alpha", "opaque"],
                   help="Coach alpha: packed = engine layout (alpha under colour), alpha = VP8/VP9 alpha")
    p.add_argument("--coach-rect", type=_parse_rect, help="Place the coach at x,y,w,h (normalized)")
    p.add_argument("--coach-fade", default="auto", choices=["auto", "on", "off"],
                   help="Mashup coach blink at each coach-block change (JD_VideoFadeBrick). auto = on for mashup")
    p.add_argument("--start", type=float, help="Song time (s) of the first frame")
    p.add_argument("--duration", type=float, help="Seconds to render")
    p.add_argument("--preview", action="store_true",
                   help="Render the song preview: starts at the musictrack's previewLoopStart, "
                        "30 s long unless --duration is given. Default output: <map>_preview.mp4")
    p.add_argument("--fps", type=float, default=None, help="Output fps (default: 30)")
    p.add_argument("--size", type=_parse_size, default=(1920, 1080), help="Output size WxH (default 1920x1080)")
    p.add_argument("--mashup-color", default=None, choices=MASHUP_COLORS,
                   help="Force the initial mashup colour set (default: as authored / driven by the mainsequence)")
    p.add_argument("--hide", action="append", default=[], metavar="ACTOR",
                   help="Hide a stage actor by name (repeatable), e.g. --hide x_stars")
    p.add_argument("--play", action="append", default=[], metavar="[ACTOR:]LABEL",
                   help="Extra tape label to play at start (repeatable)")
    p.add_argument("--renderer", default="auto", choices=["auto", "gpu", "cpu"],
                   help="gpu = OpenGL (fast, needs moderngl); cpu = software rasterizer. auto = gpu if available")
    p.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "x264"],
                   help="Video encoder. auto = NVIDIA NVENC when available, else x264")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="Render processes (default 4; the rasterizer is memory-bandwidth bound, more rarely helps)")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    p.add_argument("--ffmpeg", help="ffmpeg executable")
    p.add_argument("--still", type=float, action="append", metavar="SONG_TIME",
                   help="Write a PNG of the frame at this song time instead of a video (repeatable)")
    p.add_argument("--info", action="store_true", help="Print what would be rendered and exit")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.export_assets:
        from .export import export_stage_assets
        export_stage_assets([args.export_assets], DATA_DIR, log)
        return 0
    if not args.mainsequence:
        parser.error("a mainsequence path is required")

    # --- inputs ---------------------------------------------------------------------
    ms = MainSequence(args.mainsequence)
    map_dir = os.path.abspath(args.map_dir) if args.map_dir else map_dir_of(args.mainsequence)
    map_name = ms.map_name
    log(f"[map] {map_name}  ({map_dir})")
    log(f"[mainsequence] clock={ms.clock} clips={ms.summary() or 'none'}")

    bundles = args.bundle or []
    for b in bundles:
        if not os.path.isfile(b):
            sys.exit(f"bundle not found: {b}")
    if not bundles and not os.path.isdir(DATA_DIR):
        sys.exit(f"no stage assets: {DATA_DIR} is missing. Run once with --export-assets <bundle_pc.ipk> "
                 f"or pass --bundle")

    mt_path = args.musictrack or find_musictrack(map_dir, map_name)
    if mt_path:
        timeline = load_musictrack(mt_path)
        log(f"[musictrack] {mt_path}  beats {timeline.start_beat}..{timeline.end_beat}  "
            f"videoStartTime={timeline.video_start:+.3f}s")
    else:
        timeline = Timeline(bpm=args.bpm)
        log(f"[musictrack] none found - using {args.bpm:g} BPM")

    stage = "mashup"

    ffmpeg = V.find_ffmpeg(args.ffmpeg)
    ffprobe = V.find_ffprobe(ffmpeg)

    coach = None
    coach_path = None if args.no_coach else (args.coach or find_coach_video(map_dir, map_name))
    if coach_path:
        coach = V.CoachInfo(coach_path, ffprobe, args.coach_mode)
        log(f"[coach] {coach.describe()}")

    audio = None
    if not args.no_audio:
        audio = args.audio or find_audio(map_dir, map_name)
        if audio:
            log(f"[audio] {audio}")

    # --- time window (song seconds) --------------------------------------------------
    def ms_frame_to_s(f):
        if ms.clock in ENGINE_CLOCKS:
            return f / 48.0
        return timeline.time_at_beat(f / 24.0)

    span = ms.frame_span()
    ms_start = ms_frame_to_s(span[0]) if span else 0.0
    ms_end = ms_frame_to_s(span[1]) if span else None

    if args.preview:
        pls = timeline.preview_loop_start
        if pls is None:
            sys.exit("--preview needs a musictrack with previewLoopStart")
        if args.start is None:
            args.start = timeline.time_at_beat(pls)
        if args.duration is None:
            args.duration = 30.0
        log(f"[preview] previewLoopStart = beat {pls:g} ({args.start:.3f}s), {args.duration:g}s")
    if args.start is not None:
        t0 = args.start
    elif coach is not None:
        t0 = timeline.video_start
    else:
        t0 = min(0.0, ms_start)

    ends = []
    if timeline.end_time is not None and timeline.has_markers:
        ends.append(timeline.end_time)
    if ms_end is not None:
        ends.append(ms_end)
    if coach is not None and coach.duration:
        ends.append(timeline.video_start + coach.duration)
    if audio:
        d = V.media_duration(ffprobe, audio)
        if d:
            ends.append(d)
    if args.duration is not None:
        t1 = t0 + args.duration
    elif ends:
        t1 = max(ends)
    else:
        sys.exit("cannot determine the length: no musictrack/coach/audio and an empty mainsequence; pass --duration")
    if t1 <= t0:
        sys.exit("empty time window")
    fps = args.fps or 30.0
    width, height = args.size
    n_frames = int(round((t1 - t0) * fps))
    log(f"[window] song time {t0:.3f}s .. {t1:.3f}s  ({n_frames} frames @ {fps:g} fps, {width}x{height})")

    options = {
        "mashup_color": args.mashup_color,
        "extra_plays": args.play,
        "hide": args.hide,
    }
    cfg = RenderConfig(bundles=bundles, loose_roots=[*args.asset_root, map_dir, DATA_DIR], stage=stage, width=width, height=height,
                       options=options, timeline=timeline, mainsequence=(ms.path, ms.data),
                       t0=t0, t1=t1, map_names=[map_name])

    t_build = time.time()
    renderer = StageRenderer(cfg, log)
    info = renderer.describe()
    log(f"[scene] {info['scene']}  actors={info['actors']}  tape instances={info['tape_instances']}  "
        f"fx emitters={info['fx_windows']}  (built in {time.time() - t_build:.1f}s)")
    log("[tapes] " + ", ".join(f"{o}:{l}" for o, l in info["tapes"]))
    hit, aimed = renderer.mainsequence_targets()
    if aimed:
        log(f"[mainsequence] {hit}/{aimed} actor-targeting clips resolved against the stage")
    coach_rect = args.coach_rect
    if coach is not None and coach_rect is None and coach.mode == "opaque":
        log("[coach] note: opaque video covers the whole stage; use --coach-rect or an alpha/packed video")
    use_fade = args.coach_fade == "on" or args.coach_fade == "auto"
    if coach is not None and use_fade:
        nb = len(renderer.coach_fade_boundaries())
        log(f"[coach] fade at {nb} coach-block changes" if nb else "[coach] no coach-block changes found (no fade)")

    def coach_alpha(frame, s):
        if frame is None or not use_fade:
            return frame
        a = renderer.coach_fade_alpha(s)
        if a < 1.0:
            frame = frame.copy()
            frame[..., 3] *= a
        return frame

    gl = None
    if args.renderer in ("auto", "gpu"):
        try:
            from .gpu import GLRenderer
            gl = GLRenderer(renderer, width, height)
            log(f"[renderer] GPU (OpenGL: {gl.renderer_name})")
        except Exception as exc:
            if args.renderer == "gpu":
                sys.exit(f"GPU renderer unavailable: {exc}")
            log(f"[renderer] GPU unavailable ({exc.__class__.__name__}: {exc}); using CPU")
    if gl is None:
        log("[renderer] CPU")

    def fade_at(s):
        return renderer.coach_fade_alpha(s) if use_fade else 1.0

    if args.info:
        return 0

    # --- stills ------------------------------------------------------------------------
    if args.still:
        import cv2
        base = os.path.splitext(args.output or f"{map_name}_{stage}.png")[0]
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        for s in args.still:
            if gl is not None:
                native = raw = None
                if coach is not None:
                    rd = V.CoachReader(ffmpeg, coach, fps, s - timeline.video_start, width, height, coach_rect)
                    native = rd.next_native()
                    raw = rd.last_raw
                    rd.close()
                buf = gl.render(s, raw, native, coach_rect, fade_at(s))
                img = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
                out = f"{base}_{s:.2f}s.png"
                cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                log(f"[still] {out}")
                continue
            cf = raw = None
            if coach is not None:
                rd = V.CoachReader(ffmpeg, coach, fps, s - timeline.video_start, width, height, coach_rect)
                cf = rd.next()
                raw = rd.last_raw
                rd.close()
            vt = None
            if raw is not None:
                from .assets import Texture
                vt = Texture(raw.astype(np.float32) / 255.0, 2, 2, "coach")
            fb = renderer.render(s, vt)
            img = (np.clip(fb, 0, 1) * 255 + 0.5).astype(np.uint8)
            cf = coach_alpha(cf, s)
            if cf is not None:
                img = V.composite_over(img, cf)
            out = f"{base}_{s:.2f}s.png"
            cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            log(f"[still] {out}")
        return 0

    # --- video -------------------------------------------------------------------------
    out_path = args.output or f"{map_name}_{'preview' if args.preview else stage}.mp4"
    audio_offset = t0  # song time t0 is audio file time t0
    out_ext = os.path.splitext(out_path)[1].lower()
    shown = out_path
    if out_ext == ".png":
        use_nvenc = False
        pattern = V.Encoder.png_sequence_pattern(out_path)
        shown = pattern
        log(f"[encoder] PNG sequence -> {pattern}  (frames only, no audio)")
    else:
        use_nvenc = args.encoder == "nvenc" or (args.encoder == "auto" and out_ext != ".webm"
                                                and V.nvenc_available(ffmpeg))
        log(f"[encoder] {'NVENC (GPU)' if use_nvenc else 'x264' if out_ext != '.webm' else 'VP9'}")
    enc = V.Encoder(ffmpeg, out_path, width, height, fps, audio=audio, audio_offset=audio_offset,
                    duration=n_frames / fps, crf=args.crf, preset=args.preset, nvenc=use_nvenc)
    reader = None
    needs_video_tex = any(a.mesh is not None and a.mesh.video_texture for a in renderer.scene.actors)
    if coach is not None:
        reader = V.CoachReader(ffmpeg, coach, fps, t0 - timeline.video_start, width, height, coach_rect)

    started = time.time()

    def progress(i):
        if (i + 1) % max(1, int(fps)) == 0 or i + 1 == n_frames:
            el = time.time() - started
            rate = (i + 1) / el if el > 0 else 0
            eta = (n_frames - i - 1) / rate if rate > 0 else 0
            print(f"\r[render] {i + 1}/{n_frames} frames  {rate:.1f} fps  eta {eta:5.0f}s", end="", flush=True)

    if gl is not None:
        log(f"[render] -> {shown}  (GPU)")
        try:
            for i in range(n_frames):
                s = t0 + i / fps
                native = reader.next_native() if reader is not None else None
                raw = reader.last_raw if (reader is not None and needs_video_tex) else None
                enc.write(gl.render(s, raw, native, coach_rect, fade_at(s)))
                progress(i)
            print()
        finally:
            if reader:
                reader.close()
            code = enc.close()
        if code != 0:
            sys.exit(f"ffmpeg exited with {code}")
        log(f"[done] {shown}  ({time.time() - started:.0f}s)")
        return 0

    workers = max(1, args.workers)
    log(f"[render] -> {shown}  ({workers} worker{'s' if workers > 1 else ''}, CPU)")
    pending = collections.deque()
    pool = None
    if workers > 1:
        # one compute thread per worker process: BLAS/OpenMP pools in every worker
        # oversubscribe the CPU badly (spawned children inherit these variables)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=worker_init, initargs=(cfg,))
    else:
        worker_init(cfg)

    def submit(i):
        s = t0 + i / fps
        coach_frame = reader.next() if reader is not None else None
        tex = None
        if needs_video_tex and reader is not None and reader.last_raw is not None:
            # stage video-texture meshes get the raw (packed) frame, like the engine's TexturePatcher
            tex = reader.last_raw.astype(np.float32) * np.float32(1 / 255.0)
        task = (i, s, tex)
        res = pool.apply_async(worker_render, (task,)) if pool else _Immediate(worker_render(task))
        pending.append((res, coach_frame))

    try:
        nxt = 0
        window = workers * 3
        for i in range(n_frames):
            while nxt < n_frames and len(pending) < window:
                submit(nxt)
                nxt += 1
            res, coach_frame = pending.popleft()
            idx, frame = res.get()
            coach_frame = coach_alpha(coach_frame, t0 + idx / fps)
            if coach_frame is not None:
                frame = V.composite_over(frame, coach_frame)
            enc.write(frame)
            progress(i)
        print()
    finally:
        if pool:
            pool.terminate()
        if reader:
            reader.close()
        code = enc.close()
    if code != 0:
        sys.exit(f"ffmpeg exited with {code}")
    log(f"[done] {shown}  ({time.time() - started:.0f}s)")
    return 0


class _Immediate:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v
