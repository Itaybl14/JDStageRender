"""
ffmpeg I/O: coach video decoding (with the engine's alpha conventions) and the
final encode.

Coach video alpha, as the engine plays it:
  * "engine"/packed layout (webm_render.py ENGINE_* filters, and the
    video_texture_component_test.msh UVs): colour in the top H rows, alpha
    as luma in the next H/2 rows -> frame is W x 1.5H (e.g. 1920x1620).
  * VP8/VP9 with native alpha (alpha_mode=1): decoded with libvpx to keep it.
  * otherwise opaque.
"""

import json
import os
import shutil
import subprocess
import threading
import queue

import cv2
import numpy as np

TOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLED_FFMPEG = os.path.join(TOOL_ROOT, "bin", "ffmpeg.exe")

# ffmpeg/ffprobe are console programs: when this process has no console of its own
# (pythonw / the GUIs) Windows opens a new console window for each one. Only then are
# they started with CREATE_NO_WINDOW; under a console they keep sharing it, so ffmpeg's
# error messages still reach the terminal.
def _no_console():
    if os.name != "nt":
        return False
    try:
        import ctypes
        return not ctypes.windll.kernel32.GetConsoleWindow()
    except Exception:
        return False


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if _no_console() else 0


def _sp_run(*a, **kw):
    kw.setdefault("creationflags", NO_WINDOW)
    return subprocess.run(*a, **kw)


def _sp_Popen(*a, **kw):
    kw.setdefault("creationflags", NO_WINDOW)
    return subprocess.Popen(*a, **kw)


def find_ffmpeg(explicit=None):
    if explicit:
        return explicit
    if os.path.isfile(_BUNDLED_FFMPEG):
        return _BUNDLED_FFMPEG
    p = shutil.which("ffmpeg")
    if p:
        return p
    raise FileNotFoundError("ffmpeg not found (put ffmpeg.exe in the bin folder, on PATH, or pass --ffmpeg)")


def find_ffprobe(ffmpeg):
    d = os.path.dirname(ffmpeg)
    for name in ("ffprobe.exe", "ffprobe"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return shutil.which("ffprobe")


def probe(ffprobe, path):
    if not ffprobe:
        return None
    out = _sp_run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def media_duration(ffprobe, path):
    info = probe(ffprobe, path)
    if not info:
        return None
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return None


class CoachInfo:
    def __init__(self, path, ffprobe, mode="auto"):
        self.path = path
        info = probe(ffprobe, path)
        if not info:
            raise RuntimeError(f"cannot probe coach video {path}")
        v = next(s for s in info["streams"] if s.get("codec_type") == "video")
        self.width = int(v["width"])
        self.height = int(v["height"])
        self.codec = v.get("codec_name", "")
        tags = {k.lower(): str(val) for k, val in (v.get("tags") or {}).items()}
        self.native_alpha = tags.get("alpha_mode") == "1" or str(v.get("pix_fmt", "")).startswith(("yuva", "rgba", "bgra", "argb"))
        try:
            self.duration = float(info["format"]["duration"])
        except (KeyError, ValueError, TypeError):
            self.duration = None
        ratio = self.height / float(self.width)
        packed_guess = any(abs(ratio - r * 1.5) < 0.01 for r in (9 / 16, 3 / 4))
        if mode == "auto" and not self.native_alpha and not packed_guess:
            packed_guess = self._looks_packed(ffprobe)
        if mode == "auto":
            mode = "alpha" if self.native_alpha else ("packed" if packed_guess else "opaque")
        self.mode = mode
        self.color_h = int(round(self.height * 2 / 3)) if mode == "packed" else self.height

    def _looks_packed(self, ffprobe):
        """Engine videos pack alpha as greyscale luma in the bottom third (colour on top,
        e.g. 1216x720 = 480 colour rows + 240 alpha rows). Sample a few frames and check
        the bottom third is grey while the top has colour."""
        ffmpeg = os.path.join(os.path.dirname(ffprobe or ""), "ffmpeg.exe")
        if not os.path.isfile(ffmpeg):
            ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False
        votes = 0
        dur = self.duration or 10.0
        for t in (dur * 0.3, dur * 0.5, dur * 0.7):
            out = _sp_run([ffmpeg, "-v", "error", "-ss", f"{t:.2f}", "-i", self.path, "-frames:v", "1",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
            n = self.width * self.height * 3
            if len(out.stdout) < n:
                continue
            f = np.frombuffer(out.stdout[:n], np.uint8).reshape(self.height, self.width, 3).astype(np.int16)
            ch = int(round(self.height * 2 / 3))
            bottom, top = f[ch:], f[:ch]
            chroma_b = np.abs(bottom[..., 0] - bottom[..., 1]).mean() + np.abs(bottom[..., 1] - bottom[..., 2]).mean()
            chroma_t = np.abs(top[..., 0] - top[..., 1]).mean() + np.abs(top[..., 1] - top[..., 2]).mean()
            bright_b = (bottom.mean(axis=2) > 128).mean()
            if chroma_b < 2.0 and chroma_t > chroma_b + 1.0 and 0.002 < bright_b < 0.9:
                votes += 1
        return votes >= 2

    def describe(self):
        return f"{os.path.basename(self.path)} {self.width}x{self.height} {self.codec} mode={self.mode}"


class CoachReader:
    """Streams coach frames (RGBA uint8 at output size) resampled to the output fps."""

    def __init__(self, ffmpeg, info, fps, start_time, out_w, out_h, rect=None, prefetch=16):
        self.info = info
        self.out_w, self.out_h = out_w, out_h
        self.rect = rect  # (x, y, w, h) normalized placement, None = full frame
        self.frame_bytes = info.width * info.height * 4
        self.lead_frames = 0
        args = [ffmpeg, "-v", "error"]
        if info.mode == "alpha":
            if info.codec == "vp9":
                args += ["-c:v", "libvpx-vp9"]
            elif info.codec == "vp8":
                args += ["-c:v", "libvpx"]
        if start_time > 0:
            args += ["-ss", f"{start_time:.6f}"]
        else:
            self.lead_frames = int(round(-start_time * fps))
        args += ["-i", info.path, "-an", "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
        self.proc = _sp_Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.q = queue.Queue(maxsize=prefetch)
        self.done = False
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        self.index = 0

    def _pump(self):
        while True:
            buf = self.proc.stdout.read(self.frame_bytes)
            if not buf or len(buf) < self.frame_bytes:
                self.q.put(None)
                return
            self.q.put(buf)

    def next(self):
        """Next frame as float32 RGBA (out_h, out_w, 4) in [0,1], or None if none at this time."""
        raw = self.next_raw()
        return None if raw is None else self.place(self.unpack_rgba(raw))

    def next_native(self):
        """Next frame unpacked to uint8 RGBA at the video's own resolution (GPU path)."""
        raw = self.next_raw()
        return None if raw is None else self.unpack_rgba(raw)

    def next_raw(self):
        self.index += 1
        self.last_raw = None
        if self.index <= self.lead_frames or self.done:
            return None
        buf = self.q.get()
        if buf is None:
            self.done = True
            return None
        frame = np.frombuffer(buf, np.uint8).reshape(self.info.height, self.info.width, 4)
        self.last_raw = frame  # as decoded (still packed): what TexturePatcher feeds stage meshes
        return frame

    def unpack_rgba(self, frame):
        """Raw decoded frame -> uint8 RGBA with the engine alpha conventions applied."""
        info = self.info
        if info.mode == "packed":
            ch = info.color_h
            color = cv2.resize(frame[:ch, :, :3], (info.width, int(round(info.width * 9 / 16))))                 if abs(ch / info.width - 9 / 16) > 0.02 else frame[:ch, :, :3]
            alpha = frame[ch:, :, 0]
            alpha = cv2.resize(alpha, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR)
            rgba = np.dstack([color, alpha])
        elif info.mode == "alpha":
            rgba = frame
        else:
            rgba = frame.copy()
            rgba[..., 3] = 255
        return rgba

    def place(self, rgba):
        """Unpacked uint8 RGBA -> float32 RGBA at output size, placed in self.rect (CPU path)."""
        if self.rect is None:
            target = (self.out_w, self.out_h)
            img = cv2.resize(rgba, target, interpolation=cv2.INTER_AREA if rgba.shape[1] > self.out_w else cv2.INTER_LINEAR)
            return img.astype(np.float32) / 255.0
        x, y, w, h = self.rect
        pw, ph = max(1, int(round(w * self.out_w))), max(1, int(round(h * self.out_h)))
        img = cv2.resize(rgba, (pw, ph), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.out_h, self.out_w, 4), np.uint8)
        px, py = int(round(x * self.out_w)), int(round(y * self.out_h))
        x0, y0 = max(0, px), max(0, py)
        x1, y1 = min(self.out_w, px + pw), min(self.out_h, py + ph)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = img[y0 - py:y1 - py, x0 - px:x1 - px]
        return canvas.astype(np.float32) / 255.0

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass


def composite_over(stage_rgb_u8, coach_rgba_f):
    a = coach_rgba_f[..., 3:4]
    out = coach_rgba_f[..., :3] * a + (stage_rgb_u8.astype(np.float32) / 255.0) * (1.0 - a)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


_NVENC_OK = {}


def nvenc_available(ffmpeg):
    """True if this ffmpeg can actually encode with NVIDIA NVENC on this machine."""
    if ffmpeg not in _NVENC_OK:
        r = _sp_run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
        _NVENC_OK[ffmpeg] = r.returncode == 0
    return _NVENC_OK[ffmpeg]


class Encoder:
    def __init__(self, ffmpeg, out_path, width, height, fps, audio=None, audio_offset=0.0,
                 duration=None, crf=18, preset="medium", extra=None, nvenc=False):
        ext = os.path.splitext(out_path)[1].lower()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        args = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-"]
        if ext == ".png":
            self.args = args + self._png_sequence_args(out_path, duration)
            self.proc = _sp_Popen(self.args, stdin=subprocess.PIPE)
            return
        if audio:
            if audio_offset > 0:
                args += ["-ss", f"{audio_offset:.6f}"]
            args += ["-i", audio]
        args += ["-map", "0:v"]
        if audio:
            args += ["-map", "1:a"]
        if ext == ".webm":
            args += ["-c:v", "libvpx-vp9", "-crf", str(max(crf, 20)), "-b:v", "0", "-row-mt", "1"]
            if audio:
                args += ["-c:a", "libopus", "-b:a", "192k"]
        elif nvenc:
            # GPU encoder: constant-quality VBR at the same quality number
            nv_preset = {"ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3",
                         "fast": "p4", "medium": "p5", "slow": "p6", "slower": "p7"}.get(preset, "p5")
            args += ["-c:v", "h264_nvenc", "-preset", nv_preset, "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
            if audio:
                args += ["-c:a", "aac", "-b:a", "192k"]
        else:
            args += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
            if audio:
                args += ["-c:a", "aac", "-b:a", "192k"]
        args += ["-pix_fmt", "yuv420p"]
        if audio and audio_offset < 0:
            ms = int(round(-audio_offset * 1000))
            args += ["-af", f"adelay={ms}:all=1"]
        if duration:
            args += ["-t", f"{duration:.3f}"]
        if ext in (".mp4", ".mov", ".m4v"):
            args += ["-movflags", "+faststart"]
        args += list(extra or []) + [out_path]
        self.args = args
        self.proc = _sp_Popen(args, stdin=subprocess.PIPE)

    @staticmethod
    def png_sequence_pattern(out_path):
        """Frame pattern for a .png output (frames only, no audio). 'dir/name.png' ->
        'dir/name_%06d.png'; a path that already holds a printf pattern
        (e.g. 'frame_%04d.png') is used as-is."""
        if "%" in out_path:
            return out_path
        return f"{os.path.splitext(out_path)[0]}_%06d.png"

    @classmethod
    def _png_sequence_args(cls, out_path, duration):
        pattern = cls.png_sequence_pattern(out_path)
        os.makedirs(os.path.dirname(os.path.abspath(pattern)), exist_ok=True)
        args = ["-map", "0:v", "-c:v", "png", "-pix_fmt", "rgb24", "-f", "image2"]
        if duration:
            args += ["-t", f"{duration:.3f}"]
        return args + [pattern]

    def write(self, rgb):
        """rgb24 frame as a (H, W, 3) uint8 array or raw bytes."""
        self.proc.stdin.write(rgb if isinstance(rgb, (bytes, bytearray)) else np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.proc.stdin.close()
        return self.proc.wait()
