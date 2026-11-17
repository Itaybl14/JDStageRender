"""
Song timeline: mainsequence + musictrack.

MusicTrackStructure (engine/sound/musictrack): `markers` are sample positions
at 48 kHz (MusicTrack.cpp uses 48000.0 everywhere), one per beat starting at
`startBeat`. Song time 0 == sample 0 of the audio file. `videoStartTime` is
where the coach video starts relative to the audio (usually <= 0).
"""

import bisect
import glob
import os

from .assets import parse_data

SAMPLE_RATE = 48000.0


class Timeline:
    def __init__(self, markers=None, start_beat=0, end_beat=None, video_start=0.0, bpm=120.0):
        self.bpm = bpm
        self.preview_entry = self.preview_loop_start = self.preview_loop_end = None  # beats
        self.start_beat = int(start_beat or 0)
        self.video_start = float(video_start or 0.0)
        self.times = [m / SAMPLE_RATE for m in markers] if markers and len(markers) >= 2 else None
        if self.times:
            self.end_beat = end_beat if end_beat is not None else self.start_beat + len(self.times) - 1
        else:
            self.end_beat = end_beat

    @property
    def has_markers(self):
        return self.times is not None

    def beat_at(self, s):
        if not self.times:
            return s * self.bpm / 60.0
        t = self.times
        if s <= t[0]:
            return self.start_beat + (s - t[0]) / (t[1] - t[0])
        if s >= t[-1]:
            return self.start_beat + len(t) - 1 + (s - t[-1]) / (t[-1] - t[-2])
        i = bisect.bisect_right(t, s) - 1
        return self.start_beat + i + (s - t[i]) / (t[i + 1] - t[i])

    def time_at_beat(self, b):
        if not self.times:
            return b * 60.0 / self.bpm
        t = self.times
        i = b - self.start_beat
        if i <= 0:
            return t[0] + i * (t[1] - t[0])
        if i >= len(t) - 1:
            return t[-1] + (i - (len(t) - 1)) * (t[-1] - t[-2])
        k = int(i)
        return t[k] + (i - k) * (t[k + 1] - t[k])

    @property
    def end_time(self):
        if self.end_beat is None:
            return None
        return self.time_at_beat(self.end_beat)


def load_musictrack(path):
    """Accepts a cooked .tpl.ckd (JSON) or Lua .tpl (+ its .trk include)."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin1")
    data = None
    if "includeReference" in text and not raw.lstrip()[:1] == b"{":
        # Lua: the structure lives in the included .trk; prepend it so `structure` resolves
        import re
        from . import luatable
        inc = re.findall(r'includeReference\("([^"]+)"\)', text)
        prefix = ""
        for ref in inc:
            cand = os.path.join(os.path.dirname(path), os.path.basename(ref))
            if os.path.isfile(cand):
                with open(cand, "rb") as f:
                    prefix += f.read().decode("latin1") + "\n"
        env = luatable.parse(prefix + text)
        data = luatable.normalize(env.get("params", env))
    else:
        data = parse_data(raw, path)
    struct_ = _find(data, "MusicTrackStructure") or _find_key(data, "markers")
    if not struct_:
        raise ValueError(f"{path}: no MusicTrackStructure found")
    markers = [float(m) for m in struct_.get("markers", [])]
    tl = Timeline(markers, struct_.get("startBeat", 0), struct_.get("endBeat"),
                  struct_.get("videoStartTime", 0.0))
    # song preview (menu preview loop), in beats
    for key, attr in (("previewEntry", "preview_entry"), ("previewLoopStart", "preview_loop_start"),
                      ("previewLoopEnd", "preview_loop_end")):
        v = struct_.get(key)
        setattr(tl, attr, float(v) if v is not None else None)
    return tl


def _find(obj, cls):
    if isinstance(obj, dict):
        if obj.get("__class") == cls:
            return obj
        for v in obj.values():
            r = _find(v, cls)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, cls)
            if r:
                return r
    return None


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj
        for v in obj.values():
            r = _find_key(v, key)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r:
                return r
    return None


class MainSequence:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.data = parse_data(f.read(), path)
        if self.data.get("__class") not in (None, "Tape"):
            inner = _find(self.data, "Tape")
            if inner:
                self.data = inner
        self.clips = [c for c in (self.data.get("Clips") or []) if isinstance(c, dict)]
        self.clock = int(self.data.get("TapeClock", 0) or 0)
        self.map_name = self.data.get("MapName") or _guess_map_name(path)

    def frame_span(self):
        if not self.clips:
            return None
        start = min(int(c.get("StartTime", 0)) for c in self.clips)
        end = max(int(c.get("StartTime", 0)) + int(c.get("Duration", 0)) for c in self.clips)
        return start, end

    def summary(self):
        counts = {}
        for c in self.clips:
            counts[c.get("__class", "?")] = counts.get(c.get("__class", "?"), 0) + 1
        return counts


def _guess_map_name(path):
    base = os.path.basename(path)
    low = base.lower()
    i = low.find("_mainsequence")
    return base[:i] if i > 0 else os.path.splitext(base)[0]


def map_dir_of(mainseq_path):
    d = os.path.dirname(os.path.abspath(mainseq_path))
    if os.path.basename(d).lower() == "cinematics":
        return os.path.dirname(d)
    return d


def find_musictrack(map_dir, map_name):
    pats = []
    for sub in ("audio", "Audio", ""):
        base = os.path.join(map_dir, sub) if sub else map_dir
        pats += [os.path.join(base, "*musictrack*.tpl.ckd"), os.path.join(base, "*musictrack*.tpl"),
                 os.path.join(base, "*MusicTrack*.tpl")]
    seen = []
    for p in pats:
        for f in glob.glob(p):
            if f not in seen:
                seen.append(f)
    if not seen:
        return None
    seen.sort(key=lambda f: (map_name.lower() not in os.path.basename(f).lower(), len(f)))
    return seen[0]


def find_audio(map_dir, map_name):
    exts = (".ogg", ".wav", ".opus", ".mp3", ".m4a", ".flac")
    cands = []
    for sub in ("audio", "Audio", ""):
        base = os.path.join(map_dir, sub) if sub else map_dir
        if not os.path.isdir(base):
            continue
        for f in os.listdir(base):
            if f.lower().endswith(exts) and "amb" not in f.lower():
                cands.append(os.path.join(base, f))
    if not cands:
        return None
    cands.sort(key=lambda f: (map_name.lower() not in os.path.basename(f).lower(), -os.path.getsize(f)))
    return cands[0]


def find_coach_video(map_dir, map_name):
    exts = (".webm", ".mp4", ".mkv", ".mov")
    cands = []
    for sub in ("videoscoach", "VideosCoach", "video", "Video", ""):
        base = os.path.join(map_dir, sub) if sub else map_dir
        if not os.path.isdir(base):
            continue
        for f in os.listdir(base):
            if f.lower().endswith(exts) and "preview" not in f.lower():
                cands.append(os.path.join(base, f))
    if not cands:
        return None
    cands.sort(key=lambda f: (map_name.lower() not in os.path.basename(f).lower(), -os.path.getsize(f)))
    return cands[0]
