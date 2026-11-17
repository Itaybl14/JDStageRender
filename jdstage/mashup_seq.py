"""
Custom mashup ("MU") mainsequence model.

An MU mainsequence (e.g. GiveMeEverythingMU) is a clock-0 Tape of TapeReferenceClips:
  * bg_scroll + pulse_fg     : whole song, Loop = Infinite (4)
  * per section, same start  : color_<colour>, color_<colour>_duo, uv_left|uv_right
    (Loop = Off) - the colour tapes fade the stage's colour sets / coach-copy panels,
    the uv tape sets the scroll direction. Sections are usually ~16 beats long.
Times are tape frames: 24 per beat, from beat 0.

The support tapes only target the generic _mashup stage's actors, so they are the same
for every map; templates live in <tool>/templates/mashup_cinematics.
"""

import json
import os
import random
import shutil
import time
import zlib
from dataclasses import dataclass, field

FRAMES_PER_BEAT = 24
COLORS = ["purple", "green", "blue", "orange"]
SCROLLS = ["left", "right"]
LOOP_OFF, LOOP_INFINITE = 0, 4

TOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(TOOL_ROOT, "templates", "mashup_cinematics")
SUPPORT_TAPES = [f"color_{c}{d}" for c in COLORS for d in ("", "_duo")] + \
                ["uv_left", "uv_right", "bg_scroll", "pulse_fg"]


@dataclass
class Section:
    start: float            # beats
    length: float = 16.0    # beats
    color: str = "purple"
    scroll: str = "left"
    duo: bool = True        # also play color_<colour>_duo (what official MU maps do)

    @property
    def end(self):
        return self.start + self.length


@dataclass
class MashupSequence:
    map_name: str
    sections: list = field(default_factory=list)
    song_end: float = 0.0   # beats; length of the infinite bg/pulse loops

    def sorted(self):
        return sorted(self.sections, key=lambda s: s.start)

    def total_end(self):
        ends = [s.end for s in self.sections] + [self.song_end]
        return max(ends) if ends else 0.0


TICK = 1.0 / FRAMES_PER_BEAT  # finest step a tape can store: 1/24 beat


def snap(beats, step=TICK):
    if step == TICK:  # exact tick count, no float drift
        return round(beats * FRAMES_PER_BEAT) / FRAMES_PER_BEAT
    return round(beats / step) * step


def fmt_beat(beats):
    """'12', '12 +5/24' (beat and 1/24-beat ticks)."""
    t = int(round(beats * FRAMES_PER_BEAT))
    sign = "-" if t < 0 else ""
    b, k = divmod(abs(t), FRAMES_PER_BEAT)
    return f"{sign}{b}" + (f" +{k}/{FRAMES_PER_BEAT}" if k else "")


# ----------------------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------------------

def _load_json(path):
    with open(path, "rb") as f:
        return json.loads(f.read().rstrip(b"\0 \r\n\t").decode("utf-8", "replace"))


def load(path, map_name=None):
    """Parse an MU mainsequence .tape.ckd into sections (unknown clips are ignored)."""
    data = _load_json(path)
    name = map_name or data.get("MapName") or _map_from_filename(path)
    seq = MashupSequence(name)
    starts = {}
    loop_end = 0
    for c in data.get("Clips") or []:
        if c.get("__class") != "TapeReferenceClip":
            continue
        tape = os.path.basename(str(c.get("Path", ""))).lower().replace(".tape", "")
        st, du = int(c.get("StartTime", 0)), int(c.get("Duration", 0))
        if tape in ("bg_scroll", "pulse_fg"):
            loop_end = max(loop_end, st + du)
            continue
        slot = starts.setdefault(st, {"dur": du})
        if tape.startswith("color_"):
            col = tape[6:]
            if col.endswith("_duo"):
                slot["duo"] = True
                slot.setdefault("color", col[:-4])
            else:
                slot["color"] = col
            slot["dur"] = max(slot["dur"], du)
        elif tape in ("uv_left", "uv_right"):
            slot["scroll"] = tape[3:]
    for st in sorted(starts):
        s = starts[st]
        if "color" not in s:
            continue
        seq.sections.append(Section(start=st / FRAMES_PER_BEAT, length=s["dur"] / FRAMES_PER_BEAT,
                                    color=s["color"] if s["color"] in COLORS else "purple",
                                    scroll=s.get("scroll", "left"), duo=bool(s.get("duo", False))))
    seq.song_end = loop_end / FRAMES_PER_BEAT
    return seq


def _map_from_filename(path):
    base = os.path.basename(path)
    i = base.lower().find("_mainsequence")
    return base[:i] if i > 0 else os.path.splitext(base)[0]


# ----------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------

def autofill(start, end, block=16.0, colors=None, first_scroll="left", duo=True):
    """Back-to-back sections of `block` beats from start to end, cycling colours and
    alternating the scroll direction (the pattern official MU maps use)."""
    colors = colors or COLORS
    out = []
    b = start
    i = 0
    while b < end - 1e-6:
        length = min(block, end - b)
        if length < block * 0.5 and out:  # merge a tiny tail into the previous section
            out[-1].length += length
            break
        scroll = SCROLLS[(SCROLLS.index(first_scroll) + i) % 2]
        out.append(Section(b, length, colors[i % len(colors)], scroll, duo))
        b += length
        i += 1
    return out


def build_tape(seq):
    """MashupSequence -> mainsequence Tape dict (cooked JSON shape)."""
    rng = random.Random(zlib.crc32(seq.map_name.lower().encode()))
    track = {k: rng.getrandbits(32) for k in ("bg", "pulse", "color", "duo", "uv")}
    base = f"world/maps/{seq.map_name.lower()}/cinematics/"

    def clip(tape, start_f, dur_f, loop, track_key):
        return {"__class": "TapeReferenceClip", "Id": rng.getrandbits(32), "TrackId": track[track_key],
                "IsActive": 1, "StartTime": int(start_f), "Duration": int(dur_f),
                "Path": base + tape + ".tape", "Loop": loop}

    end_f = int(round(seq.total_end() * FRAMES_PER_BEAT))
    clips = [clip("bg_scroll", 0, end_f, LOOP_INFINITE, "bg"),
             clip("pulse_fg", 0, end_f, LOOP_INFINITE, "pulse")]
    for s in seq.sorted():
        st = int(round(s.start * FRAMES_PER_BEAT))
        du = max(1, int(round(s.length * FRAMES_PER_BEAT)))
        clips.append(clip(f"color_{s.color}", st, du, LOOP_OFF, "color"))
        if s.duo:
            clips.append(clip(f"color_{s.color}_duo", st, du, LOOP_OFF, "duo"))
        clips.append(clip(f"uv_{s.scroll}", st, du, LOOP_OFF, "uv"))
    return {"__class": "Tape", "Clips": clips, "TapeClock": 0, "TapeBarCount": 1,
            "FreeResourcesAfterPlay": 0, "MapName": seq.map_name}


def _dump(obj):
    return json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\0"


def _find(folder, name):
    if not os.path.isdir(folder):
        return None
    for f in os.listdir(folder):
        if f.lower() == name.lower():
            return os.path.join(folder, f)
    return None


def mainsequence_path(map_dir, map_name):
    cine = _find(map_dir, "cinematics") or os.path.join(map_dir, "cinematics")
    name = f"{map_name.lower()}_mainsequence.tape.ckd"
    return _find(cine, name) or os.path.join(cine, name)


def write_map(map_dir, seq, backup=True):
    """Write only the mainsequence tape into <map_dir>/cinematics (the old one is backed up).
    The support tapes are the tool's templates and are not copied into the map.
    Returns a list of (path, action) with action in written / backup."""
    cine = _find(map_dir, "cinematics") or os.path.join(map_dir, "cinematics")
    os.makedirs(cine, exist_ok=True)
    done = []
    tape_path = mainsequence_path(map_dir, seq.map_name)
    if backup and os.path.isfile(tape_path):
        bak = tape_path + time.strftime(".%Y%m%d-%H%M%S.bak")
        shutil.copy2(tape_path, bak)
        done.append((bak, "backup"))
    with open(tape_path, "wb") as f:
        f.write(_dump(build_tape(seq)))
    done.append((tape_path, "written"))
    return done


def stage_preview_root(seq, tmp_root):
    """Lay out an unsaved sequence as <tmp_root>/<map>/cinematics/... (tape + support tapes)
    so the renderer can preview it without touching the real map folder. Returns the tape path."""
    folder = os.path.join(tmp_root, seq.map_name.lower())
    cine = os.path.join(folder, "cinematics")
    os.makedirs(cine, exist_ok=True)
    for t in SUPPORT_TAPES:
        dst = os.path.join(cine, f"{t}.tape.ckd")
        if not os.path.isfile(dst):
            shutil.copy2(os.path.join(TEMPLATE_DIR, f"{t}.tape.ckd"), dst)
    path = os.path.join(cine, f"{seq.map_name.lower()}_mainsequence.tape.ckd")
    with open(path, "wb") as f:
        f.write(_dump(build_tape(seq)))
    return folder, path
