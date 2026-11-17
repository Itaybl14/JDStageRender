"""
Tape playback (engine/tapes/*), evaluated as a pure function of time so frames
can be rendered independently / in parallel.

Units: tape frames; 24 frames per beat (core/sound/Beat.h ANIMATION_FRAMES_PER_BEAT).
Clocks (tapeClock.h):
  TapeClock_Engine (2) / Custom (3) : fixed 120 BPM  -> 48 frames per second
  everything else (Conductor*/Pulse*/BeatActive)    : song beats (musictrack markers)
The "Pulse" clocks (4..7) drive bar-synced loops and are played looping.

Clip semantics follow the clip players (engine/tapes/clipplayers):
  TranslationClip              pos   = initial + curve        (per set component)
  SizeClip                     scale = curve
  ProportionClip               scale = initial * curve
  RotationClip                 angle = initial + curve (radians)
  AlphaClip / ColorClip        actor alpha / rgb (subscenes propagate to children)
  MaterialGraphicDiffuseColor  layer[LayerIdx].rgb
  MaterialGraphicDiffuseAlpha  layer[LayerIdx].a
  MaterialGraphicUVTranslation layer[LayerIdx].uvmod[UVModifierIdx].translation (anim off)
  TapeReferenceClip            nested tape; Loop: Off/Loop/PingPong/Reverse/Infinite/PingPongInfinite
                               period = min(ref tape duration, clip duration)
  TapeLauncherClip / FXClip    discrete events -> resolved ahead of time by Schedule
Actors keep the last value a clip wrote after the clip ends ("hold"), which is
what the engine does since nothing resets them.
"""

import math
import random

from .curves import Curve

ENGINE_FPS = 48.0
FRAMES_PER_BEAT = 24.0
ENGINE_CLOCKS = (2, 3)
PULSE_CLOCKS = (4, 5, 6, 7)

LOOP_OFF, LOOP_LOOP, LOOP_PINGPONG, LOOP_REVERSE, LOOP_INFINITE, LOOP_PINGPONG_INF = range(6)
INFINITE = 1 << 30


def _curve(clip, name):
    return Curve(clip.get(name))


class CClip:
    """Compiled clip."""
    __slots__ = ("cls", "start", "duration", "targets", "curves", "data", "ref", "child", "loop")

    def __init__(self, cls, start, duration, targets, curves, data):
        self.cls = cls
        self.start = start
        self.duration = duration
        self.targets = targets
        self.curves = curves
        self.data = data
        self.ref = None
        self.child = None   # compiled (clips, duration) of the referenced tape
        self.loop = LOOP_OFF

    @property
    def end(self):
        return self.start + self.duration


_CURVE_FIELDS = {
    "AlphaClip": ("Curve",),
    "ColorClip": ("CurveRed", "CurveGreen", "CurveBlue"),
    "TranslationClip": ("CurveX", "CurveY", "CurveZ"),
    "SizeClip": ("CurveX", "CurveY"),
    "ScaleClip": ("CurveX", "CurveY"),
    "ProportionClip": ("CurveX", "CurveY"),
    "RotationClip": ("CurveX", "CurveY", "CurveZ"),
    "MaterialGraphicDiffuseColorClip": ("CurveR", "CurveG", "CurveB"),
    "MaterialGraphicDiffuseAlphaClip": ("CurveA",),
    "MaterialGraphicUVTranslationClip": ("CurveU", "CurveV"),
    "MaterialGraphicUVScrollClip": ("CurveScrollU", "CurveScrollV"),
    "MaterialGraphicUVScaleClip": ("CurveU", "CurveV"),
    "MaterialGraphicUVRotationClip": ("CurveAngle", "CurvePivotX", "CurvePivotY"),
}


class CTape:
    def __init__(self, path, data):
        self.path = path
        self.clock = int(data.get("TapeClock", 0) or 0)
        self.bar_count = int(data.get("TapeBarCount", 1) or 1)
        self.actor_paths = list(data.get("ActorPaths", []) or [])
        self.raw_clips = [c for c in (data.get("Clips") or []) if isinstance(c, dict)]
        self.clips = []
        self.duration = 1
        self.compiled_for = {}

    def compile(self, library, scene, owner):
        """Resolve actor targets relative to the owner; cached per owner."""
        key = id(owner)
        if key in self.compiled_for:
            return self.compiled_for[key]
        clips = []
        result = [clips, 1]
        self.compiled_for[key] = result  # placeholder first: guards cyclic references
        ctx = owner.parent_scene if owner is not None else None
        resolved_paths = []
        for p in self.actor_paths:
            if p in ("(self)", "self"):
                resolved_paths.append(owner)
            else:
                resolved_paths.append(scene.resolve(p, ctx))
        duration = 1
        for c in self.raw_clips:
            if not c.get("IsActive", 1):
                continue
            cls = c.get("__class", "")
            start = int(c.get("StartTime", 0))
            dur = int(c.get("Duration", 0))
            targets = []
            if "ActorIndices" in c:
                for i in c.get("ActorIndices") or []:
                    if 0 <= i < len(resolved_paths) and resolved_paths[i] is not None:
                        targets.append(resolved_paths[i])
            elif "ActorPaths" in c:
                for p in c.get("ActorPaths") or []:
                    a = owner if p in ("(self)", "self") else scene.resolve(p, ctx)
                    if a is not None:
                        targets.append(a)
            curves = tuple(_curve(c, f) for f in _CURVE_FIELDS.get(cls, ()))
            cc = CClip(cls, start, dur, targets, curves, c)
            if cls == "TapeReferenceClip":
                cc.loop = int(c.get("Loop", 0) or 0)
                cc.ref = library.get(c.get("Path", ""))
                if cc.loop in (LOOP_INFINITE, LOOP_PINGPONG_INF):
                    cc.duration = INFINITE
                if cc.ref is not None:
                    cc.child = cc.ref.compile(library, scene, owner)
            clips.append(cc)
            if cc.end > duration:
                duration = cc.end
        clips.sort(key=lambda c: c.start)
        result[1] = duration
        return result


class TapeLibrary:
    def __init__(self, store, log=None):
        self.store = store
        self.cache = {}
        self.log = log or (lambda *a: None)

    def get(self, path):
        if not path:
            return None
        key = path.lower()
        if key not in self.cache:
            try:
                self.cache[key] = CTape(path, self.store.data(path))
            except Exception as exc:  # missing/corrupt tape: skip it, keep rendering
                self.log(f"[tape] cannot load {path}: {exc}")
                self.cache[key] = None
        return self.cache[key]

    def add(self, path, data):
        t = CTape(path, data)
        self.cache[path.lower()] = t
        return t


# ----------------------------------------------------------------------------------------
# Applying clips to actor states
# ----------------------------------------------------------------------------------------

def _materials(state):
    if state.material is not None:
        return [state.material]
    return state.mesh_materials or []


def apply_clip(clip, lf, states, actor_init):
    cls = clip.cls
    cv = clip.curves
    for a in clip.targets:
        s = states.get(a)
        if s is None:
            continue
        if cls == "AlphaClip":
            if cv[0].is_set:
                s.alpha = cv[0](lf)
        elif cls == "ColorClip":
            for i in range(3):
                if cv[i].is_set:
                    s.color[i] = cv[i](lf)
        elif cls == "TranslationClip":
            ip = actor_init(a)[0]
            for i in range(3):
                if cv[i].is_set:
                    s.pos[i] = ip[i] + cv[i](lf)
        elif cls in ("SizeClip", "ScaleClip"):
            for i in range(2):
                if cv[i].is_set:
                    s.scale[i] = cv[i](lf)
        elif cls == "ProportionClip":
            isc = actor_init(a)[1]
            for i in range(2):
                if cv[i].is_set:
                    s.scale[i] = isc[i] * cv[i](lf)
        elif cls == "RotationClip":
            if cv[2].is_set:
                s.angle = actor_init(a)[2] + cv[2](lf)  # radians (RotationClipPlayer)
        elif cls.startswith("MaterialGraphic"):
            layer_idx = int(clip.data.get("LayerIdx", 0))
            mod_idx = int(clip.data.get("UVModifierIdx", 0))
            for m in _materials(s):
                if not (0 <= layer_idx < 4):
                    continue
                layer = m.layers[layer_idx]
                if cls == "MaterialGraphicDiffuseColorClip":
                    for i in range(3):
                        if cv[i].is_set:
                            layer.color[i] = cv[i](lf)
                elif cls == "MaterialGraphicDiffuseAlphaClip":
                    if cv[0].is_set:
                        layer.color[3] = cv[0](lf)
                elif cls in ("MaterialGraphicUVTranslationClip", "MaterialGraphicUVScaleClip",
                             "MaterialGraphicUVRotationClip", "MaterialGraphicUVScrollClip"):
                    if not (0 <= mod_idx < len(layer.uv_mods)):
                        continue
                    mod = layer.uv_mods[mod_idx]
                    if cls == "MaterialGraphicUVTranslationClip":
                        if cv[0].is_set:
                            mod.tu = cv[0](lf)
                        if cv[1].is_set:
                            mod.tv = cv[1](lf)
                        mod.anim_u = mod.anim_v = False
                    elif cls == "MaterialGraphicUVScrollClip":
                        # UVScrollClipPlayer: translation becomes a speed (anim on)
                        if cv[0].is_set:
                            mod.tu = cv[0](lf)
                        if cv[1].is_set:
                            mod.tv = cv[1](lf)
                        mod.anim_u = mod.anim_v = True
                    elif cls == "MaterialGraphicUVScaleClip":
                        if cv[0].is_set:
                            mod.su = cv[0](lf)
                        if cv[1].is_set:
                            mod.sv = cv[1](lf)
                    else:  # MaterialGraphicUVRotationClipPlayer: angle (radians) + optional pivot
                        if cv[0].is_set:
                            mod.rot = cv[0](lf)
                        mod.anim_rot = False
                        if cv[1].is_set:
                            mod.rot_ou = cv[1](lf)
                        if cv[2].is_set:
                            mod.rot_ov = cv[2](lf)
                elif cls == "MaterialGraphicEnableLayerClip":
                    layer.enabled = bool(clip.data.get("LayerEnabled", clip.data.get("Enable", 1)))
        elif cls == "ActorEnableClip":
            s.enabled = bool(clip.data.get("ActorEnable", clip.data.get("Enable", 1)))


def _wrap(lf, period, loop):
    """TapeReferenceClipPlayer::getTimeInTape -> (time, iteration)."""
    if period <= 0:
        return lf, 0
    it = int(lf // period)
    t = lf - it * period
    if loop in (LOOP_PINGPONG, LOOP_PINGPONG_INF) and it % 2 == 1:
        t = period - t
    elif it > 0 and t == 0:
        t = period
        it -= 1
    return t, it


def eval_tape(ctape_compiled, f, states, actor_init, depth=0):
    clips, duration = ctape_compiled
    if depth > 16:
        return
    for c in clips:
        if f < c.start:
            break
        lf = f - c.start
        if lf > c.duration:
            lf = c.duration
        if c.cls == "TapeReferenceClip":
            child = c.child
            if child is None:
                continue
            child_dur = child[1]
            if c.loop == LOOP_OFF:
                eval_tape(child, min(lf, child_dur), states, actor_init, depth + 1)
            elif c.loop == LOOP_REVERSE:
                eval_tape(child, max(0.0, c.duration - lf), states, actor_init, depth + 1)
            else:
                period = min(child_dur, c.duration)
                t, it = _wrap(lf, period, c.loop)
                if it >= 1:
                    eval_tape(child, period, states, actor_init, depth + 1)  # state held from previous loop
                eval_tape(child, t, states, actor_init, depth + 1)
        elif c.cls in _CURVE_FIELDS or c.cls.startswith("MaterialGraphic") or c.cls == "ActorEnableClip":
            apply_clip(c, lf, states, actor_init)


# ----------------------------------------------------------------------------------------
# Instances & schedule
# ----------------------------------------------------------------------------------------

class TapeInstance:
    def __init__(self, tape, compiled, owner, label, start_s, start_beat, loop=False, group=None):
        self.tape = tape
        self.compiled = compiled
        self.owner = owner
        self.label = label
        self.start_s = start_s
        self.start_beat = start_beat
        self.stop_s = None
        self.loop = loop
        self.group = group

    def frame_at(self, s, timeline):
        if self.stop_s is not None and s > self.stop_s:
            s = self.stop_s
        if self.tape.clock in ENGINE_CLOCKS:
            return (s - self.start_s) * ENGINE_FPS
        return (timeline.beat_at(s) - self.start_beat) * FRAMES_PER_BEAT

    def song_time_of_frame(self, f, timeline):
        if self.tape.clock in ENGINE_CLOCKS:
            return self.start_s + f / ENGINE_FPS
        return timeline.time_at_beat(self.start_beat + f / FRAMES_PER_BEAT)


class FxWindow:
    def __init__(self, actor, descriptor, start_s, stop_s, kill_s, seed):
        self.actor = actor
        self.descriptor = descriptor
        self.start_s = start_s
        self.stop_s = stop_s
        self.kill_s = kill_s
        self.seed = seed


class Schedule:
    """All tape instances and FX emission windows for the render window."""

    def __init__(self, scene, library, timeline, t0, t1, log=None):
        self.scene = scene
        self.library = library
        self.timeline = timeline
        self.t0 = t0
        self.t1 = t1
        self.instances = []
        self.fx = []
        self.log = log or (lambda *a: None)
        self._seq_counter = {}
        self._rng = random.Random(1234)
        self._queue = []
        self._init_cache = {}

    # -- building ---------------------------------------------------------------------
    def play_label(self, actor, label, s, loop=None):
        entry = actor.tapes.get(label.lower()) if actor is not None else None
        if not entry:
            self.log(f"[tape] {actor.path if actor else '?'}: no tape labelled '{label}'")
            return None
        path, group = entry
        # same group -> the previous tape of that group stops
        for inst in self.instances:
            if inst.owner is actor and inst.group == group and inst.stop_s is None and inst.start_s <= s:
                inst.stop_s = s
        return self.play_tape(path, actor, s, loop=loop, label=label, group=group)

    def play_tape(self, path, owner, s, loop=None, label="", group=None, tape_data=None):
        tape = self.library.add(path, tape_data) if tape_data is not None else self.library.get(path)
        if tape is None:
            return None
        compiled = tape.compile(self.library, self.scene, owner)
        if loop is None:
            loop = tape.clock in PULSE_CLOCKS
        start_beat = self.timeline.beat_at(s)
        if tape.clock in PULSE_CLOCKS:
            start_beat = math.floor(start_beat + 1e-6)  # pulse tapes are bar/beat aligned
            s = self.timeline.time_at_beat(start_beat)
        inst = TapeInstance(tape, compiled, owner, label, s, start_beat, loop, group)
        self.instances.append(inst)
        self._queue.append(inst)
        return inst

    def add_fx(self, actor, fx_name, start_s, stop_s=None, kill=False):
        if actor is None or actor.fx is None:
            return
        names = actor.fx.controls.get(fx_name) or ([fx_name] if fx_name in actor.fx.descriptors else [])
        for n in names:
            d = actor.fx.descriptors.get(n)
            if d is not None:
                seed = (hash(actor.path) ^ hash(n) ^ int(start_s * 1000)) & 0x7FFFFFFF
                self.fx.append(FxWindow(actor, d, start_s, stop_s, stop_s if kill else None, seed))

    def build(self):
        guard = 0
        while self._queue and guard < 20000:
            guard += 1
            inst = self._queue.pop(0)
            self._scan(inst)
        self.instances.sort(key=lambda i: i.start_s)

    def _scan(self, inst):
        """Emit TapeLauncherClip / FXClip events of an instance within the window."""
        clips, duration = inst.compiled
        period = duration if inst.loop else None
        end_s = self.t1 if inst.stop_s is None else min(self.t1, inst.stop_s)
        iteration = 0
        while True:
            base = (period or 0) * iteration
            any_in = False
            for f, clip in _trigger_clips(inst.compiled, 0, self.library):
                abs_f = base + f
                s = inst.song_time_of_frame(abs_f, self.timeline)
                if s > end_s + 1e-6:
                    continue
                any_in = True
                self._fire(inst, clip, s, abs_f)
            if not period:
                break
            iteration += 1
            if inst.song_time_of_frame(base + period, self.timeline) > end_s or iteration > 5000:
                break
            if not any_in and iteration > 1:
                pass

    def _fire(self, inst, clip, s, abs_f):
        if clip.cls == "TapeLauncherClip":
            action = int(clip.data.get("Action", 0))
            labels = [str(l) for l in (clip.data.get("TapeLabels") or [])]
            if not labels:
                return
            for a in clip.targets:
                if action == 1:  # stop
                    for other in self.instances:
                        if other.owner is a and other.label.lower() in [l.lower() for l in labels] and other.stop_s is None:
                            other.stop_s = s
                    continue
                if int(clip.data.get("TapeChoice", 0)) == 1:
                    key = (id(clip), id(a))
                    idx = self._seq_counter.get(key, 0)
                    self._seq_counter[key] = idx + 1
                    label = labels[idx % len(labels)]
                else:
                    label = self._rng.choice(labels)
                # don't re-launch the very same tape that's still looping
                running = [o for o in self.instances if o.owner is a and o.label.lower() == label.lower()
                           and o.stop_s is None and o.loop]
                if running:
                    continue
                self.play_label(a, label, s)
        elif clip.cls == "FXClip":
            dur_s = inst.song_time_of_frame(abs_f + clip.duration, self.timeline) - s
            for a in clip.targets:
                self.add_fx(a, clip.data.get("FxName", ""), s, s + max(dur_s, 0.0),
                            kill=bool(clip.data.get("KillParticlesOnEnd", 0)))

    # -- evaluation --------------------------------------------------------------------
    def actor_init(self, actor):
        v = self._init_cache.get(actor)
        if v is None:
            pos, sc, ang = actor.initial_local()
            v = (list(pos), list(sc), ang)
            self._init_cache[actor] = v
        return v

    def evaluate(self, s, states):
        for inst in self.instances:
            if s < inst.start_s:
                continue
            f = inst.frame_at(s, self.timeline)
            if inst.loop:
                period = inst.compiled[1]
                t, it = _wrap(f, period, LOOP_LOOP)
                if it >= 1:
                    eval_tape(inst.compiled, period, states, self.actor_init)
                f = t
            eval_tape(inst.compiled, f, states, self.actor_init)


def _trigger_clips(compiled, base, library, depth=0, limit_f=INFINITE):
    """Yield (frame, clip) for every launcher/FX clip, expanding references and loops."""
    if depth > 8:
        return
    clips, duration = compiled
    for c in clips:
        if c.start + base > limit_f:
            continue
        if c.cls in ("TapeLauncherClip", "FXClip"):
            yield base + c.start, c
        elif c.cls == "TapeReferenceClip" and c.child is not None:
            child = c.child
            if c.loop == LOOP_OFF or c.loop == LOOP_REVERSE:
                yield from _trigger_clips(child, base + c.start, library, depth + 1,
                                          min(limit_f, base + c.end))
            else:
                period = max(1, min(child[1], c.duration))
                span_end = min(limit_f, base + c.start + min(c.duration, period * 2000))
                k = 0
                while base + c.start + k * period <= span_end and k < 2000:
                    yield from _trigger_clips(child, base + c.start + k * period, library, depth + 1,
                                              min(span_end, base + c.start + (k + 1) * period))
                    k += 1
