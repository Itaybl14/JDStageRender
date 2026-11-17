"""
Particle FX (FxBankComponent / ITF_ParticleGenerator_Template), evaluated as a
pure function of time: particle k of an emitter is born at start + k*period
with its own seeded random draws, so any frame can be computed independently.

Supported subset of ParticleGeneratorParameters (what the generic stages use):
  freq / maxParticles / emitParticlesCount     emission rate & cap
  initLifeTime | curveParLifeTime              lifetime (s)
  pos, genBox, circleRadius/innerCircleRadius  spawn shape (actor-local)
  velNorm/velAngle/velAngleDelta | parEmitVelocity/parEmitVelocityAngle
  grav + acc                                   constant acceleration
  initAngle/angleDelta, angularSpeed | parEmitAngularSpeed
  curveSize / curveAlpha / curveRGB            over normalized life
  defaultColor                                 ARGB
Particle curve keys store (max, min, mid); each particle draws one random
factor per curve and lerps min..max (the editor's random-range convention).
"""

import math
import random

from .curves import Spline3
from .material import argb_to_rgba


class EmitterDef:
    def __init__(self, descriptor):
        gen = descriptor.get("gen", {}) or {}
        p = gen.get("params", {}) or {}
        self.material = descriptor.get("material", {}) or {}
        self.max_particles = max(1, int(p.get("maxParticles", 16) or 16))
        emit_count = int(p.get("emitParticlesCount", 0xFFFFFFFF))
        self.emit_count = None if emit_count in (0xFFFFFFFF, -1) else emit_count
        self.freq = float(p.get("freq", 0.1) or 0.1)
        self.freq_curve = Spline3(p.get("curveFreq"))
        self.life = float(p.get("initLifeTime", 1.0) or 1.0)
        self.life_curve = Spline3(p.get("curveParLifeTime"))
        self.pos = p.get("pos", [0, 0, 0]) or [0, 0, 0]
        box = p.get("genBox", {}) or {}
        self.box_min = box.get("MIN", [0, 0])
        self.box_max = box.get("MAX", [0, 0])
        self.circle = float(p.get("circleRadius", 0) or 0)
        self.inner_circle = float(p.get("innerCircleRadius", 0) or 0)
        self.gen_type = int(p.get("genGenType", 0) or 0)
        self.vel_norm = float(p.get("velNorm", 0) or 0)
        self.vel_angle = float(p.get("velAngle", 0) or 0)
        self.vel_angle_delta = float(p.get("velAngleDelta", 0) or 0)
        self.vel_curve = Spline3(p.get("parEmitVelocity"))
        self.vel_angle_curve = Spline3(p.get("parEmitVelocityAngle"))
        g = p.get("grav", [0, 0, 0]) or [0, 0, 0]
        a = p.get("acc", [0, 0, 0]) or [0, 0, 0]
        self.accel = (float(g[0]) + float(a[0]), float(g[1]) + float(a[1]))
        self.init_angle = float(p.get("initAngle", 0) or 0)
        self.angle_delta = float(p.get("angleDelta", 0) or 0)
        self.ang_speed = float(p.get("angularSpeed", 0) or 0)
        self.ang_speed_delta = float(p.get("angularSpeedDelta", 0) or 0)
        self.ang_speed_curve = Spline3(p.get("parEmitAngularSpeed"))
        self.size_curve = Spline3(p.get("curveSize"))
        self.alpha_curve = Spline3(p.get("curveAlpha"))
        self.rgb_curve = Spline3(p.get("curveRGB"))
        self.default_color = argb_to_rgba(p.get("defaultColor", [1, 1, 1, 1]))
        self.depth = float(p.get("depth", 0) or 0)

        avg_life = self.life
        if self.life_curve.is_set:
            v = self.life_curve.at(0.0)
            avg_life = 0.5 * (v[0] + v[1])
        avg_life = max(0.05, min(avg_life, 120.0))
        self.avg_life = avg_life
        period = self.freq if self.freq > 0 else 0.1
        if self.freq_curve.is_set:
            v = self.freq_curve.at(0.0)
            mult = 0.5 * (v[0] + v[1])
            if mult > 0:
                period *= mult
        # never exceed maxParticles alive at once
        self.period = max(period, avg_life / self.max_particles)


def _rand(seed, k, n):
    r = random.Random((seed * 1000003 + k * 7919) & 0xFFFFFFFF)
    return [r.random() for _ in range(n)]


def particles_at(edef, window, s):
    """Yield (local_x, local_y, size, angle_rad, (r,g,b,a)) for particles alive at song time s."""
    if s < window.start_s:
        return
    if window.kill_s is not None and s > window.kill_s:
        return
    emit_end = window.stop_s if window.stop_s is not None else float("inf")
    period = edef.period
    max_life = edef.avg_life * 2.5 + 1.0
    k_first = max(0, int(math.floor((s - max_life - window.start_s) / period)))
    k_last = int(math.floor((min(s, emit_end) - window.start_s) / period))
    if edef.emit_count is not None:
        k_last = min(k_last, edef.emit_count - 1)
    for k in range(k_first, k_last + 1):
        birth = window.start_s + k * period
        age = s - birth
        if age < 0:
            continue
        r = _rand(window.seed, k, 12)
        life = edef.life
        if edef.life_curve.is_set:
            life = edef.life_curve.sample(0.0, r[0], life)
        life = max(0.05, min(life, 120.0))
        if age > life:
            continue
        tn = age / life
        # spawn position
        px, py = float(edef.pos[0]), float(edef.pos[1])
        bx0, by0 = edef.box_min[0], edef.box_min[1]
        bx1, by1 = edef.box_max[0], edef.box_max[1]
        if bx0 != bx1 or by0 != by1:
            px += bx0 + (bx1 - bx0) * r[1]
            py += by0 + (by1 - by0) * r[2]
        elif edef.circle > 0:
            ang = r[1] * 2 * math.pi
            rad = edef.inner_circle + (edef.circle - edef.inner_circle) * math.sqrt(r[2])
            px += math.cos(ang) * rad
            py += math.sin(ang) * rad
        # velocity
        speed = edef.vel_norm
        if edef.vel_curve.is_set:
            speed = edef.vel_curve.sample(0.0, r[3], speed)
        vang = edef.vel_angle + (r[4] * 2 - 1) * edef.vel_angle_delta
        if edef.vel_angle_curve.is_set:
            vang = edef.vel_angle_curve.sample(0.0, r[4], vang)
        vang = math.radians(vang)
        vx, vy = math.cos(vang) * speed, math.sin(vang) * speed
        ax, ay = edef.accel
        x = px + vx * age + 0.5 * ax * age * age
        y = py + vy * age + 0.5 * ay * age * age
        # rotation
        a0 = math.radians(edef.init_angle + (r[5] * 2 - 1) * edef.angle_delta)
        aspeed = edef.ang_speed + (r[6] * 2 - 1) * edef.ang_speed_delta
        if edef.ang_speed_curve.is_set:
            aspeed = edef.ang_speed_curve.sample(0.0, r[6], aspeed)
        angle = a0 + math.radians(aspeed) * age
        # appearance over life
        size = edef.size_curve.sample(tn, r[7], 1.0)
        alpha = edef.alpha_curve.sample(tn, r[8], 1.0)
        rgb = edef.rgb_curve.at(tn) if edef.rgb_curve.is_set else (1.0, 1.0, 1.0)
        dc = edef.default_color
        color = (rgb[0] * dc[0], rgb[1] * dc[1], rgb[2] * dc[2], max(0.0, min(1.0, alpha)) * dc[3])
        if color[3] <= 0.002 or size <= 0:
            continue
        yield x, y, size, angle, color
