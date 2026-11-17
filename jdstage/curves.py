"""
Curve evaluation, ported from core/math/BezierCurveFloat.cpp.

A BezierCurveFloat holds one of:
  BezierCurveFloatEmpty    -> "not set" (the clip leaves that component alone)
  BezierCurveFloatConstant -> Value
  BezierCurveFloatLinear   -> 2 keys (ValueLeft/NormalLeftOut, ValueRight/NormalRightIn)
  BezierCurveFloatMulti    -> Keys [{Value, NormalIn, NormalOut}]
Uncooked (Lua/editor) data always stores the Multi form as "Keys".

Keys are 2D points (x = clip-local tape frame, y = value) and the normals are
absolute control points of a cubic Bezier; the engine solves for x with a
20-step bisection (helpers::interpolate), reproduced exactly here.
"""

import bisect


def _bezier(p1, p2, p3, p4, t):
    it = 1.0 - t
    a = it * it * it
    b = 3.0 * it * it * t
    c = 3.0 * it * t * t
    d = t * t * t
    return (a * p1[0] + b * p2[0] + c * p3[0] + d * p4[0],
            a * p1[1] + b * p2[1] + c * p3[1] + d * p4[1])


def _interpolate(p1, p2, p3, p4, x):
    t_begin, t_end = 0.0, 1.0
    span = p4[0] - p1[0]
    t_mid = (x - p1[0]) / span if span else 0.0
    pos = p1
    for _ in range(20):
        pos = _bezier(p1, p2, p3, p4, t_mid)
        if abs(pos[0] - x) < 0.001:
            break
        if pos[0] < x:
            t_begin = t_mid
        else:
            t_end = t_mid
        t_mid = (t_begin + t_end) * 0.5
    return pos[1]


class Curve:
    __slots__ = ("kind", "value", "keys", "xs")

    def __init__(self, data):
        self.kind = "empty"
        self.value = 0.0
        self.keys = None
        self.xs = None
        if not data:
            return
        impl = data.get("Curve", data) if isinstance(data, dict) else None
        if not isinstance(impl, dict):
            return
        cls = impl.get("__class", "")
        if "Keys" in impl and impl["Keys"]:
            keys = [(tuple(k["Value"]), tuple(k.get("NormalIn", k["Value"])), tuple(k.get("NormalOut", k["Value"])))
                    for k in impl["Keys"]]
            self._set_keys(keys)
        elif cls == "BezierCurveFloatConstant" or ("Value" in impl and "ValueLeft" not in impl):
            self.kind = "const"
            self.value = float(impl.get("Value", 0.0))
        elif cls == "BezierCurveFloatLinear" or "ValueLeft" in impl:
            l = tuple(impl["ValueLeft"])
            r = tuple(impl["ValueRight"])
            keys = [(l, l, tuple(impl.get("NormalLeftOut", l))), (r, tuple(impl.get("NormalRightIn", r)), r)]
            self._set_keys(keys)

    def _set_keys(self, keys):
        if len(keys) == 1:
            self.kind = "const"
            self.value = float(keys[0][0][1])
            return
        self.kind = "keys"
        self.keys = keys
        self.xs = [k[0][0] for k in keys]

    @property
    def is_set(self):
        return self.kind != "empty"

    def __call__(self, x):
        if self.kind == "const":
            return self.value
        if self.kind == "empty":
            return 0.0
        keys = self.keys
        if x < self.xs[0]:
            return keys[0][0][1]
        i = bisect.bisect_right(self.xs, x)
        if i >= len(keys):
            return keys[-1][0][1]
        k1, k2 = keys[i - 1], keys[i]
        return _interpolate(k1[0], k1[2], k2[1], k2[0], x)


class Spline3:
    """ITF Spline used by particle curves: points of vec3 over normalized time.

    Particle curves store (max, min, mid) per key; callers pick a per-particle
    random factor and lerp between min and max.
    """

    def __init__(self, data):
        pts = []
        if isinstance(data, dict):
            spline = data.get("curve", {})
            for p in spline.get("Points", []) or []:
                pts.append((float(p.get("Time", 0.0)), tuple(float(v) for v in p.get("Point", (0, 0, 0)))))
        pts.sort(key=lambda p: p[0])
        self.points = pts
        self.times = [p[0] for p in pts]

    @property
    def is_set(self):
        return bool(self.points)

    def at(self, t):
        pts = self.points
        if not pts:
            return None
        if t <= pts[0][0] or len(pts) == 1:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        i = bisect.bisect_right(self.times, t)
        t0, a = pts[i - 1]
        t1, b = pts[i]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return tuple(a[j] + (b[j] - a[j]) * f for j in range(3))

    def sample(self, t, r, default):
        """Value at normalized time t, lerped between min (y) and max (x) by r in [0,1]."""
        v = self.at(t)
        if v is None:
            return default
        return v[1] + (v[0] - v[1]) * r
