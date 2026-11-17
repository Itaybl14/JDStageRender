"""
Frame renderer: evaluates the stage at a song time and rasterizes it.
"""

import math

import numpy as np

from . import raster
from .assets import AssetStore, Bundle, Texture
from .material import Material, material_from_serializable, BLEND_ALPHA
from .particles import EmitterDef, particles_at
from .scene import Scene
from .stages import apply_static_options, initial_plays, scene_paths
from .tapes import Schedule, TapeLibrary

_ANCHOR_OFFSETS = {  # computeMiddleCenterOffset for a 2x2 quad, y already flipped
    0: (1.0, -1.0), 1: (0.0, 0.0), 2: (1.0, 0.0), 3: (-1.0, 0.0), 4: (0.0, -1.0),
    5: (-1.0, -1.0), 6: (0.0, 1.0), 7: (1.0, 1.0), 8: (-1.0, 1.0),
}


class RenderConfig:
    """Everything a worker needs to rebuild the renderer (picklable)."""

    def __init__(self, **kw):
        self.bundles = kw["bundles"]
        self.loose_roots = kw.get("loose_roots", [])
        self.stage = kw["stage"]
        self.width = kw.get("width", 1920)
        self.height = kw.get("height", 1080)
        self.options = kw.get("options", {})
        self.timeline = kw["timeline"]
        self.mainsequence = kw.get("mainsequence")  # (path, data) or None
        self.t0 = kw["t0"]
        self.t1 = kw["t1"]
        self.map_names = kw.get("map_names", [])


class StageRenderer:
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.log = log or (lambda *a: None)
        self.store = AssetStore([Bundle(b) for b in cfg.bundles], cfg.loose_roots,
                                getattr(cfg, "map_names", []))
        paths = scene_paths(self.store)
        self.scene_path = paths[cfg.stage]
        self.scene = Scene.load(self.store, self.scene_path)
        apply_static_options(cfg.stage, self.scene, cfg.options)
        self.timeline = cfg.timeline
        self.library = TapeLibrary(self.store, self.log)
        self.schedule = Schedule(self.scene, self.library, self.timeline, cfg.t0, cfg.t1, self.log)
        for actor, label, loop in initial_plays(cfg.stage, self.scene, cfg.options):
            self.schedule.play_label(actor, label, cfg.t0, loop=loop)
        for a in self.scene.actors:
            if a.fx is not None and a.fx.default_fx:
                self.schedule.add_fx(a, a.fx.default_fx, cfg.t0 - 30.0, None)
        self.mainseq_instance = None
        if cfg.mainsequence is not None:
            path, data = cfg.mainsequence
            # The mainsequence runs on the song clock from beat 0; its ActorPaths are
            # resolved against the stage (clips aimed at map actors are ignored).
            ms_start = self.timeline.time_at_beat(0) if int(data.get("TapeClock", 0) or 0) not in (2, 3) else 0.0
            self.mainseq_instance = self.schedule.play_tape(path, None, ms_start, loop=False, tape_data=data)
            if self.mainseq_instance is not None:
                self.mainseq_instance.start_beat = 0.0
        self.schedule.build()
        self.camera = raster.Camera(cfg.width, cfg.height)
        self._clear = self._find_clear_color()
        self._emitters = {}
        self._sprites = {}
        self._fx_by_actor = {}
        for w in self.schedule.fx:
            self._fx_by_actor.setdefault(w.actor, []).append(w)

    # -- info -------------------------------------------------------------------------
    def describe(self):
        tapes = sorted({(i.owner.name if i.owner else "<mainsequence>", i.label or i.tape.path.split("/")[-1])
                        for i in self.schedule.instances})
        return {
            "scene": self.scene_path,
            "actors": len(self.scene.actors),
            "tape_instances": len(self.schedule.instances),
            "tapes": tapes,
            "fx_windows": len(self.schedule.fx),
        }

    def mainsequence_targets(self):
        if self.mainseq_instance is None:
            return 0, 0
        clips = self.mainseq_instance.compiled[0]
        aimed = [c for c in clips if c.data.get("ActorIndices") or c.data.get("ActorPaths")]
        hit = [c for c in aimed if c.targets]
        return len(hit), len(aimed)

    # -- mashup coach fade ----------------------------------------------------------------
    # JD_GSS_Gameplay_Prepare_Base::addFadeBricks(blockStart, 0.75, 0.75, 1) +
    # JD_VideoFadeBrick: at every coach-block change the main video channel's alpha
    # fades 1 -> 0 over the last 0.25 beat before the boundary and 0 -> 1 over the
    # first 0.25 beat after it. For a baked MU map the block boundaries are where the
    # mainsequence starts its (non-looping) section tapes.
    FADE_BEATS = 0.25

    def coach_fade_boundaries(self):
        if getattr(self, "_fade_beats", None) is not None:
            return self._fade_beats
        beats = set()
        if self.cfg.mainsequence is not None:
            data = self.cfg.mainsequence[1]
            engine_clock = int(data.get("TapeClock", 0) or 0) in (2, 3)
            for c in data.get("Clips") or []:
                if c.get("__class") != "TapeReferenceClip" or int(c.get("Loop", 0) or 0) in (4, 5):
                    continue
                f = int(c.get("StartTime", 0))
                if f <= 0:
                    continue
                b = self.timeline.beat_at(f / 48.0) if engine_clock else f / 24.0
                beats.add(round(b, 4))
        self._fade_beats = sorted(beats)
        return self._fade_beats

    def coach_fade_alpha(self, s):
        """Main-coach alpha multiplier at song time s (1.0 outside the fades)."""
        bounds = self.coach_fade_boundaries()
        if not bounds:
            return 1.0
        b = self.timeline.beat_at(s)
        import bisect
        i = bisect.bisect_left(bounds, b - self.FADE_BEATS)
        alpha = 1.0
        for B in bounds[max(0, i - 1):i + 2]:
            d = b - B
            if -self.FADE_BEATS <= d < self.FADE_BEATS:
                alpha = min(alpha, abs(d) / self.FADE_BEATS)
        return alpha

    def _find_clear_color(self):
        for a in self.scene.actors:
            if a.clear_color is not None and a.enabled:
                return a.clear_color[:3]
        return [0.0, 0.0, 0.0]

    # -- rendering --------------------------------------------------------------------
    def textures_for(self, material):
        return [self.store.texture(p) if p else None for p in material.textures]

    # -- backend-neutral frame description --------------------------------------------
    def frame(self, s):
        """Evaluate the stage at song time s -> (states, [(kind, actor)] in draw order).
        kind: 0 = MaterialGraphic quad, 1 = Mesh3D, 2 = particles."""
        states = self.scene.new_states()
        self.schedule.evaluate(s, states)
        self.scene.resolve_world(states)
        draws = []
        for a in self.scene.actors:
            st = states[a]
            if not st.enabled or st.world_alpha <= 0.002:
                continue
            z = st.world_pos[2]
            if a.graphic is not None and st.material is not None and st.material.has_content:
                draws.append((z, a.order, 0, a))
            if a.mesh is not None:
                draws.append((z, a.order, 1, a))
            if a in self._fx_by_actor:
                draws.append((z, a.order, 2, a))
        draws.sort(key=lambda d: (d[0], d[1], d[2]))
        return states, [(k, a) for _, _, k, a in draws]

    def quad_corners(self, a, st):
        """World-space TL, TR, BL corners of a MaterialGraphicComponent quad (uv 0,0 / 1,0 / 0,1)."""
        g = a.graphic
        if g.anchor == 9:
            ox, oy = g.custom_anchor
        else:
            ox, oy = _ANCHOR_OFFSETS.get(g.anchor, (0.0, 0.0))
        ox -= g.pos_offset[0]
        oy -= g.pos_offset[1]
        xf = self._actor_xform(st)
        return xf(ox - 1, oy + 1), xf(ox + 1, oy + 1), xf(ox - 1, oy - 1)

    def mesh_world(self, a, st):
        """(world vertices (N,3), Mesh3D) for a Mesh3DComponent, or None."""
        m = a.mesh
        try:
            mesh = self.store.mesh(m.mesh_path)
        except FileNotFoundError:
            return None
        v = mesh.vertices
        o = np.asarray(m.orientation, np.float64).reshape(4, 4)
        if np.allclose(o, 0):
            o = np.eye(4)
        vh = np.c_[v, np.ones(len(v))] @ o  # row-vector convention
        sx, sy = st.world_scale
        if st.world_flip:
            sx = -sx
        x, y, zz = vh[:, 0] * sx, vh[:, 1] * sy, vh[:, 2] * m.scale_z
        c, si = math.cos(st.world_angle), math.sin(st.world_angle)
        px, py, pz = st.world_pos
        world = np.stack([px + x * c - y * si, py + x * si + y * c, pz + zz], axis=1)
        return world, mesh

    def particles(self, a, st, s):
        """[(EmitterDef, center_world, size_world, angle, rgba)] alive at song time s."""
        sxs, sys_ = st.world_scale
        c, si = math.cos(st.world_angle), math.sin(st.world_angle)
        px, py, pz = st.world_pos
        size_scale = 0.5 * (abs(sxs) + abs(sys_))
        out = []
        for w in self._fx_by_actor.get(a, []):
            e = self._emitter(w.descriptor)
            for lx, ly, size, ang, col in particles_at(e, w, s):
                x, y = lx * sxs, ly * sys_
                wx, wy = px + x * c - y * si, py + x * si + y * c
                color = (col[0] * st.world_color[0], col[1] * st.world_color[1], col[2] * st.world_color[2],
                         col[3] * st.world_alpha)
                out.append((e, (wx, wy, pz + e.depth), size * size_scale, ang + st.world_angle, color))
        return out

    @staticmethod
    def actor_color(st):
        return (st.world_color[0], st.world_color[1], st.world_color[2], st.world_alpha)

    # -- CPU rasterizer ------------------------------------------------------------------
    def render(self, s, video_tex=None):
        """Render the stage at song time s -> float32 RGB (H, W, 3) on the CPU."""
        states, draws = self.frame(s)
        cam = self.camera
        fb = np.empty((cam.h, cam.w, 3), np.float32)
        fb[:] = self._clear
        clip = (0, 0, cam.w, cam.h)
        for kind, a in draws:
            st = states[a]
            color = self.actor_color(st)
            if kind == 0:
                mat = st.material
                raster.draw_quad(fb, cam, self.quad_corners(a, st), mat, self.textures_for(mat), color, s,
                                 clip=clip)
            elif kind == 1:
                if a.mesh.video_texture and video_tex is None:
                    continue  # TexturePatcher slot with no coach video: nothing to show
                mw = self.mesh_world(a, st)
                if mw is None:
                    continue
                world, mesh = mw
                mats = st.mesh_materials or []
                texs = [self.textures_for(mt) for mt in mats] or [[None] * 4]
                raster.draw_mesh(fb, cam, world, mesh.uvs, mesh.triangles, mats, texs, color, s,
                                 override_tex=video_tex if a.mesh.video_texture else None, clip=clip)
            else:
                for e, center, size, ang, col in self.particles(a, st, s):
                    raster.draw_sprite(fb, cam, self._sprite(e), center, size, ang, col, e.mat.blend, clip=clip)
        return fb

    def _actor_xform(self, st):
        sx, sy = st.world_scale
        if st.world_flip:
            sx = -sx
        c, si = math.cos(st.world_angle), math.sin(st.world_angle)
        px, py, pz = st.world_pos

        def xf(lx, ly):
            x, y = lx * sx, ly * sy
            return (px + x * c - y * si, py + x * si + y * c, pz)
        return xf

    def _emitter(self, d):
        key = id(d)
        e = self._emitters.get(key)
        if e is None:
            e = EmitterDef(d)
            e.mat = material_from_serializable(self.store, e.material, BLEND_ALPHA)
            self._emitters[key] = e
        return e

    def _sprite(self, e):
        key = id(e)
        spr = self._sprites.get(key)
        if spr is None:
            texs = self.textures_for(e.mat)
            base = next((t for t in texs if t is not None), None)
            res = 64 if base is None else int(min(256, max(base.w, base.h)))
            g = (np.arange(res, dtype=np.float32) + 0.5) / res
            U, V = np.meshgrid(g, g)
            spr = raster.shade(e.mat, texs, U, V, 0.0)
            if spr is None:
                spr = np.zeros((res, res, 4), np.float32)
            spr = np.ascontiguousarray(spr, np.float32)
            self._sprites[key] = spr
        return spr


# ----------------------------------------------------------------------------------------
# Worker plumbing
# ----------------------------------------------------------------------------------------

_WORKER = None


def worker_init(cfg):
    global _WORKER
    import cv2
    cv2.setNumThreads(1)
    _WORKER = StageRenderer(cfg)


def worker_render(args):
    idx, s, video = args
    vt = None
    if video is not None:
        vt = Texture(video, 2, 2, "coach")
    fb = _WORKER.render(s, vt)
    return idx, (np.clip(fb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
