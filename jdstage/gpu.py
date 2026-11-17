"""
OpenGL renderer (moderngl): same scene/tape evaluation as the CPU path
(StageRenderer.frame / quad_corners / mesh_world / particles), drawn on the GPU.

The fragment shader is a port of the CPU shading in raster.py, itself a
transcription of the engine's MultiTexture.fx:
  per layer: rgb = saturate(tex.rgb + colorSat); a = dot(tex, alphaSelector) + alphaSat
             a = thr >= 0 ? (a > thr) : a;  L = (rgb, a) * DiffuseColor
  first enabled layer is the base (premultiplied when it isn't layer 0), later layers
  combine with BlendLayerN (ALPHA / ADD / ADDALPHA / MUL / ALPHAMUL)
  out = res * vertexColor; discard if out.a < 1/255
Texture addressing (wrap / mirror / clamp / border) is done in the shader so it can be
chosen per material layer; textureGrad on the unwrapped UVs keeps mip selection
seamless across wraps. The target is RGBA8 like the engine backbuffer, so every draw
clamps exactly as the real blend stage does.
"""

import math

import numpy as np

from . import material as M
from .raster import NEAR

VERT = """
#version 330
in vec4 in_pos;
in vec2 in_uv;
in vec4 in_col;
out vec2 v_uv;
out vec4 v_col;
void main() { gl_Position = in_pos; v_uv = in_uv; v_col = in_col; }
"""

FRAG = """
#version 330
in vec2 v_uv;
in vec4 v_col;
out vec4 f_color;
uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;
uniform int on[4];
uniform vec3 r0[4];
uniform vec3 r1[4];
uniform vec4 lcol[4];
uniform vec4 sel[4];
uniform vec2 sat[4];
uniform float thr[4];
uniform ivec2 wrapm[4];
uniform int bl[4];

float wrapc(float c, int m) {
    if (m == 0) return fract(c);
    if (m == 1) { float t = mod(c, 2.0); return t < 1.0 ? t : 2.0 - t; }
    if (m == 2) return clamp(c, 0.0, 1.0);
    return c;
}

vec4 fetch(int i, vec2 uv, vec2 dx, vec2 dy) {
    ivec2 w = wrapm[i];
    if ((w.x == 3 && (uv.x < 0.0 || uv.x > 1.0)) || (w.y == 3 && (uv.y < 0.0 || uv.y > 1.0)))
        return vec4(0.0);
    vec2 g = vec2(wrapc(uv.x, w.x), wrapc(uv.y, w.y));
    if (i == 0) return textureGrad(tex0, g, dx, dy);
    if (i == 1) return textureGrad(tex1, g, dx, dy);
    if (i == 2) return textureGrad(tex2, g, dx, dy);
    return textureGrad(tex3, g, dx, dy);
}

void main() {
    vec3 base = vec3(v_uv, 1.0);
    vec2 uv[4];
    vec2 ddx[4];
    vec2 ddy[4];
    for (int i = 0; i < 4; ++i) {      // derivatives in uniform control flow
        uv[i] = vec2(dot(r0[i], base), dot(r1[i], base));
        ddx[i] = dFdx(uv[i]);
        ddy[i] = dFdy(uv[i]);
    }
    vec4 res = vec4(0.0);
    bool have = false;
    for (int i = 0; i < 4; ++i) {
        if (on[i] == 0) continue;
        vec4 t = fetch(i, uv[i], ddx[i], ddy[i]);
        vec3 rgb = clamp(t.rgb + vec3(sat[i].x), 0.0, 1.0);
        float a = dot(t, sel[i]) + sat[i].y;
        if (thr[i] >= 0.0) a = a > thr[i] ? 1.0 : 0.0;
        vec4 L = vec4(rgb * lcol[i].rgb, a * lcol[i].a);
        if (!have) {
            res = L;
            if (i > 0) res.rgb *= res.a;
            have = true;
            continue;
        }
        int op = bl[i];
        if (op == 6)       res += L;
        else if (op == 7)  res += vec4(L.rgb * L.a, L.a);
        else if (op == 10) res *= L;
        else if (op == 11) res *= L.a;
        else               res = res * (1.0 - L.a) + vec4(L.rgb * L.a, L.a);
    }
    if (!have) discard;
    vec4 o = res * v_col;
    if (o.a < 1.0 / 255.0) discard;
    f_color = o;
}
"""

QUAD_FRAG = """
#version 330
in vec2 v_uv;
in vec4 v_col;
out vec4 f_color;
uniform sampler2D tex0;
void main() { f_color = texture(tex0, v_uv) * v_col; }
"""

# engine GFX_BLENDMODE -> (equation, src_rgb, dst_rgb)
def _blend_table(mgl):
    A, ADD, SUB = mgl.FUNC_ADD, mgl.FUNC_ADD, mgl.FUNC_REVERSE_SUBTRACT
    return {
        M.BLEND_COPY: (A, mgl.ONE, mgl.ZERO),
        M.BLEND_ALPHA: (A, mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA),
        M.BLEND_ALPHAPREMULT: (A, mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA),
        M.BLEND_ADD: (ADD, mgl.ONE, mgl.ONE),
        M.BLEND_ADDALPHA: (ADD, mgl.SRC_ALPHA, mgl.ONE),
        M.BLEND_SUB: (SUB, mgl.ONE, mgl.ONE),
        M.BLEND_SUBALPHA: (SUB, mgl.SRC_ALPHA, mgl.ONE),
        M.BLEND_MUL: (A, mgl.DST_COLOR, mgl.ZERO),
        M.BLEND_MUL2X: (A, mgl.DST_COLOR, mgl.SRC_COLOR),
        M.BLEND_ALPHAMUL: (A, mgl.ZERO, mgl.SRC_ALPHA),
        M.BLEND_IALPHAMUL: (A, mgl.ZERO, mgl.ONE_MINUS_SRC_ALPHA),
        M.BLEND_IALPHA: (A, mgl.ONE_MINUS_SRC_ALPHA, mgl.SRC_ALPHA),
        M.BLEND_SCREEN: (A, mgl.ONE, mgl.ONE_MINUS_SRC_COLOR),
    }


def gpu_available():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
        name = ctx.info.get("GL_RENDERER", "?")
        ctx.release()
        return name
    except Exception:
        return None


class GLRenderer:
    """Draws StageRenderer frames with OpenGL; returns rgb24 bytes (top row first)."""

    def __init__(self, stage, width, height):
        import moderngl
        self.mgl = moderngl
        self.stage = stage
        self.w, self.h = width, height
        self.ctx = moderngl.create_standalone_context()
        self.renderer_name = self.ctx.info.get("GL_RENDERER", "?")
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.qprog = self.ctx.program(vertex_shader=VERT, fragment_shader=QUAD_FRAG)
        for i in range(4):
            self.prog[f"tex{i}"].value = i
        self.qprog["tex0"].value = 0
        self.color_tex = self.ctx.texture((width, height), 4)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex])
        self.blends = _blend_table(moderngl)
        self.black = self.ctx.texture((1, 1), 4, bytes(4))
        self._tex = {}
        self._video = None
        self._coach = None
        self.vbo = self.ctx.buffer(reserve=1 << 20, dynamic=True)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "4f 2f 4f", "in_pos", "in_uv", "in_col")])
        self.qvao = self.ctx.vertex_array(self.qprog, [(self.vbo, "4f 2f 4f", "in_pos", "in_uv", "in_col")])
        cam = stage.camera
        self.k = cam.k
        self.cam = cam.pos
        self.clear = tuple(stage._clear) + (1.0,)

    # -- resources ------------------------------------------------------------------------
    def gl_texture(self, tex):
        if tex is None:
            return self.black
        if isinstance(tex, _VideoTex):
            return tex.gl
        key = id(tex)
        t = self._tex.get(key)
        if t is None:
            data = (np.clip(tex.rgba, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            t = self.ctx.texture((tex.w, tex.h), 4, np.ascontiguousarray(data).tobytes())
            t.repeat_x = t.repeat_y = False
            t.build_mipmaps()
            t.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            t.anisotropy = 8.0
            self._tex[key] = (t, tex)  # keep the source alive so id() stays unique
            return t
        return t[0]

    def _dynamic(self, attr, rgba_u8):
        h, w = rgba_u8.shape[:2]
        t = getattr(self, attr)
        if t is None or t.size != (w, h):
            if t is not None:
                t.release()
            t = self.ctx.texture((w, h), 4)
            t.repeat_x = t.repeat_y = False
            t.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            setattr(self, attr, t)
        t.write(np.ascontiguousarray(rgba_u8).tobytes())
        t.build_mipmaps()
        return t

    # -- geometry ---------------------------------------------------------------------------
    def _clip(self, x, y, z):
        """World -> clip coords (x*w, y*w, 0, w); y flipped so readback is top-row-first."""
        d = self.cam[2] - z
        f = self.k / d
        sx = self.w * 0.5 + (x - self.cam[0]) * f
        sy = self.h * 0.5 - (y - self.cam[1]) * f
        return (2.0 * sx / self.w - 1.0) * d, (2.0 * sy / self.h - 1.0) * d, 0.0, d

    # -- material uniforms ----------------------------------------------------------------------
    def _set_material(self, mat, textures, time, override=None):
        p = self.prog
        on, r0, r1, lcol, sel, sat, thr, wrap, bl = [], [], [], [], [], [], [], [], [0]
        if mat.mat_type == M.MAT_MULTI_TEXTURE:
            for i, layer in enumerate(mat.layers):
                on.append(1 if layer.enabled else 0)
                m = layer.uv_matrix(time)
                r0.append((float(m[0, 0]), float(m[0, 1]), float(m[0, 2])))
                r1.append((float(m[1, 0]), float(m[1, 1]), float(m[1, 2])))
                lcol.append(tuple(float(c) for c in layer.color))
                csat, asat, s = M.TEXTURE_USAGE.get(layer.usage, M.TEXTURE_USAGE[5])
                sel.append(tuple(float(v) for v in s))
                sat.append((float(csat), float(asat)))
                thr.append(float(layer.alpha_threshold))
                wrap.append((int(layer.wrap_u), int(layer.wrap_v)))
            bl += [int(b) for b in mat.blend_layers]
            texs = [override or textures[i] if mat.layers[i].usage != 0 else None for i in range(4)]
        else:
            t0 = override or textures[0]
            on = [1, 0, 0, 0]
            r0 = [(1.0, 0.0, 0.0)] * 4
            r1 = [(0.0, 1.0, 0.0)] * 4
            lcol = [(1.0, 1.0, 1.0, 1.0)] * 4
            sel = [(0.0, 0.0, 0.0, 1.0)] * 4
            sat = [(0.0, 0.0)] * 4
            thr = [-1.0] * 4
            w = (int(t0.wrap_u), int(t0.wrap_v)) if t0 is not None else (2, 2)
            wrap = [w] * 4
            bl = [0, 2, 2, 2]
            texs = [t0, None, None, None]
            if t0 is None:  # default material without texture draws white (CPU parity)
                sat = [(1.0, 1.0)] + [(0.0, 0.0)] * 3
        values = {"on": on, "r0": r0, "r1": r1, "lcol": lcol, "sel": sel, "sat": sat, "thr": thr,
                  "wrapm": wrap, "bl": bl}
        for name, v in values.items():
            if name in p:
                p[name].value = v
        for i in range(4):
            self.gl_texture(texs[i]).use(i) if texs[i] is not None else self.black.use(i)

    def _set_blend(self, blend):
        eq, src, dst = self.blends.get(blend, self.blends[M.BLEND_ALPHA])
        self.ctx.blend_equation = eq
        # keep destination alpha at 1 (only rgb is read back)
        self.ctx.blend_func = (src, dst, self.mgl.ZERO, self.mgl.ONE)

    def _draw(self, verts, vao=None):
        data = np.asarray(verts, np.float32)
        if data.size == 0:
            return
        raw = data.tobytes()
        if len(raw) > self.vbo.size:
            self.vbo.orphan(len(raw) * 2)
        else:
            self.vbo.orphan()
        self.vbo.write(raw)
        (vao or self.vao).render(self.mgl.TRIANGLES, vertices=len(data))

    # -- frame --------------------------------------------------------------------------------
    def render(self, s, video_rgba=None, coach_rgba=None, coach_rect=None, coach_alpha=1.0):
        """Render at song time s. video_rgba: raw (packed) coach frame for TexturePatcher meshes;
        coach_rgba: unpacked coach (uint8 RGBA) composited on top in coach_rect (normalized)."""
        st_all, draws = self.stage.frame(s)
        ctx = self.ctx
        self.fbo.use()
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.disable(self.mgl.DEPTH_TEST | self.mgl.CULL_FACE)
        ctx.enable(self.mgl.BLEND)
        self.fbo.clear(*self.clear)
        video_tex = None
        stage = self.stage
        for kind, a in draws:
            st = st_all[a]
            col = stage.actor_color(st)
            if kind == 0:
                (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = stage.quad_corners(a, st)
                x3, y3, z3 = x1 + x2 - x0, y1 + y2 - y0, z1 + z2 - z0
                if min(self.cam[2] - z for z in (z0, z1, z2, z3)) <= NEAR:
                    continue
                p0, p1, p2, p3 = (self._clip(x0, y0, z0), self._clip(x1, y1, z1),
                                  self._clip(x2, y2, z2), self._clip(x3, y3, z3))
                v = [(*p0, 0, 0, *col), (*p1, 1, 0, *col), (*p2, 0, 1, *col),
                     (*p1, 1, 0, *col), (*p3, 1, 1, *col), (*p2, 0, 1, *col)]
                mat = st.material
                self._set_material(mat, stage.textures_for(mat), s)
                self._set_blend(mat.blend)
                self._draw(v)
            elif kind == 1:
                override = None
                if a.mesh.video_texture:
                    if video_rgba is None:
                        continue
                    if video_tex is None:
                        video_tex = _VideoTex(self._dynamic("_video", video_rgba))
                    override = video_tex
                mw = stage.mesh_world(a, st)
                if mw is None:
                    continue
                world, mesh = mw
                d = self.cam[2] - world[:, 2]
                f = self.k / np.maximum(d, NEAR)
                sx = self.w * 0.5 + (world[:, 0] - self.cam[0]) * f
                sy = self.h * 0.5 - (world[:, 1] - self.cam[1]) * f
                clip = np.stack([(2 * sx / self.w - 1) * d, (2 * sy / self.h - 1) * d,
                                 np.zeros_like(d), d], axis=1)
                mats = st.mesh_materials or []
                for mat_id, vidx, uidx in mesh.triangles:
                    mat = mats[mat_id] if mat_id < len(mats) else (mats[0] if mats else None)
                    if mat is None or not mat.has_content:
                        continue
                    ok = (d[vidx] > NEAR).all(axis=1)
                    if not ok.any():
                        continue
                    vi = vidx[ok].reshape(-1)
                    ui = uidx[ok].reshape(-1)
                    verts = np.concatenate([clip[vi], mesh.uvs[ui],
                                            np.tile(np.asarray(col, np.float32), (len(vi), 1))], axis=1)
                    self._set_material(mat, stage.textures_for(mat), s, override)
                    self._set_blend(mat.blend)
                    self._draw(verts)
            else:
                groups = {}
                for e, (cx, cy, cz), size, ang, pc in stage.particles(a, st, s):
                    d = self.cam[2] - cz
                    if d <= NEAR:
                        continue
                    f = self.k / d
                    scx = self.w * 0.5 + (cx - self.cam[0]) * f
                    scy = self.h * 0.5 - (cy - self.cam[1]) * f
                    half = 0.5 * size * f
                    if half < 0.25:
                        continue
                    c, sn = math.cos(-ang), math.sin(-ang)
                    pts = []
                    for ox, oy, u, vv in ((-1, -1, 0, 0), (1, -1, 1, 0), (-1, 1, 0, 1), (1, 1, 1, 1)):
                        px = scx + (ox * c - oy * sn) * half
                        py = scy + (ox * sn + oy * c) * half
                        pts.append(((2 * px / self.w - 1) * d, (2 * py / self.h - 1) * d, 0.0, d, u, vv, *pc))
                    q = groups.setdefault(id(e), (e, []))[1]
                    q += [pts[0], pts[1], pts[2], pts[1], pts[3], pts[2]]
                for e, verts in groups.values():
                    self._set_material(e.mat, stage.textures_for(e.mat), 0.0)
                    self._set_blend(e.mat.blend)
                    self._draw(verts)

        if coach_rgba is not None and coach_alpha > 0.0:
            t = self._dynamic("_coach", coach_rgba)
            x, y, w, h = coach_rect or (0.0, 0.0, 1.0, 1.0)
            X0, Y0, X1, Y1 = 2 * x - 1, 2 * y - 1, 2 * (x + w) - 1, 2 * (y + h) - 1
            c = (1.0, 1.0, 1.0, float(coach_alpha))
            v = [(X0, Y0, 0, 1, 0, 0, *c), (X1, Y0, 0, 1, 1, 0, *c), (X0, Y1, 0, 1, 0, 1, *c),
                 (X1, Y0, 0, 1, 1, 0, *c), (X1, Y1, 0, 1, 1, 1, *c), (X0, Y1, 0, 1, 0, 1, *c)]
            t.use(0)
            self._set_blend(M.BLEND_ALPHA)
            self._draw(v, self.qvao)

        return self.fbo.read(components=3, alignment=1)

    def release(self):
        self.ctx.release()


class _VideoTex:
    """Stands in for an assets.Texture when the TexturePatcher feeds the video channel."""

    def __init__(self, gl_tex):
        self.gl = gl_tex
        self.wrap_u = self.wrap_v = 2
