"""
Software rasterizer reproducing the engine's 2D/3D stage rendering.

Camera: JD_FixedCameraComponent (world/_common/camera/fixedcamera.tpl:
focale = 25 deg, used as the vertical FOV in GFX_computeMatrixPerspectiveFovRH)
placed at (0, 0, 10) looking down -Z, which is what the generic stage scenes
are authored against (their "Camera_JD_Dummy" sits at z=10).

Multi-texture shading is a transcription of MultiTexture.fx as shipped in
data/enginedata/Shaders/Compiled/WIN32/MultiTexture.fx_dx11.fxb:
  per layer i:  rgb_i = saturate(tex.rgb + colorSat_i)
                a_i   = dot(tex.rgba, alphaSelector_i) + alphaSat_i
                a_i   = (thr_i >= 0) ? (a_i > thr_i) : a_i
                L_i   = (rgb_i, a_i) * DiffuseColor_i
  first enabled layer is the base (premultiplied when it isn't layer 0),
  each following layer j combines with BlendLayer_j:
                ALPHA     res = res * (1 - a_j) + (rgb_j * a_j, a_j)
                ADD       res = res + L_j
                ADDALPHA  res = res + (rgb_j * a_j, a_j)
                MUL       res = res * L_j
                ALPHAMUL  res = res * a_j
  out = res * vertexColor ; discard if out.a < 1/255
"""

import math

import cv2
import numpy as np

from .assets import WRAP, MIRROR, CLAMP, BORDER
from . import material as M

FOV_Y = math.radians(25.0)
CAM_POS = (0.0, 0.0, 10.0)
NEAR = 0.05

_BORDER = {WRAP: cv2.BORDER_WRAP, MIRROR: cv2.BORDER_REFLECT, CLAMP: cv2.BORDER_REPLICATE,
           BORDER: cv2.BORDER_CONSTANT}


class Camera:
    def __init__(self, width, height, pos=CAM_POS, fov_y=FOV_Y):
        self.w = width
        self.h = height
        self.pos = pos
        self.k = (height * 0.5) / math.tan(fov_y * 0.5)

    def project(self, x, y, z):
        """World -> (sx, sy, depth). Works on scalars or numpy arrays."""
        d = self.pos[2] - z
        f = self.k / np.maximum(d, NEAR) if isinstance(d, np.ndarray) else self.k / max(d, NEAR)
        sx = self.w * 0.5 + (x - self.pos[0]) * f
        sy = self.h * 0.5 - (y - self.pos[1]) * f
        return sx, sy, d


# ----------------------------------------------------------------------------------------
# Texture sampling
# ----------------------------------------------------------------------------------------

def sample(tex_rgba, u, v, wrap_u, wrap_v, level_hint=1.0, tex=None):
    """Bilinear sample of a float32 RGBA texture at normalized (u, v) arrays."""
    img = tex_rgba
    if tex is not None and level_hint > 1.5:
        lvl = int(min(12, math.floor(math.log2(level_hint))))
        img = tex.mip(lvl)
    h, w = img.shape[:2]
    x = u * np.float32(w)
    x -= 0.5
    y = v * np.float32(h)
    y -= 0.5
    x = _wrap_coords(x, w, wrap_u)
    y = _wrap_coords(y, h, wrap_v)
    border = _BORDER.get(wrap_u if wrap_u == wrap_v else CLAMP, cv2.BORDER_REPLICATE)
    if wrap_u in (WRAP, MIRROR) or wrap_v in (WRAP, MIRROR):
        border = cv2.BORDER_WRAP if (wrap_u == WRAP or wrap_v == WRAP) else cv2.BORDER_REFLECT
    out = cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=border, borderValue=(0, 0, 0, 0))
    if out.ndim == 2:
        out = out[..., None]
    return out


def _wrap_coords(c, n, mode):
    """Fold texel coordinates into range (float32, in place where possible)."""
    if mode == WRAP:
        lo, hi = float(c.min()), float(c.max())
        if lo >= -0.5 and hi < n - 0.5:
            return c
        c = c + 0.5
        c -= np.floor(c * (1.0 / n)) * n
        c -= 0.5
        return c
    if mode == MIRROR:
        period = 2 * n
        m = c + 0.5
        m -= np.floor(m * (1.0 / period)) * period
        np.subtract(period, m, out=m, where=m >= n)
        m -= 0.5
        return m
    if mode == CLAMP:
        return np.clip(c, 0, n - 1, out=c)
    return c


# ----------------------------------------------------------------------------------------
# Shading
# ----------------------------------------------------------------------------------------

def _texture_footprint(jac, w, h):
    """Approx texels per screen pixel from d(uv)/d(screen) Jacobian."""
    du = math.hypot(jac[0][0], jac[0][1]) * w
    dv = math.hypot(jac[1][0], jac[1][1]) * h
    return max(du, dv)


def shade(material, textures, U, V, time, uv_jac=None, override_tex=None):
    """Return premultiplied-agnostic RGBA float32 (H,W,4) following the engine shader.

    textures: list of 4 Texture|None for slots diffuse..diffuse_4.
    uv_jac  : 2x2 d(uv)/d(screen) for mip selection.
    """
    if override_tex is not None:
        textures = [override_tex] * 4
    if material.mat_type == M.MAT_MULTI_TEXTURE:
        return _shade_multi(material, textures, U, V, time, uv_jac)
    tex = textures[0]
    if tex is None:
        out = np.ones(U.shape + (4,), np.float32)
    else:
        rho = _texture_footprint(uv_jac, tex.w, tex.h) if uv_jac is not None else 1.0
        out = sample(tex.rgba, U, V, tex.wrap_u, tex.wrap_v, rho, tex)
    return out


def _affine(U, V, a, b, c):
    """a*U + b*V + c in float32 with the fewest temporaries (identity is free)."""
    if a == 1.0 and b == 0.0 and c == 0.0:
        return U
    if b == 0.0:
        out = U * np.float32(a)
    else:
        out = cv2.addWeighted(U, a, V, b, 0.0)
    if c:
        out += np.float32(c)
    return out


def _shade_multi(material, textures, U, V, time, uv_jac):
    res = None
    for i, layer in enumerate(material.layers):
        if not layer.enabled:
            continue
        csat, asat, sel = M.TEXTURE_USAGE.get(layer.usage, M.TEXTURE_USAGE[5])
        m = layer.uv_matrix(time)
        Ul = _affine(U, V, float(m[0, 0]), float(m[0, 1]), float(m[0, 2]))
        Vl = _affine(U, V, float(m[1, 0]), float(m[1, 1]), float(m[1, 2]))
        tex = textures[i]
        if layer.usage == 0:
            texel = None
        elif tex is None:
            texel = np.zeros(U.shape + (4,), np.float32)  # NULL texture binding samples 0
        else:
            rho = 1.0
            if uv_jac is not None:
                j = m[:2, :2] @ np.asarray(uv_jac)
                rho = _texture_footprint(j, tex.w, tex.h)
            texel = sample(tex.rgba, Ul, Vl, layer.wrap_u, layer.wrap_v, rho, tex)
        if texel is None:
            rgb = np.ones(U.shape + (3,), np.float32)
            a = np.full(U.shape, asat, np.float32) if asat else np.zeros(U.shape, np.float32)
        else:
            if csat:
                rgb = np.ones(U.shape + (3,), np.float32)  # saturate(tex + 1) == 1
            else:
                rgb = texel[..., :3]
            if sum(sel) == 1:
                a = texel[..., sel.index(1)].copy()
            else:
                a = texel @ np.asarray(sel, np.float32)
            if asat:
                a += np.float32(asat)
        if layer.alpha_threshold >= 0:
            a = (a > layer.alpha_threshold).astype(np.float32)
        col = layer.color
        rgb = rgb * np.array(col[:3], np.float32)
        a = a * col[3]
        if res is None:
            if i > 0:
                rgb = rgb * a[..., None]
            res = np.concatenate([rgb, a[..., None]], axis=-1)
            continue
        op = material.blend_layers[i - 1]
        if op == M.BLEND_ADD:
            res[..., :3] += rgb
            res[..., 3] += a
        elif op == M.BLEND_ADDALPHA:
            res[..., :3] += rgb * a[..., None]
            res[..., 3] += a
        elif op == M.BLEND_MUL:
            res[..., :3] *= rgb
            res[..., 3] *= a
        elif op == M.BLEND_ALPHAMUL:
            res *= a[..., None]
        else:  # ALPHA (default)
            inv = (1.0 - a)[..., None]
            res *= inv
            res[..., :3] += rgb * a[..., None]
            res[..., 3] += a
    if res is None:
        return None
    return res


# ----------------------------------------------------------------------------------------
# Framebuffer blending
# ----------------------------------------------------------------------------------------

def blend_into(fb, x0, y0, src, mask, blend):
    """Blend src RGBA (h,w,4) with coverage mask into fb[y0:y0+h, x0:x0+w] (RGB float32)."""
    h, w = src.shape[:2]
    dst = fb[y0:y0 + h, x0:x0 + w]
    keep = src[..., 3] >= np.float32(1.0 / 255.0)  # shader discard
    if mask is not None:
        keep &= mask
    if not keep.any():
        return
    full = bool(keep.all())
    if full:
        a = np.clip(src[..., 3:4], 0.0, 1.0)
        rgb = src[..., :3]
    else:
        km = keep[..., None]
        a = np.where(km, np.clip(src[..., 3:4], 0.0, 1.0), np.float32(0))
        rgb = np.where(km, src[..., :3], np.float32(0))
    one = np.float32(1.0)

    def masked(new):  # ops that would alter discarded pixels
        return new if full else np.where(keep[..., None], new, dst)

    if blend == M.BLEND_COPY:
        out = masked(rgb.copy())
    elif blend == M.BLEND_ALPHAPREMULT:
        out = rgb + dst * (one - a)
    elif blend == M.BLEND_ADD:
        out = dst + rgb
    elif blend == M.BLEND_ADDALPHA:
        out = rgb * a
        out += dst
    elif blend == M.BLEND_SUB:
        out = dst - rgb
    elif blend == M.BLEND_SUBALPHA:
        out = dst - rgb * a
    elif blend == M.BLEND_MUL:
        out = masked(dst * rgb)
    elif blend == M.BLEND_MUL2X:
        out = masked(dst * rgb * np.float32(2.0))
    elif blend == M.BLEND_ALPHAMUL:
        out = masked(dst * a)
    elif blend == M.BLEND_IALPHAMUL:
        out = dst * (one - a)
    elif blend == M.BLEND_IALPHA:
        out = masked(rgb * (one - a) + dst * a)
    elif blend == M.BLEND_SCREEN:
        out = rgb + dst * (one - rgb)
    else:  # ALPHA
        out = dst - rgb
        out *= (one - a)
        out += rgb
        # rgb*a + dst*(1-a) == rgb + (dst-rgb)*(1-a)
    np.clip(out, 0.0, 1.0, out=out)
    fb[y0:y0 + h, x0:x0 + w] = out


# ----------------------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------------------

def _clip_bbox(xs, ys, clip):
    """Integer pixel bbox of the points, intersected with clip=(x0, y0, x1, y1)."""
    x0 = max(clip[0], int(math.floor(min(xs))))
    y0 = max(clip[1], int(math.floor(min(ys))))
    x1 = min(clip[2], int(math.ceil(max(xs))) + 1)
    y1 = min(clip[3], int(math.ceil(max(ys))) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def draw_quad(fb, cam, corners_world, material, textures, color, time, override_tex=None, uv_rect=(0, 0, 1, 1),
              clip=None):
    """Draw a planar quad. corners_world: TL, TR, BL (x,y,z) mapping to uv (0,0),(1,0),(0,1)."""
    (x0w, y0w, z0w), (x1w, y1w, z1w), (x2w, y2w, z2w) = corners_world
    p0 = cam.project(x0w, y0w, z0w)
    p1 = cam.project(x1w, y1w, z1w)
    p2 = cam.project(x2w, y2w, z2w)
    if min(p0[2], p1[2], p2[2]) <= NEAR:
        return
    p3 = (p1[0] + p2[0] - p0[0], p1[1] + p2[1] - p0[1])
    bb = _clip_bbox((p0[0], p1[0], p2[0], p3[0]), (p0[1], p1[1], p2[1], p3[1]), clip or (0, 0, cam.w, cam.h))
    if bb is None:
        return
    bx0, by0, bx1, by1 = bb
    ex = (p1[0] - p0[0], p1[1] - p0[1])
    ey = (p2[0] - p0[0], p2[1] - p0[1])
    det = ex[0] * ey[1] - ex[1] * ey[0]
    if abs(det) < 1e-9:
        return
    inv = ((ey[1] / det, -ey[0] / det), (-ex[1] / det, ex[0] / det))
    xs = np.arange(bx0, bx1, dtype=np.float32) + 0.5 - p0[0]
    ys = np.arange(by0, by1, dtype=np.float32) + 0.5 - p0[1]
    X, Y = np.meshgrid(xs, ys)
    U = inv[0][0] * X + inv[0][1] * Y
    V = inv[1][0] * X + inv[1][1] * Y
    mask = (U >= 0) & (U <= 1) & (V >= 0) & (V <= 1)
    if not mask.any():
        return
    u0, v0, u1, v1 = uv_rect
    Us = (u0 + U * (u1 - u0)).astype(np.float32)
    Vs = (v0 + V * (v1 - v0)).astype(np.float32)
    jac = ((inv[0][0] * (u1 - u0), inv[0][1] * (u1 - u0)), (inv[1][0] * (v1 - v0), inv[1][1] * (v1 - v0)))
    src = shade(material, textures, Us, Vs, time, jac, override_tex)
    if src is None:
        return
    src *= np.asarray(color, np.float32)
    blend_into(fb, bx0, by0, src, mask, material.blend)


def draw_mesh(fb, cam, verts_world, uvs, elements, materials, textures_per_mat, color, time, override_tex=None,
              clip=None):
    """Perspective-correct rasterization of a textured triangle mesh."""
    sx, sy, d = cam.project(verts_world[:, 0], verts_world[:, 1], verts_world[:, 2])
    if np.all(d <= NEAR):
        return
    clip = clip or (0, 0, cam.w, cam.h)
    for mat_id, vidx, uidx in elements:
        mat = materials[mat_id] if mat_id < len(materials) else (materials[0] if materials else None)
        if mat is None or not mat.has_content:
            continue
        texs = textures_per_mat[mat_id] if mat_id < len(textures_per_mat) else textures_per_mat[0]
        bb = _clip_bbox(sx[vidx].ravel(), sy[vidx].ravel(), clip)
        if bb is None:
            continue
        bx0, by0, bx1, by1 = bb
        H, W = by1 - by0, bx1 - bx0
        # In screen space u/w, v/w and 1/w are affine per triangle: f = A*x + B*y + C.
        # Solve the plane coefficients per triangle, then let OpenCV's polygon fill write
        # them into coefficient maps (C speed, 4 bits subpixel) - no per-pixel gathers.
        valid = (d[vidx] > NEAR).all(axis=1)
        tsx, tsy = sx[vidx], sy[vidx]  # cull triangles outside this clip region
        valid &= (tsx.max(axis=1) >= bx0) & (tsx.min(axis=1) <= bx1) & (tsy.max(axis=1) >= by0) & (tsy.min(axis=1) <= by1)
        if not valid.any():
            continue
        tv = vidx[valid]
        tu = uidx[valid]
        px_ = (sx[tv] - bx0)                                  # (T,3)
        py_ = (sy[tv] - by0)
        iz = 1.0 / d[tv]
        uv = uvs[tu]                                          # (T,3,2)
        Mtx = np.stack([px_ + 0.0, py_, np.ones_like(px_)], axis=-1)  # (T,3,3) rows [x y 1]
        det = np.linalg.det(Mtx)
        ok = np.abs(det) > 1e-9
        if not ok.any():
            continue
        Mtx, iz, uv, px_, py_ = Mtx[ok], iz[ok], uv[ok], px_[ok], py_[ok]
        F = np.stack([uv[..., 0] * iz, uv[..., 1] * iz, iz], axis=-1)  # (T,3 verts,3 funcs)
        coef = np.linalg.solve(Mtx, F)                        # (T, [A,B,C], funcs)
        maps = [np.zeros((H, W, 3), np.float32) for _ in range(3)]  # per func: A,B,C
        cover_u8 = np.zeros((H, W), np.uint8)
        fixed = np.round((np.stack([px_, py_], axis=-1) - 0.5) * 16).astype(np.int32)
        for t in range(len(fixed)):
            poly = fixed[t]
            c = coef[t]
            for fi in range(3):
                cv2.fillConvexPoly(maps[fi], poly, (float(c[0, fi]), float(c[1, fi]), float(c[2, fi])),
                                   lineType=cv2.LINE_8, shift=4)
            cv2.fillConvexPoly(cover_u8, poly, 1, lineType=cv2.LINE_8, shift=4)
        cover = cover_u8.astype(bool)
        if not cover.any():
            continue
        X = (np.arange(W, dtype=np.float32) + 0.5)[None, :]
        Y = (np.arange(H, dtype=np.float32) + 0.5)[:, None]

        def ev(m):
            out = m[..., 0] * X
            out += m[..., 1] * Y
            out += m[..., 2]
            return out
        inv_w = ev(maps[2])
        inv_w[~cover] = 1.0
        Ubuf = ev(maps[0]) / inv_w
        Vbuf = ev(maps[1]) / inv_w
        src = shade(mat, texs, Ubuf, Vbuf, time, None, override_tex)
        if src is None:
            continue
        src *= np.asarray(color, np.float32)
        blend_into(fb, bx0, by0, src, cover, mat.blend)


def draw_sprite(fb, cam, sprite, center_world, size_world, angle, color, blend, clip=None):
    """Particle: pre-shaded RGBA sprite drawn as a camera-facing square."""
    cx, cy, d = cam.project(*center_world)
    if d <= NEAR:
        return
    half = 0.5 * size_world * cam.k / d
    if half < 0.25:
        return
    sh, sw = sprite.shape[:2]
    rad = half * 1.4143
    bb = _clip_bbox((cx - rad, cx + rad), (cy - rad, cy + rad), clip or (0, 0, cam.w, cam.h))
    if bb is None:
        return
    x0, y0, x1, y1 = bb
    c, s = math.cos(-angle), math.sin(-angle)
    sxs = 2 * half / sw
    sys_ = 2 * half / sh
    # sprite pixel (i,j) -> screen: rotate about centre
    A = np.array([[c * sxs, -s * sys_, 0], [s * sxs, c * sys_, 0]], np.float64)
    off = A[:, :2] @ np.array([sw * 0.5, sh * 0.5])
    A[0, 2] = cx - off[0] - x0
    A[1, 2] = cy - off[1] - y0
    W, H = x1 - x0, y1 - y0
    warped = cv2.warpAffine(sprite, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    if warped.ndim == 2:
        return
    warped = warped * np.asarray(color, np.float32)
    mask = warped[..., 3] > 0
    blend_into(fb, x0, y0, warped, mask, blend)
