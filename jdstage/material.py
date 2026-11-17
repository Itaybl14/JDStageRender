"""
Material model (engine/display/material/GFXMaterialShader.{h,cpp}).

A material is a GFXMaterialShader_Template (.msh) plus the instance's texture
set. Two shading paths are used by the generic stages:

  materialtype 0  (GFX_MAT_DEFAULT)        diffuse texture * vertex color
  materialtype 10 (GFX_MAT_MULTI_TEXTURE)  4 layers, each: texture slot
                                           (diffuse, diffuse_2, diffuse_3,
                                           diffuse_4), a chain of UVModifiers,
                                           a TextureUsage, a DiffuseColor and a
                                           BlendLayerN op against the previous
                                           result.

Layer math is taken from the shipped DX11 MultiTexture.fx bytecode
(disassembled with d3dcompiler): see raster.shade_multitexture.
"""

import copy
import math

import numpy as np

# GFX_BLENDMODE
BLEND_COPY, BLEND_ALPHA, BLEND_ALPHAPREMULT = 1, 2, 3
BLEND_ADD, BLEND_ADDALPHA, BLEND_SUBALPHA, BLEND_SUB = 6, 7, 8, 9
BLEND_MUL, BLEND_ALPHAMUL, BLEND_IALPHAMUL, BLEND_IALPHA = 10, 11, 12, 13
BLEND_MUL2X, BLEND_SCREEN = 17, 21

MAT_DEFAULT, MAT_MULTI_TEXTURE = 0, 10

# GFX_TEXTURE_USAGE -> (color saturator, alpha saturator, alpha selector RGBA)
# (GFXAdapter_Common_impl.h setMultiTextureLayerParameters; Color(a,r,g,b) ctor)
TEXTURE_USAGE = {
    0: (1.0, 1.0, (0, 0, 0, 0)),  # NO_TEXTURE
    1: (1.0, 0.0, (1, 0, 0, 0)),  # ALPHA_IS_RED
    2: (1.0, 0.0, (0, 1, 0, 0)),  # ALPHA_IS_GREEN
    3: (1.0, 0.0, (0, 0, 1, 0)),  # ALPHA_IS_BLUE
    4: (1.0, 0.0, (0, 0, 0, 1)),  # ALPHA_IS_ALPHA
    5: (0.0, 0.0, (0, 0, 0, 1)),  # ENTIRE_TEXTURE
    6: (0.0, 1.0, (0, 0, 0, 0)),  # RGB_ONLY
}

TEXTURE_SLOTS = ("diffuse", "diffuse_2", "diffuse_3", "diffuse_4")


def argb_to_rgba(c):
    """JSON colors are serialized as [a, r, g, b] (ITF Color ctor order)."""
    if not c or len(c) < 4:
        return (1.0, 1.0, 1.0, 1.0)
    return (float(c[1]), float(c[2]), float(c[3]), float(c[0]))


class UVModifier:
    __slots__ = ("tu", "tv", "anim_u", "anim_v", "rot", "rot_ou", "rot_ov", "anim_rot",
                 "su", "sv", "so_u", "so_v")

    def __init__(self, d):
        self.tu = float(d.get("TranslationU", 0))
        self.tv = float(d.get("TranslationV", 0))
        self.anim_u = bool(d.get("AnimTranslationU", 0))
        self.anim_v = bool(d.get("AnimTranslationV", 0))
        self.rot = float(d.get("Rotation", 0))
        self.rot_ou = float(d.get("RotationOffsetU", 0.5))
        self.rot_ov = float(d.get("RotationOffsetV", 0.5))
        self.anim_rot = bool(d.get("AnimRotation", 0))
        self.su = float(d.get("ScaleU", 1))
        self.sv = float(d.get("ScaleV", 1))
        self.so_u = float(d.get("ScaleOffsetU", 0.5))
        self.so_v = float(d.get("ScaleOffsetV", 0.5))

    def matrix(self, time):
        """2x3 affine on (u, v, 1), UVModifier::computeUVTransform."""
        a = self.rot * time if self.anim_rot else self.rot
        c, s = math.cos(a), math.sin(a)
        # rotation about pivot: u' = u c - v s + tx ; v' = u s + v c + ty
        tx = self.rot_ou * (1 - c) + self.rot_ov * s
        ty = self.rot_ov * (1 - c) - self.rot_ou * s
        rot = np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], np.float64)
        sc = np.array([[self.su, 0, self.so_u * (1 - self.su)],
                       [0, self.sv, self.so_v * (1 - self.sv)],
                       [0, 0, 1]], np.float64)
        m = sc @ rot
        m[0, 2] += self.tu * time if self.anim_u else self.tu
        m[1, 2] += self.tv * time if self.anim_v else self.tv
        return m


class Layer:
    __slots__ = ("enabled", "color", "usage", "wrap_u", "wrap_v", "filtering", "alpha_threshold", "uv_mods")

    def __init__(self, d):
        self.enabled = bool(d.get("Enabled", 1))
        self.color = list(argb_to_rgba(d.get("DiffuseColor", [1, 1, 1, 1])))
        self.usage = int(d.get("TextureUsage", 5))
        self.wrap_u = int(d.get("TexAdressingModeU", 0))
        self.wrap_v = int(d.get("TexAdressingModeV", 0))
        self.filtering = int(d.get("Filtering", 1))
        self.alpha_threshold = float(d.get("AlphaThreshold", -1))
        self.uv_mods = [UVModifier(m) for m in (d.get("UVModifiers") or [])]

    def uv_matrix(self, time):
        m = np.eye(3)
        for mod in self.uv_mods:  # dst = dst * mod (row vectors): mod0 applied first
            m = mod.matrix(time) @ m
        return m


class Material:
    """Mutable per-actor material state (tapes write layer colors / UV params)."""

    def __init__(self, shader, textures, fallback_blend=BLEND_ALPHA):
        shader = shader or {}
        self.mat_type = int(shader.get("materialtype", MAT_DEFAULT)) if shader else MAT_DEFAULT
        self.blend = int(shader.get("blendmode", fallback_blend)) if shader else fallback_blend
        if self.blend == 0:
            self.blend = fallback_blend
        self.textures = [textures.get(s, "") for s in TEXTURE_SLOTS]
        self.layers = [Layer(shader.get(f"Layer{i}", {"Enabled": 1 if i == 1 else 0})) for i in range(1, 5)]
        self.blend_layers = [int(shader.get(f"BlendLayer{i}", BLEND_ALPHA)) for i in range(2, 5)]
        self.alpha_test = bool(shader.get("useAlphaTest", 0))
        self.alpha_ref = float(shader.get("alphaRef", 128)) / 255.0

    def clone(self):
        return copy.deepcopy(self)

    @property
    def has_content(self):
        if self.mat_type == MAT_MULTI_TEXTURE:
            return any(l.enabled for l in self.layers)
        return bool(self.textures[0])  # default material without a texture draws nothing


def material_from_serializable(store, mat_xml_or_json, fallback_blend=BLEND_ALPHA):
    """Build a Material from a GFXMaterialSerializable (dict with shaderPath + textureSet)."""
    shader_path = mat_xml_or_json.get("shaderPath", "") if mat_xml_or_json else ""
    textures = dict(mat_xml_or_json.get("textureSet", {})) if mat_xml_or_json else {}
    shader = None
    if shader_path:
        try:
            shader = store.data(shader_path)
        except FileNotFoundError:
            shader = None
    return Material(shader, textures, fallback_blend)
