"""
Scene graph: parses cooked .isc (XML) scenes into actors with the components
the generic stages use, and computes per-frame world transforms.

Transform rules (engine sources):
  * Actors store POS2D / RELATIVEZ / ANGLE (degrees) / SCALE relative to their
    owning SubSceneActor.
  * A <parentBind> makes the actor follow another actor
    (engine/actors/ActorBind.cpp, BindHandler::computeWorldCoordinates):
        world_pos   = parent_pos + rotate(offsetPos * parent_scale, parent_angle)
        world_angle = parent_angle + offsetAngle
        scale       = parent_scale * localScale   (scaleInheritProp Combine=2)
                    | parent_scale                (UseParent=0)
                    | own scale                   (UseChild=1)
  * MaterialGraphicComponent draws a 2x2-unit quad (buildWithoutAtlas /
    buildMeshFromAtlasRectangle) scaled by the actor scale, offset by its
    anchor (computeMiddleCenterOffset).
"""

import math
import xml.etree.ElementTree as ET

from .material import Material, material_from_serializable, BLEND_ALPHA

DEG = math.pi / 180.0


def _vec(s, n, default=0.0):
    if s is None:
        return [default] * n
    parts = s.replace(",", " ").split()
    vals = [float(p) for p in parts[:n]]
    return vals + [default] * (n - len(vals))


def _enum(el, name, default=0):
    for e in el.findall("ENUM"):
        if e.get("NAME") == name:
            try:
                return int(e.get("SEL", default))
            except ValueError:
                return default
    return default


def _xml_material(el):
    """GFXMaterialSerializable XML element -> dict(shaderPath, textureSet)."""
    if el is None:
        return None
    ts = el.find("textureSet/GFXMaterialTexturePathSet")
    return {"shaderPath": el.get("shaderPath", ""),
            "textureSet": dict(ts.attrib) if ts is not None else {}}


def _tpl_component(tpl, cls):
    for c in (tpl or {}).get("COMPONENTS", []) or []:
        if c.get("__class") == cls:
            return c
    return None


class Bind:
    def __init__(self, el):
        self.parent_path = el.get("parentPath", "")
        self.offset_pos = _vec(el.get("offsetPos"), 3)
        self.offset_angle = float(el.get("offsetAngle", 0)) * DEG
        self.local_scale = _vec(el.get("localScale"), 2, 1.0)
        self.scale_inherit = _enum(el, "scaleInheritProp", 2)
        self.use_parent_flip = el.get("useParentFlip", "0") == "1"
        self.use_parent_alpha = el.get("useParentAlpha", "0") == "1"
        self.use_parent_color = el.get("useParentColor", "0") == "1"


class GraphicComp:
    """MaterialGraphicComponent (quad)."""

    def __init__(self, material, color_factor, anchor, custom_anchor, pos_offset):
        self.material = material
        self.color_factor = color_factor
        self.anchor = anchor
        self.custom_anchor = custom_anchor
        self.pos_offset = pos_offset


class MeshComp:
    def __init__(self, mesh_path, materials, color_factor, scale_z, orientation, video_texture=False):
        self.mesh_path = mesh_path
        self.materials = materials
        self.color_factor = color_factor
        self.scale_z = scale_z
        self.orientation = orientation
        self.video_texture = video_texture


class FxComp:
    def __init__(self, descriptors, controls, default_fx):
        self.descriptors = descriptors   # name -> FxDescriptor_Template dict
        self.controls = controls         # control name -> [descriptor names]
        self.default_fx = default_fx


class Actor:
    def __init__(self):
        self.name = ""
        self.path = ""          # "Sub|Sub2|name"
        self.order = 0
        self.parent_scene = None  # owning SubSceneActor (or None at root)
        self.children = []        # for SubSceneActor
        self.is_subscene = False
        self.enabled = True
        self.template_path = ""
        self.template = None
        # initial (serialized) state
        self.init_pos = [0.0, 0.0, 0.0]
        self.init_angle = 0.0
        self.init_scale = [1.0, 1.0]
        self.flipped = False
        self.bind = None
        self.bind_parent = None
        # components
        self.graphic = None
        self.mesh = None
        self.fx = None
        self.tapes = {}        # label(lower) -> (path, group)
        self.clear_color = None
        self.video_patch = False
        self.init_alpha = 1.0  # stage options may start an actor hidden (tapes can still show it)

    def __repr__(self):
        return f"<Actor {self.path}>"

    @property
    def bound(self):
        """True when the actor follows a bind parent that exists in the scene. When the
        parent is missing the engine leaves the actor at its own serialized transform."""
        return self.bind is not None and self.bind_parent is not None

    def initial_local(self):
        """(pos, scale, angle) in the space tapes write to (bind-local when bound)."""
        if self.bound:
            sc = self.bind.local_scale if self.bind.scale_inherit == 2 else self.init_scale
            return self.bind.offset_pos, sc, self.bind.offset_angle
        return self.init_pos, self.init_scale, self.init_angle


class ActorState:
    """Per-frame mutable state written by tapes, then resolved to world space."""
    __slots__ = ("pos", "angle", "scale", "color", "alpha", "enabled", "material", "mesh_materials",
                 "world_pos", "world_angle", "world_scale", "world_color", "world_alpha", "world_flip", "resolved")

    def __init__(self, actor):
        pos, scale, angle = actor.initial_local()
        self.pos = list(pos)
        self.angle = angle
        self.scale = list(scale)
        # The component's GFXPrimitiveParam colorFactor *is* the actor colour: tapes
        # replace it (GraphicComponent::setColor / setCurrentAlpha), they don't multiply.
        cf = (actor.graphic.color_factor if actor.graphic is not None
              else actor.mesh.color_factor if actor.mesh is not None else (1.0, 1.0, 1.0, 1.0))
        self.color = [cf[0], cf[1], cf[2]]
        self.alpha = cf[3] * actor.init_alpha
        self.enabled = actor.enabled
        self.material = actor.graphic.material.clone() if actor.graphic else None
        self.mesh_materials = [m.clone() for m in actor.mesh.materials] if actor.mesh else None
        self.resolved = False


class Scene:
    def __init__(self, store):
        self.store = store
        self.actors = []
        self.by_path = {}
        self.by_name = {}

    # -- loading ------------------------------------------------------------------------
    @classmethod
    def load(cls, store, isc_path):
        scene = cls(store)
        root = ET.fromstring(store.read_raw(isc_path).rstrip(b"\0"))
        for sc in root:
            if sc.tag == "Scene":
                scene._parse_scene(sc, None, "")
        scene._link_binds()
        return scene

    def _parse_scene(self, scene_el, owner, prefix):
        for group in scene_el.findall("ACTORS"):
            for el in group:
                self._parse_actor(el, owner, prefix)

    def _parse_actor(self, el, owner, prefix):
        a = Actor()
        a.name = el.get("USERFRIENDLY", "")
        a.path = (prefix + "|" + a.name) if prefix else a.name
        a.order = len(self.actors)
        a.parent_scene = owner
        a.enabled = el.get("isEnabled", "1") == "1"
        a.template_path = el.get("LUA", "")
        a.init_pos = _vec(el.get("POS2D"), 2) + [float(el.get("RELATIVEZ", 0))]
        a.init_angle = float(el.get("ANGLE", 0)) * DEG
        a.init_scale = _vec(el.get("SCALE"), 2, 1.0)
        a.flipped = el.get("xFLIPPED", "0") == "1"
        pb = el.find("parentBind/Bind")
        if pb is not None:
            a.bind = Bind(pb)
        if a.template_path:
            try:
                a.template = self.store.data(a.template_path)
            except (FileNotFoundError, ValueError):
                a.template = None
        self.actors.append(a)
        self.by_path[a.path.lower()] = a
        self.by_name.setdefault(a.name.lower(), []).append(a)
        if owner is not None:
            owner.children.append(a)

        self._parse_components(a, el)

        if el.tag == "SubSceneActor":
            a.is_subscene = True
            sub = el.find("SCENE/Scene")
            if sub is not None:
                self._parse_scene(sub, a, a.path)
            else:
                rel = el.get("RELATIVEPATH", "")
                if rel:
                    try:
                        import xml.etree.ElementTree as _ET
                        r = _ET.fromstring(self.store.read_raw(rel).rstrip(b"\0"))
                        for sc in r:
                            if sc.tag == "Scene":
                                self._parse_scene(sc, a, a.path)
                    except Exception:
                        pass

    def _parse_components(self, a, el):
        tpl = a.template
        for comp_group in el.findall("COMPONENTS"):
            for c in comp_group:
                tag = c.tag
                if tag in ("MaterialGraphicComponent", "TextureGraphicComponent"):
                    t = _tpl_component(tpl, tag + "_Template") or {}
                    mat = _xml_material(c.find("material/GFXMaterialSerializable"))
                    if (not mat or (not mat["shaderPath"] and not any(mat["textureSet"].values()))) and t.get("material"):
                        mat = t["material"]
                    tpl_blend = int(t.get("blendmode", BLEND_ALPHA)) or BLEND_ALPHA
                    material = material_from_serializable(self.store, mat, tpl_blend)
                    pp = c.find("PrimitiveParameters/GFXPrimitiveParam")
                    cf = _vec(pp.get("colorFactor") if pp is not None else None, 4, 1.0)
                    a.graphic = GraphicComp(material, cf, _enum(c, "anchor", 1),
                                            _vec(c.get("customAnchor"), 2),
                                            t.get("posOffset", [0, 0]))
                elif tag == "Mesh3DComponent":
                    mats = [material_from_serializable(self.store, _xml_material(m))
                            for m in c.findall("materialList/GFXMaterialSerializable")]
                    pp = c.find("PrimitiveParameters/GFXPrimitiveParam")
                    cf = _vec(pp.get("colorFactor") if pp is not None else None, 4, 1.0)
                    orient = _vec(c.get("orientation"), 16)
                    a.mesh = MeshComp(c.get("mesh3D", ""), mats, cf, float(c.get("ScaleZ", 1)), orient)
                elif tag == "ClearColorComponent":
                    a.clear_color = _vec(c.get("clearColor"), 4, 1.0)
                elif tag == "TexturePatcherComponent":
                    a.video_patch = True
        if a.mesh is not None and a.video_patch:
            a.mesh.video_texture = True

        # Template-only components
        tc = _tpl_component(tpl, "TapeCase_Template")
        if tc:
            for gi, group in enumerate(tc.get("TapesRack", []) or []):
                for e in group.get("Entries", []) or []:
                    a.tapes[str(e.get("Label", "")).lower()] = (e.get("Path", ""), gi)
        fb = _tpl_component(tpl, "FxBankComponent_Template")
        fc = _tpl_component(tpl, "FXControllerComponent_Template")
        if fb:
            descs = {d.get("name", ""): d for d in fb.get("Fx", []) or []}
            controls = {}
            default_fx = ""
            if fc:
                for ctl in fc.get("fxControlList", []) or []:
                    controls[ctl.get("name", "")] = list(ctl.get("particles", []) or [])
                default_fx = fc.get("defaultFx", "")
            a.fx = FxComp(descs, controls, default_fx)
            a.fx_blend = int(fb.get("blendmode", BLEND_ALPHA)) or BLEND_ALPHA

    def _link_binds(self):
        for a in self.actors:
            if a.bind and a.bind.parent_path:
                a.bind_parent = self.resolve(a.bind.parent_path, a.parent_scene)

    # -- lookup -------------------------------------------------------------------------
    def resolve(self, path, context=None):
        """Resolve an ObjectPath (e.g. 'bkg', 'COLOR|BG_Scroll', '(self)')
        relative to a context subscene, falling back to a scene-wide search."""
        if not path:
            return None
        p = path.replace("..|", "").strip("|")
        lp = p.lower()
        prefix = context.path.lower() + "|" if context is not None else ""
        hit = self.by_path.get(prefix + lp)
        if hit:
            return hit
        # walk up the owning subscenes
        ctx = context
        while ctx is not None:
            ctx = ctx.parent_scene
            pre = ctx.path.lower() + "|" if ctx is not None else ""
            hit = self.by_path.get(pre + lp)
            if hit:
                return hit
        # suffix match anywhere
        for key, a in self.by_path.items():
            if key == lp or key.endswith("|" + lp):
                return a
        last = lp.split("|")[-1]
        cands = self.by_name.get(last)
        return cands[0] if cands else None

    # -- per-frame state ----------------------------------------------------------------
    def new_states(self):
        return {a: ActorState(a) for a in self.actors}

    def resolve_world(self, states):
        for a in self.actors:
            states[a].resolved = False
        for a in self.actors:
            self._resolve(a, states, 0)

    def _resolve(self, a, states, depth):
        s = states[a]
        if s.resolved:
            return s
        if depth > 64:
            s.resolved = True
            return s
        s.resolved = True
        owner = a.parent_scene
        if a.bind_parent is not None:
            p = self._resolve(a.bind_parent, states, depth + 1)
            flip = p.world_flip if a.bind.use_parent_flip else a.flipped
            ox, oy, oz = s.pos
            if flip:
                ox = -ox
            ox *= p.world_scale[0]
            oy *= p.world_scale[1]
            c, si = math.cos(p.world_angle), math.sin(p.world_angle)
            s.world_pos = [p.world_pos[0] + ox * c - oy * si, p.world_pos[1] + ox * si + oy * c, p.world_pos[2] + oz]
            s.world_angle = p.world_angle + s.angle
            inh = a.bind.scale_inherit
            if inh == 0:
                s.world_scale = list(p.world_scale)
            elif inh == 2:
                s.world_scale = [p.world_scale[0] * s.scale[0], p.world_scale[1] * s.scale[1]]
            else:
                s.world_scale = list(s.scale)
            s.world_flip = flip
            alpha = s.alpha * (p.world_alpha if a.bind.use_parent_alpha else 1.0)
            color = [s.color[i] * (p.world_color[i] if a.bind.use_parent_color else 1.0) for i in range(3)]
        else:
            if owner is not None:
                p = self._resolve(owner, states, depth + 1)
                x, y, z = s.pos
                x *= p.world_scale[0]
                y *= p.world_scale[1]
                c, si = math.cos(p.world_angle), math.sin(p.world_angle)
                s.world_pos = [p.world_pos[0] + x * c - y * si, p.world_pos[1] + x * si + y * c, p.world_pos[2] + z]
                s.world_angle = p.world_angle + s.angle
                s.world_scale = [s.scale[0] * p.world_scale[0], s.scale[1] * p.world_scale[1]]
                s.world_flip = a.flipped
            else:
                s.world_pos = list(s.pos)
                s.world_angle = s.angle
                s.world_scale = list(s.scale)
                s.world_flip = a.flipped
            alpha = s.alpha
            color = list(s.color)
        # SubScene alpha/color/enable propagate to children (AlphaClipPlayer::updateActorAlpha)
        if owner is not None:
            p = states[owner]
            alpha *= p.world_alpha
            color = [color[i] * p.world_color[i] for i in range(3)]
            if not p.enabled:
                s.enabled = False
        s.world_alpha = alpha
        s.world_color = color
        return s
