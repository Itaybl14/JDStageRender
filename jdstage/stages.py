"""
Mashup stage profile.

A song whose songdesc backgroundType is Mashup (6) gets the generic stage
genericstages["mashup"] (JD_GSS_LoadMap::addSongLayers). The scene path is read from
enginedata/gameconfig/gameconfig.isg ("genericstages") when available.
Only the Mashup stage is supported.

What gets played on load mirrors the game:
  * every TapeCase "autoplay" label (TapeCase_Component::onBecomeActive)
  * FXControllerComponent.defaultFx starts with the actor.
"""

DEFAULT_SCENES = {
    "mashup": "world/maps/_mashup/_mashup_main_scene.isc",
}

# Actors that are never rendered (none at the moment; --hide hides more per render).
ALWAYS_HIDDEN = {}

# The CentralShape*_ON/OFF_Top/Bottom quads (the star pieces at top/bottom centre) are
# drawn bigger in the game than their serialized SCALE gives with the engine's 2x2
# MaterialGraphicComponent quad. Nothing in the cooked data (scene, .act, template,
# tapes, atlases) accounts for it, so the factors are calibrated against in-game footage
# (MyWayMU). OFF (always shown): the triangle tip reaches ~13% of the screen height.
# ON (the beat pulse from pulse_fg): its star sits exactly over the OFF triangle (same
# tip) and its glow rays fan out over the FG frame bands. The two need different factors
# (likely their source textures had other sizes before being cooked to 256x512 / 1024x512).
MASHUP_CENTRALSHAPE_SCALE = {"off": (1.43, 1.65), "on": (3.37, 2.61)}

MASHUP_COLORS = ["blue", "orange", "green", "purple"]

# mashup colour set -> (CentralShape actor prefix, FG scroll actor, bg_scroll layer index)
_MASHUP_SETS = {
    "blue": ("centralshape_", "fg_scroll_blue", 0),
    "orange": ("centralshape3_", "fg_scroll_orange", 1),
    "green": ("centralshape2_", "fg_scroll_green", 2),
    "purple": ("centralshape4_", "fg_scroll_purple", 3),
}


def scene_paths(store):
    paths = dict(DEFAULT_SCENES)
    try:
        cfg = store.data("enginedata/gameconfig/gameconfig.isg")
        gs = cfg.get("genericstages") or {}
        for k, v in gs.items():
            if v and k.lower() in DEFAULT_SCENES:
                paths[k.lower()] = v
    except Exception:
        pass
    return paths


def find_actor(scene, name):
    hits = scene.by_name.get(name.lower())
    return hits[0] if hits else None


def apply_static_options(stage, scene, opts):
    """Options that change the authored initial state before playback.

    Only applied when explicitly requested: maps drive these themselves from their
    mainsequence (e.g. a mashup's color_* tapes fade the colour sets in and out), so
    the override only sets the *initial* alpha/layers and tapes can still change them.
    """
    hidden = {n.lower() for n in ALWAYS_HIDDEN.get(stage, [])}
    hidden |= {n.lower() for n in opts.get("hide", []) or []}
    for a in scene.actors:
        if a.name.lower() in hidden:
            a.enabled = False  # disables its quads/meshes/particles (and children if a subscene)

    if stage == "mashup":
        for a in scene.actors:
            n = a.name.lower()
            if n.startswith("centralshape"):
                sx, sy = MASHUP_CENTRALSHAPE_SCALE["on" if "_on_" in n else "off"]
                a.init_scale = [a.init_scale[0] * sx, a.init_scale[1] * sy]

    if stage == "mashup" and opts.get("mashup_color"):
        color = opts["mashup_color"]
        for a in scene.actors:
            n = a.name.lower()
            for col, (prefix, fg, _) in _MASHUP_SETS.items():
                if col != color and (n.startswith(prefix) or n == fg):
                    a.init_alpha = 0.0
        bg = find_actor(scene, "BG_Scroll")
        keep_layer = _MASHUP_SETS[color][2]
        if bg is not None and bg.mesh is not None:
            for m in bg.mesh.materials:
                for i, layer in enumerate(m.layers):
                    layer.enabled = (i == keep_layer)


def initial_plays(stage, scene, opts):
    """List of (actor, label, loop) to start at the beginning of the render."""
    plays = []
    for a in scene.actors:
        if "autoplay" in a.tapes:
            plays.append((a, "autoplay", None))
    for lbl in opts.get("extra_plays", []) or []:
        if ":" in lbl:
            actor_name, label = lbl.split(":", 1)
            a = find_actor(scene, actor_name)
        else:
            label = lbl
            a = next((x for x in scene.actors if label.lower() in x.tapes), None)
        if a is not None:
            plays.append((a, label, None))
    return plays
