"""
Export the generic-stage assets out of a game bundle into a loose data folder,
so the renderer runs without the .ipk.

Dependencies are traced, not guessed: each stage scene is loaded through a
recording AssetStore and every lazily-loaded dependency is forced (textures of
every material, meshes, particle materials/textures, and every tape in every
TapeCase rack, recursively through TapeReferenceClips). Exactly the bundle
entries that were read get written to <out>/<engine path>.ckd.
"""

import os

from .assets import AssetStore, Bundle, COOKED_PREFIX, AssetNotFound
from .material import material_from_serializable
from .scene import Scene
from .stages import DEFAULT_SCENES, scene_paths
from .tapes import TapeLibrary


class RecordingStore(AssetStore):
    def __init__(self, bundles):
        super().__init__(bundles, [])
        self.recorded = {}

    def read_raw(self, path):
        for b in self.bundles:
            for c in self._candidates(path):
                if c in b:
                    data = b.read(c)
                    self.recorded[c] = data
                    return data
        raise AssetNotFound(path)


def _force_material(store, mat):
    for p in mat.textures:
        if p:
            store.texture(p)


def export_stage_assets(bundle_paths, out_dir, log=print):
    store = RecordingStore([Bundle(b) for b in bundle_paths])
    try:
        store.data("enginedata/gameconfig/gameconfig.isg")
    except AssetNotFound:
        pass
    paths = scene_paths(store)
    for stage in DEFAULT_SCENES:
        isc = paths[stage]
        scene = Scene.load(store, isc)
        lib = TapeLibrary(store)
        for a in scene.actors:
            if a.graphic is not None:
                _force_material(store, a.graphic.material)
            if a.mesh is not None:
                try:
                    store.mesh(a.mesh.mesh_path)
                except AssetNotFound:
                    pass
                for m in a.mesh.materials:
                    _force_material(store, m)
            if a.fx is not None:
                for d in a.fx.descriptors.values():
                    mat = material_from_serializable(store, d.get("material") or {})
                    _force_material(store, mat)
            for label, (tape_path, _) in a.tapes.items():
                t = lib.get(tape_path)
                if t is not None:
                    t.compile(lib, scene, a)  # pulls referenced tapes recursively
        log(f"[export] {stage}: {len(scene.actors)} actors, {len(store.recorded)} files so far")

    total = 0
    for key, data in store.recorded.items():
        rel = key[len(COOKED_PREFIX):] if key.startswith(COOKED_PREFIX) else key
        dst = os.path.join(out_dir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        total += len(data)
    log(f"[export] wrote {len(store.recorded)} files ({total / 2**20:.1f} MB) -> {out_dir}")
    return len(store.recorded)
