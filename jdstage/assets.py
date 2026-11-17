"""
Asset access: .ipk bundles + loose files, and decoders for the cooked formats
the stage scenes reference.

Formats (all mirrored from this repo's engine source):
  * .ipk bundle      - src/engine/file/BundleFile.cpp / FilePackFormat.h
                       (same layout ipk_packer.py documents: big-endian boot
                       header, file table, per-file optional zlib).
  * cooked data      - "<path>.ckd": JSON for templates/tapes/materials (with
                       trailing NUL padding), XML for .isc scenes.
  * Lua data         - uncooked .tape/.tpl/.trk (see luatable.py).
  * cooked texture   - 44-byte big-endian TextureCookedHeader followed by a
                       plain DDS (pc_texture_cooker.py documents the header).
                       Wrap modes live at header bytes 40/41.
  * .m3d mesh        - Mesh3D::serialize (engine/display/Mesh3D.cpp): version,
                       link id, vertex/normal/uv arrays, then per-element
                       triangle index lists (12 u32 per triangle), big-endian.
"""

import json
import os
import struct
import threading
import zlib

import numpy as np

from . import luatable

COOKED_PREFIX = "cache/itf_cooked/pc/"
IPK_MAGIC = 0x50EC12BA

# GFX_TEXADRESSMODE (engine/display/Texture/TextureDefs.h)
WRAP, MIRROR, CLAMP, BORDER = 0, 1, 2, 3


class AssetNotFound(FileNotFoundError):
    pass


# ----------------------------------------------------------------------------------------
# IPK bundle
# ----------------------------------------------------------------------------------------

class Bundle:
    """Read-only, thread-safe random access to one .ipk file."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._f = open(path, "rb")
        hdr = struct.unpack(">11I", self._f.read(44))
        if hdr[0] != IPK_MAGIC:
            raise ValueError(f"{path}: not an .ipk bundle (bad magic)")
        self.files_start = hdr[3]
        count = struct.unpack(">I", self._f.read(4))[0]
        self.entries = {}
        rd = self._f.read
        for _ in range(count):
            pos_count, orig, comp = struct.unpack(">III", rd(12))
            rd(8)  # flush time
            positions = struct.unpack(f">{pos_count}Q", rd(8 * pos_count))
            n = struct.unpack(">I", rd(4))[0]
            base = rd(n).decode("utf-8", "replace")
            n = struct.unpack(">I", rd(4))[0]
            directory = rd(n).decode("utf-8", "replace")
            rd(8)  # string id + flags
            rel = (directory + base).replace("\\", "/").lower()
            self.entries[rel] = (positions[0] if positions else 0, orig, comp)

    def __contains__(self, rel):
        return rel.lower() in self.entries

    def read(self, rel):
        pos, orig, comp = self.entries[rel.lower()]
        with self._lock:
            self._f.seek(self.files_start + pos)
            data = self._f.read(comp or orig)
        return zlib.decompress(data) if comp else data

    def close(self):
        self._f.close()

    def __getstate__(self):  # allow pickling into worker processes
        return {"path": self.path}

    def __setstate__(self, state):
        self.__init__(state["path"])


# ----------------------------------------------------------------------------------------
# Asset store
# ----------------------------------------------------------------------------------------

def _strip_json(raw):
    return raw.rstrip(b"\0 \r\n\t")


def parse_data(raw, name=""):
    """Parse a cooked JSON or a Lua data file into the JSON shape."""
    body = _strip_json(raw)
    head = body.lstrip()[:1]
    if head in (b"{", b"["):
        return json.loads(body.decode("utf-8", "replace"))
    text = body.decode("utf-8", "replace")
    return luatable.load_lua_data(text)


def _support_tape_template(rel):
    """<tool>/templates/mashup_cinematics/<name>.tape.ckd for a mashup support tape path
    (".../cinematics/color_blue.tape"), else None."""
    from .mashup_seq import SUPPORT_TAPES, TEMPLATE_DIR
    parts = rel.lower().split("/")
    if len(parts) < 2 or parts[-2] != "cinematics":
        return None
    name = parts[-1]
    for ext in (".ckd", ".tape"):
        if name.endswith(ext):
            name = name[:-len(ext)]
    if name not in SUPPORT_TAPES:
        return None
    full = os.path.join(TEMPLATE_DIR, f"{name}.tape.ckd")
    return full if os.path.isfile(full) else None


class AssetStore:
    """Resolves engine paths ("world/maps/x/y.tpl") against bundles then loose roots."""

    def __init__(self, bundles=(), loose_roots=(), map_names=()):
        self.bundles = list(bundles)
        self.loose_roots = [r for r in loose_roots if r]
        # engine paths "world/maps/<name>/..." resolve into a loose root named <name>, or into
        # any loose root when <name> is the map's own name (folder may be named differently)
        self.map_names = [n.lower() for n in map_names if n]
        self._data_cache = {}
        self._tex_cache = {}
        self._mesh_cache = {}

    def __getstate__(self):
        return {"bundles": self.bundles, "loose_roots": self.loose_roots, "map_names": self.map_names}

    def __setstate__(self, state):
        self.__init__(state["bundles"], state["loose_roots"], state.get("map_names", ()))

    # -- raw lookup -------------------------------------------------------------------
    def _candidates(self, path):
        p = path.replace("\\", "/").lstrip("/").lower()
        yield COOKED_PREFIX + p + ".ckd"
        yield p + ".ckd"
        yield COOKED_PREFIX + p
        yield p

    def read_raw(self, path):
        for b in self.bundles:
            for c in self._candidates(path):
                if c in b:
                    return b.read(c)
        rel = path.replace("\\", "/").lstrip("/")
        for root in self.loose_roots:
            cands = [rel]
            # engine path "world/maps/<map>/x/y" -> "<map folder>/x/y"
            names = [os.path.basename(os.path.normpath(root)).lower()] + self.map_names
            for name in names:
                marker = "/maps/" + name + "/"
                i = ("/" + rel.lower()).find(marker)
                if i >= 0:
                    cands.append(rel[i - 1 + len(marker):])
                    break
            for c in [x for r in cands for x in (r, r + ".ckd")]:
                full = _find_case_insensitive(root, c)
                if full:
                    with open(full, "rb") as f:
                        return f.read()
        # mashup support tapes (color_*, uv_*, bg_scroll, pulse_fg) are identical for every
        # map; a map that only ships its mainsequence uses the tool's own copies
        tpl = _support_tape_template(rel)
        if tpl:
            with open(tpl, "rb") as f:
                return f.read()
        raise AssetNotFound(path)

    def exists(self, path):
        try:
            self.read_raw(path)
            return True
        except AssetNotFound:
            return False

    # -- typed loaders ----------------------------------------------------------------
    def data(self, path):
        key = path.lower()
        if key not in self._data_cache:
            self._data_cache[key] = parse_data(self.read_raw(path), path)
        return self._data_cache[key]

    def text(self, path):
        return self.read_raw(path).decode("latin1")

    def texture(self, path):
        key = path.lower()
        if key not in self._tex_cache:
            try:
                self._tex_cache[key] = Texture.from_cooked(self.read_raw(path), path)
            except AssetNotFound:
                self._tex_cache[key] = None
        return self._tex_cache[key]

    def mesh(self, path):
        key = path.lower()
        if key not in self._mesh_cache:
            self._mesh_cache[key] = parse_m3d(self.read_raw(path))
        return self._mesh_cache[key]


_dir_cache = {}


def _find_case_insensitive(root, rel):
    """Find root/rel on disk ignoring case of each path component."""
    cur = root
    for part in rel.split("/"):
        if not part:
            continue
        if cur not in _dir_cache:
            try:
                _dir_cache[cur] = {n.lower(): n for n in os.listdir(cur)}
            except OSError:
                return None
        real = _dir_cache[cur].get(part.lower())
        if real is None:
            return None
        cur = os.path.join(cur, real)
    return cur if os.path.isfile(cur) else None


# ----------------------------------------------------------------------------------------
# Textures
# ----------------------------------------------------------------------------------------

class Texture:
    """Decoded texture: float32 RGBA in [0,1] (gamma space, like the engine samples it)."""

    def __init__(self, rgba, wrap_u=CLAMP, wrap_v=CLAMP, name=""):
        self.rgba = rgba
        self.h, self.w = rgba.shape[:2]
        self.wrap_u = wrap_u
        self.wrap_v = wrap_v
        self.name = name
        self._mips = [rgba]

    def mip(self, level):
        import cv2
        while len(self._mips) <= level:
            prev = self._mips[-1]
            if prev.shape[0] <= 1 and prev.shape[1] <= 1:
                break
            nh, nw = max(1, prev.shape[0] // 2), max(1, prev.shape[1] // 2)
            self._mips.append(cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_AREA))
        return self._mips[min(level, len(self._mips) - 1)]

    @classmethod
    def from_cooked(cls, raw, name=""):
        wrap_u = wrap_v = CLAMP
        if raw[4:8] == b"TEX\0":
            data_off = struct.unpack(">I", raw[8:12])[0]
            wrap_u, wrap_v = raw[40], raw[41]
            dds = raw[data_off:]
        elif raw[:4] == b"DDS ":
            dds = raw
        else:
            # Not cooked: let OpenCV decode a source image (png/tga/jpg).
            import cv2
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"cannot decode texture {name}")
            return cls(_cv_to_rgba(img), wrap_u, wrap_v, name)
        return cls(decode_dds(dds), wrap_u, wrap_v, name)


def _cv_to_rgba(img):
    import cv2
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba = img[..., [2, 1, 0, 3]]
    scale = 65535.0 if rgba.dtype == np.uint16 else 255.0
    return (rgba.astype(np.float32) / scale)


def decode_dds(dds):
    import texture2ddecoder

    if dds[:4] != b"DDS ":
        raise ValueError("not a DDS payload")
    (size, flags, height, width, pitch, depth, mips) = struct.unpack_from("<7I", dds, 4)
    pf_size, pf_flags, fourcc, bitcount, rmask, gmask, bmask, amask = struct.unpack_from("<II4sIIIII", dds, 76)
    data = dds[128:]
    if pf_flags & 0x4:  # FOURCC
        fc = fourcc.decode("latin1")
        if fc == "DXT1":
            bgra = texture2ddecoder.decode_bc1(data, width, height)
        elif fc in ("DXT3", "DXT2"):
            bgra = texture2ddecoder.decode_bc2(data, width, height)
        elif fc in ("DXT5", "DXT4"):
            bgra = texture2ddecoder.decode_bc3(data, width, height)
        else:
            raise ValueError(f"unsupported DDS fourcc {fc}")
        arr = np.frombuffer(bgra, np.uint8).reshape(height, width, 4)
        return arr[..., [2, 1, 0, 3]].astype(np.float32) / 255.0
    bpp = bitcount // 8
    n = width * height
    raw = np.frombuffer(data[: n * bpp], np.uint8)
    if bpp == 1:  # L8
        l = raw.reshape(height, width).astype(np.float32) / 255.0
        return np.dstack([l, l, l, np.ones_like(l)])
    if bpp == 2 and (pf_flags & 0x20000):  # L8A8
        px = raw.reshape(height, width, 2).astype(np.float32) / 255.0
        return np.dstack([px[..., 0], px[..., 0], px[..., 0], px[..., 1]])
    if bpp in (3, 4):
        px = raw.reshape(height, width, bpp)
        word = np.zeros((height, width), np.uint32)
        for i in range(bpp):
            word |= px[..., i].astype(np.uint32) << (8 * i)

        def chan(mask, default):
            if not mask:
                return np.full((height, width), default, np.float32)
            shift = (mask & -mask).bit_length() - 1
            return ((word & mask) >> shift).astype(np.float32) / float(mask >> shift)

        a_mask = amask if (pf_flags & 0x1) else 0
        return np.dstack([chan(rmask, 0), chan(gmask, 0), chan(bmask, 0), chan(a_mask, 1.0)])
    raise ValueError(f"unsupported DDS pixel format (bitcount {bitcount})")


# ----------------------------------------------------------------------------------------
# Mesh3D
# ----------------------------------------------------------------------------------------

class Mesh3D:
    def __init__(self, vertices, uvs, triangles):
        self.vertices = vertices      # (N,3) float32
        self.uvs = uvs                # (M,2) float32
        self.triangles = triangles    # list of (material_id, (T,3) vtx idx, (T,3) uv idx)


def parse_m3d(raw):
    off = 0

    def u32():
        nonlocal off
        v = struct.unpack_from(">I", raw, off)[0]
        off += 4
        return v

    u32()  # version (Versioning::mesh3DVer)
    u32()  # serializeLink id
    nv = u32()
    verts = np.frombuffer(raw, ">f4", nv * 3, off).reshape(nv, 3).astype(np.float32)
    off += nv * 12
    nn = u32()
    off += nn * 12  # normals: unused (no lighting on these stages)
    nuv = u32()
    uvs = np.frombuffer(raw, ">f4", nuv * 2, off).reshape(nuv, 2).astype(np.float32)
    off += nuv * 8
    elements = []
    for _ in range(u32()):
        mat_id = u32()
        nt = u32()
        tri = np.frombuffer(raw, ">u4", nt * 12, off).reshape(nt, 12).astype(np.int64)
        off += nt * 48
        elements.append((mat_id, tri[:, 0:3], tri[:, 6:9]))
    return Mesh3D(verts, uvs, elements)
