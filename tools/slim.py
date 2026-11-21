"""
Make the models lighter without changing how they look (much):

  1. drop materials no mesh uses, and their textures (some kits ship rust or rock textures a model never uses);
  2. textures -> WebP (EXT_texture_webp), quality 90 by default, and no bigger than --max px (1024 by default);
  3. sibling blocks of the same material under the same node -> one block, while they fit in 65,535 vertices
     (fewer draw calls; positions are quantised again to the new bounds, still 16-bit);
  4. sibling instanced nodes with the same shape and material -> one instanced node with all the copies;
  5. byte-identical buffer views stored once.

Normals are already 8-bit and positions/UVs 16-bit (scenekit.write_model), so nothing is done there.
Running it again on a slimmed file changes nothing (WebP textures already within --max are kept as they are).

    python tools/slim.py                       # every models/*/*/model*.glb, in place
    python tools/slim.py models/item/Q25173    # one model folder (model.glb and model_built.glb)
    python tools/slim.py --max 2048 --q 90 --out some/dir  file.glb ...   # options; --out writes copies there
    python tools/slim.py --dry                 # only print what it would save

Needs numpy and Pillow (with WebP, which every Pillow wheel has).
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glbkit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_IMG_CACHE: dict = {}                     # (md5 of input, normal?, max, q) -> webp bytes: the kits share textures


# ── accessors as numpy arrays ────────────────────────────────────────────────
def read_acc(g, binb, ai):
    a = g["accessors"][ai]
    dt, n = np.dtype(NP[a["componentType"]]), glbkit.TYPE_N[a["type"]]
    v = g["bufferViews"][a["bufferView"]]
    es = dt.itemsize * n
    stride = v.get("byteStride") or es
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    raw = np.frombuffer(binb, np.uint8, count=stride * (a["count"] - 1) + es, offset=start) if a["count"] else np.zeros(0, np.uint8)
    if stride == es:
        arr = raw.view(dt)
    else:
        arr = np.lib.stride_tricks.as_strided(raw, (a["count"], es), (stride, 1)).copy().reshape(-1).view(dt)
    return arr.reshape(-1, n) if n > 1 else arr


class Out:
    """New accessors / buffer views, written at the end into one buffer (identical views stored once)."""

    def __init__(self):
        self.accs, self.views, self.blobs, self.seen = [], [], [], {}

    def view(self, data: bytes, target=None, stride=None):
        key = (hashlib.sha1(data).digest(), target, stride)
        if key in self.seen:
            return self.seen[key]
        v = {"buffer": 0, "byteLength": len(data)}
        if target:
            v["target"] = target
        if stride:
            v["byteStride"] = stride
        self.views.append(v); self.blobs.append(data)
        self.seen[key] = len(self.views) - 1
        return self.seen[key]

    def acc(self, arr, ctype, typ, normalized=False, target=None, minmax=False, pad4=False):
        arr = np.ascontiguousarray(arr)
        n = glbkit.TYPE_N[typ]
        data, stride = arr, None
        if pad4:                                             # 3 × int16 -> stride 8, 3 × int8 -> stride 4 (vertex alignment)
            data = np.hstack([arr, np.zeros((len(arr), 1), arr.dtype)])
            stride = data.shape[1] * arr.dtype.itemsize
        a = {"bufferView": self.view(data.tobytes(), target, stride), "componentType": ctype, "count": len(arr), "type": typ}
        if normalized:
            a["normalized"] = True
        if minmax:
            a["min"] = (arr.min(0) if n > 1 else [arr.min()]).tolist() if len(arr) else [0] * n
            a["max"] = (arr.max(0) if n > 1 else [arr.max()]).tolist() if len(arr) else [0] * n
            if n == 1:
                a["min"], a["max"] = a["min"][0], a["max"][0]
        self.accs.append(a)
        return len(self.accs) - 1

    def copy_acc(self, g, binb, ai):
        """an untouched accessor, copied as it was (stride kept for vertex data)"""
        a = dict(g["accessors"][ai])
        arr = read_acc(g, binb, ai)
        v = g["bufferViews"][a["bufferView"]]
        target, stride = v.get("target"), v.get("byteStride")
        n = glbkit.TYPE_N[a["type"]]
        data = arr
        if stride and stride != arr.dtype.itemsize * n:
            pad = stride // arr.dtype.itemsize - n
            data = np.hstack([arr.reshape(len(arr), -1), np.zeros((len(arr), pad), arr.dtype)])
        a["bufferView"] = self.view(np.ascontiguousarray(data).tobytes(), target, stride)
        a.pop("byteOffset", None)
        self.accs.append(a)
        return len(self.accs) - 1


# ── textures ─────────────────────────────────────────────────────────────────
def webp(data: bytes, normal: bool, max_px: int, q: int) -> bytes:
    key = (hashlib.md5(data).hexdigest(), normal, max_px, q)
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    is_webp = im.format == "WEBP"
    im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") and not normal else "RGB")
    if is_webp and max(im.size) <= max_px:
        out = data                                            # already done: leave it alone
    else:
        if max(im.size) > max_px:
            k = max_px / max(im.size)
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
            if normal:                                        # averaging normals shortens them: make them unit again
                a = np.asarray(im, np.float32)[..., :3] / 127.5 - 1
                a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-6)
                im = Image.fromarray(np.clip(np.round((a + 1) * 127.5), 0, 255).astype(np.uint8), "RGB")
        b = io.BytesIO()
        im.save(b, "WEBP", quality=q, method=6, exact=True)
        out = b.getvalue()
    _IMG_CACHE[key] = out
    return out


# ── the whole file ───────────────────────────────────────────────────────────
def slim(g, binb, max_px=1024, q=90, merge=True):
    """-> (new gltf, new bin, notes)"""
    notes = []
    mats = g.get("materials", [])
    used = sorted({p["material"] for m in g["meshes"] for p in m["primitives"] if "material" in p})
    mat_map = {old: new for new, old in enumerate(used)}
    if len(used) < len(mats):
        notes.append(f"{len(mats) - len(used)} unused material(s)")
    out = Out()

    # textures of the materials kept; which ones are normal maps (linear, renormalised when scaled)
    tex_map, images, textures = {}, [], []
    img_done = {}

    def texture(ti, normal):
        if (ti, normal) in tex_map:
            return tex_map[(ti, normal)]
        t = g["textures"][ti]
        src = (t.get("extensions") or {}).get("EXT_texture_webp", {}).get("source", t.get("source"))
        key = (src, normal)
        if key not in img_done:
            im = g["images"][src]
            data = webp(glbkit.view_bytes(g, binb, im["bufferView"]), normal, max_px, q)
            images.append({"bufferView": out.view(data), "mimeType": "image/webp"})
            img_done[key] = len(images) - 1
        textures.append({"sampler": t.get("sampler", 0), "extensions": {"EXT_texture_webp": {"source": img_done[key]}}})
        tex_map[(ti, normal)] = len(textures) - 1
        return tex_map[(ti, normal)]

    new_mats = []
    for old in used:
        m = dict(mats[old])
        pbr = dict(m.get("pbrMetallicRoughness", {}))
        if "baseColorTexture" in pbr:
            pbr["baseColorTexture"] = dict(pbr["baseColorTexture"], index=texture(pbr["baseColorTexture"]["index"], False))
        for k in ("metallicRoughnessTexture",):
            if k in pbr:
                pbr[k] = dict(pbr[k], index=texture(pbr[k]["index"], False))
        m["pbrMetallicRoughness"] = pbr
        if "normalTexture" in m:
            m["normalTexture"] = dict(m["normalTexture"], index=texture(m["normalTexture"]["index"], True))
        for k in ("occlusionTexture", "emissiveTexture"):
            if k in m:
                m[k] = dict(m[k], index=texture(m[k]["index"], False))
        new_mats.append(m)

    nodes = [dict(n) for n in g["nodes"]]
    meshes = g["meshes"]
    new_meshes, mesh_seen = [], {}

    def prim_sig(p):
        return (p.get("material"), tuple(sorted(p["attributes"])), p.get("mode", 4))

    def copy_mesh(mi):
        if mi in mesh_seen:
            return mesh_seen[mi]
        prims = []
        for p in meshes[mi]["primitives"]:
            np_ = {"attributes": {k: out.copy_acc(g, binb, a) for k, a in p["attributes"].items()}}
            if "indices" in p:
                np_["indices"] = out.copy_acc(g, binb, p["indices"])
            if "material" in p:
                np_["material"] = mat_map[p["material"]]
            if "mode" in p:
                np_["mode"] = p["mode"]
            prims.append(np_)
        new_meshes.append({"name": meshes[mi].get("name", ""), "primitives": prims})
        mesh_seen[mi] = len(new_meshes) - 1
        return mesh_seen[mi]

    def mesh_hash(mi):
        h = hashlib.sha1()
        for p in meshes[mi]["primitives"]:
            h.update(repr(prim_sig(p)).encode())
            for k in sorted(p["attributes"]):
                h.update(read_acc(g, binb, p["attributes"][k]).tobytes())
            if "indices" in p:
                h.update(read_acc(g, binb, p["indices"]).tobytes())
        return h.digest()

    # ── merge sibling blocks (unnamed, translation+scale only, one primitive, same material) ──
    def mergeable(ni):
        n = nodes[ni]
        if "mesh" not in n or n.get("name") or n.get("children") or n.get("extensions") or "rotation" in n or "matrix" in n:
            return False
        prims = meshes[n["mesh"]]["primitives"]
        if len(prims) != 1 or prims[0].get("mode", 4) != 4 or "indices" not in prims[0]:
            return False
        a = g["accessors"][prims[0]["attributes"]["POSITION"]]
        return a["componentType"] == 5122 and not a.get("normalized")

    def merge_blocks(group):
        """group of node ids -> one new node (dict) with a new mesh"""
        p0 = meshes[nodes[group[0]]["mesh"]]["primitives"][0]
        keys = sorted(p0["attributes"])
        pos, attrs, idx, base = [], {k: [] for k in keys if k != "POSITION"}, [], 0
        for ni in group:
            n = nodes[ni]; p = meshes[n["mesh"]]["primitives"][0]
            t = np.array(n.get("translation", [0, 0, 0]), float); s = np.array(n.get("scale", [1, 1, 1]), float)
            pos.append(read_acc(g, binb, p["attributes"]["POSITION"]).astype(float) * s + t)
            for k in attrs:
                attrs[k].append(read_acc(g, binb, p["attributes"][k]))
            idx.append(read_acc(g, binb, p["indices"]).astype(np.uint32) + base)
            base += len(pos[-1])
        v = np.vstack(pos)
        lo, hi = v.min(0), v.max(0)
        sc = np.maximum(hi - lo, 1e-6) / 65534.0
        mid = (lo + hi) / 2
        qv = np.round((v - mid) / sc).astype(np.int16)
        pa = {"POSITION": out.acc(qv, 5122, "VEC3", target=34962, minmax=True, pad4=True)}
        for k, arrs in attrs.items():
            a0 = g["accessors"][p0["attributes"][k]]
            arr = np.vstack(arrs)
            pad = a0["type"] == "VEC3" and arr.dtype.itemsize < 4
            pa[k] = out.acc(arr, a0["componentType"], a0["type"], a0.get("normalized", False), 34962, pad4=pad)
        ii = np.concatenate(idx)
        ia = out.acc(ii.astype(np.uint16) if base <= 65535 else ii, 5123 if base <= 65535 else 5125, "SCALAR", target=34963)
        pr = {"attributes": pa, "indices": ia, "material": mat_map[p0["material"]]}
        new_meshes.append({"name": meshes[nodes[group[0]]["mesh"]].get("name", ""), "primitives": [pr]})
        return {"mesh": len(new_meshes) - 1, "translation": [float(x) for x in mid], "scale": [float(x) for x in sc]}

    def inst_attr(n, k):
        return read_acc(g, binb, n["extensions"]["EXT_mesh_gpu_instancing"]["attributes"][k]).astype(np.float32)

    merged_blocks = merged_inst = 0
    new_nodes = []

    def emit(ni):
        """copy node ni (and its subtree) into new_nodes -> new index"""
        nonlocal merged_blocks, merged_inst
        n = dict(nodes[ni])
        kids = n.pop("children", [])
        if "mesh" in n:
            n["mesh"] = copy_mesh(n["mesh"])
        inst = (n.get("extensions") or {}).get("EXT_mesh_gpu_instancing")
        if inst:
            n["extensions"] = dict(n["extensions"])
            n["extensions"]["EXT_mesh_gpu_instancing"] = {"attributes": {k: out.copy_acc(g, binb, a) for k, a in inst["attributes"].items()}}
        me = len(new_nodes); new_nodes.append(n)
        if kids:
            n["children"] = emit_list(kids)
        return me

    def emit_list(ids):
        """siblings: merge what can be merged, copy the rest (order kept for the rest)"""
        nonlocal merged_blocks, merged_inst
        res, done = [], set()
        if merge:
            # blocks of the same material
            groups = {}
            for ni in ids:
                if mergeable(ni):
                    groups.setdefault(prim_sig(meshes[nodes[ni]["mesh"]]["primitives"][0]), []).append(ni)
            for sig, grp in groups.items():
                if len(grp) < 2:
                    continue
                grp = sorted(grp, key=lambda ni: -g["accessors"][meshes[nodes[ni]["mesh"]]["primitives"][0]["attributes"]["POSITION"]]["count"])
                bins = []                                       # first-fit decreasing into 65,535-vertex blocks
                for ni in grp:
                    c = g["accessors"][meshes[nodes[ni]["mesh"]]["primitives"][0]["attributes"]["POSITION"]]["count"]
                    for b in bins:
                        if b[0] + c <= 65535:
                            b[0] += c; b[1].append(ni); break
                    else:
                        bins.append([c, [ni]])
                for _, b in bins:
                    if len(b) > 1:
                        new_nodes.append(merge_blocks(b)); res.append(len(new_nodes) - 1)
                        done.update(b); merged_blocks += len(b) - 1
            # instanced copies of the same shape
            igroups = {}
            for ni in ids:
                n = nodes[ni]
                ext = (n.get("extensions") or {}).get("EXT_mesh_gpu_instancing")
                if ext and "mesh" in n and not n.get("children") and "rotation" not in n and "matrix" not in n \
                        and "translation" not in n and "scale" not in n and set(ext["attributes"]) == {"TRANSLATION", "ROTATION", "SCALE"}:
                    igroups.setdefault(mesh_hash(n["mesh"]), []).append(ni)
            for grp in igroups.values():
                if len(grp) < 2:
                    continue
                first = nodes[grp[0]]
                T = np.vstack([inst_attr(nodes[i], "TRANSLATION") for i in grp])
                R = np.vstack([inst_attr(nodes[i], "ROTATION") for i in grp])
                S = np.vstack([inst_attr(nodes[i], "SCALE") for i in grp])
                n = {"name": first.get("name", ""), "mesh": copy_mesh(first["mesh"]), "extensions": {"EXT_mesh_gpu_instancing": {"attributes": {
                    "TRANSLATION": out.acc(T, 5126, "VEC3"), "ROTATION": out.acc(R, 5126, "VEC4"), "SCALE": out.acc(S, 5126, "VEC3")}}}}
                new_nodes.append(n); res.append(len(new_nodes) - 1)
                done.update(grp); merged_inst += len(grp) - 1
        for ni in ids:
            if ni not in done:
                res.append(emit(ni))
        return res

    scene = g["scenes"][g.get("scene", 0)]
    top = emit_list(scene["nodes"])
    if merged_blocks:
        notes.append(f"{merged_blocks} block(s) merged into their siblings")
    if merged_inst:
        notes.append(f"{merged_inst} instanced node(s) merged")

    # the new file
    used_ext = set(g.get("extensionsUsed", [])) - {"EXT_texture_webp"}
    req_ext = set(g.get("extensionsRequired", [])) - {"EXT_texture_webp"}
    if textures:
        used_ext.add("EXT_texture_webp"); req_ext.add("EXT_texture_webp")
    if not any("KHR_texture_transform" in (t or {}).get("extensions", {}) for m in new_mats
               for t in (m["pbrMetallicRoughness"].get("baseColorTexture"), m.get("normalTexture"))):
        used_ext.discard("KHR_texture_transform"); req_ext.discard("KHR_texture_transform")
    if not any("EXT_mesh_gpu_instancing" in (n.get("extensions") or {}) for n in new_nodes):
        used_ext.discard("EXT_mesh_gpu_instancing"); req_ext.discard("EXT_mesh_gpu_instancing")
    ng = {k: v for k, v in g.items() if k not in ("nodes", "meshes", "materials", "accessors", "bufferViews", "buffers",
                                                     "images", "textures", "samplers", "scenes", "extensionsUsed", "extensionsRequired")}
    ng["asset"] = dict(g["asset"])
    if "slim" not in ng["asset"].get("generator", ""):
        ng["asset"]["generator"] = ng["asset"].get("generator", "") + " + slim"
    ng["extensionsUsed"] = sorted(used_ext)
    if req_ext:
        ng["extensionsRequired"] = sorted(req_ext)
    ng["scene"] = 0
    ng["scenes"] = [dict(scene, nodes=top)]
    ng["nodes"] = new_nodes
    ng["meshes"] = new_meshes
    ng["materials"] = new_mats
    ng["accessors"] = out.accs
    if textures:
        ng["images"], ng["textures"] = images, textures
        ng["samplers"] = g.get("samplers") or [{}]
    buf = bytearray()
    for v, blob in zip(out.views, out.blobs):
        while len(buf) % 4:
            buf.append(0)
        v["byteOffset"] = len(buf); buf.extend(blob)
    ng["bufferViews"] = out.views
    ng["buffers"] = [{"byteLength": len(buf)}]
    return ng, bytes(buf), notes


def slim_file(src, dst=None, **kw):
    """slim src (a .glb) into dst (default: in place) -> (bytes before, bytes after, notes)"""
    src = Path(src)
    g, binb = glbkit.read_glb(src)
    ng, nb, notes = slim(g, binb, **kw)
    data = glbkit.glb_bytes(ng, nb)
    before = src.stat().st_size
    if not kw.get("dry") and dst is not False:
        Path(dst or src).write_bytes(data)
    return before, len(data), notes


def main():
    args = sys.argv[1:]
    opts = {"max_px": 1024, "q": 90, "merge": True}
    out_dir, dry = None, False
    while args and args[0].startswith("--"):
        a = args.pop(0)
        if a == "--max":
            opts["max_px"] = int(args.pop(0))
        elif a == "--q":
            opts["q"] = int(args.pop(0))
        elif a == "--no-merge":
            opts["merge"] = False
        elif a == "--out":
            out_dir = Path(args.pop(0)); out_dir.mkdir(parents=True, exist_ok=True)
        elif a == "--dry":
            dry = True
        else:
            sys.exit(__doc__)
    files = []
    for a in args or [ROOT / "models"]:
        p = Path(a)
        files += sorted(p.glob("**/model*.glb")) if p.is_dir() else [p]
    tb = ta = 0
    for f in files:
        dst = False if dry else (out_dir / f"{f.parent.name}_{f.name}" if out_dir else None)
        b, a, notes = slim_file(f, dst, **opts)
        tb += b; ta += a
        rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
        print(f"{rel}: {b / 2**20:.2f} -> {a / 2**20:.2f} MB ({100 * (1 - a / b):.0f} % less){'; ' + '; '.join(notes) if notes else ''}")
    if len(files) > 1:
        print(f"{len(files)} file(s): {tb / 2**20:.1f} -> {ta / 2**20:.1f} MB ({100 * (1 - ta / max(tb, 1)):.0f} % less)")


if __name__ == "__main__":
    main()
