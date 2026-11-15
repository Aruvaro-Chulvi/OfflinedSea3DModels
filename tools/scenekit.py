"""
scenekit — richer scenes on top of meshkit (MIT), for the close-up dioramas (docs/MODEL_SPEC.md, «diorama.glb»).

- Procedural, seamlessly tiling PBR textures (colour + normal map) drawn with numpy: seabed ooze, riveted steel,
  porous rust.
- UV projection at write time (top-down «xz» for the seabed, «triplanar» for everything else), splitting the
  vertices where a smooth surface changes projection so no texture gets smeared.
- Ambient occlusion baked into the vertex colours with a voxel grid (no Blender, no ray tracer: numpy only).
- A glTF writer with several materials, embedded JPEG textures and «points of interest» (empty nodes whose
  extras carry a label in several languages).

    from meshkit import *
    from scenekit import *
    bake_ao([hull, seabed], occluders=[hull], ground=lambda x, z: 0 * x)
    write_scene("diorama.glb", [("seabed", seabed, "seabed"), ("hull", hull, "steel")],
                {"seabed": material(tex=seabed_textures(), uv=("xz", 6)), "steel": material(tex=steel_textures())})
"""
from __future__ import annotations

import io
import json
import math
import struct

import numpy as np
from PIL import Image

from meshkit import Mesh, normals, _srgb_to_linear


# ── Tiling noise for textures ────────────────────────────────────────────────
def pnoise(n, cells, seed):
    """Value noise on an n×n image that tiles seamlessly (cells × cells lattice per tile)."""
    g = np.random.default_rng(seed).random((cells, cells))
    x = np.arange(n) * cells / n
    i = np.floor(x).astype(int); f = x - i; u = f * f * (3 - 2 * f)
    i0, i1 = i % cells, (i + 1) % cells
    a, b, c, d = g[np.ix_(i0, i0)], g[np.ix_(i0, i1)], g[np.ix_(i1, i0)], g[np.ix_(i1, i1)]
    uy, ux = u[:, None], u[None, :]
    return (a * (1 - ux) + b * ux) * (1 - uy) + (c * (1 - ux) + d * ux) * uy


def pfbm(n, cells, seed, octaves=5, gain=0.5):
    s, a, t = np.zeros((n, n)), 1.0, 0.0
    for o in range(octaves):
        c = cells * 2 ** o
        if c > n // 2:
            break
        s += a * pnoise(n, c, seed + 31 * o); t += a; a *= gain
    return s / t


def splats(n, pts, radius, shape="gauss"):
    """Sum of round bumps at points (u, v in 0..1, radius in tile units) on a tiling image."""
    out = np.zeros((n, n))
    for (u, v), r in zip(pts, radius):
        rp = max(2, int(math.ceil(r * n * 2.5)))
        cx, cy = u * n, v * n
        xs = np.arange(int(cx) - rp, int(cx) + rp + 1); ys = np.arange(int(cy) - rp, int(cy) + rp + 1)
        dx = (xs + 0.5 - cx)[None, :] / (r * n); dy = (ys + 0.5 - cy)[:, None] / (r * n)
        d2 = dx * dx + dy * dy
        k = np.exp(-d2 * 2.2) if shape == "gauss" else np.clip(1 - d2, 0, 1) ** 0.5
        out[np.ix_(ys % n, xs % n)] += k
    return out


def normal_map(h, tile_m, strength=1.0):
    """Height (metres) on a tiling image -> tangent-space normal map (uint8 RGB, +Y up as glTF wants)."""
    n = h.shape[0]; px = tile_m / n
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * px) * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * px) * strength
    nv = np.stack([-dx, dy, np.ones_like(h)], -1)          # image rows go down = -v
    nv /= np.linalg.norm(nv, axis=-1, keepdims=True)
    return np.round((nv * 0.5 + 0.5) * 255).astype(np.uint8)


def _jpeg(rgb, q=90):
    b = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").save(b, "JPEG", quality=q, optimize=True, subsampling=0)
    return b.getvalue()


def _rgb(col, n):
    return np.round(np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(n, n, 3)


def seabed_textures(n=1024, tile=6.0, seed=1):
    """Deep-sea ooze: current ripples in patches, burrow holes with a rim, tiny pebbles, white foraminifera.
    Returns {"map": jpeg, "normal": jpeg, "tile": metres}."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    warp = pfbm(n, 4, seed + 1, 3)
    phase = 2 * math.pi * (12 * xx + 5 * yy) + 7.0 * warp
    rip = (0.5 + 0.5 * np.sin(phase + 0.6 * np.sin(phase))) ** 1.6                   # asymmetric ripples
    patch = np.clip((pfbm(n, 3, seed + 2, 3) - 0.42) * 4, 0, 1)
    h = 0.010 * rip * patch + 0.012 * pfbm(n, 6, seed + 3, 6)
    pits_p = r.random((45, 2)); pits_r = r.uniform(0.002, 0.006, 45)
    pits = splats(n, pits_p, pits_r)
    rims = splats(n, pits_p, pits_r * 2.2) - pits * 0.9
    h += -0.03 * pits + 0.006 * rims
    peb_p = r.random((80, 2)); peb_r = r.uniform(0.0015, 0.005, 80)
    peb = np.clip(splats(n, peb_p, peb_r, "disc"), 0, 1)
    h += 0.018 * peb
    spk = np.clip(splats(n, r.random((2200, 2)), r.uniform(0.0004, 0.0011, 2200)), 0, 1)
    base = np.array([0.67, 0.63, 0.555])
    v = (pfbm(n, 5, seed + 4, 5) - 0.5) * 0.10 + (pfbm(n, 40, seed + 5, 2) - 0.5) * 0.05
    col = base[None, None, :] * (1 + v[..., None]) - 0.035 * rip[..., None] * patch[..., None]
    col = col * (1 - 0.35 * np.clip(pits, 0, 1)[..., None])
    col = col * (1 - peb[..., None]) + np.array([0.36, 0.35, 0.34])[None, None] * peb[..., None] * (0.8 + 0.4 * pfbm(n, 64, 9, 2))[..., None]
    col = col + 0.16 * spk[..., None]
    return {"map": _jpeg(col * 255), "normal": _jpeg(normal_map(h, tile, 1.0), 92), "tile": tile}


def steel_textures(n=2048, tile=7.2, seed=2, strake=1.8, butt=3.6):
    """Riveted plates eaten by rust, as a near-white multiplier (the paint and the rust colour are in the vertex
    colours): seams, rows of rivets, pitting and flaking blisters. strake/butt: plate height and length (m)."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    U, V = xx * tile, yy * tile
    row = np.floor(V / strake)
    dv = np.abs(V / strake - np.round(V / strake)) * strake            # distance to the horizontal seams
    Us = U + (row % 2) * butt / 2
    du = np.abs(Us / butt - np.round(Us / butt)) * butt                # distance to the butts (staggered)
    seam = np.exp(-(np.minimum(dv, du) / 0.008) ** 2)
    pitch = 0.075
    ru = (U % pitch) - pitch / 2
    riv_h = np.exp(-((ru ** 2 + (dv - 0.045) ** 2) / 0.011 ** 2))
    rvv = (V % pitch) - pitch / 2
    riv_v = np.exp(-((rvv ** 2 + (du - 0.045) ** 2) / 0.011 ** 2))
    riv = np.clip(riv_h + riv_v, 0, 1)
    pit = pfbm(n, 48, seed + 1, 4)
    blis = np.clip((pfbm(n, 10, seed + 2, 5) - 0.55) * 6, 0, 1)
    h = -0.003 * seam + 0.006 * riv - 0.004 * np.clip(pit - 0.5, 0, 1) * 2 + 0.003 * blis - 0.002 * pfbm(n, 160, seed + 3, 2)
    lum = 0.80 + 0.22 * (pfbm(n, 8, seed + 4, 5) - 0.5) - 0.25 * seam + 0.05 * riv - 0.18 * np.clip(pit - 0.62, 0, 1) * 4
    tint = np.array([1.0, 0.86, 0.72])
    col = lum[..., None] * (1 - 0.6 * blis[..., None]) + lum[..., None] * tint * 0.6 * blis[..., None] * 1.08
    return {"map": _jpeg(np.clip(col, 0, 1) * 255), "normal": _jpeg(normal_map(h, tile, 1.0), 92), "tile": tile}


def rust_textures(n=512, tile=1.5, seed=3):
    """Porous, bubbly rust for the rusticles and the heaviest crusts (a near-white multiplier)."""
    r = np.random.default_rng(seed)
    pores = np.clip(splats(n, r.random((260, 2)), r.uniform(0.004, 0.014, 260)), 0, 1)
    h = 0.008 * pfbm(n, 8, seed, 6) - 0.006 * pores + 0.004 * pfbm(n, 40, seed + 1, 3)
    lum = 0.86 + 0.25 * (pfbm(n, 6, seed + 2, 5) - 0.5) - 0.45 * pores
    col = lum[..., None] * np.array([1.0, 0.93, 0.86])
    return {"map": _jpeg(np.clip(col, 0, 1) * 255), "normal": _jpeg(normal_map(h, tile, 1.0), 92), "tile": tile}


# ── Materials ────────────────────────────────────────────────────────────────
def material(color=(1, 1, 1, 1), rough=0.9, metal=0.0, tex=None, uv=None, normal_scale=1.0, double=True, emissive=None):
    """tex: a dict from *_textures(); uv: ("xz", tile) or ("tri", tile) (default: the texture's own tile, triplanar)."""
    m = {"color": list(color), "rough": rough, "metal": metal, "tex": tex, "double": double, "normal_scale": normal_scale,
         "emissive": emissive}
    if tex is not None:
        m["uv"] = uv or ("tri", tex["tile"])
    return m


# ── UV projection ────────────────────────────────────────────────────────────
def project_uv(m, nrm, mode, tile):
    """-> (v, f, c, n, uv): vertices split where faces disagree on the triplanar axis."""
    if mode == "xz":
        return m.v, m.f, m.c, nrm, np.stack([m.v[:, 0], m.v[:, 2]], 1) / tile
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fa = np.argmax(np.abs(np.cross(b - a, c - a)), axis=1)              # dominant axis of every face
    keys = (m.f * 3 + fa[:, None]).ravel()
    uk, inv = np.unique(keys, return_inverse=True)
    vi, ax = uk // 3, uk % 3
    v, col, n = m.v[vi], m.c[vi], nrm[vi]
    uv = np.where((ax == 0)[:, None], v[:, [2, 1]], np.where((ax == 1)[:, None], v[:, [0, 2]], v[:, [0, 1]])) / tile
    return v, inv.reshape(-1, 3), col, n, uv


# ── Ambient occlusion ────────────────────────────────────────────────────────
def _hemi_dirs(k, seed=7):
    r = np.random.default_rng(seed)
    out = []
    for i in range(k):                                                  # stratified, cosine-weighted
        u1 = (i + r.random()) / k; u2 = r.random()
        rr, th = math.sqrt(u1), 2 * math.pi * (u2 + i * 0.618)
        out.append((rr * math.cos(th), rr * math.sin(th), math.sqrt(max(0.0, 1 - u1))))
    return np.array(out)


def bake_ao(targets, occluders, ground=None, cell=0.7, rays=20, reach=14.0, strength=0.9, gamma=1.3, pad=4.0):
    """Darken the vertex colours of every mesh in targets by how hidden each vertex is.
    occluders: meshes that cast occlusion (voxelised); ground(x, z) -> seabed height (everything below is solid)."""
    allv = np.vstack([m.v for m in targets + occluders if len(m.v)])
    lo = allv.min(0) - pad; hi = allv.max(0) + pad
    dims = np.ceil((hi - lo) / cell).astype(int) + 1
    occ = np.zeros(dims, bool)
    # voxelise the occluders: points sampled over every triangle
    for m in occluders:
        if not len(m.f):
            continue
        a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        ns = np.clip(np.ceil(area / (cell * 0.45) ** 2), 1, 400).astype(int)
        fi = np.repeat(np.arange(len(m.f)), ns)
        r1 = np.sqrt(np.random.default_rng(3).random(len(fi))); r2 = np.random.default_rng(4).random(len(fi))
        p = (1 - r1)[:, None] * a[fi] + (r1 * (1 - r2))[:, None] * b[fi] + (r1 * r2)[:, None] * c[fi]
        p = np.vstack([p, m.v])
        ijk = np.floor((p - lo) / cell).astype(int)
        ok = np.all((ijk >= 0) & (ijk < dims), axis=1)
        occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] = True
    if ground is not None:                                              # solid below the seabed
        gx = lo[0] + (np.arange(dims[0]) + 0.5) * cell; gz = lo[2] + (np.arange(dims[2]) + 0.5) * cell
        GX, GZ = np.meshgrid(gx, gz, indexing="ij")
        gy = np.floor((ground(GX, GZ) - 0.15 - lo[1]) / cell).astype(int)
        yy = np.arange(dims[1])[None, :, None]
        occ |= yy < gy[:, None, :]
    D = _hemi_dirs(rays)
    steps = np.array([1.0, 1.5, 2.2, 3.2, 4.6, 6.5, 9.0, 12.5, 17.0]) * cell
    steps = steps[steps <= reach]
    for m in targets:
        if not len(m.v):
            continue
        nr = normals(m)
        ao = np.ones(len(m.v))
        for s in range(0, len(m.v), 40000):
            p, n = m.v[s:s + 40000], nr[s:s + 40000]
            t = np.where(np.abs(n[:, 1:2]) < 0.9, np.cross(n, [0, 1, 0]), np.cross(n, [1, 0, 0]))
            t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
            bt = np.cross(n, t)
            dirs = D[None, :, 0:1] * t[:, None] + D[None, :, 1:2] * bt[:, None] + D[None, :, 2:3] * n[:, None]   # (N,R,3)
            hit = np.zeros(dirs.shape[:2]); free = np.ones(dirs.shape[:2], bool)
            o = p + n * cell * 0.9
            for d in steps:
                q = o[:, None, :] + dirs * d
                ijk = np.floor((q - lo) / cell).astype(int)
                ok = np.all((ijk >= 0) & (ijk < dims), axis=2)
                ijk = np.clip(ijk, 0, dims - 1)
                sol = occ[ijk[..., 0], ijk[..., 1], ijk[..., 2]] & ok & free
                hit += sol * (1 - 0.55 * d / reach)
                free &= ~sol
            ao[s:s + 40000] = 1 - hit.mean(1)
        f = 1 - strength * (1 - np.clip(ao, 0, 1) ** gamma)
        m.c = m.c * f[:, None]


# ── Writer ───────────────────────────────────────────────────────────────────
def _chunks(f, limit=65535):
    """Split faces into blocks that use at most `limit` vertices -> [(vertex ids, faces renumbered)]."""
    out, start, F = [], 0, len(f)
    while start < F:
        step = min(F - start, 90000)
        while True:
            u, inv = np.unique(f[start:start + step], return_inverse=True)
            if len(u) <= limit:
                break
            step = int(step * limit / len(u) * 0.97)
        out.append((u, inv.reshape(-1, 3)))
        start += step
    return out


def write_scene(path, parts, materials, pois=(), generator="Offlined Sea 3D Models scenekit", extras=None):
    """parts: [(name, Mesh, material key)]; materials: {key: material(...)};
    pois: [{"id": str, "pos": (x, y, z), "label": {"en": .., "es": ..}, "text": {..}}] -> empty nodes with extras."""
    binbuf = bytearray()
    views, accs, meshes, nodes, images, textures, mats = [], [], [], [], [], [], []
    mat_idx = {}

    def view(data, target=None):
        while len(binbuf) % 4:
            binbuf.append(0)
        off = len(binbuf); binbuf.extend(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def tex(data):
        images.append({"bufferView": view(data), "mimeType": "image/jpeg"})
        textures.append({"source": len(images) - 1, "sampler": 0})
        return len(textures) - 1

    for key, md in materials.items():
        m = {"name": key, "doubleSided": bool(md["double"]),
             "pbrMetallicRoughness": {"baseColorFactor": md["color"], "metallicFactor": md["metal"], "roughnessFactor": md["rough"]}}
        if md.get("tex"):
            m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex(md["tex"]["map"])}
            m["normalTexture"] = {"index": tex(md["tex"]["normal"]), "scale": md["normal_scale"]}
        if md.get("emissive"):
            m["emissiveFactor"] = list(md["emissive"])
        mat_idx[key] = len(mats); mats.append(m)

    tris = 0
    for name, mesh, key in parts:
        if not len(mesh.f):
            continue
        md = materials[key]
        nrm = normals(mesh)
        if md.get("tex"):
            v, f, c, n, uv = project_uv(mesh, nrm, *md["uv"])
        else:
            v, f, c, n, uv = mesh.v, mesh.f, mesh.c, nrm, None
        prims = []
        for vi, fl in _chunks(f):                       # ≤ 65,535 vertices per primitive: 16-bit indices
            pos = v[vi].astype(np.float32)
            col = np.round(_srgb_to_linear(c[vi]) * 255).astype(np.uint8)
            col = np.hstack([col, np.full((len(col), 1), 255, np.uint8)])
            nq = np.round(np.clip(n[vi], -1, 1) * 127).astype(np.int8)          # KHR_mesh_quantization: int8 normals,
            nq = np.hstack([nq, np.zeros((len(nq), 1), np.int8)])                  # padded to 4 bytes (byteStride 4)
            idx = fl.astype(np.uint16).ravel()
            a0 = len(accs)
            vn = view(nq.tobytes(), 34962); views[vn]["byteStride"] = 4
            accs += [
                {"bufferView": view(pos.tobytes(), 34962), "componentType": 5126, "count": len(pos), "type": "VEC3",
                 "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
                {"bufferView": vn, "componentType": 5120, "normalized": True, "count": len(pos), "type": "VEC3"},
                {"bufferView": view(col.tobytes(), 34962), "componentType": 5121, "normalized": True, "count": len(pos), "type": "VEC4"},
                {"bufferView": view(idx.tobytes(), 34963), "componentType": 5123, "count": len(idx), "type": "SCALAR"},
            ]
            attrs = {"POSITION": a0, "NORMAL": a0 + 1, "COLOR_0": a0 + 2}
            if uv is not None:
                accs.append({"bufferView": view(uv[vi].astype(np.float32).tobytes(), 34962), "componentType": 5126, "count": len(pos), "type": "VEC2"})
                attrs["TEXCOORD_0"] = len(accs) - 1
            prims.append({"attributes": attrs, "indices": a0 + 3, "material": mat_idx[key]})
        meshes.append({"name": name, "primitives": prims})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
        tris += len(f)
    for p in pois:
        nodes.append({"name": "poi_" + p["id"], "translation": [float(x) for x in p["pos"]],
                      "extras": {"poi": {k: p[k] for k in ("id", "label", "text", "kind", "wiki", "view") if k in p}}})
    g = {
        "asset": {"version": "2.0", "generator": generator},
        "extensionsUsed": ["KHR_mesh_quantization"], "extensionsRequired": ["KHR_mesh_quantization"],
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": mats,
        "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(binbuf)}],
    }
    if extras:
        g["scenes"][0]["extras"] = extras
    if textures:
        g.update(images=images, textures=textures, samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])
    js = json.dumps(g, separators=(",", ":"), ensure_ascii=False).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(binbuf) % 4:
        binbuf.append(0)
    total = 12 + 8 + len(js) + 8 + len(binbuf)
    with open(path, "wb") as fh:
        fh.write(b"glTF" + struct.pack("<II", 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binbuf), 0x004E4942) + bytes(binbuf))
    return tris, total
