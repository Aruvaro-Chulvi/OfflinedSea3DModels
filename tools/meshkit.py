"""
meshkit — a tiny toolkit to build wreck models by code and write them as .glb (MIT).

Meshes are numpy arrays: vertices (N,3) in metres, faces (M,3), per-vertex colours (N,3) in sRGB 0..1.
Every primitive keeps its own vertices, so normals come out sharp at the edges of boxes and smooth
along curved surfaces. Conventions of docs/MODEL_SPEC.md: +Y up, bow to +Z, y = 0 is the seabed.

    from meshkit import *
    hull = box(10, 3, 30, (0.1, 0.1, 0.1)).T(0, 0, 0)
    write_glb("model.glb", [("hull", hull)])
"""
from __future__ import annotations

import json
import math
import struct

import numpy as np


# ── Mesh ─────────────────────────────────────────────────────────────────────
class Mesh:
    def __init__(self, v=None, f=None, c=None):
        self.v = np.zeros((0, 3)) if v is None else np.asarray(v, float).reshape(-1, 3)
        self.f = np.zeros((0, 3), np.int64) if f is None else np.asarray(f, np.int64).reshape(-1, 3)
        if c is None:
            c = (0.5, 0.5, 0.5)
        c = np.asarray(c, float)
        self.c = np.tile(c, (len(self.v), 1)) if c.ndim == 1 else c.reshape(-1, 3)

    def copy(self):
        return Mesh(self.v.copy(), self.f.copy(), self.c.copy())

    def T(self, x=0.0, y=0.0, z=0.0):
        self.v = self.v + np.array([x, y, z]); return self

    def S(self, sx, sy=None, sz=None):
        self.v = self.v * np.array([sx, sx if sy is None else sy, sx if sz is None else sz]); return self

    def R(self, rx=0.0, ry=0.0, rz=0.0):
        """Rotation like three.js Euler 'XYZ' (radians)."""
        cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        self.v = self.v @ (Rx @ Ry @ Rz).T; return self

    def paint(self, fn):
        """fn(v) -> (N,3) colours, from the current vertex positions"""
        self.c = np.asarray(fn(self.v), float).reshape(-1, 3); return self

    def warp(self, fn):
        """fn(v) -> new (N,3) positions"""
        self.v = np.asarray(fn(self.v.copy()), float); return self

    @property
    def tris(self):
        return len(self.f)


def merge(ms):
    ms = [m for m in ms if m is not None and len(m.v)]
    if not ms:
        return Mesh()
    v, f, c, o = [], [], [], 0
    for m in ms:
        v.append(m.v); c.append(m.c); f.append(m.f + o); o += len(m.v)
    return Mesh(np.vstack(v), np.vstack(f), np.vstack(c))


def normals(m):
    n = np.zeros_like(m.v)
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, m.f[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


# ── Noise ────────────────────────────────────────────────────────────────────
def _hash(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h &= 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFFFF) / 4294967295.0


def noise3(p, freq=1.0, seed=0):
    """Value noise in 0..1 at points p (N,3). freq can be a scalar or (fx, fy, fz)."""
    q = np.asarray(p, float) * np.asarray(freq, float)
    i = np.floor(q).astype(np.int64); f = q - i
    u = f * f * (3 - 2 * f)
    r = np.zeros(len(q))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = (u[:, 0] if a else 1 - u[:, 0]) * (u[:, 1] if b else 1 - u[:, 1]) * (u[:, 2] if c else 1 - u[:, 2])
                r += w * _hash(i[:, 0] + a, i[:, 1] + b, i[:, 2] + c, seed)
    return r


def fbm(p, freq=1.0, seed=0, octaves=3):
    s, a, t = 0.0, 0.5, 0.0
    f = np.asarray(freq, float)
    for o in range(octaves):
        s = s + a * noise3(p, f * (2 ** o), seed + o * 17); t += a; a *= 0.5
    return s / t


def sstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def crumple(m, amp, freq, seed, weight=None):
    """Dents and buckles: every vertex pushed by 3D noise (amp in metres, weight per vertex 0..1)."""
    w = np.ones(len(m.v)) if weight is None else np.asarray(weight, float)
    d = np.stack([fbm(m.v, freq, seed + k * 101, 3) - 0.5 for k in range(3)], 1) * 2 * amp
    m.v = m.v + d * w[:, None]
    return m


# ── Primitives ───────────────────────────────────────────────────────────────
def box(w, h, d, c, bottom=True):
    """Box w (x) × h (y) × d (z), centred on x/z; standing on y = 0 if bottom else centred."""
    x, z, y0 = w / 2, d / 2, (0 if bottom else -h / 2)
    y1 = y0 + h
    P = np.array([[-x, y0, -z], [x, y0, -z], [x, y1, -z], [-x, y1, -z], [-x, y0, z], [x, y0, z], [x, y1, z], [-x, y1, z]])
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 4, 7, 3), (1, 2, 6, 5), (3, 7, 6, 2), (0, 1, 5, 4)]
    v, f = [], []
    for q in quads:
        o = len(v); v += [P[k] for k in q]; f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    return Mesh(v, f, c)


def cyl(r1, r2, h, c, seg=12, caps=True, sx=1.0):
    """Cylinder/cone along +y from 0 to h (r1 bottom, r2 top); sx squashes it on x (elliptical)."""
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ca, sa = np.cos(a) * sx, np.sin(a)
    v = [np.stack([ca * r1, np.zeros(seg), sa * r1], 1), np.stack([ca * r2, np.full(seg, h), sa * r2], 1)]
    f = []
    for i in range(seg):
        j = (i + 1) % seg
        f += [(i, seg + i, seg + j), (i, seg + j, j)]
    V = np.vstack(v)
    if caps:
        for y, r, up in ((0, r1, False), (h, r2, True)):
            if r <= 0:
                continue
            o = len(V); ring = np.stack([ca * r, np.full(seg, y), sa * r], 1)
            V = np.vstack([V, ring, [[0, y, 0]]])
            for i in range(seg):
                j = (i + 1) % seg
                f.append((o + seg, o + j, o + i) if up else (o + seg, o + i, o + j))
    return Mesh(V, f, c)


def lathe(prof, c, seg=16, axis="y"):
    """Body of revolution: prof = [(radius, position)] along +y (or +z if axis='z')."""
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    v, f = [], []
    for r, y in prof:
        v += [(r * math.cos(t), y, r * math.sin(t)) for t in a]
    n = len(prof)
    for i in range(n - 1):
        for j in range(seg):
            k = (j + 1) % seg
            p, q = i * seg + j, i * seg + k
            f += [(p, p + seg, q + seg), (p, q + seg, q)]
    m = Mesh(v, f, c)
    if axis == "z":
        m.R(math.pi / 2, 0, 0)
    return m


def grid(x0, x1, z0, z1, nx, nz, hfn, c):
    """Height field y = hfn(X, Z) over a rectangle (for mud ridges, sediment…)."""
    xs, zs = np.linspace(x0, x1, nx + 1), np.linspace(z0, z1, nz + 1)
    X, Z = np.meshgrid(xs, zs)
    Y = hfn(X, Z)
    v = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    f = []
    W = nx + 1
    for j in range(nz):
        for i in range(nx):
            a = j * W + i
            f += [(a, a + W, a + 1), (a + 1, a + W, a + W + 1)]
    return Mesh(v, f, c)


def quad(cx, cy, cz, w, h, nrm, c, eps=0.04):
    """Flat rectangle (a porthole, a window) facing the direction nrm ('+x', '-x', '+y', '+z', '-z')."""
    s = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}[nrm]
    if nrm in ("+x", "-x"):
        P = [(0, -h / 2, -w / 2), (0, -h / 2, w / 2), (0, h / 2, w / 2), (0, h / 2, -w / 2)]
    elif nrm == "+y":
        P = [(-w / 2, 0, -h / 2), (-w / 2, 0, h / 2), (w / 2, 0, h / 2), (w / 2, 0, -h / 2)]
    else:
        P = [(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)]
    P = np.array(P, float) + np.array([cx, cy, cz]) + np.array(s) * eps
    f = [(0, 1, 2), (0, 2, 3)]
    m = Mesh(P, f, c)
    if s[0] > 0 or s[2] < 0:                 # keep the front face towards nrm
        m.f = m.f[:, ::-1]
    return m


def tube(points, r, c, seg=8):
    """A pipe along a polyline (rails, davits, booms)."""
    pts = np.asarray(points, float)
    v, f = [], []
    for i, p in enumerate(pts):
        t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        t = t / (np.linalg.norm(t) or 1)
        a = np.cross(t, [0, 1, 0]) if abs(t[1]) < 0.9 else np.cross(t, [1, 0, 0])
        a /= np.linalg.norm(a); b = np.cross(t, a)
        for k in range(seg):
            th = 2 * math.pi * k / seg
            v.append(p + r * (math.cos(th) * a + math.sin(th) * b))
    for i in range(len(pts) - 1):
        for k in range(seg):
            j = (k + 1) % seg
            p0, p1 = i * seg + k, i * seg + j
            f += [(p0, p0 + seg, p1 + seg), (p0, p1 + seg, p1)]
    return Mesh(v, f, c)


# ── Colour ───────────────────────────────────────────────────────────────────
def weather(m, rust=0.5, sediment=0.5, streaks=0.6, seed=1, rust_col=(0.45, 0.2, 0.08), sed_col=(0.56, 0.53, 0.47)):
    """Rust (in vertical streaks, like the rusticles of deep wrecks), sediment on upward faces, silt low down."""
    n = normals(m)
    p = m.v
    st = fbm(p, (0.25, 0.02, 0.25), seed, 3)                                  # long vertical streaks
    pa = fbm(p, 0.08, seed + 7, 3)                                            # patches
    r = np.clip(rust * (sstep(0.45, 0.8, pa) * 0.7 + streaks * sstep(0.5, 0.75, st)), 0, 1)
    s = np.clip(sediment * sstep(0.35, 0.9, n[:, 1]) * (0.55 + 0.45 * fbm(p, 0.15, seed + 3, 2)), 0, 1)
    low = sstep(2.0, -1.0, p[:, 1]) * 0.6
    c = m.c * (1 - r[:, None]) + np.array(rust_col) * r[:, None]
    k = np.clip(s + low, 0, 1)[:, None]
    m.c = c * (1 - k) + np.array(sed_col) * k
    return m


def _srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ── glTF binary ─────────────────────────────────────────────────────────────
def write_glb(path, parts, generator="Offlined Sea 3D Models meshkit"):
    """parts: [(name, Mesh)] or [(name, Mesh, {"translation": [x, y, z], "rotation": [qx, qy, qz, qw]})]
    -> one node per part, one shared vertex-colour material."""
    binbuf = bytearray()
    views, accs, meshes, nodes = [], [], [], []
    parts = [(p[0], p[1], p[2] if len(p) > 2 else {}) for p in parts]

    def view(data, target):
        while len(binbuf) % 4:
            binbuf.append(0)
        off = len(binbuf); binbuf.extend(data)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
        return len(views) - 1

    for name, m, trs in parts:
        if not len(m.f):
            continue
        n = normals(m)
        pos = m.v.astype(np.float32)
        col = np.round(_srgb_to_linear(m.c) * 255).astype(np.uint8)
        col = np.hstack([col, np.full((len(col), 1), 255, np.uint8)])
        idx = m.f.astype(np.uint32).ravel()
        vp = view(pos.tobytes(), 34962); vn = view(n.astype(np.float32).tobytes(), 34962)
        vc = view(col.tobytes(), 34962); vi = view(idx.tobytes(), 34963)
        a0 = len(accs)
        accs += [
            {"bufferView": vp, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": vn, "componentType": 5126, "count": len(pos), "type": "VEC3"},
            {"bufferView": vc, "componentType": 5121, "normalized": True, "count": len(pos), "type": "VEC4"},
            {"bufferView": vi, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ]
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": a0, "NORMAL": a0 + 1, "COLOR_0": a0 + 2},
                                                     "indices": a0 + 3, "material": 0}]})
        node = {"name": name, "mesh": len(meshes) - 1}
        node.update({k: [float(x) for x in trs[k]] for k in ("translation", "rotation") if k in trs})
        nodes.append(node)
    g = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
        "materials": [{"name": "wreck", "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.15, "roughnessFactor": 0.88}}],
        "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(binbuf)}],
    }
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(binbuf) % 4:
        binbuf.append(0)
    total = 12 + 8 + len(js) + 8 + len(binbuf)
    with open(path, "wb") as fh:
        fh.write(b"glTF" + struct.pack("<II", 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binbuf), 0x004E4942) + bytes(binbuf))
    return sum(len(p[1].f) for p in parts), total


def yaw_quat(heading_deg):
    """Node rotation so that a piece built bow-to-+Z points to heading_deg (clockwise from north) in a model whose
    own heading is 0 (+Z = north, +X = west)."""
    a = -math.radians(heading_deg)
    return [0.0, math.sin(a / 2), 0.0, math.cos(a / 2)]
