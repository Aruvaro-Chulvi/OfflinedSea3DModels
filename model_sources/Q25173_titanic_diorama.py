"""
RMS Titanic (Q25173) — the close-up diorama: the wreck on its own piece of North Atlantic floor, 3,800 m down.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

It reuses the wreck of Q25173_titanic.py (bow, stern, debris field) and adds what you see when you take your
time: the seabed between the two halves with its ridges, ripples, burrows and stones dropped by icebergs; rust
icicles («rusticles») hanging from every edge; the animals that really live there (rattail fish, sea cucumbers
and their trails, brittle stars, squat lobsters, anemones, sea pens, glass sponges, mushroom corals,
xenophyophores); things from the ship that are easy to recognise (pairs of shoes, rows of plates, bottles,
bench ends, a safe, copper pots, an engine telegraph, Captain Smith's bathtub); and points of interest with a
short text. All on a museum-style base that shows the layers of sediment at its edges.

Nothing grows on a wreck at 3,800 m the way it does on a reef: no light, 1–2 °C, very little food. Animals
are few and small, and the wreck itself is the colourful thing (rust from iron-eating bacteria).

    python model_sources/Q25173_titanic_diorama.py   -> models/item/Q25173/diorama.glb
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, weather, sstep, fbm, normals  # noqa: E402
from scenekit import (seabed_textures, steel_textures, rust_textures, material, bake_ao, write_scene)  # noqa: E402

t0 = time.time()
spec = importlib.util.spec_from_file_location("titanic", Path(__file__).with_name("Q25173_titanic.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
rng = np.random.default_rng(41912)


def log(*a):
    print(f"[{time.time() - t0:5.1f}s]", *a, flush=True)


# ── Small helpers ────────────────────────────────────────────────────────────
def ttube(points, radii, c, seg=6, cap=True):
    """A tube along a polyline whose radius changes from point to point (arms, tentacles, stalks)."""
    pts = np.asarray(points, float); rr = np.broadcast_to(np.asarray(radii, float), (len(pts),))
    v, f = [], []
    for i, p in enumerate(pts):
        t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        t = t / (np.linalg.norm(t) or 1)
        a = np.cross(t, [0, 1, 0]) if abs(t[1]) < 0.9 else np.cross(t, [1, 0, 0])
        a /= np.linalg.norm(a); b = np.cross(t, a)
        for k in range(seg):
            th = 2 * math.pi * k / seg
            v.append(p + rr[i] * (math.cos(th) * a + math.sin(th) * b))
    for i in range(len(pts) - 1):
        for k in range(seg):
            j = (k + 1) % seg
            p0, p1 = i * seg + k, i * seg + j
            f += [(p0, p0 + seg, p1 + seg), (p0, p1 + seg, p1)]
    if cap:
        o = len(v); v.append(pts[-1])
        n0 = (len(pts) - 1) * seg
        f += [(n0 + k, n0 + (k + 1) % seg, o) for k in range(seg)]
    return Mesh(v, f, c)


def blob(r, c, seed, sub=1, amp=0.25):
    """A lumpy ball (stones, xenophyophores): an icosphere pushed by noise."""
    t = (1 + 5 ** 0.5) / 2
    V = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = list(V / np.linalg.norm(V[0]))
    for _ in range(sub):
        mid, nf = {}, []
        for a, b, cc in F:
            ab, bc, ca = [], [], []
            for (i, j), out in (((a, b), ab), ((b, cc), bc), ((cc, a), ca)):
                k = (min(i, j), max(i, j))
                if k not in mid:
                    m = (V[i] + V[j]) / 2; V.append(m / np.linalg.norm(m)); mid[k] = len(V) - 1
                out.append(mid[k])
            nf += [(a, ab[0], ca[0]), (b, bc[0], ab[0]), (cc, ca[0], bc[0]), (ab[0], bc[0], ca[0])]
        F = nf
    V = np.array(V)
    k = 1 + amp * (fbm(V * 1.3, 1.0, seed, 3) - 0.5) * 2
    return Mesh(V * k[:, None] * r, F, c)


def place(m, pos, normal=(0, 1, 0), yaw=0.0):
    """Stand a mesh built «up = +y» on a point, its up along normal, turned yaw radians around it."""
    m.R(0, yaw, 0)
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    up = np.array([0.0, 1.0, 0.0])
    ax = np.cross(up, n); s = np.linalg.norm(ax); cth = float(np.dot(up, n))
    if s > 1e-6:
        ax /= s; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = np.eye(3) + s * K + (1 - cth) * K @ K
        m.v = m.v @ Rm.T
    elif cth < 0:
        m.v = m.v * np.array([1, -1, -1])
    return m.T(*pos)


def sample_surface(m, n, cond, r, nrm=None):
    """n random points on the faces of m that pass cond(face_normal) -> [(point, normal)] (area-weighted)."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(b - a, c - a); ar = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(ar[:, None], 1e-12)
    ok = cond(fn) & (ar > 1e-4)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    pick = r.choice(idx, size=n, p=ar[idx] / ar[idx].sum())
    r1 = np.sqrt(r.random(n)); r2 = r.random(n)
    p = (1 - r1)[:, None] * a[pick] + (r1 * (1 - r2))[:, None] * b[pick] + (r1 * r2)[:, None] * c[pick]
    return list(zip(p, fn[pick]))


def mix(c1, c2, t):
    return tuple(np.array(c1) * (1 - t) + np.array(c2) * t)


# ── The wreck, with extras that must follow the bow's deformations ──────────
HA, HT, HB = T.HA, T.HT, T.HB
extras, probes = [], {}


def probe(name, d, x, y):
    probes[name] = len(extras)
    extras.append(Mesh([(x, y, T.zd(d)), (x + 0.01, y, T.zd(d)), (x, y + 0.01, T.zd(d))], [(0, 1, 2)], (0, 0, 0)))


# Captain Smith's bathtub: the wall of his quarters has fallen, the tub is in plain view on the starboard side
tub = merge([box(0.78, 0.62, 1.72, (0.86, 0.84, 0.78)), box(0.62, 0.02, 1.52, (0.32, 0.2, 0.1)).T(0, 0.55, 0),
             tube([(0.2, 0.62, -0.86), (0.2, 0.95, -0.95), (0.2, 1.05, -0.8)], 0.025, (0.5, 0.35, 0.18), seg=5),
             tube([(-0.2, 0.62, -0.86), (-0.2, 0.92, -0.95)], 0.025, (0.5, 0.35, 0.18), seg=5)])
crumple(tub, 0.03, 2.0, 5)
tub_i = len(extras); extras.append(tub.T(-10.7, HA, T.zd(69)))
probe("bow", 0.6, 0, HB + 3)
probe("telemotor", 57.5, 0, HA + 1.8)
probe("foremast", 62, 0.3, HT + 0.8)
probe("staircase", 91, 0, HT + 0.5)
probe("funnel1", 77, 0, HT + 2.6)
probe("bathtub", 69, -10.7, HA + 1.3)
probe("anchor", 8.5, T.hw(T.zd(8.5), 17.5) + 0.7, 16.5)
probe("rail", 20, -6.5, HB + 2.4)
sizes = [len(e.v) for e in extras]
T.RUN_STYLE = "drip"
W = T.build(bow_extra=extras)
bow, stern, debris = W["bow"], W["stern"], W["debris"]
ex = W["bow_extra"]
offs = np.concatenate([[0], np.cumsum(sizes)])
tub_m = Mesh(ex.v[offs[tub_i]:offs[tub_i + 1]].copy(), ex.f[(ex.f >= offs[tub_i]).all(1) & (ex.f < offs[tub_i + 1]).all(1)] - offs[tub_i],
             ex.c[offs[tub_i]:offs[tub_i + 1]].copy())
P = {k: ex.v[offs[i]:offs[i + 1]].mean(0) for k, i in probes.items()}
log("wreck built", bow.tris + stern.tris + debris.tris, "tris")

# ── The ground ───────────────────────────────────────────────────────────────
STERN_C = np.array([14.0, -600.0])
ZTIP = 143.0 / 2                                   # BOW_END / 2 in Q25173_titanic.py
_d_tab = np.linspace(-40, 300, 700)
_hw_tab = np.array([T.B / 2 * T.w_of(T.t_of(T.zd(max(0.0, min(T.L, d))))) for d in _d_tab])


def ground_base(X, Z):
    """Gentle swell of the abyssal plain (also applied to the wreck so everything stays on the ground)."""
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    return (2.2 * (fbm(P3, 0.004, 101, 3) - 0.5) + 0.5 * (fbm(P3, 0.025, 102, 3) - 0.5)).reshape(X.shape)


def ground_feat(X, Z):
    """What is shaped by the wreck and the current: the mud the bow ploughed up, the rim around the stern, sand waves."""
    d = ZTIP - Z
    hwd = np.interp(d, _d_tab, _hw_tab)
    ax = np.abs(X)
    side = 3.6 * np.exp(-((ax - hwd - 3.5) / 4.8) ** 2) * sstep(95, 25, d) * (d > -2) * (0.3 + 0.7 * sstep(4, 34, d))
    front = 2.2 * np.exp(-(X / 8) ** 2) * np.exp(-((d + 5) / 6) ** 2)
    edge = sstep(46, 32, ax) * sstep(-28, -16, -d) * sstep(115, 95, d)
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    nz = fbm(P3, 0.2, 5, 2).reshape(X.shape)
    r_ = np.maximum(side, front) * edge
    ridge = np.where(r_ > 0, r_ * r_ / (r_ + 0.5), 0) * (0.85 + 0.3 * nz)
    # behind the bow: the furrow it dug when it hit the bottom at an angle
    furrow = -0.8 * np.exp(-(X / 9) ** 2) * sstep(-75, -95, Z - 0) * sstep(-190, -150, Z) * (Z < -70)
    # the stern: hit hard and flat; a rim of thrown-up mud, rays of it, and a hollow under the heap
    ex_, ez = (X - STERN_C[0]) / 40, (Z - STERN_C[1]) / 74
    r = np.sqrt(ex_ ** 2 + ez ** 2)
    ang = np.arctan2(ez, ex_)
    ang = ang.ravel()
    rays = fbm(np.stack([np.cos(ang) * 6, np.sin(ang) * 6, np.zeros(ang.size)], 1), 1.0, 9, 2).reshape(X.shape)
    rim = 1.7 * np.exp(-((r - 1.02) / 0.2) ** 2) * (0.6 + 0.8 * fbm(P3, 0.06, 7, 2).reshape(X.shape))
    ejecta = 0.6 * np.exp(-((r - 1.35) / 0.35) ** 2) * sstep(0.45, 0.75, rays)
    # sand waves where the current runs free (the Western Boundary Undercurrent sweeps this plain)
    warp = fbm(P3, 0.01, 11, 2).reshape(X.shape)
    sw = 0.16 * np.sin(2 * math.pi * (X * 0.35 + Z * 0.94) / 11 + 6 * warp) * sstep(0.45, 0.7, fbm(P3, 0.008, 12, 2).reshape(X.shape))
    return ridge + furrow + rim + ejecta + sw * sstep(1.4, 2.0, r)


def ground(X, Z):
    X, Z = np.asarray(X, float), np.asarray(Z, float)
    return ground_base(X, Z) + ground_feat(X, Z)


# Everything sits on the swell of the plain
for m in (bow, stern, debris, tub_m):
    m.v[:, 1] += ground_base(m.v[:, 0], m.v[:, 2])
for k in P:
    P[k] = P[k] + np.array([0, ground_base(np.array([P[k][0]]), np.array([P[k][2]]))[0], 0])

# Footprint of the wreck on a 1 m raster (to keep animals out of the hull and to stain the mud with rust)
GX0, GX1, GZ0, GZ1 = -140.0, 160.0, -830.0, 120.0
nxr, nzr = int(GX1 - GX0), int(GZ1 - GZ0)
foot = np.zeros((nxr, nzr))
for m, wgt in ((bow, 1.0), (stern, 1.0), (debris, 0.6)):
    lowv = m.v[m.v[:, 1] < ground(m.v[:, 0], m.v[:, 2]) + 2.5]
    ii = np.clip((lowv[:, 0] - GX0).astype(int), 0, nxr - 1); jj = np.clip((lowv[:, 2] - GZ0).astype(int), 0, nzr - 1)
    np.add.at(foot, (ii, jj), wgt)
occupied = foot > 0


def blur(a, r, it=3):
    for _ in range(it):
        for ax in (0, 1):
            c = np.cumsum(np.pad(a, [(r + 1, r) if k == ax else (0, 0) for k in range(2)], mode="edge"), axis=ax)
            a = (np.take(c, range(2 * r + 1, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - 2 * r - 1), axis=ax)) / (2 * r + 1)
    return a


stain = np.clip(blur(np.minimum(foot, 3.0), 4) * 1.4, 0, 1)
near = np.clip(blur(occupied.astype(float), 18) * 6, 0, 1)


def raster(a, X, Z):
    ii = np.clip((np.asarray(X) - GX0).astype(int), 0, nxr - 1); jj = np.clip((np.asarray(Z) - GZ0).astype(int), 0, nzr - 1)
    return a[ii, jj]


# ── The base: a rounded rectangle; the seabed is cut out of it like a museum diorama ──
BX0, BX1, BZ0, BZ1, BR = -132.0, 152.0, -822.0, 114.0, 46.0


def inside_d(X, Z):
    """Signed distance to the edge of the base (negative inside)."""
    cx, cz = (BX0 + BX1) / 2, (BZ0 + BZ1) / 2
    hx, hz = (BX1 - BX0) / 2 - BR, (BZ1 - BZ0) / 2 - BR
    qx, qz = np.abs(X - cx) - hx, np.abs(Z - cz) - hz
    return np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - BR


def to_edge(X, Z):
    """Move points outside the base onto its edge."""
    cx, cz = (BX0 + BX1) / 2, (BZ0 + BZ1) / 2
    hx, hz = (BX1 - BX0) / 2 - BR, (BZ1 - BZ0) / 2 - BR
    dx, dz = X - cx, Z - cz
    qx, qz = np.clip(dx, -hx, hx), np.clip(dz, -hz, hz)
    ox, oz = dx - qx, dz - qz
    ln = np.hypot(ox, oz)
    out = ln > BR
    sx = np.where(out, qx + ox / np.maximum(ln, 1e-9) * BR, dx)
    sz = np.where(out, qz + oz / np.maximum(ln, 1e-9) * BR, dz)
    return sx + cx, sz + cz


def cell_size(cx, cz):
    """Size of the ground cells: 1 m around the two halves, 2 m in the debris field and along the edge, 4 m elsewhere."""
    nb = (np.abs(cx) < 50) & (cz > -105) & (cz < 112)
    ns = (np.hypot((cx - STERN_C[0]) / 64, (cz - STERN_C[1]) / 104) < 1)
    deb = (cz > -720) & (cz < -80) & (cx > -70) & (cx < 110)
    edge = inside_d(cx, cz) > -10
    s = np.full(cx.shape, 4.0)
    s[deb | edge | (raster(near, cx, cz) > 0.05)] = 2.0
    s[nb | ns] = 1.0
    return s


def build_ground():
    cells = []                                                          # (x0, z0, size)
    xs = np.arange(BX0, BX1, 4.0); zs = np.arange(BZ0, BZ1, 4.0)
    X, Z = np.meshgrid(xs, zs, indexing="ij"); X, Z = X.ravel(), Z.ravel()
    keep = inside_d(X + 2, Z + 2) < 3.0
    X, Z = X[keep], Z[keep]
    want = cell_size(X + 2, Z + 2)
    todo = [(X, Z, np.full(X.shape, 4.0), want)]
    while todo:
        X, Z, S, Wn = todo.pop()
        done = S <= Wn
        cells += list(zip(X[done], Z[done], S[done]))
        X, Z, S = X[~done], Z[~done], S[~done] / 2
        if len(X):
            for dx in (0, 1):
                for dz in (0, 1):
                    nx_, nz_ = X + dx * S, Z + dz * S
                    todo.append((nx_, nz_, S, cell_size(nx_ + S / 2, nz_ + S / 2)))
    cells = np.array(cells)
    cx0, cz0, cs = cells[:, 0], cells[:, 1], cells[:, 2]
    keep = inside_d(cx0 + cs / 2, cz0 + cs / 2) < cs * 0.71
    cx0, cz0, cs = cx0[keep], cz0[keep], cs[keep]
    q = 0.5                                                             # vertices on a lattice of 0.5 m
    corners = np.stack([np.stack([cx0, cz0], 1), np.stack([cx0 + cs, cz0], 1), np.stack([cx0 + cs, cz0 + cs], 1), np.stack([cx0, cz0 + cs], 1)], 1)
    keys = np.round((corners - [BX0, BZ0]) / q).astype(np.int64)
    kk = keys[..., 0] * 100000 + keys[..., 1]
    uk, inv = np.unique(kk.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 4)
    vx = BX0 + (uk // 100000) * q; vz = BZ0 + (uk % 100000) * q
    # the cells that stick out of the base: their outer corners go onto the edge
    ex_, ez_ = to_edge(vx, vz)
    vx, vz = np.where(inside_d(vx, vz) > 0, ex_, vx), np.where(inside_d(vx, vz) > 0, ez_, vz)
    vy = ground(vx, vz)
    # no cracks between big and small cells: the vertices along the edge of a big cell follow that edge
    lookup = {int(k): i for i, k in enumerate(uk)}
    order = np.argsort(-cs)
    for ci in order:
        s = cs[ci]
        if s <= 1.0:
            break
        k0 = keys[ci]
        n = int(round(s / q))
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            ka, kb = k0[a], k0[b]
            ia, ib = inv[ci, a], inv[ci, b]
            for t in range(1, n):
                kp = ka + (kb - ka) * t // n
                j = lookup.get(int(kp[0] * 100000 + kp[1]))
                if j is not None:
                    vy[j] = vy[ia] + (vy[ib] - vy[ia]) * t / n
    v = np.stack([vx, vy, vz], 1)
    f = np.concatenate([inv[:, [0, 2, 1]], inv[:, [0, 3, 2]]])
    # if a big cell has finer neighbours, its two triangles still span the whole edge: no holes, only T-joints
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    good = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-6
    return Mesh(v, f[good], (1, 1, 1))


seabed = build_ground()
log("seabed", seabed.tris, "tris", len(seabed.v), "verts")

# colour of the mud: the texture carries the grain; the vertices carry stains and the light
p = seabed.v
tone = 0.93 + 0.16 * (fbm(p, 0.02, 21, 3) - 0.5) + 0.10 * (fbm(p, 0.09, 22, 2) - 0.5) + 0.05 * (fbm(p, 0.4, 25, 2) - 0.5)
col = np.ones((len(p), 3)) * tone[:, None]
st_ = raster(stain, p[:, 0], p[:, 2]) * (0.6 + 0.6 * fbm(p, 0.15, 23, 3))
rustc = np.array([0.80, 0.50, 0.32])
col = col * (1 - np.clip(st_, 0, 1)[:, None] * (1 - rustc))
rr = np.hypot((p[:, 0] - STERN_C[0]) / 40, (p[:, 2] - STERN_C[1]) / 74)
dist = np.exp(-((rr - 1.1) / 0.45) ** 2) * 0.12 * fbm(p, 0.08, 24, 2)
col = col * (1 - dist[:, None])                                         # darker, stirred-up mud around the stern
seabed.c = np.clip(col, 0, 1)

# ── The edge of the base: layers of sediment over a black plinth ────────────
def build_base():
    per = []
    cx, cz = (BX0 + BX1) / 2, (BZ0 + BZ1) / 2
    hx, hz = (BX1 - BX0) / 2 - BR, (BZ1 - BZ0) / 2 - BR
    for (sx, sz, a0) in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):       # corners and straight runs, CCW
        ccx, ccz = cx + sx * hx, cz + sz * hz
        for a in np.linspace(a0, a0 + 90, 24, endpoint=False):
            per.append((ccx + BR * math.cos(math.radians(a)), ccz + BR * math.sin(math.radians(a))))
        nxt = (cx - sx * hx, ccz) if sx * sz > 0 else (ccx, cz - sz * hz)
        a1 = math.radians(a0 + 90)
        e0 = np.array([ccx + BR * math.cos(a1), ccz + BR * math.sin(a1)])
        e1 = np.array([nxt[0] + BR * math.cos(a1), nxt[1] + BR * math.sin(a1)])
        L_ = np.linalg.norm(e1 - e0)
        for t in np.linspace(0, 1, max(2, int(L_ / 2.5)), endpoint=False):
            per.append(tuple(e0 + (e1 - e0) * t))
    per = np.array(per)
    nP = len(per)
    top = ground(per[:, 0], per[:, 1]) + 0.04
    ybot = -7.0
    depths = np.concatenate([np.linspace(0, 0.5, 4), np.linspace(0.8, 5.5, 13)])
    layers = [(0.0, (0.70, 0.63, 0.52)), (0.35, (0.58, 0.56, 0.49)), (1.1, (0.47, 0.46, 0.41)), (1.5, (0.33, 0.31, 0.29)),
              (1.65, (0.66, 0.64, 0.58)), (2.6, (0.52, 0.49, 0.43)), (3.4, (0.42, 0.40, 0.36)), (4.4, (0.56, 0.52, 0.45))]
    v, c, f = [], [], []
    outn = np.stack([per[:, 0] - cx, per[:, 1] - cz], 1)
    for i, dpt in enumerate(depths):
        y = np.maximum(top - dpt, ybot + 1.3) if dpt > 0 else top
        wav = 0.12 * np.sin(per[:, 0] * 0.05 + per[:, 1] * 0.03) + 0.08 * (fbm(np.stack([per[:, 0], np.full(nP, dpt), per[:, 1]], 1), 0.1, 31, 2) - 0.5)
        for k in range(nP):
            dd = dpt + wav[k]
            lc = layers[0][1]
            for d0, cc in layers:
                if dd >= d0:
                    lc = cc
            peb = 0.0
            if 2.6 < dd < 3.4 and ((k * 7 + i * 13) % 5 == 0):
                peb = 0.18                                                        # glacial pebbles
            c.append(np.array(lc) * (1 - peb))
            v.append((per[k, 0], y[k], per[k, 1]))
    R = len(depths)
    for i in range(R - 1):
        for k in range(nP):
            a = i * nP + k; b = i * nP + (k + 1) % nP
            f += [(a, b, a + nP), (b, b + nP, a + nP)]
    strata = Mesh(v, f, np.array(c))
    # the plinth: black, a little wider, with a bevel
    def ring(off, y):
        nn = per - [cx, cz]; ln = np.linalg.norm(nn, axis=1, keepdims=True)
        # outwards along the local normal of the edge (approximate with the corner geometry)
        qx, qz = np.clip(per[:, 0] - cx, -hx, hx), np.clip(per[:, 1] - cz, -hz, hz)
        o = per - (np.stack([qx, qz], 1) + [cx, cz]); o /= np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-9)
        q = per + o * off
        return np.stack([q[:, 0], np.full(nP, y), q[:, 1]], 1)
    rings = [ring(0.0, ybot + 1.3), ring(0.9, ybot + 1.3), ring(1.4, ybot + 0.9), ring(1.4, ybot), ring(0.0, ybot)]
    v2 = np.vstack(rings); f2 = []
    for i in range(len(rings) - 1):
        for k in range(nP):
            a = i * nP + k; b = i * nP + (k + 1) % nP
            f2 += [(a, b, a + nP), (b, b + nP, a + nP)]
    o = len(v2); v2 = np.vstack([v2, [[cx, ybot, cz]]])
    f2 += [(4 * nP + k, o, 4 * nP + (k + 1) % nP) for k in range(nP)]
    plinth = Mesh(v2, f2, (0.035, 0.035, 0.04))
    # a brass plate on the long side, under the bow
    plate = box(16, 0.7, 0.06, (0.7, 0.55, 0.25), bottom=False).R(0, 0, 0).T(cx, ybot + 0.65, BZ1 + 1.45)
    return strata, plinth, plate


strata, plinth, plate = build_base()
log("base", strata.tris + plinth.tris)

# ── Rust: icicles and runs ───────────────────────────────────────────────────
def rusticle(length, r0, seed):
    rr_ = np.random.default_rng(seed)
    n = 5
    prof = []
    ph = rr_.uniform(0, 6.28)
    for i in range(n + 1):
        t = i / n
        rad = r0 * (1 - t) ** 0.75 * (1 + 0.22 * math.sin(t * 9 + ph)) + 0.004
        prof.append((rad, -length * t))
    m = lathe(prof, (0.5, 0.25, 0.1), seg=5)
    y = m.v[:, 1] / -length
    m.c = np.array([mix((0.36, 0.14, 0.06), (0.82, 0.5, 0.2), sstep(0.15, 1.0, t) * rr_.uniform(0.6, 1.0)) for t in y])
    bend = rr_.normal(0, 0.06, 2)
    m.v[:, 0] += bend[0] * y ** 2 * length; m.v[:, 2] += bend[1] * y ** 2 * length
    return m


def rust_flow(length, w, seed):
    """A run of rust down a wall: a thin, lumpy drip (built pointing down -y, bulging towards +z), dark at the
    top where it comes out of a seam or a porthole, orange where it is fresh."""
    rr_ = np.random.default_rng(seed)
    prof = [(0.001, 0)]
    for t in np.linspace(0.08, 0.95, 6):
        prof.append((w * (0.55 + 0.45 * math.sin(t * math.pi)) * (1 + 0.35 * rr_.uniform(-1, 1)), -length * t))
    prof.append((0.001, -length))
    m = lathe(prof, (0.5, 0.25, 0.1), seg=5)
    m.v[:, 2] = np.abs(m.v[:, 2]) * 0.45
    hot = rr_.uniform(0.3, 1.0)
    m.c = np.array([mix((0.30, 0.12, 0.05), (0.72, 0.38, 0.15), sstep(0.1, 1.0, -y / length) * hot) for y in m.v[:, 1]])
    crumple(m, w * 0.25, 2.5 / w, seed)
    return m


def attach_rust(m, n_ice, n_flow, seed):
    r = np.random.default_rng(seed)
    out = []
    for p, nn in sample_surface(m, n_ice, lambda fn: fn[:, 1] < -0.35, r):
        ln = r.uniform(0.3, 1.0) ** 1.6 * 2.4 + 0.15
        out.append(rusticle(ln, r.uniform(0.04, 0.12) * (0.6 + ln / 3), int(r.integers(1e9))).T(*p))
    for p, nn in sample_surface(m, n_flow, lambda fn: np.abs(fn[:, 1]) < 0.3, r):
        ln = r.uniform(0.3, 1.0) ** 1.5 * 3.2 + 0.3
        fl = rust_flow(ln, r.uniform(0.04, 0.11), int(r.integers(1e9)))
        yaw = math.atan2(nn[0], nn[2])
        out.append(fl.R(0, yaw, 0).T(*(p + nn * 0.02)))
    return merge(out)


# more rust than the light model: up close the hull is a patchwork of rust, not black paint
for m_, sd in ((bow, 201), (stern, 202), (debris, 203)):
    lum = m_.c.mean(1)
    k = 0.5 * sstep(0.25, 0.05, lum)                                    # the black paint has mostly gone brown
    m_.c = m_.c * (1 - k[:, None]) + np.array([0.24, 0.12, 0.07]) * k[:, None]
    weather(m_, rust=0.55, sediment=0.12, streaks=1.0, seed=sd, rust_col=(0.42, 0.17, 0.07))
    m_.c = np.clip(m_.c * (0.85 + 0.3 * fbm(m_.v, 0.6, sd + 5, 2))[:, None], 0, 1)
rust = merge([attach_rust(bow, 1500, 1500, 1), attach_rust(stern, 600, 600, 2), attach_rust(debris, 150, 120, 3)])
log("rust", rust.tris)

# ── Seabed life and stones ──────────────────────────────────────────────────
def free_spot(r, weight_near=0.7, avoid=True, tries=60):
    for _ in range(tries):
        if r.random() < weight_near:
            # near the wreck: choose among the stained/near cells
            k = r.integers(0, len(_near_idx[0]))
            x = GX0 + _near_idx[0][k] + r.random(); z = GZ0 + _near_idx[1][k] + r.random()
        else:
            x = r.uniform(BX0 + 4, BX1 - 4); z = r.uniform(BZ0 + 4, BZ1 - 4)
        if inside_d(np.array([x]), np.array([z]))[0] > -3:
            continue
        if avoid and raster(occupied, x, z):
            continue
        return x, z
    return None


_near_idx = np.nonzero((near > 0.15) & ~occupied)


def gy(x, z):
    return float(ground(np.array([x]), np.array([z]))[0])


def gnormal(x, z, e=0.5):
    hx = gy(x + e, z) - gy(x - e, z); hz = gy(x, z + e) - gy(x, z - e)
    n = np.array([-hx / (2 * e), 1.0, -hz / (2 * e)]); return n / np.linalg.norm(n)


life, rocks, trails, artefacts, china, glass = [], [], [], [], [], []
pois = []


def poi(id_, pos, en, es, ten, tes, kind="wreck", view=None):
    """view: how far the camera stands when it flies to it (m)."""
    q = {"id": id_, "pos": [float(x) for x in pos], "kind": kind, "label": {"en": en, "es": es}, "text": {"en": ten, "es": tes}}
    if view:
        q["view"] = view
    pois.append(q)


# dropstones: rocks carried by icebergs and dropped when the ice melted (common on this plain)
stones = []
for i in range(46):
    s = free_spot(rng, 0.55)
    if not s:
        continue
    x, z = s
    rad = rng.uniform(0.25, 0.9) ** 2 * 2.6 + 0.15
    st = blob(rad, (0.42, 0.41, 0.4), 300 + i, sub=2, amp=0.35).S(1, rng.uniform(0.5, 0.8), 1)
    st.paint(lambda q, i=i: np.array(rng.choice([(0.38, 0.37, 0.36), (0.5, 0.46, 0.42), (0.3, 0.3, 0.32)])) * (0.8 + 0.4 * fbm(q, 2.0, i, 2))[:, None])
    weather(st, rust=0.08, sediment=0.35, streaks=0.0, seed=i, sed_col=(0.62, 0.58, 0.5))
    st.c = st.c * 0.8
    rocks.append(st.R(0, rng.uniform(0, 6.28), 0).T(x, gy(x, z) - rad * 0.25, z))
    stones.append((x, z, rad))
rocks = merge(rocks)
log("rocks", rocks.tris)


def brittle_star(sz, c):
    parts = [cyl(sz * 0.22, sz * 0.18, sz * 0.08, c, seg=6)]
    a0 = rng.uniform(0, 6.28)
    for k in range(5):
        a = a0 + k * 2 * math.pi / 5
        curl = rng.uniform(-0.9, 0.9)
        pts, rad = [], []
        for t in np.linspace(0, 1, 5):
            aa = a + curl * t * t
            rr_ = sz * (0.18 + 0.95 * t)
            pts.append((rr_ * math.cos(aa), sz * 0.04 + 0.03 * sz * math.sin(t * 3), rr_ * math.sin(aa)))
            rad.append(sz * (0.06 * (1 - t) + 0.012))
        parts.append(ttube(pts, rad, c, seg=3))
    return merge(parts)


def anemone(sz, c):
    col = mix(c, (0.95, 0.92, 0.88), 0.3)
    parts = [cyl(sz * 0.32, sz * 0.38, sz * 0.9, col, seg=7), cyl(sz * 0.4, sz * 0.4, 0.01, mix(c, (0.3, 0.1, 0.1), 0.3), seg=7).T(0, sz * 0.9, 0)]
    n = 11
    for k in range(n):
        a = 2 * math.pi * k / n + rng.uniform(-0.1, 0.1)
        ln = sz * rng.uniform(0.7, 1.1)
        pts = [(sz * 0.36 * math.cos(a), sz * 0.9, sz * 0.36 * math.sin(a)),
               ((sz * 0.36 + ln * 0.5) * math.cos(a), sz * 0.9 + ln * 0.45, (sz * 0.36 + ln * 0.5) * math.sin(a)),
               ((sz * 0.36 + ln * 0.85) * math.cos(a), sz * 0.9 + ln * 0.5, (sz * 0.36 + ln * 0.85) * math.sin(a))]
        parts.append(ttube(pts, [sz * 0.05, sz * 0.035, sz * 0.012], c, seg=3))
    return merge(parts)


def mushroom_coral(sz):
    red = (0.62, 0.08, 0.08)
    parts = [cyl(sz * 0.3, sz * 0.26, sz * 0.7, (0.75, 0.25, 0.2), seg=8),
             lathe([(0.001, sz * 1.15), (sz * 0.45, sz * 1.08), (sz * 0.7, sz * 0.9), (sz * 0.62, sz * 0.7), (sz * 0.25, sz * 0.66)], red, seg=12)]
    for k in range(7):
        a = rng.uniform(0, 6.28); rr_ = rng.uniform(0, sz * 0.45)
        parts.append(cyl(sz * 0.05, 0.002, sz * 0.18, (0.95, 0.92, 0.88), seg=4).T(rr_ * math.cos(a), sz * 1.08, rr_ * math.sin(a)))
    return merge(parts)


def squat_lobster(sz):
    c = rng.choice([(0.93, 0.9, 0.84), (0.95, 0.62, 0.35), (0.9, 0.82, 0.7)])
    body = lathe([(0.001, -sz * 0.5), (sz * 0.18, -sz * 0.35), (sz * 0.22, 0), (sz * 0.16, sz * 0.35), (0.001, sz * 0.5)], c, seg=8, axis="z").S(1, 0.6, 1)
    parts = [body.T(0, sz * 0.12, 0)]
    for s in (-1, 1):
        parts.append(ttube([(s * sz * 0.12, sz * 0.12, sz * 0.35), (s * sz * 0.5, sz * 0.25, sz * 0.9), (s * sz * 0.55, sz * 0.2, sz * 1.6)],
                           [sz * 0.05, sz * 0.04, sz * 0.02], c, seg=4))
        for k in range(3):
            zz = sz * (0.2 - 0.18 * k)
            parts.append(ttube([(s * sz * 0.15, sz * 0.12, zz), (s * sz * 0.45, sz * 0.25, zz + sz * 0.05), (s * sz * 0.65, 0.0, zz + sz * 0.1)],
                               [sz * 0.03, sz * 0.025, sz * 0.01], c, seg=3))
    return merge(parts)


def sea_cucumber(kind):
    if kind == "psychropotes":                     # yellowish, translucent, with a tall «sail» at the back
        ln, c = rng.uniform(0.35, 0.6), (0.86, 0.72, 0.42)
    else:                                          # Benthodytes: deep purple
        ln, c = rng.uniform(0.25, 0.45), (0.38, 0.12, 0.3)
    body = lathe([(0.001, -ln / 2), (ln * 0.12, -ln * 0.4), (ln * 0.17, -ln * 0.1), (ln * 0.16, ln * 0.25), (ln * 0.1, ln * 0.45), (0.001, ln / 2)],
                 c, seg=10, axis="z")
    body.v[:, 1] = np.maximum(body.v[:, 1], -ln * 0.02)
    body.S(1.25, 0.65, 1).T(0, ln * 0.06, 0)
    parts = [body]
    if kind == "psychropotes":
        sail = merge([box(ln * 0.05, ln * 0.6, ln * 0.18, mix(c, (1, 1, 1), 0.2))]).R(-0.5, 0, 0).T(0, ln * 0.1, -ln * 0.38)
        parts.append(sail)
    for k in range(6):                                                  # little tube feet along the sides
        zz = -ln * 0.35 + ln * 0.7 * k / 5
        for s in (-1, 1):
            parts.append(cyl(ln * 0.018, ln * 0.008, ln * 0.06, c, seg=3, caps=False).R(0, 0, s * 1.1).T(s * ln * 0.2, ln * 0.04, zz))
    return merge(parts)


def rattail(ln):
    c = (0.5, 0.47, 0.42)
    body = lathe([(0.001, ln * 0.0), (ln * 0.07, ln * 0.04), (ln * 0.085, ln * 0.12), (ln * 0.07, ln * 0.25), (ln * 0.035, ln * 0.5), (ln * 0.012, ln * 0.8), (0.001, ln)],
                 c, seg=10, axis="z")
    body.S(0.75, 1, 1)
    body.v[:, 2] = -body.v[:, 2]                                        # head at +z
    body.paint(lambda q: np.array(c) * (0.85 + 0.25 * sstep(-0.02, 0.06, q[:, 1]))[:, None])
    parts = [body]
    for s in (-1, 1):
        parts.append(cyl(ln * 0.026, ln * 0.026, 0.004, (0.03, 0.04, 0.06), seg=8).R(0, 0, s * math.pi / 2).T(s * ln * 0.05, ln * 0.025, -ln * 0.06))
    parts.append(box(0.004, ln * 0.12, ln * 0.12, mix(c, (0.2, 0.2, 0.2), 0.3), bottom=True).R(-0.4, 0, 0).T(0, ln * 0.05, -ln * 0.2))   # first dorsal fin
    parts.append(box(0.003, ln * 0.03, ln * 0.6, mix(c, (0.2, 0.2, 0.2), 0.3), bottom=True).T(0, ln * 0.02, -ln * 0.62))              # long fin to the tail
    parts.append(box(0.003, ln * 0.025, ln * 0.55, mix(c, (0.2, 0.2, 0.2), 0.3), bottom=False).T(0, -ln * 0.03, -ln * 0.6))
    parts.append(tube([(0, -ln * 0.06, ln * 0.06), (0, -ln * 0.1, ln * 0.04)], ln * 0.004, c, seg=3))                                 # chin barbel
    for s in (-1, 1):
        parts.append(box(0.003, ln * 0.04, ln * 0.12, c, bottom=False).R(0, s * 0.5, s * 0.6).T(s * ln * 0.06, -ln * 0.01, ln * 0.13))  # pectorals
    m = merge(parts)
    m.v[:, 2] -= ln * 0.0
    return m


def sea_pen(h):
    c = (0.9, 0.86, 0.72)
    lean = rng.uniform(0.05, 0.25)                                     # leaning with the current (to the north-east)
    pts = [(0, 0, 0)] + [(lean * h * t * t * 0.6, h * t, lean * h * t * t) for t in np.linspace(0.15, 1, 7)]
    parts = [ttube(pts, np.linspace(0.016, 0.008, len(pts)), (0.82, 0.78, 0.62), seg=5)]
    tip = np.array(pts[-1])
    for k in range(rng.integers(5, 10)):
        a = rng.uniform(0, 6.28); ln = rng.uniform(0.06, 0.13)
        e = tip + np.array([math.cos(a) * ln, -ln * rng.uniform(0.2, 0.8), math.sin(a) * ln])
        parts.append(ttube([tip, (tip + e) / 2 + [0, 0.02, 0], e], [0.022, 0.018, 0.006], c, seg=5))
    return merge(parts)


def glass_sponge(h):
    prof = [(0.03 * h, 0), (0.12 * h, 0.1 * h), (0.16 * h, 0.4 * h), (0.22 * h, 0.8 * h), (0.25 * h, h)]
    m = lathe(prof, (0.88, 0.86, 0.78), seg=14)
    m.paint(lambda q: (np.array((0.9, 0.88, 0.8)) * (0.8 + 0.25 * (np.sin(q[:, 1] * 60) * np.sin(np.arctan2(q[:, 2], q[:, 0]) * 14) > 0.3))[:, None]))
    inner = lathe([(r * 0.9, y + 0.01) for r, y in prof], (0.55, 0.52, 0.46), seg=14)
    inner.f = inner.f[:, ::-1]
    return merge([m, inner])


def bamboo_coral(h, seed):
    r = np.random.default_rng(seed)
    parts = []

    def branch(p, d, ln, rad, depth):
        pts = [p]
        for t in range(1, 5):
            d = d + r.normal(0, 0.12, 3); d[1] = abs(d[1]) + 0.2; d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * ln / 4)
        cols = [(0.92, 0.9, 0.8) if k % 2 == 0 else (0.3, 0.2, 0.12) for k in range(len(pts))]
        m = ttube(pts, np.linspace(rad, rad * 0.6, len(pts)), (0.9, 0.88, 0.78), seg=4)
        m.c = np.repeat(np.array(cols), 4, axis=0)[:len(m.v)] if len(m.v) == len(pts) * 4 else m.c
        parts.append(m)
        if depth < 3:
            for _ in range(2):
                branch(pts[r.integers(2, 5)], d + r.normal(0, 0.6, 3), ln * 0.65, rad * 0.6, depth + 1)
    branch(np.zeros(3), np.array([0, 1.0, 0]), h * 0.5, 0.025, 0)
    return merge(parts)


def xeno(r_):
    return blob(r_, (0.84, 0.8, 0.68), int(rng.integers(1e6)), sub=1, amp=0.5).S(1, 0.6, 1)


def mound(r_):
    m = lathe([(r_, 0), (r_ * 0.6, r_ * 0.25), (r_ * 0.22, r_ * 0.38), (r_ * 0.08, r_ * 0.3), (0.001, r_ * 0.1)], (0.68, 0.64, 0.56), seg=7)
    m.paint(lambda q: np.where((np.hypot(q[:, 0], q[:, 2]) < r_ * 0.12)[:, None], (0.32, 0.29, 0.25), (0.66, 0.62, 0.54)))
    return m


# animals on the seabed
counts = {"brittle": 0}
for i in range(1000):                                                    # brittle stars, everywhere, more near the wreck
    s = free_spot(rng, 0.65)
    if not s:
        continue
    x, z = s
    c = rng.choice([(0.85, 0.7, 0.58), (0.92, 0.88, 0.82), (0.78, 0.55, 0.45)])
    life.append(place(brittle_star(rng.uniform(0.06, 0.13), c), (x, gy(x, z) + 0.01, z), gnormal(x, z), rng.uniform(0, 6.28)))
for i in range(320):                                                     # burrow mounds with their hole
    s = free_spot(rng, 0.3)
    if s:
        x, z = s
        life.append(place(mound(rng.uniform(0.12, 0.35)), (x, gy(x, z) - 0.01, z), gnormal(x, z), 0))
for i in range(420):                                                     # xenophyophores: single cells as big as a fist
    s = free_spot(rng, 0.4)
    if s:
        x, z = s
        life.append(xeno(rng.uniform(0.04, 0.1)).T(x, gy(x, z) + 0.01, z))
cucumbers = []
for i in range(110):                                                     # sea cucumbers and the trails they leave
    s = free_spot(rng, 0.45)
    if not s:
        continue
    x, z = s
    kind = "psychropotes" if rng.random() < 0.45 else "benthodytes"
    yaw = rng.uniform(0, 6.28)
    life.append(place(sea_cucumber(kind), (x, gy(x, z), z), gnormal(x, z), yaw))
    cucumbers.append((x, z, kind))
    # trail: a winding strip of sediment that has been through its gut, slightly darker
    hx_, hz_ = math.sin(yaw), math.cos(yaw)
    pts, a = [], yaw
    px_, pz_ = x - hx_ * 0.3, z - hz_ * 0.3
    for k in range(int(rng.uniform(12, 40))):
        pts.append((px_, pz_)); a += rng.normal(0, 0.12)
        px_ -= math.sin(a) * 0.35; pz_ -= math.cos(a) * 0.35
    if len(pts) > 2:
        pts = np.array(pts)
        tv, tf, tc = [], [], []
        for k in range(len(pts)):
            d = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]; d /= np.linalg.norm(d) or 1
            nrm_ = np.array([-d[1], d[0]]) * 0.13
            fade = min(1.0, k / 4) * (1 - k / len(pts)) ** 0.5
            for s_ in (-1, 1):
                q = pts[k] + nrm_ * s_
                tv.append((q[0], gy(q[0], q[1]) + 0.035, q[1]))
                tc.append(np.array((1, 1, 1)) * (1 - 0.09 * fade))
        for k in range(len(pts) - 1):
            a0 = 2 * k
            tf += [(a0, a0 + 2, a0 + 1), (a0 + 1, a0 + 2, a0 + 3)]
        trails.append(Mesh(tv, tf, np.array(tc)))
pens = []
for i in range(60):                                                      # sea pens (Umbellula): a tuft on a tall stalk
    s = free_spot(rng, 0.15)
    if s:
        x, z = s
        life.append(sea_pen(rng.uniform(0.8, 2.2)).T(x, gy(x, z) - 0.05, z))
        pens.append(life[-1])
fish = []
for i in range(45):                                                      # grenadiers (rattails), hovering head-down
    s = free_spot(rng, 0.6)
    if not s:
        continue
    x, z = s
    ln = rng.uniform(0.45, 0.85)
    m = rattail(ln).R(-0.45, 0, 0).R(0, rng.uniform(0, 6.28), 0)
    life.append(m.T(x, gy(x, z) + rng.uniform(0.25, 1.4), z))
    fish.append((x, z))
# life on the stones: anemones, sponges, corals
for (x, z, rad) in stones:
    top = gy(x, z) + rad * 0.45
    for _ in range(int(rng.integers(0, 4) + rad * 3)):
        a = rng.uniform(0, 6.28); d = rng.uniform(0, rad * 0.6)
        life.append(place(anemone(rng.uniform(0.04, 0.09), rng.choice([(0.95, 0.9, 0.85), (0.95, 0.72, 0.7), (0.95, 0.6, 0.35)])),
                          (x + d * math.cos(a), top + rad * 0.15, z + d * math.sin(a)), (math.cos(a) * 0.4, 1, math.sin(a) * 0.4), rng.uniform(0, 6)))
    if rad > 0.9 and rng.random() < 0.7:
        life.append(glass_sponge(rng.uniform(0.3, 0.8)).T(x, top + rad * 0.2, z))
    if rad > 0.7 and rng.random() < 0.5:
        life.append(bamboo_coral(rng.uniform(0.5, 1.1), int(rng.integers(1e6))).T(x + 0.2, top + rad * 0.2, z))
log("seabed life done")

# life on the wreck: anemones on the rails and edges, mushroom corals, squat lobsters, brittle stars on the decks
for m_, n_an, n_mc, n_sl, n_bs in ((bow, 380, 80, 110, 120), (stern, 150, 40, 60, 60), (debris, 50, 10, 25, 25)):
    for p, nn in sample_surface(m_, n_an, lambda fn: fn[:, 1] > -0.2, rng):
        c = rng.choice([(0.95, 0.9, 0.85), (0.95, 0.9, 0.85), (0.95, 0.72, 0.7), (0.98, 0.55, 0.3)])
        life.append(place(anemone(rng.uniform(0.035, 0.08), c), p, nn, rng.uniform(0, 6)))
    for p, nn in sample_surface(m_, n_mc, lambda fn: fn[:, 1] > 0.0, rng):
        life.append(place(mushroom_coral(rng.uniform(0.05, 0.1)), p, nn, rng.uniform(0, 6)))
    for p, nn in sample_surface(m_, n_sl, lambda fn: fn[:, 1] > -0.4, rng):
        life.append(place(squat_lobster(rng.uniform(0.05, 0.09)), p + nn * 0.01, nn, rng.uniform(0, 6)))
    for p, nn in sample_surface(m_, n_bs, lambda fn: fn[:, 1] > 0.6, rng):
        life.append(place(brittle_star(rng.uniform(0.06, 0.11), (0.92, 0.85, 0.78)), p + nn * 0.01, nn, rng.uniform(0, 6)))
# a few glass sponges and bamboo corals on the stern heap (the highest, most exposed spots)
for p, nn in sample_surface(stern, 8, lambda fn: fn[:, 1] > 0.5, rng):
    life.append(place(glass_sponge(rng.uniform(0.3, 0.6)), p, nn, 0))
for p, nn in sample_surface(stern, 6, lambda fn: fn[:, 1] > 0.5, rng):
    life.append(place(bamboo_coral(rng.uniform(0.5, 0.9), int(rng.integers(1e6))), p, nn, 0))
log("wreck life done")

# ── Things from the ship ────────────────────────────────────────────────────
LEATHER = (0.17, 0.10, 0.06)


def shoe(sz=1.0, boot=False):
    ln, w = 0.27 * sz, 0.095 * sz
    up = lathe([(0.001, -ln / 2), (w * 0.35, -ln * 0.46), (w * 0.5, -ln * 0.3), (w * 0.5, ln * 0.1), (w * 0.45, ln * 0.4), (0.001, ln / 2)],
               LEATHER, seg=10, axis="z")
    up.v[:, 1] = np.maximum(up.v[:, 1], 0) * (0.9 if not boot else 1.0)
    up.S(1, 1.1, 1)
    hi = (ln * 0.55 if boot else ln * 0.28)
    up.v[:, 1] += np.where(up.v[:, 2] < ln * 0.05, sstep(ln * 0.1, -ln * 0.2, up.v[:, 2]) * hi * (up.v[:, 1] > 0.005), 0)
    sole = box(w * 1.05, 0.018 * sz, ln * 1.02, (0.08, 0.06, 0.05)).T(0, -0.012, 0)
    heel = box(w * 0.8, 0.03 * sz, ln * 0.22, (0.08, 0.06, 0.05)).T(0, -0.03, -ln * 0.37)
    hole = cyl(w * 0.32, w * 0.32, 0.004, (0.03, 0.025, 0.02), seg=10, sx=1.3).R(0, math.pi / 2, 0).T(0, hi + 0.035 * sz, -ln * 0.22)
    m = merge([up.T(0, 0.012, 0), sole, heel, hole])
    m.c = m.c * (0.85 + 0.3 * fbm(m.v, 25, 3, 2))[:, None]
    return m


def plate_(r=0.13):
    m = lathe([(0.001, 0), (r * 0.6, 0), (r * 0.72, 0.006), (r, 0.018), (r * 0.98, 0.022), (r * 0.7, 0.012), (0.001, 0.01)], (0.93, 0.92, 0.88), seg=20)
    m.paint(lambda q: np.where((np.abs(np.hypot(q[:, 0], q[:, 2]) - r * 0.86) < r * 0.05)[:, None], (0.2, 0.3, 0.6), (0.93, 0.92, 0.88)))
    return m


def cup():
    m = lathe([(0.001, 0), (0.025, 0), (0.035, 0.01), (0.045, 0.06), (0.042, 0.065), (0.038, 0.012), (0.001, 0.01)], (0.94, 0.93, 0.9), seg=16)
    h = tube([(0.044, 0.05, 0), (0.065, 0.045, 0), (0.066, 0.025, 0), (0.042, 0.02, 0)], 0.005, (0.94, 0.93, 0.9), seg=4)
    return merge([m, h])


def bottle(champ=True):
    g = (0.05, 0.14, 0.07) if champ else (0.12, 0.08, 0.04)
    m = lathe([(0.001, 0), (0.044, 0.004), (0.045, 0.2), (0.04, 0.23), (0.016, 0.27), (0.015, 0.31), (0.017, 0.315), (0.013, 0.32), (0.001, 0.32)], g, seg=12)
    cork = cyl(0.012, 0.012, 0.012, (0.4, 0.3, 0.18), seg=8).T(0, 0.318, 0)
    return merge([m, cork])


def bench_end():
    c = (0.13, 0.11, 0.1)
    curve = [(0, 0, 0.3), (0, 0.35, 0.22), (0, 0.45, 0.0), (0, 0.42, -0.25), (0, 0.6, -0.35), (0, 0.85, -0.3), (0, 0.9, -0.15)]
    leg = [(0, 0, -0.3), (0, 0.2, -0.2), (0, 0.42, -0.25)]
    scroll = [(0, 0.45, 0.0), (0, 0.55, 0.12), (0, 0.5, 0.2), (0, 0.42, 0.14)]
    return merge([tube(curve, 0.025, c, seg=5), tube(leg, 0.025, c, seg=5), tube(scroll, 0.018, c, seg=5)])


def pot(r_):
    cu = (0.6, 0.33, 0.18)
    m = lathe([(0.001, 0), (r_ * 0.95, 0), (r_, 0.03), (r_, r_ * 1.1), (r_ * 1.05, r_ * 1.12), (r_ * 0.96, r_ * 1.1), (r_ * 0.9, r_ * 0.1), (0.001, 0.02)], cu, seg=16)
    m.paint(lambda q: np.where((fbm(q, 18, 4, 2) > 0.6)[:, None], (0.3, 0.5, 0.4), cu))
    h = tube([(r_, r_ * 0.9, 0), (r_ * 1.5, r_ * 0.95, 0), (r_ * 2.2, r_ * 1.0, 0)], r_ * 0.06, (0.3, 0.2, 0.12), seg=4)
    return merge([m, h])


def telegraph():
    br = (0.6, 0.45, 0.2)
    col = lathe([(0.16, 0), (0.12, 0.08), (0.08, 0.15), (0.075, 0.9), (0.1, 0.95), (0.001, 0.96)], br, seg=14)
    head = lathe([(0.001, -0.07), (0.24, -0.06), (0.26, 0), (0.24, 0.06), (0.001, 0.07)], br, seg=20).R(math.pi / 2, 0, 0).T(0, 1.15, 0)
    dial = cyl(0.19, 0.19, 0.01, (0.8, 0.78, 0.7), seg=20).R(math.pi / 2, 0, 0).T(0, 1.15, 0.07)
    lever = box(0.04, 0.32, 0.03, br).T(0.08, 0.95, 0.12).R(0, 0, -0.35)
    return merge([col, head, dial, lever])


def safe_():
    c = (0.12, 0.1, 0.09)
    body = box(0.95, 1.25, 0.75, c)
    door = box(0.75, 1.0, 0.05, (0.15, 0.12, 0.1)).T(0, 0.12, 0.4)
    knob = cyl(0.06, 0.06, 0.04, (0.6, 0.45, 0.2), seg=12).R(math.pi / 2, 0, 0).T(0.2, 0.65, 0.43)
    handle = tube([(-0.15, 0.62, 0.43), (-0.15, 0.62, 0.5), (-0.05, 0.62, 0.5), (-0.05, 0.62, 0.43)], 0.015, (0.6, 0.45, 0.2), seg=4)
    plate_n = box(0.3, 0.08, 0.01, (0.65, 0.5, 0.22)).T(0, 1.0, 0.43)
    m = merge([body, door, knob, handle, plate_n])
    crumple(m, 0.03, 3, 7)
    return m


def suitcase():
    c = (0.3, 0.18, 0.1)
    m = box(0.7, 0.22, 0.45, c)
    crumple(m, 0.02, 4, 9)
    straps = [box(0.04, 0.23, 0.46, (0.12, 0.08, 0.05)).T(x, 0, 0) for x in (-0.2, 0.2)]
    hand = tube([(-0.08, 0.22, 0.23), (-0.08, 0.27, 0.23), (0.08, 0.27, 0.23), (0.08, 0.22, 0.23)], 0.012, (0.12, 0.08, 0.05), seg=4)
    return merge([m] + straps + [hand])


def ground_it(m, x, z, yaw, sink=0.0):
    m.R(0, yaw, 0)
    return m.T(x, gy(x, z) - sink, z)


_occ_soft = blur(occupied.astype(float), 1, 1) > 0.01


def clear_spot(x, z):
    """Nearest free sediment to (x, z)."""
    for k in range(400):
        a = k * 2.39996; d = 0.7 * math.sqrt(k)
        xx, zz = x + d * math.cos(a), z + d * math.sin(a)
        if not raster(_occ_soft, xx, zz) and inside_d(np.array([xx]), np.array([zz]))[0] < -5:
            return xx, zz
    return x, z


# shoes in pairs: where a body came to rest, the leather outlasted everything else (a memorial, not a curiosity)
shoe_sites = [clear_spot(26, -250), clear_spot(-18, -420), clear_spot(52, -515)]
for (sx, sz) in shoe_sites:
    for k in range(int(rng.integers(3, 6))):
        x, z = sx + rng.normal(0, 1.4), sz + rng.normal(0, 1.4)
        yaw = rng.uniform(0, 6.28); boot = rng.random() < 0.5; sc = rng.uniform(0.85, 1.05)
        for s in (-1, 1):
            m = shoe(sc, boot).R(0, s * 0.12, 0).T(s * 0.07, 0, 0)
            artefacts.append(ground_it(m, x, z, yaw, 0.02))
cx_, cz_ = shoe_sites[1]
for k in range(2):
    artefacts.append(ground_it(suitcase(), cx_ + 2.5 + k * 0.8, cz_ - 1.2 + k, rng.uniform(0, 6), 0.03))
# china from the pantries: rows of plates still standing on edge where their racks rotted away, cups, tureens
csx, csz = clear_spot(-30, -330)
for row in range(6):
    yaw = 0.3 + rng.normal(0, 0.08)
    x0, z0 = csx + row * 0.9 * math.cos(yaw + 1.57), csz + row * 0.9 * math.sin(yaw + 1.57)
    for k in range(int(rng.integers(8, 16))):
        x = x0 + k * 0.045 * math.sin(yaw); z = z0 + k * 0.045 * math.cos(yaw)
        r_ = rng.choice([0.13, 0.13, 0.11])
        m = plate_(r_).R(math.pi / 2 + rng.normal(0, 0.12), 0, 0).T(0, r_ * 0.75, 0)
        china.append(ground_it(m, x, z, yaw + rng.normal(0, 0.05), 0.04))
for k in range(30):
    x, z = csx + rng.normal(0, 2.5), csz + rng.normal(0, 2.5)
    china.append(ground_it(plate_(rng.choice([0.13, 0.09])).R(rng.normal(0, 0.15), 0, rng.normal(0, 0.15)), x, z, rng.uniform(0, 6), 0.01))
for k in range(22):
    x, z = csx + rng.normal(0, 2.2), csz + rng.normal(0, 2.2)
    m = cup()
    if rng.random() < 0.6:
        m.R(0, 0, math.pi / 2).T(0, 0.045, 0)
    china.append(ground_it(m, x, z, rng.uniform(0, 6), 0.0))
# bottles of champagne and wine, still corked: the crate has gone, the bottles kept its shape
bsx, bsz = clear_spot(38, -385)
for i in range(5):
    for j in range(6):
        if rng.random() < 0.15:
            continue
        m = bottle(rng.random() < 0.7).R(math.pi / 2, 0, 0).T(0, 0.045, 0)
        glass.append(ground_it(m, bsx + i * 0.1 + rng.normal(0, 0.01), bsz + j * 0.1, 0.0 + rng.normal(0, 0.05), 0.0))
for k in range(18):
    x, z = bsx + rng.normal(0, 1.6), bsz + rng.normal(0, 1.6)
    m = bottle(rng.random() < 0.6).R(math.pi / 2, 0, 0).T(0, 0.045, 0)
    glass.append(ground_it(m, x, z, rng.uniform(0, 6), 0.01))
# deck benches: only the cast-iron ends are left
bnx, bnz = clear_spot(8, -185)
for k in range(3):
    x, z = bnx + k * 3.2 + rng.normal(0, 0.5), bnz + rng.normal(0, 1.5)
    yaw = rng.uniform(0, 6.28)
    for s in (-0.9, 0.9):
        m = bench_end().T(s, 0, 0)
        if rng.random() < 0.3:
            m.R(0, 0, math.pi / 2 * 0.95).T(0, 0.03, 0)
        artefacts.append(ground_it(m, x, z, yaw, 0.03))
# a safe, upright in the mud
sfx, sfz = clear_spot(-44, -472)
artefacts.append(ground_it(safe_(), sfx, sfz, 0.7, 0.25))
# copper pots from the galleys
ptx, ptz = clear_spot(62, -468)
for k in range(9):
    m = pot(rng.uniform(0.1, 0.22))
    if rng.random() < 0.5:
        m.R(rng.uniform(1.2, 1.7), 0, 0).T(0, 0.15, 0)
    artefacts.append(ground_it(m, ptx + rng.normal(0, 1.5), ptz + rng.normal(0, 1.5), rng.uniform(0, 6), 0.02))
# an engine telegraph, from the docking bridge of the stern
tgx, tgz = clear_spot(34, -538)
artefacts.append(ground_it(telegraph().R(0.25, 0, 0.1), tgx, tgz, 2.0, 0.15))
def dust(m, amt, seed, col=(0.66, 0.62, 0.55)):
    """A film of fine sediment on everything that faces up."""
    n = normals(m)
    k = amt * sstep(0.25, 0.95, n[:, 1]) * (0.6 + 0.4 * fbm(m.v, 6.0, seed, 2))
    m.c = m.c * (1 - k[:, None]) + np.array(col) * k[:, None]


artefacts = merge(artefacts)
dust(artefacts, 0.45, 88)
china = merge(china)
dust(china, 0.35, 89)
glass = merge(glass)
dust(glass, 0.5, 90)
log("artefacts done")

# ── Points of interest ──────────────────────────────────────────────────────
up = np.array([0, 1.2, 0])
poi("bow", P["bow"] + up, "The bow", "La proa",
    "Still sharp and upright after more than a century. It ploughed some 18 m into the mud and pushed up the ridges on both sides.",
    "Sigue afilada y derecha después de más de un siglo. Se clavó unos 18 m en el barro y levantó los montículos de los dos lados.")
poi("anchor", P["anchor"], "Port anchor", "Ancla de babor",
    "The anchors still hang in their hawse pipes. Each one weighed about 8 tonnes.",
    "Las anclas siguen en sus escobenes. Cada una pesaba unas 8 toneladas.")
poi("rail", P["rail"] + up * 0.5, "Rusticles on the rail", "Carámbanos de óxido",
    "The «rusticles» are colonies of bacteria (one, Halomonas titanicae, was found here) that eat the iron. Touch them and they crumble to dust.",
    "Los «rusticles» son colonias de bacterias (una, Halomonas titanicae, se descubrió aquí) que se comen el hierro. Si los tocas, se deshacen en polvo.")
poi("telemotor", P["telemotor"], "Bridge and telemotor", "Puente y telemotor",
    "The wheelhouse was made of wood and is gone. The bronze telemotor that held the wheel still stands where the bridge was.",
    "La caseta del timón era de madera y ha desaparecido. El telemotor de bronce que sujetaba la rueda sigue en pie donde estaba el puente.")
poi("foremast", P["foremast"], "The fallen foremast", "El mástil de proa, caído",
    "It fell back across the bridge. The crow's nest from which the iceberg was sighted hung from it until it broke off.",
    "Cayó hacia atrás sobre el puente. De él colgaba la cofa desde donde se avistó el iceberg, hasta que se desprendió.")
poi("bathtub", P["bathtub"], "Captain Smith's bathtub", "La bañera del capitán Smith",
    "The wall of the captain's quarters has collapsed and his bathtub can be seen from outside.",
    "La pared del camarote del capitán se ha derrumbado y su bañera se ve desde fuera.")
poi("staircase", P["staircase"], "Grand staircase", "La gran escalera",
    "A dark hole where the glass dome over the first-class staircase used to be. The wooden staircase itself is gone.",
    "Un agujero oscuro donde estaba la cúpula de cristal sobre la escalera de primera clase. La escalera, de madera, ya no existe.")
poi("funnel1", P["funnel1"], "Base of the first funnel", "Base de la primera chimenea",
    "All four funnels broke off during the sinking. Only their torn bases are left on the deck.",
    "Las cuatro chimeneas se partieron durante el hundimiento. En la cubierta solo quedan sus bases rotas.")
st_top = stern.v[np.argmax(stern.v[:, 1])]
poi("stern", (14, st_top[1] + 2, -600), "The stern", "La popa",
    "It hit the bottom at great speed and collapsed onto itself. It lies 600 m south of the bow, facing the other way.",
    "Chocó contra el fondo a gran velocidad y se aplastó sobre sí misma. Está 600 m al sur de la proa, mirando en sentido contrario.")
poi("propellers", (14, gy(14, -546) + 6, -546), "Rudder and propellers", "Timón y hélices",
    "The rudder and the propellers, each over 7 m across, are half buried under the stern.",
    "El timón y las hélices, de más de 7 m cada una, están medio enterrados bajo la popa.")
bx_, bz_ = np.mean(W["boilers"], axis=0)
poi("boilers", (bx_, gy(bx_, bz_) + 6, bz_), "Boilers", "Calderas",
    "Five of the boilers fell out when the ship broke in two. Each is 4.8 m across. Coal is scattered around them.",
    "Cinco calderas cayeron cuando el barco se partió en dos. Cada una mide 4,8 m de diámetro. A su alrededor hay carbón esparcido.")
for i, (sx, sz) in enumerate(shoe_sites[:2]):
    poi(f"shoes{i}", (sx, gy(sx, sz) + 0.8, sz), "Pairs of shoes", "Pares de zapatos",
        "Where a person came to rest, only the leather of their shoes lasted. They are a reminder that this is a grave.",
        "Donde quedó una persona, solo ha resistido el cuero de sus zapatos. Recuerdan que este lugar es una tumba.", "memorial")
poi("china", (csx, gy(csx, csz) + 0.8, csz), "White Star china", "Vajilla de la White Star",
    "Plates still stand in rows where their wooden racks rotted away.",
    "Los platos siguen en fila donde se pudrieron sus estantes de madera.")
poi("bottles", (bsx, gy(bsx, bsz) + 0.7, bsz), "Bottles, still corked", "Botellas, aún con su corcho",
    "Champagne and wine bottles. The crate is gone but they keep its shape, and many are still corked.",
    "Botellas de champán y de vino. La caja ha desaparecido, pero conservan su forma, y muchas siguen con el corcho.")
poi("benches", (bnx, gy(bnx, bnz) + 1.3, bnz), "Deck benches", "Bancos de cubierta",
    "Only the cast-iron ends are left. The wooden seats were eaten by marine animals.",
    "Solo quedan los extremos de hierro fundido. Los asientos de madera se los comieron animales marinos.")
poi("safe", (sfx, gy(sfx, sfz) + 1.6, sfz), "A safe", "Una caja fuerte",
    "One of the ship's safes, upright in the mud. When it was found, the bottom had rusted through and it was empty.",
    "Una de las cajas fuertes del barco, de pie en el barro. Cuando se encontró, el fondo se había oxidado y estaba vacía.")
poi("pots", (ptx, gy(ptx, ptz) + 0.9, ptz), "Copper pots", "Ollas de cobre",
    "Pots from the galleys. Copper resists the sea far better than iron.",
    "Ollas de las cocinas. El cobre resiste el mar mucho mejor que el hierro.")
poi("telegraph", (tgx, gy(tgx, tgz) + 1.8, tgz), "Engine telegraph", "Telégrafo de máquinas",
    "Bronze telegraphs passed orders from the bridge to the engine room. This one stood on the docking bridge at the stern.",
    "Los telégrafos de bronce pasaban las órdenes del puente a la sala de máquinas. Este estaba en el puente de atraque de la popa.")
# animals and the ground
fx, fz = min(fish, key=lambda q: abs(q[0] - 30) + abs(q[1] + 100))
poi("rattail", (fx, gy(fx, fz) + 1.8, fz), "Grenadier (rattail)", "Pez granadero",
    "The most common fish at these depths: a big head, a tail that tapers to a thread, and a habit of hovering head-down over the mud.",
    "El pez más común a estas profundidades: cabeza grande, una cola que acaba en un hilo y la costumbre de flotar cabeza abajo sobre el barro.", "life")
cx2, cz2, _ = min(cucumbers, key=lambda q: abs(q[0] - 40) + abs(q[1] + 60) + (0 if q[2] == "psychropotes" else 50))
poi("cucumber", (cx2, gy(cx2, cz2) + 0.8, cz2), "Sea cucumber and its trail", "Holoturia y su rastro",
    "Sea cucumbers crawl over the mud swallowing it and digesting what has fallen from above. Behind them they leave a trail.",
    "Las holoturias se arrastran tragando barro y digieren lo que ha caído desde arriba. Detrás dejan un rastro.", "life")
sx2, sz2, sr2 = max(stones, key=lambda q: q[2] - 0.004 * math.hypot(q[0], q[1] + 330))
poi("dropstone", (sx2, gy(sx2, sz2) + sr2 + 1, sz2), "A stone dropped by an iceberg", "Una piedra caída de un iceberg",
    "Rocks frozen into icebergs drop to the bottom when the ice melts. They are the only hard ground here, so anemones and sponges settle on them.",
    "Las rocas atrapadas en los icebergs caen al fondo cuando el hielo se derrite. Son el único suelo duro de la zona, y en ellas se asientan anémonas y esponjas.", "life")
pv = pens[0].v[np.argmax(pens[0].v[:, 1])]
poi("seapen", pv + np.array([0, 0.4, 0]), "Sea pen (Umbellula)", "Pluma de mar (Umbellula)",
    "A colony of polyps on a stalk up to 2 m tall, catching the food the current brings.",
    "Una colonia de pólipos sobre un tallo de hasta 2 m que atrapa el alimento que trae la corriente.", "life")
poi("ripples", (95, gy(95, -150) + 2, -150), "The abyssal plain", "La llanura abisal",
    "Fine mud, 1–2 °C, total darkness. The current leaves ripples, and the animals leave holes, mounds and trails.",
    "Barro fino, 1–2 °C, oscuridad total. La corriente deja ondulaciones, y los animales, agujeros, montículos y rastros.", "life")

VIEW = {"bow": 40, "anchor": 12, "rail": 10, "telemotor": 9, "foremast": 16, "bathtub": 5, "staircase": 14, "funnel1": 12,
        "stern": 70, "propellers": 22, "boilers": 24, "shoes0": 3.5, "shoes1": 3.5, "china": 4, "bottles": 2.5, "benches": 6,
        "safe": 5, "pots": 4, "telegraph": 4, "rattail": 2.5, "cucumber": 2.5, "dropstone": 7, "seapen": 4, "ripples": 12}
for q in pois:
    if q["id"] in VIEW:
        q["view"] = VIEW[q["id"]]

# ── Light: ambient occlusion baked into the vertex colours ──────────────────
life_m = merge(life)
trails_m = merge(trails)
log("baking ambient occlusion…")
big = [bow, stern, debris, tub_m, rocks]
bake_ao([seabed, bow, stern, debris, tub_m, rocks, rust], occluders=big, ground=ground, cell=0.6, rays=24, reach=12)


def contact_shade(m, k=0.35, h=0.06):
    """Small things: darker where they touch the mud and on their undersides (the voxel AO is too coarse for them)."""
    if not len(m.v):
        return
    above = m.v[:, 1] - ground(m.v[:, 0], m.v[:, 2])
    n = normals(m)
    f = (1 - k * (1 - sstep(-0.01, h, above))) * (0.82 + 0.18 * (n[:, 1] * 0.5 + 0.5))
    m.c = m.c * f[:, None]


for m_ in (life_m, artefacts, china, glass):
    contact_shade(m_)
log("AO done")

# ── Write ───────────────────────────────────────────────────────────────────
mats = {
    "seabed": material(rough=0.97, tex=seabed_textures(2048, 9.0), uv=("xz", 9.0)),
    "strata": material(rough=0.95, tex=seabed_textures(512, 6.0, 7), uv=("tri", 3.0), normal_scale=1.5),
    "plinth": material(rough=0.32, metal=0.0),
    "brass": material(rough=0.35, metal=0.9),
    "steel": material(rough=0.8, metal=0.05, tex=steel_textures(), uv=("tri", 7.2)),
    "rusticle": material(rough=0.95, tex=rust_textures(), uv=("tri", 1.5), normal_scale=1.6),
    "rock": material(rough=0.9, tex=rust_textures(512, 2.5, 11), uv=("tri", 2.5)),
    "life": material(rough=0.55),
    "china": material(rough=0.18),
    "glass": material(rough=0.08, metal=0.0),
    "artefact": material(rough=0.7, metal=0.1),
}
parts = [
    ("seabed", seabed, "seabed"), ("trails", trails_m, "seabed"),
    ("base_strata", strata, "strata"), ("base_plinth", plinth, "plinth"), ("base_plate", plate, "brass"),
    ("bow_section", merge([bow, tub_m]), "steel"), ("stern_section", stern, "steel"), ("debris_field", debris, "steel"),
    ("rusticles", rust, "rusticle"), ("stones", rocks, "rock"),
    ("life", life_m, "life"), ("china", china, "china"), ("bottles", glass, "glass"), ("artefacts", artefacts, "artefact"),
]
out = ROOT / "models" / "item" / "Q25173" / "diorama.glb"
info = {"diorama": {"version": 1, "depth": -3800, "water": "abyssal", "light": "none",
                    "bounds": [BX0, -7.0, BZ0, BX1, 40.0, BZ1], "start": {"target": [float(x) for x in P["bow"] - [0, 6, 25]], "distance": 120}}}
tris, size = write_scene(out, parts, mats, pois, extras=info)
log(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB")
for n, m, _ in parts:
    print(f"  {n}: {m.tris:,} tris, {len(m.v):,} verts")
