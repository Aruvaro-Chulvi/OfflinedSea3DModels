"""
dresskit — dress a wreck built by code so it is worth a close look in Wiki Submarine (MIT).

What it adds to the plain geometry of a wreck (Q…_name.py builds the hull, the debris, the old mud):
- its own sediment: one height field of mud or sand piled against everything that lies on the bottom, with the
  ridges of the old mud, fading into the seabed (node «sediment»; its vertex colours carry only the baked light,
  so Offlined paints it with the seabed's own colour);
- rust: the paint turned brown, rusticles under every edge and runs down the walls (fewer the shallower);
- the life that really lives there, by depth: abyssal (deep-sea animals, very few), bathyal, or shelf (a wreck in
  tens of metres of water is covered in plumose anemones and soft corals, with shoals of fish around it);
- things from the ship (plates and bottles of a liner, shell casings of a warship, shoes where people died);
- riveted steel textures, baked ambient occlusion, and the compact writer (16-bit, instancing).
And for the ship as built: planked decks, clean plating and baked light.

    from dresskit import dress_wreck, dress_built
    dress_wreck(out, [{"steel": [("hull", hull)], "mud": [mud], "plain": [("nets", nets, "net")]}], depth=-93,
                kind="liner", seed=7, memorial=True)
    dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[DECK])
"""
from __future__ import annotations

import math

import numpy as np

import abyss
from meshkit import Mesh, merge, weather, sstep, fbm, normals
from scenekit import (material, steel_textures, rust_textures, planks_textures, bake_ao, instances, write_model)


def zone_of(depth):
    d = abs(depth)
    return "abyssal" if d >= 1500 else "bathyal" if d >= 200 else "shelf"


# Colour of the flat seabed in Offlined at each depth (calcareous ooze deep down, mud, sand on the shelf)
SED_BASE = {"abyssal": np.array([0.83, 0.79, 0.70]), "bathyal": np.array([0.62, 0.59, 0.52]), "shelf": np.array([0.72, 0.67, 0.56])}


def _area(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


# ── Sediment ───────────────────────────────────────────────────────────────
def sediment_field(steel, muds, zone, seed, cell=1.0, loose=(), drift=1.0):
    """-> (sediment Mesh, hAt(x, z)). Drifts against big pieces (hull) and small ones (debris), plus the old mud.
    loose: ids of meshes that are many small things spread over a wide area (a debris field): they drift like small ones."""
    allv = np.vstack([m.v for m in steel + muds])
    gx0, gz0 = math.floor(allv[:, 0].min()) - 30, math.floor(allv[:, 2].min()) - 30
    gx1, gz1 = math.ceil(allv[:, 0].max()) + 30, math.ceil(allv[:, 2].max()) + 30
    NX, NZ = int((gx1 - gx0) / cell) + 1, int((gz1 - gz0) / cell) + 1
    X, Z = np.meshgrid(gx0 + np.arange(NX) * cell, gz0 + np.arange(NZ) * cell, indexing="ij")
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)

    def src(ms, ymax):
        g = np.zeros((NX, NZ), bool)
        for m in ms:
            v = m.v[m.v[:, 1] < ymax]
            g[np.clip(((v[:, 0] - gx0) / cell).astype(int), 0, NX - 1), np.clip(((v[:, 2] - gz0) / cell).astype(int), 0, NZ - 1)] = True
        return g

    def spread(g, amp, reach):
        h = np.where(g, amp, 0.0)
        k1, k2 = math.exp(-cell / reach), math.exp(-cell * 1.414 / reach)
        for _ in range(int(reach * 4 / cell)):
            p = np.pad(h, 1)
            h = np.maximum.reduce([h, p[:-2, 1:-1] * k1, p[2:, 1:-1] * k1, p[1:-1, :-2] * k1, p[1:-1, 2:] * k1,
                                   p[:-2, :-2] * k2, p[2:, 2:] * k2, p[:-2, 2:] * k2, p[2:, :-2] * k2])
        return h

    big = [m for m in steel if id(m) not in loose and np.ptp(m.v[:, 0]) + np.ptp(m.v[:, 2]) > 40]
    small = [m for m in steel if m not in big]
    amp = {"abyssal": 2.6, "bathyal": 2.0, "shelf": 1.5}[zone] * drift             # sand drifts less than ooze piles up
    hb = spread(src(big, 1.0), amp, 7.0) * (0.7 + 0.6 * fbm(P3, 0.08, seed + 71, 3).reshape(X.shape)) if big else np.zeros(X.shape)
    hd = spread(src(small, 0.8), amp * 0.27, 2.2) * (0.6 + 0.8 * fbm(P3, 0.3, seed + 72, 2).reshape(X.shape)) if small else np.zeros(X.shape)
    # the old mud (ridges, slide tracks, scour mounds): its highest point in every cell, softened
    ridge = np.full(X.shape, -1.0)
    for m in muds:
        ii = np.clip(((m.v[:, 0] - gx0) / cell).astype(int), 0, NX - 1); jj = np.clip(((m.v[:, 2] - gz0) / cell).astype(int), 0, NZ - 1)
        np.maximum.at(ridge, (ii, jj), m.v[:, 1])
    if muds:
        filled = ridge > -1
        r2 = np.where(filled, ridge, 0.0)
        for _ in range(3):                                                          # fill the holes between the mud's vertices
            p = np.pad(r2, 1); q = np.pad(filled.astype(float), 1)
            s_ = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]; n_ = q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]
            r2 = np.where(filled, r2, np.where(n_ > 0, s_ / np.maximum(n_, 1), 0)); filled = filled | (n_ > 0)
        ridge = np.where(filled, r2, -1.0)
    H = np.maximum.reduce([hb, hd, ridge])
    for _ in range(3):
        p = np.pad(H, 1, mode="edge")
        H = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + 2 * H) / 6
    H = H + 0.04 * (fbm(P3, 0.6, seed + 73, 2).reshape(X.shape) - 0.5) - 0.08

    def hAt(x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        i = np.clip((x - gx0) / cell, 0, NX - 1.001); j = np.clip((z - gz0) / cell, 0, NZ - 1.001)
        a, b = i.astype(int), j.astype(int); fu, fw = i - a, j - b
        h = H[a, b] * (1 - fu) * (1 - fw) + H[a + 1, b] * fu * (1 - fw) + H[a, b + 1] * (1 - fu) * fw + H[a + 1, b + 1] * fu * fw
        return np.maximum(h, 0.0)

    keep = H > -0.02
    cellok = keep[:-1, :-1] | keep[1:, :-1] | keep[:-1, 1:] | keep[1:, 1:]
    ci, cj = np.nonzero(cellok)
    use = np.zeros((NX, NZ), bool)
    for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
        use[ci + di, cj + dj] = True
    ui, uj = np.nonzero(use)
    vid = -np.ones((NX, NZ), int); vid[ui, uj] = np.arange(len(ui))
    sh = H[ui, uj]
    sv = np.stack([X[ui, uj], np.where(sh > 0, sh, np.maximum(sh * 6, -0.6)), Z[ui, uj]], 1)   # the edge dives under the ground
    a_, b_, c_, d_ = vid[ci, cj], vid[ci + 1, cj], vid[ci, cj + 1], vid[ci + 1, cj + 1]
    sf = np.concatenate([np.stack([a_, c_, b_], 1), np.stack([b_, c_, d_], 1)])
    return Mesh(sv, sf, SED_BASE[zone]), hAt


# ── Life by depth ──────────────────────────────────────────────────────────
def _variants(fn, n):
    return [fn(k) for k in range(n)]


class Dresser:
    def __init__(self, zone, seed):
        self.zone, self.r = zone, np.random.default_rng(seed)
        abyss.seed(seed)
        self.INS = []

    def add(self, name, variants, mat, pos, nrm=None, yaw=None, scale=None, parent=None):
        r = self.r
        if pos is None or not len(pos):
            return
        pos = np.asarray(pos, float); N = len(pos)
        nrm = np.tile([0, 1.0, 0], (N, 1)) if nrm is None else np.asarray(nrm, float)
        yaw = r.uniform(0, 6.283, N) if yaw is None else np.asarray(yaw, float)
        scale = np.ones(N) if scale is None else np.asarray(scale, float)
        pick = r.integers(len(variants), size=N)
        for k, base in enumerate(variants):
            s = pick == k
            if s.any():
                self.INS.append((f"{name}_{k}{'_' + parent if parent else ''}", base, mat, instances(pos[s], nrm[s], yaw[s], scale[s]), parent))

    def on(self, m, n, cond):
        if n <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        s = abyss.sample_surface(m, int(n), cond, self.r)
        if not s:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.array([p for p, _ in s]), np.array([q for _, q in s])


def dress_site(steel, muds, plain, zone, seed, kind, memorial=False, parent=None, wood=False, life="normal", drift=1.0):
    """One site (the main wreck, or a piece lying apart): returns (wreck parts, extra parts, instanced, ground objects).
    steel: [(node name, Mesh)] or (node name, Mesh, "loose") for scattered small things (sand drifts less against them)."""
    D = Dresser(zone, seed)
    r = D.r
    steel_m = [e[1] for e in steel]
    loose = {id(e[1]) for e in steel if len(e) > 2 and e[2] == "loose"}
    # ── the paint: rust (deep) or a skin of growth (shallow) ──
    rust_k = {"abyssal": 0.55, "bathyal": 0.4, "shelf": 0.2}[zone]
    for k, m in enumerate(steel_m if wood else []):            # wood (or wood="alu", aluminium): no rust; dulled, a fine film of silt
        m.c = m.c * (0.78 + 0.22 * fbm(m.v, 0.5, seed + 205 + k, 2))[:, None]
        weather(m, rust=0.0, sediment=0.18, streaks=0.0, seed=seed + 200 + k)
        m.c = np.clip(m.c * 0.9, 0, 1)
    for k, m in enumerate([] if wood else steel_m):
        lum = m.c.mean(1)
        kk = 0.5 * sstep(0.25, 0.05, lum)
        m.c = m.c * (1 - kk[:, None]) + np.array([0.24, 0.12, 0.07]) * kk[:, None]
        weather(m, rust=rust_k, sediment=0.12, streaks=1.0 if zone != "shelf" else 0.4, seed=seed + 200 + k, rust_col=(0.42, 0.17, 0.07))
        if zone == "shelf":                                  # encrusting life: pale and greenish blotches over everything
            g = sstep(0.4, 0.75, fbm(m.v, 0.35, seed + 300 + k, 3))
            m.c = m.c * (1 - 0.6 * g[:, None]) + np.array([0.55, 0.56, 0.48]) * 0.6 * g[:, None]
        m.c = np.clip(m.c * (0.85 + 0.3 * fbm(m.v, 0.6, seed + 205 + k, 2))[:, None], 0, 1)
    # ── sediment ──
    sediment, hAt = sediment_field(steel_m, muds, zone, seed, loose=loose, drift=drift)
    if sediment.tris > 100000:                               # a big site (long debris field, slide track): coarser cells
        sediment, hAt = sediment_field(steel_m, muds, zone, seed, cell=math.sqrt(sediment.tris / 90000), loose=loose, drift=drift)
    base = SED_BASE[zone]
    for m in steel_m:
        hh = m.v[:, 1] - hAt(m.v[:, 0], m.v[:, 2])
        k = 0.8 * sstep(1.4, -0.3, hh)
        m.c = m.c * (1 - k[:, None]) + (base * 0.78)[None] * k[:, None]
    # ── where things can go on the ground ──
    occ = set()
    for m in steel_m:
        for x, z in np.floor(m.v[m.v[:, 1] < 2.5][:, [0, 2]]).astype(int):
            occ.add((int(x), int(z)))
    for m in muds:
        for x, z in np.floor(m.v[m.v[:, 1] > 0.3][:, [0, 2]]).astype(int):
            occ.add((int(x), int(z)))
    near_pts = np.array(sorted(occ), float) if occ else np.zeros((1, 2))
    allv = np.vstack(steel_m and [m.v for m in steel_m])
    lo, hi = allv.min(0), allv.max(0)
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    R = min(math.hypot(hi[0] - lo[0], hi[2] - lo[2]) / 2 + 15, 800) - 12

    def free(near=0.6):
        for _ in range(80):
            if r.random() < near:
                b = near_pts[r.integers(len(near_pts))]
                x, z = b[0] + r.normal(0, 14), b[1] + r.normal(0, 14)
            else:
                a, d = r.uniform(0, 6.283), R * math.sqrt(r.random())
                x, z = cx + d * math.cos(a), cz + d * math.sin(a)
            if math.hypot(x - cx, z - cz) < R and not any((int(math.floor(x)) + i, int(math.floor(z)) + j) in occ for i in (-1, 0, 1) for j in (-1, 0, 1)):
                return x, z
        return None

    def spots(n, near=0.6):
        out = [free(near) for _ in range(int(n))]
        return np.array([(x, 0.0, z) for x, z in (p for p in out if p)]).reshape(-1, 3)

    area = sum(_area(m) for m in steel_m)
    A = area / 1000.0                                         # thousands of m² of wreck surface
    # ── rust: rusticles and runs (deep wrecks are hung with them; shallow ones hardly) ──
    dens = {"abyssal": 1.0, "bathyal": 0.5, "shelf": 0.06}[zone] * (0 if wood else 1)
    ice = [abyss.rusticle(1.0, 0.1, seed + 50 + k) for k in range(4)]
    flows = [abyss.rust_flow(1.0, 0.1, seed + 60 + k) for k in range(4)]
    for m in (steel_m if dens else []):
        a_k = _area(m) / 1000.0
        p, _ = D.on(m, min(3000, 55 * a_k * dens), lambda fn: fn[:, 1] < -0.35)
        if len(p):
            ln = r.uniform(0.3, 1.0, len(p)) ** 1.6 * 2.4 + 0.15
            rad = r.uniform(0.5, 1.2, len(p)) * (0.6 + ln / 3)
            D.add("rusticles", ice, "rust", p, scale=np.stack([rad, ln, rad], 1), parent=parent)
        p, nn = D.on(m, min(2500, 40 * a_k * max(dens, 0.25)), lambda fn: np.abs(fn[:, 1]) < 0.3)
        if len(p):
            ln = r.uniform(0.3, 1.0, len(p)) ** 1.5 * 3.2 + 0.3
            w = r.uniform(0.4, 1.1, len(p))
            D.add("rust_runs", flows, "rust", p + nn * 0.02, yaw=np.arctan2(nn[:, 0], nn[:, 2]), scale=np.stack([w, ln, w], 1), parent=parent)
    # ── life on the wreck ──
    wreck_all = merge(steel_m)

    def on_wreck(n, cond):
        return D.on(wreck_all, min(int(n), 4000), cond)

    if life == "baltic":                                     # the brackish, cold, often airless deep Baltic: hardly anything grows
        pass
    elif zone == "shelf" and life == "tropical":                # a warm-water wreck: soft corals, sponges, crinoids' colours
        dmf = [abyss.dead_mans_fingers(0.16, c) for c in ((0.86, 0.38, 0.6), (0.95, 0.55, 0.2), (0.62, 0.32, 0.72), (0.95, 0.85, 0.6))]
        p, nn = on_wreck(min(260, 10 * A), lambda fn: fn[:, 1] > -0.3)
        D.add("soft_corals", dmf, "life", p, nn, scale=r.uniform(0.5, 1.4, len(p)), parent=parent)
        anem = [abyss.anemone(0.06, c) for c in ((0.95, 0.75, 0.3), (0.9, 0.35, 0.3), (0.55, 0.8, 0.5))]
        p, nn = on_wreck(min(160, 6 * A), lambda fn: fn[:, 1] > -0.2)
        D.add("anemones", anem, "life", p, nn, scale=r.uniform(0.7, 1.5, len(p)), parent=parent)
        sp = [abyss.glass_sponge(0.35) for _ in range(2)]
        for m in sp:
            m.c = np.clip(m.c * np.array([1.0, 0.75, 0.5]), 0, 1)
        p, nn = on_wreck(min(60, 2 * A), lambda fn: fn[:, 1] > 0.4)
        D.add("sponges", sp, "life", p, nn, scale=r.uniform(0.5, 1.1, len(p)), parent=parent)
    elif zone == "shelf":
        cols = [(0.96, 0.94, 0.9), (0.97, 0.9, 0.78), (0.96, 0.62, 0.32), (0.96, 0.94, 0.9)]
        plum = [abyss.plumose_anemone(0.35, c) for c in cols]
        p, nn = on_wreck(min(1400, 40 * A), lambda fn: fn[:, 1] > -0.5)
        D.add("plumose_anemones", plum, "life", p, nn, scale=r.uniform(0.5, 1.6, len(p)), parent=parent)
        dmf = [abyss.dead_mans_fingers(0.16, c) for c in ((0.95, 0.9, 0.82), (0.95, 0.62, 0.3), (0.95, 0.9, 0.82))]
        p, nn = on_wreck(min(500, 15 * A), lambda fn: fn[:, 1] > -0.3)
        D.add("soft_corals", dmf, "life", p, nn, scale=r.uniform(0.6, 1.5, len(p)), parent=parent)
        anem = [abyss.anemone(0.06, c) for c in ((0.9, 0.3, 0.25), (0.95, 0.7, 0.2), (0.6, 0.75, 0.45), (0.95, 0.85, 0.8))]
        p, nn = on_wreck(min(700, 20 * A), lambda fn: fn[:, 1] > -0.2)
        D.add("anemones", anem, "life", p, nn, scale=r.uniform(0.7, 1.6, len(p)), parent=parent)
        stars = [abyss.starfish(0.12, c) for c in ((0.85, 0.42, 0.18), (0.8, 0.3, 0.25), (0.9, 0.6, 0.3))]
        p, nn = on_wreck(min(300, 8 * A), lambda fn: fn[:, 1] > 0.3)
        D.add("starfish", stars, "life", p + nn * 0.01, nn, scale=r.uniform(0.7, 1.4, len(p)), parent=parent)
    else:
        deep = zone == "abyssal"
        an_cols = [(0.95, 0.9, 0.85), (0.95, 0.9, 0.85), (0.95, 0.72, 0.7), (0.98, 0.55, 0.3)]
        anem = [abyss.anemone(0.06, c) for c in an_cols]
        p, nn = on_wreck((7 if deep else 14) * A, lambda fn: fn[:, 1] > -0.2)
        D.add("anemones", anem, "life", p, nn, scale=r.uniform(0.6, 1.35, len(p)), parent=parent)
        if deep:
            coral = [abyss.mushroom_coral(0.08) for _ in range(2)]
            p, nn = on_wreck(1.5 * A, lambda fn: fn[:, 1] > 0.0)
            D.add("mushroom_corals", coral, "life", p, nn, scale=r.uniform(0.6, 1.3, len(p)), parent=parent)
        else:
            sp = [abyss.glass_sponge(0.5) for _ in range(2)]
            p, nn = on_wreck(0.8 * A, lambda fn: fn[:, 1] > 0.4)
            D.add("sponges", sp, "life", p, nn, scale=r.uniform(0.5, 1.2, len(p)), parent=parent)
        lobst = [abyss.squat_lobster(0.07) for _ in range(3)]
        p, nn = on_wreck(2 * A, lambda fn: fn[:, 1] > -0.4)
        D.add("squat_lobsters", lobst, "life", p + nn * 0.01, nn, scale=r.uniform(0.7, 1.3, len(p)), parent=parent)
        stars = [abyss.brittle_star(0.1, c) for c in ((0.85, 0.7, 0.58), (0.92, 0.88, 0.82), (0.78, 0.55, 0.45))]
        p, nn = on_wreck(2.4 * A, lambda fn: fn[:, 1] > 0.6)
        D.add("brittle_stars", stars, "life", p + nn * 0.01, nn, scale=r.uniform(0.6, 1.1, len(p)), parent=parent)
    # ── the ground around ──
    stones, ground = [], []
    n_st = {"abyssal": 30, "bathyal": 22, "shelf": 18}[zone]
    for i in range(n_st):
        s = free(0.5)
        if not s:
            continue
        rad = r.uniform(0.25, 0.9) ** 2 * 2.6 + 0.15
        st = abyss.blob(rad, (0.42, 0.41, 0.4), seed + 300 + i, sub=2, amp=0.35).S(1, r.uniform(0.5, 0.8), 1)
        st.c = np.array([(0.38, 0.37, 0.36), (0.5, 0.46, 0.42), (0.3, 0.3, 0.32)][i % 3]) * (0.8 + 0.4 * fbm(st.v, 2.0, i, 2))[:, None]
        weather(st, rust=0.08, sediment=0.35, streaks=0.0, seed=i, sed_col=tuple(base * 0.8))
        st.c *= 0.8
        stones.append(st.R(0, r.uniform(0, 6.28), 0).T(s[0], -rad * 0.25, s[1]))
    stones = merge(stones) if stones else Mesh()
    if len(stones.f):
        p, nn = D.on(stones, 120, lambda fn: fn[:, 1] > 0.3)
        anem2 = [abyss.anemone(0.06, c) for c in ((0.95, 0.9, 0.85), (0.95, 0.72, 0.7), (0.98, 0.55, 0.3))]
        D.add("anemones_on_stones", anem2, "life", p, nn, scale=r.uniform(0.7, 1.4, len(p)), parent=parent)
    if life == "baltic":                                     # a few cod about the wreck, nothing on the bottom
        fishv = [abyss.fish(0.9, c) for c in ((0.45, 0.42, 0.33), (0.5, 0.46, 0.36))]
        fp, fy = [], []
        for g in range(5):
            c0 = free(0.9)
            if not c0:
                continue
            for k in range(int(r.integers(2, 7))):
                fp.append((c0[0] + r.normal(0, 4), r.uniform(0.6, 4), c0[1] + r.normal(0, 4))); fy.append(r.uniform(0, 6.28))
        if fp:
            D.add("fish", fishv, "life", np.array(fp), yaw=np.array(fy), scale=r.uniform(0.7, 1.3, len(fp)), parent=parent)
    elif zone == "shelf":
        stars = [abyss.starfish(0.12, c) for c in ((0.85, 0.42, 0.18), (0.8, 0.3, 0.25))]
        bs = spots(260, 0.6)
        D.add("starfish_ground", stars, "life", bs + [0, 0.01, 0], scale=r.uniform(0.7, 1.5, len(bs)), parent=parent)
        # shoals of fish around the wreck: a few groups, all swimming the same way
        fishv = [abyss.fish(0.6, c) for c in (((0.85, 0.7, 0.25), (0.45, 0.6, 0.75), (0.8, 0.45, 0.35)) if life == "tropical"
                                              else ((0.42, 0.45, 0.38), (0.5, 0.48, 0.4), (0.36, 0.4, 0.36)))]
        fp, fy = [], []
        for g in range(int(6 + 3 * min(A, 20) / 10)):
            c0 = free(0.85)
            if not c0:
                continue
            yaw = r.uniform(0, 6.28); h0 = r.uniform(2, 9)
            for k in range(int(r.integers(12, 40))):
                fp.append((c0[0] + r.normal(0, 3), h0 + r.normal(0, 1.2), c0[1] + r.normal(0, 3))); fy.append(yaw + r.normal(0, 0.15))
        if fp:
            D.add("fish", fishv, "life", np.array(fp), yaw=np.array(fy), scale=r.uniform(0.6, 1.4, len(fp)), parent=parent)
        mounds = [abyss.mound(0.25) for _ in range(2)]
        mp = spots(120, 0.3)
        D.add("burrow_mounds", mounds, "life", mp - [0, 0.01, 0], scale=r.uniform(0.5, 1.4, len(mp)), parent=parent)
    else:
        deep = zone == "abyssal"
        stars = [abyss.brittle_star(0.1, c) for c in ((0.85, 0.7, 0.58), (0.92, 0.88, 0.82), (0.78, 0.55, 0.45))]
        bs = spots(500 if deep else 300, 0.65)
        D.add("brittle_stars_mud", stars, "life", bs + [0, 0.01, 0], scale=r.uniform(0.6, 1.3, len(bs)), parent=parent)
        cuc = [abyss.sea_cucumber("psychropotes"), abyss.sea_cucumber("benthodytes"), abyss.sea_cucumber("benthodytes")]
        D.add("sea_cucumbers", cuc, "life", spots(70, 0.45), parent=parent)
        pens = [abyss.sea_pen(1.6) for _ in range(3)]
        pp = spots(40, 0.15)
        D.add("sea_pens", pens, "life", pp - [0, 0.05, 0], scale=r.uniform(0.55, 1.35, len(pp)), parent=parent)
        mounds = [abyss.mound(0.25) for _ in range(2)]
        mp = spots(250, 0.3)
        D.add("burrow_mounds", mounds, "life", mp - [0, 0.01, 0], scale=r.uniform(0.5, 1.4, len(mp)), parent=parent)
        if deep:
            xen = [abyss.xeno(0.07) for _ in range(3)]
            xp = spots(260, 0.4)
            D.add("xenophyophores", xen, "life", xp + [0, 0.01, 0], scale=r.uniform(0.6, 1.4, len(xp)), parent=parent)
        fish = [abyss.rattail(0.65) for _ in range(3)]
        fp = spots(30 if deep else 45, 0.6)
        if len(fp):
            fp[:, 1] = r.uniform(0.25, 1.4, len(fp))
            tilt = np.stack([np.zeros(len(fp)), np.full(len(fp), math.cos(0.45)), np.full(len(fp), -math.sin(0.45))], 1)
            D.add("grenadiers", fish, "life", fp, tilt, scale=r.uniform(0.7, 1.3, len(fp)), parent=parent)
    # ── things from the ship ──
    objs = []
    if kind == "liner":
        plate = abyss.plate(0.13); abyss.dust(plate, 0.3, 4)
        for site in range(2):
            c0 = free(0.9)
            if not c0:
                continue
            pp, pn = [], []
            for row in range(int(r.integers(3, 6))):
                yaw = r.uniform(0, 6.28)
                x0, z0 = c0[0] + row * 0.9 * math.cos(yaw + 1.57), c0[1] + row * 0.9 * math.sin(yaw + 1.57)
                for k in range(int(r.integers(8, 15))):
                    pp.append((x0 + k * 0.045 * math.sin(yaw), 0.095, z0 + k * 0.045 * math.cos(yaw)))
                    t = r.normal(0, 0.12); pn.append((math.sin(yaw) * math.cos(t), math.sin(t), math.cos(yaw) * math.cos(t)))
            for k in range(20):
                pp.append((c0[0] + r.normal(0, 2.5), 0.0, c0[1] + r.normal(0, 2.5))); pn.append((r.normal(0, 0.15), 1, r.normal(0, 0.15)))
            objs.append(("plates", [plate], "china", np.array(pp), np.array(pn), None))
        bott = [abyss.bottle(True), abyss.bottle(False)]
        for m in bott:
            abyss.dust(m, 0.5, 6)
        c0 = free(0.9)
        if c0:
            bp = np.array([(c0[0] + r.normal(0, 1.8), 0.045, c0[1] + r.normal(0, 1.8)) for _ in range(30)])
            a_ = r.uniform(0, 6.28, len(bp))
            objs.append(("bottles", bott, "glass", bp, np.stack([np.cos(a_), np.zeros(len(bp)), np.sin(a_)], 1), np.zeros(len(bp))))
    if kind == "warship":
        cas = [abyss.shell_casing(1.0, 0.19), abyss.shell_casing(0.7, 0.13), abyss.shell_casing(0.5, 0.06)]
        for m in cas:
            abyss.dust(m, 0.4, 8)
        cp = []
        for g in range(5):
            c0 = free(0.9)
            if c0:
                cp += [(c0[0] + r.normal(0, 2.5), 0.15, c0[1] + r.normal(0, 2.5)) for _ in range(int(r.integers(6, 16)))]
        if cp:
            cp = np.array(cp); a_ = r.uniform(0, 6.28, len(cp))
            objs.append(("shell_casings", cas, "artefact", cp, np.stack([np.cos(a_), r.uniform(-0.1, 0.1, len(cp)), np.sin(a_)], 1), None))
    if kind == "galleon":                                     # round shot and broken pots about the wreck
        balls = [abyss.blob(0.08, (0.14, 0.13, 0.12), seed + 900 + k, sub=2, amp=0.08) for k in range(2)]
        bp = []
        for g in range(6):
            c0 = free(0.95)
            if c0:
                bp += [(c0[0] + r.normal(0, 1.5), 0.05, c0[1] + r.normal(0, 1.5)) for _ in range(int(r.integers(4, 14)))]
        if bp:
            objs.append(("cannonballs", balls, "artefact", np.array(bp), None, None))
        pots = [abyss.pot(0.3), abyss.pot(0.22)]
        for m in pots:
            abyss.dust(m, 0.5, 5)
        pp = [p for p in (free(0.95) for _ in range(14)) if p]
        if pp:
            pp = np.array([(x, 0.0, z) for x, z in pp]); a_ = r.uniform(0, 6.28, len(pp))
            objs.append(("pots", pots, "china", pp, np.stack([np.cos(a_) * 0.9, np.full(len(pp), 0.45), np.sin(a_) * 0.9], 1), None))
    if memorial:
        shoes = [abyss.shoe(1.0, False), abyss.shoe(1.0, True)]
        for m in shoes:
            abyss.dust(m, 0.45, 3)
        sp_, sy_ = [], []
        for site in range(3):
            c0 = free(0.85)
            if not c0:
                continue
            for k in range(int(r.integers(3, 6))):
                x, z = c0[0] + r.normal(0, 1.4), c0[1] + r.normal(0, 1.4)
                yaw = r.uniform(0, 6.28)
                for s in (-1, 1):
                    sp_.append((x + math.cos(yaw) * s * 0.07, -0.02, z - math.sin(yaw) * s * 0.07)); sy_.append(yaw + s * 0.12)
        if sp_:
            objs.append(("shoes", shoes, "artefact", np.array(sp_), None, np.array(sy_)))
    for name, var, mat, pos, nrm, yaw in objs:
        D.add(name, var, mat, pos, nrm, yaw, parent=parent)
    # ── everything on the ground rests on the sediment ──
    ON_GROUND = ("anemones_on_stones", "starfish_ground", "fish", "burrow_mounds", "brittle_stars_mud", "sea_cucumbers",
                 "sea_pens", "xenophyophores", "grenadiers", "plates", "bottles", "shell_casings", "shoes", "sponges_ground",
                 "cannonballs", "pots")
    for name, _, _, ins, _ in D.INS:
        if any(name.startswith(g + "_") for g in ON_GROUND):
            ins["t"] = ins["t"].copy(); ins["t"][:, 1] += hAt(ins["t"][:, 0], ins["t"][:, 2])
    if len(stones.v):
        stones.v[:, 1] += hAt(stones.v[:, 0], stones.v[:, 2])
    for _, m, _ in plain:
        m.v[:, 1] += hAt(m.v[:, 0], m.v[:, 2]) * (m.v[:, 1] < 0.5)
    # ── light: ambient occlusion of the big pieces, and the sediment open and flat = as bright as the seabed ──
    flat = lambda X, Z: np.zeros_like(X)
    occl = steel_m + ([stones] if len(stones.f) else []) + [sediment]
    bake_ao(steel_m + [sediment] + ([stones] if len(stones.f) else []), occluders=occl, ground=flat, cell=0.6, rays=24, reach=12)
    lum = sediment.c.mean(1); ref = np.percentile(lum, 90)
    sediment.c = np.minimum(sediment.c / ref * base.mean() * (base / base.mean())[None], base[None])
    return sediment, stones, D.INS


def _materials(wood=False):
    return {
        "steel": (material(rough=0.65, metal=0.2, tex=steel_textures(1024, 3.0, 23, wear=0.3), uv=("tri", 3.0), normal_scale=0.5) if wood == "alu"
                  else material(rough=0.9, tex=planks_textures(1024, 4.0, 21, plank=0.3, length=7.0, along_u=True, wear=1.0), uv=("tri", 4.0), normal_scale=1.4)
                  if wood else material(rough=0.8, metal=0.05, tex=steel_textures(), uv=("tri", 7.2))),
        "rust": material(rough=0.95, tex=rust_textures(), uv=("tri", 1.5), normal_scale=1.6),
        "rock": material(rough=0.9, tex=rust_textures(512, 2.5, 11), uv=("tri", 2.5)),
        "sediment": material(rough=1.0),
        "life": material(rough=0.55),
        "china": material(rough=0.18),
        "glass": material(rough=0.08),
        "artefact": material(rough=0.7, metal=0.1),
        "net": material(rough=0.95),
        "oil": material(rough=0.3),
    }


def dress_wreck(out, sites, depth, kind, seed, memorial=False, wood=False, life="normal", drift=1.0):
    """sites: [{"steel": [(node name, Mesh)], "mud": [Mesh], "plain": [(node name, Mesh, material key)],
    "piece": None | (node name, {"translation", "rotation"})}] — the first is the main wreck; a piece lying apart
    (more than 1 km, meta.json "pieces") is written as one node with its transform, its own sediment and life inside.
    wood: True for a wooden ship, "alu" for an aircraft (no rust either), False for steel.
    drift: how much sediment piles against the wreck (1 = the usual for its depth; less where the hull lies exposed)."""
    zone = zone_of(depth)
    parts, INS = [], []
    for k, s in enumerate(sites):
        piece = s.get("piece")
        sediment, stones, ins = dress_site(s["steel"], s.get("mud", []), s.get("plain", []), zone, seed + 1000 * k, kind,
                                           memorial=memorial and k == 0, parent=piece[0] if piece else None, wood=wood, life=life, drift=drift)
        INS += ins
        if piece:
            items = [(e[1], "steel") for e in s["steel"]] + [(m, key) for _, m, key in s.get("plain", [])]
            if len(stones.f):
                items.append((stones, "rock"))
            items.append((sediment, "sediment", "sediment_" + piece[0]))
            parts.append((piece[0], items, piece[1]))
        else:
            parts += [(e[0], e[1], "steel") for e in s["steel"]]
            parts += [(name, m, key) for name, m, key in s.get("plain", [])]
            if len(stones.f):
                parts.append(("stones", stones, "rock"))
            parts.append(("sediment", sediment, "sediment"))
    tris, size = write_model(out, parts, _materials(wood), instanced=INS)
    print(f"{out}: {tris:,} triangles drawn ({len(INS)} instanced shapes), {size / 2**20:.2f} MB, zone {zone}")
    return tris, size


def dress_built(out, parts, deck_colors=(), seed=12):
    """The ship as built: planked decks (faces of a deck colour that look up), clean plating, baked light.
    parts: [(node name, Mesh)] painted steel, or [(node name, Mesh, key)] with key "paint", "wood" (a planked hull),
    "cloth" (sails), "rope" or "metal" (guns, fittings)."""
    parts = [p if len(p) == 3 else (p[0], p[1], "paint") for p in parts]
    allm = [m for _, m, k in parts if k != "rope"]
    bake_ao(allm, occluders=[m for _, m, k in parts if k not in ("rope", "cloth")], ground=None, cell=0.5, rays=20, reach=10, strength=0.75)
    out_parts = []
    dc = [np.array(c) for c in deck_colors]
    for name, m, key in parts:
        if dc and key in ("paint", "wood"):
            n = normals(m)
            fc = m.c[m.f].mean(1); fn = n[m.f].mean(1)
            isdeck = np.zeros(len(m.f), bool)
            for c in dc:
                rel = fc / np.maximum(fc.mean(1, keepdims=True), 1e-3) - c / max(c.mean(), 1e-3)    # same hue, any brightness (AO)
                isdeck |= (np.abs(rel).max(1) < 0.12) & (fc.mean(1) > c.mean() * 0.4) & (fn[:, 1] > 0.7)
            items = [(Mesh(m.v, m.f[~isdeck], m.c), key)]
            if isdeck.any():
                items.append((Mesh(m.v, m.f[isdeck], m.c), "deck"))
            out_parts.append((name, items))
        else:
            out_parts.append((name, m, key))
    keys = {k for _, _, k in parts}
    mats = {"paint": material(rough=0.6, metal=0.05, tex=steel_textures(1024, 7.2, seed, wear=0.15), uv=("tri", 7.2), normal_scale=0.6),
            "deck": material(rough=0.75, tex=planks_textures(), uv=("xz", 4.0))}
    if "wood" in keys:
        mats["wood"] = material(rough=0.7, tex=planks_textures(1024, 4.0, seed + 3, plank=0.28, length=7.0, along_u=True), uv=("tri", 4.0), normal_scale=1.2)
    if "cloth" in keys:
        mats["cloth"] = material(rough=0.95)
    if "rope" in keys:
        mats["rope"] = material(rough=0.95)
    if "metal" in keys:
        mats["metal"] = material(rough=0.4, metal=0.6)
    tris, size = write_model(out, out_parts, mats)
    print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB")
    return tris, size
