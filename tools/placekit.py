"""
placekit — places under the sea that are not ships, built by code (MIT): drowned towns and ports, habitats, holes in
the seabed, volcanic craters.

Pieces (metres, +Y up, the seabed at y = 0):
- stone: ashlar walls in courses with missing and fallen blocks, rubble, column drums and capitals, statues worn by
  the sea (sphinxes, colossi), amphorae, stone anchors, megaliths, wells, mosaic floors, brick walls;
- holes: a shaft by its profile [(depth, radius)], ragged and with ledges, stalactites under overhangs;
- dress_place(): the sediment drifted against everything (node «sediment», painted by Offlined with the seabed's
  own colour), baked ambient occlusion, stone / metal / paint materials, the compact writer. For a hole, the sediment
  is cut away over the opening and meta.json gets "hole" so Offlined opens the seabed down to the floor.

    from placekit import *
    wall = ashlar_wall((0, 0), (12, 0), 1.6, 0.8, STONE, seed=1)
    dress_place(out, [("ruins", wall)], depth=-6, seed=1)
"""
from __future__ import annotations

import math

import numpy as np

from meshkit import Mesh, merge, box, cyl, lathe, grid, tube, crumple, fbm, sstep
from scenekit import material, rust_textures, planks_textures, steel_textures, bake_ao, write_model
from dresskit import sediment_field, zone_of, SED_BASE

LIMESTONE = (0.66, 0.62, 0.54)
SANDSTONE = (0.62, 0.52, 0.4)
GRANITE_RED = (0.5, 0.36, 0.32)
GRANITE_GREY = (0.46, 0.45, 0.44)
MARBLE = (0.74, 0.72, 0.66)
BASALT = (0.24, 0.23, 0.22)
BRICK = (0.5, 0.3, 0.22)
CLAY = (0.6, 0.42, 0.3)
TIMBER = (0.27, 0.2, 0.14)
REEF_ROCK = (0.55, 0.52, 0.45)


def _tone(m, seed, k=0.25, freq=1.2):
    m.c = np.clip(m.c * (1 - k / 2 + k * fbm(m.v, freq, seed, 2))[:, None], 0, 1)
    return m


def compact(m):
    if not len(m.f):
        return Mesh()
    u, inv = np.unique(m.f, return_inverse=True)
    return Mesh(m.v[u], inv.reshape(-1, 3), m.c[u])


# ── Stone ────────────────────────────────────────────────────────────────────
def ashlar_wall(p0, p1, h, th, col, seed, block=1.0, course=0.5, ruin=0.5, fallen=0.3):
    """A wall of squared blocks from p0 to p1 (x, z) in courses, standing to about h, broken down (ruin 0 whole … 1
    almost gone); some of the missing blocks lie fallen beside it. Returns one Mesh."""
    rr = np.random.default_rng(seed)
    x0, z0 = p0; x1, z1 = p1
    ln = math.hypot(x1 - x0, z1 - z0)
    if ln < 0.1:
        return Mesh()
    yaw = math.atan2(x1 - x0, z1 - z0)
    n = max(1, int(ln / block))
    P = []
    for k in range(n):
        top = h * (1 - ruin * fbm(np.array([[k * 0.37, 0.0, seed * 0.1]]), 1.0, seed, 2)[0] * 1.6)
        layers = max(0, int(top / course))
        for c in range(layers):
            off = 0.5 * block * (c % 2)
            t = (k + 0.5) * block + off
            if t > ln:
                continue
            bl = block * rr.uniform(0.85, 0.98)
            b = box(th * rr.uniform(0.9, 1.0), course * 0.96, bl, col).R(0, rr.normal(0, 0.02), 0)
            P.append(b.T(rr.normal(0, 0.03), c * course, t - ln / 2))
            if c == layers - 1 and rr.random() < fallen:           # the course above, fallen off beside the wall
                fb = box(th, course * 0.96, bl, col).R(rr.uniform(-0.6, 0.6), rr.uniform(-1, 1), rr.uniform(-0.6, 0.6))
                P.append(fb.T(rr.choice([-1, 1]) * rr.uniform(th, th + 1.5), 0.2, t - ln / 2 + rr.normal(0, 0.4)))
    m = merge(P) if P else Mesh()
    if not len(m.f):
        return m
    _tone(m, seed)
    return m.R(0, yaw, 0).T((x0 + x1) / 2, 0, (z0 + z1) / 2)


def rubble(n, cx, cz, spread, size, col, seed, flat=0.6):
    """Loose stones and broken blocks spread round (cx, cz)."""
    rr = np.random.default_rng(seed)
    P = []
    for k in range(n):
        s = size * rr.uniform(0.4, 1.3)
        b = box(s * rr.uniform(0.7, 1.4), s * rr.uniform(0.4, 0.9) * flat, s * rr.uniform(0.7, 1.4), col)
        crumple(b, s * 0.08, 2.0 / max(s, 0.2), seed + k)
        P.append(b.R(rr.uniform(-0.3, 0.3), rr.uniform(0, 6.28), rr.uniform(-0.3, 0.3)).T(cx + rr.normal(0, spread), -s * 0.1, cz + rr.normal(0, spread)))
    return _tone(merge(P), seed + 1, 0.35)


def column_drum(r, h, col, fluted=True):
    m = cyl(r, r * 0.97, h, col, seg=20 if fluted else 14)
    if fluted:
        a = np.arctan2(m.v[:, 2], m.v[:, 0])
        k = 1 - 0.04 * (np.cos(a * 20) > 0.3) * (np.hypot(m.v[:, 0], m.v[:, 2]) > r * 0.8)
        m.v[:, 0] *= k; m.v[:, 2] *= k
    return m


def column(r, h, col, seed, fallen=False, broken=0.0, drums=1):
    """A column standing (broken off at h × (1 - broken)) or fallen on its side in its drums, a little apart."""
    rr = np.random.default_rng(seed)
    if not fallen:
        m = column_drum(r, h * (1 - broken), col)
        if broken:
            crumple(m, r * 0.3, 1.5, seed, sstep(h * (1 - broken) - 0.6, h * (1 - broken), m.v[:, 1]))
        return _tone(m, seed)
    P, z = [], 0.0
    for k in range(drums):
        dl = h / drums
        d = column_drum(r, dl, col).R(math.pi / 2, 0, 0).T(0, r * 0.9, z + dl / 2)
        P.append(d.R(0, rr.normal(0, 0.06), 0))
        z += dl + rr.uniform(0.02, 0.4)
    return _tone(merge(P), seed)


def capital(col, s=1.0):
    P = [box(1.1 * s, 0.18 * s, 1.1 * s, col).T(0, 0.55 * s, 0), cyl(0.36 * s, 0.42 * s, 0.4 * s, col, seg=14).T(0, 0.15 * s, 0)]
    for k in (-1, 1):
        P.append(cyl(0.26 * s, 0.26 * s, 0.95 * s, col, seg=12).R(math.pi / 2, 0, 0).T(k * 0.48 * s, 0.38 * s, 0))
    return merge(P)


def column_base(col, s=1.0):
    return lathe([(0.0, 0.0), (0.62 * s, 0.0), (0.62 * s, 0.12 * s), (0.52 * s, 0.2 * s), (0.56 * s, 0.3 * s), (0.45 * s, 0.38 * s), (0.0, 0.38 * s)], col, seg=18)


def sphinx(length, col, seed, worn=0.3):
    """A sphinx lying on its plinth: the lion body, the forepaws, the head with its nemes headdress (worn by the sea)."""
    s = length / 4.0
    P = [box(1.3 * s, 0.35 * s, 4.0 * s, col).T(0, 0, 0)]                                    # plinth
    body = box(1.1 * s, 1.1 * s, 2.6 * s, col).T(0, 0.35 * s, -0.4 * s)
    body.v[:, 1] -= 0.25 * s * sstep(-0.2 * s, -1.6 * s, body.v[:, 2]) * (body.v[:, 1] > 1.0 * s)   # the back slopes down to the haunches
    P.append(body)
    for k in (-1, 1):
        P.append(box(0.32 * s, 0.3 * s, 1.4 * s, col).T(k * 0.36 * s, 0.35 * s, 1.4 * s))         # forepaws
    P.append(box(0.9 * s, 0.6 * s, 0.7 * s, col).T(0, 0.9 * s, 0.85 * s))                       # chest
    head = merge([box(0.62 * s, 0.75 * s, 0.6 * s, col).T(0, 1.45 * s, 1.0 * s),                 # face and nemes
                  box(1.0 * s, 0.5 * s, 0.35 * s, col).T(0, 1.35 * s, 0.82 * s)])
    P.append(head)
    m = merge(P)
    crumple(m, worn * 0.12 * s, 1.5 / s, seed)
    return _tone(m, seed, 0.3)


def colossus(h, col, seed, fallen=True, broken=()):
    """A colossal standing statue of the Egyptian kind: a striding figure against its back pillar, the crown. Lying
    on its back if fallen; broken: names of parts gone ("head", "legs")."""
    s = h / 5.0
    P = [box(0.9 * s, 4.6 * s, 0.5 * s, col).T(0, 0.3 * s, -0.35 * s)]                         # back pillar
    if "legs" not in broken:
        P += [box(0.32 * s, 2.0 * s, 0.4 * s, col).T(k * 0.25 * s, 0.3 * s, dz * s) for k, dz in ((-1, 0.0), (1, 0.25))]
    P.append(box(0.95 * s, 1.6 * s, 0.55 * s, col).T(0, 2.3 * s, 0))                           # torso
    P += [box(0.22 * s, 1.5 * s, 0.3 * s, col).T(k * 0.6 * s, 2.3 * s, 0) for k in (-1, 1)]   # arms
    if "head" not in broken:
        P.append(box(0.5 * s, 0.6 * s, 0.5 * s, col).T(0, 3.9 * s, 0.02 * s))
        P.append(cyl(0.3 * s, 0.18 * s, 0.6 * s, col, seg=10).T(0, 4.5 * s, 0))                 # crown
    P.append(box(1.1 * s, 0.3 * s, 1.0 * s, col).T(0, 0, 0))                                    # base
    m = merge(P)
    crumple(m, 0.06 * s, 1.0 / s, seed)
    if fallen:
        m.R(-math.pi / 2 + 0.08, 0, 0).T(0, 0.3 * s, 0)
    return _tone(m, seed, 0.3)


def stela(h, col, seed):
    m = box(h * 0.55, h, h * 0.12, col)
    m.v[:, 0] *= 1 - 0.12 * (m.v[:, 1] > h * 0.99)
    return _tone(m, seed)


def amphora(col=CLAY):
    return lathe([(0.0, 0.0), (0.05, 0.02), (0.14, 0.25), (0.19, 0.55), (0.16, 0.8), (0.06, 0.95), (0.05, 1.12), (0.07, 1.15)], col, seg=10)


def amphorae(n, cx, cz, spread, seed, col=CLAY, broken=0.4):
    rr = np.random.default_rng(seed)
    P = []
    for k in range(n):
        a = amphora(col); a.R(math.pi / 2 + rr.uniform(-0.4, 0.4), rr.uniform(0, 6.28), 0)
        if rr.random() < broken:
            fc = a.v[a.f].mean(1)
            a.f = a.f[fc[:, 1] < rr.uniform(-0.05, 0.12)]
        P.append(a.T(cx + rr.normal(0, spread), 0.14, cz + rr.normal(0, spread)))
    return merge(P)


def stone_anchor(size, col, seed):
    """A pierced stone anchor of the ancient kind: a slab, rounded at the top, with its hole for the rope."""
    m = box(0.7 * size, 0.2 * size, 1.0 * size, col)
    m.v[:, 0] *= 1 - 0.25 * sstep(0.3 * size, 0.5 * size, m.v[:, 2])
    crumple(m, 0.03 * size, 3.0, seed)
    hole = cyl(0.11 * size, 0.11 * size, 0.22 * size, (0.06, 0.05, 0.05), seg=8).T(0, 0.0, 0.35 * size)
    return _tone(merge([m, hole]), seed)


def megalith(h, w, col, seed):
    m = box(w, h, w * 0.55, col)
    m.v[:, 0] *= 1 - 0.3 * (m.v[:, 1] / h) ** 2
    crumple(m, w * 0.12, 1.2 / w, seed)
    return _tone(m, seed, 0.35)


def well(r, wall_h, col, seed, depth=3.0):
    """A stone-lined well: a ring of stones round a dark shaft."""
    rr = np.random.default_rng(seed)
    P = [cyl(r * 0.8, r * 0.8, 0.05, (0.05, 0.05, 0.05), seg=14).T(0, 0.01, 0)]
    n = int(2 * math.pi * r / 0.45)
    for k in range(n):
        a = 2 * math.pi * k / n
        for c in range(max(1, int(wall_h / 0.3))):
            P.append(box(0.35, 0.28, 0.45, col).R(0, -a, 0).T(math.cos(a) * r, c * 0.3, math.sin(a) * r))
    return _tone(merge(P), seed, 0.35)


def mosaic(w, d, seed, cols=((0.78, 0.76, 0.7), (0.15, 0.15, 0.16), (0.55, 0.3, 0.22)), border=0.5, lost=0.25):
    """A floor of tesserae: a black-and-white field with a border and a geometric pattern, broken away in places."""
    nx, nz = max(4, int(w / 0.25)), max(4, int(d / 0.25))
    g = grid(-w / 2, w / 2, -d / 2, d / 2, nx, nz, lambda X, Z: np.zeros_like(X) + 0.05, cols[0])
    x, z = g.v[:, 0], g.v[:, 2]
    edge = (np.abs(x) > w / 2 - border) | (np.abs(z) > d / 2 - border)
    pat = ((np.floor(x / 0.75) + np.floor(z / 0.75)) % 2 == 0) & ~edge
    g.c[pat] = cols[1]
    g.c[edge] = cols[2]
    fc = g.v[g.f].mean(1)
    g.f = g.f[fbm(fc, 0.35, seed) > lost]
    return _tone(g, seed, 0.15, 3.0)


def brick_wall(p0, p1, h, th, seed, ruin=0.5):
    """A brick wall (a single mass with the courses suggested in its colour), broken down."""
    rr = np.random.default_rng(seed)
    x0, z0 = p0; x1, z1 = p1
    ln = math.hypot(x1 - x0, z1 - z0)
    n = max(2, int(ln / 0.5))
    P = []
    for k in range(n):
        top = h * (1 - ruin * fbm(np.array([[k * 0.21, 0.0, seed * 0.1]]), 0.8, seed, 2)[0] * 1.5)
        if top > 0.1:
            P.append(box(th, top, ln / n + 0.01, BRICK).T(0, 0, (k + 0.5) * ln / n - ln / 2))
    m = merge(P) if P else Mesh()
    if not len(m.f):
        return m
    m.c = np.clip(m.c * (0.85 + 0.15 * (np.floor(m.v[:, 1] / 0.08) % 2))[:, None] * (0.85 + 0.3 * fbm(m.v, 1.0, seed, 2))[:, None], 0, 1)
    return m.R(0, math.atan2(x1 - x0, z1 - z0), 0).T((x0 + x1) / 2, 0, (z0 + z1) / 2)


def timbers(n, cx, cz, spread, seed, col=TIMBER, length=(1.0, 5.0)):
    rr = np.random.default_rng(seed)
    P = []
    for k in range(n):
        ln = rr.uniform(*length)
        p = box(rr.uniform(0.15, 0.4), rr.uniform(0.1, 0.25), ln, col)
        crumple(p, 0.04, 0.6, seed + k)
        P.append(p.R(rr.uniform(-0.1, 0.1), rr.uniform(0, 6.28), rr.uniform(-0.15, 0.15)).T(cx + rr.normal(0, spread), 0.02, cz + rr.normal(0, spread)))
    return merge(P)


# ── Holes in the seabed ─────────────────────────────────────────────────────
def shaft(profile, col, seed, seg=72, rough=0.12, floor_col=None):
    """The walls of a hole from the rim (y = 0) down: profile [(depth below the rim, radius)], top first, with a
    floor at the last depth. The wall is noisy (rough × radius) and seen from inside."""
    rings = []
    for k, (dpt, rad) in enumerate(profile):
        a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
        pts = np.stack([np.cos(a) * rad, np.full(seg, -dpt), np.sin(a) * rad], 1)
        n = fbm(pts * [1, 0.6, 1], 0.06, seed, 3) - 0.5
        n2 = fbm(pts, 0.25, seed + 1, 2) - 0.5
        r = rad * (1 + rough * 2 * n) + 1.2 * n2
        rings.append(np.stack([np.cos(a) * r, pts[:, 1], np.sin(a) * r], 1))
    v = np.vstack(rings)
    f = []
    for i in range(len(rings) - 1):
        for j in range(seg):
            a_, b_ = i * seg + j, i * seg + (j + 1) % seg
            c_, d_ = a_ + seg, b_ + seg
            f += [(a_, c_, b_), (b_, c_, d_)]
    wall = Mesh(v, f, col)
    # the floor: a shallow bowl of silt and fallen blocks
    dpt, rad = profile[-1]
    fl = grid(-rad * 1.3, rad * 1.3, -rad * 1.3, rad * 1.3, 40, 40,
              lambda X, Z: -dpt + 0.08 * rad * (np.hypot(X, Z) / rad) ** 2 + 1.5 * (fbm(np.stack([X.ravel(), X.ravel() * 0, Z.ravel()], 1), 0.05, seed + 2, 3).reshape(X.shape) - 0.5),
              floor_col or col)
    fc = fl.v[fl.f].mean(1)
    fl.f = fl.f[np.hypot(fc[:, 0], fc[:, 2]) < rad * 1.15]
    return _tone(wall, seed, 0.35, 0.08), compact(fl)


def stalactites(n, depth0, depth1, radius_at, seed, col, length=(1.0, 6.0)):
    """Stalactites hanging under overhangs on the wall between depth0 and depth1 (radius_at(depth) -> wall radius)."""
    rr = np.random.default_rng(seed)
    P = []
    for k in range(n):
        dpt = rr.uniform(depth0, depth1)
        a = rr.uniform(0, 2 * math.pi)
        rad = radius_at(dpt) * rr.uniform(0.9, 0.99)
        ln = rr.uniform(*length)
        st = lathe([(rr.uniform(0.25, 0.7), 0.0), (rr.uniform(0.15, 0.4), -ln * 0.5), (0.03, -ln)], col, seg=7)
        crumple(st, 0.06, 2.0, seed + k)
        P.append(st.T(math.cos(a) * rad, -dpt, math.sin(a) * rad))
    return _tone(merge(P), seed, 0.3)


# ── Dressing and writing ─────────────────────────────────────────────────────
def _mats():
    return {
        "stone": material(rough=0.92, tex=rust_textures(512, 2.5, 11), uv=("tri", 2.5), normal_scale=0.8),
        "rock": material(rough=0.95, tex=rust_textures(512, 4.0, 13), uv=("tri", 4.0), normal_scale=1.2),
        "metal": material(rough=0.55, metal=0.4, tex=steel_textures(1024, 3.0, 21, wear=0.4), uv=("tri", 3.0), normal_scale=0.5),
        "paint": material(rough=0.7, metal=0.1),
        "wood": material(rough=0.9, tex=planks_textures(1024, 4.0, 21, plank=0.3, length=7.0, along_u=True, wear=1.0), uv=("tri", 4.0), normal_scale=1.2),
        "artefact": material(rough=0.7, metal=0.1),
        "glass": material(rough=0.08),
        "dark": material(rough=1.0),
        "sediment": material(rough=1.0),
    }


def dress_place(out, solid, plain=(), depth=-10.0, seed=1, drift=0.5, hole=None, sediment=True):
    """solid: [(node name, Mesh, material key)] that the sediment drifts against (key "stone", "rock", "metal", "paint",
    "wood"); plain: [(node name, Mesh, material key)] laid on the sediment without drifts (small finds).
    hole: radius of an opening at the origin: the sediment is cut away over it (and the walls are not buried)."""
    zone = zone_of(depth)
    solid_m = [m for _, m, _ in solid]
    parts = list(solid)
    allv = np.vstack([m.v for m in solid_m if len(m.v)])
    cell = max(0.6, float((allv.max(0) - allv.min(0)).max()) / 260)     # big places (a hole 300 m across): a coarser AO grid
    if sediment:
        near = [m for m in solid_m if len(m.v) and m.v[:, 1].max() > -0.5]
        sed, hAt = sediment_field(near, [], zone, seed, drift=drift)
        if sed.tris > 90000:                                 # a big place (a crater 500 m across): coarser cells
            sed, hAt = sediment_field(near, [], zone, seed, cell=math.sqrt(sed.tris / 70000), drift=drift)
        if hole:
            fc = sed.v[sed.f].mean(1)
            sed.f = sed.f[np.hypot(fc[:, 0], fc[:, 2]) > hole]
            sed = compact(sed)
        base = SED_BASE[zone]
        for m in solid_m:                                    # the foot of everything dusted with silt
            hh = m.v[:, 1] - hAt(m.v[:, 0], m.v[:, 2])
            k = 0.7 * sstep(0.8, -0.2, hh) * (m.v[:, 1] > -0.5)
            m.c = m.c * (1 - k[:, None]) + (base * 0.8)[None] * k[:, None]
        for _, m, _ in plain:
            m.v[:, 1] += hAt(m.v[:, 0], m.v[:, 2]) * (m.v[:, 1] < 0.6)
        flat = lambda X, Z: np.zeros_like(X)
        bake_ao(solid_m + [sed] + [m for _, m, _ in plain], occluders=solid_m + [sed], ground=None if hole else flat,
                cell=cell, rays=22, reach=10, strength=0.8)
        lum = sed.c.mean(1); ref = np.percentile(lum, 90)
        sed.c = np.minimum(sed.c / ref * base.mean() * (base / base.mean())[None], base[None])
        parts.append(("sediment", sed, "sediment"))
    else:
        bake_ao(solid_m + [m for _, m, _ in plain], occluders=solid_m, ground=None, cell=cell, rays=22, reach=10, strength=0.8)
    parts += list(plain)
    tris, size = write_model(out, parts, _mats())
    print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB, zone {zone}")
    return tris, size
