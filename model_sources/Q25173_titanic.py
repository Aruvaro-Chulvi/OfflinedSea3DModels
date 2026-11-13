"""
RMS Titanic (Q25173) — the wreck as it lies today, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

Real size from the ship's published dimensions (length 269.1 m, beam 28.2 m, moulded depth 19.6 m to the
shelter deck, 22.3 m to the bridge deck, draught 10.5 m). The wreck, after the published surveys:
- the bow section (~143 m, up to between the 3rd and 4th funnels) stands upright, its bow ploughed into the
  mud; the funnels are gone, the foremast has fallen back over the bridge, the decks sag towards the break;
- the stern section (~109 m) lies about 600 m to the south, facing the opposite way, collapsed on itself;
- between them, a debris field with boilers, hull plates and smaller pieces, heavier near the stern.
Headings are approximate (bow roughly north).

    python model_sources/Q25173_titanic.py        -> models/item/Q25173/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, weather, sstep, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1912)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B = 269.1, 28.2
H = 19.6                 # keel to shelter (C) deck
HB = 22.3                # keel to bridge (B) deck
HA = 25.4                # top of the A-deck house = boat deck
HT = 28.3                # top of the boat-deck houses
WL = 10.5                # waterline (draught)
BLACK, RED, WHITE = (0.07, 0.07, 0.08), (0.42, 0.11, 0.07), (0.6, 0.55, 0.47)      # white paint, stained after a century
DECK, DARK, BUFF, RUST = (0.33, 0.28, 0.23), (0.025, 0.025, 0.03), (0.66, 0.5, 0.28), (0.4, 0.19, 0.08)


def zd(d):            # distance from the bow (m) -> ship coordinate z (bow at +L/2)
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):          # half-beam factor along the ship
    w = 1.0
    if t > 0.78:
        w = math.sin(((1 - t) / 0.22) * math.pi / 2) ** 1.25      # fine entrance, sharp stem
    if t < 0.12:
        w = min(w, 0.32 + 0.68 * math.sin((t / 0.12) * math.pi / 2))
    return max(w, 0.004)


def yb_of(t):         # keel line (rises at the counter stern and at the forefoot)
    y = 0.0
    if t < 0.07:
        y = H * 0.55 * ((0.07 - t) / 0.07) ** 1.6
    if t > 0.988:
        y = max(y, H * 0.85 * ((t - 0.988) / 0.012) ** 2)
    return y


def yd_of(t):         # shelter deck with a little sheer
    return H * (1 + 0.03 * (2 * t - 1) ** 2)


def k_of(t):          # section fullness (boxy midships, fine at the ends)
    return 2.2 + 6.0 * w_of(t) ** 3


def hw(z, y):         # half-width of the hull at z and height y
    t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    if y >= yd:
        return B / 2 * w
    u = max(0.0, (y - yb) / max(0.1, yd - yb))
    return B / 2 * w * (1 - (1 - u) ** k)


def hull(d0, d1, NP=22):
    """Hull from d0 to d1 (distance from bow): both sides, the deck and, at the stern end, the counter."""
    ns = max(2, int(math.ceil((d1 - d0) / 1.5)))
    st = []
    for i in range(ns + 1):
        z = zd(d1) + (zd(d0) - zd(d1)) * i / ns              # from aft (smaller z) to forward
        t = t_of(z)
        st.append((z, w_of(t), yb_of(t), yd_of(t), k_of(t)))
    W = 2 * NP + 1
    v, f = [], []
    for z, w, yb, yd, k in st:
        for j in range(W):
            side = -1 if j < NP else 1
            u = abs(NP - j) / NP
            v.append((side * B / 2 * w * (1 - (1 - u) ** k), yb + (yd - yb) * u, z))
    for i in range(ns):
        for j in range(W - 1):
            a = i * W + j; b = a + W; c = a + 1; d = b + 1
            f += [(a, c, b), (b, c, d)]
    d0i = len(v)
    for z, w, yb, yd, k in st:
        v += [(-B / 2 * w, yd, z), (B / 2 * w, yd, z)]
    for i in range(ns):
        a = d0i + 2 * i; b = a + 2; c = a + 1; d = b + 1
        f += [(a, b, c), (c, b, d)]
    if d1 >= L - 0.01:                                       # counter stern closed
        c0 = len(v); z, w, yb, yd, k = st[0]
        v.append((0, (yb + yd) / 2, z))
        f += [(c0, j + 1, j) for j in range(W - 1)]
    hl = Mesh(v, f)
    hl.paint(lambda p: np.where((p[:, 1] < WL)[:, None], RED, BLACK))
    return hl


def band(d0, d1, y_extra, color=BLACK, top=DECK):
    """A raised hull band (forecastle, bridge deck, poop): vertical sides on the hull, deck on top, closed ends."""
    ns = max(2, int(math.ceil((d1 - d0) / 1.5)))
    parts = []
    zs = [zd(d1) + (zd(d0) - zd(d1)) * i / ns for i in range(ns + 1)]
    ys = [yd_of(t_of(z)) for z in zs]
    xs = [B / 2 * w_of(t_of(z)) for z in zs]
    for side in (-1, 1):
        v, f = [], []
        for z, y0, x in zip(zs, ys, xs):
            v += [(side * x, y0 - 0.05, z), (side * x, y0 + y_extra, z)]
        for i in range(ns):
            a = 2 * i
            f += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)] if side < 0 else [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
        parts.append(Mesh(v, f, color))
    v, f = [], []
    for z, y0, x in zip(zs, ys, xs):
        v += [(-x, y0 + y_extra, z), (x, y0 + y_extra, z)]
    for i in range(ns):
        a = 2 * i; b = a + 2; c = a + 1; d = b + 1
        f += [(a, b, c), (c, b, d)]
    parts.append(Mesh(v, f, top))
    for z, y0, x, s in ((zs[0], ys[0], xs[0], -1), (zs[-1], ys[-1], xs[-1], 1)):
        if x > 0.5:
            parts.append(Mesh([(-x, y0, z), (x, y0, z), (x, y0 + y_extra, z), (-x, y0 + y_extra, z)],
                              [(0, 1, 2), (0, 2, 3)] if s > 0 else [(0, 2, 1), (0, 3, 2)], color))
    return merge(parts)


def at(m, d, x=0.0, y=0.0):
    return m.T(x, y, zd(d))


def in_range(d, d0, d1):
    return d0 <= d <= d1


def ship_part(d0, d1):
    """Everything of the ship between d0 and d1 (distance from the bow), as built."""
    P = [hull(d0, d1)]
    for a, b in ((0, 39), (58, 207), (227, L)):              # forecastle, bridge deck, poop
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 > 1:
            P.append(band(a2, b2, HB - H))
    # A-deck house (white), its windows and the open promenade
    a2, b2 = max(60, d0), min(205, d1)
    if b2 - a2 > 1:
        P.append(at(box(B * 0.94, HA - HB, b2 - a2, WHITE), (a2 + b2) / 2, y=HB))
        for d in np.arange(a2 + 1.5, b2 - 1, 2.6 if a2 < 110 else 3.0):
            big = d > 110
            for s in (-1, 1):
                P.append(quad(s * B * 0.47, HB + 1.6, zd(d), 2.5 if big else 1.2, 2.0 if big else 1.1, "+x" if s > 0 else "-x", DARK))
    # Boat-deck houses (officers' quarters, entrances, lounges) between the funnels
    for a, b, wf in ((58, 80, 0.62), (83, 99, 0.5), (107, 125, 0.5), (133, 151, 0.5), (159, 178, 0.5), (182, 200, 0.55)):
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 > 1:
            P.append(at(box(B * wf, HT - HA, b2 - a2, WHITE), (a2 + b2) / 2, y=HA))
    # Funnels: gone on the wreck; their stubs and the dark uptakes remain
    for d in (77, 103, 129, 155):
        if in_range(d, d0 + 4, d1 - 4):
            P.append(at(cyl(3.65, 3.5, 1.4, (0.18, 0.14, 0.1), seg=18, sx=0.78), d, y=HT))
            P.append(at(cyl(3.1, 3.1, 0.05, DARK, seg=18, sx=0.78), d, y=HT + 1.42))
    # Grand staircase: the dome is gone, a dark hole
    if in_range(90, d0, d1):
        P.append(quad(0, HT, zd(91), 6.5, 8.5, "+y", DARK, eps=0.03))
    # Navigating bridge: the wheelhouse is gone, the bridge wings remain
    if in_range(58, d0, d1):
        P.append(at(box(B * 1.0, 0.5, 2.6, WHITE), 57.2, y=HA))
    # Lifeboat davits (the boats are gone)
    for d in list(np.arange(60, 82, 5.2)) + list(np.arange(176, 202, 6.2)):
        if in_range(d, d0, d1):
            for s in (-1, 1):
                x = s * B * 0.46
                P.append(tube([(x, HA, zd(d)), (x, HA + 3.2, zd(d)), (x + s * 0.9, HA + 4.0, zd(d)), (x + s * 1.7, HA + 3.6, zd(d))], 0.17, WHITE, seg=6))
    # Ventilator cowls on the boat deck
    for d, x in ((86, 9), (96, 9), (111, 9.5), (122, 9.5), (138, 9), (148, 9), (186, 8.5), (196, 8.5)):
        if in_range(d, d0, d1):
            for s in (-1, 1):
                P.append(at(cyl(0.55, 0.55, 2.2, WHITE, seg=10), d, x=s * x, y=HA))
                P.append(at(box(1.3, 1.1, 1.2, WHITE), d + 0.4, x=s * x, y=HA + 2.0))
    # Portholes: rows on the hull and on the bridge-deck band
    for y, a, b, sp in ((12.4, 30, 240, 2.4), (15.0, 18, 250, 1.9), (17.6, 12, 256, 1.7), (HB - 1.4, 60, 205, 1.7), (HB - 1.4, 3, 37, 2.2), (HB - 1.4, 229, 262, 2.0)):
        for d in np.arange(max(a, d0 + 1), min(b, d1 - 1), sp):
            z = zd(d)
            x = hw(z, y)
            if x < 2:
                continue
            for s in (-1, 1):
                P.append(quad(s * x, y, z, 0.42, 0.42, "+x" if s > 0 else "-x", DARK, eps=0.05))
    # Forecastle: railing at the bow, anchor crane, capstans, the anchors on both sides
    if d0 <= 1:
        yb = lambda d: yd_of(t_of(zd(d))) + (HB - H)
        pts_r = [(hw(zd(d), HB + 2) - 0.25, yb(d) + 1.15, zd(d)) for d in np.arange(0.6, 39, 1.5)]
        for s in (-1, 1):
            P.append(tube([(s * x, y, z) for x, y, z in pts_r], 0.07, WHITE, seg=5))
            for x, y, z in pts_r[::1]:
                P.append(cyl(0.05, 0.05, 1.15, WHITE, seg=4, caps=False).T(s * x, y - 1.15, z))
        P.append(at(cyl(0.45, 0.3, 7.5, BLACK, seg=8), 4.5, y=yb(4.5)))                          # anchor crane
        P.append(tube([(0, yb(4.5) + 7.3, zd(4.5)), (0, yb(4.5) + 7.8, zd(9.5))], 0.25, BLACK, seg=6))
        for d, x in ((9, 2.5), (9, -2.5), (13, 0), (16, 3.8), (16, -3.8)):
            P.append(at(cyl(0.7, 0.6, 1.0, BLACK, seg=12), d, x=x, y=yb(d)))
        for s in (-1, 1):
            z = zd(8.5); x = s * (hw(z, 17.5) + 0.35)
            P.append(box(0.55, 3.6, 0.8, BLACK).T(x, 15.8, z))
            P.append(box(0.6, 0.7, 3.0, BLACK).T(x, 15.6, z))
    # Cargo cranes in the well decks
    for d, x in ((42, 4.5), (42, -4.5), (55, 4.5), (55, -4.5), (210, 4), (210, -4), (224, 4), (224, -4)):
        if in_range(d, d0 + 1, d1 - 1):
            y = yd_of(t_of(zd(d)))
            P.append(at(cyl(0.9, 0.8, 2.2, WHITE, seg=10), d, x=x, y=y))
            P.append(tube([(x, y + 2.2, zd(d)), (x * 1.3, y + 8.5, zd(d) + (6 if d < 100 else -6))], 0.18, WHITE, seg=5))
    return P


# ── Bow section ──────────────────────────────────────────────────────────────
BOW_END = 143.0
bow = merge(ship_part(0, BOW_END))
# the foremast, fallen back over the bridge
y0 = yd_of(t_of(zd(45)))
bow = merge([bow, tube([(0, y0, zd(45)), (0.4, HT + 0.6, zd(86))], 0.42, (0.12, 0.1, 0.09), seg=8),
             cyl(0.9, 0.9, 1.6, (0.12, 0.1, 0.09), seg=10).R(0, 0, math.pi / 2 - 0.22).T(0.6, HT - 1.2, zd(76))])   # crow's nest
dfb = L / 2 - bow.v[:, 2]                                       # distance from the bow of every vertex
w = sstep(110, BOW_END, dfb)
bow.v[:, 1] -= w * (bow.v[:, 1] / HB) * 7.0                    # the decks sag towards the break
crumple(bow, 2.2, 0.12, 11, w)
crumple(bow, 3.0, 0.35, 12, sstep(136, BOW_END, dfb))          # torn edges at the break
crumple(bow, 0.25, 0.08, 13)                                    # dents everywhere
zc = (zd(0) + zd(BOW_END)) / 2
bow.T(0, 0, -zc)
bow.R(math.atan(4 / BOW_END), 0, 0)                              # bow down: ploughed into the mud
bow.T(0, -13.0, 0)
weather(bow, rust=0.75, sediment=0.6, streaks=0.9, seed=21)

# the mud ridge the bow pushed up (it blends into the seabed at its edges)
ztip = BOW_END / 2


def mud_h(X, Z):
    d = ztip - Z
    hwd = np.array([B / 2 * w_of(t_of(zd(max(0.0, min(L, dd))))) for dd in d.ravel()]).reshape(d.shape)
    side = 3.4 * np.exp(-((np.abs(X) - hwd - 3.5) / 4.5) ** 2) * sstep(95, 25, d) * (d > -2)
    front = 1.8 * np.exp(-(X / 7) ** 2) * np.exp(-((d + 4) / 5) ** 2)
    edge = sstep(42, 30, np.abs(X)) * sstep(-25, -15, -d + 0) * sstep(110, 95, d)
    ridge = np.maximum(side, front)
    bumps = 0.5 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.2, 5).reshape(X.shape) * sstep(0.2, 1.2, ridge)
    return (ridge + bumps) * edge - 0.6               # flat parts stay under the seabed


mud = grid(-42, 42, ztip - 110, ztip + 26, 42, 68, mud_h, (0.52, 0.49, 0.43))
mud.paint(lambda p: (0.62, 0.58, 0.5) + (fbm(p, 0.12, 9)[:, None] - 0.5) * 0.1)
mud.f = mud.f[mud.v[mud.f][:, :, 1].max(axis=1) > 0.15]   # only the ridges: no flat sheet that could show above another seabed

# ── Stern section ────────────────────────────────────────────────────────────
STERN_START = 160.0
stern = merge(ship_part(STERN_START, L))
# rudder and propellers (mostly buried)
zs = zd(L) + 1.5
stern = merge([stern, box(0.8, 22, 4.2, BLACK).T(0, 0.5, zs - 1.0),
               cyl(2.6, 2.6, 0.5, (0.35, 0.27, 0.15), seg=16).R(math.pi / 2, 0, 0).T(0, 5, zs + 4.5),
               cyl(3.5, 3.5, 0.5, (0.35, 0.27, 0.15), seg=16).R(math.pi / 2, 0, 0).T(7.5, 5.5, zs + 12),
               cyl(3.5, 3.5, 0.5, (0.35, 0.27, 0.15), seg=16).R(math.pi / 2, 0, 0).T(-7.5, 5.5, zs + 12),
               tube([(0, HB, zd(218)), (2.5, HB + 4, zd(250))], 0.4, (0.12, 0.1, 0.09), seg=8)])           # fallen mainmast
dfs = L / 2 - stern.v[:, 2]
# collapsed: the decks pancaked down, the sides peeled outwards, the forward end torn apart
keep = 0.32 + 0.43 * sstep(240, 262, dfs)
y = stern.v[:, 1]
stern.v[:, 1] = np.where(y > 6, 6 + (y - 6) * keep, y)
stern.v[:, 0] *= 1 + 0.42 * sstep(6, 26, y) * (1 - 0.6 * sstep(240, 262, dfs))
crumple(stern, 4.0, 0.12, 31, sstep(190, STERN_START, dfs))
crumple(stern, 1.4, 0.1, 32)
crumple(stern, 0.5, 0.4, 33)
zc2 = (zd(STERN_START) + zd(L)) / 2
stern.T(0, 0, -zc2).R(0, math.pi, 0.1).T(14, -12.5, -600)       # facing the opposite way, ~600 m south
weather(stern, rust=0.9, sediment=0.75, streaks=0.8, seed=41)

# ── Debris field ─────────────────────────────────────────────────────────────
deb = []


def spot(near_stern=0.65):
    if rng.random() < near_stern:
        return rng.normal(18, 45), rng.normal(-600, 55)
    return rng.normal(6, 30), rng.uniform(-560, -90)


for i in range(5):                                               # boilers (Scotch boilers, 4.8 m wide)
    x, z = rng.normal(20, 18), rng.normal(-560, 25)
    m = cyl(2.4, 2.4, 6.1, (0.16, 0.12, 0.09), seg=16).T(0, -3.05, 0).R(0, 0, math.pi / 2).R(0, rng.uniform(0, math.pi), 0)
    deb.append(m.T(x, 1.6, z))
for i in range(80):                                              # hull plates and deck pieces, bent
    x, z = spot()
    m = box(rng.uniform(2, 9), 0.18, rng.uniform(1.5, 4.5), BLACK if rng.random() < 0.6 else WHITE)
    crumple(m, 0.6, 0.3, 100 + i)
    deb.append(m.R(rng.uniform(-0.35, 0.35), rng.uniform(0, 6.28), rng.uniform(-0.35, 0.35)).T(x, 0.1, z))
for i in range(28):                                              # beams
    x, z = spot()
    deb.append(box(0.4, 0.6, rng.uniform(4, 12), (0.2, 0.15, 0.1)).R(0, rng.uniform(0, 6.28), rng.uniform(-0.2, 0.2)).T(x, 0, z))
for i in range(180):                                             # smaller things
    x, z = spot(0.55)
    s = rng.uniform(0.3, 1.4)
    deb.append(box(s, s * rng.uniform(0.4, 1.0), s * rng.uniform(0.6, 1.6), (0.25, 0.2, 0.15)).R(0, rng.uniform(0, 6.28), 0).T(x, -0.1, z))
big = box(22, 2.5, 12, (0.1, 0.08, 0.07))                       # a piece of the double bottom
crumple(big, 1.5, 0.15, 77)
deb.append(big.R(0.08, 0.6, 0.05).T(-6, -0.4, -350))
debris = merge(deb)
weather(debris, rust=0.85, sediment=0.75, streaks=0.5, seed=61)

# ── Write ────────────────────────────────────────────────────────────────────
out = ROOT / "models" / "item" / "Q25173" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
tris, size = write_glb(out, [("bow_section", bow), ("mud", mud), ("stern_section", stern), ("debris_field", debris)])
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB")
for n, m in (("bow", bow), ("mud", mud), ("stern", stern), ("debris", debris)):
    print(f"  {n}: {m.tris:,} tris, x {m.v[:, 0].min():.0f}..{m.v[:, 0].max():.0f}, y {m.v[:, 1].min():.0f}..{m.v[:, 1].max():.0f}, z {m.v[:, 2].min():.0f}..{m.v[:, 2].max():.0f}")
