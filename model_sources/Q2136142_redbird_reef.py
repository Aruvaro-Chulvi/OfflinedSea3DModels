"""
Redbird Reef (Q2136142) — the artificial reef of New York subway cars off Delaware, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Delaware DNREC, Wikipedia, CBC, Scientific American): reef site 11 of
the Delaware artificial reef system, about 16 miles off the coast in some 80 ft (24 m) of water, covering about
1.3 square miles. From 2001 to 2007 997 retired New York City "Redbird" subway cars, stripped of their wheels,
doors, windows and fittings, were dropped from barges, with 86 old Army tanks, tugboats and barges, a fishing
trawler, a Chesapeake Bay cruise ship and, in 2024, a tug and the Mayor J. Harold Grady; lying where they fell,
some on their sides. On the flat sandy bottom where there was little to cling to, the steel is now crusted with
blue mussels, sponges and anemones, and black sea bass and tautog crowd about them. Here a patch about 400 m across
is drawn with a sample of the cars (hollow, their window and door openings open, the arched roof, the end walls, the
floor and centre sills; no wheels), a few M60 tanks and a tug; the layout is approximate.

    python model_sources/Q2136142_redbird_reef.py   -> models/item/Q2136142/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, quad, tube, crumple, sstep, fbm  # noqa: E402
import abyss  # noqa: E402
from shipkit import install  # noqa: E402

rng = np.random.default_rng(2001)
RED, DARK, GREY, OLIVE = (0.55, 0.12, 0.1), (0.03, 0.03, 0.035), (0.35, 0.36, 0.37), (0.3, 0.32, 0.22)


install(globals(), "kit_window")
INSIDE = (0.32, 0.12, 0.09)
ROOFC = (0.42, 0.41, 0.4)
CW, CH, CL = 2.7, 3.0, 15.5           # a Redbird car body: width, height of the sides, length


def slab(x0, x1, y0, y1, z0, z1, col):
    m = box(x1 - x0, y1 - y0, z1 - z0, col)
    return m.T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def wall(us, ys, holes, col, place):
    """A flat wall in one piece: a grid on the cuts us × ys with the cells inside holes [(u0, u1, y0, y1)] left open.
    place(u, y) -> 3-D point. (Materials are double-sided: no thickness needed.)"""
    us, ys = sorted(set(us)), sorted(set(ys))
    idx = {}
    v, f = [], []

    def vid(i, j):
        if (i, j) not in idx:
            idx[(i, j)] = len(v); v.append(place(us[i], ys[j]))
        return idx[(i, j)]

    for i in range(len(us) - 1):
        for j in range(len(ys) - 1):
            cu, cy = (us[i] + us[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(a < cu < b and c < cy < d for a, b, c, d in holes):
                continue
            a_, b_, c_, d_ = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            f += [(a_, b_, c_), (a_, c_, d_)]
    return Mesh(v, f, col)


def subway_car(seed):
    """A Redbird car stripped for the reef: a hollow body whose window and door openings are holes (you see into it),
    the arched roof, the end walls with their door openings, the floor and the centre sills; no wheels, doors or glass."""
    P = []
    win = [(z - 0.55, z + 0.55) for z in (-6.5, -4.3, -3.0, 0.0, 3.0, 4.3, 6.5)]
    doors = [(z - 0.65, z + 0.65) for z in (-5.4, -1.6, 1.6, 5.4)]
    y_floor, y_sill, y_wtop, y_dtop = 0.35, 1.25, 2.2, 2.35
    us = [-CL / 2, CL / 2] + [e for ab in win + doors for e in ab]
    ys = [y_floor, y_sill, y_wtop, y_dtop, CH]
    holes = [(a, b, y_sill, y_wtop) for a, b in win] + [(a, b, y_floor, y_dtop) for a, b in doors]
    for s in (-1, 1):
        P.append(wall(us, ys, holes, RED, lambda u, y, s=s: (s * CW / 2, y, u)))
    # the arched roof with its eaves
    nr = 8
    xs = np.linspace(-CW / 2, CW / 2, nr + 1)
    v, f = [], []
    for zz in (-CL / 2, CL / 2):
        for x in xs:
            v.append((x, CH + 0.32 * (1 - (2 * x / CW) ** 2), zz))
    for i in range(nr):
        a, b = i, i + 1
        f += [(a, b, nr + 1 + b), (a, nr + 1 + b, nr + 1 + a)]
    P.append(Mesh(v, f, ROOFC))
    # end walls: the end door opening and a window each side of it
    eu = [-CW / 2, -1.25, -0.65, -0.45, 0.45, 0.65, 1.25, CW / 2]
    ey = [y_floor, 1.65, 2.35, 2.4, CH + 0.3]
    eh = [(-0.45, 0.45, y_floor, 2.4), (-1.25, -0.65, 1.65, 2.35), (0.65, 1.25, 1.65, 2.35)]
    for s in (-1, 1):
        w = wall(eu, ey, eh, RED, lambda u, y, s=s: (u, min(y, CH + 0.32 * (1 - (2 * u / CW) ** 2)), s * CL / 2))
        P.append(w)
    P.append(slab(-CW / 2, CW / 2, 0.25, y_floor, -CL / 2, CL / 2, INSIDE))             # the floor
    for x in (-0.9, 0.9):
        P.append(slab(x - 0.12, x + 0.12, 0.0, 0.25, -CL / 2 + 0.4, CL / 2 - 0.4, (0.2, 0.18, 0.16)))   # the centre sills
    m = merge(P)
    crumple(m, 0.06, 0.4, seed)
    m.c = np.clip(m.c * (0.75 + 0.35 * fbm(m.v, 0.3, seed + 1)[:, None]), 0, 1)
    return m


def tank(seed):
    """An M60 tank, stripped: the hull with its sloped glacis, six road wheels and return rollers a side, the cast
    turret with its bustle and the commander's cupola, the 105 mm gun."""
    P = []
    hull = box(3.6, 1.25, 6.9, OLIVE).T(0, 0.55, 0)
    top = hull.v[:, 1] > 1.2
    hull.v[top & (hull.v[:, 2] > 0), 2] -= 1.2
    P.append(hull)
    for s in (-1, 1):
        x = s * 1.5
        P.append(box(0.6, 0.12, 6.6, DARK).T(x, 0, 0))
        P.append(box(0.6, 0.1, 5.8, DARK).T(x, 1.15, 0))
        for z in np.linspace(-2.6, 2.4, 6):
            P.append(cyl(0.38, 0.38, 0.45, (0.18, 0.18, 0.17), seg=10).R(0, 0, math.pi / 2).T(x + 0.22, 0.45, z))
        for z in (-1.6, 0.0, 1.6):
            P.append(cyl(0.12, 0.12, 0.3, (0.18, 0.18, 0.17), seg=6).R(0, 0, math.pi / 2).T(x + 0.15, 1.08, z))
    tur = lathe([(1.5, 0.0), (1.45, 0.4), (1.1, 0.85), (0.001, 0.9)], OLIVE, seg=14)
    tur.v[:, 2] *= 1.35
    P.append(tur.T(0, 1.8, -0.3))
    P.append(box(2.0, 0.6, 1.1, OLIVE).T(0, 1.85, -2.2))
    P.append(cyl(0.45, 0.45, 0.45, OLIVE, seg=10).T(0.6, 2.6, -0.6))
    P.append(cyl(0.11, 0.1, 5.6, OLIVE, seg=8).R(math.pi / 2, 0, 0).T(0, 2.25, 1.3))
    P.append(cyl(0.2, 0.2, 0.6, OLIVE, seg=8).R(math.pi / 2, 0, 0).T(0, 2.25, 3.4))
    m = merge(P)
    crumple(m, 0.05, 0.6, seed)
    return m


def tug():
    """The tug: round-bilged hull with its sheer and bulwark, rubber fenders along the sides, the deckhouse and the
    wheelhouse with its windows, the funnel, the mast, the towing bitts aft."""
    hull = lathe([(0.05, -11.5), (2.4, -10.0), (3.6, -5.0), (3.8, 0.0), (3.6, 5.0), (2.4, 9.0), (0.05, 11.0)], (0.15, 0.15, 0.17), seg=16, axis="z")
    hull.v[:, 1] = np.minimum(hull.v[:, 1], 0.4) + 1.8
    P = [hull]
    deck = lathe([(0.05, -11.4), (2.3, -10.0), (3.5, -5.0), (3.7, 0.0), (3.5, 5.0), (2.3, 9.0), (0.05, 10.9)], (0.3, 0.28, 0.25), seg=16, axis="z")
    deck.v[:, 1] = 2.2
    P.append(deck)
    for zz in np.linspace(-9.0, 8.0, 18):                                                 # the bulwark
        w = np.interp(zz, [-11.5, -10, -5, 0, 5, 9, 11], [0.05, 2.4, 3.6, 3.8, 3.6, 2.4, 0.05])
        for s in (-1, 1):
            P.append(box(0.08, 0.8, 1.0, (0.15, 0.15, 0.17)).T(s * (w - 0.04), 2.2, zz))
    for zz in np.linspace(-7.0, 6.0, 8):                                                  # fenders
        w = np.interp(zz, [-11.5, -10, -5, 0, 5, 9, 11], [0.05, 2.4, 3.6, 3.8, 3.6, 2.4, 0.05])
        for s in (-1, 1):
            P.append(cyl(0.45, 0.45, 0.35, (0.05, 0.05, 0.05), seg=10).R(0, 0, math.pi / 2).T(s * (w + 0.05), 2.0, zz))
    P.append(box(5.5, 2.6, 7.0, (0.8, 0.78, 0.72)).T(0, 2.2, 1.0))
    P.append(box(4.0, 2.2, 3.5, (0.8, 0.78, 0.72)).T(0, 4.8, 2.5))
    for x in np.linspace(-1.4, 1.4, 4):
        P.append(kit_window(x, 6.0, 4.26, 0.7, 0.8, "+z", glass=(0.05, 0.06, 0.07), frame=(0.55, 0.55, 0.52)))
    for s in (-1, 1):
        for zz in (1.6, 3.3):
            P.append(kit_window(s * 2.0, 6.0, zz, 0.8, 0.8, "+x" if s > 0 else "-x", glass=(0.05, 0.06, 0.07), frame=(0.55, 0.55, 0.52)))
        for zz in np.linspace(-1.6, 3.6, 4):
            P.append(kit_window(s * 2.75, 3.4, zz, 0.6, 0.6, "+x" if s > 0 else "-x", glass=(0.05, 0.06, 0.07), frame=(0.55, 0.55, 0.52)))
    P.append(cyl(0.7, 0.6, 3.0, (0.6, 0.15, 0.1), seg=12).T(0, 4.8, -1.5))
    P.append(cyl(0.72, 0.72, 0.6, (0.1, 0.1, 0.1), seg=12).T(0, 7.2, -1.5))
    P.append(tube([(0, 7.0, 3.0), (0, 11.0, 3.0)], 0.1, (0.6, 0.6, 0.58), seg=6))       # mast
    P.append(tube([(-0.9, 10.0, 3.0), (0.9, 10.0, 3.0)], 0.06, (0.6, 0.6, 0.58), seg=5))
    for s in (-1, 1):                                                                      # towing bitts
        P.append(cyl(0.22, 0.22, 0.8, (0.2, 0.2, 0.2), seg=8).T(s * 0.5, 2.2, -6.0))
    P.append(box(4.2, 0.5, 0.35, (0.2, 0.2, 0.2)).T(0, 2.2, -4.5))                         # the towing rail
    m = merge(P)
    crumple(m, 0.06, 0.4, 5)
    return m


r = np.random.default_rng(7)
cars, tanks = [], []
for k in range(110):                                                          # dropped from barges in loose rows, some on their sides
    row = k // 12
    x = -180 + row * 38 + r.normal(0, 6)
    z = -150 + (k % 12) * 26 + r.normal(0, 7)
    c = subway_car(100 + k)
    roll = math.pi / 2 if r.random() < 0.18 else r.normal(0, 0.08)
    c.R(r.normal(0, 0.03), r.uniform(-0.5, 0.5) + (math.pi / 2 if row % 2 else 0), roll)
    c.T(x, 1.4 if abs(roll) > 1 else -0.1, z)
    cars.append(c)
for k in range(8):
    t = tank(300 + k)
    tanks.append(t.R(0, r.uniform(0, 6.28), 0).T(r.uniform(-60, 60) + 140, -0.2, r.uniform(-80, 80)))
tugboat = tug().R(0, 0, math.radians(18)).R(0, 0.7, 0).T(160, -0.6, 130)
cars_m = merge(cars)
tanks_m = merge(tanks)
mussels = abyss.blob(0.25, (0.12, 0.13, 0.2), 9, sub=1, amp=0.5)
mus = []
for k, (p, n) in enumerate(abyss.sample_surface(cars_m, 2500, lambda fn: fn[:, 1] > -0.3, r)):
    if k > 1600:
        break
    m = mussels.copy(); m.v[:, 1] *= 0.4
    up = np.array([0, 1.0, 0]); ax = np.cross(up, n); s_ = np.linalg.norm(ax); co = float(np.dot(up, n))
    if s_ > 1e-6:
        ax = ax / s_; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        m.v = m.v @ (np.eye(3) + s_ * K + (1 - co) * K @ K).T
    mus.append(m.S(r.uniform(0.6, 1.6)).T(*p))
mussel_beds = merge(mus)

from dresskit import dress_wreck  # noqa: E402
out = ROOT / "models" / "item" / "Q2136142" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("subway_cars", cars_m, "loose"), ("tanks", tanks_m, "loose"), ("tug_section", tugboat)], "mud": [],
                   "plain": [("mussels", mussel_beds, "life")]}], depth=-24, kind="cargo", seed=2001, drift=0.4)
