"""Baiae (Q290097) — the drowned Roman resort in the bay of Pozzuoli: the Nymphaeum of Claudius, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (the Parco Archeologico Sommerso di Baia, Wikipedia, PADI, Scuba Diving,
Artnet): Baiae was the seaside resort of the Roman élite, with imperial palaces, villas and baths; the slow
subsidence of the Campi Flegrei volcanic area (bradyseism) has put much of it under water. Since 2002 it is an
underwater archaeological park, its remains lying between some 2 and 16 m. At Punta Epitaffio lies the Nymphaeum of
Claudius, a banqueting hall of the emperor's palace with a pool in the middle and statues in niches round its walls
and in the apse — the originals were raised in 1981 and are in the museum in the Castle of Baia; copies stand in their
places today — and along it a paved road with the foundations of rooms, baths and the black-and-white mosaic floors of
villas. Here the nymphaeum and the road beside it are drawn, about 80 m across; their layout is approximate.

    python model_sources/Q290097_baia.py   -> models/item/Q290097/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, fbm, crumple  # noqa: E402
from placekit import (ashlar_wall, brick_wall, rubble, column, mosaic, amphorae, MARBLE, BASALT, CLAY, dress_place)  # noqa: E402

rng = np.random.default_rng(41)
TUFA = (0.62, 0.56, 0.42)


def roman_statue(seed, h=1.9):
    """A draped figure on its base, worn by the sea: the robe, the torso, the head, an arm."""
    rr = np.random.default_rng(seed)
    P = [box(0.8, 0.5, 0.8, MARBLE), lathe([(0.32, 0.5), (0.3, 1.2), (0.26, 1.5), (0.001, 1.52)], MARBLE, seg=12),
         box(0.5, 0.55, 0.3, MARBLE).T(0, 1.45, 0), cyl(0.12, 0.12, 0.25, MARBLE, seg=8).T(0, 1.95, 0),
         lathe([(0.001, 0), (0.13, 0.05), (0.14, 0.18), (0.001, 0.28)], MARBLE, seg=10).T(0, 2.15, 0)]
    P.append(cyl(0.06, 0.05, 0.6, MARBLE, seg=6).R(0, 0, rr.uniform(0.3, 1.2)).T(0.3, 1.5, 0.05))
    m = merge(P)
    crumple(m, 0.02, 3.0, seed)
    m.S(h / 2.4)
    m.c = np.clip(m.c * (0.8 + 0.3 * fbm(m.v, 3.0, seed, 2))[:, None], 0, 1)
    return m


W, S_ = [], []
# the nymphaeum: a hall about 18 × 9 m, the apse at its far end, niches along the side walls, a pool in the middle
L, B = 18.0, 9.0
for s in (-1, 1):
    W.append(ashlar_wall((s * B / 2, -L / 2), (s * B / 2, L / 2), 1.8, 0.9, TUFA, 10 + s, block=0.7, course=0.35, ruin=0.35))
W.append(ashlar_wall((-B / 2, -L / 2), (-1.4, -L / 2), 1.6, 0.9, TUFA, 13, block=0.7, course=0.35, ruin=0.4))
W.append(ashlar_wall((1.4, -L / 2), (B / 2, -L / 2), 1.6, 0.9, TUFA, 14, block=0.7, course=0.35, ruin=0.4))
for k in range(13):                                                    # the apse
    a0, a1 = math.pi * k / 13, math.pi * (k + 1) / 13
    W.append(ashlar_wall((-math.cos(a0) * B / 2, L / 2 + math.sin(a0) * 4.5), (-math.cos(a1) * B / 2, L / 2 + math.sin(a1) * 4.5), 2.0, 0.9, TUFA, 20 + k, block=0.7, course=0.35, ruin=0.3))
pool = merge([box(4.0, 0.5, 9.0, TUFA).T(0, 0, 0)] + [box(3.4, 0.05, 8.4, (0.25, 0.27, 0.27)).T(0, 0.48, 0)])
for k, z in enumerate((-6.0, -2.0, 2.0, 6.0)):                         # copies of the statues in their niches
    for s in (-1, 1):
        S_.append(roman_statue(100 + k * 2 + (s > 0)).R(0, -s * math.pi / 2, 0).T(s * (B / 2 - 0.9), 0.0, z))
for k, x in enumerate((-1.6, 0.0, 1.6)):                               # the group in the apse
    S_.append(roman_statue(120 + k, 2.2).R(0, math.pi, 0).T(x, 0.0, L / 2 + 2.6))
floor = mosaic(B - 1.2, L - 1.0, 41, cols=((0.7, 0.66, 0.58), (0.4, 0.38, 0.35), (0.55, 0.32, 0.24)), border=0.6, lost=0.45)

# the road beside it: big polygonal basalt slabs, kerbs, the walls of rooms on either side with mosaic floors
road = []
for k in range(120):
    x, z = rng.uniform(-2.4, 2.4), rng.uniform(-30, 30)
    s_ = rng.uniform(0.5, 0.9)
    b = box(s_, 0.25, s_ * rng.uniform(0.8, 1.3), BASALT).R(0, rng.uniform(0, 6.28), 0)
    road.append(b.T(18 + x, -0.05, z))
for s in (-1, 1):
    road.append(box(0.3, 0.35, 60, BASALT).T(18 + s * 2.8, -0.05, 0))
rooms, floors = [], []
for k, z in enumerate(np.arange(-26, 26, 8.5)):
    for s in (-1, 1):
        if rng.random() < 0.2:
            continue
        x0 = 18 + s * 3.4; x1 = 18 + s * rng.uniform(8, 11)
        rooms.append(brick_wall((x0, z - 3.8), (x1, z - 3.8), rng.uniform(0.4, 1.3), 0.5, 200 + k * 2 + (s > 0), ruin=0.5))
        rooms.append(brick_wall((x1, z - 3.8), (x1, z + 3.8), rng.uniform(0.4, 1.3), 0.5, 220 + k * 2 + (s > 0), ruin=0.5))
        if rng.random() < 0.5:
            floors.append(mosaic(abs(x1 - x0) - 0.6, 7.0, 300 + k * 2 + (s > 0), lost=0.5).T((x0 + x1) / 2, 0, z))
cols = merge([column(0.28, rng.uniform(1.0, 3.0), MARBLE, 400 + k, fallen=k % 2 == 0, broken=0.2, drums=2).R(0, rng.uniform(0, 6.28), 0).T(rng.uniform(-14, 30), 0, rng.uniform(-28, 28)) for k in range(10)])
stones = merge([rubble(25, x, z, 4, 0.4, TUFA, 500 + k) for k, (x, z) in enumerate(((-9, 0), (8, -12), (26, 14), (10, 20)))])
pots = amphorae(12, 6, -14, 6, 600, CLAY, 0.7)

out = ROOT / "models" / "item" / "Q290097" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("nymphaeum", merge(W + [pool]), "stone"), ("statues", merge(S_), "stone"), ("road", merge(road), "stone"),
                  ("rooms", merge(rooms), "stone"), ("columns", cols, "stone"), ("stones", stones, "stone")],
            [("mosaics", merge([floor] + floors), "artefact"), ("pottery", pots, "artefact")], depth=-6, seed=41, drift=0.08)
