"""Kolumbo (Q1513696) — the hydrothermal field on the crater floor of the submarine volcano off Santorini, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, Kilias et al. 2013 in Scientific Reports, Carey et al.,
NOAA Ocean Explorer): Kolumbo, 7 km north-east of Santorini, is the most active submarine volcano of the Aegean; its
eruption of 1650 killed some seventy people on Santorini with its gases. Its cone is about 3 km wide, with a crater
about 1.5 km across whose rim rises to about 18 m below the surface and whose flat floor lies at about 500 m. On the
crater floor, at about 492-505 m, an extensive hydrothermal field — mapped by E/V Nautilus in 2010 and 2011 — vents
water as hot as 220 °C and has built chimneys of metal sulphides and sulphates up to 4-6 m high (one, «Poet's
Candle», about 6 m), on orange and grey iron-stained sediment, with bubbling carbon dioxide. Here the central part
of the field is drawn, about 200 m across; the place and height of each chimney are approximate.

    python model_sources/Q1513696_kolumbo.py   -> models/item/Q1513696/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, crumple, fbm, sstep  # noqa: E402
from placekit import (rubble, dress_place)  # noqa: E402

rng = np.random.default_rng(1650)
SULPHIDE = np.array([0.36, 0.3, 0.26])
RUST = np.array([0.62, 0.36, 0.18])
GREYS = np.array([0.55, 0.53, 0.5])


def chimney(h, r, seed):
    """A sulphide chimney: a knobbly, tapering spire, rusty orange where it has oxidised, grey-black at the top."""
    rr = np.random.default_rng(seed)
    n = max(5, int(h / 0.4))
    prof = [(max(r * (1 - 0.75 * (k / n) ** 0.8) * rr.uniform(0.85, 1.15), 0.06), h * k / n) for k in range(n + 1)] + [(0.04, h + 0.1)]
    m = lathe(prof, tuple(SULPHIDE), seg=10)
    crumple(m, r * 0.2, 1.5 / max(r, 0.3), seed)
    t = fbm(m.v, 0.8, seed + 3)
    k = sstep(0.4, 0.8, t)[:, None]
    m.c = np.clip(SULPHIDE[None] * (1 - k) + RUST[None] * k, 0, 1) * (0.85 + 0.3 * fbm(m.v, 2.0, seed + 4)[:, None])
    top = m.v[:, 1] > h * 0.85
    m.c[top] = SULPHIDE * 0.5
    return m


def mound(X, Z):
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    hills = sum(a * np.exp(-((X - x) ** 2 + (Z - z) ** 2) / (2 * s ** 2)) for x, z, s, a in ((0, 0, 18, 3.0), (-50, 30, 14, 2.0), (45, -35, 12, 1.6), (30, 50, 10, 1.2)))
    return hills + 0.5 * (fbm(P3, 0.06, 3, 3).reshape(X.shape) - 0.5)


ground = grid(-100, 100, -100, 100, 100, 100, mound, tuple(GREYS))
t = fbm(ground.v, 0.03, 5)
k = sstep(0.45, 0.7, t)[:, None]
ground.c = np.clip(GREYS[None] * (1 - k) + RUST[None] * k, 0, 1) * (0.85 + 0.25 * fbm(ground.v, 0.4, 6)[:, None])   # iron-stained patches
ground.f = ground.f[ground.v[ground.f][:, :, 1].max(axis=1) > 0.05]


def ground_y(x, z):
    return float(mound(np.array([[x]]), np.array([[z]]))[0, 0])


C = [chimney(6.0, 0.9, 1).T(0, ground_y(0, 0) - 0.1, 0)]                  # «Poet's Candle»
for k in range(60):
    cx, cz, s = [(0, 0, 25), (-50, 30, 18), (45, -35, 15), (30, 50, 12)][k % 4]
    x, z = cx + rng.normal(0, s), cz + rng.normal(0, s)
    C.append(chimney(rng.uniform(0.6, 4.5), rng.uniform(0.25, 0.7), 10 + k).T(x, ground_y(x, z) - 0.1, z))
chimneys = merge(C)
talus = merge([rubble(30, x, z, 12, 0.8, tuple(SULPHIDE), 200 + j, flat=0.6) for j, (x, z) in enumerate(((0, 0), (-50, 30), (45, -35)))])
talus.v[:, 1] += np.array([ground_y(x, z) for x, z in talus.v[:, [0, 2]]]) * 0.9

out = ROOT / "models" / "item" / "Q1513696" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("chimneys", chimneys, "rock"), ("vent_mounds", ground, "rock"), ("talus", talus, "rock")], [], depth=-500, seed=1650, drift=0.2)
