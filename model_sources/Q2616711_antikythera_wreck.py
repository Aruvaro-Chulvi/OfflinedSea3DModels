"""Antikythera wreck (Q2616711) — the Roman-era shipwreck off Antikythera that gave the Antikythera mechanism, built by
code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a large merchant ship of the first half of the 1st century BC, carrying a cargo of luxury goods from the
eastern Mediterranean towards Rome: bronze and marble statues, glassware, fine pottery, amphorae of wine, jewellery,
coins and the Antikythera mechanism, the astronomical calculator.

The site, after the published descriptions (Wikipedia, the Swiss School of Archaeology in Greece and the University
of Geneva, Smithsonian, Live Science, the Maritime Executive): found by sponge divers in 1900, it was salvaged in
1900-1901 (the Antikythera Youth, the philosopher's head, the mechanism), visited by Cousteau in 1953 and 1976, and
since 2014 excavated by Greek and international teams. It lies on a steep slope of the island's east coast at some
45-55 m: a field of broken amphorae and pottery, lead anchor stocks, bronze nails and fragments of the hull, partly
covered by large boulders fallen from the cliff (up to 8.5 tonnes, some moved with lift bags in 2022 to reach parts of
the wreck never seen before), with marble statues still turning up — in 2022 the head that belongs to the headless
Herakles in the National Archaeological Museum. A part of the hull was raised in 2025. The model sets the site on
flat ground at about 50 m: the slope, the layout and the heading (not published here: 0) are approximate.

    python model_sources/Q2616711_antikythera_wreck.py   -> models/item/Q2616711/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, crumple, fbm  # noqa: E402
import abyss  # noqa: E402

rng = np.random.default_rng(65)
MARBLE = (0.74, 0.72, 0.66)
TIMBER = (0.26, 0.2, 0.14)
LEAD = (0.38, 0.38, 0.4)
CLAY = (0.6, 0.42, 0.3)
BRONZE_SEA = (0.27, 0.32, 0.26)
ROCK = (0.5, 0.47, 0.42)


def amphora(col=CLAY):
    prof = [(0.0, 0.0), (0.05, 0.02), (0.14, 0.25), (0.19, 0.55), (0.16, 0.8), (0.06, 0.95), (0.05, 1.12), (0.07, 1.15)]
    return lathe(prof, col, seg=10)


def statue_piece(seed):
    """A fallen marble or bronze figure, worn by the sea: a torso with the stumps of the limbs, or a plinth with feet."""
    rr = np.random.default_rng(seed)
    if rr.random() < 0.5:
        P = [lathe([(0.001, 0), (0.22, 0.05), (0.3, 0.35), (0.26, 0.7), (0.18, 0.85), (0.001, 0.9)], MARBLE, seg=12)]
        P += [cyl(0.09, 0.07, 0.45, MARBLE, seg=8).R(0.0, 0.0, s * 1.2).T(s * 0.28, 0.7, 0) for s in (-1, 1)]
        P += [cyl(0.12, 0.1, 0.5, MARBLE, seg=8).R(0, 0, s * 0.15).T(s * 0.12, -0.45, 0) for s in (-1, 1)]
        m = merge(P).R(math.pi / 2, rr.uniform(0, 6.28), 0)
    else:
        m = merge([box(0.9, 0.25, 0.6, MARBLE)] + [box(0.14, 0.4, 0.3, MARBLE).T(s * 0.18, 0.25, 0.05) for s in (-1, 1)])
    crumple(m, 0.04, 2.0, seed)
    m.c = np.clip(m.c * (0.78 + 0.3 * fbm(m.v, 2.5, seed, 2)[:, None]), 0, 1)
    return m


def boulder(r, seed):
    b = abyss.blob(r, ROCK, seed, sub=2, amp=0.35)
    b.v[:, 1] *= 0.75
    return b


def mound(X, Z):
    d = np.sqrt((X / 7.0) ** 2 + (Z / 18.0) ** 2)
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    return np.maximum(0.0, 0.7 * (1 - d ** 2)) + 0.15 * (fbm(P3, 0.5, 3, 3).reshape(X.shape) - 0.5) - 0.05


hull_mound = grid(-10, 10, -22, 22, 40, 88, mound, (0.58, 0.54, 0.46))
hull_mound.f = hull_mound.f[hull_mound.v[hull_mound.f][:, :, 1].max(axis=1) > 0.02]
T_ = []
for s in (-1, 1):                                                           # hull planking and frames showing at the edges
    for z in np.arange(-14, 14, 1.4):
        if rng.random() < 0.5:
            continue
        x = s * 6.6 * math.sqrt(max(0.0, 1 - (z / 18) ** 2)) + rng.normal(0, 0.2)
        t = box(0.22, 0.18, rng.uniform(0.6, 1.6), TIMBER).R(rng.uniform(-0.3, 0.3), 0, 0)
        crumple(t, 0.04, 1.0, int(z * 10 + s * 100))
        T_.append(t.T(x, 0.05, z))
O = []
for k in range(70):                                                         # amphorae, many broken, spilled down the slope
    a = amphora(); a.R(math.pi / 2 + rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28), 0)
    fc = a.v[a.f].mean(1)
    if k % 2 == 0:
        a.f = a.f[fc[:, 1] < rng.uniform(-0.05, 0.12)]
    O.append(a.T(rng.normal(0, 6), 0.14, rng.normal(0, 14) - 4))
for k in range(5):                                                          # lead anchor stocks
    O.append(box(0.16, 0.16, rng.uniform(1.4, 2.2), LEAD).R(0, rng.uniform(0, 6.28), 0).T(rng.normal(0, 5), 0.06, rng.uniform(14, 24)))
for k in range(30):                                                         # fine pottery and glass bowls
    O.append(lathe([(0.001, 0), (0.08, 0.0), (0.12, 0.05), (0.14, 0.09)], (0.55, 0.3, 0.2) if k % 3 else (0.4, 0.55, 0.5), seg=10)
             .R(rng.uniform(-1.2, 1.2), 0, 0).T(rng.normal(0, 4), 0.02, rng.normal(0, 10)))
statues = merge([statue_piece(100 + k).T(rng.normal(0, 5), 0.2, rng.normal(0, 12)) for k in range(6)]
                + [merge([box(0.5, 0.06, 0.12, BRONZE_SEA).R(0, rng.uniform(0, 6.28), 0).T(rng.normal(0, 4), 0.02, rng.normal(0, 8)) for _ in range(12)])])
rocks = merge([boulder(rng.uniform(0.8, 1.6), 200 + k).T(rng.normal(-2, 6), 0.3, rng.normal(0, 12)) for k in range(14)])
others = merge(O + T_)

from dresskit import dress_wreck  # noqa: E402
out = ROOT / "models" / "item" / "Q2616711" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_mound", hull_mound), ("finds", others, "loose")], "mud": [],
                   "plain": [("statues", statues, "artefact"), ("boulders", rocks, "rock")]}],
            depth=-50, kind="galleon", seed=65, wood=True, drift=0.3)
