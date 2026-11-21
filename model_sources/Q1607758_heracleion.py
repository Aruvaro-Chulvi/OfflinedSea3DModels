"""Heracleion (Q1607758) — Thonis-Heracleion, the sunken port of Egypt in Aboukir Bay, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the European Institute for Underwater Archaeology of Franck
Goddio, Atlas Obscura, Divernet, The National): Egypt's great port on the Mediterranean before Alexandria, at the
mouth of the Canopic branch of the Nile, with the temple of Amun-Gereb where pharaohs were confirmed in their power.
Earthquakes, floods and the liquefaction of the clay it stood on brought it down into the sea from the 2nd century
BC; Goddio's team found it in 2000 some 6.5 km off the coast, under about 10 m of water and silt. The temple
collapsed into a canal beside it; around it lie granite and limestone blocks, the fallen pieces of colossal statues,
naoi and stelae, more than 700 ancient anchors and the remains of some 70 ships. The great finds — the 5.4 m statue
of Hapy, the colossi of a king and a queen, the stela of Thonis-Heracleion — have been raised and are in museums;
the excavation goes on (in 2023 the temples of Amun and of Aphrodite). Here the temple area is drawn, about 140 m
across; its layout is approximate.

    python model_sources/Q1607758_heracleion.py   -> models/item/Q1607758/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, fbm, crumple  # noqa: E402
from placekit import (ashlar_wall, rubble, column, colossus, stela, amphorae, stone_anchor, timbers,
                      LIMESTONE, GRANITE_RED, GRANITE_GREY, CLAY, TIMBER, dress_place)  # noqa: E402

rng = np.random.default_rng(2000)


W = []
# the temple of Amun-Gereb: its walls down to their lowest courses, collapsed towards the canal (+x)
for (a, b) in (((-20, -30), (-20, 30)), ((-20, 30), (14, 30)), ((-20, -30), (14, -30)), ((-6, -30), (-6, 30))):
    W.append(ashlar_wall(a, b, 1.6, 2.0, LIMESTONE, int(a[0] * 7 + b[1]) % 997, block=1.4, course=0.7, ruin=0.6, fallen=0.5))
blocks = merge([rubble(40, x, z, 8, 1.3, LIMESTONE, 10 + k, flat=0.7) for k, (x, z) in enumerate(((18, -10), (24, 12), (8, 0), (-14, 18)))]
               + [rubble(15, 22, 0, 10, 1.6, GRANITE_RED, 30, flat=0.8)])
# the canal the temple fell into: a long hollow of darker silt with blocks lying in it
canal = grid(16, 40, -60, 60, 12, 60, lambda X, Z: -0.4 * np.exp(-((X - 28) / 7) ** 2), (0.38, 0.35, 0.3))
# the bases and broken pieces of the colossi (the statues themselves were raised), a naos, stelae
statues = merge([colossus(5.4, GRANITE_RED, 41, fallen=True, broken=("head",)).R(0, 0.6, 0).T(10, 0, 24),
                 box(2.2, 1.0, 2.2, GRANITE_RED).T(4, 0, 34), box(2.2, 1.0, 2.2, GRANITE_RED).T(-4, 0, 34),
                 box(1.8, 2.6, 1.6, GRANITE_GREY).R(0.8, 0.3, 0).T(-10, 0.6, 10),
                 stela(2.2, GRANITE_GREY, 44).R(-1.45, 0.4, 0).T(0, 0.3, -36), stela(1.6, GRANITE_GREY, 45).R(-1.5, 2.0, 0).T(-24, 0.25, 6)])
cols = merge([column(0.45, rng.uniform(2.0, 4.5), GRANITE_GREY, 60 + k, fallen=True, drums=1).R(0, rng.uniform(0, 6.28), 0).T(rng.uniform(-24, 30), 0, rng.uniform(-36, 36)) for k in range(8)])
# anchors and wrecks of the port: stone anchors everywhere, the low mounds of buried hulls with their frames showing
anchors = merge([stone_anchor(rng.uniform(0.6, 1.2), LIMESTONE, 100 + k).R(rng.uniform(-0.2, 0.2), rng.uniform(0, 6.28), 0).T(rng.uniform(-70, 70), 0.06, rng.uniform(-70, 70)) for k in range(60)])
hulls = []
for k, (x, z, yaw) in enumerate(((-50, -40, 0.3), (48, 44, -0.8), (-46, 52, 1.9))):
    ln = rng.uniform(18, 28)
    m = grid(-3.5, 3.5, -ln / 2, ln / 2, 10, 24, lambda X, Z: np.maximum(0, 0.6 * (1 - (X / 3.5) ** 2 - (2 * Z / ln) ** 2)), (0.36, 0.32, 0.26))
    hulls.append(merge([m, timbers(14, 0, 0, 2.5, 200 + k, TIMBER, (0.8, 2.5))]).R(0, yaw, 0).T(x, 0, z))
pots = merge([amphorae(16, x, z, 8, 300 + k, CLAY, 0.6) for k, (x, z) in enumerate(((30, -40), (-40, 0), (0, 50)))])

out = ROOT / "models" / "item" / "Q1607758" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("temple", merge(W), "stone"), ("blocks", blocks, "stone"), ("statues", statues, "stone"), ("columns", cols, "stone"),
                  ("wrecks", merge(hulls), "wood"), ("canal", canal, "rock")],
            [("anchors", anchors, "stone"), ("pottery", pots, "artefact")], depth=-10, seed=2000, drift=0.25)
