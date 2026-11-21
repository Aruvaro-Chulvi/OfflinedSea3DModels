"""Atlit Yam (Q2896085) — the drowned Neolithic village off the Carmel coast, Israel, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, Galili et al. in the Journal of Field Archaeology, the Tel
Aviv University, Atlas Obscura): a village of farmers, herders and fishermen of about 6900-6300 BC, drowned by the
rising sea. Ehud Galili found it in 1984 while surveying for shipwrecks; it lies 200-400 m offshore at 8-11 m and
covers about 40,000 m², usually under sand that winter storms sometimes strip away. Its remains: the stone
foundations of rectangular houses, stone-lined wells (one about 5.5 m deep), hearths, burials, and a semicircle of
seven megaliths of about 600 kg each, some with cup marks, set round a freshwater spring. Here the heart of the
village is drawn, about 90 m across; the layout is approximate.

    python model_sources/Q2896085_atlit_yam.py   -> models/item/Q2896085/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, fbm  # noqa: E402
from placekit import (ashlar_wall, rubble, megalith, well, dress_place)  # noqa: E402

rng = np.random.default_rng(6900)
STONE = (0.58, 0.54, 0.46)


# the semicircle of seven megaliths round the spring
M = []
for k in range(7):
    a = math.pi * (0.1 + 0.8 * k / 6)
    M.append(megalith(rng.uniform(1.0, 1.4), rng.uniform(0.6, 0.8), (0.52, 0.48, 0.42), 10 + k).R(rng.uniform(-0.08, 0.08), -a, rng.uniform(-0.08, 0.08)).T(math.cos(a) * 2.6, 0, math.sin(a) * 2.6 - 18))
spring = cyl(1.2, 1.2, 0.03, (0.08, 0.08, 0.07), seg=14).T(0, 0.01, -18)
W = []
for k, (x, z, w, d) in enumerate(((-24, 6, 7, 5), (-14, 18, 6, 4.5), (8, 14, 8, 5), (20, 0, 6, 5), (-8, 30, 7, 5), (16, 28, 6, 4.5), (-28, -8, 5, 4))):
    yaw = rng.uniform(-0.3, 0.3)
    c, s = math.cos(yaw), math.sin(yaw)
    P = lambda u, v: (x + u * c + v * s, z - u * s + v * c)
    co = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    for j in range(4):
        W.append(ashlar_wall(P(*co[j]), P(*co[(j + 1) % 4]), rng.uniform(0.3, 0.6), 0.55, STONE, 100 + k * 4 + j, block=0.5, course=0.25, ruin=0.5))
walls = merge([m for m in W if len(m.f)])
wells = merge([well(0.8, 0.6, STONE, 200).T(-2, 0, 4), well(0.7, 0.5, STONE, 201).T(-30, 0, 20)])
hearths = merge([merge([box(0.25, 0.15, 0.25, (0.3, 0.28, 0.25)).T(math.cos(a) * 0.6 + x, 0, math.sin(a) * 0.6 + z) for a in np.linspace(0, 6.28, 9)[:-1]]) for x, z in ((8, 14), (-14, 18), (20, 0))])
stones = merge([rubble(25, x, z, 6, 0.35, STONE, 300 + k) for k, (x, z) in enumerate(((-10, 0), (12, 22), (-20, 28)))])

out = ROOT / "models" / "item" / "Q2896085" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("megaliths", merge(M), "rock"), ("houses", walls, "stone"), ("wells", wells, "stone"), ("hearths", hearths, "stone"), ("stones", stones, "stone")],
            [("spring", spring, "dark")], depth=-10, seed=6900, drift=0.12)
