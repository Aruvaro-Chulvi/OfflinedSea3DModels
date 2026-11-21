"""Great Blue Hole (Q86486) — the giant marine sinkhole of Lighthouse Reef, Belize, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the 2018 Aquatica Submarines / Blue Hole Belize expedition
with its sonar survey, GPS World, Southern Fried Science): a round sinkhole about 300 m across and about 125 m deep
in the middle of Lighthouse Reef atoll, some 70 km off the coast of Belize, made famous by Jacques-Yves Cousteau in
1971. It was a limestone cave whose roof collapsed while the sea was lower in the Ice Ages, then drowned: its walls
drop almost sheer from the shallow lagoon to a ledge of caverns at about 40 m with stalactites hanging from their
roofs, then slope down to a floor of silt; deep down the water is without oxygen and a layer of hydrogen sulphide
lies over the bottom. The 2018 submarine dives mapped it with sonar. The profile of the walls, the ledge and the
stalactites are approximate.

    python model_sources/Q86486_great_blue_hole.py   -> models/item/Q86486/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, crumple, fbm, sstep  # noqa: E402
from placekit import (shaft, stalactites, rubble, REEF_ROCK, LIMESTONE, dress_place)  # noqa: E402

rng = np.random.default_rng(1971)
ROCK = (0.6, 0.57, 0.5)
SILT = (0.5, 0.48, 0.42)
PROFILE = [(0, 150), (8, 149), (20, 148), (34, 148), (38, 156), (46, 160), (54, 157), (58, 148), (75, 138), (95, 122), (112, 105), (125, 92)]


def radius_at(d):
    for (d0, r0), (d1, r1) in zip(PROFILE, PROFILE[1:]):
        if d0 <= d <= d1:
            return r0 + (r1 - r0) * (d - d0) / (d1 - d0)
    return PROFILE[-1][1]


wall, floor = shaft(PROFILE, ROCK, 1971, seg=120, rough=0.04, floor_col=SILT)
stal = stalactites(260, 36, 50, radius_at, 1972, (0.66, 0.63, 0.56), length=(1.0, 7.0))
fallen = merge([rubble(40, math.cos(a) * 80, math.sin(a) * 80, 18, 2.5, ROCK, 1980 + k, flat=0.7).T(0, -124, 0) for k, a in enumerate(np.linspace(0, 6.28, 7)[:-1])])
# the rim: the reef flat round the hole, patches of reef rock
rim = merge([rubble(60, math.cos(a) * 175, math.sin(a) * 175, 10, 1.4, REEF_ROCK, 1990 + k, flat=0.6) for k, a in enumerate(np.linspace(0, 6.28, 13)[:-1])])

out = ROOT / "models" / "item" / "Q86486" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("hole_walls", wall, "rock"), ("stalactites", stal, "rock"), ("rim", rim, "rock")],
            [("hole_floor", floor, "rock"), ("fallen_blocks", fallen, "rock")], depth=-5, seed=1971, drift=0.2, hole=150)
