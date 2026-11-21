"""Dean's Blue Hole (Q1181185) — the deep blue hole of Long Island, Bahamas, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, Atlas Obscura, Scuba.com): a blue hole in a sandy bay west
of Clarence Town, Long Island, ringed on one side by a limestone cliff; at 202 m it is one of the deepest known blue
holes with sea water, and the home of free-diving competitions (Vertical Blue). Its mouth is roughly round, 25-35 m
across, in water a few metres deep; at about 20 m it opens out into a great chamber about 100 m across, then drops to
the bottom. The profile of the chamber and the cliff are approximate.

    python model_sources/Q1181185_deans_blue_hole.py   -> models/item/Q1181185/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, grid, crumple, fbm, sstep  # noqa: E402
from placekit import (shaft, stalactites, rubble, dress_place)  # noqa: E402

rng = np.random.default_rng(1181)
ROCK = (0.64, 0.6, 0.52)
SILT = (0.52, 0.5, 0.44)
PROFILE = [(0, 15), (6, 15), (14, 16), (18, 17), (20, 34), (21, 46), (24, 50), (40, 50), (70, 47), (110, 44), (150, 40), (185, 36), (202, 32)]


def radius_at(d):
    for (d0, r0), (d1, r1) in zip(PROFILE, PROFILE[1:]):
        if d0 <= d <= d1:
            return r0 + (r1 - r0) * (d - d0) / (d1 - d0)
    return PROFILE[-1][1]


wall, floor = shaft(PROFILE, ROCK, 1181, seg=72, rough=0.06, floor_col=SILT)
stal = stalactites(80, 20, 24, lambda d: radius_at(d) * 0.98, 1182, (0.68, 0.65, 0.58), length=(0.5, 3.0))
# the limestone cliff on the landward side of the bay, its foot in the water by the hole
def cliff(X, Z):
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    edge = 3.0 * (fbm(P3 * [0, 0, 1], 0.08, 3, 3).reshape(X.shape) - 0.5)
    return 4.5 * sstep(32 + edge, 38 + edge, X) * sstep(60, 45, np.abs(Z)) + 0.4 * (fbm(P3, 0.3, 4, 2).reshape(X.shape) - 0.5) * (X > 30)
rock = grid(26, 56, -62, 62, 40, 120, cliff, ROCK)
rock.f = rock.f[rock.v[rock.f][:, :, 1].max(axis=1) > 0.05]
stones = merge([rubble(25, math.cos(a) * 24, math.sin(a) * 24, 4, 0.6, ROCK, 1190 + k) for k, a in enumerate(np.linspace(0.5, 5.5, 5))])
fallen = rubble(30, 0, 0, 14, 1.5, ROCK, 1199, flat=0.7).T(0, -201, 0)

out = ROOT / "models" / "item" / "Q1181185" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("hole_walls", wall, "rock"), ("stalactites", stal, "rock"), ("cliff", rock, "rock"), ("stones", stones, "rock")],
            [("hole_floor", floor, "rock"), ("fallen_blocks", fallen, "rock")], depth=-3, seed=1181, drift=0.2, hole=15)
