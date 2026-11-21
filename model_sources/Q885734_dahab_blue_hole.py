"""Blue Hole of Dahab (Q885734) — the blue hole in the reef north of Dahab, Sinai, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, Scuba.com, the Dive Site Directory): a round sinkhole in the
fringing reef north of Dahab, some 50-60 m across and just over 100 m deep, open to the sea through a shallow notch
in its rim, «the saddle», at about 6-7 m, and through «the Arch», a tunnel about 26 m long whose roof is at about
52-56 m and which opens onto the reef wall falling away to the deep sea. Famous among free-divers and notorious for
the many technical divers who have died trying to swim through the Arch. Here the reef rim is drawn standing a few
metres above the saddle, the hole and the mouth of the Arch; the tunnel ends in the model at the outer wall of the
hole, since the seabed beyond it is not drawn. The profile and the sizes are approximate.

    python model_sources/Q885734_dahab_blue_hole.py   -> models/item/Q885734/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, grid, lathe, crumple, fbm, sstep  # noqa: E402
from placekit import (shaft, rubble, REEF_ROCK, dress_place)  # noqa: E402

rng = np.random.default_rng(885)
ROCK = (0.6, 0.56, 0.48)
SILT = (0.62, 0.58, 0.5)
PROFILE = [(0, 27), (10, 27), (30, 26), (50, 26), (60, 24), (80, 21), (100, 17), (110, 14)]


wall, floor = shaft(PROFILE, ROCK, 885, seg=64, rough=0.07, floor_col=SILT)
# the Arch: an opening in the wall on the seaward side (+x), its roof at about 52-56 m, a short tunnel into the dark
fc = wall.v[wall.f].mean(1)
ang = np.arctan2(fc[:, 2], fc[:, 0])
arch = (np.abs(ang) < 0.32) & (fc[:, 1] < -54) & (fc[:, 1] > -75)
wall.f = wall.f[~arch]
tunnel = lathe([(9.0, 0.0), (9.5, 10.0), (8.5, 20.0), (0.001, 26.0)], (0.3, 0.29, 0.26), seg=16).R(0, 0, -math.pi / 2)
tunnel.v[:, 1] *= 1.0
tunnel = tunnel.T(26, -64, 0)
# the reef rim, standing a few metres above the saddle all round except on the seaward side
def rim(X, Z):
    R = np.hypot(X, Z); a = np.arctan2(Z, X)
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    crest = 4.0 * np.exp(-((R - 36) / 7) ** 2) * (1 - 0.9 * np.exp(-(a / 0.35) ** 2))     # the saddle on the +x side
    return (crest + 0.8 * (fbm(P3, 0.15, 3, 3).reshape(X.shape) - 0.5)) * (R > 28)
reef = grid(-60, 60, -60, 60, 80, 80, rim, REEF_ROCK)
reef.f = reef.f[(reef.v[reef.f][:, :, 1].max(axis=1) > 0.05) & (np.hypot(reef.v[reef.f][:, :, 0].mean(1), reef.v[reef.f][:, :, 2].mean(1)) > 27.5)]
stones = rubble(40, 0, 0, 10, 1.2, ROCK, 886, flat=0.7).T(0, -109, 0)

out = ROOT / "models" / "item" / "Q885734" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("hole_walls", wall, "rock"), ("reef_rim", reef, "rock")],
            [("hole_floor", floor, "rock"), ("the_arch", tunnel, "dark"), ("fallen_blocks", stones, "rock")], depth=-6, seed=885, drift=0.15, hole=27)
