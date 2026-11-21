"""Kick 'em Jenny (Q1740807) — the summit crater of the active submarine volcano north of Grenada, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the Smithsonian Global Volcanism Program via VolcanoDiscovery,
Allen et al. 2018 in Geochemistry, Geophysics, Geosystems, Imperial College London): the most active volcano of the
Lesser Antilles, a submarine cone some 8 km north of Grenada that has erupted more than a dozen times since its
discovery in 1939, most recently in 2017 — boats are kept out of an exclusion zone around it. Its summit is about
185 m below the surface: an active cone about 1 km across with an almost round crater about 330 m wide and up to 80 m
deep, its floor and walls venting gas and warm water. The R.R.S. James Cook surveyed it in 2016 and, right after the
eruption, in April 2017: rather than a growing cone, the surveys showed small landslides from its flanks. The shape
of the crater is approximate.

    python model_sources/Q1740807_kick_em_jenny.py   -> models/item/Q1740807/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, grid, lathe, crumple, fbm, sstep  # noqa: E402
from placekit import (shaft, rubble, dress_place)  # noqa: E402

rng = np.random.default_rng(1939)
BASALT_ = (0.3, 0.28, 0.26)
ASH = (0.44, 0.4, 0.34)
STAIN = np.array([0.6, 0.42, 0.2])
PROFILE = [(0, 165), (12, 150), (30, 128), (50, 108), (68, 92), (78, 84), (80, 80)]


wall, floor = shaft(PROFILE, BASALT_, 1939, seg=96, rough=0.05, floor_col=ASH)
# the crater wall is a slope of scree and lava blocks, stained yellow-brown where the vents seep
for m in (wall, floor):
    k = sstep(0.55, 0.75, fbm(m.v, 0.04, 7))[:, None]
    m.c = np.clip(m.c * (1 - k) + STAIN[None] * k, 0, 1)
# the rim: a ridge rising round the crater and falling away outwards (the cone's flanks beyond are the seabed's)
def rim(X, Z):
    R = np.hypot(X, Z)
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    return (6.0 * sstep(250, 175, R) * sstep(160, 175, R) + 1.5 * (fbm(P3, 0.05, 3, 3).reshape(X.shape) - 0.5)) * (R > 163)
ridge = grid(-260, 260, -260, 260, 110, 110, rim, BASALT_)
ridge.f = ridge.f[(ridge.v[ridge.f][:, :, 1].max(axis=1) > 0.05) & (np.hypot(ridge.v[ridge.f][:, :, 0].mean(1), ridge.v[ridge.f][:, :, 2].mean(1)) > 164)]
blocks = merge([rubble(40, math.cos(a) * 110, math.sin(a) * 110, 30, 2.5, BASALT_, 1950 + k, flat=0.7).T(0, -66, 0) for k, a in enumerate(np.linspace(0, 6.28, 6)[:-1])]
               + [rubble(40, 0, 0, 40, 1.5, ASH, 1960, flat=0.6).T(0, -79, 0)])
crest = merge([rubble(50, math.cos(a) * 185, math.sin(a) * 185, 12, 2.0, BASALT_, 1970 + k, flat=0.7) for k, a in enumerate(np.linspace(0, 6.28, 9)[:-1])])

out = ROOT / "models" / "item" / "Q1740807" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("crater_walls", wall, "rock"), ("crater_rim", ridge, "rock"), ("rim_blocks", crest, "rock")],
            [("crater_floor", floor, "rock"), ("scree", blocks, "rock")], depth=-185, seed=1939, drift=0.15, hole=164)
