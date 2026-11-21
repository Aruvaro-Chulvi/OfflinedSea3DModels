"""Canopus (Q1145180) — the sunken city at the mouth of the Canopic Nile in Aboukir Bay, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the European Institute for Underwater Archaeology of Franck
Goddio, Archaeology magazine, The Art Newspaper): a city famous in antiquity for its temple of Serapis and its
festivals, west of Thonis-Heracleion; like it, it slid into the sea after earthquakes and the liquefaction of the
Nile clay. Goddio's team identified its submerged eastern part in 1999, in shallow, murky water a little off Abu Qir:
the foundations of a great building almost certainly the Serapeum — a masonry wall about 103 m long and 3 m wide —
with columns, capitals and the fragments of statues around it. The great pieces (a marble head of Serapis, sphinxes)
have been raised; in August 2025 the Egyptian authorities lifted more statues from the bay. Here the Serapeum area is
drawn, about 140 m across; the layout, the sizes and the depth are approximate.

    python model_sources/Q1145180_canopus.py   -> models/item/Q1145180/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, fbm  # noqa: E402
from placekit import (ashlar_wall, rubble, column, capital, column_base, amphorae, stone_anchor,
                      LIMESTONE, GRANITE_RED, GRANITE_GREY, MARBLE, CLAY, dress_place)  # noqa: E402

rng = np.random.default_rng(1999)


W = [ashlar_wall((-6, -51.5), (-6, 51.5), 1.4, 3.0, LIMESTONE, 1, block=1.2, course=0.6, ruin=0.5, fallen=0.4)]   # the long Serapeum wall
for k, z in enumerate((-40, -10, 22, 46)):                             # cross walls of the building, lower
    W.append(ashlar_wall((-6, z), (rng.uniform(14, 22), z), 0.9, 1.6, LIMESTONE, 10 + k, block=1.1, course=0.5, ruin=0.65))
cols = merge([column(0.4, rng.uniform(2.5, 4.5), GRANITE_GREY if k % 3 else GRANITE_RED, 100 + k, fallen=k % 4 != 0, broken=0.5, drums=2)
              .R(0, rng.uniform(0, 6.28), 0).T(rng.uniform(-2, 24), 0, rng.uniform(-50, 50)) for k in range(16)])
parts = merge([capital(MARBLE, 1.1).R(rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28), rng.uniform(-1.4, 1.4)).T(rng.uniform(-10, 26), 0.3, rng.uniform(-48, 48)) for _ in range(5)]
              + [column_base(MARBLE, 1.1).T(rng.uniform(-10, 26), 0.0, rng.uniform(-48, 48)) for _ in range(5)])
blocks = merge([rubble(35, x, z, 7, 1.0, LIMESTONE, 200 + k, flat=0.7) for k, (x, z) in enumerate(((6, -30), (12, 8), (4, 36), (-14, 0)))])
anchors = merge([stone_anchor(rng.uniform(0.6, 1.0), LIMESTONE, 300 + k).R(0, rng.uniform(0, 6.28), 0).T(rng.uniform(-60, 60), 0.05, rng.uniform(-60, 60)) for k in range(18)])
pots = merge([amphorae(14, x, z, 7, 400 + k, CLAY, 0.6) for k, (x, z) in enumerate(((-30, -30), (30, 30), (-26, 40)))])

out = ROOT / "models" / "item" / "Q1145180" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("serapeum", merge(W), "stone"), ("columns", cols, "stone"), ("blocks", blocks, "stone"), ("capitals", parts, "stone")],
            [("anchors", anchors, "stone"), ("pottery", pots, "artefact")], depth=-6, seed=1999, drift=0.25)
