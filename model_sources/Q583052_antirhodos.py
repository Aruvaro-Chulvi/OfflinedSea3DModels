"""Antirhodos (Q583052) — the sunken royal island in the eastern harbour of Alexandria, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the European Institute for Underwater Archaeology of Franck
Goddio, the Oxford Centre for Maritime Archaeology, Discovery): a small island in the Portus Magnus, the great
eastern harbour of Ptolemaic Alexandria, with a palace and a temple of Isis; it sank with the royal quarter of the
city after earthquakes and tsunamis from the 4th century AD. Goddio's team has surveyed and excavated it since 1996:
the island's outline, its jetties, a palace of modest size, about 90 by 30 m, with a paved floor, granite columns,
and the remains of the temple of Isis. The statues found here — two sphinxes, one probably of Ptolemy XII, Cleopatra's
father, a statue of a priest of Isis, a colossal head thought to be of Caesarion — have been lifted for study and
exhibition. Here the palace area is drawn, about 120 m across; the layout, the sizes and the depth are approximate.

    python model_sources/Q583052_antirhodos.py   -> models/item/Q583052/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, fbm  # noqa: E402
from placekit import (ashlar_wall, rubble, column, column_base, amphorae, mosaic,
                      LIMESTONE, GRANITE_RED, GRANITE_GREY, MARBLE, CLAY, dress_place)  # noqa: E402

rng = np.random.default_rng(1996)


W = []
for (a, b) in (((-15, -45), (15, -45)), ((15, -45), (15, 45)), ((15, 45), (-15, 45)), ((-15, 45), (-15, -45))):
    W.append(ashlar_wall(a, b, 1.0, 1.4, LIMESTONE, int(abs(a[0] * 3 + a[1])) + 1, block=1.1, course=0.5, ruin=0.6, fallen=0.4))
pave = mosaic(28, 88, 1996, cols=((0.72, 0.7, 0.64), (0.62, 0.6, 0.55), (0.55, 0.5, 0.45)), border=1.0, lost=0.55)   # the marble paving, broken up
cols = merge([column(0.5, rng.uniform(3.0, 5.5), GRANITE_RED if k % 2 else GRANITE_GREY, 10 + k, fallen=k % 3 != 0, broken=0.4, drums=1)
              .R(0, rng.uniform(0, 6.28), 0).T(rng.uniform(-14, 14), 0, rng.uniform(-44, 44)) for k in range(14)])
bases = merge([column_base(GRANITE_GREY, 1.3).T(x, 0, z) for x in (-9, 9) for z in np.arange(-36, 40, 12)])
# the jetty of the island: big blocks in a line out from the palace, and the island's edge of fallen blocks
jetty = merge([ashlar_wall((15, 10), (55, 18), 1.2, 4.0, LIMESTONE, 50, block=1.6, course=0.6, ruin=0.4)])
edge = merge([rubble(40, x, z, 8, 1.2, LIMESTONE, 60 + k, flat=0.7) for k, (x, z) in enumerate(((-30, -20), (-28, 30), (30, -40), (32, 46)))])
isis = merge([ashlar_wall((-40, -10), (-28, -10), 0.8, 1.2, LIMESTONE, 70, block=1.0, course=0.5, ruin=0.6),
              ashlar_wall((-40, -10), (-40, 4), 0.8, 1.2, LIMESTONE, 71, block=1.0, course=0.5, ruin=0.6),
              rubble(20, -34, -2, 5, 0.9, GRANITE_RED, 72)])
pots = amphorae(16, 20, -20, 10, 80, CLAY, 0.6)

out = ROOT / "models" / "item" / "Q583052" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("palace", merge(W), "stone"), ("columns", merge([cols, bases]), "stone"), ("jetty", jetty, "stone"),
                  ("blocks", edge, "stone"), ("temple_of_isis", isis, "stone")],
            [("paving", pave, "stone"), ("pottery", pots, "artefact")], depth=-6, seed=1996, drift=0.2)
