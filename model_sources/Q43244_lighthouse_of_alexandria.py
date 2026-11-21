"""Lighthouse of Alexandria (Q43244) — the underwater remains off Fort Qaitbay, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the Centre d'Études Alexandrines (CEAlex) of Jean-Yves
Empereur, the French Ministry of Culture, Archaeology magazine): the Pharos, one of the Seven Wonders, stood on the
tip of Pharos island from the 3rd century BC until earthquakes brought it down in the 14th century; Fort Qaitbay was
built on its site in 1477. In 1994 CEAlex began mapping the sea bed east of the fort, a few metres down: more than
3,300 objects over about 1.6 hectares — huge granite blocks, some of them weighing tens of tonnes, column shafts, capitals,
about 25-30 sphinxes, obelisk fragments, statues of Ptolemaic kings and queens as gods — the remains of the
lighthouse and of monuments thrown into the sea, many of them in a jumble against the breakwater. Some pieces have
been raised (a colossus of a Ptolemy, sphinxes; in 2025, 22 blocks of the lighthouse's doorway), and the rest has
been recorded block by block. Here the field is drawn about 140 by 100 m; the layout and the depth are approximate.

    python model_sources/Q43244_lighthouse_of_alexandria.py   -> models/item/Q43244/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, crumple, fbm  # noqa: E402
from placekit import (rubble, column, capital, sphinx, colossus, GRANITE_RED, GRANITE_GREY, LIMESTONE, MARBLE, dress_place)  # noqa: E402

rng = np.random.default_rng(1994)


B = []
for k in range(70):                                                     # the great granite blocks of the tower, fallen in a jumble
    ln = rng.uniform(3, 11); w = rng.uniform(1.2, 2.4); h = rng.uniform(1.0, 2.0)
    b = box(w, h, ln, GRANITE_RED if k % 3 else GRANITE_GREY)
    crumple(b, 0.12, 0.5, 10 + k)
    x, z = rng.normal(-20, 18), rng.normal(0, 25)
    B.append(b.R(rng.uniform(-0.25, 0.25), rng.uniform(0, 6.28), rng.uniform(-0.25, 0.25)).T(x, -0.2 + rng.uniform(0, 1.2) * (abs(x + 20) < 10), z))
blocks = merge(B + [rubble(80, -10, 0, 25, 1.0, LIMESTONE, 99, flat=0.7)])
cols = merge([column(rng.uniform(0.35, 0.6), rng.uniform(3, 7), GRANITE_GREY if k % 2 else GRANITE_RED, 200 + k, fallen=True, drums=1)
              .R(0, rng.uniform(0, 6.28), 0).T(rng.normal(10, 25), 0, rng.normal(0, 35)) for k in range(30)])
caps = merge([capital(MARBLE, 1.2).R(rng.uniform(-1, 1), rng.uniform(0, 6.28), rng.uniform(-1.4, 1.4)).T(rng.normal(10, 22), 0.3, rng.normal(0, 30)) for _ in range(8)])
sphx = merge([sphinx(rng.uniform(1.6, 3.2), GRANITE_GREY if k % 2 else GRANITE_RED, 300 + k).R(rng.uniform(-0.8, 0.8), rng.uniform(0, 6.28), rng.uniform(-1.5, 1.5) * (k % 3 == 0))
              .T(rng.normal(20, 22), 0.2, rng.normal(0, 30)) for k in range(14)])
obel = merge([box(1.3, 1.3, rng.uniform(3, 6), GRANITE_RED).R(rng.uniform(-0.2, 0.2), rng.uniform(0, 6.28), rng.uniform(-0.3, 0.3)).T(rng.normal(24, 10), 0.4, rng.normal(10, 20)) for _ in range(4)])
statues = merge([colossus(rng.uniform(4, 7), GRANITE_RED, 400 + k, fallen=True, broken=(("head",) if k % 2 else ("legs",))).R(0, rng.uniform(0, 6.28), 0).T(rng.normal(-5, 15), 0, rng.normal(0, 25)) for k in range(3)])

out = ROOT / "models" / "item" / "Q43244" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("blocks", blocks, "stone"), ("columns", merge([cols, caps, obel]), "stone"), ("sphinxes", sphx, "stone"), ("statues", statues, "stone")],
            [], depth=-8, seed=1994, drift=0.2)
