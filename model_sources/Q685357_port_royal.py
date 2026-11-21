"""Port Royal (Q685357) — the part of the 17th-century port town that sank in the 1692 earthquake, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the Institute of Nautical Archaeology and Texas A&M
University, the Jamaica National Heritage Trust, ICOMOS): at the end of the Palisadoes spit guarding Kingston
Harbour, Port Royal was the busiest English port in the Caribbean and a haven of privateers and pirates. On 7 June
1692 an earthquake liquefied the sand it was built on and about two thirds of the town slid and sank into the
harbour; some 2,000 people died at once. Robert Marx dug some twenty to thirty buildings in 1965-1968, and the
Institute of Nautical Archaeology and Texas A&M, with the Jamaica National Heritage Trust, excavated from 1981 to
1990 around the corner of Queen Street and High Street: brick buildings standing where they fell, their walls
collapsed outwards, the timber floors and posts, and the things of daily life — pewter, bottles, clay pipes, a
pocket watch stopped at 11:43. It is an underwater archaeological preserve, closed to divers without a permit.
Here a block of the town at that street corner is drawn, about 90 m across; the layout and the depth are approximate.

    python model_sources/Q685357_port_royal.py   -> models/item/Q685357/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, crumple, fbm  # noqa: E402
from placekit import (brick_wall, rubble, timbers, BRICK, TIMBER, dress_place)  # noqa: E402

rng = np.random.default_rng(1692)


W, F = [], []
# buildings in rows along Queen Street (along z) and High Street (along x), each a brick shell of one or two rooms
plots = [(x, z) for x in (-30, -16) for z in np.arange(-36, 40, 13)] + [(x, z) for x in np.arange(4, 44, 12) for z in (-24, -10)]
for k, (x, z) in enumerate(plots):
    if rng.random() < 0.15:
        continue
    w, d = rng.uniform(8, 11), rng.uniform(9, 12)
    c = [(x - w / 2, z - d / 2), (x + w / 2, z - d / 2), (x + w / 2, z + d / 2), (x - w / 2, z + d / 2)]
    for j in range(4):
        W.append(brick_wall(c[j], c[(j + 1) % 4], rng.uniform(0.5, 2.0), 0.45, 10 + k * 4 + j, ruin=0.6))
    if rng.random() < 0.6:
        W.append(brick_wall((x, z - d / 2), (x, z + d / 2), rng.uniform(0.3, 1.2), 0.35, 500 + k, ruin=0.7))
    # the walls that fell outwards: a spread of brick rubble on one side; the floor timbers and posts inside
    s = rng.choice([-1, 1])
    F.append(rubble(30, x + s * (w / 2 + 2), z, 2.2, 0.35, BRICK, 900 + k, flat=0.5))
    F.append(timbers(6, x, z, 2.5, 1300 + k, TIMBER, (2.0, 5.0)))
    for p in range(3):
        F.append(box(0.22, rng.uniform(0.3, 1.0), 0.22, TIMBER).T(x + rng.uniform(-w / 3, w / 3), 0, z + rng.uniform(-d / 3, d / 3)))
walls = merge([m for m in W if len(m.f)])
debris = merge(F)
small = []
for k in range(70):                                                     # bottles, pewter plates, clay pipes
    t = k % 3
    if t == 0:
        m = lathe([(0.001, 0), (0.07, 0.0), (0.075, 0.12), (0.03, 0.2), (0.022, 0.28)], (0.18, 0.24, 0.14), seg=8).R(math.pi / 2 * (k % 2), rng.uniform(0, 6.28), 0)
    elif t == 1:
        m = cyl(0.12, 0.1, 0.02, (0.5, 0.5, 0.48), seg=12)
    else:
        m = merge([cyl(0.012, 0.012, 0.3, (0.8, 0.78, 0.72), seg=5).R(0, 0, math.pi / 2), cyl(0.02, 0.025, 0.04, (0.8, 0.78, 0.72), seg=6).T(0.15, 0, 0)])
    small.append(m.T(rng.uniform(-36, 50), 0.03, rng.uniform(-42, 44)))

out = ROOT / "models" / "item" / "Q685357" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("buildings", walls, "stone"), ("collapsed", debris, "stone")], [("finds", merge(small), "artefact")],
            depth=-6, seed=1692, drift=0.3)
