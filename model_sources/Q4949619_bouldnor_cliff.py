"""Bouldnor Cliff (Q4949619) — the drowned Mesolithic landscape off the Isle of Wight, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the Maritime Archaeology Trust, Smithsonian, Current
Archaeology): about 8,000 years ago the Solent was a river valley; as the sea rose after the Ice Age it drowned the
forest and the camps on its banks. Off Bouldnor, 1 km east of Yarmouth, an underwater cliff of peat and clay at about
11 m is the edge of that old land surface, eroding year by year: the trunks and stumps of oaks stick out of it, and
the Trust's divers have found worked flints (the first in 1999, thrown out of its burrow by a lobster), hearths,
worked wood and, in 2019, a platform of more than 60 split and trimmed timbers — possibly where log boats were built —
which was lifted for conservation because the site is wearing away. Here a stretch of the drowned cliff is drawn,
about 80 m long; its shape is approximate.

    python model_sources/Q4949619_bouldnor_cliff.py   -> models/item/Q4949619/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, crumple, fbm, sstep  # noqa: E402
from placekit import (timbers, dress_place)  # noqa: E402

rng = np.random.default_rng(8000)
PEAT = (0.24, 0.2, 0.15)
CLAYC = (0.46, 0.44, 0.4)
OAK = (0.2, 0.15, 0.1)


def bank(X, Z):
    """The old land surface: a shelf of peat and clay about 3 m above the eroded floor, its edge a ragged step."""
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    edge = 6.0 * (fbm(P3 * [0, 0, 1], 0.06, 3, 3).reshape(X.shape) - 0.5) + 1.5 * (fbm(P3, 0.3, 4, 2).reshape(X.shape) - 0.5)
    step = sstep(-1.5, 1.5, X - edge)
    fade = sstep(30, 20, X) * sstep(40, 30, np.abs(Z))                    # the old land surface slopes away at the model's edges
    return (3.0 * step + 0.4 * (fbm(P3, 0.15, 5, 3).reshape(X.shape) - 0.5) * step) * fade


ground = grid(-30, 30, -40, 40, 90, 120, bank, PEAT)
y = ground.v[:, 1]
ground.c = np.where((y > 0.3)[:, None] & (y < 2.6)[:, None] & ((np.floor(y / 0.35) % 2) == 0)[:, None], np.array(CLAYC) * 0.8, ground.c)   # bands of clay in the face
ground.c = np.clip(ground.c * (0.8 + 0.4 * fbm(ground.v, 0.8, 6, 2))[:, None], 0, 1)
T = []
for k in range(22):                                                     # oak trunks and stumps sticking out of the peat
    z = rng.uniform(-36, 36)
    if k % 3 == 0:
        st = lathe([(0.6, -1.0), (0.55, 0.2), (0.45, 0.6), (0.001, 0.62)], OAK, seg=10)
        for a in np.linspace(0, 6.28, 5)[:-1]:
            st = merge([st, cyl(0.15, 0.08, 1.6, OAK, seg=6).R(0, 0, 1.25).R(0, a, 0).T(0, 0.1, 0)])
        T.append(st.T(rng.uniform(2, 20), 3.0, z))
    else:
        ln = rng.uniform(4, 11)
        t = cyl(rng.uniform(0.25, 0.5), rng.uniform(0.15, 0.3), ln, OAK, seg=8).R(0, 0, math.pi / 2 + rng.uniform(-0.15, 0.15))
        crumple(t, 0.06, 0.4, 100 + k)
        T.append(t.R(0, rng.uniform(-0.6, 0.6), 0).T(rng.uniform(-3, 4) + ln / 2 * 0.3, rng.uniform(0.4, 2.6), z))
trees = merge(T)
wood = timbers(30, -6, 0, 6, 200, OAK, (0.8, 2.5))                       # worked timbers washed out on the floor below the step

out = ROOT / "models" / "item" / "Q4949619" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("peat_cliff", ground, "rock"), ("trees", trees, "wood")], [("timbers", wood, "wood")], depth=-11, seed=8000, drift=0.15)
