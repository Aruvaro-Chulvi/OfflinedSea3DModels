"""
Mahdia shipwreck (Q3589083) — the Roman-era cargo of marble columns off Mahdia, Tunisia, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a large merchant ship of the early 1st century BC, sailing from Athens (Piraeus) towards Italy with a cargo
of works of art: some sixty to seventy marble columns (about 200 tonnes) with their capitals and bases, marble
kraters and candelabra, bronze statues and fittings, furniture.

The site, after the published descriptions (Wikipedia, the Oxford Roman Economy Project, the French underwater
archaeology department): found by Greek sponge divers in 1907 some 5 km off Mahdia at about 36-39 m; Alfred Merlin
raised most of the sculptures between 1907 and 1913 (now in the Bardo Museum); Philippe Tailliez and Jacques-Yves
Cousteau dived on it in 1948 with the first aqualungs; further surveys followed, but the hull itself has not been
excavated. On the sand the marble columns still lie in rows over the low mound of the buried hull, with capitals,
column bases, lead anchor stocks and broken amphorae around. The number and layout of the columns drawn are
approximate.

    python model_sources/Q3589083_mahdia_shipwreck.py   -> models/item/Q3589083/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, crumple, fbm  # noqa: E402

rng = np.random.default_rng(81)
MARBLE = (0.72, 0.7, 0.64)
TIMBER = (0.26, 0.2, 0.14)
LEAD = (0.38, 0.38, 0.4)
CLAY = (0.6, 0.42, 0.3)


def column(ln, r):
    """A column shaft lying on its side (along z), slightly tapered, with its fluting suggested by 16 facets."""
    m = cyl(r, r * 0.88, ln, MARBLE, seg=16).R(math.pi / 2, 0, 0)
    m.c = np.clip(m.c * (0.9 + 0.12 * fbm(m.v, 1.5, int(ln * 10), 2)[:, None]), 0, 1)
    return m


def capital():
    """An Ionic capital: a square abacus over a cushion with two volutes."""
    P = [box(1.1, 0.18, 1.1, MARBLE).T(0, 0.55, 0), cyl(0.36, 0.42, 0.4, MARBLE, seg=14).T(0, 0.15, 0)]
    for s in (-1, 1):
        P.append(cyl(0.26, 0.26, 0.95, MARBLE, seg=12).R(math.pi / 2, 0, 0).T(s * 0.48, 0.38, 0))
    return merge(P)


def base():
    return lathe([(0.0, 0.0), (0.62, 0.0), (0.62, 0.12), (0.52, 0.2), (0.56, 0.3), (0.45, 0.38), (0.0, 0.38)], MARBLE, seg=18)


def amphora():
    prof = [(0.0, 0.0), (0.06, 0.02), (0.16, 0.25), (0.2, 0.55), (0.17, 0.8), (0.07, 0.95), (0.06, 1.15), (0.08, 1.18)]
    return lathe(prof, CLAY, seg=10)


# the low mound of the buried hull, its ballast and a few timbers showing at the edges
def mound(X, Z):
    d = np.sqrt((X / 6.0) ** 2 + (Z / 17.0) ** 2)
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    return np.maximum(0.0, 1.1 * (1 - d ** 2)) + 0.12 * (fbm(P3, 0.4, 3, 3).reshape(X.shape) - 0.5) - 0.05


hull_mound = grid(-9, 9, -22, 22, 36, 88, mound, (0.6, 0.56, 0.47))
hull_mound.f = hull_mound.f[hull_mound.v[hull_mound.f][:, :, 1].max(axis=1) > 0.02]
T_ = []
for s in (-1, 1):
    for z in np.arange(-15, 15, 1.6):
        if rng.random() < 0.45:
            continue
        x = s * 5.8 * math.sqrt(max(0.0, 1 - (z / 17) ** 2)) + rng.normal(0, 0.2)
        t = box(0.25, 0.22, rng.uniform(0.6, 1.4), TIMBER).R(rng.uniform(-0.3, 0.3), 0, 0)
        crumple(t, 0.04, 1.0, int(z * 10 + s * 100))
        T_.append(t.T(x, 0.05, z))

# the cargo: columns in rows across the mound, as they were stowed, a few slid off
C = []
for row in range(6):
    xr = -3.6 + row * 1.45
    z = -14.0 + rng.uniform(0, 2)
    while z < 13.0:
        ln = rng.uniform(3.2, 4.4); r = rng.uniform(0.26, 0.33)
        zc = z + ln / 2
        if rng.random() < 0.82:
            y = max(0.0, 1.1 * (1 - (xr / 6.0) ** 2 - (zc / 17.0) ** 2)) + r - 0.12
            c = column(ln, r).R(0, rng.uniform(-0.06, 0.06), 0)
            C.append(c.T(xr + rng.normal(0, 0.12), y, zc))
        z += ln + rng.uniform(0.15, 0.8)
for k in range(9):                                                   # slid off the mound onto the sand
    ln = rng.uniform(3.0, 4.2); r = rng.uniform(0.26, 0.32)
    c = column(ln, r).R(0, rng.uniform(0, 6.28), 0)
    C.append(c.T(rng.choice([-1, 1]) * rng.uniform(6.5, 11), r - 0.1, rng.uniform(-18, 18)))
for k in range(7):
    cp = capital(); cp.R(rng.uniform(-0.3, 0.3), rng.uniform(0, 6.28), rng.uniform(-0.4, 0.4))
    C.append(cp.T(rng.normal(0, 5), 0.05 + max(0.0, rng.normal(0.3, 0.2)), rng.uniform(-16, 16)))
for k in range(6):
    b_ = base(); b_.R(rng.choice([0.0, math.pi / 2]), rng.uniform(0, 6.28), 0)
    C.append(b_.T(rng.normal(0, 6), 0.3, rng.uniform(-18, 18)))
marble = merge(C)
O = []
for k in range(4):                                                   # lead anchor stocks
    O.append(box(0.18, 0.18, rng.uniform(1.6, 2.2), LEAD).R(0, rng.uniform(0, 6.28), 0).T(rng.normal(0, 4), 0.06, rng.uniform(18, 26)))
for k in range(26):                                                  # amphorae, many broken
    a = amphora(); a.R(math.pi / 2 + rng.uniform(-0.3, 0.3), rng.uniform(0, 6.28), 0)
    fc = a.v[a.f].mean(1)
    if k % 3 == 0:
        a.f = a.f[fc[:, 1] < rng.uniform(0.0, 0.1)]                  # broken in half
    O.append(a.T(rng.normal(0, 7), 0.16, rng.uniform(-20, 22)))
others = merge(O + T_)

from dresskit import dress_wreck  # noqa: E402
out = ROOT / "models" / "item" / "Q3589083" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_mound", hull_mound), ("finds", others, "loose")], "mud": [], "plain": [("marble_cargo", marble, "artefact")]}],
            depth=-38, kind="galleon", seed=81, wood=True, drift=0.3)
