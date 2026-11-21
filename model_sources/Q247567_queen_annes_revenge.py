"""Queen Anne's Revenge (Q247567) — the wreck of Blackbeard's flagship in Beaufort Inlet, North Carolina, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: built in England around 1710, taken by the French as the slave ship La Concorde, she was captured by the
pirate Edward Teach, Blackbeard, in November 1717 and armed as his flagship with some 40 guns: about 31 m long and
7.6 m in the beam, three masts, a single gun deck under the open upper deck, a quarterdeck aft.

The wreck, after the published descriptions (Wikipedia, the North Carolina Department of Natural and Cultural
Resources' Queen Anne's Revenge Project): she ran aground on a sandbar at Beaufort Inlet in June 1718 and was
abandoned. Found in 1996 by Intersal, the site in 6-8 m of water was a low mound of ballast stones with cannon, four
large anchors and a section of the lower hull buried beneath; it was mapped and then excavated by the state's
archaeologists from 1997: some 30 guns were found (24 raised by 2015), and the anchors and some hundreds of thousands
of objects have been raised and conserved, so the model shows the site as it was recorded during the excavations. The layout of the guns,
anchors and ballast and the heading (not published here: 0) are approximate.

    python model_sources/Q247567_queen_annes_revenge.py   -> models/item/Q247567/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, tube, crumple, fbm, sstep  # noqa: E402
from sailkit import Ship, Mast, solidify, frames, cannon  # noqa: E402

rng = np.random.default_rng(1718)
TIMBER = (0.25, 0.19, 0.13)
IRON = (0.26, 0.2, 0.16)
CONCRETION = (0.45, 0.4, 0.32)


def ship():
    return Ship(L=31.0, B=7.6, D=4.6, draft=3.6, fore=[(5.0, 1.8)], aft=[(9.0, 2.2)],
                sheer=1.4, tumble=0.14, castle_tumble=0.1, bluff=0.62, transom=0.5, bowsprit=10.0, stern_windows=3, jibs=1,
                masts=[Mast(0.78, 22, "square", 3), Mast(0.5, 26, "square", 3), Mast(0.2, 16, "gaff", 1)],
                gunports=[(2.9, 0.12, 0.82, 10), (4.4, 0.18, 0.74, 7)], hull_col=(0.2, 0.15, 0.1),
                upper_col=(0.18, 0.16, 0.14), port_col=(0.5, 0.12, 0.08), lower_decks=[2.4], trim_col=(0.5, 0.4, 0.2))


def compact(m):
    if not len(m.f):
        return Mesh()
    u, inv = np.unique(m.f, return_inverse=True)
    return Mesh(m.v[u], inv.reshape(-1, 3), m.c[u])


def anchor(size=1.0):
    s = size
    P = [box(0.24 * s, 0.24 * s, 4.0 * s, IRON).T(0, 0.12 * s, 0)]
    for k in (-1, 1):
        P.append(tube([(0, 0.12 * s, -1.9 * s), (k * 1.0 * s, 0.12 * s, -1.75 * s), (k * 1.5 * s, 0.12 * s, -1.0 * s)], 0.11 * s, IRON, seg=6))
        P.append(box(0.7 * s, 0.08 * s, 0.55 * s, IRON).R(0, k * 0.6, 0).T(k * 1.45 * s, 0.08 * s, -1.1 * s))
    P.append(box(0.4 * s, 0.4 * s, 0.1, IRON).T(0, 0.25 * s, 2.0 * s))
    return merge(P)


S = ship()
L = S.L


def mound_h(X, Z):
    """The ballast mound: river stones heaped over the buried lower hull."""
    d = np.sqrt((X / 4.5) ** 2 + (Z / 11.0) ** 2)
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    return np.maximum(0.0, 0.8 * (1 - d ** 2)) + 0.25 * (fbm(P3, 1.2, 3, 3).reshape(X.shape) - 0.5) * (d < 1.1) - 0.04


mound = grid(-7, 7, -15, 15, 42, 90, mound_h, (0.46, 0.43, 0.38))
mound.f = mound.f[mound.v[mound.f][:, :, 1].max(axis=1) > 0.03]
mound.c = np.clip(mound.c * (0.75 + 0.5 * fbm(mound.v, 2.5, 4)[:, None]), 0, 1)
hull = S._hull()
fc = hull.v[hull.f].mean(1)
t = S.tz(fc[:, 2])
keep = (fc[:, 1] < 0.9 + 0.5 * (fbm(fc, 0.3, 5) - 0.5)) & (t > 0.3) & (t < 0.7) & (fc[:, 0] < 1.5)    # the bit of the lower hull that survives
timbers = solidify(compact(Mesh(hull.v, hull.f[keep], hull.c)), 0.25, TIMBER)
timbers.T(0, -timbers.v[:, 1].min() - 0.6, 0)
G, F = [], []
for i in range(30):                                                        # iron guns, heaped and concreted, many across the mound
    gn = cannon(rng.uniform(1.8, 2.6), rng.uniform(0.04, 0.06), bronze=False)
    gn.c = np.clip(np.array(CONCRETION) * (0.7 + 0.5 * fbm(gn.v, 2.0, i, 2))[:, None], 0, 1)
    crumple(gn, 0.03, 2.0, 50 + i)
    gn.R(rng.uniform(-0.2, 0.2), rng.uniform(0, 6.28), rng.uniform(0, 6.28))
    x, z = rng.normal(0, 2.8), rng.uniform(-10, 10)
    y = max(0.0, 0.8 * (1 - (x / 4.5) ** 2 - (z / 11.0) ** 2))
    G.append(gn.T(x, y, z))
for k, (x, z, yaw) in enumerate(((-3.0, 14.0, 0.3), (4.0, 12.0, -0.8), (-1.0, 22.0, 2.4), (5.0, -14.0, 1.2))):    # the four large anchors
    a = anchor(1.25)
    a.c = np.clip(np.array(CONCRETION) * (0.7 + 0.5 * fbm(a.v, 1.5, 80 + k, 2))[:, None], 0, 1)
    F.append(a.R(0.0, yaw, 0.1).T(x, 0.05, z))
for i in range(40):
    F.append(lathe([(0.001, -0.06)] + [(0.06 * math.sin(a), -0.06 * math.cos(a)) for a in np.linspace(0.3, math.pi - 0.3, 5)] + [(0.001, 0.06)], IRON, seg=6)
             .T(rng.normal(0, 3), 0.0, rng.uniform(-12, 12)))
guns = merge(G)
finds = merge(F)

P = S.build(built=True)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q247567" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", timbers), ("debris_field", finds, "loose")],
                   "plain": [("ballast", mound, "rock"), ("guns", guns, "artefact")]}],
            depth=-7, kind="galleon", seed=1718, wood=True, drift=0.5)
dress_built(out.with_name("model_built.glb"),
            [("hull", merge([P["hull"], P["deck"]]), "wood"), ("rig", P["spars"], "wood"), ("rigging", P["ropes"], "rope"),
             ("sails", P["sails"], "cloth"), ("guns", P["guns"], "metal")], deck_colors=[S.deck_col])
