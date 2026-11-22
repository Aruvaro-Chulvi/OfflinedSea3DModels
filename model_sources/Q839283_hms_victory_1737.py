"""HMS Victory (1737) (Q839283) — the wreck of the British first-rate lost in the Channel in 1744, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a 100-gun first-rate ship of the line launched at Portsmouth in 1737, the predecessor of Nelson's Victory:
about 53 m on the gun deck and 15.3 m in the beam, three gun decks of bronze guns (42-pounders on the lower deck), the
high stern of the period with its galleries, three masts and a bowsprit.

The wreck, after the published descriptions (Wikipedia, the UK government, Live Science, the archaeological
monitoring of the site): flagship of Admiral Sir John Balchin, she was lost with all her company of about 1,100 men in
a storm on 5 October 1744. Odyssey Marine Exploration found the site in 2008 in the western English Channel, some 80 km
from Guernsey, at about 75-80 m (the figures published range from 74 to 100 m) and raised two bronze guns: dozens of
bronze guns bearing the arms of George I and George II lie on the sea bed with the anchors, while the hull itself is
largely gone or buried. The site is divided into seven areas, all of them disturbed by trawling and by illicit
salvage. In 2012 the wreck was given to the Maritime Heritage Foundation. The layout of the guns and the remains, and
the heading (not published here: 0) are approximate.

    python model_sources/Q839283_hms_victory_1737.py   -> models/item/Q839283/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, fbm, sstep  # noqa: E402
from shipkit import install  # noqa: E402
from sailkit import Ship, Mast, solidify, frames, cannon  # noqa: E402

rng = np.random.default_rng(1744)
BRONZE_SEA = (0.27, 0.29, 0.21)
TIMBER = (0.24, 0.18, 0.12)
IRON = (0.22, 0.18, 0.15)


def ship():
    return Ship(L=53.0, B=15.3, D=8.8, draft=6.6, fore=[(9.0, 2.4)], aft=[(17.0, 2.4), (9.0, 2.2)],
                sheer=2.0, tumble=0.18, castle_tumble=0.12, bluff=0.72, transom=0.6, bowsprit=16.0, stern_windows=6, jibs=1,
                masts=[Mast(0.8, 32, "square", 3), Mast(0.52, 40, "square", 3), Mast(0.2, 26, "gaff", 2)],
                gunports=[(3.2, 0.12, 0.84, 14), (5.4, 0.1, 0.86, 15), (7.6, 0.12, 0.84, 14)], hull_col=(0.22, 0.17, 0.12),
                upper_col=(0.48, 0.36, 0.2), port_col=(0.55, 0.13, 0.08), lower_decks=[3.0, 5.2], trim_col=(0.72, 0.56, 0.2))


install(globals(), "compact")


def anchor(size=1.0):
    """An admiralty anchor of the period lying on its side: shank, arms, flukes and the wooden stock."""
    s = size
    P = [box(0.3 * s, 0.3 * s, 5.0 * s, IRON).T(0, 0.15 * s, 0)]
    for k in (-1, 1):
        P.append(tube([(0, 0.15 * s, -2.4 * s), (k * 1.2 * s, 0.15 * s, -2.2 * s), (k * 1.9 * s, 0.15 * s, -1.3 * s)], 0.14 * s, IRON, seg=6))
        P.append(box(0.9 * s, 0.1 * s, 0.7 * s, IRON).R(0, k * 0.6, 0).T(k * 1.8 * s, 0.1 * s, -1.4 * s))
    P.append(box(0.45 * s, 0.45 * s, 0.1, IRON).T(0, 0.3 * s, 2.5 * s))
    P.append(box(4.4 * s, 0.4 * s, 0.4 * s, TIMBER).R(0, 0, math.pi / 2).T(0, 0.4 * s, 2.2 * s))
    return merge(P)


S = ship()
L = S.L
hull = S._hull()
ribs = frames(S, np.arange(0.06, 0.9, 1.2 / L), col=TIMBER, size=0.32, inset=0.34, upto=0.4)
# what is left of the hull: little; the lowest part of the bottom, mostly buried, broken up by trawls
fc = hull.v[hull.f].mean(1)
t = S.tz(fc[:, 2])
cut = 1.6 + 1.0 * (fbm(fc, 0.18, 3) - 0.5)
keep = (fc[:, 1] < cut) & (t > 0.1 + 0.06 * fbm(fc, 0.3, 4)) & (t < 0.86) & (fbm(fc, 0.12, 5) > 0.35)
lower = solidify(compact(Mesh(hull.v, hull.f[keep], hull.c)), 0.35, TIMBER)
fr = ribs.v[ribs.f].mean(1)
ribs = compact(Mesh(ribs.v, ribs.f[(fr[:, 1] < 2.0 + 0.8 * (fbm(fr, 0.3, 6) - 0.5)) & (fbm(fr, 0.15, 7) > 0.4)], ribs.c))
wreck = merge([lower, ribs])
crumple(wreck, 0.2, 0.3, 8)
wreck.R(0, 0, math.radians(6))
wreck.T(0, -wreck.v[:, 1].min() - 1.4, 0)                            # deep in the sand and shell

# the bronze guns, lying in rows and heaps where the decks fell, with the anchors
G, F = [], []
for i in range(44):
    big = i < 14
    x = rng.normal(0, 5.0)
    z = rng.uniform(-L * 0.5, L * 0.5) if i < 34 else rng.uniform(-L * 0.9, L * 0.9)
    gn = cannon(3.1 if big else rng.uniform(2.4, 2.9), 0.1 if big else rng.uniform(0.07, 0.09), bronze=True)
    gn.c = np.clip(np.array(BRONZE_SEA) * (0.8 + 0.4 * fbm(gn.v, 2.0, i, 2))[:, None], 0, 1)
    gn.R(rng.uniform(-0.15, 0.15), rng.uniform(-0.5, 0.5) + (0 if rng.random() < 0.6 else math.pi / 2), rng.uniform(0, 6.28))
    G.append(gn.T(x, 0.02, z))
for k, (x, z, yaw) in enumerate(((-6.0, L * 0.55, 0.4), (7.0, L * 0.5, -0.9))):
    a = anchor(1.35)
    crumple(a, 0.05, 0.8, 30 + k)
    F.append(a.R(0, yaw, 0).T(x, 0.0, z))
for i in range(90):                                                     # timbers, a rudder fragment, shot
    ln = rng.uniform(1.2, 6.0)
    p = box(rng.uniform(0.22, 0.45), rng.uniform(0.1, 0.2), ln, TIMBER)
    crumple(p, 0.05, 0.6, 300 + i)
    F.append(p.R(rng.uniform(-0.1, 0.1), rng.uniform(0, 6.28), rng.uniform(-0.2, 0.2)).T(rng.normal(0, 9), 0.02, rng.uniform(-L * 0.8, L * 0.8)))
for i in range(60):
    F.append(lathe([(0.001, -0.09)] + [(0.09 * math.sin(a), -0.09 * math.cos(a)) for a in np.linspace(0.3, math.pi - 0.3, 5)] + [(0.001, 0.09)], IRON, seg=6)
             .T(rng.normal(0, 4), 0.0, rng.uniform(-20, 20)))
guns = merge(G)
finds = merge(F)

P = S.build(built=True)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q839283" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", finds, "loose")], "plain": [("guns", guns, "artefact")]}],
            depth=-75, kind="galleon", seed=1744, wood=True, memorial=True, drift=0.6)
dress_built(out.with_name("model_built.glb"),
            [("hull", merge([P["hull"], P["deck"]]), "wood"), ("rig", P["spars"], "wood"), ("rigging", P["ropes"], "rope"),
             ("sails", P["sails"], "cloth"), ("guns", P["guns"], "metal")], deck_colors=[S.deck_col])
