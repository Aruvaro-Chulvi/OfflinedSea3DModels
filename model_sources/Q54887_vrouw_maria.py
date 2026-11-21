"""
Vrouw Maria (Q54887) — the Dutch merchant snow wrecked in the Finnish archipelago in 1771, and her wreck, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Dutch two-masted merchant snow, about 26 m long and 7 m in the beam: bluff bows, a raised quarterdeck,
a square-rigged foremast and mainmast with the small trysail mast of a snow just aft of the mainmast, a bowsprit.
On her way from Amsterdam to St Petersburg in 1771 she carried, among sugar, dyes, zinc and cloth, works of art
bought for Catherine the Great.

She struck a rock off Jurmo, lost her rudder and sank on 9 October 1771; her crew were saved. The Pro Vrouw Maria
society found her in 1999 with side-scan sonar at about 41 m. The cold, brackish Baltic water, without shipworm,
kept her well: she sits upright on her keel on the clay, leaning a little to starboard, bow to the south-south-west
(156°), most of her 26 by 7 m hull in excellent condition; her lower masts still stand, rising to 22-24 m below the
surface, but the upper masts and the rigging are gone; the rudder is not by the wreck. The exact lean and the depth
she has sunk into the clay are approximate.

    python model_sources/Q54887_vrouw_maria.py   -> models/item/Q54887/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, tube, crumple, fbm, sstep  # noqa: E402
from sailkit import Ship, Mast, solidify, frames  # noqa: E402

rng = np.random.default_rng(1771)


def ship():
    return Ship(L=26.0, B=7.2, D=3.7, draft=3.0, fore=(3.5, 1.2), aft=(7.0, 1.4), sheer=1.0, tumble=0.1, castle_tumble=0.06,
                bluff=0.9, transom=0.5, bowsprit=10.0, spritsail=False, jibs=2, stern_windows=3,
                masts=[Mast(0.74, 19.0, "square", 3), Mast(0.44, 21.0, "square", 3), Mast(0.36, 15.0, "gaff", top=False)],
                gunports=[], guns=False, hull_col=(0.2, 0.16, 0.12), upper_col=(0.2, 0.3, 0.26), deck_col=(0.58, 0.48, 0.34),
                trim_col=(0.62, 0.48, 0.2))


S = ship()
L = S.L


def lower_masts():
    """What stands of the masts: the lower fore and main masts with their tops, the upper masts and rigging gone."""
    P = []
    for m in S.masts[:2]:
        z = S.zt(m.at); y0 = S._mast_base(m); h = m.height
        r0 = 0.011 * h + 0.14
        P.append(cyl(r0, r0 * 0.8, h * 0.58 + 0.6, S.spar_col, seg=10).T(0, y0, z))
        P.append(cyl(1.1 + 0.02 * h, 0.9 + 0.02 * h, 0.4, S.spar_col, seg=14).T(0, y0 + h * 0.58 - 0.6, z))     # the top
    m = S.masts[2]
    P.append(cyl(0.16, 0.13, 3.0, S.spar_col, seg=8).T(0, S._mast_base(m), S.zt(m.at)))                 # trysail mast stump
    return merge(P)


hull = S._hull()
det, _ = S._details(False)
deck = S._decks()
# the rudder (and its tiller) were lost on the rock: not by the wreck
fcd = det.v[det.f].mean(1)
ytl = S.deck(0.0) + S.castle(0.0) + 0.5
is_tiller = (np.abs(fcd[:, 0]) < 0.2) & (np.abs(fcd[:, 1] - (ytl - 0.05)) < 0.3) & (fcd[:, 2] < S.zt(0) + 2.2)
is_rud = ((fcd[:, 2] < S.zt(0) - 0.05) & (np.abs(fcd[:, 0]) < 0.4)) | is_tiller
det = Mesh(det.v, det.f[~is_rud], det.c)
hull = solidify(hull, 0.22, (0.24, 0.18, 0.13))
# the bowsprit broken short
fcd = det.v[det.f].mean(1)
det = Mesh(det.v, det.f[~(fcd[:, 2] > S.zt(1.0) + 3.0)], det.c)
masts = lower_masts()
wreck = merge([hull, det, deck, masts])
# 250 years in cold brackish water: the paint gone grey-brown, the wood dark but whole
g = wreck.c.mean(1, keepdims=True)
wreck.c = np.clip((wreck.c * 0.55 + g * 0.45) * 1.05 + np.array([0.02, 0.015, 0.0]), 0, 1)
crumple(wreck, 0.04, 1.0, 12)
wreck.R(0, 0, math.radians(-6))                                   # leaning a little to starboard (−x)
wreck.T(0, -wreck.v[:, 1].min() - 1.0, 0)                         # her keel in the clay
# a few things fallen beside her: a yard, a block of planks, cargo barrels
D = []
for (x, z, a, ln, r) in ((-6.0, 2.0, 0.3, 11.0, 0.16), (5.5, -4.0, -0.2, 8.0, 0.13)):
    c = np.array([x, 0.18, z]); d = np.array([math.sin(a), 0, math.cos(a)])
    D.append(tube([tuple(c - d * ln / 2), tuple(c), tuple(c + d * ln / 2)], r, S.spar_col, seg=8))
for i in range(8):
    b = cyl(0.32, 0.32, 0.8, (0.3, 0.23, 0.15), seg=10).R(math.pi / 2, rng.uniform(0, 6.28), 0)
    D.append(b.T(rng.choice([-1, 1]) * rng.uniform(4.5, 7.0), 0.32, rng.uniform(-8, 8)))
debris = merge(D)
debris.c = np.clip(debris.c * 0.8, 0, 1)

P = S.build(built=True)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q54887" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": []}],
            depth=-41, kind="galleon", seed=1771, wood=True, life="baltic")
dress_built(out.with_name("model_built.glb"),
            [("hull", merge([P["hull"], P["deck"]]), "wood"), ("rig", P["spars"], "wood"), ("rigging", P["ropes"], "rope"),
             ("sails", P["sails"], "cloth")], deck_colors=[S.deck_col])
