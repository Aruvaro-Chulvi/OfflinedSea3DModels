"""
Breadalbane (Q4959197) — the wreck of the barque lost in the Arctic ice off Beechey Island, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a three-masted wooden barque built at Dundee in 1843 (about 38 m), taken up in 1853 as a supply ship for
the Royal Navy's search for Sir John Franklin's lost expedition: square sails on the fore and main masts, a gaff
sail on the mizzen, a bowsprit with jibs, copper sheathing below the waterline.

The wreck, after the published descriptions: just after midnight on 21 August 1853, half a mile south of Beechey
Island, a slab of ice knifed through her starboard bow and she sank within fifteen minutes; her 21 men got off.
Joe MacInnis's team found her in 1980 with side-scan sonar at about 100 m, some 1.5 km south of Beechey Island:
upright and intact on the bottom, two of her three masts still standing, her fractured bowsprit pointing east,
one of the best-preserved wooden wrecks known (the cold water has no shipworm). The most northerly known
shipwreck and a National Historic Site of Canada. Which mast fell, the state of the yards and the size of the
hole in the bow are approximate.

    python model_sources/Q4959197_breadalbane.py   -> models/item/Q4959197/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, tube, crumple, fbm, sstep  # noqa: E402
from sailkit import Ship, Mast, solidify, frames, wheel  # noqa: E402

rng = np.random.default_rng(1853)
BLACK = (0.08, 0.08, 0.085)
COPPER = (0.42, 0.5, 0.42)                     # verdigris on the copper sheathing
HOUSE = (0.55, 0.45, 0.32)
SPAR = (0.45, 0.35, 0.22)

S = Ship(L=38.1, B=8.2, D=5.4, draft=4.2, fore=(5.0, 1.2), aft=(8.0, 1.2), sheer=1.0, tumble=0.05, castle_tumble=0.02,
         bluff=0.5, transom=0.5, bowsprit=9.0, spritsail=False, jibs=3, yard=1.8,
         masts=[Mast(0.74, 26, "square", 3), Mast(0.48, 27, "square", 3), Mast(0.22, 21, "gaff")],
         hull_col=COPPER, upper_col=BLACK, wale_col=(0.85, 0.83, 0.75), deck_col=(0.6, 0.53, 0.4),
         spar_col=SPAR, sail_col=(0.84, 0.8, 0.7),
         deckhouses=[(0.38, 0.46, 3.0, 1.9, HOUSE), (0.1, 0.18, 3.2, 1.8, HOUSE)])
L = S.L


def standing_mast(t, h):
    """A lower mast and the stump of its topmast, with the top (platform) — standing on the wreck."""
    y = S.deck(t)
    P = [cyl(0.32, 0.27, h * 0.62, SPAR, seg=10).T(0, y, S.zt(t)),
         box(2.4, 0.18, 1.8, SPAR).T(0, y + h * 0.62, S.zt(t)),
         cyl(0.2, 0.16, h * 0.38, SPAR, seg=8).T(0, y + h * 0.62, S.zt(t) - 0.25)]
    for s in (-1, 1):                                                         # a few shrouds still set up
        P.append(tube([(s * S.B * 0.47, y + 0.9, S.zt(t) - 0.4), (s * 0.4, y + h * 0.6, S.zt(t))], 0.04, (0.2, 0.16, 0.12), seg=3))
        P.append(tube([(s * S.B * 0.47, y + 0.9, S.zt(t) + 0.6), (s * 0.4, y + h * 0.6, S.zt(t))], 0.04, (0.2, 0.16, 0.12), seg=3))
    P.append(box(S.B * 1.2, 0.2, 0.25, SPAR).R(0, 0, 0.12).T(0, y + h * 0.55, S.zt(t) + 0.3))   # the lower yard, hanging askew
    return merge(P)


# ── The wreck ────────────────────────────────────────────────────────────────
hull = S._hull()
det, _ = S._details(False)
deck = S._decks()
stumps, _, _ = S._rig(False)
fch = hull.v[hull.f].mean(1)
th = S.tz(fch[:, 2])
hole = (th > 0.84) & (th < 0.94) & (fch[:, 0] < -0.3) & (np.abs(fch[:, 1] - 3.6) < 1.1 + 0.5 * (fbm(fch, 0.6, 3, 2) - 0.5))
hull = Mesh(hull.v, hull.f[~hole], hull.c)                                     # the ice cut through the starboard bow
hull = solidify(hull, 0.25, (0.3, 0.22, 0.15))
ribs = frames(S, np.arange(0.84, 0.95, 0.6 / L), col=(0.3, 0.22, 0.15), size=0.28, inset=0.3, upto=0.95)
ribs.f = ribs.f[ribs.v[ribs.f].mean(1)[:, 0] < 0]
masts_up = merge([standing_mast(0.74, 17.0), standing_mast(0.48, 18.0)])
helm = wheel().T(0, S.deck(0.07) + S.castle(0.07), S.zt(0.07))
wreck = merge([hull, det, deck, stumps, ribs, masts_up, helm])
crumple(wreck, 0.05, 1.2, 12)
crumple(wreck, 0.25, 0.5, 11, sstep(6, 0, L / 2 - wreck.v[:, 2]))
lum = wreck.c.mean(1)
worn = sstep(0.55, 0.75, fbm(wreck.v, 0.4, 21, 3)) * (lum < 0.12)
wreck.c = wreck.c * (1 - worn[:, None]) + np.array([0.3, 0.24, 0.17]) * worn[:, None]
wreck.T(0, -1.4, 0)                                                             # upright, settled into the bottom

# the fractured bowsprit, the fallen mizzen and some spars beside her
yd = S.deck(0.5) - 1.4
sp = [tube([(0, S.deck(1.0) - 0.8, S.zt(1.0)), (0.3, S.deck(1.0) + 0.2, S.zt(1.0) + 4.0)], 0.22, SPAR, seg=8),
      tube([(0.3, S.deck(1.0) + 0.2, S.zt(1.0) + 4.0), (1.5, 0.3, S.zt(1.0) + 8.5)], 0.17, SPAR, seg=8)]
sp.append(tube([(0, yd + 2.0, S.zt(0.22)), (4.5, 0.25, S.zt(0.22) - 12.0)], 0.24, SPAR, seg=8))    # the mizzen, over the port quarter
for i, (x, z, a, ln) in enumerate(((5.5, -4, 0.3, 9), (-6.0, 3, -0.2, 11), (6.5, 9, 0.1, 8))):
    c = np.array([x, 0.2, z]); d = np.array([math.sin(a), 0, math.cos(a)])
    sp.append(tube([tuple(c - d * ln / 2), tuple(c + d * ln / 2)], 0.16, SPAR, seg=6))
for i in range(6):
    x0 = rng.choice([-1, 1]) * S.B * 0.48; z0 = rng.uniform(S.zt(0.15), S.zt(0.9))
    pts = [(x0, yd + 0.9, z0)] + [(x0 + np.sign(x0) * k * 1.0, max(0.06, yd + 0.9 - k * 1.3), z0 + 0.3 * k * k * (1 if i % 2 else -1)) for k in range(1, 6)]
    sp.append(tube(pts, 0.04, (0.22, 0.18, 0.13), seg=3))
spars = merge(sp)
for m in sp:
    crumple(m, 0.05, 2.0, int(abs(m.v[0, 2]) * 10) + 1)

# ── The barque as she sailed ─────────────────────────────────────────────────
P = S.build(built=True)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q4959197" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull", wreck), ("debris_field", spars)]}], depth=-100, kind="expedition", seed=1853, wood=True)
dress_built(out.with_name("model_built.glb"),
            [("hull", merge([P["hull"], P["deck"], wheel().T(0, S.deck(0.07) + S.castle(0.07), S.zt(0.07))]), "wood"), ("rig", P["spars"], "wood"),
             ("rigging", P["ropes"], "rope"), ("sails", P["sails"], "cloth")], deck_colors=[S.deck_col])
