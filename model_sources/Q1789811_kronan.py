"""
Kronan (Q1789811) — the Swedish royal ship lost off Öland in 1676, and her wreck site, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the Royal Swedish ship Kronan, built in 1668-1672 by Francis Sheldon, one of the largest warships of her
day: about 53 m long and 13 m in the beam, three gun decks with more than a hundred guns, a high carved stern,
three masts and a bowsprit.

She was lost at the Battle of Öland on 1 June 1676: turning too sharply under full sail in a strong wind, she heeled
over, her gunpowder magazine exploded and blew away the bow and much of her starboard side, and she sank; some 800
men died. Her wreck was found in 1980 about 3.2 nautical miles east of Hulterstad, in 26 m, and the Kalmar County
Museum has surveyed and excavated it every year since (more than 30,000 finds, among them bronze guns). The part
that holds together is two thirds of her port side, from the stern forward, some 40 by 20 m, lying flat on the clay
with its outside to the bottom, the inner side and the stubs of the gun decks upward; around it lie timbers,
guns, anchors and the rest of the ship broken up. The exact layout of the scattered remains is approximate.

    python model_sources/Q1789811_kronan.py   -> models/item/Q1789811/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, tube, crumple, fbm, sstep  # noqa: E402
from shipkit import install  # noqa: E402
from sailkit import Ship, Mast, solidify, frames, cannon, gun_carriage  # noqa: E402

rng = np.random.default_rng(1676)
BRONZE_SEA = (0.24, 0.27, 0.19)
TIMBER = (0.28, 0.21, 0.14)


def ship():
    return Ship(L=53.0, B=13.0, D=7.6, draft=6.2, fore=[(10.0, 2.6)], aft=[(19.0, 2.6), (11.0, 2.4), (6.0, 2.2)],
                sheer=2.6, tumble=0.16, castle_tumble=0.12, bluff=0.72, transom=0.55, bowsprit=15.0, stern_windows=7,
                masts=[Mast(0.8, 32, "square", 3), Mast(0.53, 42, "square", 3), Mast(0.2, 26, "lateen", top=False)],
                gunports=[(4.4, 0.1, 0.82, 15), (6.6, 0.08, 0.86, 16), (9.0, 0.12, 0.8, 14)],
                hull_col=(0.25, 0.19, 0.13), upper_col=(0.42, 0.3, 0.19), port_col=(0.55, 0.13, 0.08), lower_decks=[4.0, 6.2],
                trim_col=(0.66, 0.5, 0.18))


install(globals(), "compact")


def pick(m, keep):
    return compact(Mesh(m.v, m.f[keep], m.c))


def anchor(col=(0.16, 0.13, 0.11)):
    P = [cyl(0.14, 0.12, 4.8, col, seg=8)]
    for s in (-1, 1):
        pts = [(s * 1.5 * math.sin(a), 0.25 + 1.4 * (1 - math.cos(a)), 0) for a in np.linspace(0, 1.25, 6)]
        P.append(tube(pts, 0.12, col, seg=6))
        P.append(box(0.6, 0.7, 0.07, col, bottom=False).R(0, 0, s * 0.6).T(s * 1.45, 1.4, 0))
    return merge(P)


S = ship()
L = S.L
hull = S._hull()
det, _ = S._details(False)
deck = S._decks()
ribs = frames(S, np.arange(0.03, 0.7, 1.2 / L), col=TIMBER, size=0.32, inset=0.34, upto=0.98)


def side_mask(m, seed):
    """Faces of the port side kept together: from the stern two thirds forward, the outside planking and what is fixed
    to it (ribs, the stubs of the gun decks next to it), with a ragged edge where it broke away."""
    fc = m.v[m.f].mean(1)
    t = S.tz(fc[:, 2])
    n = fbm(fc, 0.15, seed, 3) - 0.5
    tmax = 0.67 + 0.05 * n
    xmin = -0.2 + 1.8 * n                                     # how far in towards the keel the side holds
    return (t < tmax) & (fc[:, 0] > xmin) & (fc[:, 1] > 0.8 + 1.2 * n)


side = merge([solidify(pick(hull, side_mask(hull, 3)), 0.32, TIMBER), pick(det, side_mask(det, 3)), pick(ribs, side_mask(ribs, 3))])
fcd = deck.v[deck.f].mean(1)
side = merge([side, pick(deck, side_mask(deck, 3) & (fcd[:, 0] > 2.5 + 3.0 * (fbm(fcd, 0.3, 4) - 0.5)))])    # deck stubs at the side
g = side.c.mean(1, keepdims=True)
side.c = np.clip((side.c * 0.55 + g * 0.45) * 1.08 + np.array([0.02, 0.014, 0.0]), 0, 1)
crumple(side, 0.22, 0.35, 5)
crumple(side, 0.05, 1.5, 6)
# laid flat: the outside (+x) to the clay
side.R(0, 0, -math.pi / 2)
side.T(-side.v[:, 0].mean(), -side.v[:, 1].min() - 0.9, 0)
low = side.v[:, 1] < -0.2
side.v[low, 1] = -0.2 + (side.v[low, 1] + 0.2) * 0.2

# ── what the explosion and three centuries spread about: guns, timbers, planks, anchors ──
deb, guns = [], []
zf = S.zt(0.67)
for i in range(40):
    x, z = rng.normal(2, 12), rng.uniform(S.zt(0.0) - 8, zf + 30)
    g_ = cannon(rng.uniform(2.4, 3.6), rng.uniform(0.07, 0.11), bronze=i % 4 != 0)
    if i % 4:
        g_.c = np.clip(np.array(BRONZE_SEA) * (0.8 + 0.4 * fbm(g_.v, 2.0, i, 2))[:, None] + g_.c * 0.0, 0, 1)
    g_.R(rng.uniform(-0.15, 0.15), rng.uniform(0, 6.28), rng.uniform(0, 6.28)).T(x, 0.12, z)
    guns.append(g_)
for i in range(6):
    c_ = gun_carriage(2.9, TIMBER); crumple(c_, 0.06, 1.0, 600 + i)
    deb.append(c_.R(rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28), rng.uniform(-0.4, 0.4)).T(rng.normal(0, 10), 0.0, rng.uniform(zf - 10, zf + 25)))
for i in range(260):
    near = i < 180
    x, z = rng.normal(0, 9 if near else 16), rng.uniform(zf - 6, zf + 34) if near else rng.uniform(S.zt(0) - 10, zf + 40)
    ln = rng.uniform(1.5, 8.5)
    p_ = box(rng.uniform(0.22, 0.45), rng.uniform(0.08, 0.16), ln, TIMBER if i % 3 else S.upper_col)
    crumple(p_, 0.05, 0.6, 700 + i)
    deb.append(p_.R(rng.uniform(-0.1, 0.1), rng.uniform(0, 6.28), rng.uniform(-0.2, 0.2)).T(x, 0.05, z))
for i in range(30):
    a0 = rng.uniform(0.3, 1.2); R_ = rng.uniform(2.5, 5.0)
    pts = [(R_ * math.sin(a), R_ * (1 - math.cos(a)), 0) for a in np.linspace(0, a0, 5)]
    deb.append(tube(pts, 0.16, TIMBER, seg=4).R(math.pi / 2 - 0.2, rng.uniform(0, 6.28), 0).T(rng.normal(0, 11), 0.14, rng.uniform(zf - 8, zf + 30)))
mast = merge([cyl(0.7, 0.6, 18.0, S.spar_col, seg=10)]); crumple(mast, 0.08, 0.5, 900)
deb.append(mast.R(0, 0, math.pi / 2).R(0, 0.4, 0).T(10.0, 0.6, zf + 4))
for k, (x, z, a) in enumerate(((-9.0, zf + 30, 0.9), (7.0, zf + 36, 2.4), (-4.0, S.zt(0) - 12, 1.7))):
    deb.append(anchor().R(0, 0, math.pi / 2 + 0.15).R(0, a, 0).T(x, 0.14, z))
debris = merge(deb)
debris.c = np.clip((debris.c * 0.6 + debris.c.mean(1, keepdims=True) * 0.4), 0, 1)
cannons = merge(guns)

P = S.build(built=True)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1789811" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("port_side", side), ("debris_field", debris)], "plain": [("guns", cannons, "artefact")]}],
            depth=-26, kind="galleon", seed=1676, wood=True, life="baltic")
dress_built(out.with_name("model_built.glb"),
            [("hull", merge([P["hull"], P["deck"]]), "wood"), ("rig", P["spars"], "wood"), ("rigging", P["ropes"], "rope"),
             ("sails", P["sails"], "cloth"), ("guns", P["guns"], "metal")], deck_colors=[S.deck_col])
