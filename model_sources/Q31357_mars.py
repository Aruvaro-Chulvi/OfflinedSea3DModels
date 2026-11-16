"""
Mars (Q31357) — the Swedish warship of 1564 and her wreck off Öland, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: flagship of Erik XIV's navy, one of the largest warships of her day: about 48 m on deck and some
13.7 m in the beam; a high aftercastle in two tiers and a forecastle, three masts (fore and main square-rigged,
a lateen mizzen), a bowsprit with its spritsail, and over a hundred guns, many of them bronze, in two tiers of
ports along the sides and in the castles. The hull lines, the castles and the rig are drawn after the
illustrations of the period and the reconstructions of the wreck; they are approximate (no plans survive).
She caught fire and blew up on 31 May 1564, in the first battle of Öland, against the Danish and Lübeck fleets;
some 800–900 men died with her. The wreck was found in 2011 north of Öland, at about 75 m, and lies in the
brackish water of the Baltic, where the shipworm cannot live: the wood is remarkably well kept. The explosion
broke the hull: the after part lies heeled over to starboard, with its lower hull, the starboard side and the
stern; the port side has fallen away outwards and lies flat beside it, its ribs uppermost; the fore part was
blown apart, except for a piece of the lower bow; bronze guns, timbers and planks lie all about.
The heading and the places of the pieces are approximate.

    python model_sources/Q31357_mars.py      -> models/item/Q31357/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, tube, crumple, fbm, sstep  # noqa: E402
from sailkit import Ship, Mast, solidify, frames, cannon, gun_carriage  # noqa: E402

rng = np.random.default_rng(1564)
BRONZE_SEA = (0.24, 0.27, 0.19)          # bronze under water: dark, greenish
TIMBER = (0.3, 0.22, 0.15)


def ship():
    return Ship(L=48.0, B=13.7, D=6.6, draft=4.6, fore=[(10.0, 2.6), (6.0, 2.2)], aft=[(17.0, 2.6), (10.0, 2.4)],
                sheer=2.4, tumble=0.14, castle_tumble=0.12, bluff=0.75, transom=0.55, bowsprit=13.0, stern_windows=5,
                masts=[Mast(0.8, 27, "square", 2), Mast(0.53, 37, "square", 3), Mast(0.2, 22, "lateen", top=False)],
                gunports=[(5.5, 0.2, 0.78, 11), (7.4, 0.14, 0.8, 12), (10.0, 0.05, 0.27, 4)],
                hull_col=(0.27, 0.2, 0.14), upper_col=(0.45, 0.33, 0.21), port_col=(0.5, 0.14, 0.09), lower_decks=[4.9])


def compact(m):
    if not len(m.f):
        return Mesh()
    u, inv = np.unique(m.f, return_inverse=True)
    return Mesh(m.v[u], inv.reshape(-1, 3), m.c[u])


def pick(m, keep):
    return compact(Mesh(m.v, m.f[keep], m.c))


def anchor(col=(0.16, 0.13, 0.11)):
    """An iron anchor of the period: a long shank, two curved arms with flukes, a ring (the wooden stock is gone)."""
    P = [cyl(0.12, 0.1, 4.2, col, seg=8)]
    for s in (-1, 1):
        pts = [(s * 1.3 * math.sin(a), 0.25 + 1.2 * (1 - math.cos(a)), 0) for a in np.linspace(0, 1.25, 6)]
        P.append(tube(pts, 0.1, col, seg=6))
        P.append(box(0.5, 0.6, 0.06, col, bottom=False).R(0, 0, s * 0.6).T(s * 1.25, 1.2, 0))
    P.append(tube([(0.35 * math.cos(a), 4.45 + 0.35 * math.sin(a), 0) for a in np.linspace(0, 6.3, 12)], 0.05, col, seg=4))
    return merge(P)


# ── The wreck ────────────────────────────────────────────────────────────────
S = ship()
L = S.L
hull = S._hull()
det, _ = S._details(False)
deck = S._decks()
stumps, _, _ = S._rig(False)
ribs = frames(S, np.arange(0.07, 0.7, 1.15 / L), col=TIMBER, size=0.3, inset=0.32, upto=0.97)


def faces_info(m, seed):
    fc = m.v[m.f].mean(1)
    t = S.tz(fc[:, 2])
    n1 = fbm(fc, 0.12, seed, 3) - 0.5
    n2 = fbm(fc, 0.6, seed + 1, 2) - 0.5
    return fc, t, n1, n2


def zones(m, seed):
    """Which piece each face of m ends in: 0 the stern section, 1 the fallen port side, 2 the bow piece, -1 gone."""
    fc, t, n1, n2 = faces_info(m, seed)
    tb = 0.6 + 0.12 * n1 + 0.04 * n2                         # where the hull was blown open
    hb = 3.4 + 3.0 * n1 + 0.8 * n2                           # the line along which the port side broke away
    z = np.full(len(fc), -1)
    main = t <= tb
    z[main] = 0
    z[main & (fc[:, 0] > 0.4) & (fc[:, 1] > hb) & (t > 0.06)] = 1
    z[(t >= 0.74 + 0.03 * n2) & (fc[:, 1] < 4.0 + 2.0 * n1 + 0.6 * n2)] = 2
    return z


zh, zd_, zr, zt_ = zones(hull, 7), zones(det, 7), zones(ribs, 7), zones(deck, 7)
# the decks: much of them fell in; what is left of the main deck has holes where the planking is gone
fcd, td, n1d, n2d = faces_info(deck, 31)
deck_keep = (zt_ == 0) & (fcd[:, 0] < 1.5 + 6 * n1d) & (fbm(fcd, 0.25, 33, 2) < 0.6)
fcs = stumps.v[stumps.f].mean(1)
stump_keep = S.tz(fcs[:, 2]) < 0.45                          # the mizzen and main stumps stay, the foremast is gone

stern_planks = solidify(pick(hull, zh == 0), 0.3, TIMBER)
port_planks = solidify(pick(hull, zh == 1), 0.3, TIMBER)
bow_planks = solidify(pick(hull, zh == 2), 0.3, TIMBER)
stern = merge([stern_planks, pick(det, zd_ == 0), pick(ribs, zr == 0), pick(deck, deck_keep), pick(stumps, stump_keep)])
port = merge([port_planks, pick(det, zd_ == 1), pick(ribs, zr == 1)])
bow = merge([bow_planks, pick(det, zd_ == 2), pick(ribs, zr == 2)])
for k, m in enumerate((stern, port, bow)):
    g = m.c.mean(1, keepdims=True)                           # 460 years in the dark: the paint gone, the wood darkened
    m.c = np.clip((m.c * 0.6 + g * 0.4) * 1.1 + np.array([0.02, 0.014, 0.0]), 0, 1)
    crumple(m, 0.25, 0.3, 40 + k)
    crumple(m, 0.06, 1.5, 50 + k)

# the stern section heels over to starboard (−x), half buried
stern.R(0, 0, math.radians(27))
stern.T(0, -stern.v[:, 1].min() - 2.2, 0)
# the port side fell outwards and lies almost flat beside it, ribs up
hx = S.half(0.35, 3.4)
port.T(-hx, -3.4, 0).R(0, 0, -math.radians(84)).R(0, math.radians(6), 0)
port.T(11.5 - port.v[:, 0].min(), -port.v[:, 1].min() - 0.5, -2.0)
# the piece of the lower bow, ahead of the break, on its side
bc = bow.v.mean(0)
bow.T(-bc[0], -bc[1], -bc[2]).R(0, 0, math.radians(-58)).R(0, math.radians(28), 0)
bow.T(3.0, -bow.v[:, 1].min() - 1.4, bc[2] + 9.0)

# ── What the explosion threw about: guns, timbers and planks, a mast, the anchors ──
deb, guns = [], []
zf = S.zt(0.62)                                               # the forward end of the stern section


def spot(spread_x, z0, spread_z):
    return rng.normal(0, spread_x), rng.uniform(z0, z0 + spread_z)


for i in range(32):                                          # bronze and iron guns, most near the break
    x, z = (spot(9, zf - 6, 30) if i < 24 else (rng.normal(-9, 3), rng.uniform(-22, zf)))
    ln = rng.uniform(2.2, 3.6)
    g = cannon(ln, rng.uniform(0.06, 0.1), bronze=i % 5 != 0)
    if i % 5:
        g.c = np.clip(g.c * 0.0 + np.array(BRONZE_SEA) * (0.8 + 0.4 * fbm(g.v, 2.0, i, 2))[:, None], 0, 1)
    g.R(rng.uniform(-0.15, 0.15), rng.uniform(0, 6.28), rng.uniform(0, 6.28)).T(x, 0.1, z)
    guns.append(g)
for i in range(5):
    x, z = spot(7, zf - 3, 20)
    c_ = gun_carriage(2.8, TIMBER)
    crumple(c_, 0.06, 1.0, 600 + i)
    deb.append(c_.R(rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28), rng.uniform(-0.4, 0.4)).T(x, 0.0, z))
for i in range(170):                                         # planks
    near = i < 130
    x, z = spot(6 if near else 11, (zf - 3) if near else -22, 22 if near else 62)
    ln = rng.uniform(1.5, 7.5)
    p_ = box(rng.uniform(0.22, 0.42), rng.uniform(0.08, 0.14), ln, TIMBER if i % 3 else S.upper_col)
    crumple(p_, 0.05, 0.6, 700 + i)
    p_.v[:, 1] += 0.15 * np.sin(p_.v[:, 2] / max(ln, 1) * math.pi) * rng.uniform(-1, 1)
    deb.append(p_.R(rng.uniform(-0.1, 0.1), rng.uniform(0, 6.28), rng.uniform(-0.25, 0.25)).T(x, 0.04, z))
for i in range(22):                                          # broken ribs and knees
    x, z = spot(10, zf - 2, 28)
    a0 = rng.uniform(0.3, 1.2); R_ = rng.uniform(2.5, 4.5)
    pts = [(R_ * math.sin(a), R_ * (1 - math.cos(a)), 0) for a in np.linspace(0, a0, 5)]
    t_ = tube(pts, 0.15, TIMBER, seg=4)
    deb.append(t_.R(math.pi / 2 - 0.2, rng.uniform(0, 6.28), 0).T(x, 0.12, z))
mast = merge([cyl(0.56, 0.48, 15.0, S.spar_col, seg=10), cyl(0.48, 0.38, 9.0, S.spar_col, seg=10).R(0, 0, 0.35).T(-0.3, 15.2, 0)])
crumple(mast, 0.08, 0.5, 900)
deb.append(mast.R(0, 0, math.pi / 2).R(0, 0.5, 0).T(-6.0, 0.4, zf + 6))
for k, (x, z, a) in enumerate(((-7.0, zf + 26, 0.9), (5.0, zf + 32, 2.4))):
    deb.append(anchor().R(0, 0, math.pi / 2 + 0.15).R(0, a, 0).T(x, 0.12, z))
debris = merge(deb)
debris.c = np.clip((debris.c * 0.6 + debris.c.mean(1, keepdims=True) * 0.4) * 1.0, 0, 1)
cannons = merge(guns)

# ── Mars as she sailed (model_built.glb) ─────────────────────────────────────
B = S.build(built=True)

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q31357" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("stern_section", stern), ("bow_section", bow), ("port_side", port), ("debris_field", debris)],
                   "plain": [("guns", cannons, "artefact")]}],
            depth=-75, kind="galleon", seed=1564, wood=True, life="baltic")
dress_built(out.with_name("model_built.glb"),
            [("hull", merge([B["hull"], B["deck"]]), "wood"), ("rig", B["spars"], "wood"), ("rigging", B["ropes"], "rope"),
             ("sails", B["sails"], "cloth"), ("guns", B["guns"], "metal")], deck_colors=[S.deck_col])
