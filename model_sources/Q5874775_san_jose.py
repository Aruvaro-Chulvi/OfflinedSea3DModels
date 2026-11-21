"""
San José (Q5874775) — the Spanish galleon sunk off Cartagena de Indias in 1708, and her wreck site, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a 64-gun galleon built in 1698 in Guipúzcoa, the flagship (capitana) of the Tierra Firme treasure fleet:
about 45 m long and 13 m in the beam, two gun decks of bronze guns, the high stern of the period, three masts and a
bowsprit.

She was lost on 8 June 1708 in Wager's Action off the Rosario Islands: engaged by a British squadron under Commodore
Charles Wager, she blew up and sank with nearly all her 600 men and a cargo of gold, silver and emeralds from Peru.
Colombia announced the discovery of the wreck in 2015 at about 600 m; the Colombian Navy's four observation campaigns
of 2022 with a remotely operated vehicle showed bronze guns, cast with dolphins, lying half-buried in the mud,
intact Chinese porcelain in the remains of the hull, glass bottles, pottery and gold coins and objects. The first
objects were raised from the upper layer of the debris field in 2025. The position is kept secret: the catalog pin
is approximate. The layout of the remains is approximate.

    python model_sources/Q5874775_san_jose.py   -> models/item/Q5874775/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, fbm, sstep  # noqa: E402
from sailkit import Ship, Mast, solidify, frames, cannon  # noqa: E402

rng = np.random.default_rng(1708)
BRONZE_SEA = (0.26, 0.28, 0.2)
TIMBER = (0.25, 0.19, 0.13)
PORCELAIN = (0.82, 0.84, 0.86)
GOLD = (0.78, 0.6, 0.2)


def ship():
    return Ship(L=45.0, B=12.8, D=6.4, draft=5.4, fore=[(8.0, 2.4)], aft=[(15.0, 2.4), (8.0, 2.2)],
                sheer=2.2, tumble=0.16, castle_tumble=0.12, bluff=0.72, transom=0.55, bowsprit=13.0, stern_windows=5,
                masts=[Mast(0.8, 28, "square", 3), Mast(0.53, 36, "square", 3), Mast(0.2, 22, "lateen", top=False)],
                gunports=[(4.2, 0.12, 0.82, 13), (6.4, 0.1, 0.84, 13)], hull_col=(0.24, 0.18, 0.12), upper_col=(0.5, 0.18, 0.1),
                port_col=(0.55, 0.13, 0.08), lower_decks=[4.0], trim_col=(0.7, 0.55, 0.2))


def compact(m):
    if not len(m.f):
        return Mesh()
    u, inv = np.unique(m.f, return_inverse=True)
    return Mesh(m.v[u], inv.reshape(-1, 3), m.c[u])


def bowl():
    return lathe([(0.0, 0.0), (0.05, 0.0), (0.1, 0.02), (0.14, 0.07), (0.15, 0.08)], PORCELAIN, seg=12)


def bottle():
    return lathe([(0.0, 0.0), (0.07, 0.0), (0.08, 0.14), (0.03, 0.22), (0.025, 0.3)], (0.2, 0.3, 0.22), seg=8)


def jar():
    return lathe([(0.0, 0.0), (0.12, 0.02), (0.2, 0.2), (0.18, 0.42), (0.08, 0.5), (0.09, 0.55)], (0.55, 0.38, 0.26), seg=10)


S = ship()
L = S.L
hull = S._hull()
ribs = frames(S, np.arange(0.05, 0.92, 1.1 / L), col=TIMBER, size=0.3, inset=0.32, upto=0.55)
# what is left of the hull after the explosion: the bottom, below about the turn of the bilge, half buried
fc = hull.v[hull.f].mean(1)
t = S.tz(fc[:, 2])
cut = 2.3 + 1.2 * (fbm(fc, 0.18, 3) - 0.5)
keep = (fc[:, 1] < cut) & (t > 0.04 + 0.03 * fbm(fc, 0.3, 4)) & (t < 0.93)
lower = solidify(compact(Mesh(hull.v, hull.f[keep], hull.c)), 0.3, TIMBER)
fr = ribs.v[ribs.f].mean(1)
ribs = compact(Mesh(ribs.v, ribs.f[fr[:, 1] < 2.8 + 0.8 * (fbm(fr, 0.3, 5) - 0.5)], ribs.c))
wreck = merge([lower, ribs])
g = wreck.c.mean(1, keepdims=True)
wreck.c = np.clip((wreck.c * 0.5 + g * 0.5) * 1.0, 0, 1)
crumple(wreck, 0.2, 0.3, 6)
wreck.R(0, 0, math.radians(4))
wreck.T(0, -wreck.v[:, 1].min() - 1.3, 0)                            # deep in the mud

# the bronze guns, half-buried in the mud, in and about the hull; the cargo and the ship's things
G, F = [], []
for i in range(48):
    inside = i < 30
    x = rng.normal(0, 3.0 if inside else 9.0)
    z = rng.uniform(-L * 0.45, L * 0.45) if inside else rng.uniform(-L * 0.8, L * 0.8)
    gn = cannon(rng.uniform(2.6, 3.4), rng.uniform(0.08, 0.1), bronze=True)
    gn.c = np.clip(np.array(BRONZE_SEA) * (0.8 + 0.4 * fbm(gn.v, 2.0, i, 2))[:, None] + gn.c * 0.0, 0, 1)
    gn.R(rng.uniform(-0.2, 0.2), rng.uniform(0, 6.28), rng.uniform(0, 6.28))
    G.append(gn.T(x, -0.05, z))                                           # sunk to half their thickness
for i in range(90):
    m = [bowl, bottle, jar][i % 3]()
    m.R(rng.uniform(-1.4, 1.4) if i % 3 else 0.0, rng.uniform(0, 6.28), 0)
    F.append(m.T(rng.normal(0, 4.0), -0.02, rng.uniform(-L * 0.4, L * 0.4)))
for i in range(40):                                                     # gold: coins in small heaps, ingots
    c = cyl(0.02, 0.02, 0.01, GOLD, seg=8) if i % 4 else box(0.08, 0.04, 0.22, GOLD)
    F.append(c.R(0, rng.uniform(0, 6.28), 0).T(rng.normal(0, 2.0), 0.0, rng.uniform(-6, 10)))
for i in range(120):                                                    # planks and timbers thrown by the explosion
    ln = rng.uniform(1.5, 7.0)
    p = box(rng.uniform(0.22, 0.45), rng.uniform(0.08, 0.16), ln, TIMBER)
    crumple(p, 0.05, 0.6, 700 + i)
    F.append(p.R(rng.uniform(-0.1, 0.1), rng.uniform(0, 6.28), rng.uniform(-0.2, 0.2)).T(rng.normal(0, 11), 0.03, rng.uniform(-L, L)))
guns = merge(G)
finds = merge(F)

P = S.build(built=True)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q5874775" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", finds, "loose")], "plain": [("guns", guns, "artefact")]}],
            depth=-600, kind="galleon", seed=1708, wood=True, drift=0.8)
dress_built(out.with_name("model_built.glb"),
            [("hull", merge([P["hull"], P["deck"]]), "wood"), ("rig", P["spars"], "wood"), ("rigging", P["ropes"], "rope"),
             ("sails", P["sails"], "cloth"), ("guns", P["guns"], "metal")], deck_colors=[S.deck_col])
