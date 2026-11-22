"""
SS Carnatic (Q1043906) — the wreck of the P&O steamer on Sha'ab Abu Nuhas, Red Sea, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: an iron-framed P&O passenger and mail steamer built by Samuda Brothers in 1862, about 90 m long, beam
11.6 m, with a single compound engine and a sailing rig (two masts), clipper bow with bowsprit, counter stern,
one funnel, wooden decks on iron beams.

The wreck, after the published descriptions and dive guides: she ran onto Sha'ab Abu Nuhas on 12 September 1869
and broke into two, 31 people died; the two halves settled almost back together. She lies on her port side in
18-27 m; the wooden decks have rotted away, leaving the iron frames and deck beams open like a ribcage, crusted
with soft and hard corals; wine and soda-water bottles still lie in the stern. On Offlined's flat seabed she lies
at the mean depth. The heading (not published: 0), the place of the break and the rig are approximate.

    python model_sources/Q1043906_carnatic.py   -> models/item/Q1043906/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, sstep, fbm, yaw_quat  # noqa: E402
from shipkit import install  # noqa: E402

rng = np.random.default_rng(1869)

L, B, D = 89.9, 11.6, 7.4
FC, TW, TT, WL = 2.0, 4.6, 1.0, 4.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.28, 0.75, 0.3, 0.3, 0.08, 0.4, 1.6, 0.9, 2.2, 5.0
HULLC = (0.09, 0.09, 0.1)                    # black
SUPC = (0.85, 0.84, 0.8)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 2.4
HATCHES = []
BULKHEADS = [44.0]
RAISED = []
HOUSES = [(36, 52, 0.55, 2.3, 0)]
FUNNELS = [(46.0, 1.4, 8.0, 0.9)]
MASTS = [(24.0, 24.0), (62.0, 22.0)]
BOATS = [(40.0, 1), (50.0, 1)]
BOAT_X = 0.28
SCREWS = [(0.0, L - 3.2, 1.9, 1.7)]
RUDDER_D = L - 1.2
ANCHORS_GONE = ()


# ── Fittings ─────────────────────────────────────────────────────────────────


# ── Cargo ────────────────────────────────────────────────────────────────────


# ── The ship, piece by piece (driven by the description above) ───────────────
#   RAISED: [(d0, d1)] forecastle / islands / poop above the main deck, FC high
#   HOUSES: [(d0, d1, width fraction of B, height, level)] deck houses on the raised decks (level 0, 1, 2…)
#   BRIDGE_FRONT: d of the wheelhouse windows (on the top house) or None
#   FUNNELS: [(d, radius, height, squash)];  MASTS: [(d, height)];  BOATS: [(d, level)]
#   SCREWS: [(x, d, y, radius)];  RUDDER_D
LEVEL_H = 2.5


# ── Placing the wreck and things about it ────────────────────────────────────
CORAL_COLS = [(0.55, 0.48, 0.32), (0.62, 0.56, 0.38), (0.5, 0.42, 0.5), (0.72, 0.5, 0.55), (0.45, 0.5, 0.33), (0.75, 0.68, 0.45),
              (0.8, 0.45, 0.3), (0.42, 0.36, 0.55)]


install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship")


FC = 0.0                                                                      # flush deck: houses stand on the main deck


def beams_and_frames(d0, d1, deck_left):
    """The iron deck beams across the hull and the frames up the sides, where the wooden deck has rotted away."""
    out = []
    for d in np.arange(d0 + 0.6, d1 - 0.3, 1.2):
        y = deck_y(d); w = half(d) - 0.1
        out.append(box(2 * w, 0.22, 0.14, (0.25, 0.2, 0.17)).T(0, y - 0.25, zd(d)))
        if rng.random() < deck_left:
            out.append(box(2 * w, 0.08, 1.1, (0.45, 0.36, 0.25)).T(0, y - 0.03, zd(d)))
    return out


def bowsprit(built):
    p = [tube([(0, deck_y(0) + 0.6, zd(0) + 0.5), (0, deck_y(0) + 2.6, zd(0) + 10.0)], 0.25, (0.4, 0.33, 0.25), seg=6)]
    return p if built else []


def hull_part(d0, d1, built):
    P = [hull(d0, d1)]
    if not built:
        P += beams_and_frames(d0, d1, 0.12)
    return P


def ship_part2(d0, d1, built):
    P = hull_part(d0, d1, built)
    sp = ship_part(d0, d1, built, mast_left=[0.0, 0.0], funnel_state="gone")
    P += sp[1:] if built else sp[2:3] + sp[4:]                           # the wreck: no wooden deck, no 'tween deck
    if d0 == 0:
        P += bowsprit(built)
    return P


BRK = 44.0
LIST = math.radians(85)


def roll(m):
    return m.R(0, 0, -LIST)


fore = merge([m for m in ship_part2(0, BRK, False) if m.tris])
aft = merge([m for m in ship_part2(BRK, L, False) if m.tris])
for k, m in enumerate((fore, aft)):
    crumple(m, 0.12, 0.12, 11 + k)
    dd = L / 2 - m.v[:, 2]
    crumple(m, 1.2, 0.3, 21 + k, sstep(BRK + (-5 if k == 0 else 5), BRK, dd))
roll(fore); roll(aft)
LIFT = -min(np.percentile(m.v[:, 1], 0.5) for m in (fore, aft)) - 0.9
XC = -0.5 * (fore.v[:, 0].min() + fore.v[:, 0].max())
for m in (fore, aft):
    m.T(XC, LIFT, 0)
zc = zd(BRK)
aft.T(0, 0, -zc).R(0, 0.06, 0).T(0.8, 0, zc - 1.5)                           # the halves settled almost back together
onwreck = make_onwreck(roll, XC, LIFT)
side = onwreck(0, D + 3, 50)[0]
import abyss  # noqa: E402
bott = []
for k in range(70):
    bt = abyss.bottle(k % 3 == 0)
    bott.append(bt.R(math.pi / 2, rng.uniform(0, 6.28), 0).T(side + rng.normal(2, 1.5), 0.05, zd(rng.uniform(70, 86))))
bottles = merge(bott)
deb = plates(16, side + 3, zd(BRK), 6, 100, cols=((0.4, 0.33, 0.25), HULLC))
for k, (d, ln) in enumerate(((24.0, 20.0), (62.0, 18.0))):
    base = onwreck(0, deck_y(d), d)
    m = cyl(0.34, 0.2, ln, (0.4, 0.33, 0.25), seg=8, caps=False); crumple(m, 0.1, 0.4, 60 + k)
    deb.append(m.R(0, 0, -math.pi / 2 + 0.05).R(0, 0.25 * (1 - 2 * k), 0).T(base[0] + 1.0, 0.3, base[2]))
debris = merge(deb)
coral = merge([corals(fore, 380, 7, size=(0.15, 0.6)), corals(aft, 420, 8, size=(0.15, 0.6))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1043906" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", fore), ("stern_section", aft), ("debris_field", debris, "loose")], "mud": [],
                   "plain": [("bottles", bottles, "glass"), ("corals", coral, "life")]}],
            depth=-24, kind="liner", seed=1869, life="tropical")
ship = merge(ship_part2(0, L, True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
