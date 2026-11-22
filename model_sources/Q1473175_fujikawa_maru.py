"""
Fujikawa Maru (Q1473175) — the wreck of the Japanese aircraft transport in Chuuk Lagoon, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a cargo and passenger ship built by Mitsubishi at Nagasaki in 1938 (6,938 GRT, about 132 m), taken
up by the Imperial Japanese Navy in 1940 as an aircraft transport, with an old 6-inch gun on the bow and another
on the stern. Raised forecastle, holds forward and aft of the bridge and engine-room island, two masts.

The wreck, after the published descriptions and dive guides: sunk at anchor in Chuuk (Truk) Lagoon during
Operation Hailstone on 17 February 1944 (a torpedo on the starboard side aft, bombs). She sits upright on the
bottom at about 34 m, the deck at about 18 m and her masts rising to a few metres below the surface, all
crusted with coral; the bow gun still on the forecastle; in hold 2, the fuselages of Zero fighters and aircraft
parts. Heading (not published: 0), the length of the holds, the exact place of the cargo and of the torpedo hole
are approximate.

    python model_sources/Q1473175_fujikawa_maru.py   -> models/item/Q1473175/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1944)

L, B, D = 132.0, 18.0, 10.5
FC, TW, TT, WL = 2.4, 6.4, 1.2, 6.5
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.22, 0.7, 0.22, 0.22, 0.1, 0.5, 2.0, 1.0, 2.5, 7.0
HULLC = (0.2, 0.21, 0.22)                    # dark grey (wartime)
SUPC = (0.55, 0.56, 0.55)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.55, 0.56, 0.55), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.25, 0.25, 0.26)
HHW = 3.8
HATCHES = [(16, 25), (28, 38), (41, 50), (83, 92), (95, 104), (106, 115)]
BULKHEADS = [14, 26.5, 39.5, 51, 81, 93.5, 105, 116]
RAISED = [(0, 13), (51, 81), (117, L)]
HOUSES = [(53, 79, 0.78, 2.5, 0), (55, 75, 0.6, 2.5, 1), (54, 59, 0.62, 2.4, 2)]
FUNNELS = [(70.5, 2.2, 7.0, 0.8)]
MASTS = [(26.5, 17.0), (93.5, 16.0)]
BOATS = [(64.0, 2), (72.0, 2)]
BOAT_X = 0.3
SCREWS = [(0.0, L - 4.0, 2.4, 2.1)]
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


def old_gun(train=0.0):
    """A 6-inch gun of the 1890s on a pedestal with a small shield."""
    P = [cyl(0.7, 0.85, 1.0, GREY_D, seg=12), box(1.1, 0.9, 2.2, GREY).T(0, 1.0, 0.2), box(2.4, 1.5, 0.1, GREY).T(0, 1.0, 1.2),
         cyl(0.14, 0.11, 6.0, STEEL, seg=8).R(math.pi / 2 - 0.08, 0, 0).T(0, 1.5, 0.8)]
    return merge(P).R(0, train, 0)


def zero_fuselage(wings=False):
    """A Mitsubishi A6M Zero: the fuselage (engine cowling to tail), with or without its wing centre-section."""
    f = lathe([(0.05, -4.6), (0.3, -4.0), (0.55, -2.0), (0.62, 0.0), (0.66, 1.6), (0.6, 2.4), (0.45, 2.6)], (0.48, 0.5, 0.42), seg=10, axis="z")
    P = [f, box(0.08, 1.4, 1.3, (0.48, 0.5, 0.42)).T(0, 0.1, -4.0), box(3.2, 0.08, 1.0, (0.48, 0.5, 0.42)).T(0, 0, -4.0),
         lathe([(0.0, -0.6), (0.38, -0.3), (0.4, 0.4), (0.0, 0.7)], (0.35, 0.45, 0.5), seg=8, axis="z").T(0, 0.55, 0.2)]
    if wings:
        P.append(box(12.0, 0.18, 2.2, (0.48, 0.5, 0.42)).T(0, -0.4, 0.6))
    return merge(P)


def hold_cargo(wreck):
    j = (lambda s: rng.normal(0, s)) if wreck else (lambda s: 0.0)
    P = []
    # hold 2: Zero fuselages and aircraft parts in the lower hold and the 'tween deck
    for k, (d, x, y, w) in enumerate(((30.5, -1.6, TT + 0.7, False), (35.5, 1.5, TT + 0.7, False), (33.0, -6.0, TW + 0.7, True), (33.0, 6.0, TW + 0.7, False))):
        z = zero_fuselage(w)
        P.append(z.R(j(0.05), (0 if k % 2 else math.pi) + j(0.2), j(0.06)).T(x, y, zd(d)))
    for k in range(8):                                                       # drop tanks, wings and engine parts
        P.append(lathe([(0.05, -1.2), (0.3, -0.8), (0.33, 0.3), (0.05, 1.1)], (0.5, 0.5, 0.45), seg=8, axis="z").T(rng.uniform(-2.5, 2.5), TT + 0.35, zd(rng.uniform(29, 37))))
    P += [box(4.5, 0.15, 1.8, (0.48, 0.5, 0.42)).R(0, rng.uniform(-0.3, 0.3), 0.15).T(rng.uniform(-3, 3), TT + 1.6 + 0.2 * k, zd(rng.uniform(29, 37))) for k in range(4)]
    # holds 1 and 3: drums and crates; holds 4-6: crates, drums, bottles
    for (a, b) in [HATCHES[0], HATCHES[2], HATCHES[3], HATCHES[4], HATCHES[5]]:
        for d in np.arange(a + 0.8, b - 0.6, 1.3):
            for x in (-2.0, -0.7, 0.6, 1.9):
                if rng.random() < 0.6:
                    P.append(cyl(0.3, 0.3, 0.9, (0.3, 0.27, 0.2), seg=8).T(x + j(0.08), TT, zd(d)))
                elif rng.random() < 0.7:
                    P.append(crate(1.0, rng.uniform(0.5, 1.0), 0.9, (0.38, 0.31, 0.21)).T(x, TT, zd(d)))
    return P


# ── The wreck: upright, masts standing ───────────────────────────────────────
wreck = merge(ship_part(0, L, False, mast_left=[15.0, 13.5], funnel_state="stump",
                        houses_damage=lambda m: crumple(m, 0.3, 0.35, 61))
              + hold_cargo(True)
              + [at(old_gun(0.15), 5.5, y=deck_y(5.5) + FC), at(old_gun(math.pi + 0.3), 126.0, y=deck_y(126) + FC)])
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.13, 0.12, 11)
# the torpedo hole, starboard side aft (hold 5): plating pushed in around a dark opening
zt, yt = zd(99.0), 4.0
near = np.hypot(wreck.v[:, 2] - zt, wreck.v[:, 1] - yt) < 4.5
wsel = near & (wreck.v[:, 0] < -B * 0.3)
wreck.v[wsel, 0] += 0.9 * (1 - np.hypot(wreck.v[wsel, 2] - zt, wreck.v[wsel, 1] - yt) / 4.5)
hole = quad(-(hw(zt, yt) - 0.6), yt, zt, 5.0, 3.6, "-x", DARK)
wreck = merge([wreck, hole])
wreck.T(0, 0, -zd(L / 2)).T(0, -1.0, 0)

debris = merge(plates(30, -14, 6, 10, 100) + plates(10, 12, -30, 8, 101))
coral = merge([corals(wreck, 650, 7)])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1473175" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-34, kind="warship", seed=1944, life="tropical")
dress_built(out.with_name("model_built.glb"), [("main", merge([built_ship(), at(old_gun(0.0), 5.5, y=deck_y(5.5) + FC).T(0, 0, -zd(L / 2)),
                                                               at(old_gun(math.pi), 126.0, y=deck_y(126) + FC).T(0, 0, -zd(L / 2))]))])
