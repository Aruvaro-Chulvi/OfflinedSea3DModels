"""
USAT Liberty (Q2468064) — the wreck of the US Army cargo ship at Tulamben, Bali, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a steel cargo ship built in 1918 at Kearny, New Jersey (about 125 m), in the US Army transport
service in 1942, with a gun on the bow and another on the stern. Raised forecastle, bridge and engine island
amidships with a tall funnel, poop, holds fore and aft, two masts.

The wreck, after the published descriptions and dive guides: torpedoed by a Japanese submarine in the Lombok
Strait in January 1942, she was beached at Tulamben; the eruption of Mount Agung in 1963 pushed her off the
beach. She lies on her starboard side on the black-sand slope a few tens of metres from the shore, bow to the
west, stern to the east, from about 5 m to about 30 m deep; the hull has broken and its middle collapsed; both
guns are still in place (the bow gun in about 28 m); the propeller was salvaged. She is one of the most dived
wrecks in the world, covered in corals and sponges. On Offlined's flat seabed she lies at the mean depth of the
slope. The degree of collapse, the colours and the layout of the superstructure are approximate.

    python model_sources/Q2468064_usat_liberty.py   -> models/item/Q2468064/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1942)

L, B, D = 125.4, 17.1, 9.4
FC, TW, TT, WL = 2.4, 5.8, 1.1, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.2, 0.7, 0.22, 0.22, 0.1, 0.5, 2.0, 1.0, 2.5, 7.0
HULLC = (0.32, 0.34, 0.35)                   # grey
SUPC = (0.5, 0.52, 0.52)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 3.5
HATCHES = [(15, 25), (28, 38), (41, 49), (78, 88), (91, 101), (104, 111)]
BULKHEADS = [13, 26.5, 39.5, 50.5, 76, 89.5, 102.5, 113]
RAISED = [(0, 12), (50.5, 76), (113, L)]
HOUSES = [(52, 74, 0.72, 2.5, 0), (53, 64, 0.6, 2.5, 1), (53, 57.5, 0.6, 2.4, 2)]
FUNNELS = [(66.5, 1.7, 9.0, 0.85)]
MASTS = [(26.5, 16.0), (89.5, 15.0)]
BOATS = [(60.0, 1), (70.0, 1)]
BOAT_X = 0.32
SCREWS = [(0.0, L - 4.0, 2.3, 2.0)]
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


def deck_gun(train=0.0):
    P = [cyl(0.6, 0.75, 1.0, GREY_D, seg=12), box(1.0, 0.8, 2.0, GREY).T(0, 1.0, 0.2), box(2.2, 1.4, 0.1, GREY).T(0, 0.9, 1.1),
         cyl(0.11, 0.09, 4.5, STEEL, seg=8).R(math.pi / 2 - 0.08, 0, 0).T(0, 1.4, 0.8)]
    return merge(P).R(0, train, 0)


SCREWS = []                                                                   # the propeller was salvaged
LIST = math.radians(85)                                                      # on her starboard side (−X)


def roll(m):
    return m.R(0, 0, LIST)


wreck = merge(ship_part(0, L, False, mast_left=[0.0, 0.0], funnel_state="gone",
                        houses_damage=lambda m: crumple(m, 0.5, 0.3, 61))
              + [at(deck_gun(0.0), 5.5, y=deck_y(5.5) + FC), at(deck_gun(math.pi), 120.0, y=deck_y(120) + FC)])
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.18, 0.12, 11)
mid = sstep(36, 52, dfb) * sstep(92, 74, dfb)                                # the middle broken and collapsed
wreck.v[:, 0] *= 1 + 0.25 * mid * sstep(3, 10, wreck.v[:, 1])
wreck.v[:, 1] = wreck.v[:, 1] * (1 - 0.45 * mid)
crumple(wreck, 1.6, 0.25, 12, mid)
gone = (fbm(wreck.v[wreck.f].mean(1), 0.12, 13) > 0.6) & (mid[wreck.f].mean(1) > 0.5)
wreck.f = wreck.f[~gone]                                                     # plating fallen away, holes into the holds
roll(wreck)
LIFT = -np.percentile(wreck.v[:, 1], 0.5) - 1.0
wreck.T(0, LIFT, 0)
XC = -0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max())
wreck.T(XC, 0, 0).T(0, 0, -zd(L / 2))
onwreck = make_onwreck(roll, XC, LIFT)
side = onwreck(0, D + FC + 4, 60)[0]
r = np.random.default_rng(71)
deb = plates(50, side - 6, 0, 18, 100)
fun = cyl(1.7, 1.65, 8.0, FUNNELC, seg=18, sx=0.85, caps=False); crumple(fun, 0.4, 0.35, 51); fun.v[:, 0] *= 0.7
deb.append(fun.R(0, 0, math.pi / 2).R(0, -0.4, 0).T(side - 4, 1.0, zd(66.5) - zd(L / 2)))
for k, (d, ln) in enumerate(((26.5, 14.0), (89.5, 13.0))):
    m = cyl(0.36, 0.2, ln, GREY, seg=8, caps=False); crumple(m, 0.12, 0.4, 60 + k)
    deb.append(m.R(0, 0, math.pi / 2 - 0.05).R(0, 0.3 * (1 - 2 * k), 0).T(side - 2.0, 0.3, zd(d) - zd(L / 2)))
debris = merge(deb)
coral = merge([corals(wreck, 800, 7, size=(0.2, 0.8)), corals(debris, 60, 8)])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q2468064" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-18, kind="cargo", seed=1942, life="tropical")
SCREWS = [(0.0, L - 4.0, 2.3, 2.0)]
dress_built(out.with_name("model_built.glb"), [("main", merge([built_ship(), at(deck_gun(0.0), 5.5, y=deck_y(5.5) + FC).T(0, 0, -zd(L / 2)),
                                                               at(deck_gun(math.pi), 120.0, y=deck_y(120) + FC).T(0, 0, -zd(L / 2))]))])
