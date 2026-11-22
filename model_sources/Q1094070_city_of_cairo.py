"""SS City of Cairo (Q1094070) — the wreck of the Ellerman liner sunk in the South Atlantic in 1942, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: an Ellerman Lines passenger-cargo steamer built in 1915: about 137 m long and 16.8 m in the beam, a
forecastle, two hatches forward, a long midship superstructure with a single funnel, two hatches aft, a poop, two
masts, one screw.

The wreck, after the published descriptions (Wikipedia, Deep Ocean Search, Guinness World Records, Gard): on 6
November 1942, sailing from Bombay to England with passengers and 100 tonnes of silver coin, she was torpedoed by
U-68 some 480 miles south of St Helena; most aboard got away in the boats, but over the following weeks of drifting
104 died. Deep Ocean Search found the wreck in 2011 at 5,150 m on a slope in the foothills of the Mid-Atlantic Ridge:
broken in two and deeply buried in silt, with metres of mud heaped on parts of her. In 2013 they raised most of the
silver from her, the deepest salvage of cargo ever made. The break, the burial and the heading (not published here:
0) are approximate, and so is the catalog pin.

    python model_sources/Q1094070_city_of_cairo.py   -> models/item/Q1094070/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1942 + 11)

L, B, D = 137.0, 16.8, 10.6
FC, TW, TT, WL = 2.4, 7.0, 1.3, 7.8
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.24, 0.75, 0.2, 0.42, 0.07, 0.3, 1.6, 0.6, 2.6, 7.0
HULLC = (0.12, 0.12, 0.13)
SUPC = (0.75, 0.73, 0.67)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.7, 0.55, 0.3)
HHW = 3.0
HATCHES = [(16, 22), (28, 35), (106, 112), (118, 124)]
BULKHEADS = [12, 40, 100, 128]
RAISED = [(0, 12), (40, 100), (128, L)]
HOUSES = [(43, 96, 0.78, 2.5, 0), (46, 90, 0.7, 2.5, 1), (48, 56, 0.5, 2.4, 2)]
FUNNELS = [(72.0, 2.6, 12.0, 0.82)]
MASTS = [(26.0, 22.0), (114.0, 20.0)]
BOATS = [(d, 1) for d in (58.0, 64.0, 82.0, 88.0)]
BOAT_X = 0.29
SCREWS = [(0.0, L - 5.5, 2.4, 2.2)]
RUDDER_D = L - 2.2
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


# ── Damage helpers ───────────────────────────────────────────────────────────
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship collapse holes lay boiler center_x")


fore = merge(ship_part(0, 66, False, mast_left=[0.0, 0.0], funnel_state="gone", davits=False))
aft = merge(ship_part(67, L, False, mast_left=[0.0, 0.0], funnel_state="gone", davits=False))
for k, m in enumerate((fore, aft)):
    collapse(m, D + FC + 0.5, 0.35, 61 + k)
holes(aft, [(104.0, -B * 0.49, 3.5, 4.0)], 63)                                    # the torpedo, aft
lay(fore, math.radians(10), 64, dz=6.0, yaw=math.radians(-18), sink=3.5)           # broken in two, both halves deep in the silt
lay(aft, math.radians(-6), 65, dz=-4.0, yaw=math.radians(9), sink=3.0)
center_x(aft, fore)
debris = merge(plates(40, 6, zd(66), 12, 651, cols=(HULLC, SUPC, DECKC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1094070" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", fore), ("stern_section", aft), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-5150, kind="liner", seed=1942, memorial=True, drift=1.6)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())], deck_colors=[DECKC])
