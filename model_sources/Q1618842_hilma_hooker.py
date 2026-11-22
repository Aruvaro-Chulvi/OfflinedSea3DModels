"""Hilma Hooker (Q1618842) — the wreck of the small cargo ship that sank off Bonaire in 1984, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a small cargo ship built in the Netherlands in 1951, renamed several times: about 72 m long and 11 m in
the beam, a forecastle, two holds forward with their derrick masts, the bridge and engine house aft with a short
funnel, one screw.

The wreck, after the published descriptions (Wikipedia, the Dutch Cultural Heritage Agency, Scuba Diving, Infobonaire):
in 1984 she put into Kralendijk with engine trouble; her cargo hid some 11 tonnes of marijuana, her owners never came
for her and, moved to an anchorage off Angel City, she took on water and on the morning of 12 September 1984 rolled
over on her starboard side and sank in about two minutes. She lies on her starboard side on a sandy plateau at about
30 m between two coral reefs, the bow to the south and the stern, with its rudder and propeller, to the north; her
port side is at about 18 m. A favourite dive of the Bonaire National Marine Park.

    python model_sources/Q1618842_hilma_hooker.py   -> models/item/Q1618842/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1984 + 9)

L, B, D = 72.0, 11.0, 6.4
FC, TW, TT, WL = 2.2, 4.4, 1.0, 4.8
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.24, 0.75, 0.2, 0.45, 0.07, 0.3, 1.2, 0.5, 2.6, 7.0
HULLC = (0.18, 0.24, 0.3)
SUPC = (0.8, 0.79, 0.74)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.8, 0.79, 0.74)
HHW = 2.6
HATCHES = [(10, 22), (28, 44)]
BULKHEADS = [8, 52]
RAISED = [(0, 8), (52, L)]
HOUSES = [(53, 68, 0.8, 2.4, 0), (54, 64, 0.7, 2.4, 1), (54, 58, 0.6, 2.3, 2)]
FUNNELS = [(63.0, 1.4, 3.5, 0.8)]
MASTS = [(24.0, 13.0), (48.0, 12.0)]
BOATS = [(62.0, 1)]
BOAT_X = 0.36
SCREWS = [(0.0, L - 3.0, 1.8, 1.4)]
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


# ── Damage helpers ───────────────────────────────────────────────────────────
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship collapse holes lay boiler center_x")


wreck = merge(ship_part(0, L, False, mast_left=[13.0, 12.0], funnel_state="stand", davits=True))
lay(wreck, math.radians(90), 51, sink=0.4)                                         # on her starboard side
crumple(wreck, 0.04, 0.12, 52)
center_x(wreck)
debris = merge(plates(10, -8, zd(30), 6, 521, cols=(HULLC, SUPC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1618842" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-30, kind="cargo", seed=1984)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())])
