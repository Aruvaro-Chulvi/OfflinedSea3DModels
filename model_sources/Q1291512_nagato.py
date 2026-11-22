"""
Nagato (Q1291512) — the wreck of the Japanese battleship at Bikini Atoll, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the lead ship of her class, completed in 1920 and rebuilt in the 1930s: about 225 m long, beam 34.6 m,
eight 16-inch guns in four twin turrets (A and B forward, X and Y aft), the tall pagoda bridge, one funnel, four
screws; Yamamoto's flagship at Pearl Harbor. Surrendered in 1945; her secondary guns had already been taken ashore.

The wreck, after the published descriptions: a target ship of Operation Crossroads, she survived both tests; after
Baker a small list to starboard grew over five days and she capsized and sank on the night of 29/30 July 1946.
She lies upside down in about 52 m (170 ft) of the lagoon, her four great propellers, her most prominent feature,
at about 33.5 m (110 ft); her bridge can be reached at about 46 m (150 ft). Here the pagoda has broken off and lies
on its side beside the hull. The heading (not published: 0) and the state of the turrets are approximate.

    python model_sources/Q1291512_nagato.py   -> models/item/Q1291512/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1946)

L, B, D = 224.9, 34.6, 14.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.55, 0.07, 0.3, 2.0, 0.4, 2.5, 6.0
HULLC = (0.42, 0.44, 0.46)
SUPC = (0.5, 0.52, 0.53)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 3.5
HATCHES = []
BULKHEADS = []
RAISED = [(0, 12), (50.5, 76), (113, L)]
HOUSES = [(52, 74, 0.72, 2.5, 0), (53, 64, 0.6, 2.5, 1), (53, 57.5, 0.6, 2.4, 2)]
FUNNELS = [(66.5, 1.7, 9.0, 0.85)]
MASTS = [(26.5, 16.0), (89.5, 15.0)]
BOATS = [(60.0, 1), (70.0, 1)]
BOAT_X = 0.32
SCREWS = [(0.0, L - 4.0, 2.3, 2.0)]
RUDDER_D = L - 1.2
ANCHORS_GONE = ()
BS = dict(turrets=[(44.0, 0.0, 2, 0.0), (58.0, 2.8, 2, 0.0), (168.0, 2.8, 2, 3.1416), (183.0, 0.0, 2, 3.1416)], gun_len=18.0, house_w=14.0,
          bridge=(68.0, 84.0, 3, 16.0), funnels=[(100.0, 3.4, 10.0, 0.7)],
          masts=[(140.0, 14.0, "tripod")], secondary=[], pagoda=(76.0, 26.0), rudders=[0.0])


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


# ── A battleship (driven by BS below) ───────────────────────────────────────
#   BS = dict(turrets=[(d, raised, guns, train)], gun_len, house_w, bridge=(d0, d1, levels, width), funnels=[(d, r, h, sx)],
#             masts=[(d, h, kind)], casemates=[d…] (both sides), secondary=[(d, side)], pagoda=None|(d, height))
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship bs_turret bs_part pagoda capsize")


SCREWS = [(-3.8, L - 16.0, 2.6, 2.3), (3.8, L - 16.0, 2.6, 2.3), (-9.5, L - 34.0, 2.8, 2.3), (9.5, L - 34.0, 2.8, 2.3)]
RUDDER_D = L - 5.0
lv = BS["bridge"][2]
BSW = dict(BS); BSW["pagoda"] = None
_bs = BS
BS = BSW
wreck = merge(bs_part(False))
BS = _bs
crumple(wreck, 0.2, 0.08, 11)
capsize(wreck, math.radians(178), sink=0.6)
# the pagoda, broken off, on its side beside the hull
pg = pagoda(26.0); crumple(pg, 0.3, 0.3, 21)
pg.R(0, 0, -(math.pi / 2 - 0.1)).R(0, -0.25, 0).T(B / 2 + 3.0, 4.0, zd(76.0))
debris = merge([pg] + plates(40, 22, zd(90), 18, 100, cols=(HULLC, GREY_D)))
coral = merge([corals(wreck, 420, 7, size=(0.12, 0.45)), corals(pg, 60, 8)])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1291512" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris)], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-52, kind="warship", seed=1946, life="tropical")
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
