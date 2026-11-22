"""HMS Hampshire (Q1060785) — the wreck of the British armoured cruiser lost off Orkney in 1916, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Devonshire-class armoured cruiser completed in 1905: about 144 m long, beam 20.9 m, four 7.5-inch guns
in single turrets (one forward, one aft and two wing turrets abreast the bridge), six 6-inch guns in casemates, four
funnels, two pole masts, two screws.

The wreck, after the published descriptions and the HMS Hampshire 100 ROV survey (2016) and the 2023 dive survey:
she struck a mine laid by U-75 off Marwick Head on 5 June 1916, carrying Lord Kitchener to Russia; 737 died. She
capsized as she sank and lies upside down in about 60-70 m, surrounded by a large debris field; the forward part
torn by the mine, the stern shattered, the rudder lying on the seabed, the port bronze propeller still in place
(the other was taken by illegal salvors between 1977 and 1983). A war grave. The exact angle and heading are
approximate.

    python model_sources/Q1060785_hms_hampshire.py   -> models/item/Q1060785/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1916)

L, B, D = 144.3, 20.9, 12.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.7, 0.22, 0.5, 0.07, 0.3, 1.8, 0.4, 2.4, 5.5
RAISED = [(0, 30)]
SCREWS = [(-3.6, L - 14.0, 2.0, 1.9), (3.6, L - 14.0, 2.0, 1.9)]
RUDDER_D = L - 2.5
BS = dict(turrets=[(16.0, 0.0, 1, 0.0), (128.0, 0.0, 1, 3.1416)],
          gun_len=9.5, house_w=10.0, bridge=(24.0, 32.0, 3, 10.0), funnels=[(46.0, 1.7, 11.0, 1.1), (58.0, 1.7, 11.0, 1.1), (70.0, 1.7, 11.0, 1.1), (82.0, 1.7, 11.0, 1.1)],
          masts=[(36.0, 24.0, "pole"), (104.0, 20.0, "pole")], casemates=[44.0, 64.0, 96.0], secondary=[(34.0, 1), (34.0, -1)],
          rudders=[0.0])
HULLC = (0.4, 0.42, 0.44)
SUPC = (0.5, 0.52, 0.53)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 3.5
HATCHES = []
BULKHEADS = []
HOUSES = []
FUNNELS = []
MASTS = []
BOATS = []
BOAT_X = 0.32
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


# ── A battleship (driven by BS below) ───────────────────────────────────────
#   BS = dict(turrets=[(d, raised, guns, train)], gun_len, house_w, bridge=(d0, d1, levels, width), funnels=[(d, r, h, sx)],
#             masts=[(d, h, kind)], casemates=[d…] (both sides), secondary=[(d, side)], pagoda=None|(d, height))
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship bs_turret bs_part pagoda capsize")


SCREWS = [(3.6, L - 14.0, 2.0, 1.9)]                                          # the port propeller only (the other was taken)
BUILT_BS = BS
BS = dict(BS, rudders=[])                                                    # the rudder lies on the seabed
wreck = merge(bs_part(False))
crumple(wreck, 0.2, 0.08, 16)
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
mine = (fd_ > 12) & (fd_ < 34) & (fc[:, 0] > B * 0.1) & (fc[:, 1] < 7.0) & (fbm(fc, 0.12, 17) > 0.35)        # the mine: forward, port side, low
stern = (fd_ > L - 16) & (fbm(fc, 0.1, 18) > 0.3)                                                            # the shattered stern
wreck.f = wreck.f[~(mine | stern)]
capsize(wreck, math.radians(176), sink=0.8)
rud = merge(rudder(RUDDER_D, 0.8)); rud.v -= rud.v.mean(0); rud.R(math.pi / 2, 0.6, 0); rud.T(-6.0, -rud.v[:, 1].min(), zd(L + 8))
debris = merge([rud] + plates(110, 0, zd(L - 8), 18, 161, cols=(HULLC, GREY_D)) + plates(60, 12, zd(22), 14, 162, cols=(HULLC,)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1060785" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-62, kind="warship", seed=1916, memorial=True)
BS = BUILT_BS
SCREWS = [(-3.6, L - 14.0, 2.0, 1.9), (3.6, L - 14.0, 2.0, 1.9)]
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
