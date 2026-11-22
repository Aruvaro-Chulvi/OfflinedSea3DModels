"""HMS Royal Oak (Q620472) — the wreck of the British battleship sunk in Scapa Flow in 1939, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Revenge-class battleship completed in 1916: about 190 m long, 31 m across her anti-torpedo bulges, eight
15-inch guns in four twin turrets (two forward, two aft, the inner ones superfiring), twelve 6-inch guns in
casemates along the sides, a tripod foremast over the bridge, a single funnel, a pole mainmast, four screws.

The wreck, after the published descriptions and the ROYAL OAK 80 survey (2018-2019) and the ADUS multibeam survey:
torpedoed at anchor by U-47 on 14 October 1939 (835 killed), she rolled over to starboard and sank in minutes. She
lies almost upside down in Scapa Bay in about 30-33 m, the upperworks crushed into the bottom under her, the bilge
and keel uppermost, with the torpedo holes in her starboard side amidships and the bow damaged by the first hit.
She is a designated war grave: diving is prohibited, and oil still seeps from her. The exact angle and heading (not
published here: 0) are approximate.

    python model_sources/Q620472_hms_royal_oak.py   -> models/item/Q620472/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1939)

L, B, D = 190.0, 30.0, 15.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.27, 0.7, 0.2, 0.5, 0.07, 0.3, 1.6, 0.4, 2.2, 5.0
RAISED = [(0, 14)]
SCREWS = [(-7.0, L - 30.0, 2.4, 1.7), (-3.0, L - 22.0, 2.2, 1.7), (3.0, L - 22.0, 2.2, 1.7), (7.0, L - 30.0, 2.4, 1.7)]
RUDDER_D = L - 3.0
BS = dict(turrets=[(36.0, 0.0, 2, 0.0), (50.0, 2.8, 2, 0.0), (140.0, 2.8, 2, 3.1416), (154.0, 0.0, 2, 3.1416)],
          gun_len=16.5, house_w=13.0, bridge=(58.0, 74.0, 4, 14.0), funnels=[(88.0, 3.4, 10.0, 0.75)],
          masts=[(66.0, 24.0, "tripod"), (112.0, 17.0, "pole")], casemates=[56.0, 62.0, 68.0, 74.0, 80.0, 86.0], secondary=[],
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


SCREWS_BUILT = SCREWS
wreck = merge(bs_part(False))
crumple(wreck, 0.2, 0.08, 39)
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
torp = np.zeros(len(fc), bool)
for d0, r0 in ((82.0, 5.5), (96.0, 5.0), (108.0, 4.5), (12.0, 3.0)):       # three hits amidships, the first at the bow
    torp |= (fc[:, 0] < -B * 0.3) & (np.hypot(fd_ - d0, (fc[:, 1] - 4.0) * 1.3) < r0 * (0.8 + 0.4 * fbm(fc, 0.15, 40)))
wreck.f = wreck.f[~torp]
capsize(wreck, math.radians(165), sink=0.6)                                     # rolled over to starboard: almost keel up
debris = merge(plates(80, -24, zd(90), 22, 139, cols=(HULLC, GREY_D)) + plates(30, 22, zd(40), 14, 140, cols=(HULLC,)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q620472" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-31, kind="warship", seed=1939, memorial=True)
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
