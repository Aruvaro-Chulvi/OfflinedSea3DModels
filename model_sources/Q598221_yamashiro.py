"""Yamashiro (Q598221) — the wreck of the Japanese battleship sunk in the Surigao Strait in 1944, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Fusō-class battleship completed in 1917 and rebuilt in the 1930s: about 205 m long and 33 m across her
bulges, twelve 14-inch guns in six twin turrets (two forward, two amidships, two aft), the towering pagoda bridge, a
single funnel, a tripod mainmast, casemate guns along the sides, four screws.

The wreck, after the published descriptions (Wikipedia, the R/V Petrel expedition, Pacific Wrecks, X-Ray Magazine):
in the night battle of the Surigao Strait on 25 October 1944 she was hit by torpedoes from American destroyers and by
the gunfire of the battle line, capsized and sank; only about ten of some 1,600 men survived. R/V Petrel's remotely
operated vehicle found her in late 2017 lying upside down at about 180-200 m, near her sister Fusō. The damage and the
heading (not published here: 0) are approximate.

    python model_sources/Q598221_yamashiro.py   -> models/item/Q598221/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1944 + 10)

L, B, D = 205.1, 33.1, 15.5
FC, TW, TT, WL = 0.0, 6.0, 1.2, 9.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.28, 0.7, 0.2, 0.5, 0.07, 0.3, 1.8, 0.4, 2.2, 5.0
RAISED = []
SCREWS = [(-8.0, L - 30.0, 2.5, 1.8), (-3.5, L - 22.0, 2.3, 1.8), (3.5, L - 22.0, 2.3, 1.8), (8.0, L - 30.0, 2.5, 1.8)]
RUDDER_D = L - 3.0
BS = dict(turrets=[(30.0, 0.0, 2, 0.0), (42.0, 2.8, 2, 0.0), (98.0, 0.0, 2, 3.1416), (124.0, 2.8, 2, 0.0), (168.0, 2.8, 2, 3.1416), (180.0, 0.0, 2, 3.1416)],
          gun_len=16.0, house_w=14.0, bridge=(52.0, 70.0, 3, 16.0), funnels=[(84.0, 3.6, 10.0, 0.75)],
          masts=[(140.0, 18.0, "tripod")], casemates=[46.0, 52.0, 58.0, 64.0, 70.0, 146.0, 152.0], secondary=[],
          pagoda=(60.0, 34.0), rudders=[0.0])
HULLC = (0.4, 0.41, 0.43)
SUPC = (0.46, 0.47, 0.49)
GREY, GREY_L, GREY_D = (0.42, 0.43, 0.45), (0.5, 0.51, 0.53), (0.3, 0.3, 0.32)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.42, 0.43, 0.45)
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


def holes(m, spots, seed):
    fc = m.v[m.f].mean(1); fd_ = L / 2 - fc[:, 2]
    hit = np.zeros(len(fc), bool)
    for (d0, x0, y0, r0) in spots:
        hit |= np.sqrt((fd_ - d0) ** 2 + (fc[:, 0] - x0) ** 2 + ((fc[:, 1] - y0) * 1.2) ** 2) < r0 * (0.8 + 0.4 * fbm(fc, 0.15, seed))
    m.f = m.f[~hit]
    return m


w = merge(bs_part(False, gone_masts=True))
crumple(w, 0.2, 0.08, 81)
holes(w, [(70.0, -B * 0.48, 4.0, 5.0), (120.0, -B * 0.48, 4.0, 5.0), (150.0, B * 0.48, 4.0, 4.5)], 82)   # torpedo hits (sides approximate)
capsize(w, math.radians(172), sink=1.0)                                             # upside down, the pagoda crushed under her
debris = merge(plates(90, -20, zd(60), 26, 821, cols=(HULLC, GREY_D))
               + [pagoda(20.0).R(math.pi / 2 - 0.2, 0.4, 0).T(26.0, 2.0, zd(70))])           # the top of the pagoda, broken off beside her

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q598221" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-190, kind="warship", seed=1944, memorial=True, drift=0.6)
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
