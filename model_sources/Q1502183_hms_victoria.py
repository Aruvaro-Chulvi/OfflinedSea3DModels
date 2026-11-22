"""HMS Victoria (Q1502183) — the wreck of the British battleship that stands on end off Tripoli, Lebanon, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Victoria-class battleship completed in 1890, flagship of the Mediterranean Fleet: about 103.6 m long
between perpendiculars and 21.3 m in the beam, a low freeboard, one heavy turret forward with two 16.25-inch guns, a
single 10-inch gun aft, a central superstructure with two funnels side by side, a single military mast, a broadside
of 6-inch guns, two screws.

The wreck, after the published descriptions (Wikipedia, Advanced Diver Magazine, X-Ray Magazine, the Diving Almanac):
on 22 June 1893, during fleet manoeuvres off Tripoli, she was rammed by HMS Camperdown on an order of Vice-Admiral Sir
George Tryon and capsized and sank in about 13 minutes; 358 died, Tryon among them. The wreck was found in 2004 by the
Lebanese diver Christian Francis with Mark Ellyatt: she stands nearly vertical, her bow and the first 30 m or so of
her length driven into the mud by the weight of the forward turret and the momentum of the sinking, her stern with its
screws and rudder pointing up towards the surface. The bottom lies at about 140-150 m, the stern at about 75 m. A war
grave. Her lean and heading (not published here: 0) are approximate.

    python model_sources/Q1502183_hms_victoria.py   -> models/item/Q1502183/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1893 + 6)

L, B, D = 103.6, 21.3, 9.0
FC, TW, TT, WL = 0.0, 5.0, 1.2, 7.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.26, 0.62, 0.22, 0.45, 0.07, 0.28, 0.6, 0.3, 2.2, 5.0
RAISED = []
SCREWS = [(-4.0, L - 6.0, 2.6, 2.0), (4.0, L - 6.0, 2.6, 2.0)]
RUDDER_D = L - 2.5
BS = dict(turrets=[(22.0, 0.0, 2, 0.0), (90.0, 0.4, 1, 3.1416)], gun_len=13.0, house_w=13.0, bridge=(30.0, 44.0, 3, 15.0),
          funnels=[], masts=[(48.0, 22.0, "pole")], casemates=[50.0, 56.0, 62.0, 68.0, 74.0], secondary=[], rudders=[0.0])
HULLC = (0.1, 0.1, 0.11)
SUPC = (0.8, 0.78, 0.72)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.7, 0.58, 0.32)
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


def victoria(built):
    P = bs_part(built, gone_masts=True)
    P.append(box(14.0, 2.6, 30.0, SUPC).T(0, deck_y(60), zd(60)))                    # the central battery and house
    for s in (-1, 1):                                                               # two funnels side by side
        f = cyl(2.0, 1.95, 10.0, FUNNELC, seg=16, sx=0.85).T(s * 3.2, deck_y(58) + 2.6, zd(58))
        f.c[f.v[:, 1] > deck_y(58) + 11.6] = BLACK
        P.append(f)
    return merge(P)


w = victoria(False)
fc = w.v[w.f].mean(1); fd_ = L / 2 - fc[:, 2]
w.f = w.f[~((fd_ < 14) & (fc[:, 0] < 0) & (np.abs(fc[:, 0]) > B * 0.3) & (fbm(fc, 0.2, 61) > 0.35))]     # rammed on the starboard bow
crumple(w, 0.1, 0.12, 62)
w.R(math.radians(78), 0, math.radians(8))                                         # standing on end, bow down, leaning a little
w.T(0, -30.0 - w.v[:, 1].min(), 0)                                                # the first 30 m buried in the mud
fc = w.v[w.f].mean(1)
w.f = w.f[fc[:, 1] > -1.5]
k, inv = np.unique(w.f, return_inverse=True)
w = Mesh(w.v[k], inv.reshape(-1, 3), w.c[k])
w.T(-0.5 * (w.v[:, 0].min() + w.v[:, 0].max()), 0, -0.5 * (w.v[:, 2].min() + w.v[:, 2].max()))
debris = merge(plates(40, 0, 0, 14, 631, cols=(HULLC, SUPC, DECKC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1502183" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-145, kind="warship", seed=1893, memorial=True, drift=0.6)
ship = victoria(True); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
