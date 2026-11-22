"""
HMS Hermes (Q596282) — the wreck of the British aircraft carrier off Batticaloa, Sri Lanka, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the first ship designed from the keel as an aircraft carrier, completed in 1924: about 183 m long,
beam 21.4 m, an enclosed bow up to the flight deck, a single hangar, two lifts, a large island on the starboard
side with a tripod mast and the funnel, six 5.5-inch guns in sponsons along the sides, twin screws.

The wreck, after the published descriptions and dive reports: caught without aircraft by Japanese carrier dive
bombers on 9 April 1942, she was hit by some forty bombs and sank within minutes; 307 of her men died. She lies
on her port side in about 40-53 m some 8 nautical miles off Batticaloa, largely intact, her island ("control
tower") nearly whole; a technical dive. The list, the heading (not published: 0), the bomb damage in detail and
the colours are approximate. She is a war grave.

    python model_sources/Q596282_hms_hermes.py   -> models/item/Q596282/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1942 + 4)

L, B, D = 182.9, 21.4, 10.5
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.45, 0.07, 0.3, 1.2, 0.3, 2.3, 6.0
HULLC = (0.45, 0.47, 0.49)                   # grey
SUPC = (0.5, 0.52, 0.52)
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
SCREWS = [(-3.6, L - 14.0, 2.4, 2.0), (3.6, L - 14.0, 2.4, 2.0)]
RUDDER_D = L - 4.0
ANCHORS_GONE = ()
FDC = (0.3, 0.29, 0.27)
CARRIER = dict(hangar=6.0, fd=(4.0, L - 6.0), fdw=27.4, island=(68.0, 98.0, 6.0, 4), funnel=(84.0, 96.0, 7.0, 4.0), mast=(76.0, 14.0),
               lifts=[(40.0, 14.0, 10.0, 0.0), (140.0, 12.0, 10.0, 0.0)],
               sponsons=[(d, s, "single") for d in (50.0, 110.0, 130.0) for s in (-1, 1)])


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


# ── An aircraft carrier: hangar, flight deck, island (driven by CARRIER below) ──
#   CARRIER = dict(hangar=height of the hangar above the hull deck, fd=(d0, d1), fdw=flight deck width,
#                  island=(d0, d1, width, levels), funnel=(d0, d1, height, width) or None, mast=(d, height),
#                  lifts=[(d, length, width, x)], sponsons=[(d, side, kind)], open_hangar=bool)
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship fd_y carrier_part")


LIST = math.radians(88)                                                       # on her port side (+X)


def roll(m):
    return m.R(0, 0, -LIST)


wreck = merge(carrier_part(0, L, False, holes=0.18, seed=42))
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.2, 0.1, 11)
crumple(wreck, 0.6, 0.15, 12, (wreck.v[:, 1] > D + 3).astype(float))         # the flight deck torn and buckled by the bombs
roll(wreck)
LIFT = -np.percentile(wreck.v[:, 1], 0.5) - 1.2
wreck.T(0, LIFT, 0)
XC = -0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max())
wreck.T(XC, 0, 0)
onwreck = make_onwreck(roll, XC, LIFT)
side = onwreck(0, D + 6.5, 90)[0]
debris = merge(plates(60, side + 8, zd(100), 25, 100, cols=(GREY, FDC, GREY_D)))
coral = merge([corals(wreck, 380, 7, size=(0.12, 0.45))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q596282" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-52, kind="warship", seed=1942, memorial=True, life="tropical")
ship = merge(carrier_part(0, L, True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
