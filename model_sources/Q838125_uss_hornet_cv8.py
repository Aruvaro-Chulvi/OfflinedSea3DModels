"""USS Hornet (CV-8) (Q838125) — the wreck of the US aircraft carrier lost at the Battle of the Santa Cruz Islands, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the last of the Yorktown class, commissioned in 1941: about 247 m long, 25.4 m in the beam at the
waterline under a flight deck 33 m wide, the open hangar, three lifts, the island on the starboard side, eight single
5-inch guns at the corners of the flight deck and anti-aircraft guns along its edges, four screws. She launched the
Doolittle Raid on Tokyo and fought at Midway.

The wreck, after the published descriptions of the expedition of the research vessel Petrel (Vulcan, Paul Allen),
which found her in January 2019 near the Solomon Islands: crippled by bombs, torpedoes and crashing aircraft on
26 October 1942, then hit by the torpedoes and some 400 shells of US destroyers trying to scuttle her, she was sunk
by Japanese destroyers' torpedoes; 140 men died. She rests upright on the floor of the South Pacific at about
5,330-5,400 m, largely intact, identified by her hull number; anti-aircraft guns are clearly visible, and an
aircraft tug still stands upright on her hangar deck. The location is kept secret; the damage, list and heading
are approximate.

    python model_sources/Q838125_uss_hornet_cv8.py   -> models/item/Q838125/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1942 + 8)

L, B, D = 246.7, 25.4, 14.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 7.9
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.5, 0.07, 0.3, 1.2, 0.3, 2.3, 6.0
HULLC = (0.38, 0.4, 0.43)                    # navy blue-grey
SUPC = (0.42, 0.44, 0.47)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.45), (0.5, 0.52, 0.54), (0.3, 0.3, 0.32)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 3.5
HATCHES = []
BULKHEADS = []
RAISED = []
HOUSES = []
FUNNELS = []
MASTS = []
BOATS = []
BOAT_X = 0.32
SCREWS = [(-8.0, L - 32.0, 2.6, 2.0), (-3.0, L - 24.0, 2.4, 2.0), (3.0, L - 24.0, 2.4, 2.0), (8.0, L - 32.0, 2.6, 2.0)]
RUDDER_D = L - 4.0
ANCHORS_GONE = ()
FDC = (0.42, 0.36, 0.28)
CARRIER = dict(hangar=9.0, fd=(4.0, L - 2.0), fdw=33.4, island=(100.0, 128.0, 7.5, 5), funnel=(112.0, 124.0, 8.0, 5.0), mast=(106.0, 16.0),
               lifts=[(52.0, 14.0, 13.0, 0.0), (120.0, 14.0, 13.0, 0.0), (176.0, 14.0, 13.0, 0.0)], open_hangar=True,
               sponsons=[(d, s, "single") for d in (28.0, 38.0, 206.0, 216.0) for s in (-1, 1)] + [(d, s, "twin") for d in (70.0, 150.0) for s in (-1, 1)])


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
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship fd_y carrier_part sub")


def holes_in(m, spots, seed):
    """Remove faces in spheres (d, x, y, r): torpedo and bomb holes."""
    fc = m.v[m.f].mean(1); fd_ = L / 2 - fc[:, 2]
    hit = np.zeros(len(fc), bool)
    for (d0, x0, y0, r0) in spots:
        hit |= np.sqrt((fd_ - d0) ** 2 + (fc[:, 0] - x0) ** 2 + ((fc[:, 1] - y0) * 1.2) ** 2) < r0 * (0.8 + 0.4 * fbm(fc, 0.15, seed))
    m.f = m.f[~hit]
    return m


def tug():
    """A small aircraft tractor (tug) of the period."""
    return merge([box(1.6, 0.9, 2.8, (0.32, 0.34, 0.3)).T(0, 0.35, 0)] +
                 [cyl(0.35, 0.35, 0.25, RUBBER, seg=10).R(0, 0, math.pi / 2).T(s * 0.9, 0.35, dz) for s in (-1, 1) for dz in (-0.9, 0.9)])

w = merge(carrier_part(0, L, False, holes=0.08, seed=81))
crumple(w, 0.18, 0.1, 82)
holes_in(w, [(118.0, -B * 0.48, 4.0, 5.0), (150.0, -B * 0.48, 4.0, 5.0), (176.0, -B * 0.46, 3.5, 4.0),   # torpedoes, starboard side
             (96.0, B * 0.48, 3.0, 3.5), (140.0, B * 0.48, 4.0, 3.5),                                       # the scuttling torpedoes
             (150.0, 3.0, fd_y(150) + 0.5, 4.0), (176.0, -4.0, fd_y(176) + 0.5, 3.5)], 83)                  # bombs through the flight deck
w = merge([w, tug().T(2.0, deck_y(150.0), zd(150.0))])                         # the aircraft tug, upright on the hangar deck
w.R(0, 0, math.radians(3))
w.T(0, -np.percentile(w.v[:, 1], 0.5) - 2.0, 0)
debris = merge(plates(70, -18, zd(140), 30, 241, cols=(GREY, FDC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q838125" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-5350, kind="warship", seed=1950, memorial=True, drift=0.4)
ship = merge(carrier_part(0, L, True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
