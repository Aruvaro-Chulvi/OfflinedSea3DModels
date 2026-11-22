"""Kaga (Q726146) — the wreck of the Japanese aircraft carrier lost at the Battle of Midway, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: laid down as a battleship, completed as an aircraft carrier in 1928 and rebuilt in 1934-1935 with a
single full-length flight deck: about 248 m long, 32.5 m across her anti-torpedo bulges, two hangars one above the
other, a small island on the starboard side, the funnel trunked out along the starboard side and bent down towards
the sea, ten 20 cm guns in casemates low on the sides aft, twin 12.7 cm anti-aircraft mounts in sponsons, four screws.
She took part in the attack on Pearl Harbor.

The wreck, after the published descriptions of the expedition of the research vessel Petrel (Vulcan, Paul Allen),
which found her on 16 October 2019 near Midway: set on fire by dive bombers from USS Enterprise on 4 June 1942, she
sank that evening with some 800 men. She sits upright at about 5,400 m, twisted, with much of her flight deck
missing; the ROV filmed a 20 cm casemate gun on her starboard side, a gun mount and a barbette. (Debris from her had
been found before; this is the main wreck.) The damage, the extent of what is left of the flight deck and the
heading are approximate.

    python model_sources/Q726146_kaga.py   -> models/item/Q726146/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1942 + 9)

L, B, D = 240.0, 32.5, 15.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 9.5
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.28, 0.75, 0.2, 0.5, 0.07, 0.3, 1.4, 0.3, 2.3, 6.0
HULLC = (0.4, 0.42, 0.43)
SUPC = (0.45, 0.47, 0.47)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
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
SCREWS = [(-8.0, L - 30.0, 2.8, 2.1), (-3.0, L - 22.0, 2.6, 2.1), (3.0, L - 22.0, 2.6, 2.1), (8.0, L - 30.0, 2.8, 2.1)]
RUDDER_D = L - 4.0
ANCHORS_GONE = ()
FDC = (0.45, 0.38, 0.28)
CARRIER = dict(hangar=11.0, fd=(6.0, L - 2.0), fdw=30.5, island=(70.0, 82.0, 5.5, 3), funnel=None, mast=(76.0, 0.0),
               lifts=[(40.0, 12.0, 11.0, 0.0), (110.0, 12.0, 11.0, 0.0), (175.0, 12.0, 11.0, 0.0)],
               sponsons=[(d, s, "twin") for d in (36.0, 60.0, 190.0, 214.0) for s in (-1, 1)])


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


def casemates():
    """Ten 20 cm guns in casemates low on the sides aft, five a side."""
    P = []
    for d in (160.0, 172.0, 184.0, 196.0, 208.0):
        for s in (-1, 1):
            z = zd(d); y = deck_y(d) + 1.4
            x = s * (hw(z, y) - 0.3)
            P.append(box(3.0, 2.4, 3.6, GREY).T(x, y - 1.2, z))
            P.append(cyl(0.16, 0.13, 6.0, STEEL, seg=8).R(math.pi / 2, 0, 0).R(0, s * 1.35, 0).T(x + s * 1.2, y, z))
    return P


def side_funnel():
    """The funnel: trunked out along the starboard side from the hangar and bent down towards the sea."""
    P = []
    for k, d in enumerate((118.0, 126.0)):
        z = zd(d); x0 = -(hw(z, deck_y(d)) + 0.5)
        f = box(4.0, 3.2, 6.0, FUNNELC).T(0, -1.6, 0)
        f.R(0, 0, math.radians(-25))
        P.append(f.T(x0 - 2.0, deck_y(d) + 5.0, z))
    return P


w = merge(carrier_part(0, L, False, holes=0.62, seed=91) + casemates() + side_funnel())
crumple(w, 0.35, 0.12, 92)
crumple(w, 1.2, 0.03, 93, (w.v[:, 1] > D + 4.0).astype(float))          # twisted: the hangar sides buckled by the fires and the fall
fc = w.v[w.f].mean(1); fd_ = L / 2 - fc[:, 2]
hang = (fc[:, 1] > D + 3.0) & (fbm(fc, 0.05, 94) > 0.6)                  # parts of the hangar sides gone with the flight deck
w.f = w.f[~hang]
w.R(0, 0, math.radians(-3))
w.T(0, -np.percentile(w.v[:, 1], 0.5) - 2.2, 0)
debris = merge(plates(140, 0, zd(120), 40, 251, cols=(GREY, FDC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q726146" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-5400, kind="warship", seed=1951, memorial=True, drift=0.4)
ship = merge(carrier_part(0, L, True) + casemates() + side_funnel()); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
