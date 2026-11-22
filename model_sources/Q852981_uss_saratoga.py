"""
USS Saratoga (Q852981) — the wreck of the American aircraft carrier at Bikini Atoll, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the Lexington-class carrier CV-3, completed in 1927 on a battlecruiser hull: about 270 m long, beam 32 m,
a long flight deck over an enclosed hangar, the island and the huge flat funnel on the starboard side, twin
5-inch gun mounts before and abaft the island (1945 fit), four screws.

The wreck, after the published descriptions and dive guides: a target ship of Operation Crossroads, she survived
the air burst Able and sank after the underwater burst Baker on 25 July 1946. She rests upright on the bottom of
the lagoon at about 51 m, the flight deck at 27-30 m, the bridge rising to within some 12-18 m of the surface;
the unarmoured flight deck aft has collapsed onto the hangar deck below; aircraft and bombs remain in the hangar.
The only aircraft carrier divable by recreational technical divers. The heading (not published: 0), the extent
of the collapse and the state of the funnel are approximate.

    python model_sources/Q852981_uss_saratoga.py   -> models/item/Q852981/model.glb (+ model_built.glb)
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

L, B, D = 270.7, 32.3, 12.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.32, 0.75, 0.2, 0.5, 0.06, 0.25, 1.0, 0.2, 2.6, 6.5
HULLC = (0.36, 0.4, 0.45)                    # navy blue-grey
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
SCREWS = [(-4.0, L - 18.0, 2.4, 2.3), (4.0, L - 18.0, 2.4, 2.3), (-9.0, L - 32.0, 2.6, 2.3), (9.0, L - 32.0, 2.6, 2.3)]
RUDDER_D = L - 5.0
ANCHORS_GONE = ()
FDC = (0.3, 0.29, 0.27)
CARRIER = dict(hangar=8.0, fd=(4.0, L - 3.0), fdw=32.3, island=(100.0, 126.0, 7.0, 4), funnel=(128.0, 166.0, 17.0, 8.0), mast=(112.0, 12.0),
               lifts=[(32.0, 18.0, 12.0, 0.0), (225.0, 12.0, 10.0, 0.0)],
               sponsons=[(d, -1, "twin") for d in (92.0, 172.0, 180.0)] + [(d, 1, "twin") for d in (60.0, 200.0)] + [(d, s, "single") for d in (40.0, 230.0) for s in (-1, 1)])


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


def helldiver():
    """A Curtiss SB2C Helldiver, wings folded, as left in the hangar."""
    f = lathe([(0.05, -6.0), (0.35, -4.5), (0.7, -1.0), (0.75, 1.5), (0.65, 3.4), (0.5, 4.0)], (0.25, 0.3, 0.38), seg=8, axis="z")
    P = [f, box(4.6, 0.2, 2.2, (0.25, 0.3, 0.38)).T(0, -0.4, 0.8), box(4.4, 0.08, 1.2, (0.25, 0.3, 0.38)).T(0, 0.1, -5.6), box(0.08, 1.6, 1.4, (0.25, 0.3, 0.38)).T(0, 0.1, -5.5)]
    for s in (-1, 1):
        P.append(box(0.2, 4.2, 1.9, (0.25, 0.3, 0.38)).R(0, 0, s * 0.15).T(s * 2.5, 1.2, 0.8))      # folded outer wings
        P.append(cyl(0.3, 0.3, 0.2, RUBBER, seg=8).R(0, 0, math.pi / 2).T(s * 1.6, -1.7, 1.5))
    return merge(P)


wreck = merge(carrier_part(0, L, False, holes=0.05, seed=46)
              + [helldiver().R(0, a, 0).T(x, deck_y(d) + 2.0, zd(d)) for d, x, a in ((70, -6, 0.1), (78, 5, -0.2), (150, 0, 0.3))])
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.22, 0.08, 11)
aft = sstep(165, 185, dfb) * sstep(L - 4, L - 18, dfb) * (wreck.v[:, 1] > D + 3)  # the flight deck aft collapsed onto the hangar deck
wreck.v[:, 1] -= aft * (CARRIER["hangar"] - 1.2) * (0.8 + 0.2 * fbm(wreck.v, 0.1, 47))
crumple(wreck, 1.0, 0.15, 12, aft)
wreck.T(0, -1.2, 0)
r = np.random.default_rng(46)
debris = merge(plates(50, 20, zd(210), 22, 100, cols=(GREY, FDC, GREY_D)) + plates(20, -22, zd(140), 15, 101, cols=(GREY, GREY_D)))
coral = merge([corals(wreck, 520, 7, size=(0.12, 0.45))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q852981" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-52, kind="warship", seed=1946, life="tropical")
ship = merge(carrier_part(0, L, True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
