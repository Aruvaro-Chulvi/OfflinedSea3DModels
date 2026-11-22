"""
USS Oriskany (Q562053) — the aircraft carrier sunk as an artificial reef off Pensacola, Florida, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: an Essex-class carrier (CV-34) completed in 1950 and rebuilt in the 1950s with an angled flight deck, an
enclosed hurricane bow, deck-edge lifts and a new island with its stack and mast: about 271 m long, beam 30.9 m.
Veteran of Korea and Vietnam (the 1966 fire), decommissioned in 1976.

The reef, after the published descriptions: stripped of her armament, fuels and hazardous materials, she was
scuttled on 17 May 2006 some 24 miles off Pensacola, the largest ship ever sunk as an artificial reef. She rests
upright on the sand, bow to the south, the bottom at about 212 ft (65 m), the flight deck at about 145-150 ft
(44-46 m), the top of the island at about 84 ft (26 m). The layout of the island and of the lifts is approximate.

    python model_sources/Q562053_uss_oriskany.py   -> models/item/Q562053/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(2006)

L, B, D = 270.7, 30.9, 12.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.18, 0.55, 0.06, 0.25, 1.2, 0.2, 2.6, 6.5
HULLC = (0.5, 0.52, 0.54)                    # haze grey
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
SCREWS = [(-4.0, L - 18.0, 2.4, 2.3), (4.0, L - 18.0, 2.4, 2.3), (-9.0, L - 30.0, 2.6, 2.3), (9.0, L - 30.0, 2.6, 2.3)]
RUDDER_D = L - 5.0
ANCHORS_GONE = ()
FDC = (0.26, 0.27, 0.28)
CARRIER = dict(hangar=7.5, fd=(2.0, L - 2.0), fdw=33.0, island=(95.0, 128.0, 7.5, 5), funnel=(108.0, 122.0, 7.0, 6.0), mast=(102.0, 10.0),
               lifts=[(40.0, 16.0, 13.0, 0.0)], sponsons=[])


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


def angled_deck():
    """The angled deck to port and the two deck-edge lifts (port forward, starboard aft of the island)."""
    a, b = 70.0, 235.0
    outer = lambda d: CARRIER["fdw"] / 2 + 12.0 * sstep(a, a + 40, d) * sstep(b, b - 50, d)
    ds = np.arange(a, b + 0.1, 3.0)
    v, f = [], []
    for i, d in enumerate(ds):
        y = fd_y(d) + 0.02
        v += [(CARRIER["fdw"] / 2 - 0.5, y, zd(d)), (outer(d), y, zd(d))]
    for i in range(len(ds) - 1):
        k = 2 * i
        f += [(k, k + 2, k + 1), (k + 1, k + 2, k + 3)]
    deck = Mesh(v, f, FDC)
    edge = tube([(outer(d), fd_y(d) - 0.3, zd(d)) for d in ds], 0.35, GREY_D, seg=4)
    P = [deck, edge]
    for d in np.arange(a + 30, b - 30, 6.0):                                  # the angled landing line
        x = CARRIER["fdw"] / 2 + 12.0 * sstep(a, a + 40, d) * sstep(b, b - 50, d) - 6.0 - (d - a) * 0.04
        P.append(quad(x, fd_y(d) + 0.05, zd(d), 0.5, 3.0, "+y", (0.75, 0.75, 0.72), eps=0.0))
    P.append(box(13.0, 0.4, 16.0, GREY_D).T(CARRIER["fdw"] / 2 + 5.0, fd_y(82) - 0.4, zd(82)))       # port deck-edge lift
    P.append(box(13.0, 0.4, 16.0, GREY_D).T(-(CARRIER["fdw"] / 2 + 5.0), fd_y(140) - 0.4, zd(140)))  # starboard deck-edge lift
    return P


wreck = merge(carrier_part(0, L, False, holes=0.0) + angled_deck())
crumple(wreck, 0.12, 0.08, 11)
wreck.T(0, -0.6, 0)
debris = merge(plates(12, 25, zd(150), 15, 100, cols=(GREY, FDC)))
coral = merge([corals(wreck, 340, 7, size=(0.12, 0.45))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q562053" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-65, kind="cargo", seed=2006, life="tropical")
ship = merge(carrier_part(0, L, True) + angled_deck()); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
