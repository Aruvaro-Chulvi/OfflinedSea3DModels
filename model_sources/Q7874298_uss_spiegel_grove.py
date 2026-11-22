"""USS Spiegel Grove (Q7874298) — the former US Navy dock landing ship sunk as an artificial reef off Key Largo, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Thomaston-class dock landing ship (LSD-32) commissioned in 1956: about 155 m long, beam 25.6 m, a tall
superstructure forward with the bridge, two funnels side by side behind it, and aft the long well deck between high
wing walls, which could be flooded to float landing craft in and out through the stern gate; two big cranes on the
wing walls and a removable helicopter deck over the well.

The wreck, after the published descriptions (Florida Keys tourism, PADI, DAN, NBC News, Wikipedia): sunk as an
artificial reef in the Florida Keys National Marine Sanctuary six miles off Key Largo, she went down prematurely on
17 May 2002 and settled upside down with her bow above the surface; she was rolled onto her starboard side some
weeks later, and in July 2005 the currents and waves raised by Hurricane Dennis turned her upright on the bottom in
about 130 ft (40 m), as intended. One of the largest ships ever sunk as a reef, her superstructure rises to within
about 60 ft of the surface. The heading and the state of the cranes are approximate.

    python model_sources/Q7874298_uss_spiegel_grove.py   -> models/item/Q7874298/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(2002)

L, B, D = 155.5, 25.6, 11.0
FC, TW, TT, WL = 2.6, 7.0, 1.3, 5.5
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.15, 0.85, 0.05, 0.25, 1.6, 0.2, 2.2, 6.0
HULLC = (0.44, 0.46, 0.48)
SUPC = (0.48, 0.5, 0.52)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.46, 0.48, 0.5)
HHW = 3.0
HATCHES = []
BULKHEADS = [12, 62]
RAISED = [(0, 62)]
HOUSES = [(16, 60, 0.9, 2.6, 0), (20, 56, 0.8, 2.6, 1), (24, 50, 0.7, 2.6, 2), (28, 42, 0.55, 2.4, 3)]
FUNNELS = []
MASTS = [(36.0, 14.0)]
BOATS = []
BOAT_X = 0.3
SCREWS = [(-4.0, L - 8.0, 2.2, 2.0), (4.0, L - 8.0, 2.2, 2.0)]
RUDDER_D = L - 3.0
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


install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship")


def dock(built):
    """The well deck aft between its high wing walls, the stern gate, the two funnels side by side, the cranes."""
    P = []
    y0 = deck_y(100.0) - 6.0                                              # the floor of the well, low in the hull
    a, b = 62.0, L - 2.0
    P.append(box(B * 0.62, 0.4, b - a, GREY_D).T(0, y0, zd((a + b) / 2)))
    for s in (-1, 1):
        x = s * (B / 2 - 2.6)
        P.append(box(4.6, deck_y(100.0) + 2.6 - y0, b - a, HULLC).T(x, y0, zd((a + b) / 2)))                                   # the wing walls
        P.append(box(4.6, 0.4, b - a, GREY_D).T(x, deck_y(100.0) + 2.6, zd((a + b) / 2)))
        P.append(cyl(1.7, 1.6, 7.0, FUNNELC, seg=14).T(s * 4.2, deck_y(58.0) + FC + 2.6 - 0.2, zd(58.0)))                     # the funnels
        cr = merge([cyl(1.2, 1.2, 4.0, GREY, seg=10), tube([(0, 4.0, 0), (0, 12.0, -14.0 if built else -8.0)], 0.4, GREY, seg=6)])
        if not built:
            cr.R(0.0, 0.0, s * 0.35)
        P.append(cr.T(x, deck_y(100.0) + 3.0, zd(96.0 + 30.0 * (s > 0))))
    P.append(box(B * 0.62, D - 1.0, 0.5, GREY_D).T(0, y0, zd(L - 0.5)))                                                    # the stern gate
    return P


wreck = merge(ship_part(0, 62.0, False, mast_left=[8.0], houses_damage=lambda m: crumple(m, 0.08, 0.4, 71)) + [hull(62.0, L)] + dock(False)
              + rudder(RUDDER_D, 0.6) + [m.T(x, 0, 0) for (x, d, y, r) in SCREWS for m in propeller(d, y, r, x=x)])
crumple(wreck, 0.08, 0.1, 72)
wreck.R(0, 0, math.radians(2))
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 1.2, 0)
debris = merge(plates(20, 16, zd(40), 14, 301, cols=(HULLC, SUPC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q7874298" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-40, kind="cargo", seed=2002, life="tropical")
ship = merge(ship_part(0, 62.0, True) + [hull(62.0, L)] + dock(True) + rudder(RUDDER_D, 0.6) + [m.T(x, 0, 0) for (x, d, y, r) in SCREWS for m in propeller(d, y, r, x=x)])
dress_built(out.with_name("model_built.glb"), [("main", ship)])
