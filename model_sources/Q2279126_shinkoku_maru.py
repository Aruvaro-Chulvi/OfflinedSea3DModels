"""Shinkoku Maru (Q2279126) — the wreck of the Japanese tanker sunk in Chuuk (Truk) Lagoon in 1944, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a fast tanker completed in 1940, about 152 m long and 20 m in the beam, taken over by the Imperial Japanese
Navy as a fleet oiler: a forecastle with a gun, the long low tank deck with its catwalk, the bridge island amidships,
and the engine room under the poop aft with the funnel, the lifeboats and a gun on the stern.

The wreck, after the published descriptions and dive guides (Divernet, Scuba Diving, Pacific Wrecks, Dive3D): she
was sunk by US carrier aircraft in Operation Hailstone, 17-18 February 1944. She sits upright on the bottom of the
lagoon at about 38-39 m near the propeller, her deck at some 15-20 m and the top of the bridge at about 11 m; her
masts and superstructure are thickly covered with soft corals and anemones (the model is left bare: Offlined draws
the life), and her engine room is one of the most famous in the lagoon. The exact list, damage and heading are
approximate.

    python model_sources/Q2279126_shinkoku_maru.py   -> models/item/Q2279126/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1944 + 2)

L, B, D = 152.4, 20.0, 11.6
FC, TW, TT, WL = 2.4, 7.0, 1.3, 8.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.27, 0.75, 0.2, 0.45, 0.07, 0.3, 1.6, 0.5, 2.6, 7.0
HULLC = (0.32, 0.33, 0.33)
SUPC = (0.45, 0.46, 0.46)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.32, 0.33, 0.33)
HHW = 1.2
HATCHES = []
BULKHEADS = [14, 64, 88, 124]
RAISED = [(0, 14), (64, 88), (124, L)]
HOUSES = [(66, 84, 0.6, 2.5, 0), (68, 82, 0.55, 2.5, 1), (70, 78, 0.5, 2.4, 2), (126, 148, 0.78, 2.5, 0), (130, 144, 0.6, 2.5, 1)]
FUNNELS = [(138.0, 2.4, 8.0, 0.75)]
MASTS = [(12.0, 18.0), (100.0, 16.0)]
BOATS = [(134.0, 1), (142.0, 1)]
BOAT_X = 0.28
SCREWS = [(0.0, L - 6.0, 2.6, 2.2)]
RUDDER_D = L - 2.5
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


def catwalk(d0, d1):
    """The raised fore-and-aft gangway over the tank deck, on posts."""
    P = []
    for a, b in ((14, 64), (88, 124)):
        a, b = max(a, d0), min(b, d1)
        if b - a < 1:
            continue
        P.append(box(1.4, 0.15, b - a, GREY_D).T(0, deck_y((a + b) / 2) + 2.2, zd((a + b) / 2)))
        P += [box(0.15, 2.2, 0.15, GREY_D).T(0, deck_y(d), zd(d)) for d in np.arange(a + 1.5, b, 4.5)]
    for a, b in ((16, 62), (90, 122)):                                        # tank lids and the pipes along the deck
        for d in np.arange(a + 2, b - 1, 6.0):
            P.append(cyl(0.6, 0.6, 0.5, STEEL, seg=10).T(-3.5, deck_y(d), zd(d)))
        for x in (2.5, 4.0):
            P.append(cyl(0.18, 0.18, b - a, STEEL, seg=6).R(math.pi / 2, 0, 0).T(x, deck_y((a + b) / 2) + 0.3, zd((a + b) / 2)))
    return P


def guns():
    P = []
    for d in (5.0, L - 4.0):
        g = merge([cyl(0.8, 0.9, 0.9, GREY, seg=12), box(1.2, 1.0, 1.6, GREY).T(0, 0.9, 0), cyl(0.09, 0.08, 3.4, STEEL, seg=6).R(math.pi / 2, 0, 0).T(0, 1.4, 0.5)])
        P.append(g.R(0, 0.0 if d < 50 else math.pi, 0).T(0, deck_y(d) + FC, zd(d)))
    return P


wreck = merge(ship_part(0, L, False, mast_left=[14.0, 12.0], funnel_state="stand", houses_damage=lambda m: crumple(m, 0.12, 0.3, 41)) + catwalk(0, L) + guns())
crumple(wreck, 0.1, 0.1, 42)
wreck.R(0, 0, math.radians(3))
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 1.0, 0)
debris = merge(plates(30, -12, zd(40), 14, 221, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q2279126" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-38, kind="cargo", seed=1946, life="tropical")
ship = merge([built_ship()] + catwalk(0, L) + guns())
dress_built(out.with_name("model_built.glb"), [("main", ship)])
