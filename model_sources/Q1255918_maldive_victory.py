"""Maldive Victory (Q1255918) — the wreck of the cargo ship sunk off Hulhulé, Maldives, in 1981, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a general cargo ship built in the early 1970s, about 83 m long: a forecastle, three holds forward with
their hatches and derrick posts, the engine room and a four-deck accommodation block aft with the wheelhouse on top
and a single funnel.

The wreck, after the published descriptions and dive guides (PADI, Divernet, Dive Against Debris): carrying supplies
for the resorts, she struck the reef off the airport island of Hulhulé on the night of 13 February 1981 and sank, her
hull breached above the cargo holds. She sits upright, slightly tilted, in about 35 m, parallel to the house reef,
one of the best-known dives of the Maldives; anchors dropped by vessels working on the bridge to Hulhulé damaged her
wheelhouse and cabins, and sediment from the works settled on her. The exact tilt and heading are approximate.

    python model_sources/Q1255918_maldive_victory.py   -> models/item/Q1255918/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1981)

L, B, D = 83.0, 13.0, 6.8
FC, TW, TT, WL = 2.4, 6.5, 1.2, 5.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.28, 0.7, 0.22, 0.5, 0.08, 0.35, 1.4, 0.5, 2.4, 6.0
HULLC = (0.18, 0.2, 0.24)
SUPC = (0.86, 0.86, 0.82)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.82, 0.55, 0.2)
HHW = 3.2
HATCHES = [(12, 22), (26, 38), (42, 56)]
BULKHEADS = [10, 24, 40, 58]
RAISED = [(0, 10), (60, L)]
HOUSES = [(62, 78, 0.86, 2.5, 0), (63, 76, 0.82, 2.5, 1), (64, 74, 0.66, 2.5, 2), (64, 70, 0.6, 2.4, 3)]
FUNNELS = [(73.0, 1.7, 5.0, 0.7)]
MASTS = [(24.0, 11.0), (40.0, 11.0)]
BOATS = [(68.0, 2)]
BOAT_X = 0.3
SCREWS = [(0.0, L - 5.0, 1.9, 1.6)]
RUDDER_D = L - 2.0
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


def wheelhouse_damage(m):
    """The anchors of the bridge works came down on her top: the wheelhouse and cabins crushed and torn."""
    crumple(m, 0.12, 0.3, 31)
    top = m.v[:, 1] > deck_y(66) + FC + LEVEL_H * 3 - 0.3
    m.v[top, 1] -= 0.6 + 0.5 * fbm(m.v[top], 0.4, 32)
    crumple(m, 0.4, 0.4, 33, top.astype(float))


wreck = merge(ship_part(0, L, False, mast_left=[9.0, 6.0], funnel_state="stand", houses_damage=wheelhouse_damage))
crumple(wreck, 0.08, 0.1, 34)
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
breach = (fd_ > 14) & (fd_ < 30) & (fc[:, 0] < -B * 0.3) & (fc[:, 1] < 3.5) & (fbm(fc, 0.2, 35) > 0.45)   # holed on the reef
wreck.f = wreck.f[~breach]
wreck.R(0, 0, math.radians(-7))                                            # upright, slightly tilted
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 0.9, 0)
debris = merge(plates(25, 9, zd(64), 8, 211, cols=(SUPC, GREY)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1255918" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-35, kind="cargo", seed=1981, life="tropical")
dress_built(out.with_name("model_built.glb"), [("main", built_ship())])
