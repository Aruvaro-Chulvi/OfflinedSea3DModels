"""USS Kittiwake (Q7870825) — the former US Navy submarine rescue ship sunk as an artificial reef off Grand Cayman, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Chanticleer-class submarine rescue ship (ASR-13) commissioned in 1946: about 76.6 m long, beam 13 m, a
long two-deck superstructure forward and amidships with the bridge, a funnel, a tripod mast, and a working deck aft
where she carried the McCann rescue chamber and her diving gear; she served until 1994.

The wreck, after the published descriptions and dive guides (Wikipedia, Scuba Diving, Divernet, DAN): cleaned and
sunk upright as an artificial reef off Seven Mile Beach on 5 January 2011, from 15 ft (her top) to some 65 ft (the
sand); a storm in November 2011 already shifted her 60 ft. In October 2017 Tropical Storm Nate broke her anchor
chains, moved her some 18-20 m towards the neighbouring site Sand Chute and rolled her onto her port side on the
sand next to the reef, about 6 m deeper, where she lies now (most of the dive at 15 m, the sand at about 21 m). The
heading is approximate.

    python model_sources/Q7870825_uss_kittiwake.py   -> models/item/Q7870825/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(2011)

L, B, D = 76.6, 13.1, 6.4
FC, TW, TT, WL = 2.4, 6.0, 1.2, 4.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.7, 0.25, 0.5, 0.08, 0.35, 1.8, 0.4, 2.2, 5.5
HULLC = (0.44, 0.46, 0.48)
SUPC = (0.48, 0.5, 0.52)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.46, 0.48, 0.5)
HHW = 2.0
HATCHES = []
BULKHEADS = [8, 52]
RAISED = [(0, 52)]
HOUSES = [(10, 50, 0.82, 2.5, 0), (14, 34, 0.7, 2.5, 1), (16, 24, 0.6, 2.4, 2)]
FUNNELS = [(38.0, 1.8, 4.5, 0.75)]
MASTS = [(28.0, 16.0)]
BOATS = [(42.0, 1), (46.0, 1)]
BOAT_X = 0.3
SCREWS = [(0.0, L - 5.0, 1.8, 1.6)]
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


def rescue_gear():
    """The working deck aft: the rescue-chamber A-frame and winches, bollards."""
    y = deck_y(64.0)
    P = []
    for s in (-1, 1):
        P.append(tube([(s * 2.4, y, zd(70.0)), (s * 1.0, y + 7.0, zd(73.0))], 0.25, GREY, seg=6))
    P.append(box(2.6, 0.5, 0.5, GREY).T(0, y + 6.8, zd(73.0)))
    P += [box(1.8, 1.2, 1.4, STEEL).T(s * 3.0, y, zd(58.0)) for s in (-1, 1)]
    return P


wreck = merge(ship_part(0, L, False, mast_left=[10.0], funnel_state="stand", davits=False, houses_damage=lambda m: crumple(m, 0.08, 0.4, 61)) + rescue_gear())
crumple(wreck, 0.06, 0.2, 62)
wreck.R(0, 0, math.radians(-88))                                         # on her port side (+x down)
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 0.7, 0)
wreck.T(-0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max()), 0, 0)
debris = merge(plates(8, -6, zd(30), 8, 291, cols=(HULLC,)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q7870825" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-21, kind="cargo", seed=2011, drift=0.6)
ship = merge([built_ship()] + rescue_gear())
dress_built(out.with_name("model_built.glb"), [("main", ship)])
