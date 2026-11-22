"""SS Baron Gautsch (Q686128) — the wreck of the Austrian Lloyd passenger steamer lost off Rovinj in 1914, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a passenger steamer built in 1908 for the Austrian Lloyd's Dalmatian service: about 84.5 m long and 11.6 m
in the beam, a forecastle, a long white superstructure with the promenade deck, a single funnel, two masts, two
screws.

The wreck, after the published descriptions (Wikipedia, the Istria tourist board, PADI, Total Croatia News): on 13
August 1914, two weeks into the First World War, she strayed into an Austro-Hungarian minefield off Rovinj, struck a
mine on her port side and sank in a few minutes; some 147 died. She sits upright and well preserved in 38-40 m, her
upper deck at about 28 m: the railings, portholes, davits and promenade deck are still there, the masts broken, the
propellers removed, and the hole made by the mine is still clearly visible. A protected cultural monument: diving only
with permitted dive centres. The heading (not published here: 0) is approximate.

    python model_sources/Q686128_baron_gautsch.py   -> models/item/Q686128/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1914 + 8)

L, B, D = 84.5, 11.6, 7.0
FC, TW, TT, WL = 2.4, 4.8, 1.0, 4.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.26, 0.7, 0.24, 0.4, 0.08, 0.32, 1.2, 0.5, 2.4, 6.0
HULLC = (0.1, 0.1, 0.11)
SUPC = (0.8, 0.78, 0.72)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.72, 0.6, 0.3)
HHW = 2.0
HATCHES = [(12, 16), (74, 78)]
BULKHEADS = [9, 20, 70]
RAISED = [(0, 9), (20, L)]
HOUSES = [(22, 68, 0.84, 2.4, 0), (26, 62, 0.8, 2.4, 1), (28, 34, 0.52, 2.3, 2)]
FUNNELS = [(46.0, 1.6, 8.0, 0.85)]
MASTS = [(16.0, 16.0), (76.0, 14.0)]
BOATS = [(d, 1) for d in (36.0, 42.0, 52.0, 58.0)]
BOAT_X = 0.36
SCREWS = [(-2.2, L - 5.0, 1.6, 1.3), (2.2, L - 5.0, 1.6, 1.3)]
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


# ── Damage helpers ───────────────────────────────────────────────────────────
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship collapse holes lay boiler center_x")


def promenade(d0, d1, lv, built):
    """The open promenade deck: the roof on stanchions with the railings, all the way round the house."""
    y0 = deck_y((d0 + d1) / 2) + FC + 2.4 * lv
    w = B * 0.42
    P = [box(2 * w, 0.15, d1 - d0, DECKC).T(0, y0 + 2.4, zd((d0 + d1) / 2))]
    for s in (-1, 1):
        P += [box(0.12, 2.4, 0.12, SUPC).T(s * w, y0, zd(d)) for d in np.arange(d0 + 1, d1, 3.0)]
        P.append(box(0.06, 0.05, d1 - d0, SUPC).T(s * w, y0 + 1.0, zd((d0 + d1) / 2)))
    return P


wreck = merge(ship_part(0, L, False, mast_left=[3.0, 2.0], funnel_state="stump", davits=True) + promenade(24, 66, 0, False))
holes(wreck, [(14.0, B * 0.49, 2.5, 3.0)], 31)                                     # the mine, port side forward
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
wreck.f = wreck.f[~((fd_ > L - 7) & (fc[:, 1] < 2.0) & (np.abs(fc[:, 2] - zd(L - 5)) < 1.5) & (fc[:, 0] ** 2 < 9))]   # propellers removed
crumple(wreck, 0.06, 0.12, 32)
wreck.R(0, 0, math.radians(2))
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 0.7, 0)
debris = merge(plates(25, 8, zd(30), 8, 321, cols=(SUPC, DECKC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q686128" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-40, kind="liner", seed=1914)
dress_built(out.with_name("model_built.glb"), [("main", merge([built_ship(), merge(promenade(24, 66, 0, True)).T(0, 0, -zd(L / 2))]))], deck_colors=[DECKC])
