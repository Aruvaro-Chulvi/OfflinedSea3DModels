"""Armenia (Q683174) — the wreck of the Soviet hospital ship sunk off Crimea in 1941, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Black Sea passenger motor ship of the Adzharia class, built in Leningrad in 1928: about 108 m long,
beam 15.5 m, a white superstructure of three decks, a single funnel, two masts, two screws. In the war she served as
a hospital transport, marked with red crosses.

The wreck, after the descriptions of the Russian Geographical Society, whose underwater research centre identified
it in April 2020 with an underwater robot after Russian Navy sonar found it in March: evacuating wounded, medical
staff and civilians from Yalta, she was sunk by a German torpedo bomber on 7 November 1941; perhaps 5,000-7,000 or
more died, one of the worst losses at sea. She lies some 15 nautical miles off Yalta at about 1,500 m: the ship
did not split, and stands on an even keel in one piece; two zones of bomb (torpedo) damage were found. The exact
damage, the state of her masts and the heading are approximate.

    python model_sources/Q683174_armenia.py   -> models/item/Q683174/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1941 + 11)

L, B, D = 107.7, 15.5, 8.4
FC, TW, TT, WL = 2.4, 7.0, 1.3, 5.4
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.26, 0.35, 0.08, 0.35, 1.8, 0.7, 2.4, 6.0
HULLC = (0.85, 0.85, 0.83)
SUPC = (0.88, 0.88, 0.86)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.86, 0.86, 0.84)
HHW = 3.0
HATCHES = [(10, 15), (95, 100)]
BULKHEADS = [8, 17, 93]
RAISED = [(0, L)]
HOUSES = [(24, 84, 0.9, 2.5, 0), (26, 80, 0.86, 2.5, 1), (30, 70, 0.6, 2.5, 2), (30, 36, 0.55, 2.4, 3)]
FUNNELS = [(56.0, 2.6, 7.0, 0.7)]
MASTS = [(17.0, 15.0), (90.0, 13.0)]
BOATS = [(d, 2) for d in (40.0, 47.0, 63.0, 69.0)]
BOAT_X = 0.27
SCREWS = [(-3.2, L - 8.0, 2.0, 1.6), (3.2, L - 8.0, 2.0, 1.6)]
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


CROSS = (0.62, 0.08, 0.07)


def red_crosses():
    """Red crosses on both sides of the hull and of the funnel."""
    out = []
    for d in (30.0, 78.0):
        for s in (-1, 1):
            x = s * (half(d) + 0.04); y = deck_y(d) - 2.6
            out.append(quad(x, y, zd(d), 1.2, 4.0, "+x" if s > 0 else "-x", CROSS, eps=0.0))
            out.append(quad(x, y, zd(d), 4.0, 1.2, "+x" if s > 0 else "-x", CROSS, eps=0.0))
    yb = deck_y(56) + FC + LEVEL_H * 3 - 0.2
    for s in (-1, 1):
        x = s * (2.6 * 0.7 + 0.06)
        out.append(quad(x, yb + 3.8, zd(56), 0.6, 2.2, "+x" if s > 0 else "-x", CROSS, eps=0.0))
        out.append(quad(x, yb + 3.8, zd(56), 2.2, 0.6, "+x" if s > 0 else "-x", CROSS, eps=0.0))
    return out


wreck = merge(ship_part(0, L, False, mast_left=[8.0, 6.0], funnel_state="stand", houses_damage=lambda m: crumple(m, 0.12, 0.3, 21)) + red_crosses())
crumple(wreck, 0.1, 0.1, 22)
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
hit = np.zeros(len(fc), bool)
for d0, s in ((18.0, 1), (88.0, -1)):                                   # two zones of damage
    hit |= (np.hypot(fd_ - d0, (fc[:, 1] - 4.0) * 1.2) < 5.0 * (0.8 + 0.4 * fbm(fc, 0.15, 23))) & (fc[:, 0] * s > -1.0)
wreck.f = wreck.f[~hit]
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 1.6, 0)              # on an even keel, sunk into the mud of the slope
debris = merge(plates(50, 6, zd(20), 14, 201, cols=(HULLC, SUPC)) + plates(40, -6, zd(88), 14, 202, cols=(HULLC, SUPC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q683174" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-1500, kind="liner", seed=1952, memorial=True, drift=0.6)
ship = merge([built_ship()] + [m.T(0, 0, -zd(L / 2)) for m in red_crosses()])
dress_built(out.with_name("model_built.glb"), [("main", ship)])
