"""
MS Antilla (Q576959) — the wreck of the German freighter scuttled off Aruba, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a HAPAG cargo motor ship completed in 1939, about 121 m long, with a raised forecastle, holds fore and
aft of the bridge and engine island (one funnel), two masts and a poop.

The wreck, after the published descriptions and dive guides: caught in the Caribbean at the start of the war, she
was scuttled by her captain off Aruba on 10 May 1940 as Dutch troops came to seize her. She lies on her port side
in 16-18 m of water off Malmok, broken in two parts that make two dives, the largest wreck in the Caribbean,
covered in tube sponges and corals. Where exactly she broke, the gap between the parts, the heading (not
published: 0), the colours and the layout are approximate.

    python model_sources/Q576959_antilla.py   -> models/item/Q576959/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1940)

L, B, D = 121.4, 16.5, 9.6
FC, TW, TT, WL = 2.4, 5.8, 1.1, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.2, 0.7, 0.22, 0.22, 0.1, 0.5, 2.0, 1.0, 2.5, 7.0
HULLC = (0.09, 0.09, 0.1)                    # black
SUPC = (0.8, 0.79, 0.74)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 3.5
HATCHES = [(15, 25), (28, 38), (41, 49), (74, 84), (87, 97), (100, 108)]
BULKHEADS = [13, 26.5, 39.5, 50.5, 72, 85.5, 98.5, 110]
RAISED = [(0, 12), (50.5, 72), (110, L)]
HOUSES = [(52, 70, 0.74, 2.5, 0), (53, 66, 0.6, 2.5, 1), (53, 57.5, 0.62, 2.4, 2)]
FUNNELS = [(63.0, 2.4, 6.5, 0.75)]
MASTS = [(26.5, 16.0), (85.5, 15.0)]
BOATS = [(62.0, 2)]                                 # on the boat-deck house aft of the break (58-66 m), davits and all
BOAT_X = 0.28                                # the davits on the boat-deck house, which is narrow
SCREWS = [(0.0, L - 4.0, 2.3, 2.0)]
RUDDER_D = L - 1.2
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


LIST = math.radians(80)                                                      # on her port side (+X)


def roll(m):
    return m.R(0, 0, -LIST)


BRK = 58.0                                                                   # broken in two, about amidships
parts = []
for k, (a, b) in enumerate(((0, BRK), (BRK, L))):
    m = merge(ship_part(a, b, False, mast_left=[3.0, 0.0], funnel_state="gone", houses_damage=lambda m: crumple(m, 0.4, 0.3, 61)))
    dd = L / 2 - m.v[:, 2]
    crumple(m, 0.16, 0.12, 11 + k)
    crumple(m, 1.5, 0.3, 21 + k, sstep(BRK + (-6 if k == 0 else 6), BRK, dd))
    parts.append(m)
roll(parts[0]); roll(parts[1])
LIFT = -min(np.percentile(p.v[:, 1], 0.5) for p in parts) - 1.0
for p in parts:
    p.T(0, LIFT, 0)
XC = -0.5 * (parts[0].v[:, 0].min() + parts[0].v[:, 0].max())
for p in parts:
    p.T(XC, 0, 0)
zc = zd(BRK)
parts[1].T(0, 0, -zc).R(0, 0.18, 0).T(1.5, 0, zc - 4.0)                      # the after part twisted a little and apart
fore, aft = parts
onwreck = make_onwreck(roll, XC, LIFT)
side = onwreck(0, D + FC + 4, 60)[0]
deb = plates(50, side + 4, zc, 12, 100) + plates(20, side + 6, zc - 30, 14, 101)
fun = cyl(2.4, 2.35, 6.0, FUNNELC, seg=18, sx=0.75, caps=False); crumple(fun, 0.4, 0.35, 51); fun.v[:, 0] *= 0.7
deb.append(fun.R(0, 0, -math.pi / 2).R(0, 0.5, 0).T(side + 5, 1.0, zd(63.0)))
m = cyl(0.36, 0.2, 13.0, GREY, seg=8, caps=False); crumple(m, 0.12, 0.4, 60)
deb.append(m.R(0, 0, -math.pi / 2 + 0.05).R(0, -0.3, 0).T(side + 2.0, 0.3, zd(85.5)))
debris = merge(deb)
coral = merge([corals(fore, 420, 7, size=(0.2, 0.75)), corals(aft, 380, 8, size=(0.2, 0.75))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q576959" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", fore), ("stern_section", aft), ("debris_field", debris, "loose")], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-17, kind="cargo", seed=1940, life="tropical")
dress_built(out.with_name("model_built.glb"), [("main", built_ship())])
