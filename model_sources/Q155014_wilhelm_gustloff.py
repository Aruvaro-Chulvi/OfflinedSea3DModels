"""MV Wilhelm Gustloff (Q155014) — the wreck of the German liner sunk in the Baltic in 1945, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a "Strength Through Joy" cruise liner completed in 1938: about 208.5 m long, beam 23.6 m, a long white
superstructure of promenade and boat decks, a single funnel, two masts, two screws. In the war she was a hospital
ship, then an accommodation ship at Gotenhafen (Gdynia), painted grey.

The wreck, after the published descriptions and the HMS Echo sonar survey (2021): evacuating refugees, wounded and
naval personnel from East Prussia, she was hit by three torpedoes from the Soviet submarine S-13 on the port side on
the night of 30 January 1945 and sank in under 40 minutes, listing to port; some 9,000 died, the greatest loss of
life in a single sinking. She lies some 19 nautical miles north of Łeba in about 45-47 m, broken in three parts.
Declared a war grave in 1994: diving is forbidden within 500 m. The side she lies on, the gaps between the parts and
the heading are approximate.

    python model_sources/Q155014_wilhelm_gustloff.py   -> models/item/Q155014/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1945)

L, B, D = 208.5, 23.6, 14.0
FC, TW, TT, WL = 2.6, 8.0, 1.4, 7.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.26, 0.35, 0.08, 0.35, 2.0, 0.8, 2.4, 6.0
HULLC = (0.5, 0.52, 0.53)                     # wartime grey
SUPC = (0.56, 0.58, 0.58)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.52, 0.54, 0.54)
HHW = 3.5
HATCHES = [(14, 22), (186, 193)]
BULKHEADS = [12, 30, 184]
RAISED = [(0, L)]
HOUSES = [(32, 182, 0.94, 2.5, 0), (34, 176, 0.92, 2.5, 1), (38, 166, 0.88, 2.5, 2), (44, 150, 0.62, 2.5, 3), (44, 54, 0.6, 2.4, 4)]
FUNNELS = [(112.0, 4.2, 10.0, 0.62)]
MASTS = [(24.0, 20.0), (190.0, 17.0)]
BOATS = [(d, 3) for d in (62.0, 70.0, 78.0, 86.0, 94.0, 126.0, 134.0, 142.0)]
BOAT_X = 0.29
SCREWS = [(-4.8, L - 12.0, 2.8, 2.3), (4.8, L - 12.0, 2.8, 2.3)]
RUDDER_D = L - 4.0
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


LIST = math.radians(-86)                                                     # on her port side (+X down)


def lay(m, dz, yaw, seed):
    crumple(m, 0.16, 0.1, seed)
    m.R(0, 0, LIST)
    c = m.v.mean(0)
    m.T(-c[0], 0, -c[2]).R(0, yaw, 0).T(c[0], 0, c[2] + dz)
    m.T(0, -np.percentile(m.v[:, 1], 0.5) - 1.0, 0)
    return m


crush = lambda m: crumple(m, 0.3, 0.3, 77)
bow = merge(ship_part(0, 74, False, mast_left=[6.0], houses_damage=crush))
mid = merge(ship_part(76, 140, False, funnel_state="gone", houses_damage=crush, davits=False))
stern = merge(ship_part(142, L, False, mast_left=[0.0, 4.0], houses_damage=crush, davits=False))
# the middle part broken down: the superstructure collapsed, plating torn away
fc = mid.v[mid.f].mean(1)
mid.f = mid.f[~((fbm(fc, 0.08, 78) > 0.62) | ((fc[:, 1] > D + 6) & (fbm(fc, 0.15, 79) > 0.4)))]
mid.v[:, 1] = np.where(mid.v[:, 1] > D * 0.6, D * 0.6 + (mid.v[:, 1] - D * 0.6) * 0.45, mid.v[:, 1])
crumple(mid, 0.6, 0.12, 80)
bow = lay(bow, 7.0, math.radians(-6), 81)
mid = lay(mid, 0.0, math.radians(3), 82)
stern = lay(stern, -6.0, math.radians(8), 83)
XC = -0.5 * (mid.v[:, 0].min() + mid.v[:, 0].max())
for m in (bow, mid, stern):
    m.T(XC, 0, 0)
debris = merge(plates(120, 0, zd(108), 26, 191, cols=(HULLC, SUPC, GREY_D))
               + [lifeboat().R(0.3, 0.6, 0.4).T(-14, 0.4, zd(90)), lifeboat().R(-0.2, 2.6, 0.3).T(-18, 0.3, zd(128)), lifeboat().R(0.1, 1.2, 2.9).T(16, 0.6, zd(100))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q155014" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", bow), ("hull_section", mid), ("stern_section", stern), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-46, kind="liner", seed=1945, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())])
