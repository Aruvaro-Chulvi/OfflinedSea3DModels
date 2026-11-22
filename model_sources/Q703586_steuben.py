"""SS General von Steuben (Q703586) — the wreck of the German liner sunk in the Baltic in 1945, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Norddeutscher Lloyd liner completed in 1923 as München, rebuilt after a fire and renamed General von
Steuben in 1931 (Steuben from 1938): about 168 m long and 19.8 m in the beam, a long white superstructure, two
funnels, two masts, two screws. In the war an accommodation ship and then a transport for the wounded, painted grey.

The wreck, after the published descriptions (Wikipedia, the 2004 Polish Navy identification, the remote-sensing study
of the wreck in Geosciences, 2020): carrying wounded soldiers and refugees from Pillau, she was hit by two torpedoes
from the Soviet submarine S-13 on the starboard bow, below the bridge, shortly after midnight on 10 February 1945 and
sank in about 20 minutes; some 4,000 died. Found in May 2004, she lies on her port side at about 70 m, her hull
rising to about 50 m. A war grave. The collapse of the superstructure and the heading (not published here: 0) are
approximate.

    python model_sources/Q703586_steuben.py   -> models/item/Q703586/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1945 + 10)

L, B, D = 168.0, 19.8, 12.5
FC, TW, TT, WL = 2.6, 8.0, 1.3, 8.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.26, 0.75, 0.24, 0.4, 0.07, 0.32, 1.8, 0.7, 2.5, 6.5
HULLC = (0.4, 0.41, 0.42)
SUPC = (0.5, 0.51, 0.51)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.45, 0.46, 0.46)
HHW = 3.0
HATCHES = [(16, 22), (28, 34), (146, 152), (156, 161)]
BULKHEADS = [13, 38, 142, 164]
RAISED = [(0, 13), (38, 142), (164, L)]
HOUSES = [(40, 138, 0.84, 2.5, 0), (44, 132, 0.8, 2.5, 1), (50, 120, 0.72, 2.5, 2), (50, 58, 0.5, 2.4, 3)]
FUNNELS = [(80.0, 3.0, 10.0, 0.72), (100.0, 3.0, 10.0, 0.72)]
MASTS = [(26.0, 22.0), (154.0, 20.0)]
BOATS = [(d, 2) for d in (64.0, 72.0, 88.0, 110.0, 118.0)]
BOAT_X = 0.31
SCREWS = [(-4.0, L - 8.0, 2.5, 2.0), (4.0, L - 8.0, 2.5, 2.0)]
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


# ── Damage helpers ───────────────────────────────────────────────────────────
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship collapse holes lay boiler center_x")


wreck = merge(ship_part(0, L, False, mast_left=[0.0, 0.0], funnel_state="stump", davits=False, houses_damage=lambda m: crumple(m, 0.3, 0.3, 91)))
holes(wreck, [(40.0, -B * 0.48, 4.0, 5.0), (50.0, -B * 0.47, 4.5, 4.0)], 92)            # the two torpedoes, starboard bow below the bridge
fc = wreck.v[wreck.f].mean(1)
wreck.f = wreck.f[~((fc[:, 1] > D + FC + 2.0) & (fbm(fc, 0.1, 93) > 0.55))]              # the upper houses rotting through
lay(wreck, math.radians(-88), 94, sink=0.8, squash=(18.0, 0.5))                          # on her port side: the hull stands some 20 m high
crumple(wreck, 0.2, 0.1, 95)
center_x(wreck)
fun = [cyl(3.0, 2.9, 9.0, FUNNELC, seg=18, sx=0.72).R(math.pi / 2, a, 0).T(x, 2.0, zd(d)) for x, d, a in ((14.0, 80.0, 0.2), (17.0, 102.0, -0.3))]
for f in fun:
    crumple(f, 0.5, 0.4, 96)
debris = merge(plates(80, 14, zd(80), 20, 951, cols=(HULLC, SUPC, DECKC)) + fun)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q703586" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-70, kind="liner", seed=1945, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())], deck_colors=[DECKC])
