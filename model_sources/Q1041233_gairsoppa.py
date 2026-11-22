"""SS Gairsoppa (Q1041233) — the wreck of the British India cargo steamer sunk in 1941 with a cargo of silver, built by
code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a British India Steam Navigation Company cargo steamer built in 1919: about 125 m long and 16.6 m in the
beam, a forecastle, two hatches forward, the bridge and engine house amidships with a single funnel, three hatches
aft and a poop, two masts with derricks, one screw.

The wreck, after the published descriptions (Wikipedia, Odyssey Marine Exploration, gCaptain): straggling from convoy
SL-64 in a storm, she was torpedoed by U-101 on 17 February 1941 some 300 miles south-west of Galway; one of her 86
crew survived. Odyssey Marine Exploration found her in September 2011 at nearly 4,700 m: she sits upright on the
bottom, intact, her holds open, the lifeboat cradles and the engine-room skylight still there. In 2012-2013 Odyssey
cut into her holds with its remotely operated vehicles and raised some 110 tonnes of silver. The exact position is
not published (the catalog pin is approximate); the damage, the openings cut into the holds and the heading (not
published here: 0) are approximate.

    python model_sources/Q1041233_gairsoppa.py   -> models/item/Q1041233/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1941 + 2)

L, B, D = 125.0, 16.6, 10.2
FC, TW, TT, WL = 2.4, 7.0, 1.3, 7.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.24, 0.75, 0.2, 0.42, 0.07, 0.3, 1.6, 0.6, 2.6, 7.0
HULLC = (0.12, 0.12, 0.13)
SUPC = (0.6, 0.57, 0.5)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.12, 0.12, 0.13)
HHW = 3.0
HATCHES = [(15, 22), (28, 36), (74, 81), (86, 93), (98, 105)]
BULKHEADS = [12, 42, 68, 110]
RAISED = [(0, 12), (42, 68), (110, L)]
HOUSES = [(44, 66, 0.72, 2.5, 0), (46, 62, 0.62, 2.5, 1), (46, 52, 0.5, 2.4, 2)]
FUNNELS = [(58.0, 2.2, 10.0, 0.85)]
MASTS = [(26.0, 20.0), (90.0, 18.0)]
BOATS = [(d, 1) for d in (54.0, 62.0)]
BOAT_X = 0.28
SCREWS = [(0.0, L - 5.0, 2.4, 2.1)]
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


wreck = merge(ship_part(0, L, False, mast_left=[6.0, 0.0], funnel_state="stump", houses_damage=lambda m: crumple(m, 0.2, 0.3, 21)))
holes(wreck, [(70.0, -B * 0.49, 3.5, 4.5)], 22)                                               # the torpedo, in way of hold 3
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
cut = (np.abs(fd_ - 32.0) < 4.5) & (fc[:, 1] > D - 0.5) & (np.abs(fc[:, 0]) < B * 0.4)           # Odyssey's opening into hold 2
wreck.f = wreck.f[~cut]
crumple(wreck, 0.1, 0.1, 23)
wreck.R(0, 0, math.radians(2))
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 1.0, 0)
bars = merge([box(0.5, 0.15, 0.12, (0.62, 0.62, 0.6)).R(0, rng.uniform(0, 6.28), 0).T(rng.normal(-12, 1.5), 0.05, zd(32) + rng.normal(0, 2.5)) for _ in range(40)])
debris = merge(plates(40, -12, zd(40), 10, 231, cols=(HULLC, GREY_D, DECKC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1041233" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": [("silver", bars, "artefact")]}],
            depth=-4700, kind="cargo", seed=1941, memorial=True, drift=0.4)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())])
