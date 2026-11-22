"""Rainbow Warrior (Q627265) — the wreck of the Greenpeace ship scuttled off the Cavalli Islands, New Zealand, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: built in 1955 as the British research trawler Sir William Hardy, bought by Greenpeace in 1977 and turned
into the Rainbow Warrior: about 40 m long, a high bow, the wheelhouse and accommodation amidships with the funnel
behind them, rigged with two masts for sail, the hull painted green with a rainbow and a white dove at the bow.

The wreck, after the published descriptions and dive guides (Divernet, PADI, NZ Geographic, NZ History): sunk by two
bombs placed by French secret-service agents in Auckland harbour on 10 July 1985 (the photographer Fernando Pereira
died), she was raised, judged beyond repair and scuttled on 12 December 1987 between Matauri Bay and the Cavalli
Islands as a dive wreck and fish sanctuary. She sits almost upright on the sand at about 26-27 m with a slight list to
starboard, her masts removed; the superstructure rises to within about 15 m of the surface. The heading and the exact
list are approximate.

    python model_sources/Q627265_rainbow_warrior.py   -> models/item/Q627265/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1987)

L, B, D = 40.0, 7.3, 4.6
FC, TW, TT, WL = 1.8, 6.0, 1.0, 3.4
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.32, 0.7, 0.3, 0.35, 0.1, 0.4, 2.2, 0.6, 2.2, 5.0
HULLC = (0.16, 0.36, 0.24)                    # Greenpeace green
SUPC = (0.86, 0.86, 0.82)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.86, 0.86, 0.82)
HHW = 1.4
HATCHES = [(8, 11)]
BULKHEADS = [6, 13]
RAISED = [(0, 7)]
HOUSES = [(14, 27, 0.78, 2.4, 0), (15, 22, 0.6, 2.4, 1)]
FUNNELS = [(24.5, 0.8, 3.0, 0.8)]
MASTS = [(10.0, 20.0), (30.0, 16.0)]
BOATS = [(25.0, 1)]
BOAT_X = 0.26
SCREWS = [(0.0, L - 2.5, 1.4, 1.2)]
RUDDER_D = L - 1.0
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


def rainbow():
    """The rainbow and the dove at the bow, both sides."""
    out = []
    cols = [(0.75, 0.15, 0.12), (0.85, 0.45, 0.1), (0.85, 0.75, 0.15), (0.2, 0.55, 0.25), (0.15, 0.35, 0.7), (0.4, 0.2, 0.55)]
    for s in (-1, 1):
        for k, c in enumerate(cols):
            d = 3.0 + k * 0.7
            z = zd(d); y = deck_y(d) - 1.6 - k * 0.15
            out.append(quad(s * (hw(z, y) + 0.04), y, z, 0.6, 2.6 - k * 0.2, "+x" if s > 0 else "-x", c, eps=0.0))
        z = zd(9.0); y = deck_y(9.0) - 1.2
        out.append(quad(s * (hw(z, y) + 0.04), y, z, 1.6, 0.9, "+x" if s > 0 else "-x", (0.9, 0.9, 0.88), eps=0.0))
    return out


wreck = merge(ship_part(0, L, False, mast_left=[0.6, 0.6], funnel_state="stand", houses_damage=lambda m: crumple(m, 0.06, 0.4, 51)) + rainbow())
crumple(wreck, 0.05, 0.3, 52)
wreck.R(0, 0, math.radians(6))                                          # a slight list to starboard
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 0.6, 0)
debris = merge(plates(10, 5, zd(20), 6, 281, cols=(HULLC, SUPC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q627265" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-26, kind="cargo", seed=1987)
ship = merge([built_ship()] + rainbow())
dress_built(out.with_name("model_built.glb"), [("main", ship)])
