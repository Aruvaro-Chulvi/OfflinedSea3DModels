"""MV Salem Express (Q1417751) — the wreck of the Red Sea ferry lost off Safaga in 1991, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a roll-on/roll-off passenger ferry built in France in 1964 as Fred Scamaroni: about 115 m long and 17.8 m in
the beam, a bow visor and ramp for the car deck, a tall superstructure with the bridge forward, two funnels side by
side aft (each marked with a large S under her last owners), one screw on each side.

The wreck, after the published descriptions (Wikipedia, dive guides): on the night of 16-17 December 1991, bringing
pilgrims home from Mecca to Safaga, she struck the reef of Hyndman in a storm, holing her below the waterline; the car
deck flooded and she capsized and sank within minutes; officially 470 died, probably many more. She lies on her
starboard side in about 30-32 m near the Hyndman reef, largely intact, bow and screws included, with lifeboats lying on
the sand beside her. The Egyptian authorities and many dive operators ask divers not to enter her: she is a grave. The
heading (not published here: 0) is approximate.

    python model_sources/Q1417751_salem_express.py   -> models/item/Q1417751/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1991 + 12)

L, B, D = 115.0, 17.8, 10.0
FC, TW, TT, WL = 0.0, 5.5, 1.2, 5.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.24, 0.8, 0.14, 0.62, 0.07, 0.3, 1.2, 0.3, 2.6, 7.0
HULLC = (0.82, 0.82, 0.8)
SUPC = (0.86, 0.86, 0.84)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.8, 0.8, 0.78)
HHW = 2.0
HATCHES = []
BULKHEADS = []
RAISED = []
HOUSES = [(10, 104, 0.98, 2.8, 0), (12, 100, 0.94, 2.6, 1), (14, 92, 0.9, 2.6, 2), (14, 30, 0.7, 2.4, 3)]
FUNNELS = []
MASTS = [(20.0, 6.0)]
BOATS = [(d, 2) for d in (40.0, 48.0, 56.0, 64.0, 72.0)]
BOAT_X = 0.46
SCREWS = [(-3.5, L - 6.0, 2.0, 1.8), (3.5, L - 6.0, 2.0, 1.8)]
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


# ── Damage helpers ───────────────────────────────────────────────────────────
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship collapse holes lay boiler center_x")


def letter_s(col):
    """A big «S» in blocks, standing in the x=0 plane (on the side of a funnel)."""
    bars = [(0, 1.6, 1.4, 0.35), (0, 0.0, 1.4, 0.35), (0, -1.6, 1.4, 0.35), (-0.55, 0.8, 0.35, 1.6), (0.55, -0.8, 0.35, 1.6)]
    return merge([box(0.06, h, w, col).T(0, y, x) for x, y, w, h in bars])


def funnels(built):
    P = []
    for s in (-1, 1):
        y0 = deck_y(86) + 2.8 + 2.6 * 2
        f = box(3.0, 7.0, 6.0, FUNNELC).T(s * 4.5, y0, zd(86))
        f.v[:, 2] += 0.25 * (f.v[:, 1] - y0)                                   # raked aft
        f.c[f.v[:, 1] > y0 + 6.2] = BLACK
        P.append(f)
        P.append(letter_s((0.2, 0.3, 0.6)).T(s * 6.05, y0 + 3.8, zd(86) + 1.0))
    return P


def visor(open_):
    v = box(B * 0.62, D * 0.8, 6.0, HULLC)
    v.v[:, 0] *= 1 - 0.7 * np.clip(-v.v[:, 2] / 3.0, 0, 1)
    if open_:
        return v.T(0, 0, 3.0).R(-1.0, 0, 0).T(0, D * 0.55, zd(0) + 1.0)
    return v.T(0, D * 0.35, zd(2.5))


wreck = merge(ship_part(0, L, False, davits=True) + funnels(False) + [visor(False)])
holes(wreck, [(14.0, -B * 0.45, 2.0, 2.5)], 40)                                    # holed on the reef, starboard bow (approximate)
lay(wreck, math.radians(88), 41, sink=0.6)                                         # on her starboard side
crumple(wreck, 0.06, 0.1, 42)
center_x(wreck)
boats = merge([lifeboat().R(0.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.4, 0.4)).T(-rng.uniform(12, 16), 0.55, zd(d)) for d in (38.0, 50.0, 66.0)])
debris = merge(plates(20, -10, zd(50), 8, 421, cols=(HULLC, SUPC)) + [boats])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1417751" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-30, kind="liner", seed=1991, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", merge([built_ship(), merge(funnels(True) + [visor(False)]).T(0, 0, -zd(L / 2))]))])
