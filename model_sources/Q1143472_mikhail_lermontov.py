"""
MS Mikhail Lermontov (Q1143472) — the wreck of the Soviet cruise liner in Port Gore, New Zealand, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the last of the five Ivan Franko-class liners, built at Wismar (East Germany) in 1972 for the Baltic
Shipping Company: about 176 m long, beam 23.6 m, twin screws, a long white superstructure of several decks, the
bridge forward, one broad funnel aft with the red band and the hammer and sickle, cargo hatches forward and aft.

The wreck, after the published descriptions and dive guides: on 16 February 1986, on a cruise round the
Marlborough Sounds, she struck rocks off Cape Jackson and sank in Port Gore; one crewman died. She lies whole on
her starboard side in 36-37 m, her port side 12 m under the surface, the funnel with its hammer and sickle still
in place; the interior is open to technical divers. Cold-water life: anemones, sponges, kelp-fish. The heading
(not published: 0), the colours and the exact layout of the decks are approximate.

    python model_sources/Q1143472_mikhail_lermontov.py   -> models/item/Q1143472/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1986)

L, B, D = 175.8, 23.6, 13.0
FC, TW, TT, WL = 2.6, 8.0, 1.4, 7.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.28, 0.32, 0.08, 0.35, 2.2, 0.8, 2.4, 6.0
HULLC = (0.86, 0.86, 0.84)                   # white
SUPC = (0.88, 0.88, 0.86)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.86, 0.86, 0.84)
HHW = 3.5
HATCHES = [(14, 21), (24, 30), (157, 164)]
BULKHEADS = [12, 32, 154]
RAISED = [(0, L)]
HOUSES = [(36, 150, 0.92, 2.5, 0), (38, 146, 0.9, 2.5, 1), (40, 138, 0.84, 2.5, 2), (42, 120, 0.6, 2.5, 3), (42, 50, 0.6, 2.4, 4)]
FUNNELS = [(112.0, 3.6, 9.0, 0.6)]
MASTS = [(22.5, 12.0)]
BOATS = [(d, 3) for d in (58.0, 66.0, 74.0, 82.0, 90.0, 98.0)]
BOAT_X = 0.31
SCREWS = [(-4.6, L - 12.0, 2.6, 2.2), (4.6, L - 12.0, 2.6, 2.2)]
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


def funnel_paint(m, d, r, h, yb):
    """The red band and a gold hammer and sickle on both sides of the funnel, black top."""
    out = [m]
    for s in (-1, 1):
        x = s * (r * 0.6 + 0.08)
        out.append(quad(x, yb + h * 0.62, zd(d), 2 * r * 0.95, 2.0, "+x" if s > 0 else "-x", (0.62, 0.1, 0.08), eps=0.0))
        out.append(quad(x + s * 0.02, yb + h * 0.64, zd(d), 1.1, 1.1, "+x" if s > 0 else "-x", (0.85, 0.68, 0.2), eps=0.0))
    return out


def radar_mast(built):
    P = [cyl(0.3, 0.2, 9.0, SUPC, seg=8), box(3.0, 0.25, 0.4, SUPC).T(0, 6.0, 0)]
    if built:
        P.append(box(4.5, 0.3, 0.6, (0.3, 0.3, 0.3)).T(0, 8.5, 0))
    return merge(P)


yfun = deck_y(112) + FC + LEVEL_H * 3 + 2.5 - 0.2          # on the roof of the boat-deck house, like the funnel
LIST = math.radians(88)                                                       # on her starboard side (−X)


def roll(m):
    return m.R(0, 0, LIST)


extra = funnel_paint(Mesh(), 112.0, 3.6, 9.0, yfun)[1:] + [radar_mast(False).T(0, deck_y(46) + FC + LEVEL_H * 5, zd(46))]
wreck = merge(ship_part(0, L, False, mast_left=[3.0], funnel_state="stand", houses_damage=lambda m: crumple(m, 0.2, 0.3, 61)) + extra)
crumple(wreck, 0.14, 0.1, 11)
roll(wreck)
LIFT = -np.percentile(wreck.v[:, 1], 0.5) - 1.0
wreck.T(0, LIFT, 0)
XC = -0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max())
wreck.T(XC, 0, 0)
onwreck = make_onwreck(roll, XC, LIFT)
side = onwreck(0, D + 14, 90)[0]
debris = merge(plates(25, side - 5, zd(80), 20, 100) + [lifeboat().R(0.3, 0.6, 0.4).T(side - 8, 0.4, zd(70)), lifeboat().R(-0.2, 2.6, 0.3).T(side - 12, 0.3, zd(95))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1143472" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-36, kind="liner", seed=1986)
ship = built_ship()
ship = merge([ship] + funnel_paint(Mesh(), 112.0, 3.6, 9.0, yfun)[1:] + [radar_mast(True).T(0, deck_y(46) + FC + LEVEL_H * 5, zd(46))])
dress_built(out.with_name("model_built.glb"), [("main", ship)])
