"""
MS Zenobia (Q2603065) — the wreck of the ro-ro ferry off Larnaca, Cyprus, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a roll-on/roll-off ferry built by Kockums in Sweden in 1979, about 172 m long and 28 m in beam, with
two vehicle decks, the accommodation and bridge forward, the funnels aft and a stern ramp; on her maiden voyage
she carried over a hundred articulated lorries.

The wreck, after the published descriptions and dive guides: a fault in her computerised ballast system gave her
a growing list to port; she was towed out of Larnaca harbour and sank on 7 June 1980; nobody died. She lies
whole on her port side in about 42 m, her upturned starboard side 16 m under the surface; lorries are still on
her two vehicle decks, some hanging from their chains, others scattered over the seabed. One of the world's best
wreck dives. The heading (not published: 0), the layout of the superstructure and of the lorries are approximate.

    python model_sources/Q2603065_zenobia.py   -> models/item/Q2603065/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1980)

L, B, D = 172.0, 27.6, 15.0
FC, TW, TT, WL = 0.0, 9.0, 1.5, 6.5
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.18, 0.6, 0.1, 0.75, 0.05, 0.25, 1.0, 0.2, 4.0, 9.0
HULLC = (0.18, 0.3, 0.55)                    # blue
SUPC = (0.88, 0.88, 0.86)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.85, 0.85, 0.83)
HHW = 3.5
HATCHES = []
BULKHEADS = [8]
RAISED = []
HOUSES = [(12, 58, 0.96, 2.6, 0), (13, 56, 0.94, 2.6, 1), (14, 52, 0.86, 2.6, 2), (15, 30, 0.7, 2.5, 3)]
FUNNELS = []
MASTS = []
BOATS = [(d, 2) for d in (36.0, 44.0, 52.0)]
BOAT_X = 0.43
SCREWS = [(-5.5, L - 6.0, 2.6, 2.3), (5.5, L - 6.0, 2.6, 2.3)]
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


install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship")


LEVEL_H = 2.6


def lorry(col):
    """An articulated lorry: tractor unit and a box trailer, front to +Z, on y = 0."""
    P = [box(2.5, 3.8, 12.5, col).T(0, 1.2, -1.5), box(2.4, 2.6, 2.2, (0.8, 0.8, 0.78)).T(0, 0.9, 6.3), box(2.0, 0.5, 5.0, BLACK).T(0, 0.6, 5.0)]
    for z in (-6.0, -4.8, 4.6, 6.8):
        for s in (-1, 1):
            P.append(wheel(0.5, 0.35).T(s * 1.05 - 0.175, 0.5, z))
    return merge(P)


def funnels():
    return [box(3.0, 7.0, 5.0, FUNNELC).T(s * 6.0, deck_y(150) + 2.5, zd(150)) for s in (-1, 1)] + [box(15.0, 2.5, 7.0, SUPC).T(0, deck_y(150), zd(150))]


TCOL = [(0.75, 0.75, 0.72), (0.65, 0.2, 0.15), (0.2, 0.35, 0.6), (0.8, 0.7, 0.3), (0.35, 0.5, 0.35)]
slots = [(d, x) for d in np.arange(75.0, 160.0, 15.0) for x in (-10.0, -5.0, 0.0, 5.0, 10.0)]
r = np.random.default_rng(80)
fallen = set(r.choice(len(slots), 9, replace=False).tolist())
deck_trucks, loose_trucks = [], []
for k, (d, x) in enumerate(slots):
    t = lorry(TCOL[k % len(TCOL)])
    if k in fallen:
        loose_trucks.append((t, d))
    else:
        if k % 7 == 3:                                                         # hanging askew from its chains
            t.R(0, 0, -0.35).T(0.6, 0.4, 0)
        deck_trucks.append(t.T(x, deck_y(d), zd(d)))
ramp = box(12.0, 9.0, 0.4, HULLC).T(0, deck_y(L) - 0.2, zd(L) - 0.2)
LIST = math.radians(90)                                                       # on her port side (+X)


def roll(m):
    return m.R(0, 0, -LIST)


wreck = merge(ship_part(0, L, False, houses_damage=lambda m: crumple(m, 0.2, 0.3, 61)) + deck_trucks + funnels() + [ramp])
crumple(wreck, 0.12, 0.1, 11)
roll(wreck)
LIFT = -np.percentile(wreck.v[:, 1], 0.5) - 1.0
wreck.T(0, LIFT, 0)
XC = -0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max())
wreck.T(XC, 0, 0)
onwreck = make_onwreck(roll, XC, LIFT)
side = onwreck(0, D + 4, 100)[0]
lt = []
for k, (t, d) in enumerate(loose_trucks):                                     # lorries scattered over the seabed
    crumple(t, 0.25, 0.4, 70 + k)
    lt.append(t.R(0, 0, r.choice([0.0, math.pi / 2, math.pi])).R(0, r.uniform(0, 6.28), 0).T(side + r.uniform(3, 18), 1.2 if k % 3 else 0.0, zd(d) + r.normal(0, 6)))
debris = merge(lt + plates(20, side + 8, zd(110), 15, 100, cols=(SUPC, HULLC)))
coral = merge([corals(wreck, 240, 7, size=(0.12, 0.4))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q2603065" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-42, kind="cargo", seed=1980)
all_trucks = [lorry(TCOL[k % len(TCOL)]).T(x, deck_y(d), zd(d)) for k, (d, x) in enumerate(slots)]
dress_built(out.with_name("model_built.glb"), [("main", merge([built_ship()] + all_trucks + funnels() + [ramp]))])
