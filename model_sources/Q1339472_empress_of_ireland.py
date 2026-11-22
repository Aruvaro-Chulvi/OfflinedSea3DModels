"""
RMS Empress of Ireland (Q1339472) — the wreck of the Canadian Pacific liner in the St Lawrence, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: built by Fairfield at Glasgow in 1906 for Canadian Pacific: about 173 m long, beam 20 m, twin screws,
two funnels and two masts, black hull and white superstructure, holds forward and aft.

The wreck, after the published descriptions: rammed by the collier Storstad in fog off Pointe-au-Père on 29 May
1914, she sank in fourteen minutes; 1,012 people died. She lies on her starboard side in about 40 m of cold,
dark water with strong currents. Years of currents, salvage and corrosion have taken their toll: the funnels and
the superstructure have rusted away; the passage cut into the hull by the 1914 salvors (to reach the purser's
office and the mail) is still open. A national historic site since 2009 and a grave. The list, the heading (not
published: 0), the extent of the collapse and the colours are approximate.

    python model_sources/Q1339472_empress_of_ireland.py   -> models/item/Q1339472/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1914)

L, B, D = 173.0, 20.1, 11.5
FC, TW, TT, WL = 2.5, 7.0, 1.3, 7.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.26, 0.7, 0.28, 0.25, 0.08, 0.45, 1.8, 1.0, 2.5, 6.5
HULLC = (0.09, 0.09, 0.1)                    # black
SUPC = (0.86, 0.85, 0.82)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.72, 0.6, 0.38)
HHW = 3.5
HATCHES = [(22, 31), (34, 42), (138, 147), (150, 158)]
BULKHEADS = [19, 32.5, 44, 136, 148.5, 161]
RAISED = [(0, 19), (44, 136), (161, L)]
HOUSES = [(46, 132, 0.86, 2.5, 0), (48, 126, 0.78, 2.5, 1), (50, 118, 0.6, 2.5, 2), (50, 55, 0.6, 2.4, 3)]
FUNNELS = [(76.0, 2.8, 12.0, 0.72), (100.0, 2.8, 12.0, 0.72)]
MASTS = [(32.5, 26.0), (148.5, 24.0)]
BOATS = [(d, 3) for d in (62.0, 69.0, 86.0, 93.0, 108.0, 115.0)]
BOAT_X = 0.3
SCREWS = [(-3.8, L - 10.0, 2.6, 2.4), (3.8, L - 10.0, 2.6, 2.4)]
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


LIST = math.radians(70)                                                       # on her starboard side (−X)


def roll(m):
    return m.R(0, 0, LIST)


def rot_houses(m):                                                            # the superstructure rusted away: low, broken remains
    crumple(m, 0.8, 0.25, 61)
    m.f = m.f[(m.v[m.f][:, :, 1].mean(1) < D + FC + 4.0) & (fbm(m.v[m.f].mean(1), 0.1, 62) < 0.62)]


wreck = merge(ship_part(0, L, False, mast_left=[0.0, 0.0], funnel_state="gone", houses_damage=rot_houses, davits=False))
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.22, 0.1, 11)
top = wreck.v[:, 1] > D - 1.0
crumple(wreck, 0.7, 0.2, 12, top.astype(float))                              # the decks sagging and broken
gone = (wreck.v[wreck.f][:, :, 1].mean(1) > D) & (fbm(wreck.v[wreck.f].mean(1), 0.08, 13) > 0.64)
wreck.f = wreck.f[~gone]
# the 1914 salvage passage cut into the port side amidships (towards the purser's office and the mail room)
cut = merge([quad(half(88) + 0.05, D - 2.0, zd(d), 1.8, 1.6, "+x", DARK) for d in (84.0, 86.0, 88.0)])
wreck = merge([wreck, cut])
roll(wreck)
LIFT = -np.percentile(wreck.v[:, 1], 0.5) - 1.2
wreck.T(0, LIFT, 0)
XC = -0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max())
wreck.T(XC, 0, 0)
onwreck = make_onwreck(roll, XC, LIFT)
side = onwreck(0, D + 4, 90)[0]
deb = plates(90, side - 7, zd(90), 22, 100)
for k, d in enumerate((76.0, 100.0)):
    f = cyl(2.8, 2.7, 9.0, FUNNELC, seg=18, sx=0.72, caps=False); crumple(f, 0.7, 0.3, 50 + k); f.v[:, 0] *= 0.5
    f.f = f.f[fbm(f.v[f.f].mean(1), 0.2, 52 + k) < 0.62]
    deb.append(f.R(0, 0, math.pi / 2).R(0, 0.3 * (1 - 2 * k), 0).T(side - 6, 0.8, zd(d)))
for k, (d, ln) in enumerate(((32.5, 22.0), (148.5, 20.0))):
    base = onwreck(0, deck_y(d), d)
    m = cyl(0.42, 0.24, ln, GREY, seg=8, caps=False); crumple(m, 0.12, 0.4, 60 + k)
    deb.append(m.R(0, 0, math.pi / 2 - 0.05).R(0, 0.25 * (1 - 2 * k), 0).T(base[0] - 1.5, 0.3, base[2]))
debris = merge(deb)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1339472" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-40, kind="liner", seed=1914, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())])
