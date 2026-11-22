"""
RMS Rhone (Q1664018) — the wreck of the Royal Mail steamer off Salt Island, British Virgin Islands, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Royal Mail Steam Packet Company iron steamer launched in 1865, about 95 m long, beam 12 m, with a
single screw (a four-bladed bronze propeller), two masts with sails, one funnel, clipper bow with bowsprit.

The wreck, after the published descriptions and dive guides: caught by a hurricane on 29 October 1867 she was
driven onto Black Rock Point; her boilers exploded when the sea rushed in and she broke in two; most of the 147
aboard died. The bow section lies on its side in about 24 m, still whole enough to swim through, with the
foremast and its crow's nest beside it; the stern section, in 9-10 m, is more broken up (the Royal Navy
demolished part of it in 1950), with the great propeller, the shaft running to the engine and gearbox and the
rudder; the boilers are strewn between. A marine park. On Offlined's flat seabed the pieces lie at the mean
depth. The heading (not published: 0), the gaps between the pieces and the rig are approximate.

    python model_sources/Q1664018_rms_rhone.py   -> models/item/Q1664018/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1867)

L, B, D = 94.5, 12.1, 7.6
FC, TW, TT, WL = 2.0, 4.6, 1.0, 4.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.28, 0.75, 0.3, 0.3, 0.08, 0.4, 1.6, 0.9, 2.2, 5.0
HULLC = (0.09, 0.09, 0.1)                    # black
SUPC = (0.85, 0.84, 0.8)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 2.4
HATCHES = []
BULKHEADS = [44.0]
RAISED = []
HOUSES = [(38, 54, 0.55, 2.3, 0)]
FUNNELS = [(48.0, 1.5, 8.5, 0.9)]
MASTS = [(25.0, 25.0), (66.0, 23.0)]
BOATS = [(42.0, 1), (52.0, 1)]
BOAT_X = 0.28
SCREWS = [(0.0, L - 3.2, 2.2, 2.3)]
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


FC = 0.0                                                                      # flush deck: houses stand on the main deck


def beams_and_frames(d0, d1, deck_left):
    """The iron deck beams across the hull and the frames up the sides, where the wooden deck has rotted away."""
    out = []
    for d in np.arange(d0 + 0.6, d1 - 0.3, 1.2):
        y = deck_y(d); w = half(d) - 0.1
        out.append(box(2 * w, 0.22, 0.14, (0.25, 0.2, 0.17)).T(0, y - 0.25, zd(d)))
        if rng.random() < deck_left:
            out.append(box(2 * w, 0.08, 1.1, (0.45, 0.36, 0.25)).T(0, y - 0.03, zd(d)))
    return out


def bowsprit(built):
    p = [tube([(0, deck_y(0) + 0.6, zd(0) + 0.5), (0, deck_y(0) + 2.6, zd(0) + 10.0)], 0.25, (0.4, 0.33, 0.25), seg=6)]
    return p if built else []


def hull_part(d0, d1, built):
    P = [hull(d0, d1)]
    if not built:
        P += beams_and_frames(d0, d1, 0.12)
    return P


def ship_part2(d0, d1, built):
    P = hull_part(d0, d1, built)
    sp = ship_part(d0, d1, built, mast_left=[0.0, 0.0], funnel_state="gone")
    P += sp[1:] if built else sp[2:3] + sp[4:]                           # the wreck: no wooden deck, no 'tween deck
    if d0 == 0:
        P += bowsprit(built)
    return P


BRK0, BRK1 = 46.0, 70.0
LIST = math.radians(80)


def roll(m):
    return m.R(0, 0, LIST)                                                   # the bow section on its side (starboard, after dive guides)


bow = merge([m for m in ship_part2(0, BRK0, False) if m.tris])
crumple(bow, 0.12, 0.12, 11)
crumple(bow, 1.3, 0.3, 21, sstep(BRK0 - 5, BRK0, L / 2 - bow.v[:, 2]))
bow.T(0, 0, -zd(BRK0 / 2)); roll(bow)
bow.T(0, -np.percentile(bow.v[:, 1], 0.5) - 0.9, 0)
XB = -0.5 * (bow.v[:, 0].min() + bow.v[:, 0].max()); bow.T(XB, 0, zd(BRK0 / 2) + 8.0)
# the stern: the lower hull broken open, the shaft, the great propeller and the rudder
st = merge([m for m in ship_part2(BRK1, L, False) if m.tris])
dd = L / 2 - st.v[:, 2]
st.v[:, 1] = np.where(st.v[:, 1] > 3.0, 3.0 + (st.v[:, 1] - 3.0) * (0.25 + 0.5 * sstep(L - 12, L, dd)), st.v[:, 1])
st.v[:, 0] *= 1 + 0.15 * sstep(2.5, 6.0, st.v[:, 1]) * sstep(L - 8, BRK1, dd)
crumple(st, 0.9, 0.25, 22)
st.f = st.f[(fbm(st.v[st.f].mean(1), 0.12, 23) < 0.66) | (st.v[st.f][:, :, 2].mean(1) < zd(L - 10))]
st.R(0, 0, 0.12).T(0, -0.4, 0)
shaft = cyl(0.32, 0.32, L - 4 - BRK1 + 6, STEEL, seg=10).R(math.pi / 2, 0, 0).T(0, 1.4, zd(L - 4))
engine = merge([box(3.4, 3.6, 6.0, GREY_D).T(0, 0, zd(BRK1 - 4)), box(2.4, 1.6, 2.4, GREY_D).T(0, 0, zd(BRK1 + 1)),
                cyl(1.6, 1.6, 0.6, BRONZE, seg=16).R(0, 0, math.pi / 2).T(0.3, 1.6, zd(BRK1 + 1))])
crumple(engine, 0.3, 0.4, 24)
r = np.random.default_rng(67)
boilers = []
for k in range(4):                                                            # the boilers, burst and strewn
    b = cyl(1.9, 1.9, 4.0, GREY_D, seg=16, caps=k % 2 == 0).R(0, 0, math.pi / 2)
    crumple(b, 0.6, 0.3, 30 + k)
    boilers.append(b.R(0, r.uniform(0, 6.28), 0).T(r.normal(0, 6), 1.5, zd(r.uniform(BRK0 + 2, BRK1 - 2))))
stern = merge([st, shaft, engine])
onb = lambda x, y, d: roll(Mesh([(x, y, zd(d) - zd(BRK0 / 2))], [])).v[0]
mast_ = merge([cyl(0.36, 0.22, 22.0, (0.4, 0.33, 0.25), seg=8, caps=False), cyl(0.9, 0.9, 1.0, (0.4, 0.33, 0.25), seg=10, caps=False).T(0, 15.5, 0)])
crumple(mast_, 0.1, 0.4, 60)
mast_.R(0, 0, math.pi / 2 - 0.05).R(0, 0.12, 0).T(-8.0 + XB, 0.4, zd(25.0) + 8.0)   # the foremast with its crow's nest beside the bow
debris = merge(boilers + [mast_] + plates(40, 0, zd(58), 10, 100, cols=((0.4, 0.33, 0.25), HULLC)))
coral = merge([corals(bow, 360, 7), corals(stern, 260, 8), corals(debris, 60, 9)])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1664018" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", bow), ("stern_section", stern), ("debris_field", debris, "loose")], "mud": [],
                   "plain": [("corals", coral, "life")]}],
            depth=-18, kind="liner", seed=1867, memorial=True, life="tropical")
ship = merge(ship_part2(0, L, True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
