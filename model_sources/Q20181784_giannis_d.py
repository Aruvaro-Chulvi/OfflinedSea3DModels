"""
Giannis D (Q20181784) — the wreck of the Greek cargo ship on Sha'ab Abu Nuhas, Red Sea, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a general cargo ship built in Japan in 1969 (as Shoyo Maru), about 99.5 m long, engines and
accommodation aft with the funnel (marked with a large D), four holds with derrick posts, raised forecastle.

The wreck, after the published descriptions and dive guides: loaded with timber, she ran at full speed onto the
north-west corner of Sha'ab Abu Nuhas on 19 April 1983; nobody died. She lies in three sections, roughly
parallel to the reef: the bow in about 10 m, the middle broken down and flattened with timber about it, and the
stern section on its port side down to about 25 m, the propeller on the bottom near 30 m; the accommodation and
engine room, at an angle, are the classic swim-through. On Offlined's flat seabed the three pieces lie at the
mean depth. The heading (not published: 0), the angles of the pieces and the colours are approximate.

    python model_sources/Q20181784_giannis_d.py   -> models/item/Q20181784/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1983)

L, B, D = 99.5, 16.0, 8.6
FC, TW, TT, WL = 2.3, 5.4, 1.1, 5.5
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.2, 0.7, 0.22, 0.22, 0.1, 0.5, 2.0, 1.0, 2.5, 7.0
HULLC = (0.12, 0.2, 0.3)                     # dark blue
SUPC = (0.84, 0.83, 0.8)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.15, 0.25, 0.45)
HHW = 4.0
HATCHES = [(12, 24), (28, 40), (44, 56), (60, 71)]
BULKHEADS = [10.5, 26, 42, 58, 73.5]
RAISED = [(0, 10), (73.5, L)]
HOUSES = [(76, 92, 0.86, 2.5, 0), (77, 91, 0.76, 2.5, 1), (77.5, 88, 0.66, 2.5, 2), (77.5, 82, 0.62, 2.4, 3)]
FUNNELS = [(87.0, 2.0, 5.5, 0.8)]
MASTS = [(26.0, 13.0), (42.0, 12.0), (58.0, 12.0)]
BOATS = [(84.0, 2)]
BOAT_X = 0.32
SCREWS = [(0.0, L - 3.5, 2.1, 1.8)]
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


def letter_d(r, h):
    """The big white D on both sides of the funnel (block letter, as plates)."""
    out = []
    for s in (-1, 1):
        x = s * (r * 0.8 + 0.06)
        for (dz, dy, w, hh) in ((-0.9, 0.0, 0.5, 3.0), (0.0, 1.3, 1.6, 0.45), (0.0, -1.3, 1.6, 0.45), (0.9, 0.0, 0.5, 2.2)):
            out.append(quad(x, h + dy, dz, w, hh, "+x" if s > 0 else "-x", (0.92, 0.92, 0.9), eps=0.0))
    return merge(out)


def roll_by(m, ang, about_d):
    """Roll a piece onto its port side (+X) by ang, keel on the bottom, centred on its own length."""
    m.T(0, 0, -zd(about_d)).R(0, 0, -ang)
    m.T(0, -np.percentile(m.v[:, 1], 0.5) - 0.8, 0)
    return m


yF = deck_y(87) + FC + LEVEL_H * 2 + 2.5 - 0.2              # the base of the funnel (on the roof of the house under it)
dmesh = letter_d(2.0, 3.4).T(0, yF, zd(87.0))
bow = merge(ship_part(0, 30, False, mast_left=[9.0, 0, 0]))
mid = merge(ship_part(30, 73, False, mast_left=[0, 6.0, 0.0]))
stern = merge(ship_part(73, L, False, funnel_state="stand", houses_damage=lambda m: crumple(m, 0.25, 0.3, 61)) + [dmesh])
for k, m in enumerate((bow, mid, stern)):
    crumple(m, 0.15, 0.12, 11 + k)
dd = L / 2 - bow.v[:, 2]; crumple(bow, 1.4, 0.3, 21, sstep(25, 30, dd))
dd = L / 2 - stern.v[:, 2]; crumple(stern, 1.4, 0.3, 22, sstep(78, 73, dd))
dd = L / 2 - mid.v[:, 2]
mid.v[:, 1] *= 0.35 + 0.1 * fbm(mid.v, 0.1, 5)                              # the middle broken down and flattened
mid.v[:, 0] *= 1.12
crumple(mid, 1.2, 0.2, 23)
mid.f = mid.f[fbm(mid.v[mid.f].mean(1), 0.15, 24) < 0.66]
roll_by(bow, math.radians(25), 15).T(0, 0, zd(15) + 6.0)
mid.T(0, 0, 0)
roll_by(stern, math.radians(55), 86).T(-2.0, 0, zd(86) - 5.0)
stern.T(0, 0, 0)
r = np.random.default_rng(83)
timber = merge([box(0.15, 0.1, r.uniform(3, 6), (0.55, 0.45, 0.3)).R(0, r.uniform(0, 6.28), r.uniform(-0.2, 0.2)).T(r.normal(8, 9), 0.05 + 0.1 * (k % 4), zd(r.uniform(30, 75)))
                for k in range(160)])
deb = plates(40, 6, zd(52), 14, 100)
debris = merge(deb)
coral = merge([corals(bow, 220, 7), corals(mid, 160, 8), corals(stern, 380, 9)])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q20181784" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", bow), ("midships", mid), ("stern_section", stern), ("debris_field", debris, "loose")], "mud": [],
                   "plain": [("timber", timber, "artefact"), ("corals", coral, "life")]}],
            depth=-22, kind="cargo", seed=1983, life="tropical")
dress_built(out.with_name("model_built.glb"), [("main", merge([built_ship(), letter_d(2.0, 3.4).T(0, yF, zd(87.0))]))])
