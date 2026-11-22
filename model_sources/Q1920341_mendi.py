"""SS Mendi (Q1920341) — the wreck of the troopship lost off the Isle of Wight in 1917, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a passenger-cargo steamer built in 1905 for the Elder Dempster West Africa service: about 111 m long and
14.7 m in the beam, a forecastle, a long midship house with the bridge and the boats, a single funnel, a poop aft,
two masts, one screw.

The wreck, after the published descriptions and the Wessex Archaeology survey for Historic England (2007-2008, with
multibeam bathymetry): carrying 823 men of the South African Native Labour Corps to France, she was rammed in thick fog
by the SS Darro on 21 February 1917 some 11 nautical miles south of St Catherine's Point and sank in about 20
minutes; 646 died, most of them black South African volunteers. She lies in about 40 m, roughly east-west with her
bow to the west, on her starboard side and broken down: the port side stands up from the seabed, the decks have
fallen in and her boilers lie exposed. She is a protected place under the Protection of Military Remains Act. The
amount of collapse is approximate.

    python model_sources/Q1920341_mendi.py   -> models/item/Q1920341/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1917 + 2)

L, B, D = 111.0, 14.7, 9.0
FC, TW, TT, WL = 2.4, 6.0, 1.2, 6.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.24, 0.75, 0.2, 0.42, 0.07, 0.3, 1.5, 0.6, 2.6, 7.0
HULLC = (0.1, 0.1, 0.11)
SUPC = (0.74, 0.72, 0.66)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.6, 0.45, 0.2)
HHW = 2.6
HATCHES = [(14, 19), (24, 30), (82, 88), (92, 97)]
BULKHEADS = [12, 36, 78, 100]
RAISED = [(0, 12), (36, 78), (100, L)]
HOUSES = [(39, 75, 0.74, 2.5, 0), (41, 56, 0.6, 2.5, 1), (42, 48, 0.46, 2.4, 2)]
FUNNELS = [(60.0, 2.0, 10.0, 0.85)]
MASTS = [(22.0, 20.0), (90.0, 18.0)]
BOATS = [(d, 1) for d in (44.0, 52.0, 66.0, 72.0)]
BOAT_X = 0.29
SCREWS = [(0.0, L - 5.0, 2.2, 1.9)]
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


def prune(m, min_faces=60, above=1.0, gap=0.2):
    """Drop the little loose bits left hanging in the water (window panes, slivers of plating) once the walls around
    them have been cut away: connected groups of fewer than min_faces faces, wholly above `above` m, that touch no
    other part (their boxes, grown by `gap`, overlap nothing else)."""
    par = np.arange(len(m.v))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for a, b, c in m.f:
        ra, rb, rc = find(a), find(b), find(c)
        par[rb] = ra
        par[find(rc)] = ra
    roots = np.array([find(a) for a in m.f[:, 0]])
    _, inv, cnt = np.unique(roots, return_inverse=True, return_counts=True)
    tri = m.v[m.f]
    lo = np.full((len(cnt), 3), 1e9); hi = np.full((len(cnt), 3), -1e9)
    np.minimum.at(lo, inv, tri.min(1)); np.maximum.at(hi, inv, tri.max(1))
    drop = np.zeros(len(cnt), bool)
    for k in np.nonzero((cnt < min_faces) & (lo[:, 1] > above))[0]:
        ov = np.all((lo <= hi[k] + gap) & (hi >= lo[k] - gap), axis=1)
        ov[k] = False
        drop[k] = not ov.any()
    m.f = m.f[~drop[inv]]
    return m


wreck = merge(ship_part(0, L, False, mast_left=[0.0, 0.0], funnel_state="gone", davits=False))
fc = wreck.v[wreck.f].mean(1)
# the starboard side and decks broken down; plating gone over the engine room so the boilers show
gone = ((fc[:, 0] < -1.0) & (fc[:, 1] > 2.0) & (fbm(fc, 0.1, 61) > 0.42)) | ((fc[:, 1] > D + FC) & (fbm(fc, 0.1, 62) > 0.22)) \
    | ((np.abs(L / 2 - fc[:, 2] - 66) < 9) & (fc[:, 0] < 2.5) & (fc[:, 1] > 2.0))
wreck.f = wreck.f[~gone]
bo = merge([boiler(2.1, 3.2).T(-0.6, 3.0, zd(d)) for d in (62.0, 67.0, 71.5)])
wreck = merge([wreck, bo])
lay(wreck, math.radians(78), 63, sink=0.8, squash=(9.5, 0.35))                   # on her starboard side, the port side standing
crumple(wreck, 0.3, 0.15, 64)
prune(wreck)
center_x(wreck)
debris = merge(plates(70, -6, zd(55), 16, 641, cols=(HULLC, SUPC, DECKC)) + [boiler(2.0, 3.0).R(0, 0.4, 0.2).T(-9.0, 1.6, zd(76))])
prune(debris)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1920341" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-40, kind="liner", seed=1917, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())], deck_colors=[DECKC])
