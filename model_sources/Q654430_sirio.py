"""SS Sirio (Q654430) — the wreck of the Italian emigrant liner lost off Cabo de Palos in 1906, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: an iron passenger steamer built in Glasgow in 1883 for Navigazione Generale Italiana on the run from Genoa to
South America: about 115.8 m long and 12.8 m in the beam, a straight stem, a long flush deck with deck houses, a
single funnel, three masts that could still set sail, one screw.

The wreck, after the published descriptions (Wikipedia, the Murcia tourist board, the Cabo de Palos-Islas Hormigas
marine reserve, the history of the disaster in Navegación e historia de la ciencia): on 4 August 1906, carrying some
1,700 people, most of them emigrants, she ran at full speed onto the Bajo de Fuera, a ridge of rocks off the Hormigas
islands, and sank by the stern within minutes, her bow left out of the water and listing; somewhere between 150 and
more than 500 died. Beaten by the sea since, she lies broken on the steep flanks of the shoal between about 40 and 70
m, her stern the deepest. The model sets the two halves on flat ground at about 50 m: the slope, the break and the
heading (not published here: 0) are approximate.

    python model_sources/Q654430_sirio.py   -> models/item/Q654430/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1906 + 8)

L, B, D = 115.8, 12.8, 8.6
FC, TW, TT, WL = 2.2, 5.6, 1.1, 6.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.22, 0.75, 0.22, 0.35, 0.08, 0.32, 1.3, 0.6, 2.6, 7.0
HULLC = (0.1, 0.1, 0.11)
SUPC = (0.72, 0.7, 0.64)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.1, 0.1, 0.11)
HHW = 2.4
HATCHES = [(14, 19), (26, 31), (84, 89), (96, 101)]
BULKHEADS = [10, 104]
RAISED = [(0, 10), (104, L)]
HOUSES = [(36, 78, 0.6, 2.4, 0), (42, 52, 0.5, 2.3, 1)]
FUNNELS = [(62.0, 2.0, 10.0, 0.85)]
MASTS = [(22.0, 26.0), (46.0, 28.0), (92.0, 24.0)]
BOATS = [(d, 0) for d in (40.0, 48.0, 70.0, 76.0)]
BOAT_X = 0.36
SCREWS = [(0.0, L - 4.5, 2.0, 1.8)]
RUDDER_D = L - 1.8
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


bow = merge(ship_part(0, 54, False, mast_left=[0.0, 0.0, 0.0], funnel_state="gone", davits=False))
stern = merge(ship_part(57, L, False, mast_left=[0.0, 0.0, 0.0], funnel_state="gone", davits=False))
for k, m in enumerate((bow, stern)):
    fc = m.v[m.f].mean(1)
    m.f = m.f[~((fc[:, 1] > 3.0) & (fbm(fc, 0.09, 81 + k) > 0.5))]                # the iron plating beaten through by a century of storms
    collapse(m, D - 1.0, 0.4, 83 + k, rot=0.5)
holes(bow, [(6.0, 0.0, 0.5, 4.0)], 85)                                                # the forefoot torn open on the rocks
lay(bow, math.radians(28), 86, dz=6.0, yaw=math.radians(-12), sink=1.0)
lay(stern, math.radians(-12), 87, dz=-3.0, yaw=math.radians(10), sink=1.5)
prune(bow); prune(stern)
center_x(stern, bow)
bo = merge([boiler(2.0, 3.0).R(0, rng.uniform(0, 3), 0).T(rng.normal(4, 3), 1.6, zd(d)) for d in (58.0, 61.0)])
debris = prune(merge(plates(70, 0, zd(56), 14, 861, cols=(HULLC, SUPC, DECKC)) + [bo]))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q654430" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", bow), ("stern_section", stern), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-50, kind="liner", seed=1906, memorial=True, drift=0.4)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())], deck_colors=[DECKC])
