"""RMS Carpathia (Q504096) — the wreck of the Cunard liner that rescued the Titanic's survivors, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Cunard passenger liner completed in 1903: about 164.6 m long and 19.5 m in the beam, a forecastle, two
well decks with their hatches, a long midship superstructure with the bridge and the boats, a poop aft, a single tall
funnel in Cunard red with its black top, four masts and two screws. In April 1912 she picked up the 705 survivors of
the Titanic.

The wreck, after the published descriptions (Wikipedia, Britannica, Marine Link, MaritimeQuest): on 17 July 1918,
sailing in convoy from Liverpool to Boston, she was hit by three torpedoes from U-55 on her port side about 120 miles
west of Fastnet and sank in about two hours; five of her crew died. The wreck was found in 2000 by an expedition of
the National Underwater and Marine Agency (NUMA, Clive Cussler) and identified in September 2000: she sits upright
and largely intact on the bottom at about 150-160 m. Her masts and funnel are gone and her midship houses have
fallen in. The amount of collapse, the torpedo damage and the heading (not published here: 0) are approximate.

    python model_sources/Q504096_carpathia.py   -> models/item/Q504096/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1918 + 7)

L, B, D = 164.6, 19.5, 12.0
FC, TW, TT, WL = 2.5, 8.0, 1.3, 8.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.24, 0.75, 0.2, 0.42, 0.07, 0.3, 1.8, 0.7, 2.6, 7.0
HULLC = (0.09, 0.09, 0.1)                     # Cunard black hull, white superstructure
SUPC = (0.78, 0.77, 0.72)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.66, 0.3, 0.13)
HHW = 3.0
HATCHES = [(20, 26), (32, 38), (124, 130), (134, 139)]
BULKHEADS = [16, 46, 120, 142]
RAISED = [(0, 16), (46, 120), (142, L)]
HOUSES = [(50, 116, 0.78, 2.5, 0), (54, 110, 0.7, 2.5, 1), (56, 66, 0.5, 2.4, 2)]
FUNNELS = [(86.0, 2.6, 14.0, 0.8)]
MASTS = [(14.0, 26.0), (42.0, 26.0), (122.0, 24.0), (144.0, 22.0)]
BOATS = [(d, 1) for d in (62.0, 70.0, 78.0, 98.0, 106.0)]
BOAT_X = 0.3
SCREWS = [(-3.6, L - 7.0, 2.4, 2.0), (3.6, L - 7.0, 2.4, 2.0)]
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


wreck = merge(ship_part(0, L, False, mast_left=[3.0, 2.0, 2.5, 0.0], funnel_state="gone", davits=True))
collapse(wreck, D + FC + 0.6, 0.6, 51, rot=0.45)
holes(wreck, [(34.0, B * 0.48, 4.0, 4.0), (96.0, B * 0.48, 3.5, 4.5), (104.0, B * 0.47, 4.0, 3.5)], 52)   # the three torpedoes, port side (+X)
crumple(wreck, 0.12, 0.1, 53)
wreck.R(0, 0, math.radians(-4))
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 1.2, 0)
prune(wreck)
fun = cyl(2.6, 2.5, 9.0, FUNNELC, seg=18, sx=0.8)                           # the funnel, fallen to port beside the hull
fun.c[fun.v[:, 1] > 7.6] = BLACK
crumple(fun, 0.5, 0.4, 54)
fun.R(math.pi / 2, 0.25, 0).T(B * 0.5 + 3.0, 1.6, zd(80))
masts = [cyl(0.38, 0.22, 20.0, GREY, seg=8).R(math.pi / 2 - 0.05, a, 0).T(x, 0.4, zd(d)) for x, d, a in ((-6.0, 26.0, 0.4), (9.0, 132.0, -0.3))]
debris = merge(plates(60, 8, zd(70), 20, 551, cols=(HULLC, SUPC, DECKC)) + [fun] + masts)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q504096" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-157, kind="liner", seed=1918)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())], deck_colors=[DECKC])
