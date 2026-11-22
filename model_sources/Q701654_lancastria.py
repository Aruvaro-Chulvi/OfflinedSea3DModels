"""RMS Lancastria (Q701654) — the wreck of the Cunard liner bombed off Saint-Nazaire in 1940, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Cunard liner completed in 1922 as Tyrrhenia, renamed Lancastria in 1924: about 176 m long and 21.3 m in
the beam, a long superstructure of promenade and boat decks, a single funnel, two masts, two screws. Taken up as a
troopship in 1940 and painted grey.

The wreck, after the published descriptions (Wikipedia, the UK government, the Royal Navy, the National Archives):
on 17 June 1940, during the evacuation of British troops and civilians from France (Operation Aerial), she was hit by
German bombs off Saint-Nazaire and rolled over as she sank in about 20 minutes; at least 1,738 died and probably
several thousand, the greatest loss of life in British maritime history. She lies in about 24-26 m, some 17 km south
of Saint-Nazaire and 7 km west of Pointe Saint-Gildas, marked by a buoy. A war grave protected by the French
government since 2006: diving is not allowed. The published sources do not describe her attitude on the bottom: the
model lays her over on her port side, nearly keel up, as she rolled when she sank. The attitude, collapse and heading
(not published here: 0) are approximate.

    python model_sources/Q701654_lancastria.py   -> models/item/Q701654/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1940 + 6)

L, B, D = 176.0, 21.3, 13.0
FC, TW, TT, WL = 2.6, 8.0, 1.3, 8.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.26, 0.75, 0.24, 0.4, 0.07, 0.32, 1.8, 0.7, 2.5, 6.5
HULLC = (0.36, 0.37, 0.38)                    # troopship grey
SUPC = (0.46, 0.47, 0.47)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.42, 0.43, 0.43)
HHW = 3.2
HATCHES = [(18, 24), (30, 36), (150, 156), (160, 165)]
BULKHEADS = [14, 40, 146, 168]
RAISED = [(0, 14), (40, 146), (168, L)]
HOUSES = [(42, 142, 0.84, 2.5, 0), (46, 136, 0.8, 2.5, 1), (52, 124, 0.72, 2.5, 2), (54, 62, 0.5, 2.4, 3)]
FUNNELS = [(92.0, 3.4, 11.0, 0.72)]
MASTS = [(28.0, 24.0), (158.0, 22.0)]
BOATS = [(d, 2) for d in (66.0, 74.0, 82.0, 102.0, 110.0, 118.0)]
BOAT_X = 0.31
SCREWS = [(-4.0, L - 8.0, 2.6, 2.1), (4.0, L - 8.0, 2.6, 2.1)]
RUDDER_D = L - 3.0
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


def prune(m, min_faces=12, above=1.0):
    """Drop the little loose bits left hanging in the water (window panes, slivers of plating) once the walls around
    them have been cut away: connected groups of fewer than min_faces faces lying wholly above `above` m."""
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
    ymin = np.full(len(cnt), 1e9)
    np.minimum.at(ymin, inv, m.v[m.f][:, :, 1].min(1))
    drop = (cnt[inv] < min_faces) & (ymin[inv] > above)
    m.f = m.f[~drop]
    return m


wreck = merge(ship_part(0, L, False, mast_left=[0.0, 0.0], funnel_state="gone", davits=False))
holes(wreck, [(70.0, 3.0, D + 4.0, 5.0), (100.0, -2.0, D + 2.0, 4.0), (126.0, 2.0, D + 3.0, 4.0)], 71)   # the bombs, down through the decks
lay(wreck, math.radians(-150), 72, sink=3.8, squash=(6.0, 0.15))       # rolled over to port: the upperworks crushed under her
crumple(wreck, 0.2, 0.1, 73)
center_x(wreck)
debris = merge(plates(80, 14, zd(90), 22, 741, cols=(HULLC, SUPC, DECKC)) + [lifeboat().R(0.2, 0.6, 2.8).T(18, 0.5, zd(70)), lifeboat().R(-0.3, 2.0, 0.4).T(-16, 0.4, zd(110))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q701654" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-25, kind="liner", seed=1940, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())], deck_colors=[DECKC])
