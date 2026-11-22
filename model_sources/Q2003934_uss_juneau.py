"""USS Juneau (Q2003934) — the wreck of the US light cruiser lost off Guadalcanal in 1942, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: an Atlanta-class anti-aircraft light cruiser completed in 1942: about 165 m long and 16.2 m in the beam,
sixteen 5-inch guns in eight twin mounts (three on the centre line forward, three aft, two wing mounts abreast the
after superstructure), a bridge and tripod foremast, two funnels, two screws.

The wreck, after the published descriptions (Wikipedia, the R/V Petrel expedition, the Naval History and Heritage
Command, Smithsonian): damaged in the Naval Battle of Guadalcanal, she was hit by a torpedo from the Japanese
submarine I-26 on 13 November 1942; it struck near the forward magazine and she blew up and sank in about 20 seconds;
687 died, among them the five Sullivan brothers, and only ten were rescued. R/V Petrel found her on 17 March 2018 at
about 4,200 m off the Solomon Islands, broken into several large pieces, the bow and stern not far apart and debris
spread over more than a mile. The number and layout of the pieces and the heading (not published here: 0) are
approximate.

    python model_sources/Q2003934_uss_juneau.py   -> models/item/Q2003934/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1942 + 13)

L, B, D = 164.9, 16.2, 9.5
FC, TW, TT, WL = 0.0, 5.0, 1.1, 6.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.42, 0.07, 0.3, 2.4, 0.3, 2.3, 6.0
RAISED = []
SCREWS = [(-3.0, L - 14.0, 2.0, 1.6), (3.0, L - 14.0, 2.0, 1.6)]
RUDDER_D = L - 3.0
BS = dict(turrets=[(22.0, 0.0, 2, 0.0), (30.0, 2.4, 2, 0.0), (38.0, 4.8, 2, 0.0), (128.0, 4.8, 2, 3.1416), (136.0, 2.4, 2, 3.1416), (144.0, 0.0, 2, 3.1416)],
          gun_len=5.5, house_w=10.0, bridge=(44.0, 58.0, 4, 10.0), funnels=[(72.0, 2.0, 8.0, 0.75), (96.0, 2.0, 8.0, 0.75)],
          masts=[(54.0, 18.0, "tripod"), (108.0, 12.0, "pole")], casemates=[], secondary=[(118.0, -1), (118.0, 1)], rudders=[0.0])
HULLC = (0.38, 0.4, 0.43)
SUPC = (0.42, 0.44, 0.47)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.45), (0.5, 0.52, 0.54), (0.3, 0.3, 0.32)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.38, 0.4, 0.43)
HHW = 3.5
HATCHES = []
BULKHEADS = []
HOUSES = []
FUNNELS = []
MASTS = []
BOATS = []
BOAT_X = 0.32
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


# ── A battleship (driven by BS below) ───────────────────────────────────────
#   BS = dict(turrets=[(d, raised, guns, train)], gun_len, house_w, bridge=(d0, d1, levels, width), funnels=[(d, r, h, sx)],
#             masts=[(d, h, kind)], casemates=[d…] (both sides), secondary=[(d, side)], pagoda=None|(d, height))
# ── Damage helpers ───────────────────────────────────────────────────────────
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship bs_turret bs_part pagoda capsize collapse holes lay boiler center_x")


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


def cut(m, d0, d1, seed):
    """The faces between d0 and d1 from the bow, with ragged ends."""
    fc = m.v[m.f].mean(1); fd_ = L / 2 - fc[:, 2]
    j = 3.0 * (fbm(fc, 0.15, seed) - 0.5)
    keep = (fd_ + j >= d0) & (fd_ + j < d1)
    k, inv = np.unique(m.f[keep], return_inverse=True)
    return Mesh(m.v[k], inv.reshape(-1, 3), m.c[k])


full = merge(bs_part(False, gone_masts=True))
crumple(full, 0.25, 0.1, 91)
bow = cut(full, 0, 34, 92)
mid = cut(full, 66, 104, 93)
stern = cut(full, 112, L, 94)
lay(bow, math.radians(60), 95, dz=40.0, yaw=math.radians(40), sink=1.5)          # the bow on its side
lay(mid, math.radians(150), 96, dz=0.0, yaw=math.radians(-25), sink=5.0)         # a piece of the middle, upside down
lay(stern, math.radians(-12), 97, dz=-30.0, yaw=math.radians(12), sink=1.5)      # the stern, nearly upright
mid.T(-30.0, 0, 0)
center_x(stern, bow)
G = []
for k in range(4):                                                                 # 5-inch mounts and pieces thrown out by the blast
    g = bs_turret(2, rng.uniform(0, 6.28), 0.0, 5.5)
    G.append(g.R(rng.uniform(-1.4, 1.4), 0, rng.uniform(-1.4, 1.4)).T(rng.normal(-10, 25), 1.5, rng.normal(0, 40)))
debris = merge(plates(160, -12, 0, 40, 951, cols=(HULLC, GREY_D, SUPC)) + G)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q2003934" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("stern_section", stern), ("bow_section", bow), ("hull_section", mid), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-4200, kind="warship", seed=1942, memorial=True, drift=0.4)
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
