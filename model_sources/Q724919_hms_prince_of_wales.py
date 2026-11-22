"""
HMS Prince of Wales (Q724919) — the wreck of the British battleship off Kuantan, Malaysia, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a King George V-class battleship completed in 1941: about 227 m long, beam 31.4 m, ten 14-inch guns in
two quadruple turrets (A forward, Y aft) and one twin (B, superfiring forward), sixteen 5.25-inch guns in eight
twin turrets, the bridge tower, two funnels, pole masts, four screws. She fought the Bismarck in May 1941.

The wreck, after the published descriptions: sunk by Japanese torpedo bombers with HMS Repulse on 10 December
1941 (327 of her men died), she lies upside down in about 68 m of water off Kuantan. In the 21st century she has
been illegally salvaged by scrap-metal dealers; shells and cordite lie scattered about the site. A "Protected
Place" under the Protection of Military Remains Act. The angle at which she lies, the heading (not published: 0),
the torpedo damage and the extent of the looting are approximate.

    python model_sources/Q724919_hms_prince_of_wales.py   -> models/item/Q724919/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1941)

L, B, D = 227.1, 31.4, 14.6
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.55, 0.07, 0.3, 1.8, 0.4, 2.4, 6.0
HULLC = (0.42, 0.44, 0.47)
SUPC = (0.55, 0.57, 0.58)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 3.5
HATCHES = []
BULKHEADS = []
RAISED = [(0, 12), (50.5, 76), (113, L)]
HOUSES = [(52, 74, 0.72, 2.5, 0), (53, 64, 0.6, 2.5, 1), (53, 57.5, 0.6, 2.4, 2)]
FUNNELS = [(66.5, 1.7, 9.0, 0.85)]
MASTS = [(26.5, 16.0), (89.5, 15.0)]
BOATS = [(60.0, 1), (70.0, 1)]
BOAT_X = 0.32
SCREWS = [(0.0, L - 4.0, 2.3, 2.0)]
RUDDER_D = L - 1.2
ANCHORS_GONE = ()
BS = dict(turrets=[(42.0, 0.0, 4, 0.0), (58.0, 2.6, 2, 0.0), (190.0, 0.0, 4, 3.1416)], gun_len=17.0, house_w=13.0,
          bridge=(64.0, 84.0, 5, 15.0), funnels=[(98.0, 3.0, 9.0, 0.7), (128.0, 3.0, 9.0, 0.7)],
          masts=[(86.0, 18.0, "pole"), (150.0, 16.0, "pole")],
          secondary=[(d, s) for d in (88.0, 100.0, 140.0, 152.0) for s in (-1, 1)], rudders=[0.0])


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
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship bs_turret bs_part pagoda capsize")


SCREWS = [(-3.5, L - 16.0, 2.6, 2.2), (3.5, L - 16.0, 2.6, 2.2), (-9.0, L - 34.0, 2.8, 2.2), (9.0, L - 34.0, 2.8, 2.2)]
RUDDER_D = L - 5.0

wreck = merge(bs_part(False))
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.2, 0.08, 11)
# torpedo damage aft on the port side, the port outer shaft bent; plates torn off by the scrap thieves
port_aft = (wreck.v[:, 0] > 6) & (dfb > 170) & (dfb < 205) & (wreck.v[:, 1] < 8)
wreck.v[port_aft, 0] -= 1.5 * (1 - np.abs(dfb[port_aft] - 188) / 18)
fc = wreck.v[wreck.f].mean(1)
loot = (fbm(fc, 0.05, 31) > 0.7) & (fc[:, 1] < D) & (np.abs(fc[:, 0]) > B * 0.3)
wreck.f = wreck.f[~loot]
capsize(wreck, math.radians(172))                                             # upside down (a little over to starboard)
r = np.random.default_rng(41)
shells = []
for k in range(26):                                                           # 14-inch shells and cordite charges strewn about
    s = cyl(0.18, 0.12, 1.6, (0.4, 0.38, 0.32), seg=8).R(math.pi / 2, r.uniform(0, 6.28), 0)
    shells.append(s.T(r.normal(0, 14), 0.2, zd(r.choice([40.0, 60.0, 190.0])) + r.normal(0, 8)))
debris = merge(shells + plates(60, 20, zd(120), 30, 100, cols=(HULLC, GREY_D)))
coral = merge([corals(wreck, 380, 7, size=(0.12, 0.45))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q724919" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": [("corals", coral, "life")]}],
            depth=-68, kind="warship", seed=1941, memorial=True, life="tropical")
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
