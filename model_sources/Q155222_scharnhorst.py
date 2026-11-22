"""Scharnhorst (Q155222) — the wreck of the German battleship sunk at the Battle of the North Cape in 1943, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Scharnhorst-class battleship completed in 1939: about 235 m long, beam 30 m, nine 28 cm guns in three
triple turrets (Anton and Bruno forward, Caesar aft), 15 cm guns in twin turrets along the sides, 10.5 cm
anti-aircraft guns in twin mounts, a tower bridge, a single funnel with a raked cap, a mainmast aft, three screws
and twin rudders.

The wreck, after the published descriptions: sunk on 26 December 1943 by HMS Duke of York and her escorts after many
hits by 14-inch shells and some eleven torpedoes; only 36 of her 1,968 men survived. Her wreck was found in
September 2000 by a Norwegian expedition at about 290 m north of the North Cape. She lies upside down; her bow,
torn off forward of the first turret (probably by the explosion of a magazine or of torpedoes), lies apart nearby,
and a field of wreckage spreads about her. The exact layout, angle and heading are approximate.

    python model_sources/Q155222_scharnhorst.py   -> models/item/Q155222/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1943)

L, B, D = 234.9, 30.0, 14.6
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.22, 0.5, 0.07, 0.3, 2.8, 0.4, 2.4, 5.5
RAISED = [(0, 70)]
SCREWS = [(-6.0, L - 32.0, 2.6, 2.2), (0.0, L - 20.0, 2.4, 2.2), (6.0, L - 32.0, 2.6, 2.2)]
RUDDER_D = L - 4.0
BS = dict(turrets=[(52.0, 0.0, 3, 0.0), (66.0, 2.8, 3, 0.0), (192.0, 0.0, 3, 3.1416)],
          gun_len=13.0, house_w=14.0, bridge=(78.0, 98.0, 5, 15.0), funnels=[(114.0, 3.6, 10.0, 0.72)],
          masts=[(88.0, 26.0, "pole"), (150.0, 20.0, "pole")], casemates=[],
          secondary=[(102.0, 1), (102.0, -1), (160.0, 1), (160.0, -1), (124.0, 1), (124.0, -1), (136.0, 1), (136.0, -1)], rudders=[-2.6, 2.6])
HULLC = (0.4, 0.42, 0.44)
SUPC = (0.5, 0.52, 0.53)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
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
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship bs_turret bs_part pagoda capsize sub")


wreck = merge(bs_part(False))
crumple(wreck, 0.22, 0.08, 43)
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
hits = np.zeros(len(fc), bool)
for d0, s, r0 in ((90.0, 1, 4.5), (120.0, -1, 5.0), (150.0, 1, 4.5), (178.0, -1, 4.5), (205.0, 1, 4.0), (110.0, 1, 4.0)):   # torpedoes, both sides
    hits |= (fc[:, 0] * s > B * 0.3) & (np.hypot(fd_ - d0, (fc[:, 1] - 4.0) * 1.3) < r0 * (0.8 + 0.4 * fbm(fc, 0.15, 44)))
for d0 in (70.0, 100.0, 130.0, 160.0, 185.0):                                           # shell hits
    hits |= (np.hypot(fd_ - d0, (fc[:, 1] - D) * 1.2) < 2.6) & (fbm(fc, 0.2, 45) > 0.4)
cut = 40.0 + 3.0 * (fbm(fc, 0.12, 46) - 0.5)                                            # the bow, torn off ahead of Anton
bowm = sub(wreck, (fd_ < cut) & ~hits)
wreck = sub(wreck, (fd_ >= cut) & ~hits)
capsize(wreck, math.radians(178), sink=0.9)
# the bow section apart, on its side, some 60 m ahead of the torn end and off to one side
bowm.v[:, 2] -= zd(20.0)
bowm.R(0, 0, math.radians(100)).R(0, math.radians(35), 0)
bowm.T(0, -bowm.v[:, 1].min() - 1.0, 0)
low = bowm.v[:, 1] < -0.3
bowm.v[low, 1] = -0.3 + (bowm.v[low, 1] + 0.3) * 0.12
bowm.T(-34.0, 0, zd(40.0) + 60.0)
debris = merge(plates(140, 0, zd(60), 30, 181, cols=(HULLC, GREY_D)) + plates(80, -18, zd(-10), 24, 182, cols=(HULLC,)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q155222" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("bow_section", bowm), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-290, kind="warship", seed=1943, memorial=True, drift=0.5)
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
