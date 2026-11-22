"""SMS Kronprinz Wilhelm (Q470400) — the wreck of the German battleship scuttled in Scapa Flow, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a König-class dreadnought completed in 1914 (as Kronprinz, renamed Kronprinz Wilhelm in 1918), sister of
the Markgraf: about 175 m long, beam 29.5 m, ten 30.5 cm guns in five twin turrets (two forward, one amidships, two
aft), fourteen 15 cm guns in casemates, two funnels, two masts, three screws and twin rudders.

The wreck, after the published descriptions and dive guides (Canmore, Scapa Flow Wrecks, Divernet "Wreck Tour"):
scuttled with the interned High Seas Fleet on 21 June 1919 and never raised, she lies upside down on her starboard
side in some 34-45 m (the shallowest part about 12 m below the surface), extensively salvaged: the armour belt
taken and the engine and boiler rooms blasted open amidships. Half of her 30.5 cm guns are still to be seen; the two
gun turrets at the stern, with their barrels reaching out under the overhanging side, are the dive most made. The
other turrets fell out of their barbettes as she rolled over and lie under the hull. The exact angle and heading
(not published: 0) are approximate.

    python model_sources/Q470400_sms_kronprinz_wilhelm.py   -> models/item/Q470400/model.glb (+ model_built.glb)
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

L, B, D = 175.4, 29.5, 14.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.28, 0.7, 0.2, 0.55, 0.07, 0.3, 1.8, 0.5, 2.6, 6.0
RAISED = [(0, 12), (50.5, 76), (113, L)]
SCREWS = [(-5.0, L - 20.0, 2.6, 2.0), (0.0, L - 12.0, 2.4, 2.0), (5.0, L - 20.0, 2.6, 2.0)]
RUDDER_D = L - 5.0
BS = dict(turrets=[(28.0, 0.0, 2, 0.0), (40.0, 2.6, 2, 0.0), (90.0, 0.0, 2, 1.5708), (138.0, 2.6, 2, 3.1416), (150.0, 0.0, 2, 3.1416)],
          gun_len=14.0, house_w=12.0, bridge=(50.0, 62.0, 3, 13.0), funnels=[(72.0, 2.6, 9.0, 0.8), (104.0, 2.6, 9.0, 0.8)],
          masts=[(62.0, 22.0, "pole"), (118.0, 18.0, "pole")], casemates=[48.0, 56.0, 64.0, 72.0, 80.0, 112.0, 120.0], secondary=[],
          rudders=[-2.4, 2.4])
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
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship bs_turret bs_part pagoda capsize")


BUILT_BS = dict(BS)
BS = dict(BS, turrets=[t for t in BS["turrets"] if t[0] in (28.0, 138.0, 150.0)])   # two forward turrets and the midships one fell out
SCREWS = []                                                                   # the propellers were salvaged
wreck = merge(bs_part(False))
crumple(wreck, 0.2, 0.08, 14)
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
belt = (np.abs(fc[:, 0]) > B * 0.42) & (fc[:, 1] > 3.0) & (fc[:, 1] < 7.5) & (fd_ > 25) & (fd_ < 160) & (fbm(fc, 0.05, 41) < 0.7)
engine = (fd_ > 66) & (fd_ < 124) & (fc[:, 1] < 5.0) & (fbm(fc, 0.07, 42) > 0.4)
wreck.f = wreck.f[~(belt | engine)]                                            # armour belt taken, engine and boiler rooms blasted
capsize(wreck, math.radians(148), sink=0.8)                                    # upside down on her starboard side
# the turrets that fell out, lying upside down on the seabed beside the hull
fallen = []
for k, (d, tr) in enumerate(((40.0, 0.3), (90.0, -0.4))):
    t = bs_turret(2, tr, 0.0, 14.0).R(math.pi, 0, 0)
    t.T(-B * 0.72 - 2.0 * k, -t.v[:, 1].min() - 0.8, zd(d))
    crumple(t, 0.12, 0.2, 60 + k)
    fallen.append(t)
debris = merge(fallen + plates(70, -20, zd(95), 22, 141, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q470400" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-38, kind="warship", seed=1914)
BS = BUILT_BS
SCREWS = [(-5.0, L - 20.0, 2.6, 2.0), (0.0, L - 12.0, 2.4, 2.0), (5.0, L - 20.0, 2.6, 2.0)]
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
