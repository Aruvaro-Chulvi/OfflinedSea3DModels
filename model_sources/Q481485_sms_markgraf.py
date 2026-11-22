"""
SMS Markgraf (Q481485) — the wreck of the German battleship scuttled in Scapa Flow, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a König-class dreadnought completed in 1914: about 175 m long, beam 29.5 m, ten 30.5 cm guns in five twin
turrets (two forward, one amidships between the funnels, two aft), fourteen 15 cm guns in casemates along the
sides, an armour belt, two funnels, two masts with spotting tops, three screws and twin rudders. She fought at
Jutland.

The wreck, after the published descriptions and dive guides: scuttled with the interned High Seas Fleet on 21 June
1919, she was never raised. She lies almost completely upturned on her starboard side at about 45 m, held from
rolling right over by her superstructure. Salvors have taken most of the armour belt and the propellers (broken
stubs of the A-frames remain) and blasted the engine rooms open; at the bow a deep V was cut down to the seabed to
reach the torpedo rooms. Still to be seen: the stem rising from the seabed, a hawse pipe and the anchor chain
across the hull, seven casemate guns in a line (two hanging down, one pointing aft), the bent foremast with its
spotting top, two rows of portholes, the twin rudders. Considered the finest of the Scapa Flow wrecks. The exact
angle and heading (not published: 0) are approximate.

    python model_sources/Q481485_sms_markgraf.py   -> models/item/Q481485/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1919)

L, B, D = 175.4, 29.5, 14.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.28, 0.7, 0.2, 0.55, 0.07, 0.3, 1.8, 0.5, 2.6, 6.0
HULLC = (0.4, 0.42, 0.44)
SUPC = (0.5, 0.52, 0.53)
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
BS = dict(turrets=[(28.0, 0.0, 2, 0.0), (40.0, 2.6, 2, 0.0), (90.0, 0.0, 2, 1.5708), (138.0, 2.6, 2, 3.1416), (150.0, 0.0, 2, 3.1416)],
          gun_len=14.0, house_w=12.0, bridge=(50.0, 62.0, 3, 13.0), funnels=[(72.0, 2.6, 9.0, 0.8), (104.0, 2.6, 9.0, 0.8)],
          masts=[(62.0, 22.0, "pole"), (118.0, 18.0, "pole")], casemates=[48.0, 56.0, 64.0, 72.0, 80.0, 112.0, 120.0], secondary=[],
          rudders=[-2.4, 2.4])


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


SCREWS = []                                                                   # the propellers were salvaged
RUDDER_D = L - 5.0

wreck = merge(bs_part(False))
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.2, 0.08, 11)
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
belt = (np.abs(fc[:, 0]) > B * 0.42) & (fc[:, 1] > 3.0) & (fc[:, 1] < 7.5) & (fd_ > 25) & (fd_ < 160) & (fbm(fc, 0.05, 31) < 0.75)
vcut = (fd_ < 22) & (fd_ > 4) & (np.abs(fc[:, 0]) < 2.5 + (22 - fd_) * 0.3) & (fc[:, 1] < D - 1)
engine = (fd_ > 95) & (fd_ < 130) & (fc[:, 1] < 4.0) & (fbm(fc, 0.08, 32) > 0.45)
wreck.f = wreck.f[~(belt | vcut | engine)]                                     # armour belt taken, the bow cut open, engine rooms blasted
fm = (np.abs(wreck.v[:, 2] - zd(62.0)) < 2.0) & (wreck.v[:, 1] > D + 12)
wreck.v[fm, 2] += (wreck.v[fm, 1] - D - 12) * 0.5                              # the foremast bent
capsize(wreck, math.radians(152), sink=0.8)                                    # almost upturned on her starboard side
chain = tube([(B * 0.1, 15.0, zd(4.0)), (B * 0.2, 13.0, zd(12.0)), (B * 0.45, 4.0, zd(18.0)), (B * 0.7, 0.2, zd(22.0))], 0.16, BLACK, seg=5)
debris = merge([chain] + plates(60, -22, zd(100), 20, 100, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q481485" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-45, kind="warship", seed=1919)
SCREWS = [(-5.0, L - 20.0, 2.6, 2.0), (0.0, L - 12.0, 2.4, 2.0), (5.0, L - 20.0, 2.6, 2.0)]
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
