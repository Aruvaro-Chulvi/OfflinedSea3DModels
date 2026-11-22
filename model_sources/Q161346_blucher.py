"""Blücher (Q161346) — the wreck of the German heavy cruiser sunk in the Oslofjord in 1940, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: an Admiral Hipper-class heavy cruiser completed in 1939: about 203 m long, beam 22 m, eight 20.3 cm guns
in four twin turrets (two forward, two aft, the inner ones superfiring), twelve 10.5 cm anti-aircraft guns in six
twin mounts along the sides, a tower bridge, a single large funnel with a cap, a mainmast aft with the aircraft
catapult between them, three screws.

The wreck, after the published descriptions: leading the German invasion force to Oslo on 9 April 1940, she was hit
at close range in the Drøbak Sound by the 28 cm guns of Oscarsborg fortress and by two torpedoes from its shore
battery on the port side, burned, capsized and sank with some 800-1,000 dead. She lies upside down at about 90 m on
the bottom of the fjord. Oil was pumped out of her in 1994, when one of her Arado 196 floatplanes was also raised;
oil still seeps from her. Protected as cultural heritage: diving needs permission. The exact angle and heading are
approximate.

    python model_sources/Q161346_blucher.py   -> models/item/Q161346/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1940)

L, B, D = 202.8, 21.8, 12.5
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.22, 0.45, 0.07, 0.3, 2.4, 0.4, 2.4, 5.5
RAISED = [(0, 60)]
SCREWS = [(-5.0, L - 28.0, 2.3, 2.0), (0.0, L - 18.0, 2.2, 2.0), (5.0, L - 28.0, 2.3, 2.0)]
RUDDER_D = L - 4.0
BS = dict(turrets=[(40.0, 0.0, 2, 0.0), (53.0, 2.6, 2, 0.0), (150.0, 2.6, 2, 3.1416), (163.0, 0.0, 2, 3.1416)],
          gun_len=10.5, house_w=11.0, bridge=(64.0, 84.0, 5, 12.0), funnels=[(98.0, 3.2, 9.0, 0.7)],
          masts=[(74.0, 22.0, "pole"), (128.0, 18.0, "pole")], casemates=[],
          secondary=[(88.0, 1), (88.0, -1), (106.0, 1), (106.0, -1), (120.0, 1), (120.0, -1)], rudders=[0.0])
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


wreck = merge(bs_part(False))
crumple(wreck, 0.2, 0.08, 40)
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
hits = np.zeros(len(fc), bool)
for d0, r0 in ((96.0, 5.0), (116.0, 4.5)):                                        # the two torpedoes, port side
    hits |= (fc[:, 0] > B * 0.3) & (np.hypot(fd_ - d0, (fc[:, 1] - 3.5) * 1.3) < r0 * (0.8 + 0.4 * fbm(fc, 0.15, 41)))
for d0 in (66.0, 92.0, 140.0):                                                     # shell hits from Oscarsborg
    hits |= (np.hypot(fd_ - d0, (fc[:, 1] - D + 1.5) * 1.2) < 2.2) & (np.abs(fc[:, 0]) > B * 0.3)
wreck.f = wreck.f[~hits]
capsize(wreck, math.radians(-174), sink=0.8)                                       # she rolled over to port: keel up
debris = merge(plates(70, 10, zd(105), 20, 171, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q161346" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-90, kind="warship", seed=1940, memorial=True)
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
