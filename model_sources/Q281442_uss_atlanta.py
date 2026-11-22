"""USS Atlanta (Q281442) — the wreck of the US light cruiser lost off Guadalcanal in 1942, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the lead ship of the Atlanta class of anti-aircraft light cruisers, completed in December 1941: about 165 m
long and 16.2 m in the beam, sixteen 5-inch guns in eight twin mounts (three on the centre line forward, three aft,
two wing mounts abreast the after superstructure), a bridge and tripod foremast, two funnels, two screws.

The wreck, after the published descriptions (Wikipedia, the Naval History and Heritage Command, Pacific Wrecks, the
US Navy and National Geographic on the 2025 Ironbottom Sound expedition): in the night action of 13 November 1942
(First Naval Battle of Guadalcanal) she was hit by a torpedo from the destroyer Akatsuki and by the gunfire of
Japanese ships and, in the confusion, of the cruiser San Francisco; Rear Admiral Norman Scott died on her bridge with
much of her crew. Taken in tow, she was scuttled that evening off Lunga Point. Robert Ballard's Lost Ships of
Guadalcanal expedition found her in 1992; she lies on her port side at about 130 m (430 ft) in Ironbottom Sound, and
his team returned in 2025 to survey the battle's wrecks again. The damage and the heading (not published here: 0)
are approximate.

    python model_sources/Q281442_uss_atlanta.py   -> models/item/Q281442/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1942 + 14)

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


w = merge(bs_part(False, gone_masts=True))
holes(w, [(80.0, -B * 0.48, 3.5, 4.5)], 81)                                        # the torpedo, port side amidships (engine rooms)
fc = w.v[w.f].mean(1); fd_ = L / 2 - fc[:, 2]
shell = (fc[:, 1] > D) & (fbm(fc, 0.2, 82) > 0.58) & (fd_ > 40) & (fd_ < 110)     # the superstructure riddled by gunfire
w.f = w.f[~shell]
lay(w, math.radians(-85), 83, sink=0.8)                                             # on her port side
crumple(w, 0.15, 0.1, 84)
center_x(w)
debris = merge(plates(50, 12, zd(80), 16, 841, cols=(HULLC, GREY_D, SUPC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q281442" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-131, kind="warship", seed=1942, memorial=True)
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
