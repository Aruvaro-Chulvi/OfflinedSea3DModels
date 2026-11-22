"""
SMS Szent István (Q642429) — the wreck of the Austro-Hungarian battleship off Premuda, Croatia, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the last Tegetthoff-class dreadnought, completed at Fiume in 1915: about 152 m long, beam 28 m, twelve
30.5 cm guns in four triple turrets (two forward, two aft, superfiring), two funnels close together, two masts,
two screws and twin rudders.

The wreck, after the published descriptions: on 10 June 1918 two torpedoes from the Italian motor torpedo boat
MAS 15 struck her near the boilers; she capsized about three hours later off Premuda (89 men died; the sinking
was filmed). She lies upside down at about 66 m; the bow broke off when it hit the seabed while the stern was
still afloat, and lies right beside the rest of the heavily encrusted hull. The two propellers and the rudders
can be seen; an opening lets divers swim beneath the hull; a searchlight lies half buried in the sand. Located by
the Yugoslav Navy in the 1970s; a protected site of the Croatian Ministry of Culture. The heading (not published:
0), the angle of the bow and the turrets' state are approximate.

    python model_sources/Q642429_sms_szent_istvan.py   -> models/item/Q642429/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1918)

L, B, D = 152.0, 28.0, 13.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.28, 0.7, 0.2, 0.55, 0.07, 0.3, 1.6, 0.4, 2.6, 6.0
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
BS = dict(turrets=[(28.0, 0.0, 3, 0.0), (40.0, 2.6, 3, 0.0), (112.0, 2.6, 3, 3.1416), (124.0, 0.0, 3, 3.1416)],
          gun_len=14.0, house_w=12.0, bridge=(48.0, 58.0, 3, 12.0), funnels=[(66.0, 2.4, 9.0, 0.8), (74.0, 2.4, 9.0, 0.8)],
          masts=[(58.0, 22.0, "pole"), (96.0, 18.0, "pole")], casemates=[50.0, 60.0, 70.0, 80.0, 90.0, 100.0], secondary=[],
          rudders=[-2.0, 2.0])


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


SCREWS = [(-4.0, L - 16.0, 2.6, 2.1), (4.0, L - 16.0, 2.6, 2.1)]
RUDDER_D = L - 5.0
BRK = 22.0
full = merge(bs_part(False))
crumple(full, 0.2, 0.08, 11)
# the torpedo holes near the boiler rooms, starboard side
fc = full.v[full.f].mean(1); fd_ = L / 2 - fc[:, 2]
holes = (fc[:, 0] < -B * 0.35) & (fc[:, 1] < 6.0) & (np.abs(fd_ - 66) < 4 + 2 * fbm(fc, 0.3, 5)) | ((fc[:, 0] < -B * 0.35) & (fc[:, 1] < 6.0) & (np.abs(fd_ - 80) < 3.5))
full.f = full.f[~holes]
dv = L / 2 - full.v[:, 2]
fwd_v = dv < BRK
# split at the break: faces wholly forward go with the bow
fm = fwd_v[full.f].all(1)
bow = Mesh(full.v, full.f[fm], full.c); main = Mesh(full.v, full.f[~fm], full.c)
u, inv = np.unique(bow.f, return_inverse=True); bow = Mesh(bow.v[u], inv.reshape(-1, 3), bow.c[u])
u, inv = np.unique(main.f, return_inverse=True); main = Mesh(main.v[u], inv.reshape(-1, 3), main.c[u])
crumple(main, 1.4, 0.3, 21, sstep(BRK + 6, BRK, L / 2 - main.v[:, 2]))
crumple(bow, 1.4, 0.3, 22, sstep(BRK - 6, BRK, L / 2 - bow.v[:, 2]))
capsize(main, math.radians(176), sink=0.8)
zb = zd(BRK / 2)
bow.T(0, 0, -zb)
capsize(bow, math.radians(160), sink=0.8)
bow.R(0, 0.35, 0).T(5.0, 0, zb + 3.0)                                          # broken off on hitting the bottom, lying right beside
sl = merge([cyl(0.9, 0.9, 1.2, GREY, seg=14), cyl(0.85, 0.85, 0.05, (0.7, 0.75, 0.75), seg=14).T(0, 1.2, 0)]).R(1.2, 0, 0.3).T(-B * 0.7, -0.4, zd(60))
debris = merge([sl] + plates(40, -18, zd(70), 16, 100, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q642429" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", main), ("bow_section", bow), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-66, kind="warship", seed=1918, memorial=True)
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
