"""SMS Scharnhorst (Q533999) — the wreck of the German armoured cruiser sunk at the Battle of the Falkland Islands in 1914, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the lead ship of her class, completed in 1907, flagship of Vice-Admiral Graf Spee's East Asia Squadron:
about 144.6 m long, beam 21.6 m, eight 21 cm guns (two twin turrets fore and aft, and four single guns in casemates
on the upper deck amidships), six 15 cm guns in casemates, four funnels, two masts with fighting tops, three screws.

The wreck, after the published descriptions of the Falklands Maritime Heritage Trust expedition led by Mensun Bound
from the Seabed Constructor (found on 5 December 2019, announced on the eve of the battle's anniversary): after
the victory at Coronel, the squadron was caught off the Falklands on 8 December 1914 by the battlecruisers
Invincible and Inflexible; Scharnhorst went down with all hands, Graf Spee among them. She lies about 98 nautical
miles south-east of Stanley at 1,610 m, upright and intact, battle-scarred, beside her impact crater, "her big guns
pointing in all directions". The site was not touched. The damage, the trains of the guns and the heading are
approximate.

    python model_sources/Q533999_sms_scharnhorst.py   -> models/item/Q533999/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1914 + 12)

L, B, D = 144.6, 21.6, 11.5
FC, TW, TT, WL = 0.0, 6.0, 1.2, 7.5
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.22, 0.5, 0.07, 0.3, 1.8, 0.4, 2.4, 5.5
RAISED = [(0, 40)]
SCREWS = [(-4.5, L - 20.0, 2.2, 1.9), (0.0, L - 12.0, 2.0, 1.9), (4.5, L - 20.0, 2.2, 1.9)]
RUDDER_D = L - 3.0
BS = dict(turrets=[(22.0, 0.0, 2, 0.0), (124.0, 0.0, 2, 3.1416)],
          gun_len=9.5, house_w=9.0, bridge=(30.0, 40.0, 3, 9.5), funnels=[(50.0, 1.6, 10.0, 1.1), (60.0, 1.6, 10.0, 1.1), (72.0, 1.6, 10.0, 1.1), (82.0, 1.6, 10.0, 1.1)],
          masts=[(36.0, 24.0, "pole"), (104.0, 20.0, "pole")], casemates=[46.0, 66.0, 92.0], secondary=[(56.0, 1), (56.0, -1), (88.0, 1), (88.0, -1)],
          rudders=[0.0])
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


BUILT_BS = BS
# her guns pointing in all directions: the turrets left trained where the battle left them
BS = dict(BS, turrets=[(22.0, 0.0, 2, 0.7), (124.0, 0.0, 2, 3.1416 - 1.1)], secondary=[(56.0, 1), (56.0, -1), (88.0, 1), (88.0, -1)])
w = merge(bs_part(False))
crumple(w, 0.18, 0.1, 26)
fc = w.v[w.f].mean(1); fd_ = L / 2 - fc[:, 2]
hits = np.zeros(len(fc), bool)
for k in range(22):                                                         # battle-scarred: shell holes along her sides and upperworks
    d0 = rng.uniform(8, 138); s = rng.choice([-1, 1]); y0 = rng.uniform(3.0, D + 8.0); r0 = rng.uniform(0.8, 2.2)
    hits |= (fc[:, 0] * s > -0.5) & (np.hypot(fd_ - d0, fc[:, 1] - y0) < r0) & (fbm(fc, 0.3, 27) > 0.3)
fun = (fc[:, 1] > D + 6.0) & (np.abs(fd_ - 66.0) < 22.0) & (fbm(fc, 0.15, 28) > 0.55)     # funnels shot through and collapsed in part
w.f = w.f[~(hits | fun)]
w.R(0, 0, math.radians(-3))
w.T(0, -np.percentile(w.v[:, 1], 0.5) - 2.0, 0)                            # sunk in her impact crater


def crater(X, Z):
    d = np.sqrt((X / 34.0) ** 2 + (Z / 95.0) ** 2)
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    return (2.2 * np.exp(-((d - 1.0) / 0.18) ** 2) - 1.2 * np.clip(1 - d, 0, 1)) * (0.75 + 0.5 * fbm(P3, 0.05, 29, 2).reshape(X.shape))


rim = grid(-52, 52, -140, 140, 52, 140, crater, (0.5, 0.48, 0.44))
rim.f = rim.f[rim.v[rim.f][:, :, 1].max(axis=1) > 0.15]
debris = merge(plates(60, 0, zd(70), 30, 311, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q533999" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [rim], "plain": []}],
            depth=-1610, kind="warship", seed=1926, memorial=True, drift=0.4)
BS = BUILT_BS
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
