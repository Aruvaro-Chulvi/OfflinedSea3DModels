"""German cruiser Prinz Eugen (Q155430) — the capsized wreck at Kwajalein Atoll, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: an Admiral Hipper-class heavy cruiser completed in 1940: about 212.5 m long and 21.9 m in the beam, eight
20.3 cm guns in four twin turrets (two forward, two aft, the inner ones superfiring), twin 10.5 cm anti-aircraft
mounts along the sides, the tower bridge and foremast, a single large funnel, a pole mainmast, three screws. She
sailed with the Bismarck in May 1941 and survived the war.

The wreck, after the published descriptions (Wikipedia, Pacific Wrecks, the US Navy's oil-removal operation of 2018,
DAN's Alert Diver): handed to the United States, she was a target ship in the Bikini atomic tests of 1946, then towed
to Kwajalein; leaking, she developed a list of 35 degrees and on 22 December 1946 capsized and sank off Enubuj island.
She lies upside down on the reef slope, her stern on the reef with the bottom of the hull and her propellers out of the
water, her bow deeper, at about 35 m. In 2018 the US Navy removed some 250,000 gallons of fuel from her; one propeller
was removed and stands as a memorial at Laboe, Germany. The slope, the angle and the heading (not published here: 0)
are approximate.

    python model_sources/Q155430_prinz_eugen.py   -> models/item/Q155430/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1946 + 12)

L, B, D = 212.5, 21.9, 12.5
FC, TW, TT, WL = 0.0, 6.0, 1.2, 7.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.45, 0.07, 0.3, 2.4, 0.3, 2.3, 6.0
RAISED = []
SCREWS = [(-5.0, L - 26.0, 2.4, 2.0), (0.0, L - 18.0, 2.0, 2.0), (5.0, L - 26.0, 2.4, 2.0)]
RUDDER_D = L - 4.0
BS = dict(turrets=[(36.0, 0.0, 2, 0.0), (48.0, 2.8, 2, 0.0), (168.0, 2.8, 2, 3.1416), (180.0, 0.0, 2, 3.1416)],
          gun_len=10.5, house_w=12.0, bridge=(60.0, 80.0, 4, 13.0), funnels=[(100.0, 3.6, 9.0, 0.7)],
          masts=[(76.0, 18.0, "pole"), (128.0, 16.0, "pole")], casemates=[],
          secondary=[(d, s) for d in (88.0, 108.0, 118.0) for s in (-1, 1)], rudders=[0.0])
HULLC = (0.5, 0.52, 0.54)
SUPC = (0.56, 0.58, 0.6)
GREY, GREY_L, GREY_D = (0.48, 0.5, 0.52), (0.56, 0.58, 0.6), (0.32, 0.33, 0.34)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.52, 0.54, 0.56)
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


w = merge(bs_part(False, gone_masts=True))
fc = w.v[w.f].mean(1); fd_ = L / 2 - fc[:, 2]
w.f = w.f[~((np.abs(fd_ - (L - 26.0)) < 2.5) & (fc[:, 0] > 3.0) & (fc[:, 1] < 4.5))]   # the port propeller, removed (now at Laboe)
crumple(w, 0.15, 0.08, 71)
capsize(w, math.radians(178), sink=0.8)                                             # upside down: the upperworks crushed into the bottom
w.R(math.radians(8.5), 0, 0)                                                         # the stern up on the reef, its screws out of the water
w.T(0, -np.percentile(w.v[:, 1], 0.3), 0)


def reef_h(X, Z):
    """The reef slope under her stern: rising aft, ragged coral rock."""
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    n = fbm(P3, 0.08, 712, 3).reshape(X.shape) - 0.5
    under = np.clip(1 - (np.abs(X) / 55.0) ** 2, 0, 1) ** 0.5
    slope = (ZB - Z) * math.tan(math.radians(8.5))                              # the same slope as the hull resting on it
    return np.maximum(0.0, (slope - 0.6 + 2.5 * n) * under)


ZB = w.v[np.argmin(w.v[:, 1]), 2]                                              # where the bow touches the bottom
reef = grid(-55, 55, -L / 2 - 40, ZB + 20, 44, 90, reef_h, (0.5, 0.47, 0.4))
reef.f = reef.f[reef.v[reef.f][:, :, 1].max(axis=1) > 0.05]
reef.c = np.clip(reef.c * (0.8 + 0.4 * fbm(reef.v, 0.3, 713)[:, None]), 0, 1)
debris = merge(plates(40, 14, zd(40), 14, 711, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q155430" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [], "plain": [("reef", reef, "rock")]}],
            depth=-35, kind="warship", seed=1946, drift=0.4)
ship = merge(bs_part(True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
