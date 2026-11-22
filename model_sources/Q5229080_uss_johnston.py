"""USS Johnston (DD-557) (Q5229080) — the wreck of the US destroyer sunk off Samar in 1944, the deepest wreck ever dived, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Fletcher-class destroyer commissioned in 1943: about 115 m long, beam 12 m, five single enclosed 5-inch
gun mounts (two forward, three aft), two raked funnels, a pole foremast over the bridge, two quintuple torpedo-tube
mounts amidships, 40 mm and 20 mm anti-aircraft guns, two screws.

The wreck, after the published descriptions of Victor Vescovo's dives in the submersible Limiting Factor (March
2021) and of the Naval History and Heritage Command (H-Gram 060): in the Battle off Samar on 25 October 1944,
under Commander Ernest Evans, she charged the Japanese battleships and cruisers to cover the escort carriers, and
was sunk by gunfire; 186 of her 327 men died. Her forward two thirds lie upright and intact at 6,456 m in the
Philippine Sea, having slid down a cliff face into deeper water; the hull is peppered with shell holes and clearly
identified by the hull number 557 at the bow; the bow, the bridge, gun mounts, the starboard 20 mm guns and the aft
quintuple torpedo mount can be seen. The after third lies apart, broken up, up the slope (not drawn here). The
exact damage and heading are approximate.

    python model_sources/Q5229080_uss_johnston.py   -> models/item/Q5229080/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1944 + 25)

L, B, D = 114.8, 12.0, 7.2
FC, TW, TT, WL = 0.0, 6.0, 1.2, 4.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.32, 0.8, 0.25, 0.55, 0.05, 0.2, 2.2, 0.3, 2.2, 5.5
RAISED = [(0, 44)]
SCREWS = [(-2.6, L - 10.0, 1.4, 1.5), (2.6, L - 10.0, 1.4, 1.5)]
RUDDER_D = L - 2.0
BS = dict(turrets=[(14.0, 0.0, 1, 0.0), (22.0, 1.6, 1, 0.0), (84.0, 1.6, 1, 3.1416), (96.0, 0.0, 1, 3.1416), (104.0, 0.0, 1, 3.1416)],
          gun_len=5.5, house_w=6.0, bridge=(28.0, 38.0, 3, 7.0), funnels=[(44.0, 1.6, 6.0, 0.7), (58.0, 1.6, 6.0, 0.7)],
          masts=[(34.0, 14.0, "pole")], casemates=[], secondary=[(40.0, 1), (40.0, -1), (76.0, 1), (76.0, -1)], rudders=[0.0])
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


def lettering(text, h, col):
    """Hull numbers as painted block digits (seven segments, facing +z), centred on the origin."""
    SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcfgd"}
    w, t = h * 0.55, h * 0.13
    pos = {"a": (0, h / 2, w, t), "g": (0, 0, w, t), "d": (0, -h / 2, w, t), "f": (-w / 2, h / 4, t, h / 2), "b": (w / 2, h / 4, t, h / 2),
           "e": (-w / 2, -h / 4, t, h / 2), "c": (w / 2, -h / 4, t, h / 2)}
    P = []
    for i, ch in enumerate(text):
        x0 = (i - (len(text) - 1) / 2) * w * 1.5
        for s in SEG[ch]:
            x, y, sw, sh = pos[s]
            P.append(box(sw, sh, 0.04, col).T(x0 + x, y - sh / 2, 0))
    return merge(P)


def torpedo_mount(d):
    """A quintuple torpedo-tube mount on its turntable, trained fore and aft."""
    P = [cyl(1.4, 1.4, 0.5, GREY, seg=14)]
    for k in range(5):
        P.append(cyl(0.28, 0.28, 7.0, GREY_D, seg=8).R(math.pi / 2, 0, 0).T((k - 2) * 0.62, 0.9, -3.0))
    return merge(P).T(0, deck_y(d), zd(d))


def hull_number():
    out = []
    for s in (-1, 1):
        n = lettering("557", 2.2, (0.85, 0.85, 0.82))
        n.R(0, s * math.pi / 2, 0)
        z = zd(9.0); y = deck_y(9.0) - 3.6
        out.append(n.T(s * (hw(z, y) + 0.05), y, z))
    return out


extra = [torpedo_mount(50.0), torpedo_mount(66.0)] + hull_number()
whole = merge(bs_part(False) + extra)
crumple(whole, 0.15, 0.1, 57)
fc = whole.v[whole.f].mean(1); fd_ = L / 2 - fc[:, 2]
shells = np.zeros(len(fc), bool)
for k in range(26):                                                   # peppered with shell holes
    d0 = rng.uniform(6, 76); s = rng.choice([-1, 1]); y0 = rng.uniform(2.0, D + 3.0); r0 = rng.uniform(0.4, 1.1)
    shells |= (fc[:, 0] * s > 0) & (np.hypot(fd_ - d0, fc[:, 1] - y0) < r0)
cut = 77.0 + 3.0 * (fbm(fc, 0.15, 58) - 0.5)
keep = (fd_ < cut) & ~shells
k_, inv = np.unique(whole.f[keep], return_inverse=True)
fwd = Mesh(whole.v[k_], inv.reshape(-1, 3), whole.c[k_])
fwd.R(0, 0, math.radians(-4))
fwd.T(0, -np.percentile(fwd.v[:, 1], 0.5) - 1.2, 0)
debris = merge(plates(40, 0, zd(84), 10, 261, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q5229080" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", fwd), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-6456, kind="warship", seed=1969, memorial=True, drift=0.3)
ship = merge(bs_part(True) + extra); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
