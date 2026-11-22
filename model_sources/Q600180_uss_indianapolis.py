"""USS Indianapolis (CA-35) (Q600180) — the wreck of the US heavy cruiser sunk in the Philippine Sea in 1945, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Portland-class heavy cruiser commissioned in 1932: about 186 m long, beam 20 m, nine 8-inch guns in three
triple turrets (two forward, one aft), eight single 5-inch anti-aircraft guns, a tripod foremast over the bridge,
two funnels, a mainmast, aircraft catapults amidships with their floatplanes, four screws.

The wreck, after the published descriptions of the expedition of the research vessel Petrel (Vulcan, Paul Allen),
which found her in August 2017 at about 5,500 m: having delivered parts of the first atomic bomb to Tinian, she was
torpedoed by the Japanese submarine I-58 shortly after midnight on 30 July 1945; the first torpedo blew away her bow,
the second struck amidships, and she sank in about twelve minutes. Of 1,195 men some 300 went down with her; only 316
survived four days in the water. Photographs released from the site show her hull number 35, the ship's name
plating, gun turrets, a 5-inch gun, one of the two anchor windlasses of the forecastle, the remains of one of her
floatplanes and a spare-parts box. The Navy keeps the location confidential and has not published the layout: here
the hull lies upright with the bow broken away beside it and a field of debris about them, an approximation.

    python model_sources/Q600180_uss_indianapolis.py   -> models/item/Q600180/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1945 + 30)

L, B, D = 186.0, 20.1, 12.5
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.4
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.8, 0.22, 0.45, 0.07, 0.3, 2.4, 0.4, 2.4, 5.5
RAISED = [(0, 70)]
SCREWS = [(-6.0, L - 26.0, 2.2, 1.8), (-2.2, L - 18.0, 2.0, 1.8), (2.2, L - 18.0, 2.0, 1.8), (6.0, L - 26.0, 2.2, 1.8)]
RUDDER_D = L - 3.0
BS = dict(turrets=[(42.0, 0.0, 3, 0.0), (54.0, 2.6, 3, 0.0), (148.0, 0.0, 3, 3.1416)],
          gun_len=10.5, house_w=10.0, bridge=(60.0, 76.0, 4, 11.0), funnels=[(86.0, 2.1, 9.0, 0.75), (100.0, 2.1, 9.0, 0.75)],
          masts=[(68.0, 22.0, "tripod"), (124.0, 18.0, "tripod")], casemates=[],
          secondary=[(80.0, 1), (80.0, -1), (108.0, 1), (108.0, -1)], rudders=[0.0])
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


install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship bs_turret bs_part pagoda capsize sub")


def hull_number():
    out = []
    for s in (-1, 1):
        n = lettering("35", 2.6, (0.85, 0.85, 0.82))
        n.R(0, s * math.pi / 2, 0)
        z = zd(12.0); y = deck_y(12.0) - 4.0
        out.append(n.T(s * (hw(z, y) + 0.05), y, z))
    return out


def floatplane():
    """What is left of a Curtiss SOC Seagull: fuselage, lower wing, the float."""
    return merge([box(1.0, 1.1, 8.0, GREY), box(10.0, 0.12, 1.6, GREY_L).T(0, -0.2, 1.0), box(0.8, 0.8, 4.6, GREY_D).T(0, -1.6, 0.8)])


def windlass_m():
    return merge([box(2.4, 1.0, 1.4, STEEL)] + [cyl(0.55, 0.55, 0.5, STEEL, seg=12).R(0, 0, math.pi / 2).T(s * 1.55, 0.8, 0) for s in (-1, 1)])


whole = merge(bs_part(False) + hull_number())
crumple(whole, 0.2, 0.1, 35)
fc = whole.v[whole.f].mean(1); fd_ = L / 2 - fc[:, 2]
tor2 = (fc[:, 0] < -B * 0.25) & (np.hypot(fd_ - 70.0, (fc[:, 1] - 4.0) * 1.2) < 6.0 * (0.8 + 0.4 * fbm(fc, 0.15, 36)))   # the second torpedo, starboard
cut = 24.0 + 3.0 * (fbm(fc, 0.12, 37) - 0.5)                                    # the first torpedo blew the bow away
bowm = sub(whole, (fd_ < cut) & ~tor2)
hull_ = sub(whole, (fd_ >= cut) & ~tor2)
hull_.R(0, 0, math.radians(8))
hull_.T(0, -np.percentile(hull_.v[:, 1], 0.5) - 1.5, 0)
low = hull_.v[:, 1] < -0.3; hull_.v[low, 1] = -0.3 + (hull_.v[low, 1] + 0.3) * 0.12
c = bowm.v.mean(0); bowm.T(-c[0], -c[1], -c[2]).R(0, 0, math.radians(-70)).R(0, math.radians(30), 0)
bowm.T(0, -bowm.v[:, 1].min() - 0.8, 0); bowm.T(-28.0, 0, zd(0.0) + 40.0)
# a turret torn out of its barbette, a floatplane, a windlass, the debris of the sinking
tur = bs_turret(3, 0.4, 0.0, 10.5); tur.R(0.3, 0, 1.2); tur.T(22.0, -tur.v[:, 1].min() - 0.5, zd(40.0))
fp = floatplane(); fp.R(0.2, 1.1, 2.7); fp.T(-14.0, -fp.v[:, 1].min() - 0.2, zd(110.0))
wl = windlass_m(); wl.R(0.4, 0.8, 0.3); wl.T(-18.0, -wl.v[:, 1].min(), zd(0.0) + 20.0)
box_ = box(1.4, 0.9, 1.0, GREY_D).R(0.1, 0.5, 0.2).T(8.0, 0.0, zd(130.0))
debris = merge([tur, fp, wl, box_] + plates(120, 0, zd(60), 40, 271, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q600180" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", hull_), ("bow_section", bowm), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-5500, kind="warship", seed=1975, memorial=True, drift=0.3)
ship = merge(bs_part(True) + hull_number()); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
