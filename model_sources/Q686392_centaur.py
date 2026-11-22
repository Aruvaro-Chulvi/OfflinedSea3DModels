"""AHS Centaur (Q686392) — the wreck of the Australian hospital ship sunk off Queensland in 1943, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Blue Funnel Line passenger-cargo motor ship built in 1924 for the Fremantle-Singapore trade: about 96 m
long and 18 m in the beam, a forecastle, hatches fore and aft, a midship house with a single tall funnel, two masts,
one screw. Converted to a hospital ship in 1943: white hull with a broad green band, large red crosses on her sides,
and the Red Cross number 47 on her bow.

The wreck, after the published descriptions (Wikipedia, the Australian War Memorial, the Queensland State Library,
Science Illustrated): on 14 May 1943, lit up and marked as a hospital ship, she was torpedoed by the Japanese submarine
I-177 off Moreton Island and sank in about three minutes; 268 of the 332 aboard died. David Mearns's team found her on
20 December 2009 about 30 nautical miles east of Moreton Island at 2,059 m and filmed her with a remotely operated
vehicle in January 2010: very badly damaged, the hull broken in at least one and perhaps two places though in one
piece, the red crosses and the number 47 on her bow still visible. A war grave. The published sources do not describe
how she lies: the model has her upright, the hull broken through forward. Attitude, damage and heading (not published
here: 0) are approximate.

    python model_sources/Q686392_centaur.py   -> models/item/Q686392/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1943 + 5)

L, B, D = 96.2, 18.0, 9.0
FC, TW, TT, WL = 2.4, 6.0, 1.2, 6.4
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.24, 0.75, 0.2, 0.45, 0.07, 0.3, 1.4, 0.5, 2.6, 7.0
HULLC = (0.82, 0.82, 0.8)                     # hospital ship white with a green band (painted below)
SUPC = (0.84, 0.84, 0.82)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.82, 0.82, 0.8)
HHW = 2.6
HATCHES = [(12, 18), (22, 28), (72, 78), (82, 87)]
BULKHEADS = [10, 32, 66, 90]
RAISED = [(0, 10), (32, 66), (90, L)]
HOUSES = [(34, 64, 0.76, 2.5, 0), (36, 58, 0.66, 2.5, 1), (36, 42, 0.52, 2.4, 2)]
FUNNELS = [(50.0, 2.2, 11.0, 0.9)]
MASTS = [(20.0, 18.0), (80.0, 16.0)]
BOATS = [(d, 1) for d in (44.0, 54.0)]
BOAT_X = 0.3
SCREWS = [(0.0, L - 4.5, 2.0, 1.8)]
RUDDER_D = L - 1.8
ANCHORS_GONE = ()
GREEN_BAND = (0.16, 0.42, 0.22)
REDX = (0.62, 0.08, 0.06)


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


# ── Damage helpers ───────────────────────────────────────────────────────────
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship collapse holes lay boiler center_x")


def markings():
    """The green band along the hull, the red crosses on both sides and the number 47 at the bow."""
    P = []
    for s in (-1, 1):
        for d in np.arange(4, L - 3, 1.5):
            z = zd(d); y = WL + 0.6
            P.append(quad(s * (hw(z, y) + 0.03), y, z, 1.55, 0.9, "+x" if s > 0 else "-x", GREEN_BAND))
        for d in (44.0, 76.0):
            z = zd(d); y = WL + 2.4
            x = s * (hw(z, y) + 0.05)
            P += [quad(x, y, z, 1.1, 3.6, "+x" if s > 0 else "-x", REDX), quad(x, y, z, 3.6, 1.1, "+x" if s > 0 else "-x", REDX)]
        z = zd(5.0); y = D * 0.8
        x = s * (hw(z, y) + 0.05)
        for k, bars in enumerate([[(0, 0.6, 0.3, 1.5), (-0.4, 0.0, 1.1, 0.3), (0.2, -0.6, 0.3, 1.5)],           # 4
                                  [(0, 1.1, 1.0, 0.3), (0.35, 0.0, 0.3, 2.4)]]):                              # 7
            for dz, yy, w, h in bars:
                P.append(quad(x, y + yy, z + s * (-0.8 + 1.6 * k) + dz, w, h, "+x" if s > 0 else "-x", BLACK))
    return P


wreck = merge(ship_part(0, L, False, mast_left=[0.0, 3.0], funnel_state="gone", davits=False) + markings())
holes(wreck, [(24.0, B * 0.48, 3.5, 5.0), (30.0, B * 0.2, 6.0, 4.0)], 71)                          # the torpedo, port side forward
fc = wreck.v[wreck.f].mean(1); fd_ = L / 2 - fc[:, 2]
wreck.f = wreck.f[~((np.abs(fd_ - 30.0) < 1.5 + 2.0 * fbm(fc, 0.2, 72)) & (fc[:, 1] > 1.0))]          # broken through forward of the bridge
collapse(wreck, D + FC + 0.5, 0.45, 73)
wreck.R(0, 0, math.radians(-6))
crumple(wreck, 0.15, 0.1, 74)
wreck.T(0, -np.percentile(wreck.v[:, 1], 0.5) - 1.5, 0)
debris = merge(plates(50, 8, zd(30), 12, 741, cols=(HULLC, SUPC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q686392" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-2059, kind="liner", seed=1943, memorial=True, drift=0.5)
dress_built(out.with_name("model_built.glb"), [("main", merge([built_ship(), merge(markings()).T(0, 0, -zd(L / 2))]))], deck_colors=[DECKC])
