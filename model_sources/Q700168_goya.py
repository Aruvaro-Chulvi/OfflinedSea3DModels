"""MV Goya (Q700168) — the wreck of the German transport sunk in the Baltic in 1945, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Norwegian motor cargo ship built by Akers in Oslo and launched in 1940, seized by the Germans: about 146 m
long and 17.4 m in the beam, a forecastle, a long well deck forward with three hatches and king posts, the bridge and
engine house amidships with a short motor-ship funnel, two hatches aft and a poop; one screw. Used as a submarine
target ship, then in Operation Hannibal to carry refugees and soldiers out of East Prussia, in camouflage grey.

The wreck, after the published descriptions (Wikipedia, the Royal Navy's HMS Echo survey of 2021, Baltictech's
inventory dives): torpedoed by the Soviet submarine L-3 on the night of 16 April 1945 she sank in about seven minutes;
only some 183 of the more than 6,000 aboard survived. Found in 2003 by Ulrich Restemeyer's expedition, she sits
upright at about 76 m, in good condition but draped in lost fishing nets, broken towards the bow. A war grave:
diving is forbidden by Polish law. The break, the nets and the heading (not published here: 0) are approximate.

    python model_sources/Q700168_goya.py   -> models/item/Q700168/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1945 + 4)

L, B, D = 146.0, 17.4, 11.0
FC, TW, TT, WL = 2.4, 7.0, 1.3, 8.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.26, 0.75, 0.2, 0.42, 0.07, 0.3, 1.6, 0.5, 2.6, 7.0
HULLC = (0.33, 0.35, 0.34)
SUPC = (0.42, 0.44, 0.43)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.36, 0.38, 0.37)
HHW = 3.2
HATCHES = [(16, 24), (32, 40), (48, 56), (98, 106), (114, 122)]
BULKHEADS = [13, 62, 92, 130]
RAISED = [(0, 13), (62, 92), (130, L)]
HOUSES = [(64, 90, 0.76, 2.5, 0), (66, 86, 0.66, 2.5, 1), (66, 74, 0.52, 2.4, 2)]
FUNNELS = [(82.0, 2.6, 5.0, 0.7)]
MASTS = [(28.0, 18.0), (44.0, 18.0), (110.0, 17.0)]
BOATS = [(d, 1) for d in (72.0, 80.0)]
BOAT_X = 0.3
SCREWS = [(0.0, L - 5.5, 2.6, 2.4)]
RUDDER_D = L - 2.2
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


# ── Damage helpers ───────────────────────────────────────────────────────────
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship collapse holes lay boiler center_x")


def nets(m, n, seed):
    """Lost fishing nets caught on the wreck: torn curtains hanging down her sides from the deck edge and the masts."""
    rr = np.random.default_rng(seed)
    out = []
    for k in range(n):
        s = rr.choice([-1, 1])
        z = rr.uniform(m.v[:, 2].min() + 6, m.v[:, 2].max() - 6)
        near = m.v[np.abs(m.v[:, 2] - z) < 1.5]
        if not len(near):
            continue
        top = near[np.argmax(near[:, 1] + 0.3 * s * near[:, 0])]
        w, h = rr.uniform(4, 9), top[1] * rr.uniform(0.5, 0.9)
        g = grid(-w / 2, w / 2, 0, h, 8, 6, lambda X, Z: 0.5 * np.sin(Z / h * math.pi) * (1 - (2 * X / w) ** 2), (0.16, 0.18, 0.16))
        g.v = np.stack([s * (0.3 + g.v[:, 1]) + 0.0 * g.v[:, 0], -g.v[:, 2], g.v[:, 0]], 1)     # hang down the side, bellying outwards
        fc = g.v[g.f].mean(1)
        g.f = g.f[fbm(fc, 0.6, seed + k) > 0.3]                                                   # torn
        out.append(g.T(top[0], top[1], z))
    return merge(out)


bow = merge(ship_part(0, 30, False, mast_left=[9.0, 0.0, 0.0]))
main = merge(ship_part(31, L, False, mast_left=[0.0, 10.0, 8.0], funnel_state="stand", houses_damage=lambda m: crumple(m, 0.2, 0.3, 81)))
holes(main, [(84.0, -B * 0.48, 3.5, 4.0), (118.0, -B * 0.48, 3.5, 4.0)], 82)              # the two torpedoes (side approximate)
lay(bow, math.radians(25), 83, dz=4.0, yaw=math.radians(-14), sink=1.2)                      # the bow broken off, twisted to starboard
lay(main, math.radians(4), 84, sink=1.2)
center_x(main, bow)
netm = merge([nets(main, 14, 85), nets(bow, 4, 86)])
debris = merge(plates(50, 8, zd(30), 14, 851, cols=(HULLC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q700168" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", main), ("bow_section", bow), ("debris_field", debris, "loose")], "mud": [], "plain": [("nets", netm, "net")]}],
            depth=-76, kind="cargo", seed=1945, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", built_ship())])
