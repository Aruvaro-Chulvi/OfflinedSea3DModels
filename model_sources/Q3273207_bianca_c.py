"""
MV Bianca C (Q3273207) — the wreck of the Costa liner off Grenada, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: laid down in France as Maréchal Pétain, completed in 1949 as La Marseillaise, later Arosa Sky and,
from 1959, Bianca C of the Costa Line: about 180 m long, beam 23.8 m, twin screws, one large funnel, two masts,
a long superstructure with the swimming pools on the after decks.

The wreck, after the published descriptions and dive guides: an engine-room explosion and fire on 22 October 1961
in St George's harbour; towed out, she sank off Pink Gin Beach; she is called the Titanic of the Caribbean. She
sits upright on her keel at about 50 m, the superstructure and bow from about 27 m, the pools at about 40 m.
The fire and the years have taken their toll: parts of the superstructure have fallen in, most of the
starboard side has collapsed onto the seabed, the masts have toppled and the fantail has broken away among
plates from the superstructure. Strong currents. The heading (not published: 0), the colours and the extent of
the collapse are approximate.

    python model_sources/Q3273207_bianca_c.py   -> models/item/Q3273207/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1961)

L, B, D = 180.0, 23.8, 13.0
FC, TW, TT, WL = 2.6, 8.0, 1.4, 7.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.28, 0.32, 0.08, 0.35, 2.2, 0.8, 2.4, 6.0
HULLC = (0.86, 0.86, 0.84)                   # white
SUPC = (0.88, 0.88, 0.86)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.88, 0.75, 0.25)
HHW = 3.5
HATCHES = [(14, 21), (24, 30)]
BULKHEADS = [12, 32, 160]
RAISED = [(0, L)]
HOUSES = [(36, 158, 0.92, 2.5, 0), (38, 150, 0.88, 2.5, 1), (40, 136, 0.82, 2.5, 2), (42, 118, 0.6, 2.5, 3), (42, 50, 0.6, 2.4, 4)]
FUNNELS = [(98.0, 4.0, 9.5, 0.62)]
MASTS = [(31.0, 20.0), (150.0, 16.0)]
BOATS = [(d, 3) for d in (56.0, 64.0, 72.0, 80.0, 108.0, 115.0)]
BOAT_X = 0.31
SCREWS = [(-4.6, L - 12.0, 2.6, 2.2), (4.6, L - 12.0, 2.6, 2.2)]
RUDDER_D = L - 4.0
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


install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship")


def funnel_mark(d, r, h, yb):
    """The blue band and the big C of the Costa Line on both sides of the funnel."""
    out = []
    for s in (-1, 1):
        x = s * (r * 0.62 + 0.08)
        out.append(quad(x, yb + h * 0.6, zd(d), 2 * r * 0.95, 3.0, "+x" if s > 0 else "-x", (0.12, 0.25, 0.55), eps=0.0))
        for (dz, dy, w, hh) in ((-0.9, 0.0, 0.45, 2.2), (0.1, 0.9, 1.6, 0.4), (0.1, -0.9, 1.6, 0.4)):
            out.append(quad(x + s * 0.02, yb + h * 0.6 + dy, zd(d) + dz, w, hh, "+x" if s > 0 else "-x", (0.92, 0.9, 0.85), eps=0.0))
    return out


def pools():
    out = []
    for d in (128.0, 140.0):
        y = deck_y(d) + FC + LEVEL_H * 2 + 0.02
        out += [box(7.0, 0.05, 6.0, (0.2, 0.45, 0.55)).T(0, y - 1.2, zd(d)), box(7.4, 1.25, 0.2, SUPC).T(0, y - 1.2, zd(d) + 3.0),
                box(7.4, 1.25, 0.2, SUPC).T(0, y - 1.2, zd(d) - 3.0)]
    return out


yfun = deck_y(98) + FC + LEVEL_H * 3 + 2.5 - 0.2           # on the roof of the boat-deck house, like the funnel
FANT = 166.0                                                                  # the fantail broke away aft of here
main = merge(ship_part(0, FANT, False, mast_left=[0.0, 0.0], funnel_state="stand",
                       houses_damage=lambda m: (crumple(m, 0.9, 0.22, 61), m.__setattr__('f', m.f[fbm(m.v[m.f].mean(1), 0.09, 62) < 0.62])))
             + funnel_mark(98.0, 4.0, 9.5, yfun) + pools())
crumple(main, 0.18, 0.1, 11)
stb = main.v[:, 0] < -B * 0.25                                                # most of the starboard side collapsed outwards onto the seabed
hgt = sstep(3.0, D + FC + 12, main.v[:, 1])
along = sstep(20, 40, L / 2 - main.v[:, 2]) * sstep(FANT, FANT - 20, L / 2 - main.v[:, 2])
k = stb * hgt * along * (main.v[:, 1] < D + FC + 1.0)                         # the hull side falls outwards…
fc = main.v[main.f].mean(1)
zone = (fc[:, 0] < -B * 0.2) & (fc[:, 1] > D + FC + 0.5)
dark = main.c[main.f].mean(1).mean(1) < 0.08                                 # (their windows go with them)
main.f = main.f[~(zone & ((fbm(fc, 0.07, 63) > 0.42) | dark))]   # …and the houses above it are gone
main.v[:, 0] -= k * (main.v[:, 1] - 3.0) * 0.85
main.v[:, 1] -= k * (main.v[:, 1] - 3.0) * 0.7
crumple(main, 1.2, 0.2, 12, k)
main.T(0, -0.8, 0)
fant = merge(ship_part(FANT, L, False))
crumple(fant, 0.3, 0.2, 13)
crumple(fant, 1.4, 0.3, 14, sstep(FANT + 5, FANT, L / 2 - fant.v[:, 2]))
zf = zd((FANT + L) / 2)
fant.T(0, 0, -zf).R(0, 0, -0.35).R(0, 0.3, 0)
fant.T(3.0, -np.percentile(fant.v[:, 1], 0.5) - 0.8, zf - 9.0)
deb = plates(70, -14, zd(120), 22, 100, cols=(SUPC, HULLC, DECKC)) + plates(30, 6, zd(FANT) - 6, 10, 101, cols=(SUPC, DECKC))
for k_, (d, ln, s) in enumerate(((31.0, 18.0, -1), (150.0, 15.0, 1))):         # the toppled masts
    m = merge([cyl(0.45, 0.25, ln, SUPC, seg=8, caps=False), box(3.0, 0.25, 0.4, SUPC).T(0, ln * 0.6, 0)])
    crumple(m, 0.12, 0.4, 60 + k_)
    deb.append(m.R(0, 0, s * (math.pi / 2 - 0.08)).R(0, 0.3, 0).T(s * (B / 2 + 2.0), 0.4, zd(d)))
debris = merge(deb)
coral = merge([corals(main, 380, 7, size=(0.15, 0.5)), corals(fant, 60, 8, size=(0.15, 0.5))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q3273207" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", main), ("stern_section", fant), ("debris_field", debris, "loose")], "mud": [],
                   "plain": [("corals", coral, "life")]}],
            depth=-50, kind="liner", seed=1961, life="tropical")
dress_built(out.with_name("model_built.glb"), [("main", merge([built_ship()] + funnel_mark(98.0, 4.0, 9.5, yfun) + pools()))])
