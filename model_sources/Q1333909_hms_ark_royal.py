"""HMS Ark Royal (Q1333909) — the wreck of the British aircraft carrier sunk off Gibraltar in 1941, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the first purpose-built large British fleet carrier, completed in 1938: about 240 m long, beam 28.9 m, two
hangars one above the other under a flight deck that ran almost from end to end, three lifts, the island with its
big funnel on the starboard side, sixteen 4.5-inch guns in twin turrets at the four corners of the flight deck,
three screws. Her Swordfish crippled the Bismarck in May 1941.

The wreck, after the published descriptions of the survey made for the BBC in 2002 by C & C Technologies with the
AUV Hugin 3000: torpedoed by U-81 on 13 November 1941, she listed slowly to starboard, rolled over and broke in two
as she sank, the after part first; one man died. She lies some 30 nautical miles east of Gibraltar at about
1,000 m, the site strewn over some 0.27 km², oriented roughly north to south: the stern part upright, the forward
part upside down with some 20 m of the bow broken off beside it, and between them a large field of debris in which
lie the island and the funnel; several gun turrets are still on the hull, and aircraft lie about the pieces. The
distances between the pieces, the angles and the heading are approximate.

    python model_sources/Q1333909_hms_ark_royal.py   -> models/item/Q1333909/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1941)

L, B, D = 232.0, 28.9, 13.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 6.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.5, 0.07, 0.3, 1.4, 0.3, 2.3, 6.0
HULLC = (0.45, 0.47, 0.49)
SUPC = (0.5, 0.52, 0.52)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.3, 0.31, 0.32)
HHW = 3.5
HATCHES = []
BULKHEADS = []
RAISED = []
HOUSES = []
FUNNELS = []
MASTS = []
BOATS = []
BOAT_X = 0.32
SCREWS = [(-6.0, L - 26.0, 2.6, 2.2), (0.0, L - 16.0, 2.4, 2.2), (6.0, L - 26.0, 2.6, 2.2)]
RUDDER_D = L - 4.0
ANCHORS_GONE = ()
FDC = (0.3, 0.29, 0.27)
CARRIER = dict(hangar=10.0, fd=(2.0, L - 2.0), fdw=29.5, island=(96.0, 126.0, 7.0, 5), funnel=(106.0, 118.0, 8.0, 5.5), mast=(100.0, 16.0),
               lifts=[(70.0, 14.0, 11.0, 0.0), (130.0, 14.0, 11.0, 0.0), (175.0, 14.0, 11.0, 0.0)],
               sponsons=[(d, s, "twin") for d in (36.0, 48.0, 186.0, 198.0) for s in (-1, 1)])


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


# ── An aircraft carrier: hangar, flight deck, island (driven by CARRIER below) ──
#   CARRIER = dict(hangar=height of the hangar above the hull deck, fd=(d0, d1), fdw=flight deck width,
#                  island=(d0, d1, width, levels), funnel=(d0, d1, height, width) or None, mast=(d, height),
#                  lifts=[(d, length, width, x)], sponsons=[(d, side, kind)], open_hangar=bool)
install(globals(), "zd t_of w_of yb_of yd_of k_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place ship_part make_onwreck plates corals built_ship fd_y carrier_part sub")


def swordfish():
    """A Fairey Swordfish biplane, roughly: fuselage, two wings, struts, tail (no fabric left: the frames)."""
    P = [box(1.1, 1.2, 9.5, GREY_D).T(0, 0, 0), box(13.9, 0.15, 1.8, GREY).T(0, -0.3, 1.5), box(13.9, 0.15, 1.8, GREY).T(0, 1.9, 1.8),
         box(4.8, 0.12, 1.2, GREY).T(0, 0.4, -4.2), box(0.12, 1.6, 1.2, GREY).T(0, 0.6, -4.3)]
    for x in (-5.5, -2.5, 2.5, 5.5):
        P.append(box(0.1, 2.2, 0.1, STEEL).T(x, -0.3, 1.6))
    return merge(P)


whole = merge(carrier_part(0, L, False, holes=0.05, seed=91))
crumple(whole, 0.2, 0.1, 92)
fc = whole.v[whole.f].mean(1); fd_ = L / 2 - fc[:, 2]
i0, i1, iw, lv = CARRIER["island"]
xi = -(CARRIER["fdw"] / 2 - iw / 2 - 0.3)
is_isl = (fd_ > i0 - 1) & (fd_ < i1 + 1) & (np.abs(fc[:, 0] - xi) < iw / 2 + 1.2) & (fc[:, 1] > fd_y(110.0) + 0.1)
cut = 128.0 + 4.0 * (fbm(fc, 0.1, 93) - 0.5)                                  # she broke in two abaft the island
torn = np.abs(fd_ - cut) < 3.0 + 3.0 * fbm(fc, 0.2, 94)
island = sub(whole, is_isl)
stern = sub(whole, (fd_ >= cut) & ~torn & ~is_isl)
fwd = sub(whole, (fd_ < cut) & (fd_ >= 20.0 + 2.0 * (fbm(fc, 0.2, 95) - 0.5)) & ~torn & ~is_isl)
bow = sub(whole, (fd_ < 20.0 + 2.0 * (fbm(fc, 0.2, 95) - 0.5)) & ~torn)


def settle(m, sink):
    m.T(0, -np.percentile(m.v[:, 1], 0.5) - sink, 0)
    low = m.v[:, 1] < -0.3
    m.v[low, 1] = -0.3 + (m.v[low, 1] + 0.3) * 0.12
    return m


# the stern part upright on the bottom, a slight list; it sank first
stern.R(0, 0, math.radians(-5)); settle(stern, 1.4)
# the forward part upside down, some 250 m to the south across the debris field, turned a little
c = fwd.v.mean(0); fwd.T(-c[0], 0, -c[2]).R(0, 0, math.pi - math.radians(6)).R(0, math.radians(-14), 0)
settle(fwd, 1.0); fwd.T(4.0, 0, zd(128.0) + 250.0 + 52.0)
# the bow, some 20 m of it, broken off and lying beside the forward part, on its side
c = bow.v.mean(0); bow.T(-c[0], -c[1], -c[2]).R(0, 0, math.radians(100)).R(0, math.radians(40), 0)
settle(bow, 0.8); bow.T(-26.0, 0, fwd.v[:, 2].max() + 12.0)
# the island and its funnel, fallen into the debris field between the two parts, on its side
c = island.v.mean(0); island.T(-c[0], -c[1], -c[2]).R(0, 0, math.radians(-80)).R(0, math.radians(25), 0)
settle(island, 0.5); island.T(-12.0, 0, zd(128.0) + 120.0)
D_ = []
for k in range(5):
    sw = swordfish(); crumple(sw, 0.15, 0.4, 300 + k)
    sw.R(rng.uniform(-0.3, 0.3) + (math.pi if k % 2 else 0.0), rng.uniform(0, 6.28), rng.uniform(-0.2, 0.2))
    D_.append(sw.T(rng.uniform(-40, 40), -sw.v[:, 1].min() if k % 2 == 0 else -sw.v[:, 1].min() - 0.2, zd(128.0) + rng.uniform(-30, 280)))
debris = merge(D_ + plates(170, 0, zd(128.0) + 130.0, 60, 191, cols=(GREY, FDC, GREY_D)) + plates(80, 0, zd(150.0), 30, 192, cols=(GREY, FDC)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1333909" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("stern_section", stern), ("bow_section", fwd), ("bow_piece", bow), ("island", island), ("debris_field", debris, "loose")],
                   "mud": [], "plain": []}], depth=-1000, kind="warship", seed=1941, drift=0.5)
ship = merge(carrier_part(0, L, True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
