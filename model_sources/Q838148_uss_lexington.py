"""USS Lexington (CV-2) (Q838148) — the wreck of the US aircraft carrier lost at the Battle of the Coral Sea, built by
code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: laid down as a battlecruiser and completed in 1927 as one of the first American fleet carriers: about
270.7 m long and 32.3 m across her bulges, an enclosed hangar under a long flight deck, the island and the huge flat
funnel on the starboard side, two lifts, single 5-inch guns on sponsons at the corners of the flight deck (her
8-inch turrets were landed early in 1942), four screws.

The wreck, after the published descriptions (Wikipedia, the Naval History and Heritage Command, NAVSEA Carderock's
damage assessment, the Vulcan / R/V Petrel expedition): hit by two torpedoes and two bombs on 8 May 1942, she was torn
by petrol-vapour explosions and scuttled that evening by torpedoes from the destroyer Phelps; 216 died. R/V Petrel
found her on 4 March 2018 at about 3,000 m in the Coral Sea, some 430 nautical miles off Queensland: broken into
several pieces that rest together, her nameplate on the stern, and some of her aircraft — TBD Devastator torpedo
bombers and F4F Wildcat fighters — on the bottom around her, the markings still clear. The pieces, the damage, the
aircraft drawn and the heading (not published here: 0) are approximate.

    python model_sources/Q838148_uss_lexington.py   -> models/item/Q838148/model.glb (+ model_built.glb)
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
from airkit import Plane, Engine  # noqa: E402

rng = np.random.default_rng(1942 + 5)

L, B, D = 270.7, 32.3, 15.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 9.0
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.5, 0.07, 0.3, 1.2, 0.3, 2.3, 6.0
HULLC = (0.36, 0.38, 0.41)
SUPC = (0.42, 0.44, 0.47)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.45), (0.5, 0.52, 0.54), (0.3, 0.3, 0.32)
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
SCREWS = [(-9.0, L - 34.0, 2.8, 2.2), (-3.5, L - 26.0, 2.6, 2.2), (3.5, L - 26.0, 2.6, 2.2), (9.0, L - 34.0, 2.8, 2.2)]
RUDDER_D = L - 4.0
ANCHORS_GONE = ()
FDC = (0.42, 0.36, 0.28)
CARRIER = dict(hangar=7.5, fd=(6.0, L - 4.0), fdw=32.0, island=(98.0, 124.0, 6.0, 4), funnel=(126.0, 160.0, 15.0, 8.0), mast=(108.0, 14.0),
               lifts=[(42.0, 14.0, 12.0, 0.0), (232.0, 11.0, 11.0, 0.0)], open_hangar=False,
               sponsons=[(d, s, "single") for d in (16.0, 24.0, 32.0, 240.0, 248.0, 256.0) for s in (-1, 1)])
NAVY = (0.42, 0.46, 0.5)


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


def holes_in(m, spots, seed):
    """Remove faces in spheres (d, x, y, r): torpedo and bomb holes."""
    fc = m.v[m.f].mean(1); fd_ = L / 2 - fc[:, 2]
    hit = np.zeros(len(fc), bool)
    for (d0, x0, y0, r0) in spots:
        hit |= np.sqrt((fd_ - d0) ** 2 + (fc[:, 0] - x0) ** 2 + ((fc[:, 1] - y0) * 1.2) ** 2) < r0 * (0.8 + 0.4 * fbm(fc, 0.15, seed))
    m.f = m.f[~hit]
    return m


def tug():
    """A small aircraft tractor (tug) of the period."""
    return merge([box(1.6, 0.9, 2.8, (0.32, 0.34, 0.3)).T(0, 0.35, 0)] +
                 [cyl(0.35, 0.35, 0.25, RUBBER, seg=10).R(0, 0, math.pi / 2).T(s * 0.9, 0.35, dz) for s in (-1, 1) for dz in (-0.9, 0.9)])


def devastator():
    p = Plane(length=10.7, span=15.2, fus_w=1.4, fus_h=1.8, fus_y=1.8, wing_z=-2.8, root_chord=3.3, tip_chord=1.4, wing_y=0.95,
              dihedral=0.06, thick=0.17, tail_span=5.0, tail_chord=1.4, fin_h=1.8, fin_chord=1.6,
              engines=[Engine(0.0, 1.8, 0.0, r=0.72, prop=1.6, nacelle=1.4)], cockpit=(-6.8, -2.3), skin=NAVY, nose=0.1)
    return p.build()


def wildcat():
    p = Plane(length=8.8, span=11.6, fus_w=1.3, fus_h=1.7, fus_y=1.6, wing_z=-2.3, root_chord=2.6, tip_chord=1.3, wing_y=1.35,
              dihedral=0.0, thick=0.16, tail_span=4.4, tail_chord=1.2, fin_h=1.6, fin_chord=1.4,
              engines=[Engine(0.0, 1.6, 0.0, r=0.66, prop=1.5, nacelle=1.2)], cockpit=(-3.4, -2.7), skin=NAVY, nose=0.1)
    return p.build()


def star(x, y, z, r, nrm):
    """The US national insignia of 1942: a white star in a blue disc."""
    P = [cyl(r, r, 0.02, (0.12, 0.16, 0.36), seg=16)]
    pts = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        rr = r * (0.92 if k % 2 == 0 else 0.37)
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    v = [(0, 0.015, 0)] + [(px, 0.015, py) for px, py in pts]
    P.append(Mesh(v, [(0, k + 1, (k + 1) % 10 + 1) for k in range(10)], (0.9, 0.9, 0.88)))
    m = merge(P)
    return m.T(x, y, z) if nrm == "+y" else m.R(0, 0, -math.pi / 2).T(x, y, z)


def aircraft_wreck(build, seed, x, z, yaw, lost=()):
    parts = build()
    for k in lost:
        parts.pop(k, None)
    m = merge(list(parts.values()))
    yw = m.v[:, 1].max()
    m = merge([m, star(-4.0, 1.45, -3.5, 0.6, "+y")])
    crumple(m, 0.08, 0.6, seed)
    m.T(0, -m.v[:, 1].min() - 0.25, 0)
    return m.R(rng.uniform(-0.05, 0.05), yaw, rng.uniform(-0.08, 0.08)).T(x, 0, z)


def cut(m, d0, d1, seed):
    fc = m.v[m.f].mean(1); fd_ = L / 2 - fc[:, 2]
    j = 4.0 * (fbm(fc, 0.12, seed) - 0.5)
    keep = (fd_ + j >= d0) & (fd_ + j < d1)
    k, inv = np.unique(m.f[keep], return_inverse=True)
    return Mesh(m.v[k], inv.reshape(-1, 3), m.c[k])


w = merge(carrier_part(0, L, False, holes=0.16, seed=31))
crumple(w, 0.25, 0.08, 32)
holes_in(w, [(60.0, B * 0.48, 4.0, 5.0), (190.0, B * 0.48, 4.0, 5.0),                       # the Japanese torpedoes, port side
             (120.0, -B * 0.48, 4.0, 4.5), (150.0, -B * 0.48, 4.0, 4.5),                    # the scuttling torpedoes
             (30.0, 0.0, fd_y(30) + 0.5, 5.0), (140.0, 6.0, fd_y(140) + 0.5, 6.0)], 33)       # bomb hits and the explosions
fore = cut(w, 0, 92, 34)
mid = cut(w, 92, 214, 35)
aft = cut(w, 214, L, 36)
for m, dz, yaw, roll in ((fore, 6.0, -7.0, 3.0), (mid, 0.0, 0.0, -2.0), (aft, -7.0, 9.0, 5.0)):
    c = m.v.mean(0)
    m.T(-c[0], 0, -c[2]).R(0, math.radians(yaw), math.radians(roll)).T(c[0], 0, c[2] + dz)
    m.T(0, -np.percentile(m.v[:, 1], 0.5) - 2.0, 0)
planes = merge([aircraft_wreck(devastator, 41, 34.0, zd(60), 0.6), aircraft_wreck(devastator, 42, -30.0, zd(150), 2.4, lost=("tail",)),
                aircraft_wreck(wildcat, 43, 40.0, zd(200), -1.1), aircraft_wreck(devastator, 44, 26.0, zd(240), 3.6, lost=("wing_l",))])
debris = merge(plates(110, 20, zd(130), 36, 351, cols=(GREY, FDC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q838148" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", mid), ("bow_section", fore), ("stern_section", aft), ("debris_field", debris, "loose")], "mud": [],
                   "plain": [("aircraft", planes, "artefact")]}],
            depth=-3000, kind="warship", seed=1942, memorial=True, drift=0.4)
ship = merge(carrier_part(0, L, True)); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
