"""Akagi (Q320967) — the wreck of the Japanese aircraft carrier lost at the Battle of Midway, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: laid down as a battlecruiser and completed in 1927 as a carrier, rebuilt in 1935-1938 with a single
full-length flight deck: about 260.7 m long and 31.3 m in the beam, the small island on the port side amidships, the
large funnel curving down over the starboard side with a smaller upright one beside it, three lifts, six 20 cm guns in
casemates aft (three on each side), twin 12 cm anti-aircraft mounts on sponsons, four screws; the imperial chrysanthemum
crest on the bow.

The wreck, after the published descriptions (Wikipedia, the Ocean Exploration Trust / NOAA survey of 2023 and its
photogrammetry): hit by dive bombers from USS Enterprise on 4 June 1942, she burned through the night and was scuttled
by torpedoes from Japanese destroyers the next morning, sinking bow first; 267 died. Found by R/V Petrel's sonar in
2019 at about 5,490 m, she was surveyed for 14 hours by the remotely operated vehicle Atalanta from E/V Nautilus on 10
September 2023: she sits upright, the bow driven deep into the seabed with the chrysanthemum crest still on it, the
flight deck and hangar wrecked, the 20 cm casemate guns at the stern, a debris field along her port side. The damage
and the heading (not published here: 0) are approximate.

    python model_sources/Q320967_akagi.py   -> models/item/Q320967/model.glb (+ model_built.glb)
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

rng = np.random.default_rng(1942 + 6)

L, B, D = 260.7, 31.3, 15.0
FC, TW, TT, WL = 0.0, 6.0, 1.2, 8.6
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.2, 0.5, 0.07, 0.3, 1.6, 0.3, 2.3, 6.0
HULLC = (0.4, 0.41, 0.43)
SUPC = (0.44, 0.45, 0.47)
GREY, GREY_L, GREY_D = (0.42, 0.43, 0.45), (0.5, 0.51, 0.53), (0.3, 0.3, 0.32)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.4, 0.41, 0.43)
HHW = 3.5
HATCHES = []
BULKHEADS = []
RAISED = []
HOUSES = []
FUNNELS = []
MASTS = []
BOATS = []
BOAT_X = 0.32
SCREWS = [(-8.5, L - 32.0, 2.6, 2.0), (-3.5, L - 24.0, 2.4, 2.0), (3.5, L - 24.0, 2.4, 2.0), (8.5, L - 32.0, 2.6, 2.0)]
RUDDER_D = L - 4.0
ANCHORS_GONE = ()
FDC = (0.45, 0.38, 0.28)
CARRIER = dict(hangar=11.0, fd=(8.0, L - 6.0), fdw=30.5, island=(112.0, 124.0, 5.0, 3), funnel=None, mast=(118.0, 8.0),
               lifts=[(40.0, 13.0, 12.0, 0.0), (120.0, 13.0, 12.0, 0.0), (200.0, 13.0, 12.0, 0.0)], open_hangar=False,
               sponsons=[(d, s, "twin") for d in (76.0, 92.0, 108.0) for s in (-1, 1)])


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


def akagi(built, holes=0.0):
    m = merge(carrier_part(0, L, built, holes=holes, seed=51))
    m.S(-1.0, 1.0, 1.0); m.f = m.f[:, ::-1]                                       # the island on the port side (+X)
    P = [m]
    d = 112.0                                                                       # the big funnel, curving down over the starboard side
    y = fd_y(d) - 4.0
    P.append(tube([(-B / 2 + 1.0, y + 2.0, zd(d)), (-B / 2 - 3.0, y - 1.0, zd(d)), (-B / 2 - 5.0, y - 5.0, zd(d))], 2.2, FUNNELC, seg=12))
    P.append(cyl(1.0, 1.0, 9.0, FUNNELC, seg=10).T(-B / 2 - 0.5, y, zd(d + 10)))
    for dd in (186.0, 196.0, 206.0):                                               # the 20 cm casemate guns, three a side
        for s in (-1, 1):
            z = zd(dd); yy = deck_y(dd) - 3.0
            P.append(cyl(0.18, 0.15, 7.0, STEEL, seg=8).R(math.pi / 2, s * 1.4, 0).T(s * (hw(z, yy) - 0.2), yy, z))
            P.append(quad(s * (hw(z, yy) + 0.04), yy, z, 1.8, 1.8, "+x" if s > 0 else "-x", DARK))
    for s in (-1, 1):                                                              # the chrysanthemum crest on the stem
        z = zd(1.2); yy = D - 1.0
        P.append(cyl(0.7, 0.7, 0.1, (0.62, 0.5, 0.2), seg=16).R(0, 0, math.pi / 2).T(s * (hw(z, yy) + 0.05), yy, z))
    return merge(P)


w = akagi(False, holes=0.22)
crumple(w, 0.3, 0.08, 52)
fc = w.v[w.f].mean(1)
w.f = w.f[~((fc[:, 1] > D) & (fbm(fc, 0.07, 53) > 0.55))]                         # the hangar burnt out, its sides fallen in
holes_in(w, [(60.0, 0.0, fd_y(60) + 0.5, 6.0), (120.0, 2.0, fd_y(120) + 0.5, 6.0), (190.0, -3.0, fd_y(190) + 0.5, 5.0),
             (140.0, B * 0.48, 4.0, 4.0), (100.0, -B * 0.48, 4.0, 4.0)], 54)
w.R(math.radians(1.5), 0, math.radians(3))                                         # bow down, deep in the seabed
w.T(0, -np.percentile(w.v[:, 1], 0.5) - 2.0, 0)
fc = w.v[w.f].mean(1)
w.f = w.f[fc[:, 1] > -3.5]
debris = merge(plates(100, 30, zd(130), 26, 551, cols=(GREY, FDC, GREY_D)))

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q320967" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-5490, kind="warship", seed=1942, memorial=True, drift=0.5)
ship = akagi(True); ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
