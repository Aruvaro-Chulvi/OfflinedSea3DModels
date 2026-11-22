"""Aquarius Reef Base (Q623183) — the undersea laboratory off Key Largo, Florida, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, Florida International University's fact sheet and facilities
pages, NOAA): the only undersea research laboratory still working, built in 1986 and placed in 1993 on the sand beside
Conch Reef, in the Florida Keys National Marine Sanctuary, at about 18-19 m; run by NOAA and the University of North
Carolina Wilmington, and since 2013 by Florida International University. Scientists and NASA's NEEMO crews live in it
for days at a time, saturated, and dive from it. The habitat is a steel cylinder about 13-14 m long and 2.7-3 m across
(some 82 tonnes), with a wet porch and its moon pool at one end, the main lock and the living quarters with their
bunks and portholes, sitting on a 120-tonne baseplate that holds it about 4 m above the bottom; air, power and
communications come down an umbilical from a life-support buoy moored above it. A small air-filled shelter, the
«gazebo», stands nearby. Modelled: the baseplate girders on four jack legs and concrete pads, the saddles, the
hull with its stiffening rings and dished heads, round portholes and the big end viewport, the wet porch with its
moon pool and ladder, the divers' grating, handrails, gas-bottle racks, floodlights, the umbilical rising towards
the buoy, the gazebo with its viewport, ballast and air hose. Its heading, the layout round it and the exact
place of each fitting are approximate; the buoy is not drawn.

    python model_sources/Q623183_aquarius_reef_base.py   -> models/item/Q623183/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, fbm, sstep  # noqa: E402
from placekit import (rubble, REEF_ROCK, dress_place)  # noqa: E402
from shipkit import install  # noqa: E402

rng = np.random.default_rng(1993)
WHITE = (0.82, 0.82, 0.8)
STEEL = (0.45, 0.46, 0.47)
YELLOW = (0.8, 0.66, 0.2)
DARK = (0.04, 0.05, 0.06)
WHITE_D = (0.7, 0.71, 0.7)
CONC = (0.55, 0.54, 0.5)
GRATE = (0.3, 0.31, 0.3)


install(globals(), "kit_porthole")
R_H, Y_H, L_H = 1.37, 5.6, 12.6          # the pressure hull: radius, axis height, length of the cylinder
ZW = -L_H / 2                           # the wet-porch end (-z)


def beam(p0, p1, w, h, col=STEEL):
    """A steel beam of w × h section from p0 to p1 (horizontal or vertical)."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    ln = float(np.linalg.norm(d))
    m = box(w, h, ln, col, bottom=False)
    yaw = math.atan2(d[0], d[2])
    pitch = -math.asin(d[1] / ln) if ln else 0.0
    return m.R(pitch, 0, 0).R(0, yaw, 0).T(*((p0 + p1) / 2))


def ring(r0, r1, w, col, seg=32):
    return lathe([(r0, -w / 2), (r1, -w / 2), (r1, w / 2), (r0, w / 2)], col, seg=seg)


P = []
# ── the baseplate: two deep longitudinal girders, cross girders, four jack legs on their pads ──
for sx in (-1, 1):
    P.append(beam((sx * 1.9, 3.3, -8.2), (sx * 1.9, 3.3, 8.2), 0.45, 0.9))
    P.append(beam((sx * 1.9, 3.78, -8.2), (sx * 1.9, 3.78, 8.2), 0.8, 0.08))          # its top flange
for z in np.linspace(-7.6, 7.6, 7):
    P.append(beam((-1.9, 3.35, z), (1.9, 3.35, z), 0.35, 0.7))
for sx in (-1, 1):
    for sz in (-1, 1):
        x, z = sx * 1.9, sz * 7.0
        P.append(box(0.7, 3.0, 0.7, STEEL).T(x, 0.35, z))                                  # the leg
        P.append(cyl(0.18, 0.18, 0.9, (0.3, 0.3, 0.3), seg=10).T(x, 2.6, z))               # its jack screw
        P.append(box(1.8, 0.35, 1.8, CONC).T(x, 0.0, z))                                   # the concrete pad
        P.append(beam((x, 1.0, z), (x * 0.2, 2.9, z * 0.75), 0.18, 0.18))                 # a brace
# saddles that cradle the hull
for z in (-4.6, -0.6, 3.4):
    sad = lathe([(R_H + 0.02, -0.25), (R_H + 0.35, -0.25), (R_H + 0.35, 0.25), (R_H + 0.02, 0.25)], STEEL, seg=32).R(math.pi / 2, 0, 0)
    sad.f = sad.f[sad.v[sad.f][:, :, 1].mean(1) < -0.55]                                  # only its lower part
    P.append(sad.T(0, Y_H, z))
    P.append(box(3.0, Y_H - R_H * 0.75 - 3.8, 0.5, STEEL).T(0, 3.8, z))
# a grating deck round the wet porch, where divers stand
for z in np.arange(-8.0, -4.4, 0.25):
    P.append(box(4.6, 0.05, 0.05, GRATE).T(0, 3.86, z))
for x in np.arange(-2.25, 2.3, 0.45):
    P.append(box(0.06, 0.08, 3.7, GRATE).T(x, 3.8, -6.25))
# ── the pressure hull: cylinder with stiffening rings and dished heads ──
P.append(cyl(R_H, R_H, L_H, WHITE, seg=40, caps=False).R(math.pi / 2, 0, 0).T(0, Y_H, -L_H / 2))
for z in np.arange(-L_H / 2 + 0.9, L_H / 2 - 0.5, 1.5):
    P.append(ring(R_H - 0.02, R_H + 0.06, 0.14, WHITE_D).R(math.pi / 2, 0, 0).T(0, Y_H, z))
for s in (-1, 1):
    head = lathe([(R_H, 0.0), (R_H * 0.97, 0.25), (R_H * 0.85, 0.6), (R_H * 0.6, 0.9), (R_H * 0.3, 1.05), (0.001, 1.1)], WHITE, seg=40)
    P.append(head.R(s * math.pi / 2, 0, 0).T(0, Y_H, s * L_H / 2))
    P.append(ring(R_H - 0.02, R_H + 0.09, 0.18, WHITE_D).R(math.pi / 2, 0, 0).T(0, Y_H, s * L_H / 2))   # the head's flange
# the big viewport in the far head (the living quarters' window on the reef)
P.append(kit_porthole(0, Y_H + 0.1, L_H / 2 + 1.0, 0.42, "+z", rim=(0.62, 0.62, 0.6)))
# portholes along the bunk room and the main lock, on both sides
for s in (-1, 1):
    for z in (-2.6, -0.4, 1.4, 3.0, 4.4):
        P.append(kit_porthole(s * R_H, Y_H + 0.25, z, 0.17, "+x" if s > 0 else "-x", rim=(0.62, 0.62, 0.6)))
# ── the wet porch under the end: a chamber open at the bottom (the moon pool), its ladder down to the grating ──
wp = box(2.4, 1.9, 2.8, WHITE).T(0, 3.95, ZW - 0.2)
P.append(wp)
P.append(box(1.3, 0.04, 1.5, DARK).T(0, 3.93, ZW - 0.2))                                  # the moon pool opening
P.append(box(1.5, 0.12, 0.12, STEEL).T(0, 3.95, ZW - 1.0))
P.append(box(1.5, 0.12, 0.12, STEEL).T(0, 3.95, ZW + 0.6))
for sx in (-0.35, 0.35):
    P.append(tube([(sx, 3.85, ZW - 1.65), (sx, 0.3, ZW - 2.4)], 0.04, STEEL, seg=5))      # a ladder down to the sand
for k in range(1, 12):
    t = k / 12
    P.append(tube([(-0.35, 3.85 - t * 3.55, ZW - 1.65 - t * 0.75), (0.35, 3.85 - t * 3.55, ZW - 1.65 - t * 0.75)], 0.025, STEEL, seg=4))
# ── outside the hull: divers' handrails on standoffs, gas bottles, the umbilical, lights ──
for s in (-1, 1):
    zs = np.arange(-L_H / 2 + 0.5, L_H / 2 - 0.2, 1.5)
    for ang in (0.35, 1.05):
        xr, yr = s * (R_H + 0.25) * math.cos(ang), Y_H + (R_H + 0.25) * math.sin(ang)
        P.append(tube([(xr, yr, zs[0]), (xr, yr, zs[-1])], 0.035, YELLOW, seg=5))
        for z in zs:
            P.append(tube([(xr * R_H / (R_H + 0.25), Y_H + (yr - Y_H) * R_H / (R_H + 0.25), z), (xr, yr, z)], 0.03, YELLOW, seg=4))
# racks of gas bottles on the baseplate, both sides of the porch
for s in (-1, 1):
    for k in range(5):
        z = -3.6 + k * 0.38
        P.append(cyl(0.16, 0.16, 1.5, (0.2, 0.36, 0.25), seg=10).T(s * 1.55, 3.84, z))
        P.append(lathe([(0.16, 0), (0.1, 0.12), (0.04, 0.2)], (0.2, 0.36, 0.25), seg=10).T(s * 1.55, 5.34, z))
    P.append(box(0.45, 0.08, 2.1, STEEL).T(s * 1.55, 4.6, -2.84))
# on the top: the junction boxes and the equipment housing, two floodlights on a short mast
P.append(box(0.9, 0.55, 2.6, YELLOW).T(0, Y_H + R_H - 0.1, 1.2))
P.append(box(0.6, 0.4, 0.8, STEEL).T(0, Y_H + R_H - 0.1, -1.8))
P.append(box(0.5, 0.35, 0.6, STEEL).T(0.5, Y_H + R_H - 0.15, -3.4))
P.append(tube([(0, Y_H + R_H + 0.4, 3.6), (0, Y_H + R_H + 1.6, 3.6)], 0.05, STEEL, seg=6))
for s in (-1, 1):
    P.append(lathe([(0.001, 0), (0.12, 0.05), (0.15, 0.25)], (0.15, 0.15, 0.15), seg=10).R(0, 0, s * 1.2).T(s * 0.15, Y_H + R_H + 1.6, 3.6))
# the umbilical: a bundle of hoses and cables from the junction box, rising towards the life-support buoy
umb = [(0.2, Y_H + R_H + 0.4, -1.8), (0.4, Y_H + R_H + 1.2, -2.6), (0.6, Y_H + R_H + 3.0, -3.4), (0.9, Y_H + R_H + 6.5, -3.9), (1.2, Y_H + R_H + 11.0, -4.1)]
P.append(tube(umb, 0.11, (0.08, 0.08, 0.08), seg=8))
P.append(tube([(p[0] + 0.16, p[1], p[2]) for p in umb], 0.05, YELLOW, seg=6))
habitat = merge(P)
# ── the gazebo: an air-filled refuge on its legs, a bell with a viewport and its air hose from the habitat ──
G = [lathe([(1.6, 0), (1.62, 0.15), (1.52, 0.8), (1.15, 1.45), (0.55, 1.85), (0.001, 1.95)], YELLOW, seg=28).T(0, 1.6, 0)]
G.append(ring(1.58, 1.7, 0.16, STEEL, seg=28).T(0, 1.6, 0))
for a in np.linspace(0, 2 * math.pi, 5)[:-1] + math.pi / 4:
    x, z = math.cos(a) * 1.45, math.sin(a) * 1.45
    G.append(box(0.16, 1.7, 0.16, STEEL).T(x, -0.05, z))
    G.append(box(0.7, 0.3, 0.7, CONC).T(x * 1.05, -0.05, z * 1.05))                     # ballast blocks on its feet
G.append(kit_porthole(0, 2.7, 1.32, 0.22, "+z", rim=(0.62, 0.62, 0.6)))
gazebo = merge(G).T(14, 0, 8)
hose = tube([(1.5, 3.9, 3.8), (5.0, 1.0, 5.2), (9.0, 0.25, 6.6), (12.6, 0.25, 7.6), (13.6, 1.6, 8.0)], 0.07, (0.08, 0.08, 0.08), seg=6)
# Conch Reef beside it: spurs of reef rock with sand channels between them
def reef(X, Z):
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    spur = np.maximum(0, np.sin(Z / 4.5 + 1.2 * fbm(P3, 0.05, 3, 2).reshape(X.shape)))
    return (1.8 * spur + 0.6 * (fbm(P3, 0.2, 4, 3).reshape(X.shape) - 0.5)) * sstep(-14, -24, X) * sstep(-60, -50, X) * sstep(40, 30, np.abs(Z))
rock = grid(-60, -10, -40, 40, 60, 90, reef, REEF_ROCK)
rock.f = rock.f[rock.v[rock.f][:, :, 1].max(axis=1) > 0.05]
stones = rubble(30, -16, 0, 10, 0.6, REEF_ROCK, 7)

out = ROOT / "models" / "item" / "Q623183" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("habitat", habitat, "paint"), ("gazebo", merge([gazebo, hose]), "paint"), ("reef", rock, "rock"), ("stones", stones, "rock")],
            [], depth=-18, seed=1993, drift=0.15)
