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
«gazebo», stands nearby. Its heading and the layout round it are approximate; the buoy is not drawn.

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

rng = np.random.default_rng(1993)
WHITE = (0.82, 0.82, 0.8)
STEEL = (0.45, 0.46, 0.47)
YELLOW = (0.8, 0.66, 0.2)
DARK = (0.04, 0.05, 0.06)


P = []
# the baseplate: a heavy steel frame on four legs
P.append(box(4.4, 0.6, 15.0, STEEL).T(0, 3.4, 0))
for sx in (-1, 1):
    for sz in (-1, 1):
        P.append(box(0.8, 3.6, 0.8, STEEL).T(sx * 1.8, 0, sz * 6.6))
        P.append(box(1.6, 0.3, 1.6, STEEL).T(sx * 1.8, 0, sz * 6.6))
for z in (-5.0, 0.0, 5.0):                                               # saddles
    P.append(box(3.2, 0.6, 0.5, STEEL).T(0, 3.9, z))
hull = cyl(1.4, 1.4, 13.0, WHITE, seg=24).R(math.pi / 2, 0, 0).T(0, 5.4, 0)
ends = [lathe([(1.4, 0), (1.2, 0.5), (0.7, 0.9), (0.001, 1.0)], WHITE, seg=24).R(s * math.pi / 2, 0, 0).T(0, 5.4, s * 6.5) for s in (-1, 1)]
P += [hull] + ends
for s in (-1, 1):                                                        # portholes along the living quarters
    P += [quad(s * 1.42, 5.6, z, 0.4, 0.4, "+x" if s > 0 else "-x", DARK) for z in np.arange(0.3, 5.5, 1.3)]
P.append(box(2.6, 1.6, 3.0, WHITE).T(0, 3.9, -5.6))                       # the wet porch under the end of the hull
P.append(box(1.4, 0.05, 1.6, DARK).T(0, 3.88, -5.6))                       # its moon pool
P.append(box(1.0, 0.8, 4.0, YELLOW).T(0, 6.8, 1.5))                        # equipment housings on top
P.append(tube([(0.0, 7.0, 4.5), (0.6, 8.0, 5.5), (1.2, 9.0, 6.0)], 0.12, (0.1, 0.1, 0.1), seg=6))   # the umbilical, up towards the buoy
gazebo = merge([lathe([(1.6, 0), (1.5, 0.8), (1.0, 1.6), (0.001, 1.9)], YELLOW, seg=18).T(0, 1.6, 0)] +
               [box(0.2, 1.6, 0.2, STEEL).T(math.cos(a) * 1.4, 0, math.sin(a) * 1.4) for a in np.linspace(0, 6.28, 5)[:-1]]).T(14, 0, 8)
habitat = merge(P)
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
dress_place(out, [("habitat", habitat, "paint"), ("gazebo", gazebo, "paint"), ("reef", rock, "rock"), ("stones", stones, "rock")],
            [], depth=-18, seed=1993, drift=0.15)
