"""Aqaba Underwater Military Museum (L2, own catalog entry) — the military vehicles sunk off Aqaba in 2019, built by
code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (AFP via Phys.org, DIVE Magazine, PADI, Duikersgids): in July 2019 the
Aqaba Special Economic Zone Authority and the Jordanian armed forces sank some 19-21 decommissioned military vehicles
off Aqaba's South Beach, in the Red Sea, to make an artificial reef and a dive site: tanks, an ambulance, a military
crane, a troop carrier, an anti-aircraft gun and a combat helicopter. They were laid out along the coral reef as if
in a battle formation, over some 140 m, eight of them at 15-20 m and the others at 20-28 m, a couple near 5 m. The
types, the number and the places of the vehicles drawn are approximate; the model sets them on flat ground at about
22 m.

    python model_sources/L2_aqaba_military_museum.py   -> models/item/L2/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, quad, tube, crumple, fbm  # noqa: E402
from placekit import (rubble, REEF_ROCK, dress_place)  # noqa: E402

rng = np.random.default_rng(2019)
SAND = (0.6, 0.52, 0.36)
OLIVE = (0.36, 0.37, 0.26)
DARK = (0.06, 0.06, 0.06)
WHITE = (0.8, 0.8, 0.78)
RED = (0.6, 0.12, 0.1)


def tracks(L, W, col=DARK):
    P = []
    for s in (-1, 1):
        P.append(box(0.6, 0.9, L, col).T(s * (W / 2 - 0.3), 0, 0))
        P += [cyl(0.38, 0.38, 0.5, (0.12, 0.12, 0.12), seg=10).R(0, 0, math.pi / 2).T(s * (W / 2 - 0.3), 0.45, z) for z in np.linspace(-L / 2 + 0.6, L / 2 - 0.6, 6)]
    return P


def tank(col):
    P = tracks(7.4, 3.6) + [box(3.4, 1.0, 7.6, col).T(0, 0.7, 0), box(2.4, 0.9, 3.0, col).T(0, 1.7, -0.6)]
    P.append(cyl(0.12, 0.1, 5.6, col, seg=8).R(math.pi / 2, 0, 0).T(0, 2.15, 0.6))
    P.append(box(0.5, 0.4, 0.6, col).T(0.7, 2.6, -1.0))
    return merge(P)


def apc(col):
    P = tracks(5.0, 2.8) + [box(2.7, 1.4, 5.2, col).T(0, 0.6, 0)]
    P[-1].v[:, 1] -= 0.4 * np.clip((P[-1].v[:, 2] - 1.5) / 1.1, 0, 1) * (P[-1].v[:, 1] > 1.5)
    P.append(cyl(0.05, 0.05, 1.2, DARK, seg=5).R(math.pi / 2, 0, 0).T(0.5, 2.1, 0.6))
    return merge(P)


def truck(col, body):
    P = [cyl(0.5, 0.5, 0.4, DARK, seg=12).R(0, 0, math.pi / 2).T(s * 1.1, 0.5, z) for s in (-1, 1) for z in (-2.2, 2.0)]
    P += [box(2.3, 0.5, 6.4, (0.2, 0.2, 0.2)).T(0, 0.6, 0), box(2.3, 1.6, 1.8, col).T(0, 1.1, 2.1)]
    P.append(quad(0, 2.2, 3.01, 1.8, 0.6, "+z", DARK))
    if body == "box":                                                    # the ambulance body, with its red cross
        P.append(box(2.4, 2.0, 3.8, WHITE).T(0, 1.1, -1.0))
        for s in (-1, 1):
            P += [quad(s * 1.21, 2.1, -1.0, 0.35, 1.1, "+x" if s > 0 else "-x", RED), quad(s * 1.21, 2.1, -1.0, 1.1, 0.35, "+x" if s > 0 else "-x", RED)]
    else:                                                                # the crane: its boom laid along the truck
        P.append(box(1.8, 0.8, 1.6, col).T(0, 1.1, -1.6))
        P.append(box(0.5, 0.5, 7.0, (0.7, 0.55, 0.15)).R(-0.12, 0, 0).T(0, 1.9, 0.6))
    return merge(P)


def aa_gun(col):
    P = [box(2.2, 0.6, 2.2, col), cyl(0.7, 0.7, 0.5, col, seg=12).T(0, 0.6, 0), box(1.2, 0.8, 1.0, col).T(0, 1.1, 0)]
    P += [cyl(0.07, 0.06, 3.4, DARK, seg=6).R(math.pi / 2 - 0.4, 0, 0).T(s * 0.35, 1.5, 1.4) for s in (-1, 1)]
    P += [box(0.25, 0.25, 3.2, col).R(0, a, 0).T(math.sin(a) * 1.6, 0, math.cos(a) * 1.6) for a in (0.4, 2.0, 3.6, 5.2)]
    return merge(P)


def helicopter(col):
    """An attack helicopter: slim fuselage with stepped cockpits, stub wings, tail boom and fin, the rotor."""
    P = [lathe([(0.001, -2.5), (0.6, -2.0), (0.8, 0.0), (0.7, 2.0), (0.3, 3.2), (0.001, 3.6)], col, seg=12).R(math.pi / 2, 0, 0).T(0, 1.4, 0)]
    P[0].v[:, 0] *= 0.6
    P.append(box(0.9, 0.7, 3.2, (0.12, 0.16, 0.18)).T(0, 1.9, 1.4))
    P.append(cyl(0.3, 0.12, 6.5, col, seg=8).R(-math.pi / 2, 0, 0).T(0, 1.6, -5.4))
    P.append(box(0.15, 1.6, 1.0, col).T(0, 1.4, -8.4))
    P.append(box(3.2, 0.12, 0.8, col).T(0, 1.2, 0.2))
    P += [box(0.25, 0.25, 2.8, (0.2, 0.2, 0.2)).T(s * 0.7, 0, 0.2) for s in (-1, 1)]                  # skids
    P += [box(0.35, 0.06, 6.5, (0.15, 0.15, 0.15)).R(0, a, 0).T(0, 2.8, 0) for a in (0.3, 0.3 + math.pi / 2)]
    P.append(cyl(0.2, 0.2, 0.6, col, seg=8).T(0, 2.2, 0))
    return merge(P)


V = []
# the formation along the reef: tanks in front, the troop carriers and the anti-aircraft gun behind, the crane and the
# ambulance at the back, the helicopter at the head of the column
layout = [(tank, -40, 0, 0.0), (tank, -24, 6, 0.1), (tank, -8, -4, -0.1), (tank, 8, 5, 0.05), (tank, 24, -3, 0.0), (tank, 40, 4, 0.1),
          (apc, -32, 22, 0.2), (apc, -14, 24, 0.0), (apc, 4, 22, -0.1), (apc, 22, 24, 0.1),
          (aa_gun, 40, 22, 0.0), (aa_gun, -48, 20, 0.0),
          (lambda c: truck(c, "crane"), -20, 40, 0.15), (lambda c: truck(c, "box"), 0, 42, -0.05), (lambda c: truck(c, "crane"), 20, 40, 0.0),
          (helicopter, 60, 10, -0.6), (tank, -58, -2, 0.3), (apc, 58, 30, -0.2), (tank, 64, -10, 0.2)]
for k, (fn, x, z, yaw) in enumerate(layout):
    m = fn(SAND if k % 3 else OLIVE)
    crumple(m, 0.02, 1.0, k)
    V.append(m.R(0, math.pi / 2 + yaw, 0).T(x, 0, z))
vehicles = merge(V)
reef = rubble(90, 0, -26, 30, 1.4, REEF_ROCK, 31, flat=0.7)

out = ROOT / "models" / "item" / "L2" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("vehicles", vehicles, "metal"), ("reef", reef, "rock")], [], depth=-22, seed=2019, drift=0.15)
