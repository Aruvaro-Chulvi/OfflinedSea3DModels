"""Pavlopetri (Q966038) — the drowned Bronze Age town off Laconia, Greece, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the University of Nottingham's Pavlopetri Underwater
Archaeology Project 2009-2013 with the Greek Ephorate of Underwater Antiquities): a town inhabited from about 2800 BC
and through the Mycenaean period, sunk by the slow subsidence of the coast. It lies off Pounta, opposite the islet of
Pavlopetri, in 3-4 m of water: the oldest submerged town known with an almost complete plan — streets, at least 15
buildings with their courtyards, chamber tombs and cist graves — the walls standing a course or two above the sand,
spread over some 8 hectares; the surveys of 2009 and after mapped it with sonar and photogrammetry. Here the heart of
the town is drawn, about 160 m across; the plan is after the published site plans in outline only: the line of each
wall and building is approximate.

    python model_sources/Q966038_pavlopetri.py   -> models/item/Q966038/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, fbm  # noqa: E402
from placekit import (ashlar_wall, rubble, amphorae, LIMESTONE, CLAY, dress_place)  # noqa: E402

rng = np.random.default_rng(2800)
STONE = (0.6, 0.57, 0.5)


def house(x, z, w, d, yaw, seed, rooms=2):
    """A house: rough stone walls a course or two high round its rooms, a doorway on the street side."""
    rr = np.random.default_rng(seed)
    c, s = math.cos(yaw), math.sin(yaw)
    P = lambda u, v: (x + u * c + v * s, z - u * s + v * c)
    W = []
    corners = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        if k == 0:                                         # the street wall, with its doorway
            m = (a[0] + b[0]) / 2
            W.append(ashlar_wall(P(*a), P(m - 0.6, a[1]), rr.uniform(0.7, 1.1), 0.6, STONE, seed + k, block=0.55, course=0.28, ruin=0.4))
            W.append(ashlar_wall(P(m + 0.6, a[1]), P(*b), rr.uniform(0.7, 1.1), 0.6, STONE, seed + k + 10, block=0.55, course=0.28, ruin=0.4))
        else:
            W.append(ashlar_wall(P(*a), P(*b), rr.uniform(0.6, 1.1), 0.6, STONE, seed + k, block=0.55, course=0.28, ruin=0.4))
    for r in range(1, rooms):                              # cross walls between the rooms
        u = -w / 2 + w * r / rooms
        W.append(ashlar_wall(P(u, -d / 2), P(u, d / 2), rr.uniform(0.5, 0.9), 0.5, STONE, seed + 20 + r, block=0.5, course=0.28, ruin=0.5))
    return [m for m in W if len(m.f)]


def cist(x, z, yaw, seed):
    """A cist grave: four slabs set on edge round a pit, its cover gone."""
    rr = np.random.default_rng(seed)
    P = [box(0.12, 0.5, 1.9, STONE).T(k * 0.4, -0.2, 0) for k in (-1, 1)] + [box(0.9, 0.5, 0.12, STONE).T(0, -0.2, k * 0.95) for k in (-1, 1)]
    P.append(box(0.7, 0.02, 1.8, (0.2, 0.19, 0.17)).T(0, -0.15, 0))
    m = merge(P).R(0, yaw, 0).T(x, 0, z)
    return m


W = []
# the main street runs north-south, lanes cross it; houses stand in blocks along them (after the published plan, in outline)
for bx in (-52, -30, 22, 46):
    for bz in np.arange(-66, 66, 17.0):
        if rng.random() < 0.25:
            continue
        w, d = rng.uniform(8, 14), rng.uniform(7, 12)
        yaw = math.radians(rng.normal(0, 4))
        W += house(bx + rng.normal(0, 1.5), bz + rng.normal(0, 1.5), w, d, yaw + (math.pi if bx > 0 else 0), int(bx * 7 + bz * 3) % 9973, rooms=int(rng.integers(1, 4)))
for k, (x, z) in enumerate(((-8, 72), (6, -74), (70, 10))):                   # three larger buildings: courtyards with rooms round them
    W += house(x, z, 22, 16, math.radians(rng.normal(0, 3)), 500 + k, rooms=3)
    W += house(x, z, 12, 7, math.radians(rng.normal(0, 3)), 520 + k, rooms=1)
walls = merge(W)
graves = merge([cist(-78 + rng.normal(0, 3), z, rng.uniform(-0.3, 0.3), 700 + k) for k, z in enumerate(np.arange(-30, 30, 5.0))])
stones = merge([rubble(30, x, z, 6, 0.45, STONE, 800 + k) for k, (x, z) in enumerate(((-40, -20), (35, 40), (-10, 10), (55, -45), (-60, 50)))])
pots = merge([amphorae(10, x, z, 5, 900 + k, CLAY, 0.8) for k, (x, z) in enumerate(((-35, 30), (30, -20), (0, 70)))])

from placekit import dress_place  # noqa: E402,F811
out = ROOT / "models" / "item" / "Q966038" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("walls", walls, "stone"), ("graves", graves, "stone"), ("stones", stones, "stone")], [("pottery", pots, "artefact")],
            depth=-4, seed=2800, drift=0.08)
