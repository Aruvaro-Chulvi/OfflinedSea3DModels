"""Continental Shelf Station Two (Q3408831) — what is left of Cousteau's Conshelf II at Sha'ab Rumi, Sudan, built by
code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the Diving Almanac, Atlas Obscura, dive guides): in June and
July 1963 Jacques-Yves Cousteau's team lived for a month on the sea bed of the reef of Sha'ab Rumi, off Port Sudan,
in Précontinent II (Conshelf II): five oceanauts in the «Starfish House», a star of four arms round a central room,
at about 10-11 m, with a hangar for the diving saucer SP-350, a tool shed and fish pens round it; two more men lived
for a week in a «deep cabin» at about 25 m. The Starfish House and the deep cabin were taken away when the experiment
ended; the saucer hangar — a squat dome on legs, like a mushroom, with an air pocket under it — the tool shed and the
fish cages were left and are still there, now covered in coral. The layout of the pieces and the heading (not
published here: 0) are approximate. model_built.glb: the station as it was in 1963.

    python model_sources/Q3408831_conshelf_ii.py   -> models/item/Q3408831/model.glb (+ model_built.glb)
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
from dresskit import dress_built  # noqa: E402

rng = np.random.default_rng(1963)
YELLOW = (0.78, 0.62, 0.2)
STEEL = (0.42, 0.43, 0.44)
DARK = (0.04, 0.05, 0.06)
WHITE = (0.82, 0.82, 0.8)


def hangar(worn):
    """The saucer hangar: a domed shell on legs, open below, the air pocket inside."""
    dome = lathe([(5.2, 0.0), (5.0, 0.9), (4.2, 2.0), (2.6, 2.8), (0.001, 3.1)], YELLOW, seg=28)
    inner = lathe([(4.9, 0.05), (4.7, 0.9), (3.9, 1.9), (2.4, 2.6), (0.001, 2.9)], (0.2, 0.18, 0.14), seg=28)
    P = [dome.T(0, 2.4, 0), inner.T(0, 2.35, 0)]
    P += [cyl(0.25, 0.25, 2.5, STEEL, seg=8).T(math.cos(a) * 4.6, 0, math.sin(a) * 4.6) for a in np.linspace(0, 6.28, 9)[:-1]]
    m = merge(P)
    if worn:
        crumple(m, 0.08, 0.8, 5)
    return m


def shed(worn):
    m = merge([box(2.4, 2.0, 2.0, STEEL), box(2.6, 0.15, 2.2, STEEL).T(0, 2.0, 0), quad(0, 1.0, 1.01, 0.8, 1.6, "+z", DARK)])
    if worn:
        crumple(m, 0.05, 1.0, 6)
    return m


def pen(w, h):
    """A fish pen: a cage of bars."""
    P = []
    for x in (-w / 2, w / 2):
        for z in (-w / 2, w / 2):
            P.append(cyl(0.05, 0.05, h, STEEL, seg=5).T(x, 0, z))
    for y in (0.05, h):
        for a, b in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            P.append(tube([(a[0] * w / 2, y, a[1] * w / 2), (b[0] * w / 2, y, b[1] * w / 2)], 0.04, STEEL, seg=4))
    for k in range(1, 6):
        x = -w / 2 + w * k / 6
        P.append(tube([(x, 0.05, -w / 2), (x, h, -w / 2)], 0.02, STEEL, seg=4))
        P.append(tube([(x, 0.05, w / 2), (x, h, w / 2)], 0.02, STEEL, seg=4))
    return merge(P)


def starfish():
    """The Starfish House: a round central room and four arms on legs."""
    P = [cyl(2.6, 2.6, 2.8, WHITE, seg=24).T(0, 1.6, 0), lathe([(2.6, 0), (1.6, 0.7), (0.001, 0.9)], WHITE, seg=24).T(0, 4.4, 0)]
    for k in range(4):
        a = k * math.pi / 2 + math.pi / 4
        arm = merge([cyl(1.2, 1.2, 6.0, WHITE, seg=16).R(0, 0, math.pi / 2).T(3.0, 2.8, 0),
                     lathe([(1.2, 0), (0.6, 0.4), (0.001, 0.5)], WHITE, seg=16).R(0, 0, -math.pi / 2).T(6.0, 2.8, 0)]
                    + [quad(x, 3.0, 1.22, 0.5, 0.5, "+z", DARK) for x in (2.0, 3.6, 5.0)])
        P.append(arm.R(0, a, 0))
        P.append(cyl(0.2, 0.2, 1.8, STEEL, seg=6).R(0, a, 0).T(math.cos(a) * 5.0, 0, -math.sin(a) * 5.0))
    return merge(P)


H = hangar(True).T(0, 0, 0)
S = shed(True).R(0, 0.3, 0).T(14, 0, -6)
pens = merge([pen(2.2, 2.0).R(0, 0.2, 0).T(-12, 0, 8), pen(2.2, 2.0).R(0, -0.4, 0).T(10, 0, 14)])
reef = rubble(80, -10, -14, 14, 1.2, REEF_ROCK, 11, flat=0.7)

out = ROOT / "models" / "item" / "Q3408831" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("saucer_hangar", H, "paint"), ("tool_shed", S, "metal"), ("fish_pens", pens, "metal"), ("reef", reef, "rock")],
            [], depth=-10, seed=1963, drift=0.15)
built = merge([hangar(False), shed(False).R(0, 0.3, 0).T(14, 0, -6), pen(2.2, 2.0).R(0, 0.2, 0).T(-12, 0, 8),
               pen(2.2, 2.0).R(0, -0.4, 0).T(10, 0, 14), starfish().T(-4, 0, -22)])
dress_built(out.with_name("model_built.glb"), [("main", built)])
