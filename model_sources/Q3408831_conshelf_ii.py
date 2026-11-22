"""Continental Shelf Station Two (Q3408831) — what is left of Cousteau's Conshelf II at Sha'ab Rumi, Sudan, built by
code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Wikipedia, the Diving Almanac, Atlas Obscura, dive guides): in June and
July 1963 Jacques-Yves Cousteau's team lived for a month on the sea bed of the reef of Sha'ab Rumi, off Port Sudan,
in Précontinent II (Conshelf II): five oceanauts in the «Starfish House», a star of four arms round a central room,
at about 10-11 m, with a hangar for the diving saucer SP-350, a tool shed and fish pens round it; two more men lived
for a week in a «deep cabin» at about 25 m. The Starfish House and the deep cabin were taken away when the experiment
ended; the saucer hangar — a squat dome on legs, like a mushroom, with an air pocket under it — the tool shed and the
fish cages were left and are still there, now covered in coral. Modelled: the hangar's dome with its ribs, the row
of portholes round its shoulder, its lip and top hatch, the braced legs on ballast feet and the saucer's cradle
inside; the tool shed on its skids; the fish pens. The layout of the pieces and the heading (not published here:
0) are approximate. model_built.glb: the station as it was in 1963, with the Starfish House (round portholes, the
entry trunk) and the diving saucer SP-350 resting on its cradle in the hangar.

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
from shipkit import install  # noqa: E402

rng = np.random.default_rng(1963)
YELLOW = (0.78, 0.62, 0.2)
STEEL = (0.42, 0.43, 0.44)
DARK = (0.04, 0.05, 0.06)
WHITE = (0.82, 0.82, 0.8)
YELLOW_D = (0.62, 0.48, 0.15)
STEEL_D = (0.3, 0.3, 0.31)
BALLAST = (0.5, 0.49, 0.45)


install(globals(), "kit_porthole kit_window")


def ring(r0, r1, w, col, seg=32):
    return lathe([(r0, -w / 2), (r1, -w / 2), (r1, w / 2), (r0, w / 2)], col, seg=seg)


def hangar(worn):
    """The saucer hangar: a squat dome on eight legs, open below (the air pocket inside), a row of portholes round
    its shoulder, meridian ribs, a lip round its rim, a hatch on top; braced legs on ballast feet; inside, the
    cradle the saucer was winched up onto."""
    prof = [(5.2, 0.0), (5.0, 0.9), (4.2, 2.0), (2.6, 2.8), (0.001, 3.1)]
    dome = lathe(prof, YELLOW, seg=40)
    inner = lathe([(4.9, 0.05), (4.7, 0.9), (3.9, 1.9), (2.4, 2.6), (0.001, 2.9)], (0.2, 0.18, 0.14), seg=40)
    P = [dome.T(0, 2.4, 0), inner.T(0, 2.35, 0)]
    P.append(ring(5.15, 5.45, 0.22, STEEL, seg=40).T(0, 2.45, 0))                       # the lip
    for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):                           # meridian ribs
        P.append(tube([(math.cos(a) * (r + 0.04), 2.4 + y, math.sin(a) * (r + 0.04)) for r, y in prof[:-1]], 0.06, YELLOW_D, seg=4))
    for a in np.linspace(0, 2 * math.pi, 10, endpoint=False) + math.pi / 10:            # portholes round the shoulder
        r, y = 4.62, 2.4 + 1.45
        pt = kit_porthole(0, 0, 0, 0.2, "+x", rim=(0.55, 0.52, 0.45))
        pt.R(0, 0, 0.62).R(0, -a, 0).T(math.cos(a) * r, y, math.sin(a) * r)
        P.append(pt)
    P.append(cyl(0.55, 0.55, 0.35, STEEL, seg=16).T(0, 5.45, 0))                       # the top hatch
    P.append(cyl(0.62, 0.62, 0.08, STEEL, seg=16).T(0, 5.8, 0))
    legs = np.linspace(0, 2 * math.pi, 9)[:-1]
    for k, a in enumerate(legs):
        x, z = math.cos(a) * 4.6, math.sin(a) * 4.6
        P.append(cyl(0.25, 0.25, 2.55, STEEL, seg=8).T(x, 0, z))
        P.append(box(0.9, 0.45, 0.9, BALLAST).T(x, -0.05, z))                         # ballast feet
        a2 = legs[(k + 1) % 8]
        x2, z2 = math.cos(a2) * 4.6, math.sin(a2) * 4.6
        P.append(tube([(x, 0.4, z), (x2, 2.2, z2)], 0.07, STEEL, seg=5))               # cross braces
    # the saucer's cradle inside, at the height of the air pocket's floor of water
    for s in (-1, 1):
        P.append(box(0.25, 0.25, 4.2, STEEL).T(s * 1.1, 1.6, 0))
        P.append(tube([(s * 1.1, 0.0, -1.6), (s * 1.1, 1.6, -1.6)], 0.09, STEEL, seg=5))
        P.append(tube([(s * 1.1, 0.0, 1.6), (s * 1.1, 1.6, 1.6)], 0.09, STEEL, seg=5))
    m = merge(P)
    if worn:
        crumple(m, 0.06, 0.8, 5)
    return m


def saucer():
    """The diving saucer SP-350 «Denise»: a flattened yellow disc 2.85 m across, its two viewports and lights forward."""
    body = lathe([(0.001, -0.75), (0.9, -0.7), (1.35, -0.35), (1.42, 0.0), (1.3, 0.35), (0.9, 0.62), (0.4, 0.75), (0.001, 0.78)], YELLOW, seg=28)
    P = [body]
    P.append(ring(1.4, 1.48, 0.1, (0.2, 0.2, 0.2), seg=28))
    for s in (-1, 1):
        pt = kit_porthole(0, 0, 0, 0.1, "+z", rim=(0.5, 0.5, 0.5))
        P.append(pt.R(-0.3, s * 0.35, 0).T(s * 0.45, 0.05, 1.33))
        P.append(lathe([(0.001, 0), (0.09, 0.05), (0.1, 0.2)], (0.15, 0.15, 0.15), seg=8).R(math.pi / 2, 0, 0).T(s * 0.8, -0.15, 1.15))
    P.append(cyl(0.3, 0.3, 0.2, STEEL, seg=12).T(0, 0.72, -0.2))                      # its hatch
    return merge(P)


def shed(worn):
    """The tool shed: a steel hut on skids, its door in a frame, shelves of tools behind a window."""
    P = [box(2.4, 2.0, 2.0, STEEL).T(0, 0.2, 0), box(2.7, 0.15, 2.3, STEEL).T(0, 2.2, 0)]
    for s in (-1, 1):
        P.append(box(0.2, 0.25, 2.5, STEEL_D).T(s * 1.0, 0.0, 0))                       # skids
    P.append(kit_window(0.3, 1.15, 1.0, 0.8, 1.6, "+z", glass=DARK, frame=(0.3, 0.3, 0.3)))
    P.append(kit_window(1.2, 1.4, -0.3, 0.6, 0.5, "+x", glass=DARK, frame=(0.3, 0.3, 0.3)))
    for y in (0.9, 1.5):                                                                # tool racks along the wall outside
        P.append(box(0.15, 0.06, 1.6, STEEL_D).T(-1.27, y, 0))
    m = merge(P)
    if worn:
        crumple(m, 0.04, 1.0, 6)
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
        for z in (-w / 2, w / 2):
            P.append(tube([(x, 0.05, z), (x, h, z)], 0.02, STEEL, seg=4))
            P.append(tube([(z, 0.05, x), (z, h, x)], 0.02, STEEL, seg=4))
    return merge(P)


def starfish():
    """The Starfish House: a round central room and four arms on legs, round portholes, the entry trunk below."""
    P = [cyl(2.6, 2.6, 2.8, WHITE, seg=32).T(0, 1.6, 0), lathe([(2.6, 0), (1.6, 0.7), (0.001, 0.9)], WHITE, seg=32).T(0, 4.4, 0)]
    P.append(cyl(0.6, 0.6, 1.7, WHITE, seg=16).T(0, 0, 0))                              # the entry trunk, open below
    for k in range(4):
        a = k * math.pi / 2 + math.pi / 4
        arm = merge([cyl(1.2, 1.2, 6.0, WHITE, seg=20).R(0, 0, math.pi / 2).T(3.0, 2.8, 0),
                     lathe([(1.2, 0), (0.6, 0.4), (0.001, 0.5)], WHITE, seg=20).R(0, 0, -math.pi / 2).T(6.0, 2.8, 0)]
                    + [kit_porthole(x, 3.0, 1.2, 0.22, "+z") for x in (3.4, 4.6, 5.6)]
                    + [kit_porthole(x, 3.0, -1.2, 0.22, "-z") for x in (3.4, 4.6, 5.6)])
        P.append(arm.R(0, a, 0))
        P.append(cyl(0.2, 0.2, 1.8, STEEL, seg=6).T(math.cos(a) * 5.0, 0, -math.sin(a) * 5.0))
        P.append(cyl(0.2, 0.2, 1.8, STEEL, seg=6).T(math.cos(a) * 2.4, 0, -math.sin(a) * 2.4))
    for a in np.linspace(0, 2 * math.pi, 6, endpoint=False):
        P.append(kit_porthole(0, 0, 0, 0.25, "+x").R(0, -a, 0).T(math.cos(a) * 2.6, 3.2, math.sin(a) * 2.6))
    return merge(P)


H = hangar(True).T(0, 0, 0)
S = shed(True).R(0, 0.3, 0).T(14, 0, -6)
pens = merge([pen(2.2, 2.0).R(0, 0.2, 0).T(-12, 0, 8), pen(2.2, 2.0).R(0, -0.4, 0).T(10, 0, 14)])
reef = rubble(80, -10, -14, 14, 1.2, REEF_ROCK, 11, flat=0.7)

out = ROOT / "models" / "item" / "Q3408831" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_place(out, [("saucer_hangar", H, "paint"), ("tool_shed", S, "metal"), ("fish_pens", pens, "metal"), ("reef", reef, "rock")],
            [], depth=-10, seed=1963, drift=0.15)
built = merge([hangar(False), saucer().T(0, 2.45, 0), shed(False).R(0, 0.3, 0).T(14, 0, -6), pen(2.2, 2.0).R(0, 0.2, 0).T(-12, 0, 8),
               pen(2.2, 2.0).R(0, -0.4, 0).T(10, 0, 14), starfish().T(-4, 0, -22)])
dress_built(out.with_name("model_built.glb"), [("main", built)])
