"""Aqaba Underwater Military Museum (L2, own catalog entry) — the military vehicles sunk off Aqaba in 2019, built by
code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (AFP via Phys.org, DIVE Magazine, PADI, Duikersgids): in July 2019 the
Aqaba Special Economic Zone Authority and the Jordanian armed forces sank some 19-21 decommissioned military vehicles
off Aqaba's South Beach, in the Red Sea, to make an artificial reef and a dive site: tanks, an ambulance, a military
crane, a troop carrier, an anti-aircraft gun and a combat helicopter. They were laid out along the coral reef as if
in a battle formation, over some 140 m, eight of them at 15-20 m and the others at 20-28 m, a couple near 5 m. The
types, the number and the places of the vehicles drawn are approximate; the model sets them on flat ground at about
22 m. Modelled after the types: Chieftain tanks (sloped glacis, side skirts, the long turret and 120 mm gun,
searchlight box, cupola), M113 troop carriers (ramp, cupola, trim vane), army trucks (an ambulance with the Red
Crescent, cranes with turntable, telescopic boom and outriggers), towed twin anti-aircraft guns and an AH-1 Cobra.

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
from shipkit import install  # noqa: E402

rng = np.random.default_rng(2019)
SAND = (0.6, 0.52, 0.36)
OLIVE = (0.36, 0.37, 0.26)
DARK = (0.06, 0.06, 0.06)
WHITE = (0.8, 0.8, 0.78)
RED = (0.6, 0.12, 0.1)
YELLOW = (0.72, 0.56, 0.14)


install(globals(), "kit_window")
GLASS = (0.1, 0.13, 0.15)
BLACK = (0.1, 0.1, 0.1)
STEELC = (0.35, 0.35, 0.34)


def wheel(r, w, col=BLACK, seg=12, hub=None):
    m = cyl(r, r, w, col, seg=seg).R(0, 0, math.pi / 2).T(w / 2, 0, 0)
    if hub:
        m = merge([m, cyl(r * 0.45, r * 0.45, w + 0.04, hub, seg=8).R(0, 0, math.pi / 2).T(w / 2 + 0.02, 0, 0)])
    return m


def wedge(w, h, ln, slope_f, slope_r, col):
    """A hull box whose front and back faces lean in (glacis and rear plate)."""
    m = box(w, h, ln, col)
    top = m.v[:, 1] > h / 2
    m.v[top & (m.v[:, 2] > 0), 2] -= slope_f
    m.v[top & (m.v[:, 2] < 0), 2] += slope_r
    return m


def running_gear(L, W, n, r, skirt, col):
    """Tracks on road wheels: the track run, sprocket at the back, idler at the front, the side skirts over the top."""
    P = []
    for s in (-1, 1):
        x = s * (W / 2 - 0.32)
        P.append(box(0.55, 0.1, L, BLACK).T(x, 0, 0))                                   # the track on the ground
        P.append(box(0.55, 0.1, L - 0.6, BLACK).T(x, 2 * r + 0.05, 0))                  # its upper run
        for z in np.linspace(-L / 2 + 0.9, L / 2 - 1.1, n):
            P.append(wheel(r, 0.45, (0.2, 0.2, 0.19), hub=col).T(x - 0.22, r + 0.08, z))
        P.append(wheel(r * 0.85, 0.45, (0.2, 0.2, 0.19), hub=col).T(x - 0.22, r * 1.25, -L / 2 + 0.35))   # sprocket
        P.append(wheel(r * 0.8, 0.45, (0.2, 0.2, 0.19), hub=col).T(x - 0.22, r * 1.25, L / 2 - 0.3))     # idler
        if skirt:
            P.append(box(0.06, skirt, L - 0.4, col).T(s * (W / 2 + 0.02), 2 * r + 0.15 - skirt * 0.6, 0))
    return P


def tank(col):
    """A Chieftain: low hull with a sharply sloped glacis, side skirts, the long cast turret with its 120 mm gun
    (thermal sleeve, fume extractor), commander's cupola, the searchlight box on the left of the turret, stowage
    bins behind it, engine deck grilles."""
    P = running_gear(7.5, 3.5, 6, 0.36, 0.55, col)
    hull = wedge(3.3, 1.05, 7.6, 1.5, 0.4, col).T(0, 0.62, 0)
    P.append(hull)
    for z in (-2.9, -2.2):                                                           # engine deck grilles
        P.append(box(2.4, 0.04, 0.5, (0.16, 0.16, 0.15)).T(0, 1.67, z))
    tur = lathe([(1.55, 0.0), (1.5, 0.45), (1.2, 0.8), (0.001, 0.85)], col, seg=16)
    tur.v[:, 2] *= 1.45
    tur.v[tur.v[:, 2] > 0.3, 1] *= 0.75                                              # the long sloped front
    P.append(tur.T(0, 1.67, -0.3))
    P.append(box(2.4, 0.55, 0.9, col).T(0, 1.72, -2.25))                            # bins at the back of the turret
    P.append(cyl(0.38, 0.38, 0.3, col, seg=12).T(0.6, 2.4, -0.7))                  # commander's cupola
    P.append(cyl(0.42, 0.42, 0.06, BLACK, seg=12).T(0.6, 2.7, -0.7))
    P.append(box(0.55, 0.6, 0.6, col).T(-1.65, 1.95, 0.4))                         # the searchlight box
    P.append(kit_window(-1.93, 2.25, 0.4, 0.45, 0.45, "-x", glass=GLASS, frame=(0.25, 0.25, 0.24)))
    gun = [cyl(0.22, 0.2, 1.0, col, seg=10), cyl(0.12, 0.11, 6.0, col, seg=10).T(0, 1.0, 0),
           cyl(0.2, 0.2, 0.8, col, seg=10).T(0, 3.2, 0), cyl(0.15, 0.13, 0.5, col, seg=10).T(0, 6.6, 0)]
    P.append(merge(gun).R(math.pi / 2 - 0.04, 0, 0).T(0, 2.05, 1.6))
    for s in (-1, 1):                                                                # headlights and towing eyes
        P.append(cyl(0.1, 0.1, 0.12, GLASS, seg=8).R(math.pi / 2, 0, 0).T(s * 1.3, 1.15, 3.35))
    return merge(P)


def apc(col):
    """An M113 troop carrier: the aluminium box with its sloped nose, five road wheels a side, the rear ramp, the
    commander's cupola and its machine gun, the trim vane folded on the nose."""
    P = running_gear(4.9, 2.7, 5, 0.33, 0.0, col)
    body = wedge(2.68, 1.55, 4.9, 1.1, 0.0, col).T(0, 0.45, 0)
    P.append(body)
    P.append(box(2.2, 0.08, 0.7, col).R(0.9, 0, 0).T(0, 1.2, 2.55))                # trim vane
    P.append(box(1.9, 1.3, 0.08, (0.3, 0.3, 0.26)).T(0, 0.55, -2.49))              # the ramp
    P.append(cyl(0.4, 0.4, 0.35, col, seg=12).T(0.55, 2.0, 0.2))                    # cupola
    P.append(cyl(0.04, 0.04, 1.4, BLACK, seg=5).R(math.pi / 2, 0, 0).T(0.55, 2.3, 0.4))
    P.append(box(0.6, 0.06, 0.6, col).T(-0.6, 2.0, -1.2))                           # the hatch
    for s in (-1, 1):
        P.append(cyl(0.08, 0.08, 0.1, GLASS, seg=8).R(math.pi / 2, 0, 0).T(s * 1.05, 1.35, 1.8))
    return merge(P)


def red_crescent(r, col=RED, seg=20):
    """The Red Crescent (a Jordanian army ambulance): a crescent opening to the right, in the xy plane facing +z."""
    a = np.linspace(-math.pi / 2, math.pi / 2, seg)
    outer = np.stack([-np.cos(a) * r, np.sin(a) * r], 1)
    c2, r2 = r * 0.35, r * 0.8
    yi = np.sin(a) * r * 0.98
    xi = -np.sqrt(np.maximum(r2 * r2 - yi * yi, 0)) + c2
    inner = np.stack([xi, yi], 1)
    v = [(x, y, 0.0) for x, y in outer] + [(x, y, 0.0) for x, y in inner]
    f = []
    for i in range(seg - 1):
        f += [(i, i + 1, seg + i + 1), (i, seg + i + 1, seg + i)]
    return Mesh(v, f, col)


def truck(col, body):
    """A three-axle army truck: the cab with its bonnet and windscreen, bumper and mudguards, six wheels; the body
    behind it: an ambulance box with the Red Crescent, or a crane with its turntable, cab, telescopic boom laid along
    the truck and the outriggers."""
    P = [wheel(0.55, 0.42, BLACK, hub=(0.3, 0.3, 0.28)).T(s * 1.05 - (0.42 if s > 0 else 0), 0.55, z)
         for s in (-1, 1) for z in (-2.4, -1.25, 2.2)]
    P += [box(2.1, 0.35, 6.6, (0.18, 0.18, 0.18)).T(0, 0.75, 0)]                   # chassis
    cab = box(2.4, 1.5, 1.7, col).T(0, 1.15, 1.75)
    P.append(cab)
    P.append(box(2.0, 0.8, 1.2, col).T(0, 1.1, 3.1))                                 # the bonnet
    P.append(box(2.5, 0.25, 0.2, BLACK).T(0, 0.85, 3.8))                              # bumper
    P.append(kit_window(0, 2.15, 2.6, 2.0, 0.6, "+z", glass=GLASS, frame=col))
    for s in (-1, 1):
        P.append(kit_window(s * 1.2, 2.15, 1.75, 0.8, 0.55, "+x" if s > 0 else "-x", glass=GLASS, frame=col))
        P.append(box(0.5, 0.06, 1.0, col).T(s * 1.05, 1.15, 2.2))                    # mudguards
        P.append(cyl(0.12, 0.12, 0.1, GLASS, seg=8).R(math.pi / 2, 0, 0).T(s * 0.8, 1.6, 3.7))
    if body == "box":
        P.append(box(2.4, 2.1, 3.9, WHITE).T(0, 0.95, -1.1))
        for s in (-1, 1):
            cr = red_crescent(0.5)
            P.append(cr.R(0, s * math.pi / 2, 0).T(s * 1.2, 2.0, -1.1))
        P.append(red_crescent(0.45).R(0, math.pi, 0).T(0, 2.0, -3.05))
        P.append(kit_window(0, 1.8, -3.05, 1.6, 1.5, "-z", glass=(0.6, 0.6, 0.58), frame=(0.5, 0.5, 0.48)))
    else:
        P.append(cyl(0.9, 0.9, 0.3, col, seg=14).T(0, 1.0, -1.6))                      # turntable
        P.append(box(1.5, 1.2, 1.6, col).T(0.3, 1.3, -1.6))                            # crane cab
        P.append(kit_window(0.3, 2.05, -0.8, 1.1, 0.5, "+z", glass=GLASS, frame=col))
        boom = merge([box(0.7, 0.6, 7.6, YELLOW, bottom=False), box(0.5, 0.42, 3.0, YELLOW, bottom=False).T(0, 0, 4.8)])
        P.append(boom.R(-0.1, 0, 0).T(-0.55, 2.2, 1.9))
        P.append(box(0.12, 1.2, 0.12, BLACK).T(-0.55, 2.3, 7.8))                       # the hook block
        for s in (-1, 1):
            for z in (-2.9, 0.9):
                P.append(box(1.2, 0.18, 0.25, col).T(s * 1.55, 0.8, z))                  # outriggers
                P.append(box(0.4, 0.8, 0.4, col).T(s * 2.05, 0.05, z))
    return merge(P)


def aa_gun(col):
    """A towed twin anti-aircraft gun: the carriage with its two wheels and folding outriggers, the mount, the
    shield, the two long barrels raised, the gunners' seats."""
    P = [box(1.6, 0.4, 3.2, col).T(0, 0.55, 0)]
    for s in (-1, 1):
        P.append(wheel(0.45, 0.3, BLACK, hub=col).T(s * 0.95 - (0.3 if s > 0 else 0), 0.45, -0.2))
        P.append(box(0.25, 0.2, 2.0, col).R(0, s * 0.9, 0).T(s * 1.2, 0.0, 0.6))       # outriggers
        P.append(box(0.4, 0.06, 0.4, col).T(s * 2.0, 0.0, 1.3))
        P.append(box(0.45, 0.08, 0.45, BLACK).T(s * 0.55, 1.75, -0.5))                  # seats
    P.append(cyl(0.6, 0.6, 0.4, col, seg=14).T(0, 0.95, 0))
    P.append(box(1.3, 0.8, 1.0, col).T(0, 1.3, 0.1))
    P.append(box(1.6, 0.9, 0.06, col).R(-0.2, 0, 0).T(0, 1.3, 0.75))                 # shield
    for s in (-1, 1):
        P.append(cyl(0.06, 0.05, 3.6, BLACK, seg=6).R(math.pi / 2 - 0.5, 0, 0).T(s * 0.25, 1.75, 0.5))
        P.append(cyl(0.1, 0.1, 0.4, BLACK, seg=6).R(math.pi / 2 - 0.5, 0, 0).T(s * 0.25, 1.75, 0.5))
    P.append(box(0.3, 0.2, 2.2, col).T(0, 0.45, -2.5))                               # the towing bar
    return merge(P)


def helicopter(col):
    """An AH-1 Cobra: the narrow fuselage with the stepped tandem canopy, the chin turret and its gun, stub wings with
    rocket pods, the engine and exhaust, the tail boom with its fin and tail rotor, the two-blade main rotor on its
    mast, skids on their cross tubes."""
    f = lathe([(0.001, -3.0), (0.45, -2.6), (0.62, -1.2), (0.66, 0.4), (0.55, 1.8), (0.35, 2.9), (0.001, 3.4)], col, seg=14)
    f.v[:, 0] *= 0.62
    P = [f.R(math.pi / 2, 0, 0).T(0, 1.55, 0)]
    for z0, h in ((1.9, 0.55), (0.7, 0.75)):                                         # gunner forward, pilot higher behind
        can = lathe([(0.001, -0.7), (0.32, -0.4), (0.36, 0.2), (0.25, 0.6), (0.001, 0.75)], GLASS, seg=10).R(math.pi / 2, 0, 0)
        can.v[:, 1] = np.maximum(can.v[:, 1], 0) * (h / 0.36)
        P.append(can.T(0, 1.95, z0))
    P.append(cyl(0.28, 0.25, 0.3, col, seg=10).T(0, 1.0, 2.6))                       # chin turret
    P.append(cyl(0.04, 0.04, 1.2, BLACK, seg=5).R(math.pi / 2, 0, 0).T(0, 1.05, 2.7))
    P.append(box(0.7, 0.45, 1.8, col).T(0, 2.05, -0.6))                              # engine
    P.append(cyl(0.18, 0.2, 0.5, BLACK, seg=8).R(-math.pi / 2, 0, 0).T(0, 2.25, -1.4))
    for s in (-1, 1):                                                                 # stub wings and rocket pods
        P.append(box(1.4, 0.1, 0.8, col).T(s * 0.95, 1.55, 0.0))
        P.append(cyl(0.2, 0.2, 1.5, (0.3, 0.32, 0.28), seg=8).R(math.pi / 2, 0, 0).T(s * 1.4, 1.3, -0.7))
    P.append(cyl(0.3, 0.12, 6.0, col, seg=10).R(-math.pi / 2, 0, 0).T(0, 1.7, -2.6))
    P.append(box(0.12, 1.5, 1.0, col).R(-0.25, 0, 0).T(0, 1.8, -8.4))                 # fin
    P.append(box(1.4, 0.08, 0.5, col).T(0, 1.75, -7.3))                              # tail planes
    P.append(cyl(0.08, 0.08, 0.2, col, seg=6).R(0, 0, math.pi / 2).T(0.1, 2.6, -8.5))
    for a in (0.0, math.pi):
        P.append(box(0.08, 0.04, 1.4, (0.15, 0.15, 0.15)).R(a, 0, 0).T(0.2, 2.6, -8.5))  # tail rotor blades
    P.append(cyl(0.12, 0.12, 0.55, BLACK, seg=8).T(0, 2.3, -0.2))                    # rotor mast
    P.append(box(0.45, 0.05, 13.4, (0.15, 0.15, 0.15)).R(0, 0.5, 0).T(0, 2.85, -0.2))  # the two-blade rotor
    for s in (-1, 1):                                                                 # skids on cross tubes
        P.append(tube([(s * 0.95, 0.06, -1.4), (s * 0.95, 0.06, 1.6), (s * 0.95, 0.25, 2.0)], 0.06, BLACK, seg=5))
        for z in (-0.9, 0.9):
            P.append(tube([(s * 0.95, 0.06, z), (s * 0.4, 1.1, z)], 0.05, BLACK, seg=5))
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
