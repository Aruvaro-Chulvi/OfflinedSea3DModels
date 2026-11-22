"""
B-17 «Black Jack» (L1, an own catalog entry: no Wikipedia article) — the Flying Fortress off Cape Vogel, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The aircraft: Boeing B-17F Flying Fortress 41-24521 of the 63rd Bomb Squadron, 43rd Bomb Group, US Army Air
Forces, one of the first B-17Fs built at Seattle (1942); named Black Jack after the last two digits of its serial
(21, a blackjack). Four Wright Cyclone radial engines, the glazed nose, the top turret behind the cockpit, the
ball turret under the fuselage, the waist guns, the tail guns under the great dorsal fin; olive drab above, neutral
grey below, the US star on the wings and fuselage, the serial in yellow on the fin. About 22.7 m long, 31.6 m span.
On the night of 10–11 July 1943, coming back from a raid on Rabaul with two engines failing and nearly out of
fuel, the crew ditched off Boga Boga at Cape Vogel (Papua New Guinea); all of them got ashore. The aircraft sank
just beyond the reef. Divers found it in December 1986: it sits upright on the sand at about 45–50 m, almost whole,
the nose badly crumpled, the propellers of all four engines twisted, the machine guns still in their turrets, the
twin tail guns free in their mount, the cockpit windows open.
The heading is not published; the depth and the position are approximate.

    python model_sources/L1_black_jack.py      -> models/item/L1/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, fbm, sstep  # noqa: E402
from airkit import Plane, Engine, standing  # noqa: E402
from sailkit import lettering, FONT5x7  # noqa: E402

OD, GREY = (0.29, 0.29, 0.2), (0.52, 0.53, 0.52)          # olive drab, neutral grey
DARK, GLASS, YELLOW = (0.1, 0.1, 0.1), (0.2, 0.26, 0.3), (0.85, 0.7, 0.15)
BLUE, WHITE = (0.1, 0.14, 0.32), (0.88, 0.88, 0.86)
FONT5x7.update({
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
})


def plane(gear=True):
    return Plane(length=22.7, span=31.6, fus_w=2.35, fus_h=2.75, fus_y=2.9, wing_z=-7.0, root_chord=6.1, tip_chord=2.3,
                 wing_y=2.15, dihedral=0.08, sweep=2.6, thick=0.18, tail_span=13.2, tail_chord=3.0, fin_h=3.4, fin_chord=5.0,
                 engines=[Engine(s * x, 2.2 + (0.05 if x > 5 else 0.0), -5.6 - (0.6 if x > 5 else 0.0), r=0.72, prop=1.75, nacelle=4.2 if x < 5 else 3.0)
                          for s in (-1, 1) for x in (3.6, 7.4)],
                 gear=(3.6, -7.6, 0.66, 1.3, False) if gear else None, corrugated=False, nose=0.35, upsweep=0.35,
                 windows=(), cockpit=(-3.9, -3.0), skin=OD, dark=DARK, glass=GLASS)


def two_tone(m, cy):
    """Olive drab above, grey below, with a soft line between."""
    k = sstep(cy - 0.25, cy + 0.15, m.v[:, 1])
    m.c = np.array(GREY)[None] * (1 - k[:, None]) + np.array(OD)[None] * k[:, None]
    return m


def star(r, nrm):
    """The US national insignia: a white five-pointed star in a blue disc, facing nrm ('+y', '-y', '+x', '-x')."""
    disc = cyl(r, r, 0.01, BLUE, seg=24)
    v = [(0, 0.012, 0)]
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        rr = r * 0.95 if k % 2 == 0 else r * 0.37
        v.append((rr * math.cos(a), 0.012, -rr * math.sin(a)))
    st = Mesh(v, [(0, 1 + k, 1 + (k + 1) % 10) for k in range(10)], WHITE)
    m = merge([disc, st])
    if nrm == "-y":
        m.R(math.pi, 0, 0)
    elif nrm in ("+x", "-x"):
        m.R(0, 0, -math.pi / 2 if nrm == "+x" else math.pi / 2).R(math.pi / 2, 0, 0)    # facing sideways, a point up
    return m


def details(P, built=True):
    """Glazed nose, turrets, guns, the big dorsal fin, waist windows."""
    out = []
    z_nose = 0.0
    hw, hh, cy = P.section(0.97)
    nose = lathe([(0.001, 0.0), (0.55, 0.25), (0.85, 0.9), (0.98, 1.6)], GLASS, seg=18).R(-math.pi / 2, 0, 0).T(0, cy, z_nose + 0.05)
    out.append(nose)
    # top turret behind the cockpit, ball turret under the fuselage, tail guns
    u_t = (-4.6 - P.zf(0)) / P.length; hw, hh, cy = P.section(u_t)
    dome = lathe([(0.75, 0.0), (0.72, 0.25), (0.55, 0.6), (0.001, 0.78)], GLASS, seg=16).T(0, cy + hh - 0.08, -4.6)
    out += [dome] + [tube([(dx, cy + hh + 0.45, -4.6), (dx, cy + hh + 0.5, -3.3)], 0.04, DARK, seg=4) for dx in (-0.15, 0.15)]
    u_b = (-11.0 - P.zf(0)) / P.length; hw, hh, cy = P.section(u_b)
    ball = lathe([(0.001, -0.6), (0.45, -0.45), (0.6, 0.0), (0.5, 0.3), (0.001, 0.35)], GREY, seg=16).T(0, cy - hh + 0.05, -11.0)
    out += [ball] + [tube([(dx, cy - hh - 0.25, -11.0), (dx, cy - hh - 0.55, -9.9)], 0.04, DARK, seg=4) for dx in (-0.12, 0.12)]
    hw, hh, cy = P.section(0.015)
    out.append(box(0.9, 1.0, 1.2, OD).T(0, cy - 0.5, P.zf(0) + 0.5))
    out += [tube([(dx, cy - 0.1, P.zf(0) + 0.1), (dx, cy - 0.05, P.zf(0) - 1.0)], 0.04, DARK, seg=4) for dx in (-0.12, 0.12)]
    # the dorsal fin: long and low from the middle of the fuselage, rising into the tall fin
    hw, hh, cy = P.section(0.12)
    fin = P.surface(0.0, 4.2, P.zf(0) + 6.8, cy + hh * 0.6, 6.8, 1.9, 0.09, 0.08, 4.9, 0.0, OD, vertical=True, ns=10)
    out.append(fin)
    # waist gun windows and the radio-room hatch
    u_w = (-13.2 - P.zf(0)) / P.length; hw, hh, cy = P.section(u_w)
    for s in (-1, 1):
        out.append(box(0.05, 0.85, 1.4, (0.05, 0.05, 0.06), bottom=False).T(s * (hw + 0.01), cy + 0.35, -13.2))
        out.append(tube([(s * (hw + 0.05), cy + 0.35, -13.2), (s * (hw + 0.95), cy + 0.3, -13.0)], 0.04, DARK, seg=4))
    # markings: stars on the left wing top, the right wing bottom, both sides of the fuselage; the serial on the fin
    out.append(star(1.25, "+y").T(11.0, 2.15 + 11.0 * math.tan(P.dihedral) + 0.62, -9.4))
    out.append(star(1.25, "-y").T(-11.0, 2.15 + 11.0 * math.tan(P.dihedral) - 0.25, -9.4))
    u_s = (-15.5 - P.zf(0)) / P.length; hw, hh, cy = P.section(u_s)
    for s in (-1, 1):
        out.append(star(0.75, "+x" if s > 0 else "-x").T(s * (hw + 0.02), cy, -15.5))
        ser = lettering("124521", 0.42, YELLOW, depth=0.015)
        ser.R(0, -s * math.pi / 2, 0).T(s * 0.1, P.section(0.04)[2] + 2.0, P.zf(0) + 2.4)
        out.append(ser)
    return out


# ── Black Jack as she flew (model_built.glb) ─────────────────────────────────
P = plane(gear=True)
B = P.build()
for k in ("fuselage", "wing_l", "wing_r", "tail"):
    two_tone(B[k], P.fus_y if k in ("fuselage", "tail") else P.wing_y + 0.1)
for k in [k for k in B if k.startswith("engine_")]:
    two_tone(B[k], 2.2)
B["details"] = merge(details(P))
B = standing(B, -7.6)

# ── The wreck ────────────────────────────────────────────────────────────────
W = plane(gear=False).build()
for k in ("fuselage", "wing_l", "wing_r", "tail"):
    two_tone(W[k], P.fus_y if k in ("fuselage", "tail") else P.wing_y + 0.1)
W["details"] = merge(details(P))
# the propellers twisted: every blade bent back from the hub as they hit the water
for k in [k for k in W if k.startswith("engine_")]:
    e = W[k]
    two_tone(e, 2.2)
    cen = np.array([e.v[:, 0].mean(), 2.2])
    d = np.hypot(e.v[:, 0] - cen[0], e.v[:, 1] - cen[1])
    blade = (e.v[:, 2] > e.v[:, 2].max() - 0.35) & (d > 0.45)
    e.v[blade, 2] -= (d[blade] - 0.45) ** 1.3 * 0.55
    e.v[blade, 0] += np.sign(e.v[blade, 0] - cen[0]) * (d[blade] - 0.45) * 0.15
airframe = merge([W[k] for k in W])
# the nose badly crumpled: the glazing gone, the front pushed in and down
dz = -airframe.v[:, 2]
k_n = sstep(3.2, 0.0, dz)
airframe.v[:, 2] -= k_n * 1.1
airframe.v[:, 1] -= k_n * k_n * 0.45
crumple(airframe, 0.35, 0.9, 7, k_n)
fc = airframe.v[airframe.f].mean(1)
glass_gone = (np.abs(airframe.c[airframe.f].mean(1) - np.array(GLASS)).sum(1) < 0.05) & (fc[:, 2] > -1.8)
airframe.f = airframe.f[~glass_gone]
crumple(airframe, 0.05, 1.5, 8)
crumple(airframe, 0.12, 0.25, 9, sstep(-12, -22, airframe.v[:, 2]))           # the tail section a little twisted
# 83 years in warm water: the paint faded, a skin of growth
airframe.c = np.clip(airframe.c * 0.8 + np.array([0.05, 0.05, 0.04]), 0, 1)
# on the sand, wheels up, settled a little into it, a slight list to port
airframe.R(0, 0, math.radians(3)).R(math.radians(1.5), 0, 0)
airframe.T(0, -(airframe.v[:, 1].min()) - 0.45, 0)
# beside her: the two waist guns thrown out before the ditching are not here (jettisoned at sea); a few panels
deb = []
rng = np.random.default_rng(1943)
for i in range(10):
    m = box(rng.uniform(0.4, 1.5), 0.02, rng.uniform(0.3, 1.0), OD if i % 2 else GREY)
    crumple(m, 0.1, 1.0, 50 + i)
    deb.append(m.R(rng.uniform(-0.2, 0.2), rng.uniform(0, 6.28), 0).T(rng.normal(0, 4), 0.02, rng.uniform(-2, 6)))
debris = merge(deb)

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "L1" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("main", airframe), ("debris_field", debris)]}], depth=-48, kind="aircraft", seed=1943,
            wood="alu", life="tropical")
dress_built(out.with_name("model_built.glb"),
            [("airframe", merge([B[k] for k in B if not k.startswith("engine_") and k != "details"]), "paint"),
             ("engines", merge([B[k] for k in B if k.startswith("engine_")]), "paint"),
             ("details", B["details"], "paint")])
