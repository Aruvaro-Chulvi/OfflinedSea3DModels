"""
Kaleva (Q1464299) — the Junkers Ju 52/3m of Aero O/Y shot down off Keri in 1940, and its remains, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The aircraft: a Junkers Ju 52/3m airliner of the Finnish company Aero O/Y, registered OH-ALL and named Kaleva;
three BMW radial engines (the nose one in a ring cowling, the two on the wings in full cowlings), corrugated
duralumin skin, a low wing with the Junkers «double wing» (the flaps and ailerons hung below and behind the
trailing edge), fixed landing gear with spats, a tail wheel; about 18.9 m long, 29.25 m in span.
On 14 June 1940, flying from Tallinn to Helsinki, she was shot down over the Gulf of Finland by two Soviet
bombers, days before the Soviet occupation of Estonia; the nine people on board died. Her remains were found in
June 2024 by the Estonian firm Tuukritoode north of the island of Keri, at 71–76 m: the aircraft is in pieces,
badly crushed by the impact at full speed, on a rocky bottom of ridges and hollows; the recognisable parts are
the landing gear with its wheel, one of the engines and parts of the wings.
The places of the pieces are approximate (only these parts have been described).

    python model_sources/Q1464299_kaleva.py      -> models/item/Q1464299/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, crumple, fbm, sstep  # noqa: E402
from airkit import Plane, Engine, standing  # noqa: E402
from sailkit import lettering, solidify  # noqa: E402
import abyss  # noqa: E402

rng = np.random.default_rng(1940)
SKIN = (0.74, 0.74, 0.72)                 # natural duralumin
DARK = (0.12, 0.12, 0.13)
LETTER = (0.08, 0.08, 0.09)


def plane():
    return Plane(length=18.9, span=29.25, fus_w=1.9, fus_h=2.4, fus_y=2.5, wing_z=-5.2, root_chord=5.4, tip_chord=2.1,
                 wing_y=1.35, dihedral=0.06, sweep=1.4, thick=0.17, tail_span=8.4, tail_chord=1.9, fin_h=2.6, fin_chord=2.4,
                 engines=[Engine(0, 2.55, 0.2, r=0.72, prop=1.45), Engine(4.6, 1.75, -3.6, r=0.62, prop=1.4, nacelle=2.6),
                          Engine(-4.6, 1.75, -3.6, r=0.62, prop=1.4, nacelle=2.6)],
                 gear=(2.55, -5.6, 0.62, 1.2, True), corrugated=True, nose=0.15, upsweep=0.5,
                 windows=(-5.5, -11.5, 7, 3.0, 0.5), cockpit=(-3.6, -2.4), skin=SKIN, dark=DARK)


P = plane()


def double_wing(s):
    """The Junkers «double wing»: flaps and ailerons as a separate narrow wing below and behind the trailing edge."""
    hw = P.fus_w / 2
    x0, x1 = s * (hw + 0.3), s * (P.span / 2 - 0.4)
    z_te0 = P.wing_z - P.root_chord
    z_te1 = P.wing_z - P.sweep - P.tip_chord
    return P.surface(x0, x1, z_te0 - 0.15, P.wing_y - 0.3, 1.05, 0.7, 0.12, 0.1, z_te0 - z_te1, P.dihedral, SKIN, ribs=True, ns=12)


def markings():
    """OH-ALL on both sides of the fuselage and across the wings; KALEVA by the cockpit."""
    out = []
    for s in (-1, 1):
        u = 0.25
        hw, hh, cy = P.section(u)
        reg = lettering("OH-ALL", 0.75, LETTER, depth=0.02)
        reg.R(0, -s * math.pi / 2, 0).T(s * (hw + 0.03), cy - 0.1, P.zf(u))
        out.append(reg)
        nm = lettering("KALEVA", 0.28, LETTER, depth=0.02)
        u2 = 0.82
        hw2, hh2, cy2 = P.section(u2)
        nm.R(0, -s * math.pi / 2, 0).T(s * (hw2 + 0.03), cy2 + 0.35, P.zf(u2))
        out.append(nm)
    wings = P.wings()
    for s, up in ((1, True), (-1, False)):                           # on the top of the left wing, under the right
        w = lettering("OH-ALL", 1.4, LETTER, depth=0.02)
        w.R(math.pi / 2 if up else -math.pi / 2, 0, 0)
        wv = wings["wing_l" if s > 0 else "wing_r"].v
        x, z = s * 9.0, P.wing_z - 0.25 * P.sweep - 1.5
        near = wv[(np.abs(wv[:, 0] - x) < 1.2) & (np.abs(wv[:, 2] - z) < 0.8)]
        y = near[:, 1].max() + 0.015 if up else near[:, 1].min() - 0.015
        w.T(x, y, z)
        out.append(w)
    return merge(out)


# the letters: the kit's 5×7 alphabet has no hyphen and no K, V… add them
from sailkit import FONT5x7  # noqa: E402
FONT5x7.update({
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
})


# ── Kaleva as she flew (model_built.glb) ─────────────────────────────────────
B = P.build()
B["flaps"] = merge([double_wing(1), double_wing(-1)])
B["markings"] = markings()
B = standing(B, -5.6)

# ── The remains ──────────────────────────────────────────────────────────────
W = P.build()
W["flaps"] = merge([double_wing(1), double_wing(-1)])


def wreck_piece(m, amp, seed, flat=0.0):
    m = m.copy()
    crumple(m, amp, 0.6, seed)
    crumple(m, amp * 0.3, 2.5, seed + 1)
    if flat:
        c = m.v.mean(0)
        m.v[:, 1] = c[1] + (m.v[:, 1] - c[1]) * (1 - flat)
    return m


def place(m, x, z, yaw, roll=0.0, pitch=0.0, sink=0.25):
    c = m.v.mean(0)
    m.T(-c[0], -c[1], -c[2]).R(pitch, 0, roll).R(0, yaw, 0)
    m.T(x, -m.v[:, 1].min() - sink, z)
    return m


pieces = []
# the left wing outboard of the engine, torn off, crumpled and lying upside down
wl = W["wing_l"]; keep = wl.v[wl.f].mean(1)[:, 0] > 5.6
wing = Mesh(wl.v, wl.f[keep], wl.c)
wing = wreck_piece(solidify(wing, 0.02), 0.35, 11)
pieces.append(place(wing, 6.0, 4.0, 0.7, roll=math.pi - 0.15, pitch=0.05, sink=0.2))
# a piece of the right wing, its root torn, with the flap hanging from it
wr = W["wing_r"]; keep = (wr.v[wr.f].mean(1)[:, 0] < -2.0) & (wr.v[wr.f].mean(1)[:, 0] > -9.5)
fl = W["flaps"]; kf = (fl.v[fl.f].mean(1)[:, 0] < -2.0) & (fl.v[fl.f].mean(1)[:, 0] > -7.5)
wing2 = merge([Mesh(wr.v, wr.f[keep], wr.c), Mesh(fl.v, fl.f[kf], fl.c)])
wing2 = wreck_piece(wing2, 0.5, 21, flat=0.3)
pieces.append(place(wing2, -7.0, -3.0, -0.4, roll=0.25, sink=0.3))
# the engine the divers filmed: the left wing engine with what is left of its nacelle, propeller blades bent
eng = W["engine_1"].copy()
v = eng.v; front = v[:, 2] > -3.6 + 0.05
eng.v[front & (np.abs(v[:, 0] - 4.6) > 0.5), 2] -= 0.25 * np.abs(v[front & (np.abs(v[:, 0] - 4.6) > 0.5), 0] - 4.6)   # blades bent back
eng = wreck_piece(eng, 0.12, 31)
pieces.append(place(eng, -1.5, 7.5, 2.3, roll=1.2, sink=0.35))
# the landing gear leg with its wheel (the spat broken away)
gear = wreck_piece(P.gear_leg(2.55, -5.6, 0.62, 1.2, spat=False), 0.05, 41)
pieces.append(place(gear, 2.0, -6.5, 0.9, roll=1.45, sink=0.15))
# the tailplane and a twisted part of the fin
t = W["tail"]
tail = wreck_piece(t, 0.4, 51, flat=0.2)
pieces.append(place(tail, -12.0, 9.0, 1.9, roll=0.5, sink=0.4))
# the fuselage, burst open: strips of corrugated skin and frames scattered about
fus = W["fuselage"]
fc = fus.v[fus.f].mean(1)
strip = np.floor((fc[:, 2] + 30) / 2.4).astype(int) * 7 + np.floor((np.arctan2(fc[:, 1] - 2.5, fc[:, 0]) + math.pi) / (2 * math.pi) * 3).astype(int)
skins = []
for k in np.unique(strip):
    if rng.random() < 0.45:
        continue
    sel = fus.f[strip == k]
    u, inv = np.unique(sel, return_inverse=True)
    m = Mesh(fus.v[u], inv.reshape(-1, 3), fus.c[u])
    m = wreck_piece(m, 0.25, 100 + int(k), flat=0.55)
    r_, a_ = rng.uniform(2, 14), rng.uniform(0, 6.28)
    skins.append(place(m, r_ * math.cos(a_) - 1, r_ * math.sin(a_) + 1, rng.uniform(0, 6.28), roll=rng.uniform(-0.4, 0.4), sink=0.1))
# the nose engine and the other wing engine: deep in the bottom, only the cylinders and a blade showing
for k, (x, z) in enumerate(((-4.0, 0.5), (9.0, -4.5))):
    e = wreck_piece(W[f"engine_{0 if k == 0 else 2}"], 0.15, 60 + k)
    skins.append(place(e, x, z, rng.uniform(0, 6.28), roll=rng.uniform(0.6, 1.4), sink=0.9))
debris = merge(skins)

# the rocky bottom of ridges and hollows: outcrops of rock among the pieces
rocks = []
for i in range(26):
    a_, r_ = rng.uniform(0, 6.28), rng.uniform(7, 30)
    rad = rng.uniform(0.35, 1.0) ** 1.5 * 1.8
    st = abyss.blob(rad, (0.4, 0.39, 0.37), 900 + i, sub=2, amp=0.45).S(rng.uniform(1.0, 1.6), rng.uniform(1.1, 1.7), 1.0)
    st.c = np.array([(0.42, 0.4, 0.38), (0.36, 0.35, 0.34), (0.48, 0.45, 0.4)][i % 3]) * (0.8 + 0.4 * fbm(st.v, 1.5, i, 2))[:, None]
    rocks.append(st.R(0, rng.uniform(0, 6.28), 0).T(r_ * math.cos(a_), rad * 0.45, r_ * math.sin(a_)))
rocks = merge(rocks)

wreck_main = merge(pieces)
for m in (wreck_main, debris):                                     # 84 years in the Baltic: dull, with a film of silt
    m.c = np.clip(m.c * (0.8 + 0.2 * fbm(m.v, 0.7, 3, 2))[:, None], 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1464299" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("main", wreck_main), ("debris_field", debris)],
                   "plain": [("rocks", rocks, "rock")]}], depth=-74, kind="aircraft", seed=1940, wood="alu", life="baltic")
dress_built(out.with_name("model_built.glb"),
            [("airframe", merge([B[k] for k in ("fuselage", "wing_l", "wing_r", "tail", "flaps", "gear_l", "gear_r", "tail_wheel")]), "paint"),
             ("engines", merge([B[k] for k in B if k.startswith("engine_")]), "paint"),
             ("glass", B["glass"], "metal"), ("markings", B["markings"], "paint")])
