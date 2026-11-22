"""
K-278 Komsomolets (Q499298) — the Soviet nuclear submarine lost in the Norwegian Sea in 1989, and her wreck, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The boat: the one and only Project 685 "Plavnik" (NATO: Mike) nuclear attack submarine, commissioned in 1983, with a
titanium pressure hull that let her dive to 1,000 m: 117.5 m long, 10.7 m in the beam, a long low sail with a
streamlined fairing and the pop-up escape capsule in its top, retractable bow planes, cruciform tail fins with a
large pod on top of the upper rudder, a single propeller; black, covered in rubber anechoic tiles.

She sank on 7 April 1989 after a fire in her aft compartments, south-west of Bjørnøya; 42 of her 69 men died. The
escape capsule came free as she sank and surfaced (the recess in the sail is empty). The wreck lies upright on the
bottom at about 1,680 m. Soviet and Russian expeditions with the Mir submersibles (1989-1998) found the outer hull
torn in places, the anechoic tiles gone in patches, and sealed the holes in the torpedo compartment, where two
torpedoes with nuclear warheads lie; the Norwegian survey of 2019 with the ROV Ægir 6000 filmed the wreck and a
plume leaking from a ventilation pipe aft of the sail, near the reactor compartment. The exact list, damage and
heading are approximate.

    python model_sources/Q499298_k278_komsomolets.py   -> models/item/Q499298/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, fbm  # noqa: E402
from shipkit import install  # noqa: E402

rng = np.random.default_rng(1989)
L, R = 117.5, 5.35
BLACK, DGREY, BRONZE, TI = (0.08, 0.08, 0.085), (0.2, 0.2, 0.21), (0.5, 0.4, 0.22), (0.45, 0.46, 0.47)


def radius(u):
    """Hull radius at u (0 stern … 1 bow): a blunt round bow, a long parallel middle body, a long taper aft."""
    u = np.asarray(u, float)
    bow = np.sqrt(np.clip(1 - ((u - 0.9) / 0.1) ** 2, 0, 1)) ** 0.8
    aft = np.clip(u / 0.36, 0, 1) ** 0.6
    r = np.where(u > 0.9, bow, np.where(u < 0.36, aft, 1.0))
    return np.maximum(R * r, 0.4)


install(globals(), "zu")


def hull(u0=0.0, u1=1.0, n=110, seg=40):
    us = np.linspace(u0, u1, n)
    prof = [(float(radius(u)), zu(u)) for u in us]
    if u1 >= 1.0:
        prof[-1] = (0.001, zu(1.0))
    m = lathe(prof, BLACK, seg=seg)
    m.R(math.pi / 2, 0, 0)
    m.T(0, R, 0)
    return m


def sail(capsule=True):
    """The long low sail, faired into the hull, with the escape capsule in the top of its forward part."""
    zs = zu(0.62)
    body = cyl(11.0, 9.5, 5.2, BLACK, seg=32, sx=0.2).T(0, 2 * R - 0.8, zs)
    body.v[:, 2] += np.where(body.v[:, 2] < zs, (body.v[:, 2] - zs) * 0.45, 0)      # long faired trailing edge
    hy = 2 * R + 4.4
    P = [body, box(1.6, 0.08, 9.0, DGREY).T(0, hy, zs + 1.0)]
    if capsule:
        P.append(cyl(1.5, 1.5, 0.5, DGREY, seg=16, sx=1.6).T(0, hy - 0.4, zs + 5.0))
    else:
        P.append(cyl(1.55, 1.55, 0.06, (0.02, 0.02, 0.02), seg=16, sx=1.6).T(0, hy + 0.01, zs + 5.0))   # the empty recess
    P += [cyl(0.2, 0.16, 1.4, DGREY, seg=8).T(0, hy, zs + dz) for dz in (2.0, 0.6, -1.0)]
    return merge(P)


def stern_gear():
    zt = zu(0.06)
    P = []
    for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        fin = box(0.35, 4.6, 6.0, BLACK, bottom=True).T(0, 0.8, 0)
        fin.v[:, 2] -= (fin.v[:, 1] - 0.8) / 4.6 * 1.6
        fin.R(0, 0, a).T(0, R, zt)
        P.append(fin)
    P.append(lathe([(0.01, -3.0), (1.0, -2.2), (1.1, 2.0), (0.6, 3.6)], BLACK, seg=14).R(math.pi / 2, 0, 0).T(0, R + 6.2, zt))   # the pod, on top of the upper fin
    zp = zu(0.0) - 0.9
    P.append(lathe([(0.001, 0.0), (0.55, 0.3), (0.6, 1.0), (0.4, 1.4)], BRONZE, seg=12).R(-math.pi / 2, 0, 0).T(0, R, zp + 1.4))
    for b in range(7):
        P.append(box(0.9, 2.7, 0.08, BRONZE).R(0, 0.5, 0).R(0, 0, 2 * math.pi * b / 7).T(0, R, zp + 0.8))
    return merge(P)


def bow_planes():
    return merge([box(3.2, 0.25, 2.0, BLACK, bottom=False).T(s * (R + 1.4), R + 0.6, zu(0.86)) for s in (-1, 1)])


def as_built():
    hatches = [box(0.9, 0.06, 0.9, DGREY).T(0, 2 * R - 0.02, zu(u)) for u in (0.9, 0.48, 0.3)]
    m = merge([hull(), sail(True), stern_gear(), bow_planes()] + hatches)
    return m


# ── The wreck: upright on the bottom, sunk a little into the mud, a slight list to starboard ──
w = merge([hull(), sail(False), stern_gear()])
fc = w.v[w.f].mean(1); u = fc[:, 2] / L + 0.5
# the forward hull torn about the torpedo compartment (holes sealed by the Mir expeditions)
tear = (u > 0.86) & (u < 0.94) & (fc[:, 0] < -R * 0.4) & (fc[:, 1] > R * 0.5) & (fc[:, 1] < R * 1.5) & (fbm(fc, 0.4, 3) > 0.55)
w.f = w.f[~tear]
# anechoic tiles gone in patches: lighter grey titanium showing through
vu = w.v[:, 2] / L + 0.5
bare = (fbm(w.v, 0.12, 5) > 0.66) & (vu > 0.08) & (vu < 0.95)
w.c[bare] = TI
crumple(w, 0.05, 1.2, 6)
w.R(0, 0, math.radians(-4))
w.T(0, -w.v[:, 1].min() - 1.4, 0)                                    # her keel in the mud
# the ventilation pipe aft of the sail (the leak filmed in 2019) and a few loose plates and tiles
pipe = tube([(1.0, 2 * R - 1.3, zu(0.47)), (1.2, 2 * R - 0.3, zu(0.465)), (1.4, 2 * R + 0.3, zu(0.455))], 0.18, DGREY, seg=6)
pipe.R(0, 0, math.radians(-4)).T(0, -1.4 - 0.0, 0)
D = [pipe]
for i in range(120):                                                   # rubber tiles fallen beside the hull
    x, z = rng.choice([-1, 1]) * rng.uniform(R + 0.5, R + 9), rng.uniform(-50, 55)
    D.append(box(0.6, 0.06, 0.6, BLACK).R(rng.uniform(-0.3, 0.3), rng.uniform(0, 6.28), 0).T(x, 0.0, z))
for i in range(14):
    x, z = rng.normal(-9, 4), rng.uniform(40, 62)
    p = box(rng.uniform(0.8, 2.0), 0.08, rng.uniform(0.8, 2.5), BLACK); crumple(p, 0.12, 0.6, 30 + i)
    D.append(p.R(rng.uniform(-0.2, 0.2), rng.uniform(0, 6.28), 0).T(x, 0.05, z))
debris = merge(D)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q499298" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": []}],
            depth=-1680, kind="submarine", seed=1989, memorial=True, drift=0.5)
dress_built(out.with_name("model_built.glb"), [("boat", as_built())])
