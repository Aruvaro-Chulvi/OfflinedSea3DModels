"""USS Thresher (Q2164015) — the nuclear submarine lost off New England in 1963, and her wreck, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The boat: the lead boat of a new class of US nuclear attack submarines, commissioned in 1961: about 84.9 m long and
9.6 m in the beam, a teardrop hull with a long parallel middle body, the big spherical sonar array in the bow, a
small sail with its fairwater planes, cruciform rudders and stern planes ahead of a single propeller; black all over.

The wreck, after the published descriptions (Wikipedia, the US Navy's records released under the Freedom of
Information Act from 2020, the Naval History and Heritage Command photographs, GlobalSecurity): on 10 April 1963,
during deep-diving trials some 220 miles east of Cape Cod, a flooding casualty in the engine room led to a loss of
propulsion and she sank past her collapse depth with all 129 men aboard. The bathyscaphe Trieste and the research
ships of the search found and photographed her remains in 1963-1964 at about 2,560 m (8,400 ft): the hull broke into
six major sections — the sail, the sonar dome, the bow, the operations spaces, the engineering spaces and the tail
section — most of them in one debris field some 400 yards square, the sail lying upside down with its planes fully
reversed and the tail section off to starboard. The layout of the pieces and their headings are approximate.

    python model_sources/Q2164015_uss_thresher.py   -> models/item/Q2164015/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, fbm, sstep, grid  # noqa: E402
from sailkit import solidify  # noqa: E402

rng = np.random.default_rng(1963)
L, R = 84.9, 4.8
BOW_U, AFT_U = 0.86, 0.5
SAIL_U, SAIL_L, SAIL_W, SAIL_H, SAIL_PLANES = 0.76, 7.6, 1.6, 4.4, True
FIN, PROP, BLADES = 3.2, 2.2, 7
HULLC, DGREY, BRONZE = (0.08, 0.08, 0.085), (0.2, 0.2, 0.21), (0.5, 0.4, 0.22)


def radius(u):
    """Hull radius at u (0 stern … 1 bow): the round bow, a parallel middle, the long taper aft."""
    u = np.asarray(u, float)
    bow = np.sqrt(np.clip(1 - ((u - BOW_U) / (1 - BOW_U)) ** 2, 0, 1)) ** 0.9
    aft = np.clip(u / AFT_U, 0, 1) ** 0.62
    r = np.where(u > BOW_U, bow, np.where(u < AFT_U, aft, 1.0))
    return np.maximum(R * r, 0.35)


def zu(u):
    return (u - 0.5) * L


def hull(u0=0.0, u1=1.0, n=90, seg=40):
    us = np.linspace(u0, u1, n)
    prof = [(float(radius(u)), zu(u)) for u in us]
    if u1 >= 1.0:
        prof[-1] = (0.001, zu(1.0))
    m = lathe(prof, HULLC, seg=seg)
    m.R(math.pi / 2, 0, 0)                                          # lathe along +y -> along +z
    m.T(0, R, 0)
    return m


def sail():
    """The sail (fin) with its fairwater planes."""
    zs = zu(SAIL_U)
    body = cyl(SAIL_L / 2, SAIL_L / 2 * 0.9, SAIL_H, HULLC, seg=28, sx=SAIL_W / SAIL_L).T(0, 2 * R - 0.6, zs)
    body.v[:, 2] += np.where(body.v[:, 2] < zs, (body.v[:, 2] - zs) * 0.35, 0)     # longer, finer trailing edge
    top = box(SAIL_W * 0.9, 0.08, SAIL_L * 0.8, DGREY).T(0, 2 * R - 0.6 + SAIL_H, zs + 0.3)
    planes = [box(3.0, 0.25, 2.0, HULLC, bottom=False).T(s * (SAIL_W / 2 + 1.5), 2 * R + SAIL_H * 0.55, zs + 1.2) for s in (-1, 1)] if SAIL_PLANES else []
    masts = [cyl(0.16, 0.14, 1.2, DGREY, seg=8).T(dx, 2 * R - 0.6 + SAIL_H, zs + dz) for dx, dz in ((0, 1.2), (0, 0.2), (0.25, -0.8))]
    return merge([body, top] + planes + masts)


def stern_gear():
    """Cruciform rudders and stern planes, ahead of the single propeller."""
    zt = zu(0.07)
    P = []
    for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        fin = box(0.3, FIN, FIN * 1.35, HULLC, bottom=True).T(0, 0.6, 0)
        fin.v[:, 2] -= (fin.v[:, 1] - 0.6) / FIN * 1.2               # swept trailing edge
        fin.R(0, 0, a).T(0, R, zt)
        P.append(fin)
    zp = zu(0.0) - 0.9
    P.append(lathe([(0.001, 0.0), (0.45, 0.3), (0.5, 0.9), (0.35, 1.3)], BRONZE, seg=12).R(-math.pi / 2, 0, 0).T(0, R, zp + 1.3))
    for b in range(BLADES):
        a = 2 * math.pi * b / BLADES
        P.append(box(0.9, PROP, 0.08, BRONZE).R(0, 0.5, 0).R(0, 0, a).T(0, R, zp + 0.7))
    return merge(P)


def as_built():
    h = hull()
    hatches = [box(0.9, 0.06, 0.9, DGREY).T(0, 2 * R - 0.02, zu(u)) for u in (0.9, 0.62, 0.35)]
    return merge([h, sail(), stern_gear()] + hatches)


def jagged(m, u_cut, keep_above, seed, depth=1.6):
    """Cut the hull at u_cut with a torn, uneven edge."""
    fc = m.v[m.f].mean(1)
    u = fc[:, 2] / L + 0.5
    edge = u_cut + depth / L * (fbm(fc, 0.35, seed, 3) - 0.5) * 2
    keep = u > edge if keep_above else u < edge
    sel = m.f[keep]
    k, inv = np.unique(sel, return_inverse=True)
    return Mesh(m.v[k], inv.reshape(-1, 3), m.c[k])


def piece(u0, u1, seed, crush=0.0):
    """A section of the hull between u0 and u1 with torn ends, its plating thickened, optionally crushed."""
    m = jagged(jagged(hull(max(0.0, u0 - 0.03), min(1.0, u1 + 0.03)), u0, True, seed), u1, False, seed + 1)
    if crush:
        k = crush * (0.6 + 0.8 * fbm(m.v, 0.3, seed + 2, 2))
        m.v[:, :2] = np.array([0, R]) + (m.v[:, :2] - np.array([0, R])) * (1 - 0.35 * k[:, None])
        crumple(m, 0.4 * crush, 0.3, seed + 3)
    m = solidify(m, 0.3, DGREY)
    crumple(m, 0.06, 1.2, seed + 4)
    m.v -= m.v.mean(0)
    return m


def place_on(m, x, z, yaw, roll=0.0, sink=0.5):
    m.R(0, 0, roll).R(0, yaw, 0)
    m.T(x, -m.v[:, 1].min() - sink, z)
    return m


def scatter(n, cx, cz, sx, sz, seed):
    """Plating, frames, piping and battery cells thrown out of the hull."""
    rr = np.random.default_rng(seed)
    D = []
    for i in range(n):
        x, z = rr.normal(cx, sx), rr.normal(cz, sz)
        k = i % 3
        if k == 0:
            p = box(rr.uniform(0.6, 2.5), 0.06, rr.uniform(0.6, 2.5), HULLC)
            crumple(p, 0.15, 0.6, seed + i)
        elif k == 1:
            ln = rr.uniform(1.5, 5)
            p = tube([(0, 0.1, 0), (ln * 0.4, 0.1 + rr.uniform(0, 0.5), ln * 0.2), (ln, 0.1, ln * rr.uniform(-0.3, 0.3))], rr.uniform(0.05, 0.15), DGREY, seg=5)
        else:
            p = box(0.36, 0.55, 0.4, (0.15, 0.15, 0.16))
        D.append(p.R(rr.uniform(-0.4, 0.4), rr.uniform(0, 6.28), rr.uniform(-0.4, 0.4)).T(x, 0.02, z))
    return merge(D)


# the six major sections spread over a field some 400 yards (365 m) square; the tail section off to starboard
bow = place_on(piece(0.84, 0.98, 11, crush=0.3), 0.0, 60.0, math.radians(20), math.radians(12), sink=1.2)
ops = place_on(piece(0.55, 0.84, 21, crush=0.8), -30.0, 0.0, math.radians(-35), math.radians(40), sink=1.0)
eng = place_on(piece(0.18, 0.55, 31, crush=0.6), 25.0, -60.0, math.radians(160), math.radians(-25), sink=1.2)
tail = merge([jagged(hull(0.0, 0.2), 0.18, False, 41), stern_gear()])
tail = solidify(tail, 0.3, DGREY); crumple(tail, 0.15, 0.6, 42); tail.v -= tail.v.mean(0)
tail = place_on(tail, 120.0, -100.0, math.radians(70), math.radians(-60), sink=0.8)
sl = sail(); sl.v -= sl.v.mean(0); crumple(sl, 0.15, 0.4, 51)
sl.R(0, 0, math.pi).R(0, math.radians(110), 0)                                     # upside down, the planes reversed
sl.T(-70.0, -sl.v[:, 1].min() - 0.4, 40.0)
dome = merge([lathe([(0.001, -2.6), (2.0, -1.8), (2.6, 0.0), (2.0, 1.8), (0.001, 2.6)], (0.24, 0.24, 0.22), seg=16)])
crumple(dome, 0.3, 0.6, 52)
dome.T(-40.0, 1.8, 110.0)
debris = merge([scatter(220, 0, 0, 45, 60, 61), sl, dome])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q2164015" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", eng), ("bow_section", bow), ("operations_section", ops), ("stern_section", tail),
                             ("debris_field", debris, "loose")], "mud": []}],
            depth=-2560, kind="submarine", seed=1963, memorial=True, drift=0.6)
dress_built(out.with_name("model_built.glb"), [("boat", as_built())])
