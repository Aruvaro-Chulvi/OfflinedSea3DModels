"""ARA San Juan (Q5653859) — the Argentine submarine lost in the South Atlantic in 2017, and her wreck, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The boat: a TR-1700-class diesel-electric submarine built in Germany by Thyssen Nordseewerke and commissioned in 1985:
about 65.9 m long and 7.3 m in the beam, a single cylindrical pressure hull with a rounded bow, the sail amidships
forward with its fairwater planes, cruciform rudders and stern planes, one seven-bladed propeller; black all over.

The wreck, after the published descriptions (Wikipedia, Ocean Infinity, the Argentine Navy, the Maritime Executive):
on 15 November 2017, on her way back to Mar del Plata, she reported water in her battery tank and a short circuit,
then fell silent; she was lost with her 44 crew. Ocean Infinity's Seabed Constructor found her a year later, on 16
November 2018, at about 907-920 m in a ravine some 460 km east of the Valdés peninsula: she had imploded after passing
her collapse depth; the pressure hull lies in one larger section, the stern crushed, and debris is spread over an
area some 70 m across. The damage, the layout and the heading (not published here: 0) are approximate.

    python model_sources/Q5653859_ara_san_juan.py   -> models/item/Q5653859/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, fbm, sstep, grid  # noqa: E402
from shipkit import install  # noqa: E402
from sailkit import solidify  # noqa: E402

rng = np.random.default_rng(2017)
L, R = 65.9, 3.65
BOW_U, AFT_U = 0.9, 0.3
SAIL_U, SAIL_L, SAIL_W, SAIL_H, SAIL_PLANES = 0.66, 9.0, 1.4, 4.2, True
FIN, PROP, BLADES = 2.6, 1.8, 7
HULLC, DGREY, BRONZE = (0.08, 0.08, 0.085), (0.2, 0.2, 0.21), (0.5, 0.4, 0.22)


def radius(u):
    """Hull radius at u (0 stern … 1 bow): the round bow, a parallel middle, the long taper aft."""
    u = np.asarray(u, float)
    bow = np.sqrt(np.clip(1 - ((u - BOW_U) / (1 - BOW_U)) ** 2, 0, 1)) ** 0.9
    aft = np.clip(u / AFT_U, 0, 1) ** 0.62
    r = np.where(u > BOW_U, bow, np.where(u < AFT_U, aft, 1.0))
    return np.maximum(R * r, 0.35)


install(globals(), "zu")


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


# the pressure hull in one larger piece, imploded: the stern telescoped and crushed, the plating split
h = merge([hull(0.0, 1.0), stern_gear(), sail()])
u = h.v[:, 2] / L + 0.5
k = sstep(0.42, 0.05, u)                                            # the after part crushed by the implosion
h.v[:, 2] += k * k * 6.0
h.v[:, :2] = np.array([0, R]) + (h.v[:, :2] - np.array([0, R])) * (1 - 0.3 * k[:, None] * (0.6 + 0.8 * fbm(h.v, 0.4, 3, 2)[:, None]))
crumple(h, 0.5, 0.3, 4, k)
k2 = np.exp(-((u - 0.56) / 0.05) ** 2)                               # a ring of the hull pressed in ahead of the sail
h.v[:, :2] = np.array([0, R]) + (h.v[:, :2] - np.array([0, R])) * (1 - 0.25 * k2[:, None])
fc = h.v[h.f].mean(1)
h.f = h.f[~((fbm(fc, 0.25, 5) > 0.68) & (fc[:, 2] < zu(0.5)))]     # split plating aft
h = solidify(h, 0.3, DGREY)
crumple(h, 0.08, 1.0, 6)
h.R(0, 0, math.radians(-15))
h.T(0, -h.v[:, 1].min() - 1.0, 0)
debris = scatter(160, 0, -10, 18, 25, 61)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q5653859" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", h), ("debris_field", debris, "loose")], "mud": []}],
            depth=-907, kind="submarine", seed=2017, memorial=True, drift=0.5)
dress_built(out.with_name("model_built.glb"), [("boat", as_built())])
