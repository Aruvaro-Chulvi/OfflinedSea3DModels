"""
USS Scorpion (Q1757204) — the nuclear submarine lost in the Atlantic in 1968, and her wreck, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The boat: a Skipjack-class nuclear attack submarine of the US Navy, commissioned in 1960: the teardrop hull of
the class, 76.7 m long and 9.6 m in the beam, a tall sail well forward with the fairwater (diving) planes on it,
the cruciform rudders and stern planes ahead of a single propeller; black all over.
She was lost with her 99 men on 22 May 1968, south-west of the Azores, on her way home to Norfolk. The research
ship Mizar found the wreck in October 1968 at about 3,000 m, and the bathyscaphe Trieste II photographed it in
1969: the hull is broken. The bow section, with the torpedo room, lies in one impact trench; the reactor
compartment and the engine room made a second trench, the engine room driven forward into the auxiliary
machinery space as the hull collapsed; the operations compartment between them was torn apart; the sail came
off and lies nearby, the starboard fairwater plane standing out of it; a large debris field, with battery cells
and the contents of the operations compartment, lies mainly to the north of the big pieces.
The headings and the distances between the pieces are approximate.

    python model_sources/Q1757204_scorpion.py      -> models/item/Q1757204/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, weather, fbm, sstep, grid  # noqa: E402
from shipkit import install  # noqa: E402
from sailkit import solidify  # noqa: E402

rng = np.random.default_rng(1968)
L, R = 76.7, 4.8
BLACK, DGREY, BRONZE = (0.08, 0.08, 0.085), (0.2, 0.2, 0.21), (0.5, 0.4, 0.22)


def radius(u):
    """Hull radius at u (0 stern … 1 bow): the round bow, a short parallel middle, the long taper aft."""
    u = np.asarray(u, float)
    bow = np.sqrt(np.clip(1 - ((u - 0.84) / 0.16) ** 2, 0, 1)) ** 0.9
    aft = np.clip(u / 0.58, 0, 1) ** 0.62
    r = np.where(u > 0.84, bow, np.where(u < 0.58, aft, 1.0))
    return np.maximum(R * r, 0.35)


install(globals(), "zu")


def hull(u0=0.0, u1=1.0, n=90, seg=40):
    us = np.linspace(u0, u1, n)
    prof = [(float(radius(u)), zu(u)) for u in us]
    if u1 >= 1.0:
        prof[-1] = (0.001, zu(1.0))
    m = lathe(prof, BLACK, seg=seg)
    m.R(math.pi / 2, 0, 0)                                          # lathe along +y -> along +z
    m.T(0, R, 0)
    return m


def sail():
    """The sail (fin) with its fairwater planes: 9 m long, about 6 m above the hull."""
    zs = zu(0.74)
    body = cyl(4.6, 4.2, 6.6, BLACK, seg=28, sx=0.19).T(0, 2 * R - 0.6, zs)
    body.v[:, 2] += np.where(body.v[:, 2] < zs, (body.v[:, 2] - zs) * 0.35, 0)     # longer, finer trailing edge
    top = box(1.2, 0.08, 6.0, DGREY).T(0, 2 * R + 6.0, zs + 0.3)
    planes = [box(3.4, 0.25, 2.2, BLACK, bottom=False).T(s * (0.85 + 1.7), 2 * R + 3.6, zs + 1.6) for s in (-1, 1)]
    masts = [cyl(0.18, 0.15, 1.2, DGREY, seg=8).T(dx, 2 * R + 6.0, zs + dz) for dx, dz in ((0, 1.6), (0, 0.4), (0.3, -0.8))]
    return merge([body, top] + planes + masts)


def stern_gear():
    """Cruciform rudders and stern planes, ahead of the single five-bladed propeller."""
    zt = zu(0.07)
    P = []
    for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        fin = box(0.3, 3.4, 4.6, BLACK, bottom=True).T(0, 0.6, 0)
        fin.v[:, 2] -= (fin.v[:, 1] - 0.6) / 3.4 * 1.2               # swept trailing edge
        fin.R(0, 0, a).T(0, R, zt)
        P.append(fin)
    zp = zu(0.0) - 0.9
    P.append(lathe([(0.001, 0.0), (0.45, 0.3), (0.5, 0.9), (0.35, 1.3)], BRONZE, seg=12).R(-math.pi / 2, 0, 0).T(0, R, zp + 1.3))
    for b in range(5):
        a = 2 * math.pi * b / 5
        P.append(box(0.9, 2.3, 0.08, BRONZE).R(0, 0.5, 0).R(0, 0, a).T(0, R, zp + 0.7))
    return merge(P)


def as_built():
    h = hull()
    hatches = [box(0.9, 0.06, 0.9, DGREY).T(0, 2 * R - 0.02, zu(u)) for u in (0.9, 0.62, 0.35)]
    return merge([h, sail(), stern_gear()] + hatches)


# ── The wreck ────────────────────────────────────────────────────────────────
def jagged(m, u_cut, keep_above, seed, depth=1.6):
    """Cut the hull at u_cut with a torn, uneven edge."""
    fc = m.v[m.f].mean(1)
    u = fc[:, 2] / L + 0.5
    edge = u_cut + depth / L * (fbm(fc, 0.35, seed, 3) - 0.5) * 2
    keep = u > edge if keep_above else u < edge
    sel = m.f[keep]
    k, inv = np.unique(sel, return_inverse=True)
    return Mesh(m.v[k], inv.reshape(-1, 3), m.c[k])


# the bow section: the torpedo room and the forward part, broken off ahead of the sail, in its trench
bow = solidify(jagged(hull(0.62, 1.0), 0.75, True, 3), 0.35, DGREY)
bow.v[:, 2] -= zu(0.75)
crumple(bow, 0.35, 0.3, 4, sstep(4, 0, bow.v[:, 2]))
crumple(bow, 0.06, 1.2, 5)
bow.R(0, 0, math.radians(14)).R(math.radians(-6), math.radians(25), 0)
bow.T(0, -bow.v[:, 1].min() - 2.6, 0)                               # half its height in the trench
# the stern section: reactor compartment, machinery space and engine room, the engine room driven forward into it
st = merge([hull(0.0, 0.5), stern_gear()])
st = jagged(st, 0.46, False, 7)
u = st.v[:, 2] / L + 0.5
k = sstep(0.3, 0.46, u)                                             # the forward 12 m telescoped into 5 m, crushed
st.v[:, 2] -= k * k * 7.0
st.v[:, :2] = np.array([0, R]) + (st.v[:, :2] - np.array([0, R])) * (1 - 0.18 * k[:, None] * (0.6 + 0.8 * fbm(st.v, 0.4, 8, 2)[:, None]))
crumple(st, 0.5, 0.3, 9, k)
st = solidify(st, 0.35, DGREY)
crumple(st, 0.06, 1.2, 10)
st.v[:, 2] -= zu(0.4)
st.R(0, 0, math.radians(-20)).R(math.radians(4), math.radians(-150), 0)
st.T(-25.0, -st.v[:, 1].min() - 2.4, -62.0)
# the sail, torn off, on its side with the starboard fairwater plane standing out of it
sl = sail()
sl.v -= sl.v.mean(0)
crumple(sl, 0.2, 0.4, 11)
sl.R(0, 0, math.radians(-84)).R(0, math.radians(70), 0)
sl.T(14.0, -sl.v[:, 1].min() - 0.5, -24.0)

# the trenches: raised rims of mud thrown up round both big pieces
def rim(cx, cz, rx, rz, h, seed):
    def f(X, Z):
        d = np.sqrt(((X - cx) / rx) ** 2 + ((Z - cz) / rz) ** 2)
        return h * np.exp(-((d - 1.0) / 0.22) ** 2) * (0.7 + 0.6 * fbm(np.stack([X, X * 0, Z], -1).reshape(-1, 3), 0.1, seed, 2).reshape(X.shape))
    return grid(cx - rx * 1.5, cx + rx * 1.5, cz - rz * 1.5, cz + rz * 1.5, 40, 40, f, (0.5, 0.48, 0.44))


bc = bow.v.mean(0); sc = st.v.mean(0)
muds = [rim(bc[0], bc[2], 9.0, 15.0, 1.6, 1), rim(sc[0], sc[2], 9.0, 17.0, 1.8, 2)]

# ── The debris field, mostly to the north (+z): the operations compartment torn apart ──
D = []
ops = hull(0.46, 0.76, n=30, seg=40)
fc = ops.v[ops.f].mean(1)
cell = np.floor((fc[:, 2] + 40) / 3.0).astype(int) * 11 + np.floor((np.arctan2(fc[:, 1] - R, fc[:, 0]) + math.pi) / (2 * math.pi) * 6).astype(int)
for c in np.unique(cell):
    if rng.random() < 0.3:
        continue
    sel = ops.f[cell == c]
    kk, inv = np.unique(sel, return_inverse=True)
    m = solidify(Mesh(ops.v[kk], inv.reshape(-1, 3), ops.c[kk]), 0.3, DGREY)
    crumple(m, 0.22, 0.3, 200 + int(c))
    m.v -= m.v.mean(0)
    m.R(rng.uniform(-0.5, 0.5), rng.uniform(0, 6.28), rng.uniform(-0.5, 0.5))
    x, z = rng.normal(5, 28), rng.uniform(-10, 95)
    m.T(x, -m.v[:, 1].min() - 0.3, z)
    D.append(m)
for i in range(260):                                                 # battery cells
    x, z = rng.normal(0, 24), rng.uniform(-20, 100)
    c_ = box(0.36, 0.55, 0.4, (0.15, 0.15, 0.16)).R(rng.uniform(-1.5, 1.5), rng.uniform(0, 6.28), rng.uniform(-1.5, 1.5))
    D.append(c_.T(x, 0.0, z))
for i in range(40):                                                  # piping, cable runs, frames
    x, z = rng.normal(3, 22), rng.uniform(-15, 90)
    ln = rng.uniform(1.5, 6)
    pts = [(0, 0.1, 0), (ln * 0.4, 0.1 + rng.uniform(0, 0.6), ln * 0.2), (ln, 0.1, ln * rng.uniform(-0.3, 0.3))]
    t_ = tube(pts, rng.uniform(0.05, 0.18), DGREY if i % 2 else (0.45, 0.32, 0.2), seg=5)
    D.append(t_.R(0, rng.uniform(0, 6.28), 0).T(x, 0, z))
debris = merge(D)

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1757204" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("bow_section", bow), ("stern_section", st), ("sail", sl), ("debris_field", debris)], "mud": muds}],
            depth=-3000, kind="submarine", seed=1968)
dress_built(out.with_name("model_built.glb"), [("boat", as_built())])
