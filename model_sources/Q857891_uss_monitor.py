"""
USS Monitor (Q857891) — the wreck of the Civil War ironclad off Cape Hatteras, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: John Ericsson's ironclad of 1862, about 52 m long and 12.6 m in beam: a flat armoured "raft" (deck and
armour belt) riding barely above the water on a narrower iron lower hull, a single revolving turret with two
11-inch Dahlgren guns amidships, a small pilot house forward, one screw and the rudder in a well under the
overhanging stern. She fought CSS Virginia at Hampton Roads on 9 March 1862.

The wreck, after NOAA's descriptions: she foundered in a storm off Cape Hatteras on 31 December 1862 (16 men died)
and was found in 1973 in about 73 m (240 ft) of water. She lies upside down, her stern once resting on the turret
that fell out as she sank; to raise the propeller (1998), the engine (2001) and the turret (2002) a large part of
the stern hull and armour belt was cut away; hull plating lies in the sand at the stern; the hull is
deteriorating; grout bags were laid in 2000 to support it. The heart of the Monitor National Marine Sanctuary.
The list, the heading (not published: 0) and the size of the cut-away are approximate.

    python model_sources/Q857891_uss_monitor.py   -> models/item/Q857891/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, quad, tube, crumple, sstep, fbm  # noqa: E402

rng = np.random.default_rng(1862)
L, B = 52.4, 12.6
RAFT_H, HULL_H = 1.6, 2.4
IRON, BLACK, DARK, WOOD = (0.2, 0.2, 0.21), (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.45, 0.38, 0.28)


def raft_w(z):                                    # the armoured raft: pointed bow, rounded stern
    t = (z + L / 2) / L
    w = 1.0
    if t > 0.82:
        w = math.sqrt(max(0.0, 1 - ((t - 0.82) / 0.18) ** 2))
    if t < 0.12:
        w = min(w, math.sqrt(max(0.0, 1 - ((0.12 - t) / 0.12) ** 2)) * 0.7 + 0.3)
    return B / 2 * max(w, 0.02)


def hull_w(z, y):                                 # the lower hull, narrower, sharp at both ends, under the raft
    t = (z + L / 2) / L
    if t < 0.1 or t > 0.93:
        return 0.0
    k = math.sin(min(1.0, (t - 0.1) / 0.2) * math.pi / 2) * math.sin(min(1.0, (0.93 - t) / 0.25) * math.pi / 2)
    u = y / HULL_H                                # 0 keel … 1 under the raft
    return (B / 2 - 1.2) * k * (0.35 + 0.65 * u ** 0.6)


def prism(wfn, y0, y1, col, n=60):
    """A flat-sided solid between y0 and y1 following the half-width wfn(z)."""
    zs = np.linspace(-L / 2, L / 2, n)
    v, f = [], []
    for z in zs:
        w = wfn(z)
        v += [(-w, y0, z), (w, y0, z), (w, y1, z), (-w, y1, z)]
    for i in range(n - 1):
        a = 4 * i; b = a + 4
        for (p, q) in ((0, 1), (1, 2), (2, 3), (3, 0)):
            f += [(a + p, b + p, b + q), (a + p, b + q, a + q)]
    return Mesh(v, f, col)


def lower_hull():
    zs = np.linspace(-L / 2, L / 2, 70)
    NP = 10
    v, f = [], []
    for z in zs:
        for j in range(2 * NP + 1):
            s = -1 if j < NP else 1
            u = abs(NP - j) / NP
            y = HULL_H * u
            v.append((s * hull_w(z, y), y, z))
    W = 2 * NP + 1
    for i in range(len(zs) - 1):
        for j in range(W - 1):
            a = i * W + j; b = a + W
            f += [(a, a + 1, b), (b, a + 1, b + 1)]
    return Mesh(v, f, IRON)


def ship(built):
    raft = prism(raft_w, HULL_H, HULL_H + RAFT_H, IRON)
    deck = prism(lambda z: raft_w(z) - 0.05, HULL_H + RAFT_H - 0.02, HULL_H + RAFT_H, WOOD if built else IRON)
    P = [lower_hull(), raft, deck]
    y = HULL_H + RAFT_H
    P.append(quad(0, y + 0.02, L / 2 - 4.0, 1.4, 1.4, "+y", DARK))             # the anchor well
    P.append(box(1.4, 0.8, 1.4, IRON).T(0, y, L / 2 - 8.0) if built else Mesh())   # pilot house
    if built:
        tur = cyl(6.1, 6.1, 2.7, IRON, seg=32)
        tur.c[tur.v[:, 1] > 2.65] = (0.3, 0.3, 0.32)
        P.append(tur.T(0, y, 1.0))
        for s in (-1, 1):
            P.append(cyl(0.45, 0.4, 2.8, BLACK, seg=10).R(math.pi / 2, 0, 0).T(s * 1.3, y + 1.4, 5.5))
        P += [box(1.2, 1.8, 1.2, BLACK).T(x, y, -10.0) for x in (-1.0, 1.0)]       # the removable smoke-pipe boxes
        prop = [cyl(0.4, 0.3, 0.8, (0.55, 0.4, 0.2), seg=10).R(math.pi / 2, 0, 0).T(0, 1.2, -L / 2 + 5.5)]
        for k in range(4):
            prop.append(box(1.4, 0.12, 0.7, (0.55, 0.4, 0.2), bottom=False).T(0.9, 0, 0).R(0.4, 0, 0).R(0, 0, k * math.pi / 2).T(0, 1.2, -L / 2 + 5.0))
        P += prop + [box(0.25, 2.0, 1.6, IRON).T(0, 0.3, -L / 2 + 3.4)]
    return merge(P)


# ── The wreck: upside down, the stern cut open by the recovery work ─────────
w = ship(False)
fc = w.v[w.f].mean(1)
cut = (fc[:, 2] < -L / 2 + 16) & (fc[:, 2] > -L / 2 + 2) & (fbm(fc, 0.15, 7) < 0.75) & ((fc[:, 1] < HULL_H + 0.2) | (np.abs(fc[:, 0]) < 4.5))
w.f = w.f[~cut]                                                               # stern hull and belt cut away for the propeller, engine and turret
crumple(w, 0.12, 0.3, 11)
crumple(w, 0.5, 0.25, 12, sstep(-L / 2 + 20, -L / 2 + 6, w.v[:, 2]))           # the deteriorating stern
w.R(0, 0, math.pi - math.radians(12))                                         # upside down, listing onto one side of the belt
w.T(0, -np.percentile(w.v[:, 1], 1.0) - 0.4, 0)
w.T(-0.5 * (w.v[:, 0].min() + w.v[:, 0].max()), 0, 0)
r = np.random.default_rng(73)
deb = []
for k in range(26):                                                           # hull plating lying in the sand at the stern
    p = box(r.uniform(1.0, 3.0), 0.12, r.uniform(0.6, 1.8), IRON); crumple(p, 0.25, 0.5, 100 + k)
    deb.append(p.R(r.uniform(-0.3, 0.3), r.uniform(0, 6.28), r.uniform(-0.3, 0.3)).T(r.normal(0, 6), 0.1, -L / 2 + r.normal(4, 5)))
for k in range(18):                                                           # the grout bags laid in 2000 to support the hull
    g = box(1.0, 0.45, 0.6, (0.58, 0.56, 0.5)); crumple(g, 0.08, 1.2, 200 + k)
    deb.append(g.R(0, r.uniform(0, 6.28), 0).T(r.uniform(-5, 5), 0.0, -L / 2 + r.uniform(6, 16)))
debris = merge(deb)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q857891" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": [], "plain": []}], depth=-73, kind="warship", seed=1862, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", ship(True))], deck_colors=[WOOD])
