"""
USS Grunion (Q6147254) — the wreck of the American submarine lost off Kiska, Aleutian Islands, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Gato-class fleet submarine (SS-216) commissioned in April 1942: about 95 m long, beam 8.3 m; the pressure
hull inside a light outer casing with the deck on top, the conning-tower fairwater with the periscope shears, a
3-inch deck gun, bow and stern planes, two screws with their guards.

The wreck, after the published descriptions: she disappeared with all 70 men off Kiska at the end of July 1942.
Sixty-five years later the sons of her commander, Mannert Abele, had her searched for: a sonar target found in
2006, an ROV survey in 2007, and the Navy confirmed her in October 2008 from the footage (her markings, the
propeller guards and limber holes matching her design). She lies at about 1,000 m (3,000-3,200 ft) on the slope
off Kiska, the cold water and slack current keeping much of her intact; the hull has partly collapsed (imploded).
Here the bow is broken off and lies a little apart, with the furrow she cut in the slope: approximate, like the
heading (not published: 0). She is a war grave.

    python model_sources/Q6147254_uss_grunion.py   -> models/item/Q6147254/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, sstep, fbm  # noqa: E402

rng = np.random.default_rng(1942)
L, B = 95.0, 8.3
GREY, DARK, BLACK, STEEL, BRONZE = (0.2, 0.21, 0.22), (0.03, 0.03, 0.035), (0.07, 0.07, 0.08), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
DECKW = (0.32, 0.3, 0.27)


def sub(built):
    prof = [(0.05, -L / 2), (0.9, -L / 2 + 4), (2.2, -L / 2 + 14), (2.5, -10), (2.5, 15), (2.2, L / 2 - 14), (1.1, L / 2 - 4), (0.05, L / 2)]
    ph = lathe(prof, GREY, seg=18, axis="z").T(0, 3.0, 0)                      # pressure hull
    ph.v[:, 0] *= 1.45                                                         # with the saddle ballast tanks, wider than deep
    casing = box(2.8, 1.3, L - 10, GREY).T(0, 4.9, 0)
    casing.v[:, 0] *= np.where(np.abs(casing.v[:, 2]) > L / 2 - 14, np.clip((L / 2 - 4 - np.abs(casing.v[:, 2])) / 10, 0.1, 1), 1)
    deck = box(2.6, 0.08, L - 12, DECKW).T(0, 6.2, 0)
    P = [ph, casing, deck]
    for x in np.arange(-L / 2 + 8, L / 2 - 8, 1.2):                             # limber holes along the casing
        for s in (-1, 1):
            P.append(quad(s * 1.42, 5.2, x, 0.5, 0.35, "+x" if s > 0 else "-x", DARK))
    sail = box(2.4, 4.4, 10.0, GREY).T(0, 6.2, 6.0)
    sail.v[:, 2] += 0.6 * (sail.v[:, 1] > 9.0) * (sail.v[:, 2] > 8)
    P.append(sail)
    for z in (5.0, 7.0):                                                       # periscope shears
        P.append(cyl(0.25, 0.18, 6.0 if built else 3.5, STEEL, seg=8).T(0, 10.6, z))
    gun = merge([cyl(0.35, 0.4, 1.0, GREY, seg=10), cyl(0.09, 0.08, 4.0, STEEL, seg=6).R(math.pi / 2 - 0.1, 0, 0).T(0, 1.2, 0.2)])
    P.append(gun.T(0, 6.2, -6.0))
    for s in (-1, 1):
        P.append(box(3.2, 0.15, 1.6, GREY).T(s * 3.5, 3.2, L / 2 - 9.0))         # bow planes
        P.append(box(3.0, 0.15, 1.4, GREY).T(s * 3.0, 2.6, -L / 2 + 5.0))        # stern planes
        hub = cyl(0.3, 0.25, 0.8, BRONZE, seg=10).R(math.pi / 2, 0, 0).T(s * 2.0, 2.4, -L / 2 + 6.5)
        blades = [box(1.3, 0.1, 0.6, BRONZE, bottom=False).T(0.8, 0, 0).R(0.4, 0, 0).R(0, 0, k * math.pi / 2).T(s * 2.0, 2.4, -L / 2 + 6.4) for k in range(4)]
        guard = tube([(s * 2.0 + s * 1.4, 2.4 - 1.6, -L / 2 + 8.0), (s * 3.6, 2.4, -L / 2 + 7.0), (s * 3.2, 2.4 + 1.6, -L / 2 + 6.0)], 0.08, GREY, seg=4)
        P += [hub, guard] + blades
    P.append(box(0.2, 2.4, 1.8, GREY).T(0, 1.6, -L / 2 + 3.5))                 # rudder
    return merge(P)


# ── The wreck ────────────────────────────────────────────────────────────────
BRK = L / 2 - 16.0                                                            # the bow broke off here (approximate)
w = sub(False)
crumple(w, 0.15, 0.3, 11)
imp = (fbm(w.v, 0.05, 12) > 0.55) & (w.v[:, 1] > 1.0)                          # the hull partly collapsed: dished in by the pressure
w.v[imp] -= (w.v[imp] - np.array([0, 3.0, 0]) * np.array([1, 1, 0]) - np.c_[np.zeros(imp.sum()), np.zeros(imp.sum()), w.v[imp][:, 2]]) * 0.25
crumple(w, 0.5, 0.2, 13, imp.astype(float))
fwd = w.v[w.f].mean(1)[:, 2] > BRK
bow = Mesh(w.v, w.f[fwd], w.c); hullm = Mesh(w.v, w.f[~fwd], w.c)
u, inv = np.unique(bow.f, return_inverse=True); bow = Mesh(bow.v[u], inv.reshape(-1, 3), bow.c[u])
u, inv = np.unique(hullm.f, return_inverse=True); hullm = Mesh(hullm.v[u], inv.reshape(-1, 3), hullm.c[u])
crumple(hullm, 1.0, 0.3, 21, sstep(BRK - 4, BRK, hullm.v[:, 2]))
crumple(bow, 1.0, 0.3, 22, sstep(BRK + 4, BRK, bow.v[:, 2]))
hullm.R(0, 0, math.radians(12)).T(0, -1.2, 0)                                 # settled on the slope with a list
bow.T(0, 0, -(BRK + 8)).R(0, 0, math.radians(-35)).R(0, 0.6, 0).T(9.0, -1.0, BRK + 38.0)   # the bow, broken off, a little apart


def furrow():
    """The furrow she ploughed in the soft slope as she slid to rest."""
    def h(X, Z):
        edge = 1.2 * np.exp(-((np.abs(X) - 5.0) / 2.0) ** 2) - 0.8 * np.exp(-(X / 3.5) ** 2)
        return edge * sstep(L / 2 + 160, L / 2 + 10, Z) * sstep(L / 2 - 20, L / 2 + 10, Z) + 0.1 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.2, 3).reshape(X.shape) - 0.35
    g = grid(-12, 12, L / 2 - 20, L / 2 + 160, 24, 90, h, (0.6, 0.57, 0.5))
    g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > 0.05]
    return g


r = np.random.default_rng(42)
deb = []
for k in range(24):
    p = box(r.uniform(0.6, 2.4), 0.1, r.uniform(0.5, 1.6), GREY); crumple(p, 0.2, 0.6, 100 + k)
    deb.append(p.R(r.uniform(-0.4, 0.4), r.uniform(0, 6.28), 0).T(r.normal(4, 6), 0.1, r.uniform(BRK - 5, BRK + 40)))
debris = merge(deb)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q6147254" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", hullm), ("bow_section", bow), ("debris_field", debris, "loose")], "mud": [furrow()], "plain": []}],
            depth=-975, kind="warship", seed=1942, memorial=True)
dress_built(out.with_name("model_built.glb"), [("main", sub(True))], deck_colors=[DECKW])
