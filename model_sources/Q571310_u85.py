"""U-85 (Q571310) — the German Type VIIB U-boat sunk off North Carolina in 1942, and her wreck, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The boat: a Type VIIB U-boat commissioned in 1941: about 66.5 m long, 6.2 m in the beam over the saddle tanks, a
pressure hull inside a flat-topped outer casing, the conning tower with its open bridge and the anti-aircraft platform
aft, an 8.8 cm deck gun forward of it, four bow torpedo tubes and one stern tube, bow and stern diving planes, twin
screws.

The wreck, after the published descriptions (NOAA Monitor National Marine Sanctuary and its Battle of the Atlantic
expeditions, Wikipedia): the first U-boat sunk by the US Navy in the war, she was caught on the surface by the
destroyer USS Roper off Bodie Island on the night of 14 April 1942, hit by her gunfire and depth-charged as she went
down; all 46 men died. One of only 24 boats of her type and the only Type VIIB lost off the US East Coast, she lies
in about 30 m (100 ft) of water off the Bodie Island lighthouse, largely intact, with the shell damage of the Roper's
gunfire behind the conning tower; NOAA surveyed her in 2008 and made a photomosaic and 3D model of the wreck. The list,
the rot of the outer casing and the heading are approximate.

    python model_sources/Q571310_u85.py   -> models/item/Q571310/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, fbm, sstep  # noqa: E402

rng = np.random.default_rng(1942 + 4)
L, RP = 66.5, 2.35
HULLC, DECKC, DARK, BRONZE = (0.36, 0.37, 0.38), (0.3, 0.29, 0.27), (0.05, 0.05, 0.05), (0.5, 0.4, 0.22)
CY = 2.6                                                    # height of the pressure hull's axis above the keel


def zu(u):
    return (u - 0.5) * L


def pressure_hull():
    us = np.linspace(0.06, 0.92, 60)
    prof = [(RP * (np.clip((u - 0.06) / 0.18, 0, 1) ** 0.5) * (np.clip((0.92 - u) / 0.16, 0, 1) ** 0.45) + 0.25, zu(u)) for u in us]
    return lathe(prof, HULLC, seg=24).R(math.pi / 2, 0, 0).T(0, CY, 0)


def casing():
    """The outer casing: the flat deck on top, the free-flooding bow and stern, the saddle tanks on the sides."""
    P = []
    n = 70
    for k in range(n):
        u0, u1 = k / n, (k + 1) / n
        uc = (u0 + u1) / 2
        w = 1.6 * min(1.0, uc / 0.14, (1 - uc) / 0.12) + 0.15
        h = 1.4 * min(1.0, uc / 0.1, (1 - uc) / 0.08) + 0.4
        P.append(box(2 * w, 0.12, L / n + 0.02, DECKC).T(0, CY + RP + 0.2 + (1.2 if uc > 0.9 else 0.0) * sstep(0.9, 1.0, uc), zu(uc)))
        for s in (-1, 1):
            P.append(box(0.08, h, L / n + 0.02, HULLC).T(s * w, CY + RP + 0.2 - h, zu(uc)))
    for s in (-1, 1):                                                  # saddle tanks
        st = lathe([(0.01, -14.0), (0.75, -12.0), (0.95, 0.0), (0.75, 12.0), (0.01, 14.0)], HULLC, seg=14).R(math.pi / 2, 0, 0)
        st.v[:, 0] *= 0.85
        P.append(st.T(s * 2.45, CY + 0.2, zu(0.52)))
    stem = merge([box(0.5, 3.6, 4.0, HULLC).T(0, CY - 1.4, zu(0.97))]); P.append(stem)
    return merge(P)


def tower(damaged):
    """The conning tower: armoured tower, the open bridge with its fairing, the winter garden behind, periscopes."""
    zt = zu(0.6)
    yb = CY + RP + 0.3
    P = [cyl(2.0, 1.7, 3.2, HULLC, seg=20, sx=0.55).T(0, yb, zt)]
    P.append(box(2.4, 0.15, 5.6, DECKC).T(0, yb + 3.2, zt - 0.4))
    P.append(box(3.0, 0.15, 3.2, DECKC).T(0, yb + 2.6, zt - 4.0))                              # the winter garden
    P.append(box(2.0, 2.6, 3.4, HULLC).T(0, yb, zt - 3.6))                                     # its fairing, down to the casing
    P.append(cyl(0.6, 0.6, 1.0, DARK, seg=12).T(0, yb + 2.75, zt - 4.0))
    if not damaged:
        P += [cyl(0.12, 0.1, 5.5, HULLC, seg=8).T(0, yb + 3.2, zt + dz) for dz in (0.4, -0.6)]
    else:
        P += [cyl(0.12, 0.1, 1.5, HULLC, seg=8).T(0, yb + 3.2, zt + dz) for dz in (0.4, -0.6)]
    return merge(P)


def deck_gun(gun):
    z = zu(0.68); y = CY + RP + 0.3
    P = [cyl(0.35, 0.45, 0.8, HULLC, seg=12).T(0, y, z)]
    if gun:
        P += [box(1.2, 1.0, 1.0, HULLC).T(0, y + 0.6, z), cyl(0.06, 0.05, 4.7, HULLC, seg=8).R(math.pi / 2, 0, 0).T(0, y + 1.3, z + 0.3)]
    return merge(P)


def stern_gear():
    P = []
    zr = zu(0.025)
    P.append(box(0.15, 2.8, 2.0, HULLC).T(0, CY - 1.4, zr))
    for s in (-1, 1):
        P.append(box(3.4, 0.12, 1.6, HULLC).T(s * 1.5, CY - 0.2, zu(0.05)))                    # stern planes
        P.append(cyl(0.1, 0.1, 3.0, HULLC, seg=6).R(math.pi / 2, 0, 0).T(s * 1.1, CY - 1.3, zu(0.06)))
        hub = cyl(0.18, 0.18, 0.4, BRONZE, seg=10).R(math.pi / 2, 0, 0).T(s * 1.1, CY - 1.3, zu(0.045))
        P.append(hub)
        for b in range(3):
            P.append(box(0.06, 0.75, 0.3, BRONZE).T(0, 0.45, 0).R(0, 0, 2 * math.pi * b / 3).T(s * 1.1, CY - 1.3, zu(0.045)))
    for s in (-1, 1):
        P.append(box(2.6, 0.12, 1.3, HULLC).T(s * 1.6, CY + 0.8, zu(0.93)))                    # bow planes
    return merge(P)


def torpedo_tubes():
    return merge([cyl(0.3, 0.3, 6.0, (0.5, 0.48, 0.4), seg=12).R(math.pi / 2, 0, 0).T(x, CY + y, zu(0.86)) for x, y in ((-0.6, 0.5), (0.6, 0.5), (-0.6, -0.5), (0.6, -0.5))])


def as_built():
    return merge([pressure_hull(), casing(), tower(False), deck_gun(True), stern_gear()])


def prune(m, min_faces=60, above=1.0, gap=0.2):
    """Drop the little loose bits left hanging in the water (window panes, slivers of plating) once the walls around
    them have been cut away: connected groups of fewer than min_faces faces, wholly above `above` m, that touch no
    other part (their boxes, grown by `gap`, overlap nothing else)."""
    par = np.arange(len(m.v))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for a, b, c in m.f:
        ra, rb, rc = find(a), find(b), find(c)
        par[rb] = ra
        par[find(rc)] = ra
    roots = np.array([find(a) for a in m.f[:, 0]])
    _, inv, cnt = np.unique(roots, return_inverse=True, return_counts=True)
    tri = m.v[m.f]
    lo = np.full((len(cnt), 3), 1e9); hi = np.full((len(cnt), 3), -1e9)
    np.minimum.at(lo, inv, tri.min(1)); np.maximum.at(hi, inv, tri.max(1))
    drop = np.zeros(len(cnt), bool)
    for k in np.nonzero((cnt < min_faces) & (lo[:, 1] > above))[0]:
        ov = np.all((lo <= hi[k] + gap) & (hi >= lo[k] - gap), axis=1)
        ov[k] = False
        drop[k] = not ov.any()
    m.f = m.f[~drop[inv]]
    return m


# ── The wreck ──
cs = casing()
fc = cs.v[cs.f].mean(1)
rot = fbm(fc, 0.22, 3) > 0.52                                     # the outer hull largely rotted away over the frames
cs.f = cs.f[~rot]
frames = [tube([(-1.6, CY + RP + 0.2, zu(u)), (-1.75, CY + 0.6, zu(u)), (-1.2, CY - 1.0, zu(u))], 0.06, HULLC, seg=4) for u in np.arange(0.12, 0.9, 0.025)]
frames += [m.S(-1.0, 1.0, 1.0) for m in [f_.copy() for f_ in frames]]
ph = pressure_hull()
fp = ph.v[ph.f].mean(1)
crack = (fp[:, 2] > zu(0.84)) & (fp[:, 0] < 0.4) & (fbm(fp, 0.5, 5) > 0.35)    # the bow cracked open: the torpedo tubes show
ph.f = ph.f[~crack]
w = merge([ph, cs, tower(False), deck_gun(True), stern_gear(), torpedo_tubes()] + frames)
crumple(w, 0.06, 0.8, 7)
crumple(w, 0.12, 0.35, 8, (np.abs(w.v[:, 2] - zu(0.6)) < 3.5) * (w.v[:, 1] > CY + RP + 0.4))   # the tower holed by the Roper's gunfire
w.R(0, 0, math.radians(45))                                       # a 45-degree list to starboard
w.T(0, -np.percentile(w.v[:, 1], 0.3) - 0.5, 0)
w.T(-0.5 * (w.v[:, 0].min() + w.v[:, 0].max()), 0, 0)
prune(w)
D_ = []
for i in range(40):
    p = box(rng.uniform(0.5, 1.8), 0.06, rng.uniform(0.5, 2.0), HULLC if i % 2 else DECKC)
    crumple(p, 0.08, 0.8, 100 + i)
    D_.append(p.R(rng.uniform(-0.2, 0.2), rng.uniform(0, 6.28), 0).T(rng.normal(-3.0, 2.5), 0.03, rng.uniform(-30, 30)))
debris = merge(D_)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q571310" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", w), ("debris_field", debris, "loose")], "mud": []}],
            depth=-30, kind="submarine", seed=1942, memorial=True)
dress_built(out.with_name("model_built.glb"), [("boat", as_built())])
