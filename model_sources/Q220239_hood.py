"""
HMS Hood (Q220239) — the wreck of the battlecruiser in the Denmark Strait, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: Admiral-class battlecruiser of 1920, length 262.3 m overall, beam 31.8 m, draught about 9.8 m; a long
raised forecastle with a step down to the quarterdeck aft; four twin 15-inch turrets (A and B forward, X and Y
aft), the armoured conning tower, the bridge with the tripod foremast and its spotting top, two funnels, the
tripod mainmast; as she sailed in May 1941: twin 4-inch high-angle mounts and multiple 2-pounder «pom-poms»
along the superstructure, wooden decks. Four propellers and one rudder. The guns and fittings come from the
Bismarck's model (Q150857_bismarck.py), with the Hood's own sizes.
She blew up and sank in three minutes on 24 May 1941 in the Battle of the Denmark Strait, fired on by Bismarck
and Prinz Eugen; three of her 1,418 men survived. The wreck was found by David Mearns in 2001 at about
2,800 m, in pieces among two debris fields: the after magazine explosion destroyed the after part of the ship;
the bow section, sheared off just ahead of A turret, lies on its port side; the long midships section, the
biggest piece, lies upside down in a large impact crater south of the eastern debris field; a small piece of
the stern rises from the bottom at an angle; the conning tower lies by itself, away from the rest; the
propellers lie in the eastern field, a 4-inch director in the western one.
The headings and the distances between the pieces are approximate.

    python model_sources/Q220239_hood.py      -> models/item/Q220239/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, weather, sstep, fbm, write_glb  # noqa: E402
from shipkit import install  # noqa: E402

rng = np.random.default_rng(1920)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B = 262.3, 31.8
D = 15.0                  # keel to upper deck amidships
WL = 9.8                  # draught
FB = 247.0                # frame number at the bow tip (frames are metres from the aft perpendicular)
LV = 2.7                  # height of one superstructure deck
S0 = D                    # base of the superstructure
S = [S0 + LV * (i + 1) for i in range(4)]        # tops of superstructure levels 1..4
GREY, DGREY, BLACK, RED = (0.55, 0.57, 0.58), (0.36, 0.38, 0.4), (0.08, 0.08, 0.09), (0.36, 0.13, 0.1)     # Home Fleet grey
TEAK, STEEL, SILVER, DARK = (0.52, 0.42, 0.3), (0.42, 0.43, 0.45), (0.74, 0.75, 0.76), (0.025, 0.025, 0.03)
BRONZE, MACH, GLASS, PLANE = (0.55, 0.4, 0.2), (0.2, 0.18, 0.16), (0.1, 0.12, 0.14), (0.32, 0.37, 0.32)
MUD = (0.55, 0.52, 0.45)


def fr(frame):          # frame number -> distance from the bow
    return FB - frame


A_T, B_T, X_T, Y_T = 52.0, 70.0, 196.0, 214.0      # the 15-inch turrets (distance from the bow)


def w_of(t):            # half-beam factor: long fine entrance, short parallel middle body, rounded cruiser stern
    w = 1.0
    if t > 0.60:
        w = math.sin(((1 - t) / 0.40) * math.pi / 2) ** 0.9
    if t < 0.30:
        w = min(w, 0.38 + 0.62 * math.sin((t / 0.30) * math.pi / 2))
    if t < 0.035:
        w *= math.sqrt(max(0.0, 1 - ((0.035 - t) / 0.035) ** 2)) * 0.85 + 0.15     # rounded in plan
    return max(w, 0.004)


def yb_of(t):           # keel line: cut up aft for the propellers and rudders, raked clipper stem forward
    y = 0.0
    if t < 0.10:
        y = 7.5 * ((0.10 - t) / 0.10) ** 1.6
    if t > 0.955:
        y = max(y, 17.0 * ((t - 0.955) / 0.045) ** 1.8)
    return y


def yd_of(t):           # the forecastle deck, a deck higher than the quarterdeck aft, with the sheer at the bow
    y = D + 2.6 * sstep(0.30, 0.33, t)
    if t > 0.6:
        y += 3.5 * ((t - 0.6) / 0.4) ** 2
    return y


def k_of(t):            # section fullness: boxy amidships (CM 0.97), V-shaped and flared at the ends
    return 1.8 + 10.0 * w_of(t) ** 4


install(globals(), "zd t_of hw deck_y at")


def outward(m):
    """Turn every face of a convex piece outwards (from the centre of its vertices)."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, c - a)
    flip = (n * ((a + b + c) / 3 - m.v.mean(0))).sum(1) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def prism(prof, half_w, c, cols=None):
    """A gunhouse-like prism across x (from -half_w to +half_w) of a convex side profile [(z, y)];
    cols = {edge index: colour} (e.g. a darker roof)."""
    n = len(prof)
    parts = [Mesh([(s * half_w, y, z) for z, y in prof], [(0, i, i + 1) for i in range(1, n - 1)], c) for s in (-1, 1)]
    for i in range(n):
        (z0, y0), (z1, y1) = prof[i], prof[(i + 1) % n]
        parts.append(Mesh([(-half_w, y0, z0), (half_w, y0, z0), (half_w, y1, z1), (-half_w, y1, z1)], [(0, 1, 2), (0, 2, 3)],
                          (cols or {}).get(i, c)))
    return outward(merge(parts))


def barrel(r1, r2, length, c, seg=8):
    """A gun barrel along +z from 0 to length."""
    return cyl(r1, r2, length, c, seg=seg).R(math.pi / 2, 0, 0)


def hull(d0, d1, NP=22):
    """Hull from d0 to d1 (distance from bow): both sides, the deck and, at the stern end, the stern closed."""
    ns = max(2, int(math.ceil((d1 - d0) / 1.4)))
    st = []
    for i in range(ns + 1):
        z = zd(d1) + (zd(d0) - zd(d1)) * i / ns
        t = t_of(z)
        st.append((z, w_of(t), yb_of(t), yd_of(t), k_of(t)))
    W = 2 * NP + 1
    v, f = [], []
    for z, w, yb, yd, k in st:
        for j in range(W):
            side = -1 if j < NP else 1
            u = abs(NP - j) / NP
            v.append((side * B / 2 * w * (1 - (1 - u) ** k), yb + (yd - yb) * u, z))
    for i in range(ns):
        for j in range(W - 1):
            a = i * W + j; b = a + W; c = a + 1; d = b + 1
            f += [(a, c, b), (b, c, d)]
    if d1 >= L - 0.01:
        c0 = len(v); z, w, yb, yd, k = st[0]
        v.append((0, (yb + yd) / 2, z))
        f += [(c0, j + 1, j) for j in range(W - 1)]
    sides = Mesh(v, f)
    sides.paint(lambda p: np.where((p[:, 1] < WL - 0.5)[:, None], RED, np.where((p[:, 1] < WL + 0.6)[:, None], BLACK, GREY)))
    v, f = [], []
    for z, w, yb, yd, k in st:
        v += [(-B / 2 * w, yd, z), (B / 2 * w, yd, z)]
    for i in range(ns):
        a = 2 * i; b = a + 2; c = a + 1; d = b + 1
        f += [(a, b, c), (c, b, d)]
    deck = Mesh(v, f, TEAK)
    return merge([sides, deck])


# ── Guns and fittings ────────────────────────────────────────────────────────
def main_turret(rf=True, el=0.0, guns=(True, True)):
    """38 cm twin turret (Drh LC/34), pivot at the origin, base on y = 0, guns towards +z, elevation el (rad)."""
    prof = [(-7.8, 0), (5.2, 0), (5.2, 1.3), (4.3, 3.15), (-7.8, 3.15)]
    P = [prism(prof, 5.7, GREY, {3: DGREY})]
    for s, g in zip((-1.6, 1.6), guns):
        if not g:
            P.append(cyl(0.75, 0.75, 0.6, DARK, seg=10).R(math.pi / 2, 0, 0).T(s, 1.55, 4.9))   # the empty port
            continue
        gun = merge([barrel(0.85, 0.8, 1.1, DGREY, seg=10), barrel(0.56, 0.33, 17.0, GREY, seg=10).T(0, 0, 1.0),
                     barrel(0.38, 0.38, 0.5, GREY, seg=10).T(0, 0, 17.2)])
        P.append(gun.R(-el, 0, 0).T(s, 1.55, 4.6))
    if rf:                                                                    # the 10.5 m rangefinder across the rear
        P.append(box(10.5, 0.95, 1.2, GREY, bottom=False).T(0, 2.3, -5.6))
        for s in (-1, 1):
            P.append(cyl(0.75, 0.75, 1.2, GREY, seg=10).R(0, 0, math.pi / 2).T(s * 5.25 + s * 0.6, 2.3, -5.6))
    P.append(cyl(0.8, 0.7, 0.6, DGREY, seg=10).T(0, 3.15, -4.0))           # commander's cupola
    for s in (-1, 1):
        P.append(box(0.8, 0.5, 1.2, DGREY).T(s * 3.6, 3.15, 2.2))           # sight hoods
    return merge(P)


def turret_machinery(depth=7.0, seed=0):
    """What tore out with the gunhouse: the roller ring, the turret trunk, hoists and pipes (below y = 0)."""
    r = np.random.default_rng(seed)
    P = [cyl(5.5, 5.5, 0.7, MACH, seg=20).T(0, -0.7, 0),
         cyl(4.6, 4.3, depth, MACH, seg=18, caps=False).T(0, -depth - 0.7, 0)]
    for i in range(7):
        x, z = r.uniform(-3.2, 3.2), r.uniform(-3.2, 3.2)
        P.append(tube([(x, -0.5, z), (x + r.uniform(-2, 2), -depth * r.uniform(0.6, 1.3), z + r.uniform(-2, 2)),
                       (x + r.uniform(-3, 3), -depth * r.uniform(1.0, 1.6), z + r.uniform(-3, 3))], r.uniform(0.15, 0.5), MACH, seg=6))
    return merge(P)


def sec_turret(rf=False, el=0.05):
    """15 cm twin turret (Drh LC/34), guns towards +z."""
    prof = [(-3.6, 0), (3.0, 0), (3.0, 1.0), (2.4, 2.5), (-3.6, 2.5)]
    P = [cyl(3.2, 3.2, 0.5, GREY, seg=16), prism(prof, 3.0, GREY, {3: DGREY}).T(0, 0.5, 0)]
    for s in (-0.85, 0.85):
        P.append(merge([barrel(0.3, 0.3, 0.6, DGREY), barrel(0.22, 0.15, 7.6, GREY).T(0, 0, 0.5)]).R(-el, 0, 0).T(s, 1.7, 2.6))
    if rf:
        P.append(box(7.0, 0.6, 0.7, GREY, bottom=False).T(0, 2.3, -2.4))
    return merge(P)


def flak105(el=0.45):
    """10.5 cm twin flak mount (shielded), guns towards +z."""
    prof = [(-1.8, 0), (1.8, 0), (1.8, 1.6), (1.2, 2.4), (-1.8, 2.4)]
    P = [cyl(1.4, 1.4, 0.9, GREY, seg=12), prism(prof, 1.6, GREY).T(0, 0.9, 0)]
    for s in (-0.45, 0.45):
        P.append(barrel(0.12, 0.09, 6.6, GREY, seg=6).R(-el, 0, 0).T(s, 2.1, 1.2))
    return merge(P)


def flak37(el=0.35):
    P = [cyl(0.7, 0.7, 0.6, GREY, seg=8), box(1.8, 1.0, 0.15, GREY).T(0, 0.6, 0.7)]
    for s in (-0.3, 0.3):
        P.append(barrel(0.06, 0.05, 3.0, GREY, seg=5).R(-el, 0, 0).T(s, 1.0, 0.3))
    return merge(P)


def cupola(r=2.2, h=2.0, rf=10.5, radar=True):
    """Fire-control cupola with its rangefinder arms and the FuMO 23 radar mattress on the front."""
    P = [cyl(r, r * 0.92, h, GREY, seg=18), box(rf, 0.8, 0.9, GREY, bottom=False).T(0, h * 0.55, 0)]
    for s in (-1, 1):
        P.append(cyl(0.55, 0.55, 0.9, GREY, seg=8).R(0, 0, math.pi / 2).T(s * (rf / 2 + 0.45), h * 0.55, 0))
    if radar:
        P.append(box(4.0, 2.0, 0.25, DGREY).T(0, h, r * 0.6))
    return merge(P)


def director():
    """SL-8 stabilised flak director: a dome on a drum."""
    return merge([cyl(1.4, 1.4, 1.0, GREY, seg=14), lathe([(1.5, 0), (1.35, 0.6), (0.9, 1.15), (0.01, 1.4)], GREY, seg=14).T(0, 1.0, 0)])


def searchlight():
    return merge([cyl(0.3, 0.3, 0.7, GREY, seg=6), cyl(0.75, 0.75, 1.0, GREY, seg=12).R(math.pi / 2, 0, 0).T(0, 1.4, -0.5),
                  cyl(0.7, 0.7, 0.02, (0.55, 0.6, 0.62), seg=12).R(math.pi / 2, 0, 0).T(0, 1.4, 0.52)])


def boat(c=GREY):
    m = lathe([(0.05, -5.0), (0.9, -4.2), (1.45, -2.4), (1.55, 0), (1.45, 2.4), (0.9, 4.2), (0.05, 5.2)], c, seg=10, axis="z")
    m.v[:, 1] = np.minimum(m.v[:, 1], 0.4)
    return m.S(1, 0.9, 1)


def arado():
    """Arado Ar 196 floatplane, nose towards +z."""
    P = [lathe([(0.05, -5.5), (0.35, -4.0), (0.7, -1.0), (0.75, 2.6), (0.6, 4.3), (0.3, 5.3)], PLANE, seg=10, axis="z"),
         box(12.4, 0.18, 2.0, PLANE, bottom=False).T(0, 0.35, 1.3), box(4.4, 0.12, 1.2, PLANE, bottom=False).T(0, 0.3, -4.8),
         box(0.12, 1.6, 1.3, PLANE).T(0, 0.2, -5.0), cyl(1.1, 1.1, 0.05, DARK, seg=10).R(math.pi / 2, 0, 0).T(0, 0, 5.4)]
    for s in (-1, 1):
        P.append(lathe([(0.05, -3.6), (0.4, -2.4), (0.45, 0.5), (0.35, 2.8), (0.05, 3.9)], (0.5, 0.52, 0.5), seg=8, axis="z").T(s * 1.8, -2.0, 0.6))
        P += [tube([(s * 1.8, -1.6, 1.8), (s * 0.5, 0.3, 1.8)], 0.06, PLANE, seg=4), tube([(s * 1.8, -1.6, -0.6), (s * 0.5, 0.3, -0.6)], 0.06, PLANE, seg=4)]
    return merge(P)


def propeller(r=2.35):
    P = [cyl(0.55, 0.45, 1.6, BRONZE, seg=12).R(math.pi / 2, 0, 0).T(0, 0, -0.8)]
    for k in range(3):
        a = 2 * math.pi * k / 3
        P.append(box(r * 0.95, 0.16, 1.2, BRONZE, bottom=False).T(r * 0.5 + 0.3, 0, 0).R(0.35, 0, 0).R(0, 0, a))
    return merge(P)


# ── Superstructure layout (distance from the bow d0..d1, width, level 0..3) ──
BLOCKS = [(80, 104, 17, 0), (104, 148, 21, 0), (148, 170, 13, 0),
          (83, 102, 13, 1), (108, 140, 10, 1), (150, 166, 9, 1),
          (86, 100, 10, 2), (88, 98, 8, 3)]
FLAK = [(112, 10.0), (120, 10.0), (128, 10.0), (136, 10.0), (152, 8.5), (160, 8.5)]   # 4-inch twin HA mounts on level 0
LIGHT = [(106, 6.5, S[1]), (143, 7.5, S[0]), (166, 5.5, S[1])]                      # 2-pdr «pom-poms»
FUNNELS = (112.0, 133.0)
FOREMAST_D, MAINMAST_D, CT_D = 92.0, 148.0, 82.0


def turret_base_y(d, raised):
    return deck_y(d) + (3.6 if raised else 0.8)


TURRETS = [("a", A_T, False, False), ("b", B_T, True, True), ("x", X_T, True, True), ("y", Y_T, False, False)]


def tripod(h, col):
    """A tripod mast: the main leg and two raked struts, base at the origin."""
    return merge([cyl(0.75, 0.55, h, col, seg=10), tube([(0, 0, -6.0), (0, h * 0.75, 0)], 0.38, col, seg=6),
                  tube([(-3.5, 0, 2.5), (0, h * 0.75, 0)], 0.38, col, seg=6), tube([(3.5, 0, 2.5), (0, h * 0.75, 0)], 0.38, col, seg=6)])


def ship_part(d0, d1, built=False):
    """Hull and everything fixed on it between d0 and d1. Returns (hull parts, superstructure parts)."""
    H = [hull(d0, d1)]
    U = []
    inr = lambda d, m=0: d0 + m <= d <= d1 - m
    for name, d, raised, rf in TURRETS:
        if not inr(d, 4):
            continue
        y0 = turret_base_y(d, raised)
        H.append(at(cyl(5.4, 5.4, y0 - deck_y(d) + 0.4, GREY, seg=24, caps=False), d, y=deck_y(d) - 0.4))
        if not built:
            H.append(at(cyl(5.1, 5.1, 0.05, DARK, seg=24), d, y=y0 - 0.25))
            H.append(at(cyl(5.55, 5.55, 0.35, MACH, seg=24, caps=False), d, y=y0 - 0.3))
    if d0 <= 1:                                                                # forecastle: breakwater, capstans, anchors
        for s in (-1, 1):
            H.append(at(box(11.0, 1.3, 0.3, GREY).R(0, s * 0.42, 0), 40.0, x=s * 5.0, y=deck_y(40) - 0.1))
            H.append(at(cyl(0.8, 0.7, 1.0, DGREY, seg=12), 20, x=s * 3.0, y=deck_y(20)))
            z = zd(9.0); y = deck_y(9.0) - 4.5
            x = s * (hw(z, y) + 0.3)
            H.append(box(0.5, 3.4, 0.9, DGREY).T(x, y - 1.5, z))
            H.append(box(0.55, 0.6, 2.8, DGREY).T(x, y - 1.6, z))
    for y, a, b, sp in ((12.5, 8, 60, 2.2), (15.5, 6, 75, 2.2), (12.5, 180, 255, 2.4)):     # portholes
        for d in np.arange(max(a, d0 + 1), min(b, d1 - 1), sp):
            z = zd(d)
            if y > deck_y(d) - 0.8:
                continue
            x = hw(z, y)
            for s in (-1, 1):
                H.append(quad(s * x, y, z, 0.4, 0.4, "+x" if s > 0 else "-x", DARK, eps=0.05))
    for x, d in ((3.0, 240.0), (-3.0, 240.0), (8.5, 228.0), (-8.5, 228.0)):     # four propellers
        if inr(d, 1):
            H.append(propeller(2.3).T(x, 3.3, zd(d)))
            H.append(tube([(x * 0.85, 5.5, zd(d - 20)), (x, 3.3, zd(d - 1.0))], 0.3, GREY, seg=8))
    if built and inr(252, 1):
        H.append(box(0.8, 6.5, 5.0, RED).T(0, 2.0, zd(252)))
    if built:
        for s in (-1, 1):
            pts = [(s * (hw(zd(d), deck_y(d)) - 0.2), deck_y(d) + 1.0, zd(d)) for d in np.arange(max(2, d0), min(260, d1), 3.0)]
            if len(pts) > 1:
                H.append(tube(pts, 0.04, GREY, seg=4))
    for a, b, wd, lvl in BLOCKS:                                               # superstructure
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 < 1 or (not built and lvl == 3):
            continue
        y0 = S0 + lvl * LV + (deck_y(a2) - D)
        U.append(at(box(wd, LV, b2 - a2, GREY), (a2 + b2) / 2, y=y0 - (0.6 if lvl == 0 else 0)))
        for d in np.arange(a2 + 1.5, b2 - 1, 2.6):
            for s in (-1, 1):
                if rng.random() < 0.5:
                    U.append(quad(s * wd / 2, y0 + 1.6, zd(d), 0.45, 0.45, "+x" if s > 0 else "-x", DARK))
        if lvl >= 2:
            for x in np.arange(-wd / 2 + 1.0, wd / 2 - 0.8, 1.4):
                U.append(quad(x, y0 + 1.7, zd(a2), 1.0, 0.8, "+z", GLASS))
    base = lambda d: S0 + (deck_y(d) - D)
    if inr(CT_D, 2):                                                           # the armoured conning tower
        U.append(at(cyl(3.0, 2.9, 2 * LV + 1.0, DGREY, seg=18, sx=1.2), CT_D, y=base(CT_D) + LV))
    if inr(FOREMAST_D, 2):                                                     # tripod foremast with the spotting top
        if built:
            U.append(at(tripod(24.0, GREY), FOREMAST_D, y=base(FOREMAST_D) + 2 * LV))
            U.append(at(box(6.0, 3.5, 6.0, GREY), FOREMAST_D, y=base(FOREMAST_D) + 2 * LV + 22.0))
            U.append(at(cupola(2.2, 2.0, rf=9.0, radar=False), FOREMAST_D - 6.0, y=base(FOREMAST_D) + 4 * LV))
        else:
            stub = tripod(5.0, GREY); crumple(stub, 0.6, 0.6, 7, sstep(2.0, 5.0, stub.v[:, 1]))
            U.append(at(stub, FOREMAST_D, y=base(FOREMAST_D) + 2 * LV))
    for k, d in enumerate(FUNNELS):                                            # two funnels
        if not inr(d, 3):
            continue
        if built:
            U.append(at(cyl(4.2, 4.1, 13.0, GREY, seg=24, sx=0.7), d, y=base(d) + LV))
            U.append(at(cyl(4.15, 4.1, 1.2, BLACK, seg=24, sx=0.7), d, y=base(d) + LV + 13.0))
        else:
            stub = cyl(4.2, 4.15, 1.6, GREY, seg=24, sx=0.7, caps=False)
            crumple(stub, 0.5, 0.6, 9 + k, sstep(0.5, 1.6, stub.v[:, 1]))
            U += [at(stub, d, y=base(d) + LV), at(cyl(3.8, 3.8, 0.05, DARK, seg=24, sx=0.7), d, y=base(d) + LV + 0.3)]
        for s in (-1, 1):
            if built:
                U.append(boat((0.7, 0.72, 0.72)).T(s * 7.0, base(d) + LV + 0.9, zd(d + 8)))
    if inr(MAINMAST_D, 2):
        if built:
            U.append(at(tripod(20.0, GREY), MAINMAST_D, y=base(MAINMAST_D) + LV))
            U.append(at(director(), MAINMAST_D + 6, y=base(MAINMAST_D) + LV))
        else:
            stub = tripod(3.0, GREY); crumple(stub, 0.5, 0.6, 19, sstep(1.0, 3.0, stub.v[:, 1]))
            U.append(at(stub, MAINMAST_D, y=base(MAINMAST_D) + LV))
    for d, x in FLAK:
        if not inr(d, 2):
            continue
        for s in (-1, 1):
            el = 0.45 if built else rng.uniform(0.0, 0.9)
            U.append(at(flak105(el).R(0, 0.0 if built else rng.uniform(-1.2, 1.2), 0), d, x=s * x, y=base(d)))
    for d, x, y in LIGHT:
        if not inr(d, 2):
            continue
        for s in (-1, 1):
            U.append(at(flak37().R(0, 0 if built else rng.uniform(-1, 1), 0), d, x=s * x, y=y + (deck_y(d) - D)))
    return H, U


def turrets_built():
    out = {}
    for name, d, raised, rf in TURRETS:
        t = main_turret(rf, 0.0)
        if d > 150:
            t.R(0, math.pi, 0)                                                 # X and Y face aft
        out[name] = at(t, d, y=turret_base_y(d, raised))
    return out


# ── The wreck ───────────────────────────────────────────────────────────────
# The magazine explosion destroyed the after part of the ship (X and Y); what is left: the bow (sheared off just
# ahead of A turret) on its port side, the long midships section upside down in a crater, a small piece of the
# stern standing at an angle, the conning tower alone; two debris fields (east and west).
def section(d0, d1, seed):
    Hh, Uu = ship_part(d0, d1)
    m = merge(Hh + Uu)
    dp = L / 2 - m.v[:, 2]
    crumple(m, 2.6, 0.13, seed, sstep(d0 + 10, d0, dp) + sstep(d1 - 10, d1, dp))
    crumple(m, 3.4, 0.35, seed + 1, sstep(d0 + 4, d0, dp) + sstep(d1 - 4, d1, dp))
    crumple(m, 0.25, 0.08, seed + 2)
    m.T(0, 0, -(zd(d0) + zd(d1)) / 2)
    return m


r = np.random.default_rng(1941)
H_ = 0.0                                                                       # file +Z = north (no single obvious heading)


def place(m, east, north, heading):
    a = math.radians(heading)
    m.R(0, -a, 0)
    return m.T(-east, 0, north)                                                # +X is west


mid = section(46, 178, 20)
mid.R(0, 0, math.pi).T(0, D - 3.5, 0)                                          # upside down, sunk into its crater
weather(mid, rust=0.5, sediment=0.75, streaks=0.5, seed=21)
mid = place(mid, 0, 0, 200)
bow = section(0, 44, 30)
bow.R(0, 0, math.radians(-85))                                                 # on its port side (+x down)
bow.T(0, -bow.v[:, 1].min() - 3.5, 0)
weather(bow, rust=0.5, sediment=0.6, streaks=0.6, seed=31)
bow = place(bow, 330, 520, 70)
stern = section(232, L, 40)
stern.R(math.radians(35), 0, 0)                                                # rising from the bottom at an angle
stern.T(0, -stern.v[:, 1].min() - 6.0, 0)
weather(stern, rust=0.5, sediment=0.6, streaks=0.6, seed=41)
stern = place(stern, 420, 430, 250)
ct = merge([cyl(3.0, 2.9, 2 * LV + 1.0, DGREY, seg=18, sx=1.2), box(5.0, 0.6, 5.0, GREY, bottom=False)])
crumple(ct, 0.4, 0.3, 51)
ct = place(ct.R(math.pi, 0, 0).T(0, 2 * LV, 0), -260, 340, 30)               # upside down, alone
# A and B turrets, fallen out of the inverted midships, upside down near it
tur = []
for k, (e, n_, h_) in enumerate(((-60, 60, 30), (55, -70, 160))):
    t = merge([main_turret(k == 1, 0.2), turret_machinery(7.0, 60 + k)])
    t.R(0, 0, math.pi).T(0, 3.15 - 1.5, 0)
    crumple(t, 0.4, 0.15, 62 + k)
    tur.append(place(t, e, n_, h_))
turrets_w = merge(tur)
# the crater around the midships section: a raised rim of thrown-up mud
crater = grid(-120, 120, -120, 120, 60, 60,
              lambda X, Z: 2.6 * np.exp(-((np.hypot(X, Z) - 85) / 18) ** 2) * (0.7 + 0.6 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.04, 5).reshape(X.shape)) - 0.4,
              MUD)
deb = []
for field, (ce, cn, sp, n) in enumerate(((360, 470, 90, 160), (-240, 330, 110, 140))):        # east and west debris fields
    for i in range(n):
        e, nn = r.normal(ce, sp), r.normal(cn, sp * 0.7)
        k = r.random()
        if k < 0.55:
            m = box(r.uniform(2, 9), 0.18, r.uniform(1.5, 5), GREY if r.random() < 0.7 else (RED if r.random() < 0.5 else TEAK))
            crumple(m, 0.6, 0.3, 1000 * field + i)
            m.R(r.uniform(-0.35, 0.35), 0, r.uniform(-0.35, 0.35))
        elif k < 0.7:
            m = box(0.4, 0.6, r.uniform(4, 12), (0.22, 0.16, 0.11))
        elif k < 0.8:
            m = flak105(r.uniform(0, 0.9)).R(r.uniform(-1.4, 1.4), 0, r.uniform(-1.4, 1.4)).T(0, 0.6, 0)
        elif k < 0.86:
            m = director().R(r.uniform(-1.5, 1.5), 0, 0)
        else:
            s_ = r.uniform(0.3, 1.4); m = box(s_, s_ * r.uniform(0.4, 1.0), s_ * r.uniform(0.6, 1.6), (0.25, 0.2, 0.15))
        deb.append(place(m, e, nn, r.uniform(0, 360)))
for k, (e, n_) in enumerate(((395, 445), (440, 470))):                         # propellers in the eastern field
    deb.append(place(propeller(2.3).R(1.3, 0, 0).T(0, 1.2, 0), e, n_, r.uniform(0, 360)))
fdir = r.uniform(0, 360)
f1 = cyl(4.2, 4.1, 13.0, GREY, seg=24, sx=0.7, caps=False); crumple(f1, 1.0, 0.2, 77); f1.v[:, 0] *= 1.3
deb.append(place(f1.R(math.pi / 2, 0, 0).T(0, 2.6, 0), -200, 260, fdir))                 # a funnel in the western field
debris = merge(deb)
weather(debris, rust=0.5, sediment=0.6, streaks=0.4, seed=61)

# ── The battlecruiser as she sailed (model_built.glb): whole, clean, keel on y = 0, centred ──
rng = np.random.default_rng(1940)
Hb, Ub = ship_part(0, L, built=True)
bturrets = turrets_built()
ship = merge(Hb + Ub + list(bturrets.values()))
ship.c = np.clip(ship.c * (0.95 + 0.05 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q220239" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("midships_section", mid), ("bow_section", bow), ("stern_section", stern), ("conning_tower", ct),
                             ("main_turrets", turrets_w), ("debris_field", debris)], "mud": [crater]}],
            depth=-2838, kind="warship", seed=1941)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[TEAK])
