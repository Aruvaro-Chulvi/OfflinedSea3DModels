"""
Battleship Bismarck (Q150857) — the wreck as it lies today, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: Bismarck-class battleship of 1940, length 250.5 m overall (241.55 m at the waterline), beam 36 m,
depth 15 m to the upper deck amidships, draught 9.3 m, clipper bow, a deck sheer rising towards the bow.
Four twin 38 cm turrets — Anton and Bruno forward, Caesar and Dora aft (frames 192.55, 174.35, 64.35, 46.15,
frames counted in metres from the aft perpendicular), barrels 19.6 m long; six twin 15 cm turrets (three a
side), eight twin 10.5 cm flak mounts (four a side), 3.7 cm twin mounts; the armoured conning tower with the
forward fire-control cupola, the tower mast with the foretop cupola, one large funnel with a silver-grey cap,
single aircraft hangars abreast the funnel, a double-ended catapult across the ship, the double hangar under the
mainmast, the after fire-control cupola, two aircraft cranes, Arado Ar 196 floatplanes; three propellers and
two rudders.
The wreck, after the published surveys (Ballard 1989; the 2001 and 2002 expeditions; Cameron et al. 2002):
she lies upright at about 4,790 m on the flank of an extinct seamount, about two-thirds of the way down a slope
she slid down after hitting it bow first, which started a landslide some 2 km long; the hull sits deep in the
sediment and its bow is nearly buried in the avalanche debris; the stern broke away at frame 10.5 (the
after armoured bulkhead) and has not been found; the four main turrets fell out when she capsized and lie upside
down on the bottom; the secondary turrets and most flak guns are still in place; the conning towers and the
bridge are heavily damaged but attached; the foremast top, mainmast, funnel and rangefinders lie in the
debris field around the hull; shell holes in the superstructure and above the waterline, long gashes along
the lower edge of the armour belt; teak decks and paint mostly survive at that depth.
The heading, the direction of the slope, the places of the turrets and of the debris are approximate.

    python model_sources/Q150857_bismarck.py      -> models/item/Q150857/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, weather, sstep, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1941)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B = 250.5, 36.0
D = 15.0                  # keel to upper deck amidships
WL = 9.3                  # draught
FB = 247.0                # frame number at the bow tip (frames are metres from the aft perpendicular)
LV = 2.7                  # height of one superstructure deck
S0 = D                    # base of the superstructure
S = [S0 + LV * (i + 1) for i in range(4)]        # tops of superstructure levels 1..4
GREY, DGREY, BLACK, RED = (0.6, 0.62, 0.63), (0.36, 0.38, 0.4), (0.08, 0.08, 0.09), (0.36, 0.13, 0.1)
TEAK, STEEL, SILVER, DARK = (0.52, 0.42, 0.3), (0.42, 0.43, 0.45), (0.74, 0.75, 0.76), (0.025, 0.025, 0.03)
BRONZE, MACH, GLASS, PLANE = (0.55, 0.4, 0.2), (0.2, 0.18, 0.16), (0.1, 0.12, 0.14), (0.32, 0.37, 0.32)
MUD = (0.55, 0.52, 0.45)


def fr(frame):          # frame number -> distance from the bow
    return FB - frame


ANTON, BRUNO, CAESAR, DORA = fr(192.55), fr(174.35), fr(64.35), fr(46.15)
BREAK = fr(10.5)        # where the stern broke away (the after armoured bulkhead)


def zd(d):              # distance from the bow (m) -> ship coordinate z (bow at +L/2)
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


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


def yd_of(t):           # upper deck, with the sheer rising to the bow
    y = D
    if t > 0.55:
        y += 5.5 * ((t - 0.55) / 0.45) ** 2
    if t < 0.25:
        y += 0.8 * ((0.25 - t) / 0.25) ** 2
    return y


def k_of(t):            # section fullness: boxy amidships (CM 0.97), V-shaped and flared at the ends
    return 1.8 + 10.0 * w_of(t) ** 4


def hw(z, y):
    t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    if y >= yd:
        return B / 2 * w
    u = max(0.0, (y - yb) / max(0.1, yd - yb))
    return B / 2 * w * (1 - (1 - u) ** k)


def deck_y(d):
    return yd_of(t_of(zd(d)))


def at(m, d, x=0.0, y=0.0):
    return m.T(x, y, zd(d))


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
BLOCKS = [(78, 100, 20, 0), (100, 162, 27, 0), (162, 174, 17, 0),
          (81, 99, 15, 1), (103, 131, 14, 1), (150, 168, 12, 1),
          (84, 97, 11, 2), (153, 165, 8, 2),
          (87, 96, 9, 3)]
SEC = [(86, 12.5, S0, 0, False), (104, 11.0, S[0], 0, True), (165, 12.5, S0, math.pi, False)]      # 15 cm turrets (d, |x|, base y, yaw, rf)
FLAK = [(112, 12.0), (124, 12.0), (147, 11.5), (156, 11.5)]                                          # 10.5 cm mounts on level 1
LIGHT = [(96, 8.8, S[0]), (133, 11.5, S[0]), (170, 7.0, S[0]), (108, 6.0, S[1])]                    # 3.7 cm twins
FUNNEL_D, MAST_D, TOWER_D, CT_D, CAT_D, AFT_D = 117.0, 145.0, 93.0, 81.0, 133.0, 158.0


def turret_base_y(d, raised):
    return deck_y(d) + (4.0 if raised else 0.8)


TURRETS = [("anton", ANTON, False, False), ("bruno", BRUNO, True, True), ("caesar", CAESAR, True, True), ("dora", DORA, False, True)]


def ship_part(d0, d1, built=False):
    """Hull and everything fixed on it between d0 and d1. Returns (hull parts, superstructure parts).
    built=True: as she sailed; built=False: what is left on the wreck."""
    H = [hull(d0, d1)]
    U = []
    inr = lambda d, m=0: d0 + m <= d <= d1 - m
    # barbettes (with the turrets as built; on the wreck: empty, dark rings)
    for name, d, raised, rf in TURRETS:
        y0 = turret_base_y(d, raised)
        H.append(at(cyl(5.6, 5.6, y0 - deck_y(d) + 0.4, GREY, seg=24, caps=False), d, y=deck_y(d) - 0.4))
        if not built:
            H.append(at(cyl(5.3, 5.3, 0.05, DARK, seg=24), d, y=y0 - 0.25))
            H.append(at(cyl(5.75, 5.75, 0.35, MACH, seg=24, caps=False), d, y=y0 - 0.3))           # roller path, torn
    # forecastle: breakwater, capstans, windlass, bollards, anchors and chains
    if d0 <= 1:
        for s in (-1, 1):
            H.append(at(box(13.0, 1.4, 0.3, GREY).R(0, s * 0.42, 0), 41.0, x=s * 5.8, y=deck_y(41) - 0.1))
            H.append(at(cyl(0.8, 0.7, 1.0, DGREY, seg=12), 20, x=s * 3.2, y=deck_y(20)))
            H.append(at(cyl(0.6, 0.5, 0.9, DGREY, seg=10), 27, x=s * 2.5, y=deck_y(27)))
            for d in (6, 14, 33):
                x = s * (hw(zd(d), deck_y(d)) - 1.4)
                H += [at(cyl(0.3, 0.3, 0.8, DGREY, seg=6), d, x=x, y=deck_y(d)), at(cyl(0.3, 0.3, 0.8, DGREY, seg=6), d + 0.8, x=x, y=deck_y(d))]
            z = zd(9.0); y = deck_y(9.0) - 4.2
            x = s * (hw(z, y) + 0.3)
            H.append(box(0.5, 3.4, 0.9, DGREY).T(x, y - 1.5, z))                                  # anchors in the hawse
            H.append(box(0.55, 0.6, 2.8, DGREY).T(x, y - 1.6, z))
            if built or s > 0:                                                                     # chains to the windlass
                H.append(tube([(s * 3.2, deck_y(20) + 0.4, zd(20)), (s * 3.0, deck_y(13) + 0.2, zd(13)),
                               (s * (hw(zd(8.5), deck_y(8.5)) - 0.9), deck_y(8.5) + 0.1, zd(8.5))], 0.2, (0.2, 0.2, 0.21), seg=5))
    # portholes forward and aft of the belt
    for y, a, b, sp in ((12.8, 10, 46, 2.2), (16.6, 6, 40, 2.2), (12.6, 205, 246, 2.4)):
        for d in np.arange(max(a, d0 + 1), min(b, d1 - 1), sp):
            z = zd(d)
            if y > deck_y(d) - 0.8:
                continue
            x = hw(z, y)
            for s in (-1, 1):
                H.append(quad(s * x, y, z, 0.4, 0.4, "+x" if s > 0 else "-x", DARK, eps=0.05))
    # quarterdeck: capstans, bollards, (as built) the flagstaff
    for d, x in ((228, 3.5), (228, -3.5)):
        if inr(d, 1):
            H.append(at(cyl(0.8, 0.7, 1.0, DGREY, seg=12), d, x=x, y=deck_y(d)))
    for d in (214, 232):
        for s in (-1, 1):
            if inr(d, 1):
                x = s * (hw(zd(d), deck_y(d)) - 1.4)
                H += [at(cyl(0.3, 0.3, 0.8, DGREY, seg=6), d, x=x, y=deck_y(d)), at(cyl(0.3, 0.3, 0.8, DGREY, seg=6), d + 0.8, x=x, y=deck_y(d))]
    if built:
        H.append(at(cyl(0.12, 0.06, 7.0, DGREY, seg=6), 249.0, y=deck_y(249)))
        H.append(at(cyl(0.12, 0.06, 6.0, DGREY, seg=6), 1.5, y=deck_y(1.5)))                       # jackstaff
        for s in (-1, 1):                                                                          # guard rails
            pts = [(s * (hw(zd(d), deck_y(d)) - 0.2), deck_y(d) + 1.0, zd(d)) for d in np.arange(2, 249, 3.0)]
            H.append(tube(pts, 0.04, GREY, seg=4))
    # propellers, shaft brackets and, as built, the two rudders
    for x, d in ((0.0, 232.0), (7.6, 220.0), (-7.6, 220.0)):
        y = 3.4
        U_ = propeller().T(x, y, zd(d))
        H.append(U_)
        H.append(tube([(x * 0.8, 5.5, zd(d - 22)), (x, y, zd(d - 1.0))], 0.32, GREY, seg=8))
        H.append(box(0.35, 4.5, 1.2, GREY).T(x, y, zd(d - 1.6)))
    if built:
        for x in (-2.8, 2.8):
            H.append(box(0.7, 6.0, 4.6, RED).T(x, 1.8, zd(242.5)))
    # superstructure blocks with portholes and, on the wreck, shell holes
    for a, b, wd, lvl in BLOCKS:
        if not built and lvl == 3:
            continue                                                                               # the upper bridge: gone
        y0 = S0 + lvl * LV
        U.append(at(box(wd, LV, b - a, GREY), (a + b) / 2, y=y0 - (0.6 if lvl == 0 else 0)))
        for d in np.arange(a + 1.5, b - 1, 2.6):
            for s in (-1, 1):
                if rng.random() < 0.5:
                    U.append(quad(s * wd / 2, y0 + 1.6, zd(d), 0.45, 0.45, "+x" if s > 0 else "-x", DARK))
        if lvl >= 2:
            for x in np.arange(-wd / 2 + 1.0, wd / 2 - 0.8, 1.4):
                U.append(quad(x, y0 + 1.7, zd(a), 1.0, 0.8, "+z", GLASS))
        if not built:
            for i in range(int((b - a) / 9) + 1):
                s = rng.choice([-1, 1]); sz = rng.uniform(0.8, 2.6)
                U.append(quad(s * wd / 2, y0 + rng.uniform(0.8, 1.9), zd(rng.uniform(a + 1, b - 1)), sz, sz * rng.uniform(0.5, 1.0), "+x" if s > 0 else "-x", DARK, eps=0.06))
    # forward armoured conning tower with the forward fire-control cupola
    U.append(at(cyl(3.4, 3.3, S[2] + 0.8 - S[0], DGREY, seg=20, sx=1.15), CT_D, y=S[0]))
    for k in range(10):
        a_ = -0.9 + 1.8 * k / 9
        U.append(quad(0, 0, 0, 0.5, 0.25, "+z", DARK, eps=0.0).R(0, a_, 0).T(3.9 * math.sin(a_), S[2] - 0.4, zd(CT_D) + 3.35 * math.cos(a_)))
    if built:
        U.append(at(cupola(2.1, 1.9), CT_D, y=S[2] + 0.8))
    else:
        U.append(at(cupola(2.1, 1.9, rf=7.0, radar=False), CT_D, y=S[2] + 0.8))                   # the arms broken short
    # tower mast and foretop
    if built:
        U.append(at(cyl(2.7, 2.5, 35.5 - S[2], GREY, seg=18), TOWER_D, y=S[2]))
        U.append(at(cyl(4.2, 4.2, 0.3, GREY, seg=18), TOWER_D, y=31.0))
        for s in (-1, 1):
            U.append(at(searchlight(), TOWER_D, x=s * 3.3, y=31.3))
            U.append(at(box(1.4, 1.1, 1.4, GREY), TOWER_D - 2.5, x=s * 3.0, y=31.3))           # 2 cm quadruple mounts
        U.append(at(cyl(3.8, 3.8, 0.35, GREY, seg=18), TOWER_D, y=35.5))
        U.append(at(cupola(2.3, 2.2), TOWER_D, y=35.85))
        U.append(at(cyl(0.35, 0.15, 12.0, DGREY, seg=8), TOWER_D + 0.5, y=38.0))
        U.append(tube([(-4.0, 46.0, zd(TOWER_D + 0.5)), (4.0, 46.0, zd(TOWER_D + 0.5))], 0.1, DGREY, seg=4))
    else:
        stub = cyl(2.7, 2.6, 29.5 - S[2], GREY, seg=18, caps=False)
        crumple(stub, 0.9, 0.5, 7, sstep(3.0, 6.5, stub.v[:, 1]))
        U.append(at(stub, TOWER_D, y=S[2]))
    for s in (-1, 1):
        U.append(at(director(), 98.5, x=s * 4.2, y=S[1]))
    # funnel with the searchlight platform and silver-grey cap
    if built:
        U.append(at(cyl(5.0, 4.9, 33.5 - S[1] - 1.6, GREY, seg=26, sx=0.68), FUNNEL_D, y=S[1]))
        U.append(at(cyl(4.9, 4.8, 1.6, SILVER, seg=26, sx=0.68), FUNNEL_D, y=33.5 - 1.6))
        U.append(at(cyl(4.3, 4.3, 0.05, DARK, seg=26, sx=0.68), FUNNEL_D, y=33.52))
        U.append(at(cyl(6.6, 6.6, 0.3, GREY, seg=26, sx=0.8), FUNNEL_D, y=27.0))
        for x, dz in ((4.6, -2.5), (-4.6, -2.5), (4.6, 3.0), (-4.6, 3.0)):
            U.append(at(searchlight(), FUNNEL_D + dz, x=x, y=27.3))
    else:
        stub = cyl(5.0, 4.95, 1.8, GREY, seg=26, sx=0.68, caps=False)
        crumple(stub, 0.6, 0.6, 9, sstep(0.6, 1.8, stub.v[:, 1]))
        U += [at(stub, FUNNEL_D, y=S[1]), at(cyl(4.6, 4.6, 0.05, DARK, seg=26, sx=0.68), FUNNEL_D, y=S[1] + 0.3)]
    # single hangars abreast the funnel, aircraft cranes, boats
    for s in (-1, 1):
        U.append(at(box(4.6, 3.6, 13.0, GREY), FUNNEL_D, x=s * 9.5, y=S[0]))
        if not built and s > 0:
            U.append(quad(s * 9.5, S[0] + 3.62, zd(FUNNEL_D + 2), 3.4, 6.0, "+y", DARK, eps=0.0))     # roof blown in
        if built or s < 0:
            U.append(at(cyl(0.9, 0.8, 3.0, GREY, seg=10), 129, x=s * 8.5, y=S[0]))
            U.append(tube([(s * 8.5, S[0] + 3.0, zd(129)), (s * 9.5, S[0] + (11.0 if built else 3.6), zd(129) + (7.0 if built else 0.3))], 0.35, GREY, seg=6))
        if built:
            for d in (122, 129.5):
                U.append(boat((0.7, 0.72, 0.72)).T(s * 4.8, S[1] + 0.9, zd(d)))
    # the catapult across the ship (an Arado on it as built)
    cat = merge([cyl(1.2, 1.2, 2.2, GREY, seg=12), box(32.0, 0.9, 1.7, GREY).T(0, 2.2, 0)])
    if not built:
        crumple(cat, 0.5, 0.15, 17, sstep(4.0, 16.0, np.abs(cat.v[:, 0])))
    U.append(at(cat, CAT_D, y=S[0]))
    # double hangar and the mainmast on it
    U.append(at(box(13.0, 3.8, 12.0, GREY), MAST_D - 2.0, y=S[0]))
    U.append(at(director(), MAST_D + 2.5, y=S[0] + 3.8))
    if built:
        U.append(at(cyl(0.7, 0.3, 41.0 - S[0] - 3.8, DGREY, seg=10), MAST_D - 3.5, y=S[0] + 3.8))
        U.append(tube([(-5.0, 37.0, zd(MAST_D - 3.5)), (5.0, 37.0, zd(MAST_D - 3.5))], 0.1, DGREY, seg=4))
    else:
        stub = cyl(0.7, 0.65, 1.4, DGREY, seg=10)
        crumple(stub, 0.25, 0.8, 19, sstep(0.5, 1.4, stub.v[:, 1]))
        U.append(at(stub, MAST_D - 3.5, y=S[0] + 3.8))
        U.append(quad(4.0, S[0] + 3.82, zd(MAST_D - 6), 4.0, 3.5, "+y", DARK, eps=0.0))
    # after fire-control post
    U.append(at(cyl(2.8, 2.7, 2.0, DGREY, seg=18), AFT_D, y=S[2] - 0.1))
    if built:
        U.append(at(cupola(2.1, 1.9), AFT_D, y=S[2] + 1.9))
    U.append(at(director(), 171.0, y=S[0]))
    # secondary turrets, flak, light flak
    for d, x, y, yaw, rf in SEC:
        for s in (-1, 1):
            el = 0.05 if built else rng.uniform(-0.05, 0.25)
            U.append(at(sec_turret(rf, el).R(0, yaw + (0 if built else rng.uniform(-0.5, 0.5)), 0), d, x=s * x, y=y))
    for d, x in FLAK:
        for s in (-1, 1):
            el = 0.45 if built else rng.uniform(0.0, 0.9)
            yaw = 0.0 if built else -s * rng.uniform(0.6, 1.4)                         # turned inboard, towards each other
            U.append(at(flak105(el).R(0, yaw, 0), d, x=s * x, y=S[0]))
    for d, x, y in LIGHT:
        for s in (-1, 1):
            U.append(at(flak37().R(0, 0 if built else rng.uniform(-1, 1), 0), d, x=s * x, y=y))
    return H, U


def turrets_built():
    out = {}
    for name, d, raised, rf in TURRETS:
        t = main_turret(rf, 0.0)
        if d > 150:
            t.R(0, math.pi, 0)                                                         # Caesar and Dora face aft
        out[name] = at(t, d, y=turret_base_y(d, raised))
    return out


# ── The wreck: hull and what is left on it ───────────────────────────────────
H, U = ship_part(0, BREAK)
hullm = merge(H)
sup = merge(U)
dfb = L / 2 - sup.v[:, 2]
crumple(sup, 0.45, 0.25, 21, sstep(S0 + 1, S0 + 12, sup.v[:, 1]))          # dented, shell-torn superstructure
bridge = (dfb > 83.5) & (dfb < 98) & (sup.v[:, 1] > S[1]) & (sup.v[:, 1] < S[2] + 1.0) & \
         (np.hypot(sup.v[:, 0], sup.v[:, 2] - zd(TOWER_D)) > 3.0)
sup.v[bridge, 1] = S[1] + (sup.v[bridge, 1] - S[1]) * 0.55                    # the bridge crushed
# shell holes above the waterline (seven on the port side = +X, one on the starboard side), the bow deck hole with
# the starboard anchor chain fallen through it, the hole in the deck abreast the catapult, gashes along the belt
marks = []
for d, y, s, sz in ((96, 12.5, 1, 1.6), (104, 11.2, 1, 1.0), (112, 13.6, 1, 1.4), (121, 12.0, 1, 2.0), (133, 10.8, 1, 1.2),
                    (146, 13.0, 1, 1.5), (60, 13.8, 1, 1.1), (114, 12.4, -1, 1.3)):
    marks.append(quad(s * hw(zd(d), y), y, zd(d), sz, sz * 0.8, "+x" if s > 0 else "-x", DARK, eps=0.07))
marks.append(quad(-4.5, deck_y(24) + 0.03, zd(24), 3.0, 4.5, "+y", DARK, eps=0.0))
marks.append(tube([(-3.2, deck_y(20) + 0.4, zd(20)), (-4.0, deck_y(22) + 0.3, zd(22)), (-4.5, deck_y(24) - 1.0, zd(24))], 0.2, (0.2, 0.2, 0.21), seg=5))
marks.append(quad(15.0, D + 0.03, zd(CAT_D), 3.0, 5.0, "+y", DARK, eps=0.0))
for d, ln, s in ((70, 22, 1), (118, 30, 1), (160, 18, 1), (90, 26, -1), (140, 34, -1), (196, 14, -1)):
    y = 6.8
    marks.append(quad(s * hw(zd(d), y), y, zd(d), ln, 0.5, "+x" if s > 0 else "-x", DARK, eps=0.06))
# the torn end at frame 10.5: the after armoured bulkhead and plates bent backwards
zb = zd(BREAK)
tb = t_of(zb)
sec = [(s_ * B / 2 * w_of(tb) * (1 - (1 - u) ** k_of(tb)) * 0.97, yb_of(tb) + (yd_of(tb) - yb_of(tb)) * u, zb + 0.5)
       for s_, us in ((-1, np.linspace(1, 0, 12)), (1, np.linspace(0, 1, 12)[1:])) for u in us]
torn = [Mesh(sec + [(0, (yb_of(tb) + yd_of(tb)) / 2, zb + 0.5)], [(len(sec), i, i + 1) for i in range(len(sec) - 1)], (0.3, 0.29, 0.28))]
for i in range(22):
    y = rng.uniform(3, deck_y(BREAK))
    wdt = hw(zb, y)
    s_ = rng.choice([-1, 1])
    p = box(rng.uniform(1.2, 3.5), 0.1, rng.uniform(1.2, 3.5), GREY if y > WL else (0.3, 0.12, 0.09), bottom=False)
    crumple(p, 0.5, 0.35, 300 + i)
    x = s_ * rng.uniform(0.55, 1.0) * wdt if rng.random() < 0.7 else rng.uniform(-wdt, wdt)        # mostly the shell plating
    torn.append(p.R(rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3), rng.uniform(-0.6, 0.6)).T(x, y, zb - rng.uniform(0.3, 2.2)))
for i in range(10):                                                           # frames and beams sticking out of the break
    y = rng.uniform(4, deck_y(BREAK) - 0.5)
    x = rng.uniform(-0.8, 0.8) * hw(zb, y)
    torn.append(box(0.3, 0.4, rng.uniform(2, 5), MACH).R(rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0).T(x, y, zb - 1.5))
wreck = merge([hullm, sup, merge(marks), merge(torn)])
dfw = L / 2 - wreck.v[:, 2]
crumple(wreck, 2.6, 0.3, 24, sstep(229, BREAK, dfw))                         # ragged edges at the break
crumple(wreck, 0.18, 0.07, 25)                                                # dents everywhere
zc = (zd(0) + zb) / 2
wreck.T(0, 0, -zc)
wreck.R(math.atan(6.5 / BREAK), 0, 0.006)                                     # bow down in the avalanche debris
wreck.T(0, -5.8, 0)                                                           # deep in the sediment, the propellers just clear
weather(wreck, rust=0.4, sediment=0.5, streaks=0.5, seed=31, rust_col=(0.42, 0.27, 0.16))
HALF = (BREAK) / 2                                                            # the hull runs from z = -HALF (break) to +HALF (bow)

# ── Sediment around the hull and the avalanche over the bow ─────────────────


def mud_h(X, Z):
    d = HALF - Z                                                              # distance from the bow along the hull
    hwd = np.array([B / 2 * w_of(t_of(zd(max(0.0, min(L, dd))))) for dd in d.ravel()]).reshape(d.shape)
    side = (2.0 + 3.5 * sstep(120, 10, d)) * np.exp(-((np.abs(X) - hwd - 3.0) / 4.5) ** 2) * sstep(-6, 2, d) * sstep(BREAK + 6, BREAK - 4, d)
    front = 8.0 * np.exp(-(X / 14) ** 2) * np.exp(-((d + 6) / 11) ** 2)        # avalanche debris heaped on the stem
    p = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    bumps = (0.5 * fbm(p, 0.12, 5).reshape(X.shape) - 0.25) * sstep(0.3, 1.5, np.maximum(side, front))
    return np.maximum(side, front) + bumps - 0.6


mud = grid(-48, 48, -HALF - 14, HALF + 40, 40, 110, mud_h, MUD)
mud.paint(lambda p: np.array(MUD) + (fbm(p, 0.12, 9)[:, None] - 0.5) * 0.07)
mud.f = mud.f[mud.v[mud.f][:, :, 1].max(axis=1) > 0.15]
lumps = []
for i in range(60):                                                           # blocks of the avalanche
    x, z = rng.normal(0, 16), HALF + rng.uniform(-4, 30)
    s = rng.uniform(0.8, 3.2)
    m = box(s, s * rng.uniform(0.4, 0.8), s * rng.uniform(0.7, 1.4), MUD, bottom=False)
    crumple(m, s * 0.15, 0.6, 900 + i)
    lumps.append(m.R(rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28), rng.uniform(-0.4, 0.4)).T(x, rng.uniform(0.0, 1.5), z))
mudm = merge([mud, merge(lumps)])
mudm.paint(lambda p: np.array(MUD) * (0.9 + 0.1 * fbm(p, 0.3, 11))[:, None])

# ── The slide track up the slope behind the stern (its lower ~550 m) ────────
Z0, Z1 = -HALF - 10, -HALF - 560


def track_h(X, Z):
    far = sstep(Z0, Z1, Z)                                                    # 0 near the hull, 1 at the far end
    p = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    pz = np.stack([np.zeros(X.size), np.zeros(X.size), Z.ravel()], 1)
    wob = (fbm(pz, 0.006, 3).reshape(X.shape) - 0.5) * 50 * far                # the track wanders a little
    xc = X - wob
    side = np.sign(xc)
    n_l, n_r = fbm(pz, 0.02, 21).reshape(X.shape), fbm(pz, 0.02, 22).reshape(X.shape)
    nn = np.where(side < 0, n_l, n_r)
    wd = 21 + 10 * (np.where(side < 0, fbm(pz, 0.008, 23).reshape(X.shape), fbm(pz, 0.008, 24).reshape(X.shape)) - 0.5) + 6 * far
    h = (2.8 - 1.6 * far) * (0.25 + 1.1 * sstep(0.35, 0.7, nn))                # broken, uneven levees
    ridge = h * np.exp(-((np.abs(xc) - wd) / (4.5 + 3 * far + 3 * (nn - 0.5))) ** 2)
    trough = 1.0 * sstep(0.6, 0.78, fbm(p, 0.035, 25).reshape(X.shape)) * sstep(18, 10, np.abs(xc))   # hummocks in the scar
    ridge = np.maximum(ridge, trough)
    ridge *= sstep(Z1 - 10, Z1 + 90, Z) * sstep(Z0 + 12, Z0 - 6, Z)
    bumps = (fbm(p, 0.09, 4).reshape(X.shape) - 0.5) * 1.0 * sstep(0.3, 1.2, ridge)
    return ridge + bumps - 0.6


track = grid(-75, 75, Z1, Z0 + 12, 44, 130, track_h, MUD)
track.f = track.f[track.v[track.f][:, :, 1].max(axis=1) > 0.15]
clods = []
for i in range(90):
    z = rng.uniform(Z1 + 20, Z0)
    x = rng.choice([-1, 1]) * rng.uniform(5, 32)
    s = rng.uniform(0.6, 2.6)
    m = box(s, s * rng.uniform(0.4, 0.8), s * rng.uniform(0.7, 1.4), MUD, bottom=False)
    crumple(m, s * 0.15, 0.6, 1200 + i)
    clods.append(m.R(rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28), 0).T(x, rng.uniform(-0.2, 0.8), z))
trackm = merge([track, merge(clods)])
trackm.paint(lambda p: np.array(MUD) * (0.88 + 0.12 * fbm(p, 0.2, 13))[:, None])

# ── The four main turrets, upside down on the bottom ────────────────────────
TPOS = {"anton": (72.0, 38.0, 0.6), "bruno": (-78.0, -26.0, 2.1), "caesar": (46.0, -232.0, 4.0), "dora": (-44.0, -385.0, 5.2)}
wturrets = {}
for i, (name, d, raised, rf) in enumerate(TURRETS):
    guns = (True, False) if name == "caesar" else (True, True)                     # Caesar lost a barrel (it lies nearby)
    t = merge([main_turret(rf, rng.uniform(0.0, 0.25), guns), turret_machinery(rng.uniform(5.5, 8.0), 70 + i)])
    crumple(t, 0.25, 0.3, 80 + i)
    crumple(t, 1.3, 0.25, 90 + i, sstep(-1.5, -6.0, t.v[:, 1]))                    # the trunk torn and twisted
    x, z, yaw = TPOS[name]
    t.R(0, 0, math.pi)                                                              # upside down: roof on the mud
    t.R(rng.uniform(-0.12, 0.12), yaw, rng.uniform(-0.1, 0.1))
    t.T(0, 2.3, 0)                                                                  # roof sunk ~1 m into the sediment
    heap = grid(-12, 12, -12, 12, 16, 16, lambda X, Z: 1.4 * np.exp(-((np.hypot(X, Z) - 7.0) / 2.6) ** 2) - 0.5, MUD)
    heap.f = heap.f[heap.v[heap.f][:, :, 1].max(axis=1) > 0.15]
    heap.paint(lambda p: np.array(MUD) + (fbm(p, 0.2, 40 + i)[:, None] - 0.5) * 0.06)
    weather(t, rust=0.4, sediment=0.45, streaks=0.3, seed=50 + i, rust_col=(0.42, 0.27, 0.16))
    wturrets[name] = (merge([t, heap]), [x, 0.0, z])

# ── Debris field ─────────────────────────────────────────────────────────────
deb = []


def spot(near=0.7):
    if rng.random() < near:
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(26, 110)
        return r * math.cos(a), max(-HALF - 40, min(HALF + 40, rng.normal(0, 70)))
    return rng.normal(0, 40), rng.uniform(Z1 + 30, Z0)


# the foretop: the top of the tower mast with its cupola, searchlight platform and pole mast
top = merge([cyl(2.6, 2.6, 6.0, GREY, seg=18), cyl(4.2, 4.2, 0.3, GREY, seg=18).T(0, 1.5, 0), cyl(3.8, 3.8, 0.35, GREY, seg=18).T(0, 6.0, 0),
             cupola(2.3, 2.2).T(0, 6.35, 0), cyl(0.35, 0.15, 12.0, DGREY, seg=8).T(0.5, 8.5, 0)])
crumple(top, 0.6, 0.3, 601)
deb.append(top.R(0, 0, math.pi / 2 - 0.25).R(0, 0.7, 0).T(-52, 2.2, 58))
# the funnel, crushed flat, with its silver-grey cap
fun = merge([cyl(5.0, 4.9, 9.5, GREY, seg=26, sx=0.68, caps=False), cyl(4.9, 4.8, 1.6, SILVER, seg=26, sx=0.68, caps=False).T(0, 9.5, 0)])
fun.S(1, 1, 0.55)
crumple(fun, 1.3, 0.25, 602)
deb.append(fun.R(math.pi / 2, 0, 0).R(0, 1.9, 0).T(58, 2.3, -14))
# the mainmast and its yard
mm = merge([cyl(0.7, 0.3, 20.0, DGREY, seg=10), tube([(-5, 16, 0), (5, 16, 0)], 0.1, DGREY, seg=4)])
crumple(mm, 0.4, 0.2, 603)
deb.append(mm.R(0, 0, math.pi / 2 - 0.05).R(0, -0.8, 0).T(-34, 0.5, -82))
# the after fire-control cupola and a rangefinder arm
deb.append(cupola(2.1, 1.9).R(0.3, 0.9, 1.4).T(33, 1.6, -150))
deb.append(box(10.5, 0.8, 0.9, GREY).R(0, 2.4, 0.1).T(-30, 0.1, 104))
# a crane jib, searchlights, the 38 cm barrel of Caesar
deb.append(tube([(0, 0, 0), (2.0, 0.6, 9.0)], 0.35, GREY, seg=6).R(0, 0.5, 0).T(36, 0.3, 14))
for i in range(4):
    x, z = spot()
    deb.append(searchlight().R(rng.uniform(0, 3), rng.uniform(0, 6.28), 1.4).T(x, 0.6, z))
cx, cz = TPOS["caesar"][0], TPOS["caesar"][1]
deb.append(barrel(0.56, 0.33, 17.0, GREY, seg=10).R(0.02, 2.3, 0).T(cx + 10, 0.2, cz + 7))
# two motor boats, flattened
for i in range(2):
    x, z = spot()
    b_ = boat((0.6, 0.62, 0.62))
    crumple(b_, 0.4, 0.4, 610 + i)
    deb.append(b_.S(1, 0.6, 1).R(0, rng.uniform(0, 6.28), rng.uniform(-0.4, 0.4)).T(x, 0.2, z))
# hull and superstructure plates, splinter shields, beams, smaller pieces
for i in range(95):
    x, z = spot()
    m = box(rng.uniform(1.5, 7), 0.14, rng.uniform(1.2, 4), GREY if rng.random() < 0.7 else (TEAK if rng.random() < 0.5 else DGREY))
    crumple(m, 0.5, 0.3, 100 + i)
    deb.append(m.R(rng.uniform(-0.35, 0.35), rng.uniform(0, 6.28), rng.uniform(-0.35, 0.35)).T(x, 0.1, z))
for i in range(30):
    x, z = spot()
    deb.append(box(0.35, 0.5, rng.uniform(3, 10), STEEL).R(0, rng.uniform(0, 6.28), rng.uniform(-0.2, 0.2)).T(x, -0.05, z))
for i in range(12):                                                         # tangled pipes and cables
    x, z = spot()
    pts = [(x, 0.3, z)]
    for k in range(4):
        px, _, pz = pts[-1]
        pts.append((px + rng.uniform(-3, 3), rng.uniform(0.1, 1.2), pz + rng.uniform(-3, 3)))
    deb.append(tube(pts, rng.uniform(0.08, 0.25), MACH, seg=5))
for i in range(170):
    x, z = spot(0.6)
    s = rng.uniform(0.3, 1.3)
    deb.append(box(s, s * rng.uniform(0.4, 1.0), s * rng.uniform(0.6, 1.6), (0.32, 0.31, 0.3)).R(0, rng.uniform(0, 6.28), 0).T(x, -0.1, z))
debris = merge(deb)
weather(debris, rust=0.35, sediment=0.6, streaks=0.3, seed=61, rust_col=(0.42, 0.27, 0.16))

# ── The ship as she sailed (model_built.glb): whole, clean, keel on y = 0, centred ──
rng = np.random.default_rng(1940)
Hb, Ub = ship_part(0, L, built=True)
bhull, bsup = merge(Hb), merge(Ub)
bturrets = turrets_built()
plane = arado().R(0, math.pi / 2, 0)                                       # on the catapult, ready to launch to port
plane.T(7.0, S[0] + 3.2 + 2.1, zd(CAT_D))


def paint_life(m, seed):
    m.c = np.clip(m.c * (0.94 + 0.06 * fbm(m.v, 0.05, seed)[:, None]), 0, 1)
    return m


# ── Write ────────────────────────────────────────────────────────────────────
# Dressed with tools/dresskit.py: its own sediment, rust, the life of its depth, things from the ship, textures,
# baked light, 16-bit positions and instancing (see docs/MODEL_SPEC.md).
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q150857" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
steel = [("hull", wreck)]
for name in ("anton", "bruno", "caesar", "dora"):                # the turrets fell out when she capsized: where they lie
    m, tr = wturrets[name]
    steel.append((f"turret_{name}", m.copy().T(*tr)))
steel.append(("debris_field", debris))
dress_wreck(out, [{"steel": steel, "mud": [mudm, trackm]}], depth=-4791, kind="warship", seed=1941)
bparts = [("hull", paint_life(bhull, 3)), ("superstructure", paint_life(bsup, 4))]
bparts += [(f"turret_{n}", paint_life(bturrets[n], 5)) for n in ("anton", "bruno", "caesar", "dora")]
bparts.append(("aircraft", plane))
dress_built(out.with_name("model_built.glb"), bparts, deck_colors=[TEAK])
