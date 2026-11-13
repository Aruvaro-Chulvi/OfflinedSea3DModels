"""
Yamato (Q215010) — the wreck of the battleship in the East China Sea, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: length 263.0 m overall (256 m on the waterline), beam 38.9 m, draught about 10.4–11 m, flush deck
with a strong sheer forward and a step down to the aviation deck aft; bulbous bow, four shafts with
three-bladed propellers of about 5 m, a main and an auxiliary rudder. Three triple 46 cm turrets (two
forward, one aft), two triple 15.5 cm turrets on the centreline superfiring over them, the tower bridge,
one big funnel raked aft, the mainmast and the after fire-control tower; as she sailed on her last sortie
(April 1945): twelve twin 12.7 cm mounts on the superstructure and some 150 triple 25 mm mounts, two
catapults and the aircraft crane on the quarterdeck.

The wreck, after the published surveys (1985, 1999 and the Kure city survey of May 2016) at about 345 m:
- the bow section (about 90 m, broken between the barbettes of turrets No. 1 and No. 2) points north-west
  (~310°); upright with a list in 1999, in 2016 it had fallen over on its starboard side; turret No. 1 is
  gone but its barbette is intact; the bulbous bow is cracked and its tip is crumbling; a gap has opened
  between the upper and lower parts of the bow; the chrysanthemum crest is still on the stem (its gold leaf
  peeled off); one anchor is missing; the tower bridge lies crushed under the starboard side;
- the after part (midships and stern, ~170 m) lies capsized, keel up and almost level, running west–east
  with the rudders to the east, less than 50 m from the bow section; the midships part is shattered and
  joined to the stern by only a little structure (the two sank "arched"); a large hole in the port bottom
  shows the shafts and boilers; three of the four propellers are still on, the starboard outer one lies
  stuck in the seabed; both rudders are intact and amidships; the after fire-control tower lies by the
  propellers;
- every main and secondary turret fell out when she capsized: the 2016 survey found the three main turrets
  in the order No. 2, No. 3, No. 1 from north to south, upside down with their barrels in the mud (No. 1
  standing upside down, the best preserved, with its rangefinder; No. 2 badly broken, its handling room
  tilted; No. 3 half buried), the top of No. 2 about 170 m from the bow section; a 15.5 cm turret with its
  middle muzzle burst; heavily damaged midships wreckage as three mounds about 70 m south of the hull;
  main-gun powder cans scattered everywhere.
Headings, the length of each section, the exact spots of the turrets and mounds are approximate.

    python model_sources/Q215010_yamato.py   -> models/item/Q215010/model.glb (+ model_built.glb)

The file is laid out with +Z pointing to 310° (the heading of the bow section, meta.json "heading"), +X to
port of it, the origin under the middle of the capsized after part (the biggest piece). Everything lies
within ~250 m of the origin (the 2016 survey area was 400 × 450 m), so there are no separate "pieces".
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, weather, sstep, fbm, normals, write_glb  # noqa: E402

rng = np.random.default_rng(1945)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B = 263.0, 38.9
D = 18.9                                   # keel to the upper deck amidships
WL = 10.4                                  # waterline (top of the red bottom paint)
STEP = 2.6                                 # the quarterdeck aft is one deck lower
LV = 2.6                                   # one level of the superstructure
GREY, RED, DECKW = (0.43, 0.45, 0.45), (0.38, 0.14, 0.09), (0.55, 0.47, 0.36)
DARK, STEEL, BRONZE = (0.025, 0.025, 0.03), (0.34, 0.35, 0.36), (0.55, 0.4, 0.2)
GOLD, GOLD_OLD = (0.86, 0.66, 0.22), (0.42, 0.33, 0.2)
INSIDE, BOILER = (0.12, 0.1, 0.09), (0.16, 0.13, 0.11)
T1, T2, T3 = 77.0, 97.5, 200.0             # main turrets (distance from the bow)
S1, S2 = 114.0, 183.0                      # 15.5 cm turrets
RB = 7.1                                   # barbette radius of a main turret
RAKE = 7.0                                 # the stem leans forward: the deck at the bow is ~7 m ahead of the forefoot


def zd(d):
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):                                # beamy hull, fine entrance, broad cruiser stern
    w = 1.0
    if t > 0.6:
        w = math.sin(((1 - t) / 0.4) * math.pi / 2) ** 0.95
    if t < 0.2:
        w = min(w, 0.42 + 0.58 * math.sin((t / 0.2) * math.pi / 2))
    return max(w, 0.004)


def yb_of(t):                               # keel line: cut up aft over the propellers and rudders
    y = 0.0
    if t < 0.13:
        y = D * 0.6 * ((0.13 - t) / 0.13) ** 1.3
    if t > 0.99:
        y = max(y, D * 0.6 * ((t - 0.99) / 0.01) ** 2)
    return y


def yd_of(t):                               # upper deck: strong sheer forward, step down to the quarterdeck aft
    y = D
    if t > 0.62:
        y += 5.6 * ((t - 0.62) / 0.38) ** 2
    y -= STEP * sstep(0.205, 0.197, t)
    return float(y)


def k_of(t):
    return 2.4 + 7.5 * w_of(t) ** 4


def hw(z, y):
    t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    if y >= yd:
        return B / 2 * w
    u = max(0.0, (y - yb) / max(0.1, yd - yb))
    return B / 2 * w * (1 - (1 - u) ** k)


def hull(d0, d1, NP=24, step=1.5):
    ns = max(2, int(math.ceil((d1 - d0) / step)))
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
    d0i = len(v)
    for z, w, yb, yd, k in st:
        v += [(-B / 2 * w, yd, z), (B / 2 * w, yd, z)]
    for i in range(ns):
        a = d0i + 2 * i; b = a + 2; c = a + 1; d = b + 1
        f += [(a, b, c), (c, b, d)]
    if d1 >= L - 0.01:                                       # the rounded cruiser stern, closed
        c0 = len(v); z, w, yb, yd, k = st[0]
        v.append((0, (yb + yd) / 2, z))
        f += [(c0, j + 1, j) for j in range(W - 1)]
    m = Mesh(v, f)
    m.paint(lambda p: np.where((p[:, 1] < WL)[:, None], RED, GREY))
    top = np.arange(len(m.v)) >= d0i
    m.c[top] = DECKW
    return m


def at(m, d, x=0.0, y=0.0):
    return m.T(x, y, zd(d))


def deck_y(d):
    return yd_of(t_of(zd(d)))


def hexa(P, c):
    """A box from 8 corners in meshkit's box() order (4 at the rear, 4 at the front)."""
    P = np.asarray(P, float)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 4, 7, 3), (1, 2, 6, 5), (3, 7, 6, 2), (0, 1, 5, 4)]
    v, f = [], []
    for q in quads:
        o = len(v); v += [P[k] for k in q]; f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    return Mesh(v, f, c)


def gunhouse(w0, w1, lb, cut, h, c):
    """Turret gunhouse: wider at the bottom, sloped front face (cut), front towards +Z."""
    zb0, zb1 = -0.55 * lb, 0.45 * lb
    zt0, zt1 = zb0 + 0.3, zb1 - cut
    return hexa([(-w0 / 2, 0, zb0), (w0 / 2, 0, zb0), (w1 / 2, h, zt0), (-w1 / 2, h, zt0),
                 (-w0 / 2, 0, zb1), (w0 / 2, 0, zb1), (w1 / 2, h, zt1), (-w1 / 2, h, zt1)], c), zb1


def barrel(r0, r1, ln, elev, col=GREY, seg=12):
    return cyl(r0, r1, ln, col, seg=seg).R(math.pi / 2 - elev, 0, 0)


def main_turret(elev=0.0, trunk=0.0, col=GREY):
    """46 cm triple turret: gunhouse, 15 m rangefinder across the rear, three 20.7 m guns (~17.5 m outside).
    Origin at the top of the barbette, guns towards +Z. trunk > 0 adds the rotating structure below."""
    gh, zf = gunhouse(13.2, 11.6, 20.5, 3.4, 5.6, col)
    P = [gh, box(15.2, 1.4, 1.6, col).T(0, 4.0, -7.2)]
    for s in (-1, 1):
        P.append(box(1.3, 1.9, 1.9, col).T(s * 7.6, 3.75, -7.2))                # rangefinder hoods
        P.append(box(1.1, 0.7, 1.7, col).T(s * 4.3, 5.6, 1.5))                  # periscope hoods
    for x in (-3.1, 0.0, 3.1):
        P.append(barrel(0.62, 0.42, 17.5, elev, col).T(x, 2.5, zf - 1.2))
        P.append(box(1.9, 2.1, 1.2, (0.3, 0.3, 0.28)).T(x, 1.45, zf - 0.7))     # blast bags
    if trunk > 0:
        P.append(cyl(6.0, 6.0, trunk, col, seg=24).T(0, -trunk, 0))
        P.append(cyl(6.9, 6.9, 0.7, col, seg=24).T(0, -0.7, 0))
        P.append(cyl(4.2, 4.2, 2.2, col, seg=16).T(0, -trunk - 2.2, 0))          # the shell handling room
    return merge(P)


def sec_turret(elev=0.0, trunk=0.0, col=GREY):
    """15.5 cm triple turret (from the Mogami class): smaller gunhouse, three 9.3 m guns."""
    gh, zf = gunhouse(8.4, 7.2, 11.5, 2.0, 3.3, col)
    P = [gh, box(9.0, 0.9, 1.0, col).T(0, 2.3, -3.6)]
    for x in (-1.9, 0.0, 1.9):
        P.append(barrel(0.27, 0.19, 9.3, elev, col, seg=8).T(x, 1.5, zf - 0.6))
    if trunk > 0:
        P.append(cyl(3.6, 3.6, trunk, col, seg=16).T(0, -trunk, 0))
    return merge(P)


def mount127(col=GREY):
    """Twin 12.7 cm mount with its shield."""
    P = [hexa([(-2.0, 0, -2.4), (2.0, 0, -2.4), (1.7, 2.5, -2.2), (-1.7, 2.5, -2.2),
               (-2.0, 0, 2.2), (2.0, 0, 2.2), (1.7, 2.5, 1.0), (-1.7, 2.5, 1.0)], col)]
    for x in (-0.75, 0.75):
        P.append(barrel(0.15, 0.11, 5.0, 0.2, col, seg=6).T(x, 1.4, 1.8))
    return merge(P)


def mount25(col=GREY):
    """Triple 25 mm mount."""
    P = [cyl(0.7, 0.7, 0.6, col, seg=8), box(1.5, 0.9, 1.4, col).T(0, 0.6, 0)]
    for x in (-0.4, 0.0, 0.4):
        P.append(barrel(0.06, 0.05, 2.4, 0.3, col, seg=4).T(x, 1.15, 0.5))
    return merge(P)


def tower(col=GREY):
    """The tower bridge, from its base: tapering tower, platforms, bridge with windows, 15 m rangefinder, radar."""
    P = [cyl(6.0, 3.5, 17.0, col, seg=10)]
    for y, r in ((4.0, 7.2), (8.0, 6.3), (11.5, 5.6)):
        P.append(cyl(r, r, 0.35, col, seg=12).T(0, y, 0))
    P.append(box(11.0, 3.2, 7.5, col).T(0, 13.0, 0.6))                         # bridge
    for x in np.arange(-4.5, 4.6, 1.0):
        P.append(quad(x, 14.9, 4.35, 0.75, 0.9, "+z", DARK))
    for s in (-1, 1):
        P.append(quad(s * 5.5, 14.9, 1.5, 3.5, 0.9, "+x" if s > 0 else "-x", DARK))
    P.append(cyl(3.1, 3.0, 2.6, col, seg=12).T(0, 17.0, 0))                    # rangefinder tower
    P.append(box(15.0, 1.3, 1.4, col).T(0, 18.3, 0))
    for s in (-1, 1):
        P.append(box(1.2, 1.7, 1.7, col).T(s * 7.5, 18.1, 0))
    P.append(box(4.4, 2.4, 0.3, (0.3, 0.3, 0.3)).T(0, 19.9, 0))               # Type 21 radar ("mattress")
    for s in (-1, 1):
        P.append(box(0.25, 2.2, 3.4, (0.3, 0.3, 0.3)).T(s * 3.0, 16.2, -2.5))   # Type 22 radar horns (simplified)
    P.append(tube([(0, 17.0, -2.0), (0, 25.5, -3.2)], 0.18, col, seg=6))       # signal mast
    P.append(tube([(-4.0, 23.5, -3.0), (4.0, 23.5, -3.0)], 0.08, col, seg=4))  # yard
    return merge(P)


def funnel(col=GREY):
    rake = 0.4
    P = [cyl(5.6, 4.6, 17.0, col, seg=20, sx=0.74).R(-rake, 0, 0)]
    top = cyl(4.6, 4.6, 0.6, DARK, seg=20, sx=0.74).T(0, 17.0, 0).R(-rake, 0, 0)
    P.append(top)
    for k in range(-2, 3):                                                       # the cap grille
        P.append(box(6.4, 0.25, 0.3, (0.2, 0.2, 0.2)).T(0, 17.6, k * 1.5).R(-rake, 0, 0))
    return merge(P)


def searchlight(col=GREY):
    return merge([cyl(0.5, 0.5, 1.0, col, seg=8), cyl(0.85, 0.85, 1.2, col, seg=10).R(math.pi / 2, 0, 0).T(0, 1.6, -0.6)])


def catapult(col=STEEL):
    return merge([cyl(1.6, 1.6, 0.9, col, seg=12), box(1.4, 0.9, 19.0, col).T(0, 0.9, 5.5)])


def crane(col=GREY):
    return merge([cyl(1.3, 1.0, 6.5, col, seg=10), box(2.6, 2.2, 3.2, col).T(0, 6.5, 0),
                  tube([(0, 7.6, 1.0), (0, 13.5, 13.0)], 0.35, col, seg=6)])


def propeller(r=2.5, col=BRONZE):
    P = [cyl(0.55, 0.55, 1.6, col, seg=12).R(math.pi / 2, 0, 0).T(0, 0, -0.8)]
    for k in range(3):
        a = 2 * math.pi * k / 3
        P.append(box(r * 0.95, 0.2, 1.7, col, bottom=False).T(r * 0.5 + 0.35, 0, 0).R(0.42, 0, 0).R(0, 0, a))
    return merge(P)


SHAFTS = ((10.4, 3.6, 232.0, 196.0), (4.6, 3.2, 246.0, 214.0))   # x, y, propeller at d, leaves the hull at d


def stern_gear(drop_stbd_outer=False):
    P = []
    for x, y, dp, dh in SHAFTS:
        for s in (-1, 1):
            if drop_stbd_outer and s < 0 and x > 8:
                P.append(tube([(s * x, y, zd(dh)), (s * x, y, zd(dp) + 4.0)], 0.4, STEEL, seg=8))   # shaft stub only
                continue
            P.append(tube([(s * x, y, zd(dh)), (s * x, y, zd(dp) + 0.6)], 0.42, STEEL, seg=8))
            P.append(propeller().T(s * x, y, zd(dp)))
            P.append(tube([(s * x, y + 0.2, zd(dp + 3.5) + 0), (s * x * 0.55, y + 4.5, zd(dp + 3.5))], 0.25, STEEL, seg=5))  # A-bracket
            P.append(tube([(s * x, y - 0.2, zd(dp + 3.5) + 0), (s * x * 0.75, y + 5.0, zd(dp + 5))], 0.25, STEEL, seg=5))
    P.append(box(0.9, 8.0, 5.8, RED).T(0, 1.8, zd(257.5)))                         # main rudder
    P.append(box(0.8, 5.5, 4.4, RED).T(0, 0.4, zd(240.5)))                         # auxiliary rudder
    P.append(box(0.7, 2.6, 18.0, RED).T(0, 1.0, zd(229)))                          # skeg
    return P


def bulb():
    prof = [(0.05, -6.5), (1.7, -5.8), (2.7, -4.3), (3.2, -2.2), (3.3, 0), (3.0, 2.0), (2.2, 3.6), (1.0, 4.8), (0.05, 5.3)]
    m = lathe(prof, RED, seg=16, axis="z")
    m.S(0.85, 1.0, 1)
    return m.T(0, 4.6, zd(0) - 1.6)


def rake(m):
    """Raked, flared stem: the lower bow pulled aft (the deck line keeps the overall length of 263 m)."""
    d = L / 2 - m.v[:, 2]
    f = np.clip(m.v[:, 1] / (D + 5.6), 0, 1) ** 1.6
    m.v[:, 2] += RAKE * (f - 1) * sstep(26, 0, d)
    return m


def crest(col):
    """The chrysanthemum crest on the stem (1.5 m across): a disc of 16 petals."""
    P = [cyl(0.75, 0.75, 0.2, col, seg=16).R(math.pi / 2, 0, 0)]
    for k in range(16):
        a = 2 * math.pi * k / 16
        P.append(box(0.22, 0.3, 0.12, col, bottom=False).T(0.68, 0, 0.12).R(0, 0, a))
    P.append(cyl(0.22, 0.22, 0.32, col, seg=10).R(math.pi / 2, 0, 0))
    return merge(P).T(0, deck_y(0.6) - 1.4, zd(0.6) + 0.5)


def ship_part(d0, d1, built=False, anchors=(-1, 1), crest_col=GOLD, open_barbettes=False, drop_prop=False):
    """The hull and everything fixed to it between d0 and d1 (distance from the bow). The turrets,
    the superstructure and the deck guns are added separately (superstructure())."""
    P = [hull(d0, d1)]
    inr = lambda d, m=0: d0 + m <= d <= d1 - m
    if d0 <= 0.5:
        P.append(bulb())
        P.append(crest(crest_col))
        for s in anchors:                                                         # hawse pipes and anchors
            z = zd(9.0); y = deck_y(9.0) - 4.6; x = s * (hw(z, y) + 0.3)
            P.append(box(0.5, 3.8, 0.9, DARK).T(x, y - 2.8, z))
            P.append(box(0.55, 0.7, 2.6, DARK).T(x, y - 3.1, z))
        if built:
            P.append(at(cyl(0.12, 0.08, 6.0, GREY, seg=5), 1.0, y=deck_y(1.0)))   # jackstaff
    # forecastle: windlasses, cable to the hawse pipes, bollards, the breakwater ahead of turret No. 1
    if inr(22):
        for s in (-1, 1):
            P.append(at(cyl(1.2, 1.0, 1.4, STEEL, seg=12), 21, x=s * 3.0, y=deck_y(21)))
            P.append(tube([(s * 3.0, deck_y(21) + 0.4, zd(21)), (s * 4.5, deck_y(15) + 0.25, zd(15)),
                           (s * (hw(zd(9.5), deck_y(9.5)) - 1.0), deck_y(9.5) + 0.2, zd(9.5))], 0.28, (0.16, 0.13, 0.1), seg=5))
        for d in (6, 14, 30, 42):
            for s in (-1, 1):
                if inr(d):
                    x = s * (hw(zd(d), deck_y(d)) - 1.8)
                    P += [at(cyl(0.38, 0.38, 0.9, STEEL, seg=8), d, x=x, y=deck_y(d)), at(cyl(0.38, 0.38, 0.9, STEEL, seg=8), d + 1.0, x=x, y=deck_y(d))]
    if inr(56, 1):
        th = 0.5
        for s in (-1, 1):
            P.append(box(14.0, 1.7, 0.35, GREY).R(0, s * th, 0).T(s * 7.0 * math.cos(th), deck_y(56), zd(56) - 7.0 * math.sin(th)))
    # barbettes of the main and secondary turrets (open rings where the turrets fell out)
    for d, rb, top in ((T1, RB, 1.2), (T2, RB, 5.0), (T3, RB, 1.2), (S1, 4.4, 10.0), (S2, 4.4, 10.0)):
        if inr(d, rb):
            y0 = deck_y(d)
            if built or not open_barbettes:
                P.append(at(cyl(rb, rb, top, GREY, seg=28), d, y=y0))
            else:
                P.append(at(cyl(rb, rb, top, GREY, seg=28, caps=False), d, y=y0))
                P.append(at(cyl(rb - 0.6, rb - 0.6, 0.05, DARK, seg=28), d, y=y0 + top - 0.9))
                P.append(at(cyl(rb - 0.6, rb - 0.6, top, INSIDE, seg=28, caps=False), d, y=y0))
    # portholes (two rows forward and aft of the armoured citadel)
    for y_off, a, b, sp in ((1.6, 8, 62, 3.0), (3.6, 14, 60, 3.2), (1.6, 212, 256, 3.0), (3.4, 214, 252, 3.2)):
        for d in np.arange(max(a, d0 + 1), min(b, d1 - 1), sp):
            z = zd(d); y = deck_y(d) - y_off
            x = hw(z, y)
            if x < 3:
                continue
            for s in (-1, 1):
                P.append(quad(s * x, y, z, 0.42, 0.42, "+x" if s > 0 else "-x", DARK, eps=0.06))
    # bilge keels
    a2, b2 = max(80, d0 + 1), min(185, d1 - 1)
    if b2 - a2 > 3:
        for s in (-1, 1):
            pts = [(s * (hw(zd(d), 2.0) + 0.5), 2.0, zd(d)) for d in np.arange(a2, b2 + 0.1, 5.0)]
            P.append(tube(pts, 0.35, RED, seg=4))
    # aviation deck aft: catapults, aircraft crane, hangar lift; shafts, propellers and rudders
    if built and inr(240):
        for s in (-1, 1):
            P.append(at(catapult().R(0, s * 0.18, 0), 236, x=s * 10.5, y=deck_y(236)))
        P.append(at(box(9.0, 0.05, 10.0, DARK), 222, y=deck_y(222)))
        P.append(at(crane(), 255, y=deck_y(255)).R(0, 0, 0))
    if d1 >= L - 0.01:
        P += stern_gear(drop_prop)
    if d0 <= 26:
        P = [rake(m) for m in P]
    return P


def superstructure():
    """Everything above the upper deck amidships, as she sailed in April 1945."""
    P = []
    y0 = D
    P.append(at(box(24.0, LV, 82.0, GREY), 149, y=y0))                             # level 1, d 108–190
    P.append(at(box(17.0, LV, 60.0, GREY), 148, y=y0 + LV))                        # level 2, d 118–178
    P.append(at(box(13.0, LV, 16.0, GREY), 127, y=y0 + 2 * LV))                    # tower base
    for lv, (wd, a, b) in enumerate(((24.0, 108, 190), (17.0, 118, 178))):
        for s in (-1, 1):
            for d in np.arange(a + 2, b - 1, 3.2):
                P.append(quad(s * wd / 2, y0 + lv * LV + 1.5, zd(d), 0.7, 0.7, "+x" if s > 0 else "-x", DARK))
    for d, top, back in ((T1, 1.2, False), (T2, 5.0, False), (T3, 1.2, True)):
        P.append(at(main_turret(0.05).R(0, math.pi if back else 0, 0), d, y=deck_y(d) + top))
    for d, back in ((S1, False), (S2, True)):
        P.append(at(sec_turret(0.05).R(0, math.pi if back else 0, 0), d, y=deck_y(d) + 10.0))
    P.append(at(tower(), 127, y=y0 + 3 * LV))
    P.append(at(funnel(), 152, y=y0 + 2 * LV))
    for s in (-1, 1):                                                               # searchlight platforms by the funnel
        P.append(at(box(6.0, 0.4, 9.0, GREY), 151, x=s * 7.0, y=y0 + 2 * LV + 6.0))
        P.append(at(box(1.0, 6.0, 1.0, GREY), 151, x=s * 7.0, y=y0 + 2 * LV))
        for dz in (-3.0, 0.0, 3.0):
            P.append(at(searchlight(), 151 + dz, x=s * 8.0, y=y0 + 2 * LV + 6.4))
    # mainmast (raked aft) with its yard, the after fire-control tower and its 10 m rangefinder
    P.append(tube([(0, y0 + 2 * LV, zd(162)), (0, y0 + 30.0, zd(167))], 0.5, GREY, seg=8))
    for s in (-1, 1):
        P.append(tube([(s * 4.0, y0 + 2 * LV, zd(166)), (0, y0 + 18.0, zd(164.5))], 0.3, GREY, seg=6))
    P.append(tube([(-6.0, y0 + 24.0, zd(166)), (6.0, y0 + 24.0, zd(166))], 0.12, GREY, seg=4))
    P.append(at(box(8.0, 6.5, 7.0, GREY), 172, y=y0 + 2 * LV))
    P.append(at(box(10.0, 1.2, 1.3, GREY), 172.5, y=y0 + 2 * LV + 6.5))
    P.append(at(box(3.6, 2.0, 0.25, (0.3, 0.3, 0.3)), 171, y=y0 + 2 * LV + 7.9))
    # twelve twin 12.7 cm mounts on the roof of level 1, six a side
    for d in (128, 136, 144, 152, 160, 168):
        for s in (-1, 1):
            P.append(at(mount127().R(0, s * 1.2, 0), d, x=s * 10.0, y=y0 + LV))
    # triple 25 mm mounts: along the deck edges, on the superstructure roofs and on the tower platforms
    spots = []
    for d in list(np.arange(60, 106, 5.5)) + list(np.arange(186, 214, 5.0)):
        for s in (-1, 1):
            spots.append((d, s * (hw(zd(d), deck_y(d)) - 2.2), deck_y(d)))
    for d in np.arange(110, 189, 6.0):
        for s in (-1, 1):
            if not (122 <= d <= 172):
                spots.append((d, s * 9.5, y0 + LV))
            spots.append((d, s * 6.5, y0 + 2 * LV) if 118 < d < 178 else (d, s * 9.5, y0 + LV))
    for y, r in ((4.0, 6.3), (8.0, 5.4)):
        for s in (-1, 1):
            spots.append((127, s * r, y0 + 3 * LV + y + 0.35))
    seen = set()
    for d, x, y in spots:
        key = (round(d), round(x), round(y))
        if key in seen:
            continue
        seen.add(key)
        P.append(at(mount25().R(0, (1 if x > 0 else -1) * 1.3, 0), d, x=x, y=y))
    return P


# ── Pieces of wreckage ───────────────────────────────────────────────────────
def plate(r, w, h, d, col, amp=0.5, seed=0):
    p = box(w, h, d, col, bottom=False)
    crumple(p, amp, 0.3, seed)
    return p


def torn_end(d_break, y_lo, y_hi, n, seed, aft_side):
    """Torn plates and frames sticking out of a broken end, the inside dark."""
    r = np.random.default_rng(seed)
    out = []
    z0 = zd(d_break)
    for i in range(n):
        y = r.uniform(y_lo, y_hi)
        wdt = hw(z0, y) - 0.5
        p = plate(r, r.uniform(2, 6), 0.12, r.uniform(2.5, 7), GREY if y > WL else RED, 0.8, seed + i)
        dz = r.uniform(0.5, 4.5) * (1 if aft_side else -1)
        out.append(p.R(r.uniform(-0.6, 0.6), r.uniform(-0.4, 0.4), r.uniform(-0.8, 0.8)).T(r.uniform(-wdt, wdt), y, z0 + dz))
    # dark bulkhead just inside the break and the decks seen through it
    inner = box(B * 0.92, D * 0.86, 0.25, INSIDE).T(0, 0.8, z0 + (-4.0 if aft_side else 4.0))
    crumple(inner, 1.0, 0.2, seed + 99)
    out.append(inner)
    for yy in (D - 2.6, D - 5.2, D - 8.0, 6.0):
        dk = box(B * 0.86, 0.3, 7.0, (0.2, 0.18, 0.16), bottom=False).T(0, yy, z0 + (-2.0 if aft_side else 2.0))
        crumple(dk, 0.7, 0.3, seed + int(yy * 10))
        out.append(dk)
    return merge(out)


def silt(m, k, seed):
    """A fine dusting of silt over everything that faces upwards (80 years at 345 m)."""
    n = normals(m)
    a = k * sstep(-0.1, 0.7, n[:, 1]) * (0.6 + 0.4 * fbm(m.v, 0.1, seed, 2))
    m.c = m.c * (1 - a[:, None]) + np.array((0.52, 0.49, 0.42)) * a[:, None]
    return m


def mud_for(m, h0, seed, pad=12.0, cell=2.0, col=(0.5, 0.46, 0.38)):
    """Sediment heaped against a piece lying on the seabed (only the heaps, no flat sheet)."""
    low = m.v[m.v[:, 1] < 1.0]
    if len(low) == 0:
        return Mesh()
    r = np.random.default_rng(seed)
    pts = low[:, [0, 2]][r.choice(len(low), size=min(2500, len(low)), replace=False)]
    x0, z0 = pts.min(0) - pad; x1, z1 = pts.max(0) + pad
    nx, nz = max(4, int((x1 - x0) / cell)), max(4, int((z1 - z0) / cell))

    def h(X, Z):
        q = np.stack([X.ravel(), Z.ravel()], 1)
        dist = np.full(len(q), 1e9)
        for k in range(0, len(pts), 500):
            dd = np.linalg.norm(q[:, None, :] - pts[None, k:k + 500, :], axis=2).min(1)
            dist = np.minimum(dist, dd)
        dist = dist.reshape(X.shape)
        hh = h0 * np.exp(-((dist - 2.5) / 4.0) ** 2) * sstep(0.3, 1.5, dist)
        bumps = 0.25 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.2, seed).reshape(X.shape) * sstep(0.2, 1.0, hh)
        return hh + bumps - 0.5
    g = grid(x0, x1, z0, z1, nx, nz, h, col)
    g.paint(lambda p: np.asarray(col) + (fbm(p, 0.12, seed + 1)[:, None] - 0.5) * 0.06)
    g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > 0.15]
    return g


# ── Bow section: from the bow to the break between barbettes No. 1 and No. 2 (~88 m) ──
BOW_END = 88.0
bow = merge(ship_part(0, BOW_END, anchors=(-1,), crest_col=GOLD_OLD, open_barbettes=True) + [torn_end(BOW_END, 1.0, D + 1.0, 26, 300, False)])
dfb = L / 2 - bow.v[:, 2]
# the gap between the upper and lower parts of the bow: the upper part lifted, the faces across it gone
YS = 13.0
gap = 1.8 * sstep(40, 14, dfb)
above = bow.v[:, 1] > YS
fv = bow.v[bow.f]
cross = (fv[:, :, 1].min(1) < YS) & (fv[:, :, 1].max(1) > YS) & ((L / 2 - fv[:, :, 2].mean(1)) < 36) & ((L / 2 - fv[:, :, 2].mean(1)) > 3)
bow.f = bow.f[~cross]
bow.v[:, 1] += np.where(above, gap, 0.0)
inside_gap = []
for d in np.arange(6, 36, 3.0):                                                # dark inside seen through the gap
    z = zd(d); wd = 2 * hw(z, YS) - 1.0
    if wd > 1:
        inside_gap.append(box(wd, 0.2, 3.0, INSIDE, bottom=False).T(0, YS - 0.6, z))
bow = merge([bow] + inside_gap)
dfb = L / 2 - bow.v[:, 2]
crumple(bow, 2.2, 0.13, 11, sstep(70, BOW_END, dfb))                           # buckled towards the break
crumple(bow, 3.0, 0.35, 12, sstep(83, BOW_END, dfb))                           # torn edges
crumple(bow, 1.6, 0.4, 14, sstep(5, -4, dfb) * sstep(8, 2, bow.v[:, 1]))       # the cracked, crumbling bulb
crumple(bow, 0.22, 0.08, 13)                                                   # dents everywhere
zcb = (zd(0) + zd(BOW_END)) / 2
bow.T(0, 0, -zcb)
ROLL = math.radians(70)                                                        # fallen over on her starboard side
bow.R(0, 0, ROLL)
bow.R(math.radians(-1.5), 0, 0)
bow.T(0, -bow.v[:, 1].min() - 4.5, 0)                                          # the starboard side sunk in the sediment
# the tower bridge, crushed and lying under the starboard side near the bow, its top sticking out
tw = tower(GREY)
crumple(tw, 1.4, 0.25, 21)
tw.v[:, 0] *= 1 + 0.3 * sstep(0, 12, tw.v[:, 1])
tw.R(0, 0, math.pi / 2 + 0.15).R(0, 0.35, 0)
xs_near = bow.v[(np.abs(bow.v[:, 2] - (zd(32) - zcb)) < 3) & (bow.v[:, 1] < 2.5)][:, 0]
x_edge = xs_near.min() if len(xs_near) else -10.0
tw.T(x_edge + 9.0, 3.2, zd(32) - zcb)
bow = merge([bow, tw])
weather(bow, rust=0.6, sediment=0.75, streaks=0.6, seed=21, rust_col=(0.36, 0.2, 0.11))
silt(bow, 0.4, 22)
bow_mud = mud_for(bow, 2.6, 5)

# ── After part: midships and stern (~168 m), capsized ────────────────────────
STERN_START = 95.0
JOINT = 140.0                                                                  # where the midships part hangs on to the stern
stern = merge(ship_part(STERN_START, L, open_barbettes=True, drop_prop=True) + [torn_end(STERN_START, 1.0, D, 30, 400, True)])
dfs = L / 2 - stern.v[:, 2]
# the joint: plating torn through all round except a strip along the bottom
fv = stern.v[stern.f]
fd = L / 2 - fv[:, :, 2].mean(1)
fy = fv[:, :, 1].mean(1); fx = fv[:, :, 0].mean(1)
tear = (np.abs(fd - JOINT) < 1.6) & ~((fy < 3.5) & (np.abs(fx) < 7))
# the hole in the port bottom, around the engine rooms
rim = 1.3 * (fbm(fv.mean(1), 0.22, 77, 2) - 0.5)                                # ragged edge
hole = (((fd - 163) / 14) ** 2 + ((fx - 9.5) / 6.5) ** 2 < 1 + rim) & (fy < 7.0) & (fx > 0)
stern.f = stern.f[~(tear | hole)]
inner = [box(13.0, 0.3, 26.0, INSIDE, bottom=False).T(9.5, 7.5, zd(163))]   # inside the hole: a deck, boilers, shafts
for dz in (-8, 0, 8):
    inner.append(box(4.5, 5.0, 5.0, BOILER).T(9.0, 2.5, zd(163 + dz)))
for x in (4.6, 10.4):
    inner.append(tube([(x, 3.3, zd(150)), (x, 3.3, zd(180))], 0.42, STEEL, seg=8))
for dz in np.arange(-10, 11, 4.0):                                              # frames across the hole, bent
    fr = box(9.0, 0.4, 0.3, (0.22, 0.12, 0.07), bottom=False)
    crumple(fr, 0.4, 0.3, int(dz) + 90)
    inner.append(fr.T(8.5, 3.2 + 0.08 * abs(dz), zd(163 + dz)))
for i in range(12):                                                             # plates torn outwards round the hole
    th = 2 * math.pi * i / 12 + rng.uniform(-0.2, 0.2)
    dz, x = 14 * math.cos(th), 9.5 + 6.5 * math.sin(th)
    p = box(rng.uniform(1.5, 3.5), 0.15, rng.uniform(2, 4), RED, bottom=False)
    crumple(p, 0.4, 0.4, 950 + i)
    yb = max(0.3, yb_of(t_of(zd(163 + dz))))
    inner.append(p.R(0, 0, -0.6 * math.sin(th)).R(0.6 * math.cos(th), 0, 0).T(x, yb - 0.6, zd(163 + dz)))
stern = merge([stern] + inner)
dfs = L / 2 - stern.v[:, 2]
# the midships part bent away from the stern at the joint ("arched")
fw = sstep(JOINT + 1.5, JOINT - 1.5, dfs)
zj = zd(JOINT)
yaw, pitch = math.radians(9), math.radians(4)
dz_ = stern.v[:, 2] - zj
stern.v[:, 0] += fw * dz_ * math.sin(yaw)
stern.v[:, 1] += fw * dz_ * math.sin(pitch)                                    # after the capsize: down into the mud
crumple(stern, 2.6, 0.12, 31, sstep(JOINT, STERN_START, dfs))                  # the shattered midships
crumple(stern, 3.4, 0.35, 32, sstep(STERN_START + 6, STERN_START, dfs))        # the torn forward end
crumple(stern, 1.5, 0.3, 35, np.exp(-((dfs - JOINT) / 3.0) ** 2))
crumple(stern, 0.3, 0.08, 33)
zcs = (zd(STERN_START) + zd(L)) / 2
stern.T(0, 0, -zcs)
stern.R(0, 0, math.pi)                                                         # keel up
stern.T(0, D - 2.4, 0)                                                         # resting on her deck, settled in the sediment
weather(stern, rust=0.6, sediment=0.75, streaks=0.55, seed=41, rust_col=(0.36, 0.2, 0.11))
silt(stern, 0.4, 42)
stern_mud = mud_for(stern, 2.2, 6)

# ── The site: where everything lies (geographic metres from the after part: east, north) ──
H = 310.0                                                                      # heading of the bow section = the file's +Z
ZA = np.array([math.sin(math.radians(H)), math.cos(math.radians(H))])
XA = np.array([math.sin(math.radians(H - 90)), math.cos(math.radians(H - 90))])


def place(m, east, north, heading):
    """Put a piece built bow-to-+Z at (east, north) metres from the origin, its bow to `heading`."""
    p = np.array([east, north])
    return m.R(0, -math.radians(heading - H), 0).T(float(p @ XA), 0, float(p @ ZA))


STERN_HDG = 270.0                                                              # the broken forward end to the west, rudders east
BOW_BREAK = (-118.0, 32.0)                                                     # less than 50 m from the after part's forward end
bow_mid = (BOW_BREAK[0] + (BOW_END / 2) * math.sin(math.radians(H)), BOW_BREAK[1] + (BOW_END / 2) * math.cos(math.radians(H)))
place(bow, *bow_mid, H); place(bow_mud, *bow_mid, H)
place(stern, 0, 0, STERN_HDG); place(stern_mud, 0, 0, STERN_HDG)

# ── The main turrets, upside down (north to south: No. 2, No. 3, No. 1) ──────
turrets = []
sed = []
tA = main_turret(0.3, trunk=9.0)                                              # No. 1: standing upside down, the best kept
tA.R(0, 0, math.pi).T(0, 5.6 - 1.8, 0)
crumple(tA, 0.25, 0.1, 51)
turrets.append(place(tA, -20, -105, 160))
tB = main_turret(0.3, trunk=9.0)                                              # No. 2: badly broken, the handling room tilted
tB.R(0, 0, math.pi).T(0, 5.6 - 2.4, 0)
yy = tB.v[:, 1] - 8.0
a_ = 0.32 * sstep(7, 10, tB.v[:, 1])                                          # tilted above y = 8 m
tB.v[:, 1] = np.where(yy > 0, 8.0 + yy * np.cos(a_), tB.v[:, 1])
tB.v[:, 0] = tB.v[:, 0] + np.where(yy > 0, yy * np.sin(a_), 0.0)
crumple(tB, 1.3, 0.2, 52, sstep(4, 14, tB.v[:, 1]))
crumple(tB, 0.4, 0.1, 53)
turrets.append(place(tB, 15, 68, 75))
tC = main_turret(0.3, trunk=9.0)                                              # No. 3: half buried
tC.R(0, 0, math.pi).R(0.2, 0, 0.12).T(0, 5.6 - 8.0, 0)
crumple(tC, 0.5, 0.15, 54)
turrets.append(place(tC, 40, -48, 220))
for t_ in turrets:
    weather(t_, rust=0.55, sediment=0.7, streaks=0.5, seed=55, rust_col=(0.36, 0.2, 0.11))
    silt(t_, 0.35, 56)
    sed.append(mud_for(t_, 1.0, 57, pad=8.0))
turrets_m = merge(turrets)

# ── Debris field ─────────────────────────────────────────────────────────────
deb = []
# a 15.5 cm turret on its side, three barrels showing, the middle muzzle burst
s1 = sec_turret(0.1, trunk=5.0)
burst = cyl(0.2, 0.55, 1.4, GREY, seg=8, caps=False).R(math.pi / 2 - 0.1, 0, 0).T(0, 1.5 + 9.3 * math.sin(0.1), 3.75 + 9.3 * math.cos(0.1) - 0.2)
crumple(burst, 0.25, 0.8, 61)
s1 = merge([s1, burst]).R(0, 0, 1.45).T(0, 4.2, 0)
crumple(s1, 0.3, 0.2, 62)
deb.append(place(s1, -62, -36, 200))
s2 = sec_turret(0.3, trunk=5.0).R(0, 0, math.pi).T(0, 3.3 - 1.2, 0)          # the other, upside down, barrels in the mud
crumple(s2, 0.5, 0.2, 63)
deb.append(place(s2, 5, 42, 20))
# the after fire-control tower, lying by the propellers
rt = merge([box(8.0, 6.5, 7.0, GREY), box(10.0, 1.2, 1.3, GREY).T(0, 6.5, 0.5), box(3.6, 2.0, 0.25, (0.3, 0.3, 0.3)).T(0, 7.9, -1)])
crumple(rt, 0.9, 0.25, 64)
deb.append(place(rt.R(0, 0, 1.4).T(0, 3.6, 0), 100, 30, 60))
# the starboard outer propeller, stuck in the seabed
pr = propeller().R(0.5, 0, 0.3).T(0, 1.4, 0)
deb.append(place(pr, 72, -32, 300))
# three mounds of shattered midships wreckage (superstructure, armour, boilers, the funnel) ~70 m south
r = np.random.default_rng(7)
MOUNDS = ((-122, -58, 14), (-86, -72, 16), (-50, -62, 12))
for mi, (me, mn, rad) in enumerate(MOUNDS):
    items = []
    for i in range(46):
        a, rr = r.uniform(0, 2 * math.pi), rad * math.sqrt(r.uniform(0, 1))
        x, z = rr * math.cos(a), rr * math.sin(a)
        hgt = 6.5 * (1 - (rr / rad) ** 2) * r.uniform(0.5, 1.0)
        k = r.random()
        if k < 0.4:                                                            # crumpled superstructure and hull plates
            p = plate(r, r.uniform(4, 11), r.uniform(0.2, 2.6), r.uniform(3, 8), GREY if r.random() < 0.75 else RED, 0.9, 700 + mi * 100 + i)
        elif k < 0.55:                                                         # armour slabs
            p = box(r.uniform(3, 6), 0.45, r.uniform(2.5, 4), (0.24, 0.22, 0.2), bottom=False)
        elif k < 0.7:                                                          # beams
            p = box(0.45, 0.7, r.uniform(5, 13), (0.22, 0.16, 0.11), bottom=False)
        elif k < 0.8:                                                          # boilers
            p = box(4.5, 5.0, 5.0, BOILER, bottom=False); crumple(p, 0.5, 0.3, 800 + i)
        elif k < 0.9:
            p = mount25()
        else:
            p = mount127()
        items.append(p.R(r.uniform(-0.6, 0.6), r.uniform(0, 6.28), r.uniform(-0.6, 0.6)).T(x, hgt, z))
    if mi == 1:                                                                # a piece of the funnel
        f = cyl(5.4, 5.0, 9.0, GREY, seg=20, sx=0.74, caps=False)
        keep = np.arctan2(f.v[:, 2], f.v[:, 0]) > -1.2
        f.f = f.f[keep[f.f].all(axis=1)]
        crumple(f, 0.8, 0.2, 811)
        items.append(f.R(math.pi / 2 - 0.3, 0.7, 0).T(4, 3.0, -3))
    if mi == 0:                                                                # the mainmast, broken
        items.append(tube([(-12, 0.6, -4), (2, 2.4, 3), (14, 0.4, 9)], 0.5, GREY, seg=8))
    heap = merge(items)

    def hh(X, Z, rad=rad, s=mi):
        d = np.hypot(X, Z) / rad
        return 5.0 * np.clip(1 - d ** 2, 0, 1) + 0.6 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.15, 900 + s).reshape(X.shape) - 0.5
    g = grid(-rad * 1.2, rad * 1.2, -rad * 1.2, rad * 1.2, 16, 16, hh, (0.5, 0.46, 0.38))
    g.paint(lambda p: (0.5, 0.46, 0.38) + (fbm(p, 0.12, 910 + mi)[:, None] - 0.5) * 0.06)
    g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > 0.15]
    deb.append(place(heap, me, mn, 0))
    sed.append(place(g, me, mn, 0))
# a catapult, torn off
cp = catapult()
crumple(cp, 0.6, 0.3, 66)
deb.append(place(cp.R(0, 0, 0.3).T(0, 0.4, 0), 60, 34, 250))


def spot():
    """Somewhere in the debris field: mostly around the hull, the turrets and the mounds."""
    k = r.random()
    if k < 0.35:
        return r.normal(-90, 25), r.normal(-62, 14)
    if k < 0.55:
        return r.normal(-100, 30), r.normal(20, 30)
    if k < 0.75:
        e, n = [(-20, -105), (15, 68), (40, -48)][r.integers(3)]
        return r.normal(e, 14), r.normal(n, 14)
    return r.uniform(-170, 110), r.uniform(-110, 90)


loose = []
for i in range(90):                                                            # hull plates and deck pieces, bent
    e, n = spot()
    p = plate(r, r.uniform(2, 8), 0.18, r.uniform(1.5, 5), GREY if r.random() < 0.7 else (RED if r.random() < 0.5 else DECKW), 0.6, 1000 + i)
    loose.append(place(p.R(r.uniform(-0.35, 0.35), 0, r.uniform(-0.35, 0.35)).T(0, 0.1, 0), e, n, r.uniform(0, 360)))
for i in range(26):                                                            # beams
    e, n = spot()
    loose.append(place(box(0.4, 0.6, r.uniform(4, 12), (0.22, 0.16, 0.11)).R(0, 0, r.uniform(-0.2, 0.2)), e, n, r.uniform(0, 360)))
for i in range(14):                                                            # 25 mm mounts and 12.7 cm mounts, tumbled
    e, n = spot()
    m_ = mount25() if i < 10 else mount127()
    loose.append(place(m_.R(r.uniform(-1.2, 1.2), 0, r.uniform(-1.6, 1.6)).T(0, 0.6, 0), e, n, r.uniform(0, 360)))
for i in range(150):                                                           # main-gun powder cans, everywhere
    e, n = spot()
    c_ = cyl(0.3, 0.3, 1.2, (0.42, 0.36, 0.26), seg=6).R(math.pi / 2, 0, 0).R(r.uniform(-0.3, 0.3), 0, 0).T(0, 0.15, 0)
    loose.append(place(c_, e, n, r.uniform(0, 360)))
for i in range(160):                                                           # smaller things
    e, n = spot()
    s = r.uniform(0.3, 1.4)
    loose.append(place(box(s, s * r.uniform(0.4, 1.0), s * r.uniform(0.6, 1.6), (0.25, 0.2, 0.15)).T(0, -0.1, 0), e, n, r.uniform(0, 360)))
debris = merge(deb + loose)
weather(debris, rust=0.55, sediment=0.7, streaks=0.45, seed=61, rust_col=(0.36, 0.2, 0.11))
silt(debris, 0.35, 62)
sediment = merge([bow_mud, stern_mud] + sed)

# ── The battleship as she sailed (model_built.glb): whole, clean, keel on y = 0, centred ──
ship = merge(ship_part(0, L, built=True) + superstructure())
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
out = ROOT / "models" / "item" / "Q215010" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
tris, size = write_glb(out, [("bow_section", bow), ("stern_section", stern), ("main_turrets", turrets_m),
                             ("debris_field", debris), ("sediment", sediment)])
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB")
for n, m in (("bow", bow), ("stern", stern), ("turrets", turrets_m), ("debris", debris), ("sediment", sediment)):
    print(f"  {n}: {m.tris:,} tris, x {m.v[:, 0].min():.0f}..{m.v[:, 0].max():.0f}, y {m.v[:, 1].min():.0f}..{m.v[:, 1].max():.0f}, z {m.v[:, 2].min():.0f}..{m.v[:, 2].max():.0f}")
tris2, size2 = write_glb(out.with_name("model_built.glb"), [("main", ship)])
print(f"model_built.glb: {tris2:,} triangles, {size2 / 2**20:.2f} MB, y {ship.v[:, 1].min():.1f}..{ship.v[:, 1].max():.1f}, z {ship.v[:, 2].min():.1f}..{ship.v[:, 2].max():.1f}")
