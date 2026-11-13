"""
SS Andrea Doria (Q493798) — the wreck of the Italian Line liner south of Nantucket, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: Italian Line liner of 1951/1953, length 213.8 m, beam 27.5 m, draught 9.1 m, 29,083 GRT, twin
screws. A long black hull with a raked stem and a cruiser stern; a white superstructure rising in tiers towards
the middle (promenade deck, boat-deck house, belvedere deck with the bridge forward), one streamlined white
funnel with a red top, 16 lifeboats (8 a side) on the boat deck, and aft the three outdoor swimming pools, one
per class, on terraced decks stepping down to the stern. The heights of the decks, the funnel and the masts,
and the exact layout of hatches, kingposts and pools are approximate (from photographs, not plans).

The wreck, after the published surveys and divers' accounts:
- she lies on her starboard side in about 73 m of water (240 ft), bow to the north (Life, 1956);
- the top of the wreck was at 49 m (162 ft) when she sank and is now at about 58 m (190 ft): the hull is
  collapsing on itself, flattening, the decks no longer vertical;
- the superstructure decks have sheared off and fallen onto the seabed beside her (to the east, the side the
  decks face), with the bridge and the funnel in twisted heaps; a swimming pool is encased in debris;
- the bow has broken off forward of the bridge (OceanGate sonar, 2016) and there is a large crack at the
  stern; fishing nets and lines are draped over the hull and the debris;
- the hole torn by the Stockholm (starboard side, under the bridge, about a third of the length from the bow,
  some 12 m deep) is on the side now against the sand: it is modelled, but lies below the seabed line.

    python model_sources/Q493798_andrea_doria.py   -> models/item/Q493798/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, weather, sstep, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1956)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B = 213.8, 27.5
H = 16.0                 # keel to the upper deck (top of the black hull), approximate
WL = 9.1                 # draught
LV = 2.8                 # height of one superstructure deck
RAKE = 6.0               # how far the stem rakes back from the deck to the forefoot
BLACK, RED, WHITE = (0.06, 0.06, 0.07), (0.46, 0.13, 0.09), (0.86, 0.85, 0.81)
DECK, DARK, STEEL, BRONZE = (0.52, 0.4, 0.27), (0.025, 0.025, 0.03), (0.5, 0.5, 0.5), (0.6, 0.43, 0.2)
FRED, POOL, TILE, GLASS = (0.72, 0.12, 0.09), (0.3, 0.62, 0.68), (0.74, 0.78, 0.76), (0.16, 0.22, 0.26)
NETC = (0.25, 0.27, 0.22)

SUP = [(52, 172, 0.97), (56, 152, 0.86), (60, 140, 0.66)]     # promenade deck, boat-deck house, belvedere deck
POOLS = [(146, 2, 8.0, 5.5), (161, 1, 9.0, 6.5), (182, 0, 10.0, 7.0)]   # first, cabin, tourist class: (d, level, l, w)
BOATS = [66, 76, 86, 96, 106, 116, 126, 136]                    # 8 lifeboats a side
FUNNEL_D, FORE_D, MAIN_D = 118, 71, 139


def zd(d):            # distance from the bow (m) -> ship coordinate z (bow at +L/2)
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):          # half-beam factor along the ship
    w = 1.0
    if t > 0.76:
        w = math.sin(((1 - t) / 0.24) * math.pi / 2) ** 1.1          # fine entrance
    if t < 0.2:
        w = min(w, math.sin((t / 0.2) * math.pi / 2) ** 0.7)         # fine run to a rounded cruiser stern
    return max(w, 0.004)


def yb_of(t):         # keel line: cut up aft for the screws and rudder, forefoot rising at the bow
    y = 0.0
    if t < 0.075:
        y = H * 0.5 * ((0.075 - t) / 0.075) ** 1.4
    if t > 0.975:
        y = max(y, H * 0.5 * ((t - 0.975) / 0.025) ** 2)
    return y


def yd_of(t):         # upper deck with sheer, strongest at the bow
    return H + 2.4 * float(sstep(0.62, 1.0, t)) ** 1.6 + 0.8 * float(sstep(0.3, 0.0, t))


def k_of(t):          # section fullness
    return 2.4 + 6.0 * w_of(t) ** 3


def hw(z, y):         # half-width of the hull at z and height y
    t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    if y >= yd:
        return B / 2 * w
    u = max(0.0, (y - yb) / max(0.1, yd - yb))
    return B / 2 * w * (1 - (1 - u) ** k)


def dy(d):
    return yd_of(t_of(zd(d)))


def at(m, d, x=0.0, y=0.0):
    return m.T(x, y, zd(d))


def hull(d0, d1, NP=26, ds=1.2, deck=DECK):
    """Hull from d0 to d1 (distance from bow): both sides and the upper deck; raked stem."""
    ns = max(2, int(math.ceil((d1 - d0) / ds)))
    st = []
    for i in range(ns + 1):
        z = zd(d1) + (zd(d0) - zd(d1)) * i / ns
        t = t_of(z)
        st.append((z, w_of(t), yb_of(t), yd_of(t), k_of(t)))
    W = 2 * NP + 1
    v, f = [], []
    for z, w, yb, yd, k in st:
        rk = RAKE * float(sstep(L / 2 - 26, L / 2, z))
        for j in range(W):
            side = -1 if j < NP else 1
            u = abs(NP - j) / NP
            v.append((side * B / 2 * w * (1 - (1 - u) ** k), yb + (yd - yb) * u, z - rk * (1 - u) ** 1.4))
    for i in range(ns):
        for j in range(W - 1):
            a = i * W + j; b = a + W; c = a + 1; d = b + 1
            f += [(a, c, b), (b, c, d)]
    hl = Mesh(v, f)
    hl.paint(lambda p: np.where((p[:, 1] < WL)[:, None], RED, BLACK))
    dv, df = [], []
    for z, w, yb, yd, k in st:
        dv += [(-B / 2 * w, yd, z), (B / 2 * w, yd, z)]
    for i in range(ns):
        a = 2 * i; b = a + 2; c = a + 1; d = b + 1
        df += [(a, b, c), (c, b, d)]
    return merge([hl, Mesh(dv, df, deck)])


def bulwark(d0, d1, h, color=BLACK):
    ns = max(2, int(math.ceil((d1 - d0) / 1.5)))
    zs = [zd(d1) + (zd(d0) - zd(d1)) * i / ns for i in range(ns + 1)]
    parts = []
    for side in (-1, 1):
        v, f = [], []
        for z in zs:
            t = t_of(z)
            v += [(side * B / 2 * w_of(t), yd_of(t) - 0.05, z), (side * B / 2 * w_of(t), yd_of(t) + h, z)]
        for i in range(ns):
            a = 2 * i
            f += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)] if side < 0 else [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
        parts.append(Mesh(v, f, color))
    return merge(parts)


def boat():
    b = lathe([(0.05, -4.8), (0.8, -4.1), (1.4, -2.6), (1.5, 0), (1.4, 2.6), (0.8, 4.1), (0.05, 4.8)], WHITE, seg=10, axis="z")
    b.v[:, 1] = np.minimum(b.v[:, 1], 0.35)
    return b.S(1, 0.9, 1)


def davit(x, y, z, s):
    return tube([(x, y, z), (x, y + 2.6, z), (x + s * 0.8, y + 3.3, z), (x + s * 1.5, y + 3.0, z)], 0.15, WHITE, seg=5)


def funnel():
    """Streamlined oval funnel, white with a red top, raked aft (approximate shape)."""
    hgt = 13.0
    f = cyl(5.0, 4.4, hgt - 2.8, WHITE, seg=24, sx=0.55, caps=False)
    top = cyl(4.4, 4.2, 2.8, FRED, seg=24, sx=0.55).T(0, hgt - 2.8, 0)
    rim = cyl(3.9, 3.9, 0.05, DARK, seg=24, sx=0.55).T(0, hgt + 0.02, 0)
    return merge([f, top, rim]).R(-0.1, 0, 0)


def pool(l, w):
    """An outdoor swimming pool: tiled rim, water."""
    return merge([box(w + 1.2, 0.35, l + 1.2, TILE), quad(0, 0.35, 0, w, l, "+y", POOL, eps=0.02)])


def ship_part(d0, d1, built=False):
    """Everything of the ship between d0 and d1 (distance from the bow). built=True: as she sailed; built=False:
    what is left on the hull of the wreck (the superstructure above the promenade deck goes to the debris)."""
    P = [hull(d0, d1, deck=DECK if built else (0.3, 0.25, 0.2))]            # on the wreck the teak has gone
    inr = lambda d, m=0: d0 + m <= d <= d1 - m
    clip = lambda a, b: (max(a, d0), min(b, d1))
    for a, b in ((0, 36), (186, L)):                                  # bulwarks at the bow and the stern
        a2, b2 = clip(a, b)
        if b2 - a2 > 1:
            P.append(bulwark(a2, b2, 1.1))
    # Superstructure: three white tiers, windows; bridge on top forward
    for i, (a, b, wf) in enumerate(SUP):
        a2, b2 = clip(a, b)
        if b2 - a2 < 1:
            continue
        y0 = H + i * LV
        if built:
            P.append(at(box(B * wf, LV, b2 - a2, WHITE), (a2 + b2) / 2, y=y0))
            P.append(at(box(B * wf - 0.2, 0.03, b2 - a2 - 0.2, DECK), (a2 + b2) / 2, y=y0 + LV))
            ww, wh, sp = ((1.8, 1.3, 2.6), (1.0, 1.0, 2.4), (1.3, 1.1, 2.4))[i]
            for d in np.arange(a2 + 1.6, b2 - 1, sp):
                for s in (-1, 1):
                    P.append(quad(s * B * wf / 2, y0 + 1.5, zd(d), ww, wh, "+x" if s > 0 else "-x", GLASS))
            if a2 == a:
                for x in np.arange(-B * wf / 2 + 1.5, B * wf / 2 - 1, 2.2):
                    P.append(quad(x, y0 + 1.5, zd(a), 1.6, 1.2, "+z", GLASS))
        elif i == 0:                                                  # promenade deck: ragged side walls only
            for d in np.arange(a2, b2 - 0.5, 5.0):
                ln = min(5.0, b2 - d)
                for s in (-1, 1):
                    h = rng.uniform(0.6, 2.4)
                    P.append(at(box(0.3, h, ln, WHITE), d + ln / 2, x=s * B * wf / 2, y=y0))
                    if h > 1.9:
                        P.append(quad(s * B * wf / 2, y0 + 1.3, zd(d + ln / 2), 1.8, 1.1, "+x" if s > 0 else "-x", DARK))
    if built:
        if inr(65):                                                   # wheelhouse and bridge wings
            yb_ = H + 3 * LV
            P.append(at(box(13, 2.6, 7, WHITE), 65.5, y=yb_))
            P.append(at(box(B * 1.0, 0.4, 3.0, WHITE), 63.5, y=yb_ + 2.2))
            for x in np.arange(-5.5, 5.6, 1.1):
                P.append(quad(x, yb_ + 1.6, zd(62), 0.9, 1.0, "+z", GLASS))
        if inr(FUNNEL_D):
            P.append(at(funnel(), FUNNEL_D, y=H + 3 * LV))
        for d, top in ((FORE_D, 44.0), (MAIN_D, 38.0)):               # pole masts (approximate heights)
            if inr(d):
                y0 = H + 3 * LV
                P.append(at(cyl(0.35, 0.18, top - y0, WHITE, seg=8).R(-0.04, 0, 0), d, y=y0))
                P.append(at(box(5.0, 0.2, 0.3, WHITE), d + 0.4, y=y0 + (top - y0) * 0.6))
        if inr(70):
            P.append(at(box(3.0, 0.4, 3.0, WHITE), 70.5, y=H + 3 * LV + 8))                     # radar platform
        for d in BOATS:                                                # lifeboats in their davits on the boat deck
            if inr(d, 5):
                for s in (-1, 1):
                    x = s * (B * 0.97 / 2 - 0.6)
                    P.append(boat().T(x, H + LV + 1.3, zd(d)))
                    P += [davit(x - s * 0.9, H + LV, zd(d - 3.6), s), davit(x - s * 0.9, H + LV, zd(d + 3.6), s)]
        for d, s in ((84, 1), (84, -1), (100, 1), (100, -1), (132, 1), (132, -1)):           # ventilators
            if inr(d):
                P.append(at(cyl(0.6, 0.6, 2.0, WHITE, seg=10), d, x=s * 7.0, y=H + 3 * LV))
        for dd, lev, l_, w_ in POOLS:                                  # the three pools on the terraces aft
            if inr(dd, 5):
                y0 = dy(dd) if lev == 0 else H + lev * LV
                P.append(at(pool(l_, w_), dd, y=y0))
                for s in (-1, 1):
                    P.append(tube([(s * (w_ / 2 + 1.5), y0 + 0.9, zd(dd - l_ / 2)), (s * (w_ / 2 + 1.5), y0 + 0.9, zd(dd + l_ / 2))], 0.04, WHITE, seg=4))
        for d, lev, wf in ((152, 2, 0.86), (172, 1, 0.97)):           # railings at the aft edge of each terrace
            if inr(d):
                x = B * wf / 2
                P.append(tube([(-x, H + lev * LV + 1.0, zd(d)), (x, H + lev * LV + 1.0, zd(d))], 0.05, WHITE, seg=4))
    # Foredeck and afterdeck: hatches, kingposts with derricks, windlass and capstans
    for d, w_ in ((20, 6.5), (34, 7.0), (194, 6.0)):
        if inr(d, 3):
            P.append(at(box(w_, 0.8, 5.0, BLACK), d, y=dy(d)))
            P.append(at(box(w_ - 0.5, 0.02, 4.5, DECK if built else DARK), d, y=dy(d) + 0.8))
    for d, h in ((27, 12.0), (41, 13.0), (199, 10.0)):
        if inr(d, 1):
            for s in (-1, 1):
                hh = h if built else rng.uniform(1.5, 3.5)
                P.append(at(cyl(0.45, 0.35, hh, WHITE, seg=10), d, x=s * 4.2, y=dy(d)))
                if built:
                    zt = zd(d) + (-7 if d < 100 else 7)
                    P.append(tube([(s * 4.2, dy(d) + 1.5, zd(d)), (s * 3.0, dy(d) + h * 0.7, zt)], 0.15, WHITE, seg=5))
    if inr(8, 2):
        for s in (-1, 1):
            P.append(at(cyl(0.9, 0.9, 1.4, BLACK, seg=12).R(0, 0, math.pi / 2).T(0.7, 0.9, 0), 8, x=s * 2.6, y=dy(8)))
            z = zd(5.5); x = s * (hw(z, H - 2) + 0.3)                                # anchors in the hawse pipes
            P.append(box(0.5, 3.2, 0.8, BLACK).T(x, H - 3.8, z))
            P.append(box(0.55, 0.6, 2.8, BLACK).T(x, H - 4.0, z))
        P.append(at(cyl(0.6, 0.5, 1.0, BLACK, seg=10), 13, y=dy(13)))
    for d in (204, 208):
        if inr(d, 1):
            for s in (-1, 1):
                P.append(at(cyl(0.6, 0.5, 1.0, BLACK, seg=10), d, x=s * 4.5, y=dy(d)))
    # Portholes in three rows along the black hull
    for y, a, b, sp in ((10.6, 18, 198, 2.3), (12.4, 14, 202, 2.3), (14.2, 10, 204, 2.3)):
        for d in np.arange(max(a, d0 + 1), min(b, d1 - 1), sp):
            z = zd(d)
            x = hw(z, y)
            if x < 2:
                continue
            for s in (-1, 1):
                P.append(quad(s * x, y, z, 0.42, 0.42, "+x" if s > 0 else "-x", DARK, eps=0.06))
                if not built and rng.random() < 0.3:
                    ln = rng.uniform(0.6, 1.8)
                    P.append(quad(s * hw(z, y - ln / 2), y - 0.2 - ln / 2, z, 0.3, ln, "+x" if s > 0 else "-x", (0.45, 0.24, 0.1), eps=0.05))
    return P


def rudder_props():
    zs = zd(L)
    parts = [box(0.8, 6.6, 4.4, BLACK).T(0, 1.0, zs + 4.2)]                         # rudder
    for s in (-1, 1):                                                                # twin screws on shaft brackets
        x, y, z = s * 5.4, 3.6, zs + 12.5
        parts.append(tube([(x, y, z + 0.5), (x * 0.75, y + 0.6, z + 14)], 0.35, BRONZE, seg=8))
        parts.append(tube([(x, y, z + 1.4), (x * 0.4, y + 2.6, z + 1.9)], 0.18, BLACK, seg=5))
        parts.append(cyl(0.55, 0.55, 1.1, BRONZE, seg=12).R(math.pi / 2, 0, 0).T(x, y, z - 0.6))
        for k in range(4):
            a = 2 * math.pi * k / 4 + (0.4 if s > 0 else 0)
            parts.append(box(2.3 * 2 * 0.42, 0.16, 1.2, BRONZE, bottom=False).T(2.3 * 0.42 + 0.4, 0, 0).R(0.35 * s, 0, 0).R(0, 0, a).T(x, y, z))
    return merge(parts)


FONT = {  # 5×7 block letters, rows from the top
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
}


def name_on_bow(side, text="ANDREA DORIA", d_start=7.0, px=0.26):
    """White block letters on the bow, readable from outside on each side."""
    out, col = [], 0
    n = len(text) * 6 * px
    y_top = dy(d_start) - 0.9
    for ch in text:
        if ch == " ":
            col += 3; continue
        for r_, row in enumerate(FONT[ch]):
            for c_, bit in enumerate(row):
                if bit != "1":
                    continue
                y = y_top - r_ * px
                dd = d_start + (col + c_) * px if side > 0 else d_start + n - (col + c_) * px
                z = zd(dd); x = side * (hw(z, y) + 0.07)
                out.append(quad(x, y, z, px * 0.95, px * 0.95, "+x" if side > 0 else "-x", WHITE, eps=0.0))
        col += 6
    return out


# ── The wreck ────────────────────────────────────────────────────────────────
BREAK = 48.0          # the bow broke off forward of the bridge
CRACK = 181.0         # the crack across the stern


def torn_edge(d, direction, n, seed):
    """Plates torn and bent at a break (ship coordinates)."""
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = r.uniform(1.0, H + 1)
        wdt = hw(zd(d), min(y, H - 0.1))
        p = box(r.uniform(1.0, 3.5), 0.12, r.uniform(1.0, 3.5), BLACK if y < H else WHITE, bottom=False)
        crumple(p, 0.5, 0.35, seed + i)
        out.append(p.R(r.uniform(-0.7, 0.7), r.uniform(-0.5, 0.5), r.uniform(-0.9, 0.9))
                   .T(r.choice([-1, 1]) * r.uniform(wdt * 0.4, wdt), y, zd(d) + direction * r.uniform(0.3, 3)))
    inner = box(B * 0.66, H * 0.7, 0.15, DARK).T(0, 2.2, zd(d) - direction * 2.0)          # the dark inside at the break
    crumple(inner, 0.6, 0.25, seed + 99)
    return out + [inner]


def deck_tears(d0, d1, n, seed):
    """Where the superstructure sheared off the upper deck: stumps of walls, bent plates (ship coordinates)."""
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        d = r.uniform(d0, d1)
        p = box(0.12, r.uniform(0.8, 2.6), r.uniform(1.5, 5), WHITE if r.random() < 0.7 else DECK)
        crumple(p, 0.4, 0.4, seed + i)
        out.append(p.R(0, 0, r.uniform(-0.7, 0.7)).T(r.uniform(-B * 0.42, B * 0.42), H - 0.1, zd(d)))
    return out


def deck_openings(d0, d1, n, seed):
    """On the exposed upper deck: dark holes (stairwells, skylights, torn plating) with beams across some of them."""
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        d, w_, l_ = r.uniform(d0, d1), r.uniform(1.5, 6), r.uniform(2, 9)
        x = r.uniform(-B * 0.4 + w_ / 2, B * 0.4 - w_ / 2)
        out.append(quad(x, dy(d), zd(d), w_, l_, "+y", DARK, eps=0.04))
        if r.random() < 0.5:
            for k in np.arange(-l_ / 2 + 0.7, l_ / 2, 1.4):
                out.append(at(box(w_, 0.3, 0.25, (0.3, 0.24, 0.18)), d + k, x=x, y=dy(d)))
    return out


def stockholm_hole():
    """The wedge torn by the Stockholm's bow: starboard side, under the bridge (now against the sand)."""
    x = -(B / 2 + 0.06)
    tri = Mesh([(x, H + 0.2, zd(64)), (x, 1.2, zd(71)), (x, H + 0.2, zd(79))], [(0, 2, 1)], DARK)
    bent = []
    for i in range(10):
        p = box(0.12, rng.uniform(1.5, 4), rng.uniform(1, 3), BLACK, bottom=False)
        crumple(p, 0.5, 0.4, 900 + i)
        y = rng.uniform(3, H)
        bent.append(p.R(rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.6), 0).T(x + rng.uniform(0.3, 2.0), y, zd(rng.uniform(66, 77))))
    return [tri] + bent


def stern_crack():
    """The crack across the stern: dark gaps on the port side (now the top) and on the deck."""
    out = []
    for y0 in np.arange(3.0, H, 1.6):
        z = zd(CRACK + rng.uniform(-0.6, 0.6))
        out.append(quad(hw(z, y0 + 0.8), y0 + 0.8, z, 0.7, 1.7, "+x", DARK, eps=0.08))
    out.append(quad(rng.uniform(-3, 3), dy(CRACK) + 0.05, zd(CRACK), B * 0.85, 0.7, "+y", DARK, eps=0.02))
    return out


def roll_and_settle(m, flat, sink=2.0):
    """Starboard side down (deck facing −X = east when the bow points north), then the hull flattens: the
    side now on top sags towards the sand, the keel side holds up better, the deck side spreads out."""
    m.R(0, 0, math.pi / 2)
    m.T(H * 0.55, B / 2 - sink, 0)
    x, y = m.v[:, 0].copy(), m.v[:, 1].copy()
    f = flat + (1 - flat) * 0.45 * sstep(-6, 9, x)
    yn = np.where(y > 0, y * f, y)
    m.v[:, 0] = x - 0.25 * (y - yn) * sstep(10, -6, x)
    m.v[:, 1] = yn
    return m


def growth(m, seed, amount=1.0, rust=0.5):
    """Paint faded under seventy years of growth, rust and silt, then what grows on wrecks in these cold waters:
    olive hydroid turf and white anemone clumps."""
    red = np.clip((m.c[:, 0] - m.c[:, 2]) * 3, 0, 1)[:, None] * 0.7                 # antifouling red fades most
    m.c = m.c * (1 - 0.5 * amount) + np.array((0.3, 0.28, 0.24)) * 0.5 * amount
    m.c = m.c * (1 - red) + np.array((0.33, 0.25, 0.19)) * red
    weather(m, rust=rust, sediment=0.45, streaks=0.3, seed=seed, rust_col=(0.42, 0.22, 0.1), sed_col=(0.5, 0.49, 0.42))
    p = m.v
    g = sstep(0.4, 0.7, fbm(p, 0.1, seed + 60, 2)) * 0.35 * amount
    m.c = m.c * (1 - g[:, None]) + np.array((0.32, 0.34, 0.24)) * g[:, None]
    a = sstep(0.6, 0.78, fbm(p, 0.3, seed + 50, 3)) * 0.8 * amount
    m.c = m.c * (1 - a[:, None]) + np.array((0.86, 0.83, 0.76)) * a[:, None]
    return m


# Main hull: from the break to the stern
hullw = merge(ship_part(BREAK + 1.5, L) + [rudder_props()] + torn_edge(BREAK + 1.5, -1, 22, 300) + deck_tears(52, 172, 46, 400) + deck_openings(52, 175, 26, 450)
              + stockholm_hole() + stern_crack())
dfs = L / 2 - hullw.v[:, 2]
crumple(hullw, 2.6, 0.35, 11, sstep(56, BREAK + 1.5, dfs))                        # torn edges at the break
crumple(hullw, 1.6, 0.3, 12, np.exp(-((dfs - CRACK) / 4.0) ** 2))                   # buckled at the stern crack
roll_and_settle(hullw, 0.56, 2.0)
hullw.v[:, 1] -= 1.2 * sstep(zd(CRACK - 2), zd(CRACK + 10), hullw.v[:, 2]) * sstep(0, 12, hullw.v[:, 1])   # stern sagging past the crack
crumple(hullw, 1.7, 0.12, 13, sstep(4, -8, hullw.v[:, 0]))                          # the deck side, where the superstructure tore away
yy = hullw.v[:, 1]
hullw.v[:, 1] -= (2.6 * np.exp(-((hullw.v[:, 2] - zd(112)) / 32) ** 2) + 1.4 * np.exp(-((hullw.v[:, 2] - zd(160)) / 12) ** 2)) * sstep(2, 15, yy)   # the side on top sagging in
crumple(hullw, 0.9, 0.07, 15, sstep(8, 15, hullw.v[:, 1]))                         # its plating buckled
crumple(hullw, 0.35, 0.08, 14)                                                      # dents everywhere
growth(hullw, 21)

# Bow section: broken off forward of the bridge, better preserved, slewed a little and a few metres away
bow = merge(ship_part(0, BREAK) + torn_edge(BREAK, 1, 18, 500))
dfb = L / 2 - bow.v[:, 2]
crumple(bow, 2.4, 0.35, 31, sstep(BREAK - 7, BREAK, dfb))
roll_and_settle(bow, 0.72, 2.4)
zb = zd(BREAK)
bow.T(0, 0, -zb).R(0, 0.08, 0).T(-1.2, -0.3, zb + 3.5)
crumple(bow, 0.3, 0.08, 32)
growth(bow, 41)

# ── Debris field: the superstructure on the seabed beside the hull (east, −X), nets over everything ──
DECK_X = -H * 0.45 - 1.0          # world x of the deck edge after the roll


def plate(w, l, col, seed, nx=6, nz=4, amp=0.8):
    g = grid(-w / 2, w / 2, -l / 2, l / 2, nx, nz, lambda X, Z: np.zeros_like(X), col)
    return crumple(g, amp, 0.25, seed)


def wall_panel(l, h, seed, col=WHITE):
    """A fallen superstructure wall, lying on the sand with its row of windows."""
    ps = [grid(-l / 2, l / 2, -h / 2, h / 2, max(2, int(l / 2)), 2, lambda X, Z: np.zeros_like(X), col)]
    for x in np.arange(-l / 2 + 1.3, l / 2 - 1, 2.5):
        ps.append(quad(x, 0, 0.1, 1.6, 1.1, "+y", DARK, eps=0.04))
    return crumple(merge(ps), 0.6, 0.3, seed)


deb = []
for lev in range(3):                                                # decks of the three tiers, fallen and stacked
    a, b, wf = SUP[lev]
    d = a
    while d < b - 3:
        ln = min(rng.uniform(8, 16), b - d)
        w_ = rng.uniform(5, 10)
        p = plate(w_, ln, DECK if rng.random() < 0.5 else WHITE, 1000 + lev * 50 + int(d), nx=5, nz=max(3, int(ln / 3)))
        x = DECK_X - 2 - lev * 3.5 - rng.uniform(0, 7)
        y = 0.4 + (2 - lev) * 0.9 * sstep(-30, -10, x) + rng.uniform(0, 1.2)
        deb.append(p.R(rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(-0.3, 0.3)).T(x - w_ / 2, y, zd(d + ln / 2)))
        d += ln + rng.uniform(-1.5, 2)
for i in range(22):                                                 # walls with their windows
    d = rng.uniform(55, 170)
    l_ = rng.uniform(6, 14)
    p = wall_panel(l_, LV, 1300 + i)
    x = DECK_X - rng.uniform(2, 20)
    deb.append(p.R(rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3) + (math.pi / 2 if rng.random() < 0.3 else 0), 0).T(x, rng.uniform(0.6, 2.5), zd(d)))
# the funnel, lying on its side in a crushed heap
fn = funnel().R(0, 0, math.pi / 2)
fn.S(1, 0.6, 1)
crumple(fn, 1.2, 0.2, 1400)
deb.append(fn.T(DECK_X - 4.0, 2.0, zd(FUNNEL_D)))
# the bridge: wheelhouse and wings in a twisted heap
br = merge([box(13, 2.6, 7, WHITE)] + [quad(x, 1.6, 3.5, 0.9, 1.0, "+z", DARK) for x in np.arange(-5.5, 5.6, 1.1)]
           + [box(B, 0.4, 3.0, WHITE).T(0, 2.2, 2.0)])
br.S(1, 0.55, 1).R(0.12, 0.35, 0.18)
crumple(br, 1.4, 0.2, 1500)
deb.append(br.T(DECK_X - 16, 1.2, zd(65)))
# masts and kingposts lying on the sand, pointing away from the hull
for d, ln, r_ in ((FORE_D, 26, 0.35), (MAIN_D, 20, 0.33), (27, 11, 0.45), (41, 12, 0.45)):
    a_ = rng.uniform(-0.3, 0.3)
    x0 = DECK_X - (0.5 if d < 45 else 3.0)
    deb.append(tube([(x0, 1.5, zd(d)), (x0 - ln * math.cos(a_), 0.35, zd(d) + ln * math.sin(a_))], r_, WHITE, seg=8))
# davits, twisted, and lifeboat chocks along the boat deck line
for i in range(14):
    d = rng.choice(BOATS) + rng.uniform(-4, 4)
    x, y, z = DECK_X - rng.uniform(3, 14), rng.uniform(0.3, 1.5), zd(d)
    tw = tube([(0, 0, 0), (0, 2.6, 0), (0.8, 3.3, 0), (1.5, 3.0, 0)], 0.15, WHITE, seg=5)
    crumple(tw, 0.5, 0.4, 1600 + i)
    deb.append(tw.R(rng.uniform(1.0, 1.6), rng.uniform(0, 6.28), 0).T(x, y, z))
# the first- and cabin-class pools, tilted and half full of debris
for dd, lev, l_, w_ in POOLS[:2]:
    p = merge([box(w_ + 1.2, 0.35, l_ + 1.2, TILE), quad(0, 0.35, 0, w_, l_, "+y", (0.4, 0.42, 0.36), eps=0.02)])
    crumple(p, 0.5, 0.25, 1700 + dd)
    deb.append(p.R(rng.uniform(-0.3, 0.3), 0, rng.uniform(-0.4, -0.15)).T(DECK_X - 6 - lev * 3, 0.9, zd(dd)))
# ventilators, deck fittings, smaller pieces, beams
for i in range(10):
    d = rng.uniform(60, 165)
    deb.append(cyl(0.6, 0.6, 2.0, WHITE, seg=8).R(math.pi / 2, rng.uniform(0, 6.28), 0).T(DECK_X - rng.uniform(2, 18), 0.6, zd(d)))
for i in range(30):
    d = rng.uniform(50, 175)
    deb.append(box(0.35, 0.5, rng.uniform(4, 11), (0.3, 0.24, 0.18)).R(0, rng.uniform(-0.8, 0.8), rng.uniform(-0.3, 0.3))
               .T(DECK_X - 1 - min(rng.exponential(8), 30), rng.uniform(0, 1.2), zd(d)))
for i in range(170):
    d = rng.uniform(46, 180)
    s = rng.uniform(0.3, 1.5)
    col = (WHITE, DECK, (0.25, 0.2, 0.15), (0.35, 0.3, 0.22))[rng.integers(4)]
    deb.append(box(s, s * rng.uniform(0.4, 1.0), s * rng.uniform(0.6, 1.6), col).R(rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28), 0)
               .T(DECK_X - 0.5 - min(rng.exponential(7), 32), rng.uniform(-0.1, 0.8), zd(d)))
for i in range(14):                                                 # hull plates fallen on the keel side and at the ends
    d = rng.choice([rng.uniform(40, 60), rng.uniform(170, 200), rng.uniform(60, 170)])
    p = plate(rng.uniform(2, 6), rng.uniform(1.5, 4), BLACK, 1800 + i, nx=3, nz=3, amp=0.5)
    side = 1 if rng.random() < 0.5 else -1
    x = H * 0.55 + rng.uniform(2, 9) if side > 0 else DECK_X - rng.uniform(1, 10)
    deb.append(p.R(rng.uniform(-0.3, 0.3), rng.uniform(0, 6.28), rng.uniform(-0.3, 0.3)).T(x, 0.15, zd(d)))
debris = merge(deb)
growth(debris, 61, amount=0.75, rust=0.4)

# ── Nets: lost trawl and gill nets draped over the hull and the debris ─────────
wreck_v = np.vstack([hullw.v, bow.v, debris.v])
X0, Z0, CS = -50.0, -115.0, 1.5
NX, NZ = int(80 / CS), int(232 / CS)
hm = np.zeros((NZ, NX))
ix = np.clip(((wreck_v[:, 0] - X0) / CS).astype(int), 0, NX - 1)
iz = np.clip(((wreck_v[:, 2] - Z0) / CS).astype(int), 0, NZ - 1)
np.maximum.at(hm, (iz, ix), wreck_v[:, 1])
for _ in range(2):                                                   # fill the gaps between vertices
    p_ = np.pad(hm, 1, mode="edge")
    hm = np.max([p_[1 + a:NZ + 1 + a, 1 + b:NX + 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)
for _ in range(2):                                                   # soften: the net sags between high points
    p_ = np.pad(hm, 1, mode="edge")
    hm = np.mean([p_[1 + a:NZ + 1 + a, 1 + b:NX + 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0) * 0.6 + hm * 0.4


def hm_at(X, Z):
    fx = np.clip((X - X0) / CS - 0.5, 0, NX - 1.001); fz = np.clip((Z - Z0) / CS - 0.5, 0, NZ - 1.001)
    i, j = fz.astype(int), fx.astype(int); u, v = fz - i, fx - j
    return (hm[i, j] * (1 - u) * (1 - v) + hm[i + 1, j] * u * (1 - v) + hm[i, j + 1] * (1 - u) * v + hm[i + 1, j + 1] * u * v)


def net_patch(cx, cz, w, l, seed, cell=0.55):
    nx, nz = max(4, int(w / cell)), max(4, int(l / cell))
    g = grid(cx - w / 2, cx + w / 2, cz - l / 2, cz + l / 2, nx, nz, lambda X, Z: hm_at(X, Z) + 0.3, NETC)
    k = np.arange(len(g.f)) // 2
    ci, cj = k % nx, k // nx
    edge = np.sqrt(((ci - nx / 2) / (nx / 2)) ** 2 + ((cj - nz / 2) / (nz / 2)) ** 2) + 0.4 * (fbm(g.v[g.f[:, 0]], 0.2, seed) - 0.5)
    a, b, c = g.v[g.f[:, 0]], g.v[g.f[:, 1]], g.v[g.f[:, 2]]
    nrm = np.cross(b - a, c - a)
    flat = np.abs(nrm[:, 1]) / (np.linalg.norm(nrm, axis=1) + 1e-9) > 0.45
    keep = ((ci % 3 == 0) | (cj % 3 == 0)) & (edge < 1.0) & flat      # strands of mesh, ragged outline, lying on top
    g.f = g.f[keep]
    crumple(g, 0.4, 0.3, seed)
    return g


nets_ = []
for cz, w, l in ((zd(30), 16, 14), (zd(62), 18, 12), (zd(95), 20, 16), (zd(128), 16, 12), (zd(160), 18, 14), (zd(192), 14, 12)):
    row = hm[int(np.clip((cz - Z0) / CS, 0, NZ - 1))]
    cx = X0 + (np.argmax(row) + 0.5) * CS + rng.uniform(-3, 3)
    nets_.append(net_patch(cx, cz, w, l, 2000 + int(cz)))
for cz, cx in ((zd(FUNNEL_D), DECK_X - 14), (zd(65), DECK_X - 18)):    # over the funnel and the bridge heap
    nets_.append(net_patch(cx, cz, 12, 10, 2100 + int(cz)))
for i in range(16):                                                  # float lines and ropes hanging down the sides
    cz = rng.uniform(-L / 2 + 20, L / 2 - 10)
    x0 = rng.uniform(-4, 8)
    x1 = x0 + rng.choice([-1, 1]) * rng.uniform(6, 14)
    pts = [(x, float(hm_at(np.array([x]), np.array([cz]))[0]) + 0.25, cz + rng.uniform(-0.5, 0.5)) for x in np.linspace(x0, x1, 7)]
    nets_.append(tube(pts, 0.05, (0.3, 0.3, 0.26), seg=4))
nets = merge(nets_)
nets.paint(lambda p: np.array(NETC) + (fbm(p, 0.3, 77)[:, None] - 0.5) * 0.12)

# ── Sand banked against the hull (the seabed itself comes from GEBCO) ─────────
KEEL_X = H * 0.55


def sand_h(X, Z):
    along = sstep(L / 2 + 6, L / 2 - 12, np.abs(Z - 1.5))
    keel = 1.5 * np.exp(-((X - KEEL_X - 1.5) / 4.5) ** 2)
    deck = 0.9 * np.exp(-((X - DECK_X + 0.5) / 4.0) ** 2)
    bumps = 0.25 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.2, 5).reshape(X.shape)
    return (np.maximum(keel, deck) + bumps * sstep(0.2, 1.0, np.maximum(keel, deck))) * along - 0.6


sand = grid(DECK_X - 10, KEEL_X + 12, -L / 2 - 14, L / 2 + 16, 30, 96, sand_h, (0.58, 0.54, 0.45))
sand.paint(lambda p: (0.6, 0.56, 0.46) + (fbm(p, 0.12, 9)[:, None] - 0.5) * 0.07)
sand.f = sand.f[sand.v[sand.f][:, :, 1].max(axis=1) > 0.15]

# ── The ship as she sailed (model_built.glb): whole, clean, keel on y = 0, centred ──
ship = merge(ship_part(0, L, built=True) + [rudder_props()] + name_on_bow(1) + name_on_bow(-1))
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
out = ROOT / "models" / "item" / "Q493798" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
tris, size = write_glb(out, [("hull", hullw), ("bow_section", bow), ("debris_field", debris), ("nets", nets), ("sediment", sand)])
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB")
for n, m in (("hull", hullw), ("bow_section", bow), ("debris_field", debris), ("nets", nets), ("sediment", sand)):
    print(f"  {n}: {m.tris:,} tris, x {m.v[:, 0].min():.0f}..{m.v[:, 0].max():.0f}, y {m.v[:, 1].min():.0f}..{m.v[:, 1].max():.0f}, z {m.v[:, 2].min():.0f}..{m.v[:, 2].max():.0f}")
tris2, size2 = write_glb(out.with_name("model_built.glb"), [("main", ship)])
print(f"model_built.glb: {tris2:,} triangles, {size2 / 2**20:.2f} MB, y {ship.v[:, 1].min():.1f}..{ship.v[:, 1].max():.1f}, z {ship.v[:, 2].min():.1f}..{ship.v[:, 2].max():.1f}")
