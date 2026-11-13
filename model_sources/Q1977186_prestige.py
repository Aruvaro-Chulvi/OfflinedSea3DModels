"""
MV Prestige (Q1977186) — the wreck of the single-hull tanker off Galicia, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: single-hull Aframax tanker of 1976, length 243.5 m, beam 34.4 m, depth 18.7 m, draught 14 m
(bulbous bow, forecastle, long tank deck with a centre catwalk and cargo pipelines, manifold with two hose
cranes amidships, accommodation block and funnel aft, one propeller).
The wreck, after the published surveys: she broke in two on 19 November 2002, some 250 km west of Cape
Finisterre, and sank on the south-western flank of the Galicia Bank: the bow section at 3,830 m, the stern
section at 3,565 m, a few kilometres apart. In 2003 the French submersible Nautile sealed the leaking cracks,
and in 2004 the fuel left in the tanks was pumped out through fittings fixed on the deck. Headings, lengths
of each section and the exact layout of cracks, patches and fittings are approximate.

    python model_sources/Q1977186_prestige.py   -> models/item/Q1977186/model.glb (+ model_built.glb)

The stern section is a separate node ("stern_section") with its own position: in this file it sits at its
offset from the bow (north +Z, west +X); Offlined places it with "pieces" in meta.json on its own seabed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, weather, sstep, fbm, write_glb, yaw_quat  # noqa: E402

rng = np.random.default_rng(2002)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B, D, T = 243.5, 34.4, 18.7, 14.0
FC = 2.8                                   # forecastle height above the main deck
LV = 2.8                                   # one deck of the accommodation
BLACK, RED, WHITE = (0.07, 0.07, 0.08), (0.45, 0.12, 0.08), (0.84, 0.83, 0.79)
DECKC, PIPE, DARK = (0.4, 0.2, 0.15), (0.38, 0.42, 0.34), (0.025, 0.025, 0.03)
ORANGE, STEEL, BRONZE, OIL = (0.9, 0.42, 0.1), (0.45, 0.45, 0.45), (0.55, 0.4, 0.2), (0.03, 0.025, 0.02)
FUNNEL = (0.1, 0.1, 0.11)
WL = 9.5                                   # top of the red bottom paint (boot-top)


def zd(d):
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):                                # very full hull: short entrance, short run
    w = 1.0
    if t > 0.86:
        w = math.sin(((1 - t) / 0.14) * math.pi / 2) ** 0.55
    if t < 0.16:
        w = min(w, 0.5 + 0.5 * math.sin((t / 0.16) * math.pi / 2))
    return max(w, 0.004)


def yb_of(t):                               # keel line: cut up at the stern (propeller and rudder), stem at the bow
    y = 0.0
    if t < 0.12:
        y = D * 0.55 * ((0.12 - t) / 0.12) ** 1.5
    if t > 0.992:
        y = max(y, D * 0.75 * ((t - 0.992) / 0.008) ** 2)
    return y


def yd_of(t):
    return D * (1 + 0.02 * (2 * t - 1) ** 2)


def k_of(t):
    return 2.5 + 9.0 * w_of(t) ** 4


def hw(z, y):
    t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    if y >= yd:
        return B / 2 * w
    u = max(0.0, (y - yb) / max(0.1, yd - yb))
    return B / 2 * w * (1 - (1 - u) ** k)


def hull(d0, d1, NP=26):
    ns = max(2, int(math.ceil((d1 - d0) / 1.3)))
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
    if d1 >= L - 0.01:                                       # transom
        c0 = len(v); z, w, yb, yd, k = st[0]
        v.append((0, (yb + yd) / 2, z))
        f += [(c0, j + 1, j) for j in range(W - 1)]
    m = Mesh(v, f)
    deck_top = np.array([p[1] >= yd_of(t_of(p[2])) - 0.02 and abs(p[0]) < B / 2 * w_of(t_of(p[2])) - 0.05 for p in m.v])
    m.paint(lambda p: np.where((p[:, 1] < WL)[:, None], RED, BLACK))
    m.c[deck_top] = DECKC
    return m


def at(m, d, x=0.0, y=0.0):
    return m.T(x, y, zd(d))


def deck_y(d):
    return yd_of(t_of(zd(d)))


def bulb():
    """Bulbous bow: a rounded nose below the waterline, ahead of the stem."""
    prof = [(0.05, -7), (1.6, -6.2), (2.6, -4.6), (3.2, -2.4), (3.3, 0), (3.0, 2.2), (2.2, 4.0), (1.0, 5.2), (0.05, 5.7)]
    m = lathe(prof, RED, seg=18, axis="z")
    m.S(1, 1.2, 1)
    return m.T(0, 6.0, zd(0) - 2.0)


def rudder_prop():
    z = zd(L) + 4.0
    parts = [box(0.9, 10.5, 6.0, RED).T(0, 2.0, z - 3.5)]                       # rudder
    parts.append(cyl(0.9, 0.9, 1.8, BRONZE, seg=14).R(math.pi / 2, 0, 0).T(0, 4.6, z + 1.0))
    for k in range(5):                                                           # five-bladed propeller
        a = 2 * math.pi * k / 5
        parts.append(box(3.1, 0.2, 1.6, BRONZE, bottom=False).T(2.0, 0, 0).R(0.4, 0, 0).R(0, 0, a).T(0, 4.6, z + 1.9))
    return merge(parts)


def ship_part(d0, d1, built=False):
    """Everything of the ship between d0 and d1 (distance from the bow)."""
    P = [hull(d0, d1)]
    inr = lambda d, m=0: d0 + m <= d <= d1 - m
    if d0 <= 0.5:
        P.append(bulb())
    # forecastle: raised deck, bulwark, mooring winches, bollards, foremast
    a2, b2 = max(0, d0), min(22, d1)
    if b2 - a2 > 1:
        n = max(2, int((b2 - a2) / 1.3))
        zs = [zd(b2) + (zd(a2) - zd(b2)) * i / n for i in range(n + 1)]
        for s in (-1, 1):
            v, f = [], []
            for z in zs:
                x = s * B / 2 * w_of(t_of(z)); y0 = yd_of(t_of(z))
                v += [(x, y0 - 0.05, z), (x, y0 + FC + 1.1, z)]
            for i in range(n):
                a = 2 * i
                f += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)] if s < 0 else [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
            P.append(Mesh(v, f, BLACK))
        v, f = [], []
        for z in zs:
            x = B / 2 * w_of(t_of(z)); y0 = yd_of(t_of(z)) + FC
            v += [(-x, y0, z), (x, y0, z)]
        for i in range(n):
            a = 2 * i; b = a + 2; c = a + 1; d = b + 1
            f += [(a, b, c), (c, b, d)]
        P.append(Mesh(v, f, DECKC))
        x22 = B / 2 * w_of(t_of(zd(b2)))
        P.append(Mesh([(-x22, deck_y(b2), zd(b2)), (x22, deck_y(b2), zd(b2)), (x22, deck_y(b2) + FC, zd(b2)), (-x22, deck_y(b2) + FC, zd(b2))], [(0, 2, 1), (0, 3, 2)], WHITE))
        for d, x in ((12, 5.5), (12, -5.5), (17, 7), (17, -7)):
            if inr(d):
                P.append(at(cyl(1.0, 1.0, 3.2, STEEL, seg=12).R(0, 0, math.pi / 2).T(1.6, 0.9, 0), d, x=x, y=deck_y(d) + FC))
        for d in (5, 9, 15, 20):
            for s in (-1, 1):
                if inr(d):
                    x = s * (B / 2 * w_of(t_of(zd(d))) - 1.6)
                    P += [at(cyl(0.35, 0.35, 0.9, BLACK, seg=8), d, x=x, y=deck_y(d) + FC), at(cyl(0.35, 0.35, 0.9, BLACK, seg=8), d + 0.9, x=x, y=deck_y(d) + FC)]
        if inr(8):
            y0 = deck_y(8) + FC
            P.append(at(cyl(0.45, 0.3, 11, (0.75, 0.6, 0.2), seg=8), 8, y=y0))                # foremast (yellow)
            P.append(at(box(2.4, 0.15, 1.2, (0.75, 0.6, 0.2)), 8, y=y0 + 9))
    # tank deck: catwalk on posts, four cargo pipelines, tank hatches, vent risers, railing
    a2, b2 = max(24, d0), min(202, d1)
    if b2 - a2 > 2:
        yc = lambda d: deck_y(d) + 2.6
        P.append(at(box(1.3, 0.15, b2 - a2, STEEL), (a2 + b2) / 2, y=yc((a2 + b2) / 2)))
        for s in (-1, 1):
            P.append(tube([(s * 0.65, yc(d) + 1.0, zd(d)) for d in np.arange(a2, b2 + 0.1, 6)], 0.04, STEEL, seg=4))
        for d in np.arange(a2 + 2, b2, 6.0):
            P.append(at(box(0.25, 2.6, 0.25, STEEL), d, y=deck_y(d)))
        for x in (-7.5, -5.5, 5.5, 7.5):
            P.append(tube([(x, deck_y(d) + 0.7, zd(d)) for d in np.arange(a2 + 1, b2, 6)], 0.36, PIPE, seg=8))
        for d in np.arange(a2 + 6, b2 - 2, 14.5):
            for s in (-1, 1):
                P.append(at(cyl(0.9, 0.9, 0.55, PIPE, seg=12), d, x=s * 3.0, y=deck_y(d)))
        for d in (60, 100, 140, 180):
            if inr(d, 1):
                for s in (-1, 1):
                    P.append(at(cyl(0.22, 0.22, 7.5, PIPE, seg=6), d, x=s * 10.5, y=deck_y(d)))
        for s in (-1, 1):
            P.append(tube([(s * (B / 2 - 0.4), deck_y(d) + 1.05, zd(d)) for d in np.arange(max(22, d0), min(203, d1) + 0.1, 3.0)], 0.04, STEEL, seg=4))
    # manifold amidships with its two hose cranes
    if inr(120, 3):
        for dz in (-1.2, 1.2):
            P.append(tube([(-15.5, deck_y(120) + 1.2, zd(120 + dz)), (15.5, deck_y(120) + 1.2, zd(120 + dz))], 0.3, PIPE, seg=8))
        for s in (-1, 1):
            for dz in (-1.2, 1.2):
                P.append(at(box(0.9, 0.9, 0.7, (0.6, 0.15, 0.1)), 120 + dz, x=s * 14.5, y=deck_y(120) + 0.8))
        for d, s in ((115, 1), (125, -1)):
            y0 = deck_y(d)
            P.append(at(cyl(0.7, 0.55, 7.0, WHITE, seg=10), d, x=s * 4.5, y=y0))
            P.append(tube([(s * 4.5, y0 + 6.5, zd(d)), (s * 13.0, y0 + 11.0, zd(d) + 6)], 0.22, WHITE, seg=6))
    # accommodation block aft: five decks, bridge with wings, windows
    levels = [(203, 230, 0.9), (204, 229, 0.86), (205, 228, 0.8), (205, 227, 0.74), (203, 215, 0.6)]
    for i, (a, b, wf) in enumerate(levels):
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 > 1:
            y0 = deck_y((a + b) / 2) + i * LV
            P.append(at(box(B * wf, LV, b2 - a2, WHITE), (a2 + b2) / 2, y=y0))
            for s in (-1, 1):
                for d in np.arange(a2 + 1.5, b2 - 1, 2.4):
                    P.append(quad(s * B * wf / 2, y0 + 1.6, zd(d), 1.0, 0.9, "+x" if s > 0 else "-x", DARK))
            for x in np.arange(-B * wf / 2 + 1.2, B * wf / 2 - 1, 2.2 if i < 4 else 1.5):
                P.append(quad(x, y0 + 1.6, zd(a2), 1.0 if i < 4 else 1.25, 0.9 if i < 4 else 1.2, "+z", DARK))
    if inr(209):
        yb_ = deck_y(209) + 4 * LV
        P.append(at(box(B * 1.0, 0.4, 4.0, WHITE), 205, y=yb_ + LV - 0.4))                      # bridge wings
        P.append(at(cyl(0.35, 0.25, 7.0, WHITE, seg=8), 210, y=yb_ + LV))                         # radar mast
        P.append(at(box(5.0, 0.3, 0.5, WHITE), 210, y=yb_ + LV + 5.0))
        P.append(at(box(4.2, 0.25, 0.45, STEEL), 210, y=yb_ + LV + 6.6))                          # radar scanner
    # funnel behind the accommodation, engine casing, lifeboats (enclosed, orange)
    if inr(234, 2):
        y0 = deck_y(234) + LV
        P.append(at(cyl(3.6, 3.3, 19.0, FUNNEL, seg=20, sx=0.72), 234, y=y0))
        P.append(at(cyl(3.3, 3.3, 0.05, DARK, seg=20, sx=0.72), 234, y=y0 + 19.02))
        P.append(at(box(B * 0.6, LV, 12, WHITE), 236.5, y=deck_y(236.5)))
    if inr(224):
        for s in (-1, 1):
            boat = lathe([(0.05, -4.2), (1.0, -3.6), (1.45, -2.0), (1.5, 0), (1.45, 2.0), (1.0, 3.6), (0.05, 4.2)], ORANGE, seg=12, axis="z")
            boat.S(1, 0.95, 1)
            y0 = deck_y(224) + 2 * LV + 1.2
            P.append(boat.T(s * (B / 2 - 0.2), y0, zd(224)))
            P.append(tube([(s * (B / 2 - 2.5), y0 - 1.4, zd(221.5)), (s * (B / 2 - 2.5), y0 + 2.2, zd(221.5)), (s * (B / 2 + 0.4), y0 + 2.6, zd(221.5))], 0.18, STEEL, seg=5))
            P.append(tube([(s * (B / 2 - 2.5), y0 - 1.4, zd(226.5)), (s * (B / 2 - 2.5), y0 + 2.2, zd(226.5)), (s * (B / 2 + 0.4), y0 + 2.6, zd(226.5))], 0.18, STEEL, seg=5))
    # stern mooring deck: winches and bollards
    for d, x in ((239, 6), (239, -6)):
        if inr(d, 1):
            P.append(at(cyl(1.0, 1.0, 3.0, STEEL, seg=12).R(0, 0, math.pi / 2).T(1.5, 0.9, 0), d, x=x, y=deck_y(d) + LV))
    # the name at the bow and on the stern, white letters (as simple plates)
    return P


FONT = {  # 5×7 block letters, rows from the top
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
}


def lettering(text, d_start, side, y_top, px=0.3, along=+1, transom=False):
    """White block letters painted on the hull: on a side (going aft from d_start) or across the transom."""
    out = []
    col = 0
    for ch in text:
        if ch == " ":
            col += 3; continue
        for r_, row in enumerate(FONT[ch]):
            for c_, bit in enumerate(row):
                if bit != "1":
                    continue
                y = y_top - r_ * px
                if transom:
                    x = (col + c_) * px - len(text) * 6 * px / 2
                    z = zd(L) + 0.0
                    out.append(quad(-x, y, z - 0.06, px * 0.95, px * 0.95, "-z", WHITE, eps=0.0))
                else:
                    # reading from bow to stern on the starboard side, from stern to bow on port
                    dd = d_start + ((col + c_) * px if side > 0 else -(col + c_) * px) * along   # +X side reads aft, −X side reads forward
                    z = zd(dd); x = side * (hw(z, y) + 0.07)
                    out.append(quad(x, y, z, px * 0.95, px * 0.95, "+x" if side > 0 else "-x", WHITE, eps=0.0))
        col += 6
    return out


def name_plates(d, side, yb):
    """'PRESTIGE' on the bow, readable from outside on both sides."""
    n = len("PRESTIGE") * 6 * 0.3
    return lettering("PRESTIGE", d if side > 0 else d + n, side, yb)


def oil_stain(m, centres, radius, seed):
    """Darken the vertices near some points (fuel residue)."""
    k = np.zeros(len(m.v))
    for c in centres:
        d = np.linalg.norm(m.v - np.asarray(c), axis=1)
        k = np.maximum(k, sstep(radius, radius * 0.25, d) * sstep(0.35, 0.65, fbm(m.v, 0.12, seed)))
    m.c = m.c * (1 - k[:, None] * 0.9) + np.array(OIL) * k[:, None] * 0.9
    return m


def cracks_and_patches(d_list, side, ys, built):
    out = []
    for d, y in zip(d_list, ys):
        z = zd(d); x = side * hw(z, y)
        out.append(quad(x, y, z, 0.35, rng.uniform(3, 7), "+x" if side > 0 else "-x", DARK, eps=0.05))      # crack
        if not built:                                                                                      # patch over it (2003)
            out.append(box(0.15, 2.4, 1.8, (0.55, 0.55, 0.5), bottom=False).T(x + side * 0.12, y, z))
    return out


def extraction_fittings(ds):
    """The fittings fixed on the tank deck in 2004 to pump out the fuel (flanged valves)."""
    out = []
    for d in ds:
        y = deck_y(d)
        out += [at(cyl(1.6, 1.6, 0.25, (0.6, 0.58, 0.5), seg=16), d, x=6.5 if d % 2 else -6.5, y=y),
                at(cyl(0.7, 0.7, 1.8, (0.6, 0.58, 0.5), seg=12), d, x=6.5 if d % 2 else -6.5, y=y + 0.25),
                at(cyl(1.0, 1.0, 0.2, (0.85, 0.5, 0.1), seg=12), d, x=6.5 if d % 2 else -6.5, y=y + 2.05)]
    return out


def mud_around(length, half_beam, height, seed):
    def h(X, Z):
        side = height * np.exp(-((np.abs(X) - half_beam - 3.0) / 5.5) ** 2) * sstep(length / 2 + 8, length / 2 - 10, np.abs(Z))
        bumps = 0.2 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.2, seed).reshape(X.shape) * sstep(0.2, 1.0, side)
        return side + bumps - 0.6
    g = grid(-half_beam - 14, half_beam + 14, -length / 2 - 12, length / 2 + 12, 30, int(length / 2.5), h, (0.5, 0.45, 0.36))
    g.paint(lambda p: (0.5, 0.45, 0.36) + (fbm(p, 0.12, seed + 1)[:, None] - 0.5) * 0.06)
    g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > 0.15]
    return g


def debris_near(cx, cz, n, spread, seed):
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x, z = cx + r.normal(0, spread), cz + r.normal(0, spread)
        kind = r.random()
        if kind < 0.45:
            p = box(r.uniform(2, 8), 0.16, r.uniform(1.5, 5), BLACK if r.random() < 0.5 else (DECKC if r.random() < 0.5 else WHITE))
            crumple(p, 0.6, 0.3, seed + i)
            out.append(p.R(r.uniform(-0.3, 0.3), r.uniform(0, 6.28), r.uniform(-0.3, 0.3)).T(x, 0.05, z))
        elif kind < 0.65:
            out.append(tube([(0, 0, 0), (r.uniform(4, 12), 0, 0)], 0.36, PIPE, seg=8).R(0, r.uniform(0, 6.28), 0).T(x, 0.2, z))
        elif kind < 0.72:
            out.append(box(1.3, 0.15, r.uniform(4, 9), STEEL).R(r.uniform(-0.4, 0.4), r.uniform(0, 6.28), 0).T(x, 0.3, z))     # catwalk piece
        else:
            s = r.uniform(0.3, 1.4)
            out.append(box(s, s * r.uniform(0.4, 1.0), s * r.uniform(0.6, 1.6), (0.25, 0.2, 0.15)).R(0, r.uniform(0, 6.28), 0).T(x, -0.1, z))
    return merge(out)


def oil_patches(cx, cz, n, spread, seed):
    """Dark fuel residue on the sediment."""
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x, z, R = cx + r.normal(0, spread), cz + r.normal(0, spread), r.uniform(4, 14)

        def h(X, Z, x=x, z=z, R=R, s=seed + i):
            d = np.hypot(X - x, Z - z) / R
            edge = d + 0.35 * (fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.15, s).reshape(X.shape) - 0.5)
            return np.where(edge < 1, 0.08, -0.5)
        g = grid(x - R * 1.4, x + R * 1.4, z - R * 1.4, z + R * 1.4, 14, 14, h, OIL)
        g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > 0.0]
        out.append(g)
    return merge(out)


# ── Bow section: from the bow to the break (about 112 m), at 3,830 m ────────
BOW_END = 112.0
bow = merge(ship_part(0, BOW_END) + name_plates(4, 1, D - 3.0) + name_plates(4, -1, D - 3.0)
            + cracks_and_patches((40, 72, 95), 1, (9, 12, 7), False) + cracks_and_patches((58,), -1, (11,), False)
            + extraction_fittings((33, 48, 63, 78, 93)))
# the inside of the tanks at the break, black with fuel
inner = box(B * 0.9, D * 0.85, 0.2, OIL, bottom=True).T(0, 1.0, zd(BOW_END - 3))
crumple(inner, 1.2, 0.2, 7)
bow = merge([bow, inner])
dfb = L / 2 - bow.v[:, 2]
w = sstep(85, BOW_END, dfb)
bow.v[:, 1] -= w * np.sin(dfb * 0.45) * 1.4 * (bow.v[:, 1] / D)            # the deck buckled towards the break
crumple(bow, 2.4, 0.13, 11, w)
crumple(bow, 3.2, 0.35, 12, sstep(105, BOW_END, dfb))                      # torn edges
crumple(bow, 0.2, 0.08, 13)
zc = (zd(0) + zd(BOW_END)) / 2
bow.T(0, 0, -zc).R(0.012, 0, 0.07)                                         # a little bow-down and a list to starboard
bow.T(0, -7.0, 0)                                                           # settled into the sediment
weather(bow, rust=0.45, sediment=0.5, streaks=0.6, seed=21)
oil_stain(bow, [(0, 6, -BOW_END / 2 + 2), (14, 9, 0), (-14, 11, 10)], 14, 31)
bow_site = merge([mud_around(BOW_END, B / 2, 1.6, 5), debris_near(0, -BOW_END / 2 - 25, 70, 22, 101), oil_patches(0, -BOW_END / 2 - 18, 6, 25, 201)])

# ── Stern section: from the break to the stern (about 127 m), at 3,565 m ────
STERN_START = 116.5
stern = merge(ship_part(STERN_START, L) + [rudder_prop()] + lettering("PRESTIGE", 0, 0, D - 0.8, px=0.32, transom=True)
              + lettering("NASSAU", 0, 0, D - 3.6, px=0.26, transom=True) + cracks_and_patches((135, 160), 1, (8, 11), False)
              + cracks_and_patches((150,), -1, (10,), False) + extraction_fittings((128, 143, 158, 173, 188)))
inner2 = box(B * 0.9, D * 0.85, 0.2, OIL).T(0, 1.0, zd(STERN_START + 3))
crumple(inner2, 1.2, 0.2, 8)
stern = merge([stern, inner2])
dfs = L / 2 - stern.v[:, 2]
w2 = sstep(145, STERN_START, dfs)
stern.v[:, 1] -= w2 * np.sin(dfs * 0.5) * 1.6 * (stern.v[:, 1] / D)
crumple(stern, 2.6, 0.13, 31, w2)
crumple(stern, 3.2, 0.35, 32, sstep(124, STERN_START, dfs))
crumple(stern, 0.25, 0.08, 33)
crumple(stern, 0.6, 0.25, 34, sstep(D + 2, D + 14, stern.v[:, 1]))     # dents in the accommodation and funnel
zc2 = (zd(STERN_START) + zd(L)) / 2
stern.T(0, 0, -zc2).R(0, 0, -0.1).T(0, -7.5, 0)                          # list to port, settled deeper (the propeller half buried)
weather(stern, rust=0.45, sediment=0.5, streaks=0.6, seed=41)
oil_stain(stern, [(0, 6, (L - STERN_START) / 2 - 2), (-14, 9, 20)], 14, 41)
stern_site = merge([stern, mud_around(L - STERN_START, B / 2, 1.2, 6),
                    debris_near(0, (L - STERN_START) / 2 + 22, 60, 20, 301), oil_patches(0, (L - STERN_START) / 2 + 16, 5, 22, 401)])

# ── The ship as built ────────────────────────────────────────────────────────
ship = merge(ship_part(0, L, built=True) + [rudder_prop()] + name_plates(4, 1, D - 3.0) + name_plates(4, -1, D - 3.0)
             + lettering("PRESTIGE", 0, 0, D - 0.8, px=0.32, transom=True) + lettering("NASSAU", 0, 0, D - 3.6, px=0.26, transom=True))
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
# Stern site: 42°12'N 12°02'W vs bow 42°10'N 12°03'W (published positions, rounded): ~3.7 km north, ~1.4 km east.
LAT_B, LON_B, LAT_S, LON_S = 42.1667, -12.05, 42.2, -12.0333
north = (LAT_S - LAT_B) * 111320.0
east = (LON_S - LON_B) * 111320.0 * math.cos(math.radians((LAT_B + LAT_S) / 2))
STERN_HEADING = 40                                                       # approximate
out = ROOT / "models" / "item" / "Q1977186" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
tris, size = write_glb(out, [("bow_section", bow), ("bow_site", bow_site),
                             ("stern_section", stern_site, {"translation": [-east, 0, north], "rotation": yaw_quat(STERN_HEADING)})])
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB  (stern {north:.0f} m N, {east:.0f} m E)")
for n, m in (("bow", bow), ("bow_site", bow_site), ("stern_site", stern_site)):
    print(f"  {n}: {m.tris:,} tris, x {m.v[:, 0].min():.0f}..{m.v[:, 0].max():.0f}, y {m.v[:, 1].min():.0f}..{m.v[:, 1].max():.0f}, z {m.v[:, 2].min():.0f}..{m.v[:, 2].max():.0f}")
tris2, size2 = write_glb(out.with_name("model_built.glb"), [("main", ship)])
print(f"model_built.glb: {tris2:,} triangles, {size2 / 2**20:.2f} MB, y {ship.v[:, 1].min():.1f}..{ship.v[:, 1].max():.1f}")
