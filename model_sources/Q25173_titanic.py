"""
RMS Titanic (Q25173) — the wreck as it lies today, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

Real size from the ship's published dimensions (length 269.1 m, beam 28.2 m, moulded depth 19.6 m to the
shelter deck, 22.3 m to the bridge deck, draught 10.5 m). The wreck, after the published surveys:
- the bow section (~143 m, up to between the 3rd and 4th funnels) stands upright, its bow ploughed into the
  mud; the funnels are gone, the foremast has fallen back over the bridge, the decks sag towards the break;
- the stern section (~109 m) lies about 600 m to the south, facing the opposite way, collapsed on itself;
- between them, a debris field with boilers, hull plates and smaller pieces, heavier near the stern.
Headings are approximate (bow roughly north).

    python model_sources/Q25173_titanic.py        -> models/item/Q25173/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, weather, sstep, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1912)
RUN_STYLE = "quad"       # rust runs under the portholes: "quad" (this model) or "drip" (the diorama sets it)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B = 269.1, 28.2
H = 19.6                 # keel to shelter (C) deck
HB = 22.3                # keel to bridge (B) deck
HA = 25.4                # top of the A-deck house = boat deck
HT = 28.3                # top of the boat-deck houses
WL = 10.5                # waterline (draught)
BLACK, RED, WHITE = (0.07, 0.07, 0.08), (0.42, 0.11, 0.07), (0.6, 0.55, 0.47)      # white paint, stained after a century
DECK, DARK, BUFF, RUST = (0.33, 0.28, 0.23), (0.025, 0.025, 0.03), (0.66, 0.5, 0.28), (0.4, 0.19, 0.08)
GOLD, RUSTY, BRONZE = (0.72, 0.56, 0.22), (0.5, 0.24, 0.08), (0.55, 0.4, 0.2)


def zd(d):            # distance from the bow (m) -> ship coordinate z (bow at +L/2)
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):          # half-beam factor along the ship
    w = 1.0
    if t > 0.78:
        w = math.sin(((1 - t) / 0.22) * math.pi / 2) ** 1.25      # fine entrance, sharp stem
    if t < 0.12:
        w = min(w, 0.32 + 0.68 * math.sin((t / 0.12) * math.pi / 2))
    return max(w, 0.004)


def yb_of(t):         # keel line (rises at the counter stern and at the forefoot)
    y = 0.0
    if t < 0.07:
        y = H * 0.55 * ((0.07 - t) / 0.07) ** 1.6
    if t > 0.988:
        y = max(y, H * 0.85 * ((t - 0.988) / 0.012) ** 2)
    return y


def yd_of(t):         # shelter deck with a little sheer
    return H * (1 + 0.03 * (2 * t - 1) ** 2)


def k_of(t):          # section fullness (boxy midships, fine at the ends)
    return 2.2 + 6.0 * w_of(t) ** 3


def hw(z, y):         # half-width of the hull at z and height y
    t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    if y >= yd:
        return B / 2 * w
    u = max(0.0, (y - yb) / max(0.1, yd - yb))
    return B / 2 * w * (1 - (1 - u) ** k)


def hull(d0, d1, NP=28):
    """Hull from d0 to d1 (distance from bow): both sides, the deck and, at the stern end, the counter."""
    ns = max(2, int(math.ceil((d1 - d0) / 1.2)))
    st = []
    for i in range(ns + 1):
        z = zd(d1) + (zd(d0) - zd(d1)) * i / ns              # from aft (smaller z) to forward
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
    if d1 >= L - 0.01:                                       # counter stern closed
        c0 = len(v); z, w, yb, yd, k = st[0]
        v.append((0, (yb + yd) / 2, z))
        f += [(c0, j + 1, j) for j in range(W - 1)]
    hl = Mesh(v, f)
    hl.paint(lambda p: np.where((p[:, 1] < WL)[:, None], RED, BLACK))
    return hl


def band(d0, d1, y_extra, color=BLACK, top=DECK):
    """A raised hull band (forecastle, bridge deck, poop): vertical sides on the hull, deck on top, closed ends."""
    ns = max(2, int(math.ceil((d1 - d0) / 1.5)))
    parts = []
    zs = [zd(d1) + (zd(d0) - zd(d1)) * i / ns for i in range(ns + 1)]
    ys = [yd_of(t_of(z)) for z in zs]
    xs = [B / 2 * w_of(t_of(z)) for z in zs]
    for side in (-1, 1):
        v, f = [], []
        for z, y0, x in zip(zs, ys, xs):
            v += [(side * x, y0 - 0.05, z), (side * x, y0 + y_extra, z)]
        for i in range(ns):
            a = 2 * i
            f += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)] if side < 0 else [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
        parts.append(Mesh(v, f, color))
    v, f = [], []
    for z, y0, x in zip(zs, ys, xs):
        v += [(-x, y0 + y_extra, z), (x, y0 + y_extra, z)]
    for i in range(ns):
        a = 2 * i; b = a + 2; c = a + 1; d = b + 1
        f += [(a, b, c), (c, b, d)]
    parts.append(Mesh(v, f, top))
    for z, y0, x, s in ((zs[0], ys[0], xs[0], -1), (zs[-1], ys[-1], xs[-1], 1)):
        if x > 0.5:
            parts.append(Mesh([(-x, y0, z), (x, y0, z), (x, y0 + y_extra, z), (-x, y0 + y_extra, z)],
                              [(0, 1, 2), (0, 2, 3)] if s > 0 else [(0, 2, 1), (0, 3, 2)], color))
    return merge(parts)


def at(m, d, x=0.0, y=0.0):
    return m.T(x, y, zd(d))


def in_range(d, d0, d1):
    return d0 <= d <= d1


def ship_part(d0, d1, built=False):
    """Everything of the ship between d0 and d1 (distance from the bow). built=True: as she sailed (funnels, masts,
    boats, glass dome…); built=False: what is left on the wreck (stubs, holes, rusticles)."""
    P = [hull(d0, d1)]
    inr = lambda d, m=0: d0 + m <= d <= d1 - m
    deck_y = lambda d: yd_of(t_of(zd(d)))                         # shelter deck (well decks)
    top_y = lambda d: deck_y(d) + (HB - H)                        # top of the forecastle / bridge deck / poop
    for a, b in ((0, 39), (58, 207), (227, L)):                   # forecastle, bridge deck, poop
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 > 1:
            P.append(band(a2, b2, HB - H))
            # the gilded line along the top of the black hull
            for s in (-1, 1):
                pts = [(s * (B / 2 * w_of(t_of(zd(d)))) + s * 0.04, top_y(d) - 0.5, zd(d)) for d in np.arange(a2 + 0.3, b2, 2.0)]
                if len(pts) > 1:
                    P.append(tube(pts, 0.09, GOLD, seg=4))
    # A deck: closed forward (windows), open promenade aft (columns), deck slab on top = the boat deck
    a2, b2 = max(60, d0), min(205, d1)
    if b2 - a2 > 1:
        P.append(at(box(B * 0.94, 0.45, b2 - a2, WHITE), (a2 + b2) / 2, y=HA - 0.45))
        P.append(at(box(B * 0.94, 0.03, b2 - a2, DECK), (a2 + b2) / 2, y=HA))
        c1 = min(b2, max(a2, 110))
        if c1 - a2 > 1:                                            # closed part
            P.append(at(box(B * 0.94, HA - HB - 0.45, c1 - a2, WHITE), (a2 + c1) / 2, y=HB))
            for d in np.arange(a2 + 1.5, c1 - 1, 2.6):
                for s in (-1, 1):
                    P.append(quad(s * B * 0.47, HB + 1.6, zd(d), 1.2, 1.1, "+x" if s > 0 else "-x", DARK))
        if b2 - c1 > 1:                                            # open promenade: inner wall and columns
            P.append(at(box(B * 0.78, HA - HB - 0.45, b2 - c1, WHITE), (c1 + b2) / 2, y=HB))
            for d in np.arange(c1 + 1, b2, 3.0):
                for s in (-1, 1):
                    P.append(at(box(0.32, HA - HB - 0.45, 0.32, WHITE), d, x=s * B * 0.455, y=HB))
                    P.append(quad(s * B * 0.39, HB + 1.5, zd(d + 1.5), 1.2, 1.1, "+x" if s > 0 else "-x", DARK))
    # Boat-deck houses (officers' quarters, entrances, lounges) between the funnels, with their windows
    for a, b, wf in ((60, 80, 0.62), (83, 99, 0.5), (107, 125, 0.5), (133, 151, 0.5), (159, 178, 0.5), (182, 200, 0.55)):
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 > 1:
            P.append(at(box(B * wf, HT - HA, b2 - a2, WHITE), (a2 + b2) / 2, y=HA))
            for d in np.arange(a2 + 1.2, b2 - 0.8, 2.2):
                for s in (-1, 1):
                    P.append(quad(s * B * wf / 2, HA + 1.6, zd(d), 0.8, 0.9, "+x" if s > 0 else "-x", DARK))
    # Boat-deck railing
    for a, b in ((60, 205),):
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 > 2:
            for s in (-1, 1):
                pts = [(s * B * 0.465, HA + 1.05, zd(d)) for d in np.arange(a2 + 0.5, b2, 2.5)]
                P.append(tube(pts, 0.05, WHITE, seg=4))
                for x, y, z in pts[::2]:
                    P.append(cyl(0.04, 0.04, 1.05, WHITE, seg=4, caps=False).T(x, HA, z))
    # Funnels: 7.3 × 5.8 m, buff with black tops, raked aft, 53.3 m above the keel. On the wreck: torn stubs.
    rake = 0.05
    for d in (77, 103, 129, 155):
        if not inr(d, 4):
            continue
        if built:
            hgt = 53.3 - HA
            f = cyl(3.65, 3.55, hgt - 4.4, BUFF, seg=24, sx=0.79).R(-rake, 0, 0)
            top = cyl(3.55, 3.5, 4.4, BLACK, seg=24, sx=0.79).T(0, hgt - 4.4, 0).R(-rake, 0, 0)
            P += [at(f, d, y=HA), at(top, d, y=HA)]
        else:
            stub = cyl(3.65, 3.6, 2.2, (0.2, 0.15, 0.1), seg=24, sx=0.79, caps=False)
            crumple(stub, 0.5, 0.6, int(d), sstep(0.8, 2.2, stub.v[:, 1]))
            P += [at(stub, d, y=HT - 0.3), at(cyl(3.2, 3.2, 0.05, DARK, seg=24, sx=0.79), d, y=HT + 0.4)]
    # Grand staircase: the glass dome (as built) / a dark hole (wreck)
    if inr(91):
        if built:
            P.append(at(lathe([(4.2, 0), (4.0, 0.6), (3.2, 1.6), (1.8, 2.3), (0.01, 2.6)], (0.62, 0.72, 0.74), seg=20), 91, y=HT))
        else:
            P.append(quad(0, HT, zd(91), 6.5, 8.5, "+y", DARK, eps=0.03))
    # Navigating bridge: wheelhouse and wings (on the wreck only the wings and the telemotor remain)
    if inr(58):
        P.append(at(box(B * 1.0, 0.5, 2.6, WHITE), 57.2, y=HA))
        if built:
            P.append(at(box(9.5, 2.7, 4.0, WHITE), 58.3, y=HA))
            for x in np.arange(-4.0, 4.1, 1.0):
                P.append(quad(x, HA + 1.7, zd(56.3), 0.7, 0.9, "+z", DARK))
        else:
            P.append(at(cyl(0.3, 0.25, 1.3, (0.55, 0.4, 0.2), seg=10), 57.5, y=HA + 0.5))      # the telemotor (bronze)
    # Lifeboat davits and, as built, the boats
    davits = list(np.arange(60, 82, 5.2)) + list(np.arange(176, 202, 6.2))
    for d in davits:
        if inr(d):
            for s in (-1, 1):
                x = s * B * 0.46
                P.append(tube([(x, HA, zd(d)), (x, HA + 3.2, zd(d)), (x + s * 0.9, HA + 4.0, zd(d)), (x + s * 1.7, HA + 3.6, zd(d))], 0.17, WHITE, seg=6))
                if built and d + 3 in [dd + 3 for dd in davits[::2]]:
                    boat = lathe([(0.05, -4.6), (0.8, -3.9), (1.35, -2.4), (1.45, 0), (1.35, 2.4), (0.8, 3.9), (0.05, 4.6)], WHITE, seg=10, axis="z")
                    boat.v[:, 1] = np.minimum(boat.v[:, 1], 0.3)                               # open on top
                    boat.S(1, 0.85, 1)
                    P.append(boat.T(x - s * 0.6, HA + 1.4, zd(d + 2.6)))
    # Ventilator cowls on the boat deck
    for d, x in ((86, 9), (96, 9), (111, 9.5), (122, 9.5), (138, 9), (148, 9), (186, 8.5), (196, 8.5)):
        if inr(d):
            for s in (-1, 1):
                P.append(at(cyl(0.55, 0.55, 2.2, WHITE, seg=10), d, x=s * x, y=HA))
                P.append(at(box(1.3, 1.1, 1.2, WHITE), d + 0.4, x=s * x, y=HA + 2.0))
    # Portholes (on the wreck, each with a rust run below it)
    for y, a, b, sp in ((12.4, 30, 240, 2.4), (15.0, 18, 250, 1.9), (17.6, 12, 256, 1.7), (HB - 1.4, 60, 205, 1.7), (HB - 1.4, 3, 37, 2.2), (HB - 1.4, 229, 262, 2.0)):
        for d in np.arange(max(a, d0 + 1), min(b, d1 - 1), sp):
            z = zd(d)
            x = hw(z, y)
            if x < 2:
                continue
            for s in (-1, 1):
                P.append(quad(s * x, y, z, 0.42, 0.42, "+x" if s > 0 else "-x", DARK, eps=0.06))
                if not built and rng.random() < 0.6:
                    ln = rng.uniform(0.8, 2.6)
                    run = quad(s * hw(z, y - ln / 2), y - 0.2 - ln / 2, z, 0.3, ln, "+x" if s > 0 else "-x", RUSTY, eps=0.05)
                    if RUN_STYLE == "drip":                # the diorama: a drip that narrows and fades downwards
                        low = run.v[:, 1] < run.v[:, 1].mean()
                        run.v[low, 2] = z + (run.v[low, 2] - z) * 0.12
                        run.c = np.where(low[:, None], (0.62, 0.32, 0.12), (0.36, 0.15, 0.06))
                    P.append(run)
    # Hatches (open on the wreck)
    for d, wd in ((26, 6), (49, 7), (217, 7), (240, 6)):
        if inr(d, 3):
            y = top_y(d) if (d < 39 or d > 227) else deck_y(d)
            P.append(at(box(wd, 0.7, 5.5, BLACK), d, y=y))
            P.append(at(box(wd - 0.6, 0.02, 4.9, (0.38, 0.34, 0.27) if built else DARK), d, y=y + 0.7))
    # Forecastle: railing at the bow, anchor crane, capstans and windlass, chains, the anchors on both sides
    if d0 <= 1:
        pts_r = [(hw(zd(d), HB + 2) - 0.25, top_y(d) + 1.15, zd(d)) for d in np.arange(0.6, 39, 1.5)]
        for s in (-1, 1):
            P.append(tube([(s * x, y, z) for x, y, z in pts_r], 0.07, WHITE, seg=5))
            for x, y, z in pts_r:
                P.append(cyl(0.05, 0.05, 1.15, WHITE, seg=4, caps=False).T(s * x, y - 1.15, z))
        P.append(at(cyl(0.45, 0.3, 7.5, BLACK, seg=8), 4.5, y=top_y(4.5)))                       # anchor crane
        P.append(tube([(0, top_y(4.5) + 7.3, zd(4.5)), (0, top_y(4.5) + 7.8, zd(9.5))], 0.25, BLACK, seg=6))
        for d, x in ((9, 2.5), (9, -2.5), (13, 0), (16, 3.8), (16, -3.8)):
            P.append(at(cyl(0.7, 0.6, 1.0, BLACK, seg=12), d, x=x, y=top_y(d)))
        for s in (-1, 1):                                                                        # chains to the hawse pipes
            z0, z1 = zd(9), zd(4.2)
            P.append(tube([(s * 2.5, top_y(9) + 0.5, z0), (s * 3.4, top_y(6.5) + 0.25, zd(6.5)), (s * (hw(z1, HB) - 0.7), top_y(4.2) + 0.15, z1)], 0.22, (0.16, 0.12, 0.09), seg=5))
            z = zd(8.5); x = s * (hw(z, 17.5) + 0.35)
            P.append(box(0.55, 3.6, 0.8, BLACK).T(x, 15.8, z))
            P.append(box(0.6, 0.7, 3.0, BLACK).T(x, 15.6, z))
    # Cargo cranes in the well decks and on the poop; the docking bridge at the stern
    for d, x in ((42, 4.5), (42, -4.5), (55, 4.5), (55, -4.5), (210, 4), (210, -4), (224, 4), (224, -4), (246, 4.5), (246, -4.5)):
        if inr(d, 1):
            y = top_y(d) if d > 227 else deck_y(d)
            P.append(at(cyl(0.9, 0.8, 2.2, WHITE, seg=10), d, x=x, y=y))
            P.append(tube([(x, y + 2.2, zd(d)), (x * 1.3, y + 8.5, zd(d) + (6 if d < 100 else -6))], 0.18, WHITE, seg=5))
    if inr(258, 2):
        P.append(at(box(B * 0.6, 0.4, 2.2, WHITE), 258, y=top_y(258) + 3.2))
        for s in (-1, 1):
            P.append(at(cyl(0.12, 0.12, 3.2, WHITE, seg=5), 258, x=s * 3, y=top_y(258)))
    # Masts and the wireless aerial between them (as built)
    if built:
        for d, h in ((45, 62.0), (222, 58.0)):
            if inr(d):
                P.append(at(cyl(0.45, 0.22, h - deck_y(d), (0.55, 0.42, 0.25), seg=10).R(-0.05, 0, 0), d, y=deck_y(d)))
        if inr(45):
            P.append(at(cyl(1.1, 1.1, 1.7, (0.55, 0.42, 0.25), seg=12), 45.7, y=48.0))            # crow's nest
            P.append(tube([(0, 61.5, zd(46.5)), (0, top_y(1) + 1.2, zd(1.2))], 0.04, (0.2, 0.2, 0.2), seg=4))   # forestay
        if d0 <= 45 and d1 >= 222:
            P.append(tube([(0, 61.0, zd(46.5)), (0, 57.0, zd(223.5))], 0.03, (0.2, 0.2, 0.2), seg=4))           # aerial
    return P


def rusticles(m, n, seed, ymin=None):
    """Rust icicles hanging under edges: under the vertices of mesh m that face down or sideways."""
    r = np.random.default_rng(seed)
    idx = r.choice(len(m.v), size=min(n, len(m.v)), replace=False)
    out = []
    for i in idx:
        x, y, z = m.v[i]
        if ymin is not None and y < ymin:
            continue
        h = r.uniform(0.4, 2.2)
        out.append(cyl(0.0, r.uniform(0.08, 0.22), h, (0.55, 0.26, 0.09), seg=5, caps=False).T(x, y - h, z))
    return merge(out)


def rudder_props():
    zs = zd(L) + 1.5
    parts = [box(0.8, 22, 4.2, BLACK).T(0, 0.5, zs - 1.0)]
    for x, y, z, rr, nb in ((0, 5.0, zs + 4.5, 2.6, 4), (7.5, 5.5, zs + 12, 3.5, 3), (-7.5, 5.5, zs + 12, 3.5, 3)):
        parts.append(cyl(0.6, 0.6, 1.2, BRONZE, seg=12).R(math.pi / 2, 0, 0).T(x, y, z - 0.6))
        for k in range(nb):
            a = 2 * math.pi * k / nb
            parts.append(box(rr * 2 * 0.42, 0.18, 1.3, BRONZE, bottom=False).T(rr * 0.42 + 0.4, 0, 0).R(0.35, 0, 0).R(0, 0, a).T(x, y, z))
    return merge(parts)


def build(bow_extra=None):
    """The wreck: bow, mud, stern and debris (and the ship as built). bow_extra: meshes in ship coordinates
    (bow at +L/2) that get the same sagging, crumpling and tilting as the bow; they come back apart, in
    "bow_extra", so the diorama can use the positions they end up at."""
    global rng
    rng = np.random.default_rng(1912)
    # ── Bow section (the wreck) ──────────────────────────────────────────────────
    BOW_END = 143.0
    bow = merge(ship_part(0, BOW_END))
    y0 = yd_of(t_of(zd(45)))
    fore = merge([tube([(0, y0, zd(45)), (0.4, HT + 0.6, zd(86))], 0.42, (0.12, 0.1, 0.09), seg=8),        # foremast, fallen back over the bridge
                  cyl(0.9, 0.9, 1.6, (0.12, 0.1, 0.09), seg=10).R(0, 0, math.pi / 2 - 0.22).T(0.6, HT - 1.2, zd(76))])   # its crow's nest
    # torn plates at the break, sticking out backwards
    torn = []
    for i in range(30):
        y = rng.uniform(4, HT)
        wdt = hw(zd(BOW_END), min(y, H)) if y < H else B * 0.45
        p = box(rng.uniform(2, 6), 0.12, rng.uniform(2, 7), BLACK if y < HB else WHITE, bottom=False)
        crumple(p, 0.8, 0.35, 300 + i)
        torn.append(p.R(rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4), rng.uniform(-0.8, 0.8)).T(rng.uniform(-wdt, wdt), y, zd(BOW_END) - rng.uniform(0.5, 4)))
    # where the wooden deck has gone: holes with the steel beams across
    holes = []
    for d, x, w_, l_ in ((66, 5, 4, 6), (70, -6, 3, 5), (88, -9, 3, 7), (112, 8, 4, 8), (128, -7, 5, 9)):
        holes.append(quad(x, HA + 0.035, zd(d), w_, l_, "+y", DARK, eps=0.0))
        for k in np.arange(-l_ / 2 + 0.6, l_ / 2, 1.2):
            holes.append(at(box(w_, 0.25, 0.22, (0.22, 0.15, 0.09)), d + k, x=x, y=HA))
    bow = merge([bow, fore, merge(torn), merge(holes)])
    # parts that must follow every deformation of the bow (the diorama adds its own: a bathtub, probes…)
    ev0, ef0 = len(bow.v), len(bow.f)
    if bow_extra:
        bow = merge([bow] + list(bow_extra))
    ev1, ef1 = len(bow.v), len(bow.f)
    dfb = L / 2 - bow.v[:, 2]                                       # distance from the bow of every vertex
    w = sstep(110, BOW_END, dfb)
    bow.v[:, 1] -= w * (bow.v[:, 1] / HB) * 7.0                    # the decks sag towards the break
    sag = np.exp(-((dfb - 69) / 7) ** 2) * sstep(HA - 0.5, HT, bow.v[:, 1])
    bow.v[:, 1] -= sag * 1.6                                        # the roof of the officers' quarters has partly fallen in
    crumple(bow, 2.2, 0.12, 11, w)
    crumple(bow, 3.0, 0.35, 12, sstep(136, BOW_END, dfb))          # torn edges at the break
    crumple(bow, 0.25, 0.08, 13)                                    # dents everywhere
    # rusticles under the bow railing, the forecastle edge and along the hull top
    rail_v = Mesh(np.array([(s * (hw(zd(d), HB + 2) - 0.25), yd_of(t_of(zd(d))) + (HB - H), zd(d)) for d in np.arange(0.6, 120, 0.9) for s in (-1, 1)]))
    bow = merge([bow, rusticles(rail_v, 220, 5)])
    zc = (zd(0) + zd(BOW_END)) / 2
    bow.T(0, 0, -zc)
    bow.R(math.atan(4 / BOW_END), 0, 0)                              # bow down: ploughed into the mud
    bow.T(0, -13.0, 0)
    weather(bow, rust=0.75, sediment=0.6, streaks=0.95, seed=21)

    # the mud ridge the bow pushed up (it blends into the seabed at its edges)
    ztip = BOW_END / 2


    def mud_h(X, Z):
        d = ztip - Z
        hwd = np.array([B / 2 * w_of(t_of(zd(max(0.0, min(L, dd))))) for dd in d.ravel()]).reshape(d.shape)
        side = 3.4 * np.exp(-((np.abs(X) - hwd - 3.5) / 4.5) ** 2) * sstep(95, 25, d) * (d > -2)
        front = 1.8 * np.exp(-(X / 7) ** 2) * np.exp(-((d + 4) / 5) ** 2)
        edge = sstep(42, 30, np.abs(X)) * sstep(-25, -15, -d + 0) * sstep(110, 95, d)
        ridge = np.maximum(side, front)
        bumps = 0.22 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.2, 5).reshape(X.shape) * sstep(0.2, 1.2, ridge)
        return (ridge + bumps) * edge - 0.6               # flat parts stay under the seabed


    mud = grid(-42, 42, ztip - 110, ztip + 26, 42, 68, mud_h, (0.52, 0.49, 0.43))
    mud.paint(lambda p: (0.5, 0.45, 0.36) + (fbm(p, 0.12, 9)[:, None] - 0.5) * 0.06)
    mud.f = mud.f[mud.v[mud.f][:, :, 1].max(axis=1) > 0.15]   # only the ridges: no flat sheet that could show above another seabed

    # ── Stern section (the wreck) ────────────────────────────────────────────────
    STERN_START = 160.0
    stern = merge(ship_part(STERN_START, L) + [rudder_props(), tube([(0, HB, zd(218)), (2.5, HB + 4, zd(250))], 0.4, (0.12, 0.1, 0.09), seg=8)])
    dfs = L / 2 - stern.v[:, 2]
    # collapsed: the decks pancaked down, the sides peeled outwards, the forward end torn apart; the poop held better
    keep = 0.3 + 0.5 * sstep(232, 258, dfs)
    y = stern.v[:, 1]
    stern.v[:, 1] = np.where(y > 6, 6 + (y - 6) * keep, y)
    stern.v[:, 0] *= 1 + 0.45 * sstep(6, 26, y) * (1 - 0.7 * sstep(235, 260, dfs))
    crumple(stern, 4.5, 0.12, 31, sstep(195, STERN_START, dfs))
    crumple(stern, 1.4, 0.1, 32)
    crumple(stern, 0.5, 0.4, 33)
    # hull plates torn off and standing up around the heap
    ups = []
    for i in range(36):
        d = rng.uniform(STERN_START, 235)
        s = rng.choice([-1, 1])
        p = box(rng.uniform(2, 7), rng.uniform(3, 9), 0.12, BLACK if rng.random() < 0.7 else WHITE)
        crumple(p, 0.9, 0.3, 500 + i)
        ups.append(p.R(rng.uniform(-0.9, 0.9), rng.uniform(0, 6.28), rng.uniform(-0.5, 0.5)).T(s * rng.uniform(B * 0.3, B * 0.85), rng.uniform(2, 8), zd(d)))
    stern = merge([stern, merge(ups)])
    zc2 = (zd(STERN_START) + zd(L)) / 2
    stern.T(0, 0, -zc2).R(0, math.pi, 0.1).T(14, -12.5, -600)       # facing the opposite way, ~600 m south
    weather(stern, rust=0.9, sediment=0.75, streaks=0.8, seed=41)

    # ── Debris field ─────────────────────────────────────────────────────────────
    deb = []


    def spot(near_stern=0.65):
        if rng.random() < near_stern:
            return rng.normal(18, 45), rng.normal(-600, 55)
        return rng.normal(6, 30), rng.uniform(-560, -90)


    boilers = []
    for i in range(5):                                               # Scotch boilers, 4.8 m wide, with their three furnaces
        x, z = rng.normal(20, 18), rng.normal(-560, 25)
        boilers.append((x, z))
        b_ = merge([cyl(2.4, 2.4, 6.1, (0.16, 0.12, 0.09), seg=20)] +
                   [cyl(0.55, 0.55, 0.06, DARK, seg=12).T(dx, 6.1, dz) for dx, dz in ((-1.1, -0.5), (1.1, -0.5), (0, 0.9))])
        deb.append(b_.T(0, -3.05, 0).R(0, 0, math.pi / 2).R(0, rng.uniform(0, math.pi), 0).T(x, 1.7, z))
    for i in range(70):                                              # coal, spilled near the boilers
        x, z = rng.normal(20, 22), rng.normal(-560, 30)
        s = rng.uniform(0.15, 0.5)
        deb.append(box(s, s * 0.7, s, (0.05, 0.05, 0.05)).R(rng.uniform(0, 3), rng.uniform(0, 3), 0).T(x, -0.05, z))
    for i in range(3):                                               # pieces of funnel (curved plates, buff under the rust)
        x, z = spot()
        f = cyl(3.6, 3.6, rng.uniform(3, 6), BUFF, seg=24, sx=0.79, caps=False)
        keep_ = np.arctan2(f.v[:, 2], f.v[:, 0]) > rng.uniform(0.5, 2.0)
        f.f = f.f[keep_[f.f].all(axis=1)]
        crumple(f, 0.6, 0.25, 700 + i)
        deb.append(f.R(math.pi / 2, rng.uniform(0, 6.28), 0).T(x, 0.6, z))
    for i in range(2):                                               # cargo cranes, fallen
        x, z = spot()
        deb.append(merge([cyl(0.9, 0.8, 2.2, WHITE, seg=10), tube([(0, 2.2, 0), (2, 8.5, 6)], 0.18, WHITE, seg=5)]).R(math.pi / 2, rng.uniform(0, 6.28), 0).T(x, 0.9, z))
    for i in range(80):                                              # hull plates and deck pieces, bent
        x, z = spot()
        m = box(rng.uniform(2, 9), 0.18, rng.uniform(1.5, 4.5), BLACK if rng.random() < 0.6 else WHITE)
        crumple(m, 0.6, 0.3, 100 + i)
        deb.append(m.R(rng.uniform(-0.35, 0.35), rng.uniform(0, 6.28), rng.uniform(-0.35, 0.35)).T(x, 0.1, z))
    for i in range(28):                                              # beams
        x, z = spot()
        deb.append(box(0.4, 0.6, rng.uniform(4, 12), (0.2, 0.15, 0.1)).R(0, rng.uniform(0, 6.28), rng.uniform(-0.2, 0.2)).T(x, 0, z))
    for i in range(180):                                             # smaller things
        x, z = spot(0.55)
        s = rng.uniform(0.3, 1.4)
        deb.append(box(s, s * rng.uniform(0.4, 1.0), s * rng.uniform(0.6, 1.6), (0.25, 0.2, 0.15)).R(0, rng.uniform(0, 6.28), 0).T(x, -0.1, z))
    big = box(22, 2.5, 12, (0.1, 0.08, 0.07))                       # a piece of the double bottom
    crumple(big, 1.5, 0.15, 77)
    deb.append(big.R(0.08, 0.6, 0.05).T(-6, -0.4, -350))
    debris = merge(deb)
    weather(debris, rust=0.85, sediment=0.75, streaks=0.5, seed=61)

    # ── The ship as she sailed (model_built.glb): whole, clean, keel on y = 0, centred ──
    ship = merge(ship_part(0, L, built=True) + [rudder_props()])
    ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)          # a little life in the paint

    extra = None
    if ev1 > ev0:
        sel = np.zeros(len(bow.v), bool); sel[ev0:ev1] = True
        fe = bow.f[ef0:ef1] - ev0
        extra = Mesh(bow.v[ev0:ev1].copy(), fe, bow.c[ev0:ev1].copy())
        keep = np.ones(len(bow.f), bool); keep[ef0:ef1] = False
        remap = np.cumsum(~sel) - 1
        bow = Mesh(bow.v[~sel], remap[bow.f[keep]], bow.c[~sel])
    return {"bow": bow, "mud": mud, "stern": stern, "debris": debris, "ship": ship, "bow_extra": extra, "boilers": boilers}


# ── Dressing: what makes the wreck worth a close look in Wiki Submarine ──────
def dress(seed=41912):
    """The wreck (model.glb) and the ship as built (model_built.glb), with rust, life, objects, textures and
    baked ambient occlusion. Returns (wreck parts, instanced parts, materials, built parts, built materials)."""
    import abyss
    from scenekit import (material, steel_textures, rust_textures, planks_textures, bake_ao, instances)
    global RUN_STYLE
    RUN_STYLE = "drip"
    r = np.random.default_rng(seed)
    abyss.seed(seed)

    # Captain Smith's bathtub: the wall of his quarters has fallen and the tub is in plain view (starboard side)
    tub = merge([box(0.78, 0.62, 1.72, (0.86, 0.84, 0.78)), box(0.62, 0.02, 1.52, (0.32, 0.2, 0.1)).T(0, 0.55, 0),
                 tube([(0.2, 0.62, -0.86), (0.2, 0.95, -0.95), (0.2, 1.05, -0.8)], 0.025, (0.5, 0.35, 0.18), seg=5)])
    crumple(tub, 0.03, 2.0, 5)
    W = build(bow_extra=[tub.T(-10.7, HA, zd(69))])
    bow, mud, stern, debris = merge([W["bow"], W["bow_extra"]]), W["mud"], W["stern"], W["debris"]
    RUN_STYLE = "quad"

    # Up close the hull is a patchwork of rust: most of the black paint has turned brown
    for m_, sd in ((bow, 201), (stern, 202), (debris, 203)):
        lum = m_.c.mean(1)
        k = 0.5 * sstep(0.25, 0.05, lum)
        m_.c = m_.c * (1 - k[:, None]) + np.array([0.24, 0.12, 0.07]) * k[:, None]
        weather(m_, rust=0.55, sediment=0.12, streaks=1.0, seed=sd, rust_col=(0.42, 0.17, 0.07))
        m_.c = np.clip(m_.c * (0.85 + 0.3 * fbm(m_.v, 0.6, sd + 5, 2))[:, None], 0, 1)

    # Where the wreck touches the ground (to keep things off the hull and on the flat ground Offlined makes)
    occ = {}
    for m_ in (bow, stern, debris, mud):
        low = m_.v[m_.v[:, 1] < 2.5]
        for x, z in np.floor(low[:, [0, 2]]).astype(int):
            occ[(x, z)] = True
    for x, z in np.floor(mud.v[:, [0, 2]]).astype(int):
        occ[(x, z)] = True
    circles = []                                            # the flat ground Offlined puts under each part
    for m_ in (bow, mud, stern, debris):
        lo, hi = m_.v.min(0), m_.v.max(0)
        cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
        circles.append((cx, cz, min(math.hypot(hi[0] - lo[0], hi[2] - lo[2]) / 2 + 15, 800) - 12))
    near_pts = np.array([k for k in occ.keys()], float)

    def free(near=0.65):
        for _ in range(80):
            if r.random() < near:
                b = near_pts[r.integers(len(near_pts))]
                x, z = b[0] + r.normal(0, 14), b[1] + r.normal(0, 14)
            else:
                c = circles[r.integers(len(circles))]
                a, d = r.uniform(0, 6.283), c[2] * math.sqrt(r.random())
                x, z = c[0] + d * math.cos(a), c[1] + d * math.sin(a)
            if any(math.hypot(x - c[0], z - c[1]) < c[2] for c in circles) and not any(
                    occ.get((int(math.floor(x)) + i, int(math.floor(z)) + j)) for i in (-1, 0, 1) for j in (-1, 0, 1)):
                return x, z
        return None

    def spots(n, near=0.65):
        out = [free(near) for _ in range(n)]
        return np.array([(x, 0.0, z) for x, z in (p for p in out if p)])

    def on(mesh, n, cond):
        s = abyss.sample_surface(mesh, n, cond, r)
        return np.array([p for p, _ in s]), np.array([q for _, q in s])

    INS = []                                                # (name, base mesh, material, instances)

    def add(name, variants, mat, pos, nrm=None, yaw=None, scale=None):
        """Split the copies among the variants of a shape (so they don't all look the same)."""
        if not len(pos):
            return
        pos = np.asarray(pos, float); N = len(pos)
        nrm = np.tile([0, 1.0, 0], (N, 1)) if nrm is None else np.asarray(nrm, float)
        yaw = r.uniform(0, 6.283, N) if yaw is None else np.asarray(yaw, float)
        scale = np.ones(N) if scale is None else np.asarray(scale, float)
        pick = r.integers(len(variants), size=N)
        for k, base in enumerate(variants):
            s = pick == k
            if s.any():
                INS.append((f"{name}_{k}", base, mat, instances(pos[s], nrm[s], yaw[s], scale[s])))

    # ── rust: icicles under every edge, runs down the walls ──
    ice = [abyss.rusticle(1.0, 0.1, 50 + k) for k in range(4)]
    flows = [abyss.rust_flow(1.0, 0.1, 60 + k) for k in range(4)]
    for m_, n_i, n_f in ((bow, 2600, 2000), (stern, 1000, 800), (debris, 240, 160)):
        p, _ = on(m_, n_i, lambda fn: fn[:, 1] < -0.35)
        ln = r.uniform(0.3, 1.0, len(p)) ** 1.6 * 2.4 + 0.15
        rad = r.uniform(0.5, 1.2, len(p)) * (0.6 + ln / 3)
        add("rusticles", ice, "rust", p, scale=np.stack([rad, ln, rad], 1))
        p, nn = on(m_, n_f, lambda fn: np.abs(fn[:, 1]) < 0.3)
        ln = r.uniform(0.3, 1.0, len(p)) ** 1.5 * 3.2 + 0.3
        w = r.uniform(0.4, 1.1, len(p))
        add("rust_runs", flows, "rust", p + nn * 0.02, yaw=np.arctan2(nn[:, 0], nn[:, 2]), scale=np.stack([w, ln, w], 1))

    # ── life on the wreck ──
    an_cols = [(0.95, 0.9, 0.85), (0.95, 0.9, 0.85), (0.95, 0.72, 0.7), (0.98, 0.55, 0.3)]
    anem = [abyss.anemone(0.06, c) for c in an_cols]
    coral = [abyss.mushroom_coral(0.08) for _ in range(2)]
    lobst = [abyss.squat_lobster(0.07) for _ in range(3)]
    stars = [abyss.brittle_star(0.1, c) for c in ((0.85, 0.7, 0.58), (0.92, 0.88, 0.82), (0.78, 0.55, 0.45))]
    for m_, a_, c_, l_, s_ in ((bow, 420, 90, 120, 140), (stern, 170, 45, 60, 70), (debris, 60, 10, 25, 25)):
        p, nn = on(m_, a_, lambda fn: fn[:, 1] > -0.2)
        add("anemones", anem, "life", p, nn, scale=r.uniform(0.6, 1.35, len(p)))
        p, nn = on(m_, c_, lambda fn: fn[:, 1] > 0.0)
        add("mushroom_corals", coral, "life", p, nn, scale=r.uniform(0.6, 1.3, len(p)))
        p, nn = on(m_, l_, lambda fn: fn[:, 1] > -0.4)
        add("squat_lobsters", lobst, "life", p + nn * 0.01, nn, scale=r.uniform(0.7, 1.3, len(p)))
        p, nn = on(m_, s_, lambda fn: fn[:, 1] > 0.6)
        add("brittle_stars", stars, "life", p + nn * 0.01, nn, scale=r.uniform(0.6, 1.1, len(p)))

    # ── the ground around: stones dropped by icebergs, and the animals of the abyssal plain ──
    stones, stone_pts = [], []
    for i in range(40):
        s = free(0.5)
        if not s:
            continue
        rad = r.uniform(0.25, 0.9) ** 2 * 2.6 + 0.15
        st = abyss.blob(rad, (0.42, 0.41, 0.4), 300 + i, sub=2, amp=0.35).S(1, r.uniform(0.5, 0.8), 1)
        base = np.array([(0.38, 0.37, 0.36), (0.5, 0.46, 0.42), (0.3, 0.3, 0.32)][i % 3])
        st.c = base * (0.8 + 0.4 * fbm(st.v, 2.0, i, 2))[:, None]
        weather(st, rust=0.08, sediment=0.35, streaks=0.0, seed=i, sed_col=(0.62, 0.58, 0.5))
        st.c *= 0.8
        stones.append(st.R(0, r.uniform(0, 6.28), 0).T(s[0], -rad * 0.25, s[1]))
        stone_pts.append((s[0], s[1], rad))
    stones = merge(stones)
    p, nn = on(stones, 160, lambda fn: fn[:, 1] > 0.3)
    add("anemones_on_stones", anem, "life", p, nn, scale=r.uniform(0.7, 1.4, len(p)))
    big = [(x, z, rad) for x, z, rad in stone_pts if rad > 0.8]
    if big:
        sp = [abyss.glass_sponge(0.6) for _ in range(2)]
        bc = [abyss.bamboo_coral(0.9, 70 + k) for k in range(3)]
        pts = np.array([(x, rad * 0.35, z) for x, z, rad in big])
        add("glass_sponges", sp, "life", pts, scale=r.uniform(0.6, 1.3, len(pts)))
        pts2 = pts + np.array([0.25, 0, 0.1])
        add("bamboo_corals", bc, "life", pts2[r.random(len(pts2)) < 0.6], scale=None)
    bs = spots(900, 0.65)
    add("brittle_stars_mud", stars, "life", bs + [0, 0.01, 0], scale=r.uniform(0.6, 1.3, len(bs)))
    cuc = [abyss.sea_cucumber("psychropotes"), abyss.sea_cucumber("psychropotes"), abyss.sea_cucumber("benthodytes"), abyss.sea_cucumber("benthodytes")]
    add("sea_cucumbers", cuc, "life", spots(120, 0.45))
    pens = [abyss.sea_pen(1.6) for _ in range(4)]
    pp = spots(60, 0.15)
    add("sea_pens", pens, "life", pp - [0, 0.05, 0], scale=r.uniform(0.55, 1.35, len(pp)))
    xen = [abyss.xeno(0.07) for _ in range(4)]
    xp = spots(420, 0.4)
    add("xenophyophores", xen, "life", xp + [0, 0.01, 0], scale=r.uniform(0.6, 1.4, len(xp)))
    mounds = [abyss.mound(0.25) for _ in range(2)]
    mp = spots(380, 0.3)
    add("burrow_mounds", mounds, "life", mp - [0, 0.01, 0], scale=r.uniform(0.5, 1.4, len(mp)))
    fish = [abyss.rattail(0.65) for _ in range(3)]
    fp = spots(45, 0.6)
    fp[:, 1] = r.uniform(0.25, 1.4, len(fp))
    tilt = np.stack([np.zeros(len(fp)), np.cos(0.45) * np.ones(len(fp)), -np.sin(0.45) * np.ones(len(fp))], 1)   # head down
    add("grenadiers", fish, "life", fp, tilt, scale=r.uniform(0.7, 1.3, len(fp)))

    # ── things from the ship, in the debris field ──
    def near_free(x, z):
        for k in range(400):
            a = k * 2.39996; d = 0.7 * math.sqrt(k)
            xx, zz = x + d * math.cos(a), z + d * math.sin(a)
            if not any(occ.get((int(math.floor(xx)) + i, int(math.floor(zz)) + j)) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)):
                return xx, zz
        return x, z

    shoes = [abyss.shoe(1.0, False), abyss.shoe(1.0, True)]
    for m_ in shoes:
        abyss.dust(m_, 0.45, 3)
    sp_, sy_, ss_ = [], [], []
    shoe_sites = [near_free(26, -250), near_free(-18, -420), near_free(52, -515)]
    for sx, sz in shoe_sites:
        for k in range(int(r.integers(3, 6))):
            x, z = sx + r.normal(0, 1.4), sz + r.normal(0, 1.4)
            yaw = r.uniform(0, 6.28); sc = r.uniform(0.85, 1.05)
            for s in (-1, 1):                                   # pairs: where a person came to rest
                sp_.append((x + math.cos(yaw) * s * 0.07, -0.02, z - math.sin(yaw) * s * 0.07)); sy_.append(yaw + s * 0.12); ss_.append(sc)
    add("shoes", shoes, "artefact", np.array(sp_), yaw=np.array(sy_), scale=np.array(ss_))
    plate = abyss.plate(0.13); abyss.dust(plate, 0.3, 4)
    cup = abyss.cup(); abyss.dust(cup, 0.3, 5)
    csx, csz = near_free(-30, -330)
    pp, pn, py = [], [], []
    for row in range(6):                                        # rows of plates still on edge where the racks rotted
        yaw = 0.3 + r.normal(0, 0.08)
        x0, z0 = csx + row * 0.9 * math.cos(yaw + 1.57), csz + row * 0.9 * math.sin(yaw + 1.57)
        for k in range(int(r.integers(8, 16))):
            pp.append((x0 + k * 0.045 * math.sin(yaw), 0.095, z0 + k * 0.045 * math.cos(yaw)))
            t = r.normal(0, 0.12)
            pn.append((math.sin(yaw) * math.cos(t), math.sin(t), math.cos(yaw) * math.cos(t))); py.append(r.uniform(0, 6.28))
    for k in range(30):
        pp.append((csx + r.normal(0, 2.5), 0.0, csz + r.normal(0, 2.5))); pn.append((r.normal(0, 0.15), 1, r.normal(0, 0.15))); py.append(r.uniform(0, 6.28))
    add("plates", [plate], "china", np.array(pp), np.array(pn), np.array(py))
    cp = np.array([(csx + r.normal(0, 2.2), 0.0, csz + r.normal(0, 2.2)) for _ in range(22)])
    side = r.random(len(cp)) < 0.6
    cn = np.where(side[:, None], np.stack([np.cos(r.uniform(0, 6.28, len(cp))), np.zeros(len(cp)), np.sin(r.uniform(0, 6.28, len(cp)))], 1), [0, 1, 0])
    cp[side, 1] = 0.045
    add("cups", [cup], "china", cp, cn)
    bottles = [abyss.bottle(True), abyss.bottle(False)]
    for m_ in bottles:
        abyss.dust(m_, 0.5, 6)
    bsx, bsz = near_free(38, -385)
    bp = [(bsx + i * 0.1, 0.045, bsz + j * 0.1) for i in range(5) for j in range(6) if r.random() > 0.15]
    bp += [(bsx + r.normal(0, 1.6), 0.045, bsz + r.normal(0, 1.6)) for _ in range(18)]
    bp = np.array(bp)
    bn = np.stack([np.zeros(len(bp)), np.zeros(len(bp)), np.ones(len(bp))], 1)
    bn[30:] = np.stack([np.cos(a := r.uniform(0, 6.28, len(bp) - 30)), np.zeros(len(a)), np.sin(a)], 1)
    add("bottles", bottles, "glass", bp, bn, yaw=np.zeros(len(bp)))
    bench = abyss.bench_end(); abyss.dust(bench, 0.4, 7)
    bnx, bnz = near_free(8, -185)
    bep, bey = [], []
    for k in range(3):
        x, z = bnx + k * 3.2 + r.normal(0, 0.5), bnz + r.normal(0, 1.5)
        yaw = r.uniform(0, 6.28)
        for s in (-0.9, 0.9):
            bep.append((x + math.cos(yaw) * s, -0.03, z - math.sin(yaw) * s)); bey.append(yaw)
    add("bench_ends", [bench], "artefact", np.array(bep), yaw=np.array(bey))
    pots = [abyss.pot(0.16) for _ in range(2)]
    for m_ in pots:
        abyss.dust(m_, 0.4, 8)
    ptx, ptz = near_free(62, -468)
    pp = np.array([(ptx + r.normal(0, 1.5), -0.02, ptz + r.normal(0, 1.5)) for _ in range(9)])
    pn = np.where((r.random(len(pp)) < 0.5)[:, None], [0, 1, 0], [1, 0.2, 0])
    add("pots", pots, "artefact", pp, pn, scale=r.uniform(0.7, 1.35, len(pp)))
    singles = []
    sfx, sfz = near_free(-44, -472)
    singles.append(abyss.safe().R(0, 0.7, 0).T(sfx, -0.25, sfz))
    tgx, tgz = near_free(34, -538)
    singles.append(abyss.telegraph().R(0.25, 0, 0.1).R(0, 2.0, 0).T(tgx, -0.15, tgz))
    for k in range(2):
        singles.append(abyss.suitcase().R(0, r.uniform(0, 6), 0).T(shoe_sites[1][0] + 2.5 + k * 0.8, -0.03, shoe_sites[1][1] - 1.2 + k))
    singles = merge(singles)
    abyss.dust(singles, 0.45, 9)

    # ── light: ambient occlusion baked into the colours of the big pieces ──
    flat = lambda X, Z: np.zeros_like(X)
    bake_ao([bow, stern, debris, mud, stones], occluders=[bow, stern, debris, stones], ground=flat, cell=0.6, rays=24, reach=12)

    mats = {
        "steel": material(rough=0.8, metal=0.05, tex=steel_textures(), uv=("tri", 7.2)),
        "rust": material(rough=0.95, tex=rust_textures(), uv=("tri", 1.5), normal_scale=1.6),
        "rock": material(rough=0.9, tex=rust_textures(512, 2.5, 11), uv=("tri", 2.5)),
        "mud": material(rough=1.0),
        "life": material(rough=0.55),
        "china": material(rough=0.18),
        "glass": material(rough=0.08),
        "artefact": material(rough=0.7, metal=0.1),
    }
    parts = [("bow_section", bow, "steel"), ("mud", mud, "mud"), ("stern_section", stern, "steel"),
             ("debris_field", [(debris, "steel"), (singles, "artefact"), (stones, "rock")])]

    # ── the ship as built: planked decks, clean plating, and its own ambient occlusion ──
    ship = W["ship"]
    deckc = np.array(DECK)
    isdeck = (np.abs(ship.c - deckc).sum(1) < 0.02)
    fd = isdeck[ship.f].all(1)
    rigging = []
    for d, hgt, sx in ((45, 61.0, 1), (222, 57.0, 1)):         # shrouds from the mast heads to the sides
        for s in (-1, 1):
            for dd in (-3, 0, 3):
                rigging.append(tube([(0, hgt - 2, zd(d)), (s * B * 0.48, yd_of(t_of(zd(d + dd))) + 0.5, zd(d + dd))], 0.03, (0.18, 0.17, 0.16), seg=3))
    rig = merge(rigging)
    bake_ao([ship], occluders=[ship], ground=None, cell=0.5, rays=20, reach=10, strength=0.75)
    deck = Mesh(ship.v, ship.f[fd], ship.c)
    rest = Mesh(ship.v, ship.f[~fd], ship.c)
    bmats = {"paint": material(rough=0.6, metal=0.05, tex=steel_textures(1024, 7.2, 12, wear=0.15), uv=("tri", 7.2), normal_scale=0.6),
             "deck": material(rough=0.75, tex=planks_textures(), uv=("xz", 4.0)),
             "rigging": material(rough=0.8)}
    bparts = [("main", [(rest, "paint"), (deck, "deck"), (rig, "rigging")])]
    return parts, INS, mats, bparts, bmats


# ── Write ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from scenekit import write_model
    parts, ins, mats, bparts, bmats = dress()
    out = ROOT / "models" / "item" / "Q25173" / "model.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    tris, size = write_model(out, parts, mats, instanced=ins)
    print(f"{out}: {tris:,} triangles drawn ({len(ins)} instanced shapes), {size / 2**20:.2f} MB")
    tris2, size2 = write_model(out.with_name("model_built.glb"), bparts, bmats)
    print(f"model_built.glb: {tris2:,} triangles, {size2 / 2**20:.2f} MB")
