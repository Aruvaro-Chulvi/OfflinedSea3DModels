"""
RMS Lusitania (Q187425) — the wreck off the Old Head of Kinsale, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: Cunard four-funnel liner of 1907, length overall 239.9 m (787 ft), beam 26.8 m, moulded depth 17.3 m,
draught 10.2 m, 43.9 m from the keel to the top of the funnels; black hull with a red bottom and a gilt line,
white superstructure, Cunard red funnels with black tops, two masts, four propellers (four-bladed from 1909),
22 lifeboats in davits in 1915. Deck heights, funnel positions and the layout of the fittings are approximate.

The wreck, after the published surveys and dive reports: she lies in 93 m of water, 11.5 nautical miles off the
Old Head of Kinsale, on her starboard side at an angle of some 30-40 degrees, bow to the north-east (Marine
Institute / INFOMAR multibeam; NNE in an older wreck list). The superstructure and the four funnels are gone;
the upper port side has fallen onto the starboard decks and the hull has slumped to about half its beam,
standing about 14 m above the seabed; the bow is the best-preserved part, the stern is split open and nearly
flattened (the counter stern can no longer be seen), and she rests in two parts with a break amidships (placed
here just aft of the third funnel, a guess). Boilers lie exposed, the docking bridge lies on the seabed on the
starboard (deck) side among a debris field, fishing nets drape the wreck, and only one of the four propellers is
left (three were blasted off by Oceaneering in 1982): a starboard one, trapped under the hull. The angle of the
roll, the amount of collapse along the hull, the place of the break, the nets and the debris layout are
approximate.

    python model_sources/Q187425_lusitania.py   -> models/item/Q187425/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, weather, sstep, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1915)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B = 239.9, 26.8
H1 = 17.3                # keel to the upper deck (moulded depth): the well deck
H = 19.8                 # keel to the shelter deck = top of the black hull (forecastle and main hull)
HB = 22.6                # top of the B-deck house
HA = 25.4                # top of the A deck = boat deck
HT = 28.2                # top of the boat-deck houses
HF = 43.9                # top of the funnels above the keel
WL = 10.2                # waterline (draught)
BLACK, RED, WHITE = (0.07, 0.07, 0.08), (0.45, 0.12, 0.08), (0.86, 0.84, 0.78)
DECK, DARK, GOLD = (0.62, 0.52, 0.38), (0.025, 0.025, 0.03), (0.72, 0.56, 0.22)
FRED, MAST, BRONZE = (0.78, 0.3, 0.12), (0.62, 0.48, 0.3), (0.55, 0.4, 0.2)
WOOD = (0.55, 0.42, 0.28)
FUNNELS = (76, 100, 124, 148)          # distance of each funnel from the bow (m), approximate
SPLIT = 131.0                          # the split amidships, just aft of the third funnel


def zd(d):            # distance from the bow (m) -> ship coordinate z (bow at +L/2)
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):          # half-beam factor along the ship: a long fine entrance (she was built for speed)
    w = 1.0
    if t > 0.74:
        w = math.sin(((1 - t) / 0.26) * math.pi / 2) ** 1.15
    if t < 0.14:
        w = min(w, 0.3 + 0.7 * math.sin((t / 0.14) * math.pi / 2))
    return max(w, 0.004)


def yb_of(t):         # keel line (rises at the counter stern and a little at the forefoot)
    y = 0.0
    if t < 0.075:
        y = H1 * 0.6 * ((0.075 - t) / 0.075) ** 1.6
    if t > 0.99:
        y = max(y, H1 * 0.8 * ((t - 0.99) / 0.01) ** 2)
    return y


def yd_of(t):         # upper deck with a little sheer
    return H1 * (1 + 0.035 * (2 * t - 1) ** 2)


def k_of(t):          # section fullness (a fine, V-shaped run aft where the shafts come out)
    k = 2.0 + 5.5 * w_of(t) ** 3
    return 1.1 + (k - 1.1) * float(sstep(0.03, 0.22, t))


def hw(z, y):         # half-width of the hull at z and height y
    t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    if y >= yd:
        return B / 2 * w
    u = max(0.0, (y - yb) / max(0.1, yd - yb))
    return B / 2 * w * (1 - (1 - u) ** k)


def hull(d0, d1, NP=26):
    """Hull from d0 to d1 (distance from bow): both sides, the upper deck and, at the stern, the counter."""
    ns = max(2, int(math.ceil((d1 - d0) / 1.25)))
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
    if d1 >= L - 0.01:                                       # counter stern closed
        c0 = len(v); z, w, yb, yd, k = st[0]
        v.append((0, (yb + yd) / 2, z))
        f += [(c0, j + 1, j) for j in range(W - 1)]
    m = Mesh(v, f)
    deck = np.array([p[1] >= yd_of(t_of(p[2])) - 0.02 and abs(p[0]) < B / 2 * w_of(t_of(p[2])) - 0.05 for p in m.v])
    m.paint(lambda p: np.where((p[:, 1] < WL)[:, None], RED, BLACK))
    m.c[deck] = DECK
    return m


def band(d0, d1, y_extra, color=BLACK, top=DECK, ends=True):
    """A raised hull band on the upper deck (forecastle, shelter deck; or the bulwark of the well deck if top=None)."""
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
    if top is not None:
        v, f = [], []
        for z, y0, x in zip(zs, ys, xs):
            v += [(-x, y0 + y_extra, z), (x, y0 + y_extra, z)]
        for i in range(ns):
            a = 2 * i; b = a + 2; c = a + 1; d = b + 1
            f += [(a, b, c), (c, b, d)]
        parts.append(Mesh(v, f, top))
    if ends:
        for z, y0, x, s in ((zs[0], ys[0], xs[0], -1), (zs[-1], ys[-1], xs[-1], 1)):
            if x > 0.5:
                parts.append(Mesh([(-x, y0, z), (x, y0, z), (x, y0 + y_extra, z), (-x, y0 + y_extra, z)],
                                  [(0, 1, 2), (0, 2, 3)] if s > 0 else [(0, 2, 1), (0, 3, 2)], color))
    return merge(parts)


def rot_holes(m, da, db, ymin, seed, port_only=True):
    """Where whole plates have rotted away: faces removed in patches on the (port, upper) side of the hull."""
    c = m.v[m.f].mean(axis=1)
    d = L / 2 - c[:, 2]
    n = fbm(c, (0.07, 0.1, 0.045), seed, 3)
    w = sstep(da, da + 30, d) * sstep(db, db - 10, d) * sstep(ymin, ymin + 1.5, c[:, 1]) * (c[:, 0] > 0.5 if port_only else 1)
    m.f = m.f[~((n > 0.585 - 0.05 * sstep(150, 230, d)) & (w > 0.5))]
    return m


INNER = (0.051, 0.047, 0.043)          # the dark inside of the hull, seen through the holes (kept unweathered)


def inner_wall(d0, d1):
    """A dark wall along the centreline inside the hull: what you see through the rotted plates."""
    ds = np.arange(d0, d1 + 0.01, 2.0)
    v, f = [], []
    for d in ds:
        t = t_of(zd(d))
        v += [(0.2, max(yb_of(t) + 0.6, 1.2), zd(d)), (0.2, yd_of(t) + (H - H1) - 0.4, zd(d))]
    for i in range(len(ds) - 1):
        a = 2 * i
        f += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)]
    return Mesh(v, f, INNER)


def at(m, d, x=0.0, y=0.0):
    return m.T(x, y, zd(d))


def deck_y(d):        # the upper deck (well deck)
    return yd_of(t_of(zd(d)))


def top_y(d):         # the shelter deck (forecastle, main hull)
    return deck_y(d) + (H - H1)


def boiler(length=5.6, r=2.5):
    """A Scotch boiler lying on its side (axis along x), with its furnace mouths at one end."""
    b = merge([cyl(r, r, length, (0.16, 0.12, 0.09), seg=18)] +
              [cyl(0.5, 0.5, 0.06, DARK, seg=10).T(dx, length, dz) for dx, dz in ((-1.1, -0.6), (1.1, -0.6), (0, 0.9), (0, -1.6))])
    return b.T(0, -length / 2, 0).R(0, 0, math.pi / 2)


def propeller(r=2.4, nb=4):
    parts = [cyl(0.55, 0.55, 1.3, BRONZE, seg=12).R(math.pi / 2, 0, 0).T(0, 0, -0.65)]
    for k in range(nb):
        a = 2 * math.pi * k / nb
        parts.append(box(r * 0.95, 0.16, 1.25, BRONZE, bottom=False).T(r * 0.5 + 0.35, 0, 0).R(0.38, 0, 0).R(0, 0, a))
    return merge(parts)


# Four screws: the wing (outer) shafts come out further forward than the inner ones. (x, y, distance from bow)
SCREWS = {"port_outer": (8.8, 4.6, L - 22), "port_inner": (3.4, 4.0, L - 10),
          "stbd_inner": (-3.4, 4.0, L - 10), "stbd_outer": (-8.8, 4.6, L - 22)}


def shafts_and_props(keep=("port_outer", "port_inner", "stbd_inner", "stbd_outer"), stubs=True):
    """Shaft brackets, shafts and propellers; screws not in keep are left as bare shaft ends (removed by salvors)."""
    P = []
    for name, (x, y, d) in SCREWS.items():
        z = zd(d)
        zin = zd(d - 16)                                       # where the shaft comes out of the hull
        xin = math.copysign(min(abs(x), hw(zin, y) - 0.3), x)
        P.append(tube([(xin, y + 0.4, zin), (x, y, z + 1.0)], 0.32, BRONZE if name in keep else (0.25, 0.2, 0.15), seg=8))
        P.append(tube([(x, y + 0.2, z + 1.6), (x * 0.45, y + 2.6, z + 1.6)], 0.2, BLACK, seg=5))       # A-bracket
        if name in keep:
            P.append(propeller().T(x, y, z))
        elif stubs:
            P.append(cyl(0.45, 0.45, 0.6, (0.3, 0.22, 0.12), seg=10).R(math.pi / 2, 0, 0).T(x, y, z + 0.6))  # cut shaft end
    return P


def rudder():
    zs = zd(L) + 2.0
    return box(0.8, 9.0, 5.0, RED).T(0, 1.2, zs + 2.5)


def ship_part(d0, d1, built=False, port_only=False):
    """Everything of the ship between d0 and d1 (distance from the bow). built=True: as she sailed;
    built=False: what is left on the wreck (hull, forecastle gear, open hatches; no superstructure)."""
    P = [hull(d0, d1)]
    if not built:
        P[0] = rot_holes(P[0], 40, L - 8, 5.0, seed=int(d0) + 7)
    inr = lambda d, m=0: d0 + m <= d <= d1 - m
    sides = (1,) if port_only else (-1, 1)
    # forecastle and main hull raised to the shelter deck; the well deck between them, with its bulwark
    for a, b in ((0, 30), (52, L)):
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 > 1:
            bd = band(a2, b2, H - H1)
            P.append(bd if built else rot_holes(bd, 40, L - 8, 5.0, seed=int(a2) + 17))
            for s in (-1, 1):                                 # the gilt line along the top of the black hull
                pts = [(s * (B / 2 * w_of(t_of(zd(d)))) + s * 0.04, top_y(d) - 0.45, zd(d)) for d in np.arange(a2 + 0.3, b2, 2.0)]
                if len(pts) > 1:
                    P.append(tube(pts, 0.08, GOLD, seg=4))
    a2, b2 = max(30, d0), min(52, d1)
    if b2 - a2 > 1:
        P.append(band(a2, b2, 1.2, top=None, ends=False))
    # Bilge keels along the turn of the bilge
    for s_ in sides:
        pts = []
        for d in np.arange(max(70, d0), min(170, d1) + 0.1, 4.0):
            z = zd(d); y = 1.6
            pts.append((s_ * (hw(z, y) + 0.25), y - 0.35, z))
        if len(pts) > 1:
            P.append(tube(pts, 0.3, RED if built else BLACK, seg=4).S(1, 1, 1))
    # Portholes (on the wreck, each with a rust run below it)
    for y, a, b, sp in ((12.2, 26, 222, 2.3), (14.6, 12, 228, 1.9), (16.6, 8, 232, 1.8), (H - 1.3, 54, 234, 1.8), (H - 1.3, 3, 28, 2.2)):
        for d in np.arange(max(a, d0 + 1), min(b, d1 - 1), sp):
            z = zd(d)
            x = hw(z, y)
            if x < 2:
                continue
            for s in sides:
                P.append(quad(s * x, y, z, 0.42, 0.42, "+x" if s > 0 else "-x", DARK, eps=0.06))
                if not built and rng.random() < 0.45:
                    ln = rng.uniform(0.8, 2.2)
                    P.append(quad(s * hw(z, y - ln / 2), y - 0.2 - ln / 2, z, 0.3, ln, "+x" if s > 0 else "-x", (0.42, 0.22, 0.09), eps=0.05))
    # Hatches (well deck and aft deck; open on the wreck)
    for d, wd, yfn in ((34, 6, deck_y), (46, 6, deck_y), (214, 6, top_y)):
        if inr(d, 3):
            y = yfn(d)
            P.append(at(box(wd, 0.7, 5.0, BLACK), d, y=y))
            P.append(at(box(wd - 0.6, 0.02, 4.4, (0.38, 0.34, 0.27) if built else DARK), d, y=y + 0.7))
    # Forecastle: railing, capstans and windlass, cables to the hawse pipes, the anchors in their pockets
    if d0 <= 1:
        if built:
            pts_r = [(hw(zd(d), H + 2) - 0.25, top_y(d) + 1.15, zd(d)) for d in np.arange(0.6, 30, 1.5)]
            for s in (-1, 1):
                P.append(tube([(s * x, y, z) for x, y, z in pts_r], 0.07, WHITE, seg=5))
                for x, y, z in pts_r[::2]:
                    P.append(cyl(0.05, 0.05, 1.15, WHITE, seg=4, caps=False).T(s * x, y - 1.15, z))
        for d, x in ((9, 2.4), (9, -2.4), (13, 0), (17, 3.6), (17, -3.6)):
            P.append(at(cyl(0.75, 0.62, 1.0, BLACK, seg=12), d, x=x, y=top_y(d)))
        for s in (-1, 1):                                                                        # the huge cable links
            z1 = zd(4.0)
            pts = [(s * 2.4, top_y(9) + 0.5, zd(9)), (s * 3.3, top_y(6.5) + 0.25, zd(6.5)), (s * (hw(z1, H) - 0.6), top_y(4.0) + 0.15, z1)]
            P.append(tube(pts, 0.26, (0.16, 0.12, 0.09), seg=6))
            z = zd(7.5); x = s * (hw(z, 15.8) + 0.3)
            P.append(box(0.55, 3.4, 0.8, BLACK).T(x, 14.2, z))                                   # anchor shank and stock
            P.append(box(0.6, 0.7, 2.8, BLACK).T(x, 14.0, z))
    # Cargo cranes in the well deck and aft
    for d, x in ((38, 4.2), (38, -4.2), (50, 4.2), (50, -4.2), (210, 4), (210, -4), (220, 4), (220, -4)):
        if inr(d, 1) and built:
            y = deck_y(d) if d < 60 else top_y(d)
            P.append(at(cyl(0.85, 0.75, 2.0, WHITE, seg=10), d, x=x, y=y))
            P.append(tube([(x, y + 2.0, zd(d)), (x * 1.3, y + 8.0, zd(d) + (5.5 if d < 100 else -5.5))], 0.17, WHITE, seg=5))
    if not built:
        return P
    # ── what only the ship as built has: superstructure, funnels, masts, boats ──
    a2, b2 = max(54, d0), min(196, d1)                         # B-deck house, white, with a row of windows
    if b2 - a2 > 1:
        P.append(at(box(B * 0.96, HB - H, b2 - a2, WHITE), (a2 + b2) / 2, y=H))
        for d in np.arange(a2 + 1.2, b2 - 1, 2.3):
            for s in (-1, 1):
                P.append(quad(s * B * 0.48, H + 1.5, zd(d), 1.1, 1.0, "+x" if s > 0 else "-x", DARK))
    a2, b2 = max(56, d0), min(190, d1)                         # A deck: closed forward, open promenade aft
    if b2 - a2 > 1:
        P.append(at(box(B * 0.95, 0.4, b2 - a2, WHITE), (a2 + b2) / 2, y=HA - 0.4))
        P.append(at(box(B * 0.95, 0.03, b2 - a2, DECK), (a2 + b2) / 2, y=HA))
        c1 = min(b2, max(a2, 72))
        if c1 - a2 > 1:
            P.append(at(box(B * 0.95, HA - HB - 0.4, c1 - a2, WHITE), (a2 + c1) / 2, y=HB))
            for d in np.arange(a2 + 1.4, c1 - 1, 2.5):
                for s in (-1, 1):
                    P.append(quad(s * B * 0.475, HB + 1.5, zd(d), 1.2, 1.1, "+x" if s > 0 else "-x", DARK))
        if b2 - c1 > 1:
            P.append(at(box(B * 0.8, HA - HB - 0.4, b2 - c1, WHITE), (c1 + b2) / 2, y=HB))
            for d in np.arange(c1 + 1, b2, 3.0):
                for s in (-1, 1):
                    P.append(at(box(0.3, HA - HB - 0.4, 0.3, WHITE), d, x=s * B * 0.46, y=HB))
                    P.append(quad(s * B * 0.4, HB + 1.4, zd(d + 1.5), 1.2, 1.1, "+x" if s > 0 else "-x", DARK))
    a2, b2 = max(196, d0), min(212, d1)                        # small deckhouse on the after deck
    if b2 - a2 > 1:
        P.append(at(box(B * 0.5, 2.6, b2 - a2, WHITE), (a2 + b2) / 2, y=top_y((a2 + b2) / 2)))
    # boat-deck houses between the funnels (bridge house, entrances, lounges)
    for a, b, wf in ((58, 68, 0.55), (81, 95, 0.45), (105, 119, 0.45), (129, 143, 0.45), (153, 172, 0.5), (176, 186, 0.45)):
        a2, b2 = max(a, d0), min(b, d1)
        if b2 - a2 > 1:
            P.append(at(box(B * wf, HT - HA, b2 - a2, WHITE), (a2 + b2) / 2, y=HA))
            for d in np.arange(a2 + 1.2, b2 - 0.8, 2.2):
                for s in (-1, 1):
                    P.append(quad(s * B * wf / 2, HA + 1.6, zd(d), 0.8, 0.9, "+x" if s > 0 else "-x", DARK))
    if inr(57):                                                 # the navigating bridge: wheelhouse and wings
        P.append(at(box(B * 1.0, 0.45, 2.4, WHITE), 57.0, y=HT - 0.45))
        P.append(at(box(9.0, 2.6, 4.0, WHITE), 57.5, y=HT))
        for x in np.arange(-3.6, 3.7, 0.9):
            P.append(quad(x, HT + 1.7, zd(55.5), 0.65, 0.9, "+z", DARK))
    for s in (-1, 1):                                           # boat-deck railing
        pts = [(s * B * 0.47, HA + 1.05, zd(d)) for d in np.arange(max(57, d0) + 0.5, min(190, d1), 2.5)]
        if len(pts) > 1:
            P.append(tube(pts, 0.05, WHITE, seg=4))
            for x, y, z in pts[::2]:
                P.append(cyl(0.04, 0.04, 1.05, WHITE, seg=4, caps=False).T(x, HA, z))
    # Funnels: Cunard red with black tops and thin black bands, oval, raked aft, 43.9 m above the keel
    rake = 0.07
    for d in FUNNELS:
        if not inr(d, 4):
            continue
        hgt = HF - HA
        parts = [cyl(3.5, 3.4, hgt - 4.0, FRED, seg=24, sx=0.74),
                 cyl(3.4, 3.35, 4.0, BLACK, seg=24, sx=0.74).T(0, hgt - 4.0, 0),
                 cyl(3.36, 3.36, 0.05, DARK, seg=24, sx=0.74).T(0, hgt + 0.01, 0)]
        for yy in (hgt * 0.62, hgt * 0.7):
            parts.append(cyl(3.44, 3.44, 0.35, BLACK, seg=24, sx=0.745, caps=False).T(0, yy, 0))
        P.append(at(merge(parts).R(-rake, 0, 0), d, y=HA))
    # Ventilator cowls on the boat deck
    for d, x in ((86, 8.5), (92, 8.5), (110, 8.5), (116, 8.5), (134, 8.5), (140, 8.5), (158, 8.0), (166, 8.0)):
        if inr(d):
            for s in (-1, 1):
                P.append(at(cyl(0.5, 0.5, 2.2, WHITE, seg=10), d, x=s * x, y=HA))
                P.append(at(box(1.2, 1.0, 1.1, WHITE), d + 0.4, x=s * x, y=HA + 2.0))
    # 22 lifeboats in their davits (1915), 11 a side
    for d in (61, 70, 83.5, 92, 107.5, 116, 131.5, 140, 156, 165, 181):
        if not inr(d):
            continue
        for s in (-1, 1):
            x = s * B * 0.47
            for dd in (d - 3.6, d + 3.6):
                P.append(tube([(x, HA, zd(dd)), (x, HA + 3.0, zd(dd)), (x + s * 0.8, HA + 3.8, zd(dd)), (x + s * 1.5, HA + 3.4, zd(dd))], 0.16, WHITE, seg=6))
            boat = lathe([(0.05, -4.3), (0.8, -3.6), (1.3, -2.2), (1.4, 0), (1.3, 2.2), (0.8, 3.6), (0.05, 4.3)], WHITE, seg=10, axis="z")
            boat.v[:, 1] = np.minimum(boat.v[:, 1], 0.3)
            boat.S(1, 0.85, 1)
            P.append(boat.T(x + s * 0.5, HA + 2.0, zd(d)))
    # Masts (with the crow's nest on the foremast) and the wireless aerial between them
    for d, h in ((40, 62.0), (178, 60.0)):
        if inr(d):
            y0 = deck_y(d) if d < 52 else HA
            P.append(at(cyl(0.45, 0.2, h - y0, MAST, seg=10).R(-0.06, 0, 0), d, y=y0))
    if inr(40):
        P.append(at(cyl(1.0, 1.0, 1.6, MAST, seg=12), 40.6, y=34.0))
        P.append(tube([(0, 61.5, zd(41.5)), (0, top_y(1) + 1.2, zd(1.2))], 0.04, (0.2, 0.2, 0.2), seg=4))      # forestay
    if d0 <= 40 and d1 >= 178:
        P.append(tube([(0, 60.5, zd(41.5)), (0, 58.5, zd(179.5))], 0.03, (0.2, 0.2, 0.2), seg=4))
    if inr(228, 2):                                             # docking bridge at the stern
        P.append(at(box(B * 0.62, 0.4, 2.0, WHITE), 228, y=top_y(228) + 3.4))
        for s in (-1, 1):
            P.append(at(cyl(0.12, 0.12, 3.4, WHITE, seg=5), 228, x=s * 3, y=top_y(228)))
    return P


def torn_plates(d, n, seed, colors=(BLACK,), toward=+1):
    """Torn plates sticking out of a break at distance d from the bow (toward=+1: aft, -1: forward)."""
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = r.uniform(2, H)
        wdt = hw(zd(d), y) * 0.95
        p = box(r.uniform(2, 5.5), 0.12, r.uniform(2, 6), colors[r.integers(len(colors))], bottom=False)
        crumple(p, 0.7, 0.35, seed + i)
        out.append(p.R(r.uniform(-0.6, 0.6), r.uniform(-0.4, 0.4), r.uniform(-0.8, 0.8)).T(r.uniform(-wdt, wdt), y, zd(d) - toward * r.uniform(0.3, 3.0)))
    return merge(out)


def pancaked_decks(d0, d1, seed):
    """The collapsed decks: long buckled slabs stacked on the (starboard) deck side of the slumped hull."""
    r = np.random.default_rng(seed)
    out = []
    cols = ((0.28, 0.2, 0.14), (0.58, 0.54, 0.47), (0.4, 0.32, 0.24))
    for i, y in enumerate((H + 0.3, H + 1.4, H + 2.6)):
        d = d0
        while d < d1 - 4:
            ln = r.uniform(5, 12)
            ln = min(ln, d1 - d)
            wf = r.uniform(0.45, 0.85)
            p = box(B * wf, 0.22, ln, cols[r.integers(3)])
            crumple(p, 1.1, 0.22, seed + i * 50 + int(d))
            p.R(r.uniform(-0.08, 0.08), r.uniform(-0.06, 0.06), r.uniform(-0.15, 0.15))
            out.append(at(p, d + ln / 2, x=r.uniform(-1.5, 1.5) - (B * (1 - wf) / 2) * 0.6, y=y))
            d += ln + r.uniform(1.0, 6.0)
    return merge(out)


def growth(m, seed, amount=0.5):
    """Marine growth of the Celtic Sea (anemones, hydroids, dead man's fingers): pale specks on exposed faces."""
    sp = fbm(m.v, 1.6, seed, 2)
    pa = fbm(m.v, 0.1, seed + 5, 2)
    k = np.clip(amount * sstep(0.56, 0.72, sp) * sstep(0.35, 0.65, pa), 0, 0.85)[:, None]
    tint = np.where((fbm(m.v, 0.5, seed + 9, 1) > 0.55)[:, None], (0.8, 0.55, 0.32), (0.82, 0.78, 0.68))
    m.c = m.c * (1 - k) + tint * k
    return m


# ── The wreck ────────────────────────────────────────────────────────────────
ROLL = math.radians(58)          # rolled onto her starboard side (the deck faces starboard and down)
LIFT = 7.0                       # how far the rolled hull is raised so the starboard bilge sits ~4 m in the mud
Y0 = 1.0                         # the collapse squeezes everything above this height


def lay_down(m, d_of_v, keep, roll=ROLL, lean=0.38):
    """Roll the piece onto starboard about the keel, settle it into the mud and let the upper (port) side slump
    down onto the starboard decks: heights above Y0 are scaled by keep(d) and pushed towards the deck side."""
    m.R(0, 0, roll)
    m.v[:, 1] += LIFT
    k = keep(d_of_v)
    y = m.v[:, 1]
    up = np.maximum(0.0, y - Y0)
    m.v[:, 0] -= up * (1 - k) * lean
    m.v[:, 1] = np.where(y > Y0, Y0 + up * k, y)
    return m


def keep_fn(d):
    d = np.asarray(d, float)
    k = 0.66 - 0.2 * sstep(18, 70, d) - 0.1 * sstep(90, 128, d)       # bow proud, slumping aft
    k = k - 0.1 * sstep(SPLIT + 2, 175, d) - 0.12 * sstep(195, 236, d)  # stern nearly flat
    return k


# Bow section: from the stem to the split
bow_parts = ship_part(0, SPLIT, port_only=True) + [pancaked_decks(54, SPLIT - 2, 71), inner_wall(40, SPLIT - 1)]
for i, (d, y) in enumerate(((78, 7.5), (92, 7.0), (114, 7.5), (126, 6.8))):    # boilers inside, lying on their sides
    bow_parts.append(at(boiler(), d, x=rng.uniform(-3, 3), y=y))
bow_parts.append(torn_plates(SPLIT, 26, 300, (BLACK, BLACK, DECK)))
bow = merge(bow_parts)
dfb = L / 2 - bow.v[:, 2]
crumple(bow, 1.6, 0.12, 11, sstep(85, SPLIT, dfb))                       # buckling towards the split
crumple(bow, 2.6, 0.35, 12, sstep(SPLIT - 6, SPLIT, dfb))                # torn edges at the split
crumple(bow, 0.3, 0.08, 13)                                              # dents everywhere
lay_down(bow, dfb, keep_fn)
crumple(bow, 0.9, 0.06, 14, sstep(40, 120, dfb))                         # the slumped side, wavy
bow.v[:, 1] -= 1.2 * np.sin(np.clip(dfb / SPLIT, 0, 1) * math.pi) * sstep(-1, 6, bow.v[:, 1])   # the curved keel

# Stern section: from the split to the counter, split open and nearly flat; one starboard propeller left
stern_parts = ship_part(SPLIT + 1.5, L, port_only=True) + [pancaked_decks(SPLIT + 4, 226, 81), inner_wall(SPLIT + 2.5, L - 8)]
stern_parts += shafts_and_props(keep=("stbd_outer",))
_px, _py, _pd = SCREWS["stbd_outer"]
for d, y in ((138, 7.0), (146, 7.6)):
    stern_parts.append(at(boiler(), d, x=rng.uniform(-3, 3), y=y))
stern_parts.append(torn_plates(SPLIT + 1.5, 24, 400, (BLACK, DECK), toward=-1))
stern_parts.append(Mesh([(_px, _py, zd(_pd))] * 3, [(0, 1, 2)]))        # marker: where the last propeller ends up
stern = merge(stern_parts)
dfs = L / 2 - stern.v[:, 2]
stern.v[:, 0] *= 1 + 0.35 * sstep(150, 200, dfs) * sstep(4, 16, stern.v[:, 1])   # split open: the sides peeled apart
crumple(stern, 2.6, 0.12, 31, sstep(SPLIT + 30, SPLIT, dfs))
crumple(stern, 3.0, 0.12, 32, sstep(222, L, dfs))                        # the counter stern torn and crushed
crumple(stern, 0.35, 0.08, 33)
lay_down(stern, dfs, keep_fn, roll=math.radians(64), lean=0.45)
stern.v[:, 1] -= 2.2 * sstep(222, L, dfs) * sstep(-2, 8, stern.v[:, 1])  # the stern sunk into the mud
crumple(stern, 1.0, 0.06, 34)
# a little out of line with the bow at the split
zs0 = zd(SPLIT)
stern.T(0, 0, -zs0).R(0, math.radians(-3.0), 0).T(1.2, 0, zs0 - 1.0)
PROP = stern.v[-1].copy()
stern.f = stern.f[:-1]
stern.v, stern.c = stern.v[:-3], stern.c[:-3]

hull_m = merge([bow, stern])
inside = np.all(np.abs(hull_m.c - np.array(INNER)) < 1e-6, axis=1)
hull_m.c = hull_m.c * 0.42 + np.array((0.27, 0.2, 0.15)) * 0.58              # a century of rust: the paint is long gone
weather(hull_m, rust=0.55, sediment=0.55, streaks=0.35, seed=21, sed_col=(0.5, 0.48, 0.42))
growth(hull_m, 22, 0.55)
hull_m.c[inside] = np.array(INNER) * (0.8 + 0.5 * fbm(hull_m.v[inside], 0.2, 23, 2))[:, None]

# centre the wreck: origin under the middle of the hull, on the seabed
ZC = 0.5 * (hull_m.v[:, 2].min() + hull_m.v[:, 2].max())
XC = 0.5 * (hull_m.v[:, 0].min() + hull_m.v[:, 0].max())
hull_m.T(-XC, 0, -ZC)
PROP = PROP - np.array([XC, 0, ZC])


def wz(d):            # distance from the bow -> z in the final (centred) wreck frame, roughly
    return zd(d) - ZC


# ── Height field of the wreck (to drape nets and heap mud) ──────────────────
CELL = 1.0
gx0, gz0 = -50.0, -135.0
NXg, NZg = 100, 270
HF_ = np.full((NZg, NXg), -10.0)
ix = np.clip(((hull_m.v[:, 0] - gx0) / CELL).astype(int), 0, NXg - 1)
iz = np.clip(((hull_m.v[:, 2] - gz0) / CELL).astype(int), 0, NZg - 1)
np.maximum.at(HF_, (iz, ix), hull_m.v[:, 1])
for _ in range(2):                                         # close small gaps between vertices
    S = HF_.copy()
    S[1:, :] = np.maximum(S[1:, :], HF_[:-1, :]); S[:-1, :] = np.maximum(S[:-1, :], HF_[1:, :])
    S[:, 1:] = np.maximum(S[:, 1:], HF_[:, :-1]); S[:, :-1] = np.maximum(S[:, :-1], HF_[:, 1:])
    HF_ = S


def top_at(X, Z):
    i = np.clip(((np.asarray(Z) - gz0) / CELL).astype(int), 0, NZg - 1)
    j = np.clip(((np.asarray(X) - gx0) / CELL).astype(int), 0, NXg - 1)
    return HF_[i, j]


# ── Fishing nets: draped over the bow and the hull, some held up like curtains by their floats ──
NET = (0.3, 0.33, 0.28)
nets = []


def drape(cx, cz, w, l, seed, lift=0.35):
    def h(X, Z):
        t = top_at(X, Z)
        sag = 0.4 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.3, seed).reshape(X.shape)
        return np.maximum(t + lift - sag, 0.08)
    g = grid(cx - w / 2, cx + w / 2, cz - l / 2, cz + l / 2, max(4, int(w / 0.9)), max(4, int(l / 0.9)), h, NET)
    c = g.v[g.f].mean(axis=1)
    holes = fbm(c, 0.35, seed + 3, 3)
    edge = np.maximum(np.abs(c[:, 0] - cx) / (w / 2), np.abs(c[:, 2] - cz) / (l / 2))
    g.f = g.f[(holes < 0.6) & (edge + 0.35 * (fbm(c, 0.2, seed + 4, 2) - 0.5) < 0.92)]
    g.paint(lambda p: np.array(NET) * (0.8 + 0.4 * fbm(p, 0.4, seed + 6, 2))[:, None])
    return g


def curtain(points, height, seed):
    """A net standing up from the wreck, held by its float line."""
    pts = np.asarray(points, float)
    seg_len = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.arange(0, seg_len[-1], 0.9)
    P = np.stack([np.interp(s, seg_len, pts[:, k]) for k in range(3)], 1)
    base = top_at(P[:, 0], P[:, 2])
    base = np.where(base < 0, 0.05, base)
    nr = 7
    v, f = [], []
    hts = height * (0.6 + 0.6 * fbm(P, 0.15, seed, 2))
    for i, p in enumerate(P):
        for j in range(nr):
            u = j / (nr - 1)
            v.append((p[0] + 0.6 * math.sin(u * 3 + i * 0.4), base[i] + hts[i] * u, p[2]))
    for i in range(len(P) - 1):
        for j in range(nr - 1):
            a = i * nr + j
            f += [(a, a + nr, a + 1), (a + 1, a + nr, a + nr + 1)]
    m = Mesh(v, f, NET)
    c = m.v[m.f].mean(axis=1)
    m.f = m.f[fbm(c, 0.4, seed + 2, 3) < 0.62]
    tops = [(p[0] + 0.6 * math.sin(3 + i * 0.4), base[i] + hts[i], p[2]) for i, p in enumerate(P)]
    out = [m, tube(tops, 0.05, (0.25, 0.22, 0.18), seg=4)]
    for x, y, z in tops[::2]:
        out.append(cyl(0.18, 0.18, 0.3, (0.62, 0.38, 0.18) if rng.random() < 0.5 else (0.7, 0.68, 0.6), seg=6).T(x, y - 0.15, z))
    return merge(out)


nets.append(drape(-2, wz(14), 22, 26, 501))               # the forecastle, draped in nets
nets.append(drape(0, wz(46), 24, 16, 502))
nets.append(drape(-4, wz(95), 26, 14, 503))
nets.append(drape(-2, wz(SPLIT), 26, 12, 504))
nets.append(drape(2, wz(190), 24, 18, 505))
for i, (d0_, d1_, x0, x1, h_) in enumerate(((8, 30, -1, 3, 6.0), (52, 74, 2, -2, 5.0), (104, 120, -5, 2, 7.0), (160, 172, 1, -3, 4.5))):
    nets.append(curtain([(x0, 0, wz(d0_)), (x1, 0, wz(d1_))], h_, 600 + i))
nets_m = merge(nets)
weather(nets_m, rust=0.0, sediment=0.35, streaks=0.0, seed=51, sed_col=(0.5, 0.48, 0.42))

# ── Sediment heaped against the hull where it lies in the mud ──────────────
DIST = np.where(HF_ > -5, 0.0, 99.0)                      # distance (m) from the hull outline, by dilation
_occ = HF_ > -5
for r in range(1, 13):
    o = _occ.copy()
    o[1:, :] |= _occ[:-1, :]; o[:-1, :] |= _occ[1:, :]; o[:, 1:] |= _occ[:, :-1]; o[:, :-1] |= _occ[:, 1:]
    DIST[o & ~_occ] = r * CELL
    _occ = o


def mud_h(X, Z):
    i = np.clip(((Z - gz0) / CELL).astype(int), 0, NZg - 1)
    j = np.clip(((X - gx0) / CELL).astype(int), 0, NXg - 1)
    dist = DIST[i, j] + 0.6 * (fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.15, 8).reshape(X.shape) - 0.5) * 4
    side = np.where(X < 0, 2.6, 1.6)                        # more mud on the starboard (deck) side
    near = np.exp(-np.maximum(dist, 0) / 3.2)
    bumps = 0.3 * fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.25, 7).reshape(X.shape)
    clear = 1 - 0.9 * np.exp(-((X - PROP[0]) ** 2 + (Z - PROP[2]) ** 2) / 30.0)   # divers see the last propeller
    return near * (side + bumps) * clear - 0.5


mud = grid(gx0 + 5, gx0 + NXg * CELL - 5, gz0 + 5, gz0 + NZg * CELL - 5, 60, 150, mud_h, (0.5, 0.47, 0.4))
mud.paint(lambda p: (0.5, 0.47, 0.4) + (fbm(p, 0.12, 9)[:, None] - 0.5) * 0.06)
mud.f = mud.f[mud.v[mud.f][:, :, 1].max(axis=1) > 0.15]

# ── Debris field: on the starboard (deck) side, around the split and the stern ──
deb = []
xmin_hull = hull_m.v[:, 0].min()


def spot(where):
    if where == "side":
        return xmin_hull - abs(rng.normal(6, 9)), rng.uniform(wz(225), wz(60))
    if where == "split":
        return rng.normal(-6, 12), rng.normal(wz(SPLIT), 10)
    return rng.normal(-10, 12), rng.normal(wz(228), 14)


# the docking bridge, fallen on the starboard seabed near the stern
db = merge([box(B * 0.62, 0.4, 2.0, WHITE), tube([(-3, 0, 0), (-3, -3.2, 0.3)], 0.14, WHITE, seg=5), tube([(3, 0, 0), (3, -3.0, -0.4)], 0.14, WHITE, seg=5)])
crumple(db, 0.4, 0.3, 801)
deb.append(db.R(0.25, 0.4, 0.15).T(xmin_hull - 6, 1.2, wz(226)))
for i in range(14):                                               # davits, fallen along the deck side
    x, z = xmin_hull - rng.uniform(1, 7), rng.uniform(wz(185), wz(60))
    dv = tube([(0, 0, 0), (0, 3.0, 0), (0.8, 3.8, 0), (1.5, 3.4, 0)], 0.16, WHITE, seg=6)
    deb.append(dv.R(math.pi / 2 * rng.uniform(0.8, 1.0), rng.uniform(0, 6.28), 0).T(x, 0.2, z))
for i in range(3):                                                # pieces of funnel, Cunard red under the rust
    x, z = spot("side")
    fp = cyl(3.5, 3.5, rng.uniform(3, 5), FRED, seg=24, sx=0.74, caps=False)
    keep_ = np.arctan2(fp.v[:, 2], fp.v[:, 0]) > rng.uniform(0.5, 2.0)
    fp.f = fp.f[keep_[fp.f].all(axis=1)]
    crumple(fp, 0.6, 0.25, 820 + i)
    deb.append(fp.R(math.pi / 2, rng.uniform(0, 6.28), 0).T(x, 0.5, z))
for i in range(4):                                                # ventilator cowls
    x, z = spot("side")
    deb.append(merge([cyl(0.5, 0.5, 2.0, WHITE, seg=10), box(1.2, 1.0, 1.1, WHITE).T(0, 2.0, 0.4)]).R(math.pi / 2, rng.uniform(0, 6.28), 0).T(x, 0.5, z))
for i in range(110):                                              # plates of the superstructure and the hull, bent
    x, z = spot(rng.choice(["side", "side", "split", "stern"]))
    m = box(rng.uniform(2, 8), 0.16, rng.uniform(1.5, 4.5), WHITE if rng.random() < 0.5 else BLACK)
    crumple(m, 0.6, 0.3, 900 + i)
    deb.append(m.R(rng.uniform(-0.35, 0.35), rng.uniform(0, 6.28), rng.uniform(-0.35, 0.35)).T(x, 0.1, z))
for i in range(30):                                               # beams and frames
    x, z = spot(rng.choice(["side", "split", "stern"]))
    deb.append(box(0.4, 0.55, rng.uniform(4, 12), (0.2, 0.15, 0.1)).R(0, rng.uniform(0, 6.28), rng.uniform(-0.2, 0.2)).T(x, 0, z))
for i in range(160):                                              # smaller things
    x, z = spot(rng.choice(["side", "split", "stern"]))
    s = rng.uniform(0.3, 1.3)
    deb.append(box(s, s * rng.uniform(0.4, 1.0), s * rng.uniform(0.6, 1.6), (0.25, 0.2, 0.15)).R(0, rng.uniform(0, 6.28), 0).T(x, -0.1, z))
for i in range(2):                                                # boilers that rolled out at the split
    deb.append(boiler().R(0, rng.uniform(0, 6.28), 0).T(rng.normal(-14, 4), 1.6, wz(SPLIT) + rng.normal(0, 6)))
debris = merge(deb)
weather(debris, rust=0.75, sediment=0.65, streaks=0.3, seed=61, sed_col=(0.5, 0.48, 0.42))
growth(debris, 62, 0.45)

# ── The ship as she sailed (model_built.glb): whole, clean, keel on y = 0, centred ──
ship = merge(ship_part(0, L, built=True) + shafts_and_props() + [rudder()])
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
ship.v[:, 1] -= ship.v[:, 1].min()

# ── Write ────────────────────────────────────────────────────────────────────
out = ROOT / "models" / "item" / "Q187425" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
tris, size = write_glb(out, [("hull", hull_m), ("nets", nets_m), ("mud", mud), ("debris_field", debris)])
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB  (last propeller at {PROP.round(1)})")
for n, m in (("hull", hull_m), ("nets", nets_m), ("mud", mud), ("debris", debris)):
    print(f"  {n}: {m.tris:,} tris, x {m.v[:, 0].min():.0f}..{m.v[:, 0].max():.0f}, y {m.v[:, 1].min():.1f}..{m.v[:, 1].max():.1f}, z {m.v[:, 2].min():.0f}..{m.v[:, 2].max():.0f}")
for d in (5, 20, 40, 70, 100, 125, 150, 180, 210, 232):
    sel = np.abs(hull_m.v[:, 2] - wz(d)) < 2
    if sel.any():
        print(f"    d={d:3d} m: height {hull_m.v[sel, 1].max():.1f} m, width {np.ptp(hull_m.v[sel, 0]):.1f} m")
tris2, size2 = write_glb(out.with_name("model_built.glb"), [("main", ship)])
print(f"model_built.glb: {tris2:,} triangles, {size2 / 2**20:.2f} MB, y {ship.v[:, 1].min():.1f}..{ship.v[:, 1].max():.1f}, z {ship.v[:, 2].min():.1f}..{ship.v[:, 2].max():.1f}")
