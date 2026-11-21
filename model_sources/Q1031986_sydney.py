"""
HMAS Sydney (Q1031986) — the wreck of the Australian light cruiser off Western Australia, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a Modified Leander-class light cruiser of 1935 (ex HMS Phaeton): length 171.4 m, beam 17.3 m; four
twin 6-inch turrets (A and B forward, X and Y aft), a bridge tower with the director control tower on top, a
tripod foremast, two widely spaced funnels with the aircraft catapult (and its Walrus) between them, four twin
4-inch guns about the after funnel, two quadruple torpedo-tube mounts, four screws.

The wreck, after the Finding Sydney Foundation's final report (2008): lost with all 645 aboard after her fight
with the raider Kormoran on 19 November 1941, she lies at about 2,470 m, 11.4 nautical miles south-east of the
Kormoran. The hull is upright, oriented north-west/south-east with the stern to the north, about 166 m long and
14 m high, its lower hull exposed rather than buried; the stern is buckled. The bow broke off forward of
A turret (the torpedo hit) and lies upside down some 450 m to the north-north-west, about 28 m long and 10.5 m
high. Between them, the debris field: a 27-ft whaler, a boat lying across a 36-ft motor pinnace, a berthing drum
and wire, one of the quadruple torpedo-tube mounts, the aircraft catapult with a boat, the director control
tower and the bridge roof, and beyond the bow the foremast. A turret is wrecked and B turret has lost its roof;
X and Y were left trained to port; shell holes run along the starboard side and through the bridge. The funnels
are not described: here they are collapsed stumps. Headings of the pieces, the shape of each piece of debris and
the damage in detail are approximate. She is a war grave.

    python model_sources/Q1031986_sydney.py   -> models/item/Q1031986/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, sstep, fbm, yaw_quat  # noqa: E402

rng = np.random.default_rng(1941)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B, D = 171.4, 17.3, 9.6                 # length, beam, depth to the upper deck
FC = 2.5                                   # the long forecastle deck above the upper deck
TW, TT = 5.0, 1.0
WL = 5.0                                   # top of the red bottom paint
HULLC = (0.46, 0.48, 0.5)                  # wartime grey
GREY, GREY_L, GREY_D = (0.46, 0.48, 0.5), (0.52, 0.54, 0.56), (0.32, 0.33, 0.35)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.3, 0.3, 0.31), (0.5, 0.42, 0.3)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06)
HHW = 3.0
HATCHES = []
BULKHEADS = [28.0]
FCL = 102.0                                # the forecastle deck ends here; quarterdeck aft of it


def zd(d):
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):                                # a fast cruiser: long fine entrance, broad transom stern
    w = 1.0
    if t > 0.7:
        w = math.sin(((1 - t) / 0.3) * math.pi / 2) ** 0.75
    if t < 0.14:
        w = min(w, 0.62 + 0.38 * math.sin((t / 0.14) * math.pi / 2))
    return max(w, 0.004)


def yb_of(t):                               # shallow cut-up aft for the four shafts, cut-away forefoot
    y = 0.0
    if t < 0.08:
        y = D * 0.32 * ((0.08 - t) / 0.08)
    if t > 0.965:
        y = max(y, D * 0.75 * ((t - 0.965) / 0.035) ** 2)
    return y


def yd_of(t):
    s = (t - 0.5) * 2
    return D + (2.4 if s > 0 else 0.3) * s * s


def k_of(t):
    return 2.2 + 5.5 * w_of(t) ** 4


def hw(z, y):
    t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    if y >= yd:
        return B / 2 * w
    u = max(0.0, (y - yb) / max(0.1, yd - yb))
    return B / 2 * w * (1 - (1 - u) ** k)


def deck_y(d):
    return yd_of(t_of(zd(d)))


def half(d):
    return B / 2 * w_of(t_of(zd(d)))


def at(m, d, x=0.0, y=0.0):
    return m.T(x, y, zd(d))


def stations(d0, d1, step, extra=()):
    ds = set(np.arange(d0, d1, step).round(3).tolist()) | {d0, d1}
    ds |= {e for e in extra if d0 < e < d1}
    return sorted(ds)


def hull(d0, d1, NP=24):
    """The shell plating and bottom between d0 and d1 (open at both ends), up to the main deck."""
    ds = stations(d0, d1, 1.3, [b for b in BULKHEADS] + [h for hh in HATCHES for h in hh])[::-1]
    W = 2 * NP + 1
    v, f = [], []
    for d in ds:
        z = zd(d); t = t_of(z)
        yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
        rake = 3.0 * max(0.0, (t - 0.93) / 0.07)
        for j in range(W):
            side = -1 if j < NP else 1
            u = abs(NP - j) / NP
            y = yb + (yd - yb) * u
            v.append((side * B / 2 * w * (1 - (1 - u) ** k), y, z + rake * (y / yd) ** 1.5))
    for i in range(len(ds) - 1):
        for j in range(W - 1):
            a = i * W + j; b = a + W; c = a + 1; dd = b + 1
            f += [(a, c, b), (b, c, dd)]
    if d1 >= L - 0.01:                                       # the small flat of the cruiser stern
        c0 = len(v); z = zd(L); t = t_of(z)
        v.append((0, (yb_of(t) + yd_of(t)) / 2, z))
        f += [(c0, j + 1, j) for j in range(W - 1)]
    m = Mesh(v, f)
    m.paint(lambda p: np.where((p[:, 1] < WL)[:, None], RED, HULLC))
    return m


def strip(d0, d1, yfn, wfn, holes, col, step=1.5):
    """A deck between d0 and d1 at height yfn(d), half width wfn(d), with openings [(da, db, half width)]."""
    ds = stations(d0, d1, step, [e for h in holes for e in h[:2]])
    v, f = [], []
    for a, b in zip(ds[:-1], ds[1:]):
        m_ = (a + b) / 2
        hole = next((h for h in holes if h[0] <= m_ <= h[1]), None)
        w0, w1, y0, y1, z0, z1 = wfn(a), wfn(b), yfn(a), yfn(b), zd(a), zd(b)
        spans = [(-w0, w0, -w1, w1)] if not hole else [(-w0, -hole[2], -w1, -hole[2]), (hole[2], w0, hole[2], w1)]
        for xa0, xb0, xa1, xb1 in spans:
            o = len(v)
            v += [(xa0, y0, z0), (xb0, y0, z0), (xb1, y1, z1), (xa1, y1, z1)]
            f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    return Mesh(v, f, col)


def side_strips(d0, d1, ybot, ytop, col, step=1.5, inset=0.0):
    """Vertical plating on both sides (raised decks, bulwarks) following the hull's half width at the deck."""
    ds = stations(d0, d1, step)
    out = []
    for s in (-1, 1):
        v, f = [], []
        for d in ds:
            x = s * (half(d) - inset)
            v += [(x, ybot(d), zd(d)), (x, ytop(d), zd(d))]
        for i in range(len(ds) - 1):
            a = 2 * i
            f += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)] if s > 0 else [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
        out.append(Mesh(v, f, col))
    return merge(out)


def section(d, ytop, col, NP=16, xmax=None):
    """A transverse wall filling the hull's cross-section at d up to ytop (a bulkhead, the end of a deck)."""
    z = zd(d); t = t_of(z)
    yb, yd, w, k = yb_of(t), yd_of(t), w_of(t), k_of(t)
    pts = []
    for j in range(2 * NP + 1):
        side = -1 if j < NP else 1
        u = abs(NP - j) / NP
        pts.append((side * B / 2 * w * (1 - (1 - u) ** k), min(yb + (yd - yb) * u, ytop), z))
    v = pts + [(0, (yb + ytop) / 2, z)]
    c = len(pts)
    f = [(c, i, i + 1) for i in range(len(pts) - 1)] + [(c, len(pts) - 1, 0)]
    return Mesh(v, f, col)


def face(d, y0, y1, col, doors=(), sign=1):
    """The front or back wall of a raised deck, across the hull at d, with doors (dark quads)."""
    w = half(d)
    m = [Mesh([(-w, y0, zd(d)), (w, y0, zd(d)), (w, y1, zd(d)), (-w, y1, zd(d))], [(0, 1, 2), (0, 2, 3)] if sign > 0 else [(0, 2, 1), (0, 3, 2)], col)]
    for x in doors:
        m.append(quad(x, y0 + 0.95, zd(d), 0.8, 1.9, "+z" if sign > 0 else "-z", DARK))
    return m


def portholes(d0, d1, y, every, xfn, r=0.22):
    out = []
    for d in np.arange(d0, d1, every):
        for s in (-1, 1):
            out.append(quad(s * xfn(d), y, zd(d), r * 2, r * 2, "+x" if s > 0 else "-x", DARK))
    return out


# ── Fittings ─────────────────────────────────────────────────────────────────


def anchor():
    """A stockless anchor hanging in its hawse pipe (shank up), seen from the side."""
    parts = [box(0.32, 2.6, 0.32, BLACK).T(0, -1.3, 0), box(0.5, 0.5, 1.6, BLACK).T(0, -1.55, 0)]
    for s in (-1, 1):
        parts.append(box(0.3, 1.3, 0.5, BLACK).R(s * 0.5, 0, 0).T(0, -1.2, s * 0.75))
    return merge(parts)


def windlass(d):
    y = deck_y(d) + FC
    return [at(box(2.4, 1.0, 1.4, STEEL), d, y=y)] + [at(cyl(0.55, 0.55, 0.5, STEEL, seg=12).R(0, 0, math.pi / 2).T(s * 1.55, 0.8, 0), d, y=y) for s in (-1, 1)]


def chains(d_w, built):
    out = []
    y0 = deck_y(d_w) + FC + 0.9
    for s in (-1, 1):
        pts = [(s * 1.6, y0, zd(d_w)), (s * 2.6, y0 - 0.5, zd(d_w - 1.5)), (s * (half(2.4) - 0.9), deck_y(2.4) + FC + 0.1, zd(2.4))]
        out.append(tube(pts, 0.13, BLACK, seg=5))
    return out


def bollards(ds, yoff, inset=1.3):
    out = []
    for d in ds:
        for s in (-1, 1):
            x = s * (half(d) - inset)
            out += [at(cyl(0.25, 0.25, 0.7, BLACK, seg=8), d, x=x, y=deck_y(d) + yoff), at(cyl(0.25, 0.25, 0.7, BLACK, seg=8), d + 0.8, x=x, y=deck_y(d) + yoff)]
    return out


def ventilator(h=2.2, r=0.38, col=GREY):
    cowl = lathe([(r, 0), (r * 1.4, 0.35), (r * 1.6, 0.7), (r * 1.2, 0.95)], col, seg=10).R(-1.2, 0, 0).T(0, h, 0)
    return merge([cyl(r, r, h, col, seg=10, caps=False), cowl])


def hatch_coaming(d0, d1, built, wreck_beams=0):
    y = deck_y((d0 + d1) / 2)
    ln, cz = d1 - d0, (d0 + d1) / 2
    P = [at(box(0.25, 0.9, ln + 0.25, GREY_D), cz, x=s * (HHW + 0.12), y=y) for s in (-1, 1)]
    P += [at(box(2 * HHW + 0.5, 0.9, 0.25, GREY_D), d, y=y) for d in (d0, d1)]
    if built:                                                # hatch boards under a tarpaulin, battened down
        cover = box(2 * HHW, 0.35, ln, TARP)
        cover.v[:, 1] += 0.18 * (1 - (cover.v[:, 0] / HHW) ** 2) * (cover.v[:, 1] > 0.1)
        P.append(at(cover, cz, y=y + 0.75))
    else:                                                    # boards gone; some of the steel hatch beams still across
        for k in range(wreck_beams):
            d = d0 + ln * (k + 1) / (wreck_beams + 1)
            P.append(at(box(2 * HHW + 0.3, 0.45, 0.22, STEEL), d, y=y + 0.45))
    return P


def mast(d, h, built):
    y = deck_y(d)
    P = [at(box(4.0, 2.2, 3.0, GREY_L), d, y=y)]                                # mast house (winches inside)
    for s in (-1, 1):
        P.append(at(cyl(0.45, 0.45, 1.0, STEEL, seg=10).R(0, 0, math.pi / 2).T(s * 1.2, 0.6, 0), d - 2.2, y=y))   # winches on deck
        P.append(at(cyl(0.45, 0.45, 1.0, STEEL, seg=10).R(0, 0, math.pi / 2).T(s * 1.2, 0.6, 0), d + 2.2, y=y))
    P.append(at(cyl(0.38, 0.22, h, GREY, seg=10), d, y=y + 2.2))
    if built:
        P.append(at(box(3.4, 0.25, 0.4, GREY), d, y=y + 2.2 + h * 0.62))                     # crosstree
        for dd in (-1, 1):                                                                   # derricks, two each way, topped down
            for s in (-1, 1):
                P.append(tube([(s * 0.8, y + 3.0, zd(d)), (s * 1.6, y + 7.5, zd(d + dd * 10.5))], 0.17, GREY, seg=6))
        for s in (-1, 1):                                                                    # shrouds
            P.append(tube([(0, y + 2.2 + h * 0.6, zd(d)), (s * (half(d) - 0.3), y + 0.9, zd(d + 1.2))], 0.03, BLACK, seg=4))
            P.append(tube([(0, y + 2.2 + h * 0.6, zd(d)), (s * (half(d) - 0.3), y + 0.9, zd(d - 1.2))], 0.03, BLACK, seg=4))
    return P


def lifeboat():
    m = lathe([(0.05, -3.6), (0.9, -3.0), (1.2, -1.5), (1.25, 0), (1.2, 1.5), (0.9, 3.0), (0.05, 3.6)], (0.75, 0.74, 0.7), seg=10, axis="z")
    m.v[:, 1] = np.minimum(m.v[:, 1], 0.35)
    return m


def davit(x, y, z, s, bent=0.0):
    pts = [(x, y - 1.0, z), (x, y + 2.4, z), (x + s * (0.9 + bent), y + 3.0 - bent * 1.5, z)]     # its foot set into the deck
    return tube(pts, 0.12, GREY, seg=5)


def propeller(d, y, r=1.9):
    P = [at(cyl(0.5, 0.4, 1.0, BRONZE, seg=12).R(math.pi / 2, 0, 0), d, y=y)]
    for k in range(4):
        a = 2 * math.pi * k / 4
        P.append(at(box(r, 0.16, 1.3, BRONZE, bottom=False).T(r / 2 + 0.3, 0, 0).R(0.45, 0, 0).R(0, 0, a), d - 0.4, y=y))
    P.append(at(cyl(0.3, 0.3, 10.0, STEEL, seg=8).R(math.pi / 2, 0, 0), d - 10.0, y=y))   # shaft, long enough to run into the hull
    return P


def rudder(d, y0):
    return [at(box(0.55, 5.4, 2.6, RED), d, y=y0), at(cyl(0.22, 0.22, 2.0, STEEL, seg=8), d - 1.0, y=y0 + 5.0)]


# ── Cargo ────────────────────────────────────────────────────────────────────


def wheel(r, w, col=RUBBER, seg=8):
    return cyl(r, r, w, col, seg=seg).R(0, 0, math.pi / 2).T(w / 2, 0, 0)


def crate(w, h, d, col=(0.36, 0.3, 0.2)):
    return box(w, h, d, col)


def tyre(r=0.48, t=0.13, seg=10):
    prof = [(r + t * math.cos(a), t * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 7)]
    return lathe(prof, RUBBER, seg=seg)


def place(m, d, x, y, yaw=0.0, tilt=(0.0, 0.0)):
    return m.R(tilt[0], yaw, tilt[1]).T(x, y, zd(d))


# ── Armament and fittings ────────────────────────────────────────────────────
def turret(yb, train, state="ok", raised=0.0):
    """A twin 6-inch turret on its barbette (origin at its centre, on the deck), guns towards +Z before training."""
    P = [cyl(3.3, 3.3, 1.2 + raised, GREY_D, seg=20)]
    y0 = 1.2 + raised
    house = box(6.2, 2.7, 7.4, GREY).T(0, y0, 0.5)
    front = (house.v[:, 2] > 3.0) & (house.v[:, 1] > y0 + 1.5)
    house.v[front, 2] -= 1.1                                                # sloped face
    if state == "roofless":                                                 # the roof blown off
        top = house.v[house.f][:, :, 1].min(1) > y0 + 2.6
        house.f = house.f[~top]
        crumple(house, 0.35, 0.8, 31, sstep(y0 + 1.6, y0 + 2.7, house.v[:, 1]))
    P.append(house)
    for x in (-0.95, 0.95):
        P.append(cyl(0.17, 0.15, 7.6, STEEL, seg=8).R(math.pi / 2, 0, 0).T(x, y0 + 1.3, 3.6))
    m = merge(P)
    if state == "wrecked":
        crumple(m, 0.7, 0.35, 32)
        m.v[:, 1] -= 0.6 * sstep(1.5, 4.0, m.v[:, 1])
    return m.R(0, train, 0).T(0, yb, 0)


def twin_4in(train=0.0):
    P = [cyl(0.9, 1.0, 0.8, GREY_D, seg=12), box(2.4, 1.6, 1.4, GREY).T(0, 0.8, 0.2)]
    for x in (-0.35, 0.35):
        P.append(cyl(0.08, 0.07, 4.2, STEEL, seg=6).R(math.pi / 2 - 0.25, 0, 0).T(x, 1.6, 0.6))
    return merge(P).R(0, train, 0)


def tt_mount(loaded=4):
    """A quadruple 21-inch torpedo-tube mount."""
    P = [cyl(1.1, 1.2, 0.7, GREY_D, seg=12)]
    for k in range(4):
        x = (k - 1.5) * 0.72
        P.append(cyl(0.33, 0.33, 7.2, GREY, seg=10).R(math.pi / 2, 0, 0).T(x, 1.1, -3.0))
        if k < loaded:
            P.append(cyl(0.27, 0.27, 0.05, (0.55, 0.5, 0.3), seg=10).R(math.pi / 2, 0, 0).T(x, 1.1, 4.21))
    return merge(P)


def funnel(h, built):
    f = cyl(2.4, 2.3, h, GREY, seg=20, sx=0.62).R(-0.06, 0, 0)
    if built:
        f.c[f.v[:, 1] > h - 1.0] = BLACK
    return f


def tripod(h):
    P = [cyl(0.42, 0.3, h, GREY, seg=8, caps=False)]
    for s in (-1, 1):
        P.append(tube([(s * 3.2, 0, -2.6), (0, h * 0.62, 0)], 0.25, GREY, seg=6))
    P += [cyl(0.9, 0.9, 1.2, GREY, seg=10).T(0, h * 0.72, 0), box(2.4, 0.2, 0.3, GREY).T(0, h * 0.8, 0),
          cyl(0.12, 0.08, 7.0, GREY, seg=6).T(0, h, 0)]
    return merge(P)


def catapult(walrus):
    P = [cyl(1.4, 1.6, 1.0, GREY_D, seg=14)]
    truss = box(1.4, 0.9, 16.0, GREY).T(0, 1.0, 0)
    P.append(truss)
    for k in range(9):
        P.append(tube([(-0.7, 1.0, -8 + 2 * k), (0.7, 1.9, -7 + 2 * k)], 0.06, GREY, seg=4))
    if walrus:
        P.append(walrus_plane().T(0, 2.0, 0))
    return merge(P)


def walrus_plane():
    """The Supermarine Walrus amphibian: a boat hull, biplane wings, a pusher engine."""
    hull_ = lathe([(0.05, -5.8), (0.5, -4.0), (0.75, -1.5), (0.8, 0.5), (0.75, 2.0), (0.4, 3.6), (0.05, 4.2)], (0.6, 0.62, 0.6), seg=8, axis="z")
    hull_.S(1, 1.1, 1)
    P = [hull_.T(0, 0.8, 0), box(14.0, 0.15, 1.8, (0.6, 0.62, 0.6)).T(0, 1.9, 0.6), box(13.0, 0.15, 1.7, (0.6, 0.62, 0.6)).T(0, 3.6, 0.5),
         cyl(0.45, 0.4, 1.4, (0.3, 0.3, 0.3), seg=8).R(math.pi / 2, 0, 0).T(0, 2.9, -0.6), box(3.8, 0.1, 0.9, (0.6, 0.62, 0.6)).T(0, 1.6, -5.2)]
    for x in (-5.5, 5.5):
        P.append(box(0.4, 0.4, 1.4, (0.6, 0.62, 0.6)).T(x, 1.2, 0.6))
        P.append(tube([(x * 0.6, 2.0, 0.6), (x * 0.6, 3.6, 0.6)], 0.05, (0.4, 0.4, 0.4), seg=4))
    return merge(P)


def boat(ln, cabin=False, col=(0.78, 0.77, 0.73)):
    m = lathe([(0.05, -ln / 2), (ln * 0.11, -ln * 0.42), (ln * 0.15, -ln * 0.2), (ln * 0.155, 0), (ln * 0.15, ln * 0.2), (ln * 0.11, ln * 0.42), (0.05, ln / 2)], col, seg=10, axis="z")
    m.v[:, 1] = np.minimum(m.v[:, 1], ln * 0.05)
    P = [m]
    if cabin:
        P.append(box(ln * 0.22, 1.0, ln * 0.35, col).T(0, ln * 0.04, ln * 0.05))
    return merge(P)


def shell_holes(n, side, d0, d1, y0, y1, seed):
    rr = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = rr.uniform(d0, d1); y = rr.uniform(y0, y1); z = zd(d)
        s = rr.uniform(0.35, 0.8)
        out.append(quad(side * (hw(z, min(y, yd_of(t_of(z)) - 0.05)) + 0.05), y, z, s, s * rr.uniform(0.7, 1.2), "+x" if side > 0 else "-x", DARK))
    return out


# ── The ship, piece by piece ─────────────────────────────────────────────────
def deck_parts(d0, d1, built):
    P = []
    a, b = max(d0, 0), min(d1, FCL)
    if b > a:
        P.append(strip(a, b, lambda d: deck_y(d) + FC, half, [], WOOD))
        P.append(side_strips(a, b, lambda d: deck_y(d), lambda d: deck_y(d) + FC, HULLC))
        if b == FCL:
            P += face(FCL, deck_y(FCL), deck_y(FCL) + FC, GREY_L, doors=(-3.0, 3.0), sign=-1)
    a, b = max(d0, FCL), min(d1, L)
    if b > a:
        P.append(strip(a, b, deck_y, half, [], WOOD))
        P += [tube([(s * (half(d) - 0.1), deck_y(d) + 1.0, zd(d)) for d in np.arange(a, b + 0.1, 2.5)], 0.03, GREY, seg=4) for s in (-1, 1)] if built else []
    return P


def bow_part(built):
    """The bow forward of the break at 28 m: forecastle deck, cable gear, anchors, breakwater."""
    P = [hull(0, 28.0)] + deck_parts(0, 28.0, built)
    y = deck_y(10) + FC
    P += [at(cyl(0.6, 0.7, 0.8, STEEL, seg=12), d, x=x, y=y) for d, x in ((9, 1.6), (9, -1.6), (16, 0))]       # capstans
    for s in (-1, 1):
        P.append(tube([(s * 1.6, y + 0.4, zd(9)), (s * (half(3.5) - 0.8), deck_y(3.5) + FC + 0.1, zd(3.5))], 0.13, BLACK, seg=5))
        z = zd(4.5); yy = deck_y(4.5) - 0.8
        P.append(anchor().T(s * (hw(z, yy) + 0.25), yy, z))
    bw = box(B * 0.7, 1.1, 0.25, GREY).T(0, deck_y(24) + FC, zd(24))           # breakwater
    P.append(bw)
    return P


def main_part(built):
    """The hull from the break (28 m) to the stern, with everything on it."""
    P = [hull(28.0 if not built else 0.0, L)] + deck_parts(28.0 if not built else 0.0, L, built)
    if built:
        P += bow_part(True)[1:]
    yF = lambda d: deck_y(d) + FC
    # turrets: A and B forward, X and Y aft (left trained to port in the fight)
    P.append(turret(yF(38), 0.0, "ok" if built else "wrecked").T(0, 0, zd(38)))
    P.append(turret(yF(48), 0.0, "ok" if built else "roofless", raised=2.4).T(0, 0, zd(48)))
    trainX = math.pi if built else math.pi - 1.35
    P.append(turret(deck_y(126), trainX, "ok", raised=2.4).T(0, 0, zd(126)))
    P.append(turret(deck_y(137), trainX, "ok").T(0, 0, zd(137)))
    # bridge tower
    y0 = yF(57)
    tower = [box(B * 0.56, 2.6, 10.0, GREY_L).T(0, y0, zd(57.5)), box(B * 0.46, 2.6, 8.0, GREY_L).T(0, y0 + 2.6, zd(57.0))]
    tower += [quad(x, y0 + 4.0, zd(53.0) + 0.02, 0.8, 0.6, "+z", DARK) for x in np.arange(-3.2, 3.3, 1.1)]
    compass = [box(B * 0.52, 1.2, 5.0, GREY_L).T(0, y0 + 5.2, zd(55.5)), box(B * 0.9, 0.2, 2.4, GREY_L).T(0, y0 + 5.2, zd(55.0))]
    dct = [box(3.6, 2.3, 3.6, GREY).T(0, y0 + 6.4, zd(57.5)), box(5.0, 0.6, 0.8, GREY).T(0, y0 + 8.7, zd(57.5)), box(B * 0.5, 0.2, 5.4, GREY_L).T(0, y0 + 6.4, zd(55.5))]
    if built:
        P += tower + compass + dct
        P.append(tripod(20.0).T(0, y0, zd(62.0)))
    else:
        tw = merge(tower + compass); crumple(tw, 0.25, 0.4, 41)
        P.append(tw)
        P.append(quad(-B * 0.26 - 0.05, y0 + 5.8, zd(55.5), 0.15 * 2, 0.3, "-x", DARK))         # the 15 cm hole through the compass platform
        P.append(cyl(0.5, 0.5, 2.0, GREY, seg=8, caps=False).T(0, y0, zd(62.0)))               # foremast stump
    # funnels, catapult, searchlight platform, 4-inch guns, torpedo tubes, boats, crane, mainmast
    for d in (70.0, 104.0):
        if built:
            P.append(funnel(10.0, True).T(0, deck_y(d) + FC + (0 if d < FCL else 0), zd(d)))
        else:
            st = funnel(1.4, False); st.f = st.f[st.v[st.f][:, :, 1].min(1) < 1.3]
            crumple(st, 0.5, 0.6, int(d), sstep(0.3, 1.4, st.v[:, 1]))
            P.append(st.T(0, deck_y(d) + FC, zd(d)))
    P.append(at(cyl(1.6, 1.7, 1.0, GREY_D, seg=14), 88, y=yF(88)))                      # catapult turntable
    if built:
        P.append(catapult(True).T(0, yF(88), zd(88)))
        P.append(at(merge([cyl(0.5, 0.45, 6.0, GREY, seg=8), tube([(0, 6.0, 0), (0, 8.0, 9.0)], 0.25, GREY, seg=6)]), 97, x=6.0, y=yF(97)))   # crane
    plat = merge([box(6.5, 0.25, 4.0, GREY_L).T(0, 3.0, 0)] + [box(0.3, 3.0, 0.3, GREY).T(x, 0, z) for x in (-2.8, 2.8) for z in (-1.6, 1.6)]
                 + [cyl(0.6, 0.6, 0.9, GREY, seg=10).T(x, 3.25, 0) for x in (-2.2, 2.2)])
    P.append(at(plat, 77.0, y=yF(77)))                                                    # searchlight platform
    for d in (99.0, 109.0):
        for s in (-1, 1):
            P.append(at(twin_4in(0.0 if d < 104 else math.pi), d, x=s * 5.4, y=deck_y(d) + FC + 0.4))
    for s in (-1, 1):                                                                       # quadruple torpedo tubes
        if built or s > 0:
            P.append(at(tt_mount(), 112, x=s * 6.0, y=deck_y(112)).R(0, 0, 0))
    if built:
        for d in (74.0, 80.0):
            for s in (-1, 1):
                P.append(boat(8.2 if d < 77 else 10.0, cabin=d > 77).T(s * 5.9, yF(d) + 0.9, zd(d)))
    else:
        for d in (74.0, 80.0):
            for s in (-1, 1):
                P.append(at(box(0.3, 0.9, 3.0, GREY_D), d, x=s * 5.9, y=yF(d)))           # empty boat cradles
    P.append(at(box(B * 0.4, 2.4, 7.0, GREY_L), 116.0, y=deck_y(116)))                   # after deckhouse
    P.append(at(cyl(0.38, 0.22, 18.0 if built else 11.0, GREY, seg=8), 119.0, y=deck_y(119) + 2.4))   # mainmast
    # quarterdeck: bollards, kedge anchor
    P += bollards((145.0, 160.0), 0.0, inset=1.0)
    P.append(at(anchor().R(math.pi / 2, 0, 0), 150.0, x=-2.5, y=deck_y(150) + 0.2))
    # four screws and the rudder
    for x, d, y in ((-5.6, L - 22, 2.0), (5.6, L - 22, 2.0), (-2.3, L - 12, 2.2), (2.3, L - 12, 2.2)):
        P += [m.T(x, 0, 0) for m in propeller(d, y, 1.6)]
    P += rudder(L - 3.0, 0.9)
    if not built:                                                                            # shell holes on the starboard side (−X)
        P += shell_holes(26, -1, 30, 125, 6.0, 11.5, 51)
        P += shell_holes(6, 1, 40, 110, 7.0, 11.0, 52)
    return P


# ── Where things are: offsets on the seabed from the hull centre ─────────────
HEAD = 135.0                                                                 # the bow points SE (hull NW/SE, stern to the north)
M_LAT, M_LON = 30.87, 30.87 * math.cos(math.radians(26.246))                 # metres per arc-second


def loc(dn_s, de_s):
    """(north, east) offset in arc-seconds from the hull centre → (x, z) in the model (bow = +Z, starboard = −X)."""
    n, e = dn_s * M_LAT, de_s * M_LON
    h = math.radians(HEAD)
    return n * math.sin(h) - e * math.cos(h), n * math.cos(h) + e * math.sin(h)


# ── The wreck ────────────────────────────────────────────────────────────────
main = merge(main_part(False))
dfb = L / 2 - main.v[:, 2]
crumple(main, 0.15, 0.12, 11)
crumple(main, 1.4, 0.35, 12, sstep(34, 28, dfb))                            # torn at the break
w_st = sstep(150, L, dfb)
main.v[:, 1] -= w_st * 1.4 * (1 - main.v[:, 1] / 16)                        # buckled stern (she struck the bottom stern first)
crumple(main, 0.9, 0.2, 13, w_st)
main.T(0, 0, -zd((28 + L) / 2))                                              # origin under the middle of the hull
main.T(0, -0.3, 0)                                                           # lower hull exposed, barely sunk in

r = np.random.default_rng(78)


def plates(n, cx, cz, spread, seed):
    rr = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = box(rr.uniform(1.0, 5.0), 0.14, rr.uniform(0.8, 3.0), GREY if rr.random() < 0.7 else WOOD)
        crumple(p, 0.4, 0.4, seed + i)
        out.append(p.R(rr.uniform(-0.5, 0.5), rr.uniform(0, 6.28), rr.uniform(-0.5, 0.5)).T(cx + rr.normal(0, spread), 0.2, cz + rr.normal(0, spread)))
    return out


def lump(w, h, d, seed):
    """A large piece of torn structure."""
    m = merge([box(w, h, d, GREY), box(w * 0.6, h * 0.5, d * 0.4, GREY_L).T(w * 0.15, h, 0)])
    crumple(m, 0.6, 0.3, seed)
    return m


Z0 = 0.0
near = []
x, z = loc(1, -6); near.append(boat(8.2).R(0.25, 1.2, 0.1).T(x, 0.4, z))                                    # S1 27-ft whaler
for k, (dn, de) in enumerate(((2, -1), (3, -1), (3, 1))):                                                    # S2-S4 large pieces
    x, z = loc(dn, de); near.append(lump(r.uniform(5, 9), r.uniform(1.5, 3.5), r.uniform(4, 8), 60 + k).R(0, r.uniform(0, 6.28), 0).T(x, 0, z))
x, z = loc(10, 1)
near.append(merge([cyl(1.0, 1.0, 2.2, STEEL, seg=14).R(0, 0, math.pi / 2).T(1.1, 1.0, 0)] +
                  [tube([(math.cos(a) * 2.5, 0.15, math.sin(a) * 2.5) for a in np.linspace(0, 6.0, 14)], 0.06, BLACK, seg=4)]).T(x, 0, z))   # S5 berthing drum and wire
x, z = loc(8, -1); near.append(tt_mount(2).R(0.15, 0.7, 0.2).T(x, 0.1, z))                                  # S6 the torpedo tubes, two loaded
x, z = loc(6, -7); near += [boat(11.0, cabin=True).R(0, 0.4, 0.15).T(x, 0.2, z), boat(8.0).R(0.3, 1.5, 0.2).T(x + 1.0, 1.1, z + 0.5)]   # S7
for (dn, de) in ((1, -3), (4, -2), (6, -4), (8, -3)):
    x, z = loc(dn, de); near += plates(8, x, z, 7, int(dn * 10 + de))
debris_near = merge(near)

# the bow and the far end of the debris field: a piece of its own about 470 m NNW, modelled about the bow
BOW_N, BOW_E = 14, -7                                                         # S11, arc-seconds from the hull centre
bx, bz = loc(BOW_N, BOW_E)
bow = merge(bow_part(False))
crumple(bow, 0.15, 0.12, 21)
crumple(bow, 1.3, 0.35, 22, sstep(22, 28, L / 2 - bow.v[:, 2]))
bow.T(0, 0, -zd(14.0)).R(0, 0, math.pi).R(0, 0.6, 0)                          # upside down
bow.T(0, 10.5 - bow.v[:, 1].max(), 0)                                       # 10.5 m high on the sonar: the forecastle deck pressed into the ooze
far = []


def at_bow(dn, de):
    x, z = loc(dn, de)
    return x - bx, z - bz


x, z = at_bow(12, -5)
far += [catapult(False).R(0.08, 0.9, 0.12).T(x, 0.3, z), boat(8.2).R(0.2, 0.3, 0.6).T(x + 6, 0.5, z + 3)]   # S9 catapult and a boat
x, z = at_bow(12, -6)
dctw = merge([box(3.6, 2.3, 3.6, GREY), box(5.0, 0.6, 0.8, GREY).T(0, 2.3, 0)]); crumple(dctw, 0.3, 0.5, 71)
roof = box(B * 0.5, 0.2, 5.4, GREY_L); crumple(roof, 0.4, 0.4, 72)
far += [dctw.R(0.4, 1.0, 0.3).T(x, 0.6, z), roof.R(0.2, 0.5, 0.15).T(x + 4, 0.4, z - 3)]                     # S10 director tower and bridge roof
x, z = at_bow(17, -10)
fm = tripod(18.0); crumple(fm, 0.2, 0.4, 73)
far.append(fm.R(0, 0, math.pi / 2 - 0.1).R(0, 2.2, 0).T(x, 0.9, z))                                         # S12 the foremast
for (dn, de) in ((13, -6), (15, -8), (11, -4), (16, -9)):
    x, z = at_bow(dn, de); far += plates(9, x, z, 9, int(dn * 100 + de))
debris_far = merge(far)

# ── The ship as built ────────────────────────────────────────────────────────
ship = merge(main_part(True))
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
ship.T(0, 0, -zd(L / 2))

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1031986" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
bn, be = BOW_N * M_LAT, BOW_E * M_LON
sites = [{"steel": [("hull_section", main), ("debris_field", debris_near, "loose")], "mud": [], "plain": []},
         {"steel": [("bow", bow), ("bow_debris", debris_far, "loose")], "mud": [], "plain": [],
          "piece": ("bow_section", {"translation": [bx, 0, bz]})}]          # in the hull's frame here; Offlined places it from meta "pieces"
dress_wreck(out, sites, depth=-2468, kind="warship", seed=1941, memorial=True, drift=0.35)
print(f"  bow {bn:.0f} m N, {be:.0f} m E")
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
