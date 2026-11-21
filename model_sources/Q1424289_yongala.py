"""
SS Yongala (Q1424289) — the wreck of the Adelaide Steamship Company's passenger steamer off Queensland, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: built by Armstrong, Whitworth & Co at Newcastle upon Tyne in 1903 for the Adelaide Steamship Company:
length 106.7 m, beam 13.8 m, depth 8.3 m, two decks, single screw (triple-expansion engine), berths for 110
first-class and 130 second-class passengers, holds for some 3,000 t of cargo, a steam windlass on the
forecastle, cranes and derricks, the steering-gear house aft. One funnel and two masts (after period photographs).

The wreck, after the published descriptions: lost with all 122 aboard in a cyclone on 23 March 1911, found in
1958 some 12 nautical miles east of Cape Bowling Green. She lies whole on her starboard side, listing 60-70°,
the bow pointing 347°, on sand at about 28-30 m, her upper side 14-16 m under the surface. The hull is intact
and crusted with coral, the superstructure still much as it was; the ventilators and railings have collapsed,
the sand is scoured from under the starboard side. She is a protected site and a grave. The funnel and masts
are not described: here they have fallen and lie on the sand; their place, the colours and the exact layout of
the superstructure are approximate.

    python model_sources/Q1424289_yongala.py   -> models/item/Q1424289/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, sstep, fbm  # noqa: E402
import abyss  # noqa: E402

rng = np.random.default_rng(1911)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B, D = 106.7, 13.8, 8.3                 # length, beam, depth
FC = 2.4                                   # forecastle, the central island and the poop above the main deck
TW, TT = 5.0, 1.1                          # 'tween deck and tank top above the keel
WL = 5.4                                   # top of the red bottom paint
HULLC = (0.09, 0.09, 0.1)                  # black topsides
GREY, GREY_L, GREY_D = (0.62, 0.55, 0.42), (0.83, 0.82, 0.78), (0.3, 0.3, 0.31)   # buff masts and vents, white houses
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06)
FUNNEL = (0.1, 0.1, 0.1)

HHW = 3.0                                   # half width of the hatches
HATCHES = [(13, 21.5), (25, 33), (79, 86), (89, 95.5)]
BULKHEADS = [11.5, 23.5, 34.5, 77.5, 87.5, 97]
RAISED = [(0, 11.5), (34.5, 77.5), (97, L)]  # forecastle, central island (saloons), poop
ISLAND = (34.5, 77.5)
SIDE = -1                                    # she lies on her starboard side (−X)


def zd(d):
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):                                # a fast passenger steamer: fine entrance, long run to a counter stern
    w = 1.0
    if t > 0.76:
        w = math.sin(((1 - t) / 0.24) * math.pi / 2) ** 0.65
    if t < 0.26:
        w = min(w, 0.2 + 0.8 * math.sin((t / 0.26) * math.pi / 2) ** 0.8)
    return max(w, 0.004)


def yb_of(t):
    y = 0.0
    if t < 0.1:
        y = D * 0.55 * ((0.1 - t) / 0.1) ** 1.2
    if t > 0.985:
        y = max(y, D * 0.6 * ((t - 0.985) / 0.015) ** 2)
    return y


def yd_of(t):
    s = (t - 0.5) * 2
    return D + (1.5 if s > 0 else 0.9) * s * s


def k_of(t):
    return 2.5 + 7.0 * w_of(t) ** 4


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
    pts = [(x, y, z), (x, y + 2.4, z), (x + s * (0.9 + bent), y + 3.0 - bent * 1.5, z)]
    return tube(pts, 0.12, GREY, seg=5)


def propeller(d, y, r=1.9):
    P = [at(cyl(0.5, 0.4, 1.0, BRONZE, seg=12).R(math.pi / 2, 0, 0), d, y=y)]
    for k in range(4):
        a = 2 * math.pi * k / 4
        P.append(at(box(r, 0.16, 1.3, BRONZE, bottom=False).T(r / 2 + 0.3, 0, 0).R(0.45, 0, 0).R(0, 0, a), d - 0.4, y=y))
    P.append(at(cyl(0.3, 0.3, 3.0, STEEL, seg=8).R(math.pi / 2, 0, 0), d - 3.0, y=y))     # shaft
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


# ── Cargo: general cargo in the holds (crates, bales, casks) ─────────────────
def cask(r=0.32, h=0.85):
    return lathe([(r * 0.85, 0), (r, h * 0.25), (r * 1.05, h * 0.5), (r, h * 0.75), (r * 0.85, h)], (0.4, 0.3, 0.2), seg=8)


def hold_cargo(wreck):
    j = (lambda s: rng.normal(0, s)) if wreck else (lambda s: 0.0)
    P = []
    for k, (a, b) in enumerate(HATCHES):
        for level, y in ((0, TT), (1, TW)):
            xs = (-1.6, 0.0, 1.6) if level == 0 else (-4.2, 4.2)
            for d in np.arange(a + 0.8, b - 0.6, 1.4):
                for x in xs:
                    h = rng.uniform(0.5, 1.1)
                    kind = rng.random()
                    if kind < 0.55:
                        P.append(crate(rng.uniform(0.8, 1.3), h, rng.uniform(0.7, 1.2), (0.38, 0.31, 0.21)).R(0, j(0.08), 0).T(x + j(0.1), y, zd(d)))
                    elif kind < 0.8:
                        P.append(cask().T(x + j(0.1), y, zd(d)))
                    else:
                        bale = box(1.1, 0.8, 0.9, (0.55, 0.5, 0.4)); crumple(bale, 0.07, 1.5, int(d * 10 + x))
                        P.append(bale.T(x, y, zd(d)))
    return P


# ── The ship, piece by piece ─────────────────────────────────────────────────
def superstructure(built):
    """The saloon and boat decks on the central island, the bridge, the funnel, the boats."""
    a, b = ISLAND
    y0 = deck_y((a + b) / 2) + FC
    P = [box(B * 0.86, 2.5, 37.0, GREY_L).T(0, y0, zd(56.5))]                  # saloon deck house
    for s in (-1, 1):
        P += [quad(s * B * 0.43, y0 + 1.4, zd(d), 0.9, 0.85, "+x" if s > 0 else "-x", DARK) for d in np.arange(39.5, 74.5, 1.6)]
    yB = y0 + 2.5
    P.append(box(B * 0.66, 2.4, 27.0, GREY_L).T(0, yB, zd(54.0)))            # boat-deck house (smoke room, music room)
    for s in (-1, 1):
        P += [quad(s * B * 0.33, yB + 1.3, zd(d), 0.45, 0.45, "+x" if s > 0 else "-x", DARK) for d in np.arange(42, 67, 1.5)]
    yC = yB + 2.4
    P.append(box(B * 0.5, 2.3, 4.0, GREY_L).T(0, yC, zd(42.5)))             # wheelhouse
    P += [quad(x, yC + 1.4, zd(40.5) + 0.02, 0.75, 0.75, "+z", DARK) for x in np.arange(-2.7, 2.8, 1.1)]
    P.append(box(B * 1.0, 0.22, 2.4, GREY_L).T(0, yC, zd(41.5)))            # bridge wings
    P.append(box(2.6, 0.9, 4.0, GREY_L).T(0, yC, zd(60.5)))                 # engine-room skylight
    if built:
        for d, x in ((50, 2.8), (50, -2.8), (64, 2.6), (64, -2.6), (46, 3.5), (46, -3.5)):
            P.append(at(ventilator(2.3, 0.38), d, x=x, y=yC))
        for s in (-1, 1):                                                    # promenade railings
            P.append(tube([(s * (B * 0.43 + 0.05), y0 + 3.5, zd(d)) for d in np.arange(36, 77.1, 2.0)], 0.03, GREY_L, seg=4))
    for d in (45.0, 51.0, 63.0):                                             # six boats, three a side
        for s in (-1, 1):
            if built:
                P.append(lifeboat().T(s * (B * 0.33 + 1.1), yC + 0.9, zd(d)))
            P += [davit(s * (B * 0.33 + 0.25), yC, zd(d + dd), s, bent=0.0 if built else rng.uniform(0.0, 0.6)) for dd in (-2.6, 2.6)]
    if built:
        fun = cyl(1.9, 1.85, 9.5, FUNNEL, seg=20, sx=0.76).R(-0.09, 0, 0)
        P.append(fun.T(0, yC, zd(56.0)))
    else:
        stump = cyl(1.9, 1.88, 0.8, FUNNEL, seg=20, sx=0.76, caps=False)
        crumple(stump, 0.3, 0.9, 41, sstep(0.2, 0.8, stump.v[:, 1]))
        P.append(stump.T(0, yC, zd(56.0)))
    return P


def ship_part(built):
    P = [hull(0, L)]
    holes = [(a, b, HHW) for a, b in HATCHES]
    for a, b in ((11.5, 34.5), (77.5, 97)):
        P.append(strip(a, b, deck_y, half, holes, DECKC))
        P.append(side_strips(a, b, lambda d: deck_y(d), lambda d: deck_y(d) + 1.0, HULLC))
    for a, b in RAISED:
        P.append(strip(a, b, lambda d: deck_y(d) + FC, half, [], WOOD if a == ISLAND[0] else DECKC))
        P.append(side_strips(a, b, lambda d: deck_y(d), lambda d: deck_y(d) + FC, HULLC))
        if built:
            P.append(side_strips(a, b, lambda d: deck_y(d) + FC, lambda d: deck_y(d) + FC + 1.0, GREY_L if a == ISLAND[0] else HULLC))
        if a > 0:
            P += face(a, deck_y(a), deck_y(a) + FC, GREY_L, doors=(-3.0, 3.0), sign=1)
        if b < L:
            P += face(b, deck_y(b), deck_y(b) + FC, GREY_L, doors=(-3.0, 3.0), sign=-1)
        a2, b2 = a + 1.2, b - 1.2
        P += [p.T(0, deck_y((a2 + b2) / 2) + FC * 0.55, 0) for p in portholes(a2, b2, 0, 2.0, lambda d: half(d) + 0.02, r=0.19)]
    for a, b in ((11.5, 34.5), (77.5, 97)):
        P.append(strip(a, b, lambda d: TW, lambda d: hw(zd(d), TW) - 0.05, holes, GREY_D))
        P.append(strip(a, b, lambda d: TT, lambda d: hw(zd(d), TT) - 0.05, [], (0.24, 0.22, 0.2)))
    for bh in BULKHEADS:
        P.append(section(bh, deck_y(bh), GREY_D))
    P += windlass(5) + chains(5, built)
    for s in (-1, 1):
        z = zd(2.8); y = deck_y(2.8) - 0.4
        P.append(anchor().T(s * (hw(z, y) + 0.25), y, z))
    P += bollards((3.5, 9.0), FC)
    for k, (a, b) in enumerate(HATCHES):
        P += hatch_coaming(a, b, built, wreck_beams=1)
    for d in (23.5, 87.5):                                                   # the two steam cranes stood by the masts
        P += mast(d, (20.0 if d < 50 else 18.0) if built else 2.0, built)
    P += superstructure(built)
    yp = deck_y(100) + FC
    P.append(at(box(5.0, 2.2, 4.0, GREY_L), L - 4.5, y=deck_y(L - 4.5) + FC))  # steering-gear house at the after end
    P += bollards((99.0, 103.0), FC, inset=1.0)
    if built:
        P.append(at(cyl(0.07, 0.05, 5.0, GREY, seg=6), L - 0.8, y=deck_y(L - 0.8) + FC))
    P += propeller(L - 4.5, 2.0, 1.75) + rudder(L - 1.5, 0.6)
    return P


# ── The wreck: whole, on her starboard side ─────────────────────────────────
LIST = math.radians(65)
wreck = merge(ship_part(False) + hold_cargo(True))
crumple(wreck, 0.13, 0.12, 11)
up = wreck.v[:, 1] > D + FC + 0.5
crumple(wreck, 0.25, 0.25, 12, up.astype(float))                            # the houses sagged a little


def roll(m):
    """Roll onto the starboard side (−X) about the keel — the same move for the wreck and for points on it."""
    return m.R(0, 0, -SIDE * LIST)


roll(wreck)
LIFT = -np.percentile(wreck.v[:, 1], 0.4) - 0.9
wreck.T(0, LIFT, 0)
XC = -0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max())
wreck.T(XC, 0, 0)


def onwreck(x, y, d):
    p = roll(Mesh([(x, y, zd(d))], [])).v[0]
    return p + np.array([XC, LIFT, 0.0])


deck_side = onwreck(0, deck_y(56) + FC + 6.0, 56)[0]
r = np.random.default_rng(71)
# the funnel and the masts, fallen on the sand on the side the deck faces
fun = cyl(1.9, 1.85, 8.5, FUNNEL, seg=20, sx=0.76, caps=False)
crumple(fun, 0.4, 0.3, 51)
fun.v[:, 0] *= 0.8
fun.R(0, 0, -SIDE * math.pi / 2).R(0, -0.3, 0).T(deck_side + SIDE * 1.5, 1.4, zd(57.0))
masts = []
for k, (d, ln) in enumerate(((23.5, 16.0), (87.5, 14.0))):
    base = onwreck(0, deck_y(d) + 4.2, d)
    m = merge([cyl(0.36, 0.2, ln, GREY, seg=10, caps=False), box(2.8, 0.22, 0.35, GREY).T(0, ln * 0.6, 0)])
    crumple(m, 0.1, 0.4, 60 + k)
    masts.append(m.R(0, 0, -SIDE * (math.pi / 2 - 0.05)).R(0, 0.25 * (1 - 2 * k), 0).T(base[0] + SIDE * 1.5, 0.3, base[2]))
fallen = []                                                                  # the collapsed ventilators and railings, crates spilt from the holds
for k in range(10):
    v = ventilator(r.uniform(1.4, 2.3), 0.38); crumple(v, 0.12, 0.8, 90 + k)
    fallen.append(v.R(math.pi / 2, r.uniform(0, 6.28), 0).T(deck_side + SIDE * r.uniform(1, 9), 0.4, zd(r.uniform(40, 72))))
for k in range(8):
    p = tube([(0, 0, 0), (0, r.uniform(-0.3, 0.3), r.uniform(4, 9))], 0.04, GREY_L, seg=4)
    fallen.append(p.R(0, r.uniform(-0.6, 0.6), 0).T(deck_side + SIDE * r.uniform(1, 8), 0.1, zd(r.uniform(36, 76))))
for k in range(16):
    fallen.append(crate(r.uniform(0.6, 1.2), r.uniform(0.4, 0.8), r.uniform(0.5, 1.0), (0.38, 0.31, 0.21)).R(r.uniform(-0.3, 0.3), r.uniform(0, 6.28), r.uniform(-0.3, 0.3))
                  .T(deck_side + SIDE * r.uniform(1, 10), 0.0, zd(r.choice([17.0, 29.0, 82.0, 92.0]) + r.normal(0, 3))))
debris = merge(fallen)

# ── Coral: the hull is crusted all over ─────────────────────────────────────
CORAL_COLS = [(0.55, 0.48, 0.32), (0.62, 0.56, 0.38), (0.5, 0.42, 0.5), (0.72, 0.5, 0.55), (0.45, 0.5, 0.33), (0.75, 0.68, 0.45),
              (0.8, 0.45, 0.3), (0.42, 0.36, 0.55)]


def corals(m, n, seed):
    rr = np.random.default_rng(seed)
    out = []
    for k, (p, nn) in enumerate(abyss.sample_surface(m, n, lambda fn: fn[:, 1] > -0.15, rr)):
        rad = rr.uniform(0.15, 0.6) * (1.0 if nn[1] > 0.6 else 0.7)
        c = abyss.blob(rad, CORAL_COLS[rr.integers(len(CORAL_COLS))], seed * 1000 + k, sub=1, amp=0.4)
        c.v[:, 1] *= rr.uniform(0.35, 0.7)
        c.c = np.clip(c.c * (0.8 + 0.4 * fbm(c.v * 6, 1.0, seed + k)[:, None]), 0, 1)
        upv = np.array([0.0, 1.0, 0.0]); ax = np.cross(upv, nn); s_ = np.linalg.norm(ax); co = float(np.dot(upv, nn))
        if s_ > 1e-6:
            ax = ax / s_; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            c.v = c.v @ (np.eye(3) + s_ * K + (1 - co) * K @ K).T
        out.append(c.T(*(p + nn * 0.05)))
    return merge(out)


coral = merge([corals(wreck, 700, 7), corals(fun, 16, 8)] + [corals(m, 6, 9 + k) for k, m in enumerate(masts)])

# ── The ship as built ────────────────────────────────────────────────────────
ship = merge(ship_part(True))
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1424289" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
sites = [{"steel": [("hull_section", wreck), ("funnel", fun), ("masts", merge(masts), "loose"), ("debris_field", debris, "loose")],
          "mud": [],
          "plain": [("corals", coral, "life")]}]
dress_wreck(out, sites, depth=-29, kind="liner", seed=1911, memorial=True, life="tropical")
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
