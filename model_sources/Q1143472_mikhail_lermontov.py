"""
MS Mikhail Lermontov (Q1143472) — the wreck of the Soviet cruise liner in Port Gore, New Zealand, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the last of the five Ivan Franko-class liners, built at Wismar (East Germany) in 1972 for the Baltic
Shipping Company: about 176 m long, beam 23.6 m, twin screws, a long white superstructure of several decks, the
bridge forward, one broad funnel aft with the red band and the hammer and sickle, cargo hatches forward and aft.

The wreck, after the published descriptions and dive guides: on 16 February 1986, on a cruise round the
Marlborough Sounds, she struck rocks off Cape Jackson and sank in Port Gore; one crewman died. She lies whole on
her starboard side in 36-37 m, her port side 12 m under the surface, the funnel with its hammer and sickle still
in place; the interior is open to technical divers. Cold-water life: anemones, sponges, kelp-fish. The heading
(not published: 0), the colours and the exact layout of the decks are approximate.

    python model_sources/Q1143472_mikhail_lermontov.py   -> models/item/Q1143472/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, sstep, fbm, yaw_quat  # noqa: E402

rng = np.random.default_rng(1986)

L, B, D = 175.8, 23.6, 13.0
FC, TW, TT, WL = 2.6, 8.0, 1.4, 7.2
ENTRY, ENTRY_P, RUN, STERN_W, CUTUP_T, CUTUP_H, SHEER_F, SHEER_A, K0, K1 = 0.3, 0.75, 0.28, 0.32, 0.08, 0.35, 2.2, 0.8, 2.4, 6.0
HULLC = (0.86, 0.86, 0.84)                   # white
SUPC = (0.88, 0.88, 0.86)
GREY, GREY_L, GREY_D = (0.4, 0.42, 0.43), (0.5, 0.52, 0.52), (0.3, 0.3, 0.31)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER, FUNNELC = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06), (0.86, 0.86, 0.84)
HHW = 3.5
HATCHES = [(14, 21), (24, 30), (157, 164)]
BULKHEADS = [12, 32, 154]
RAISED = [(0, L)]
HOUSES = [(36, 150, 0.92, 2.5, 0), (38, 146, 0.9, 2.5, 1), (40, 138, 0.84, 2.5, 2), (42, 120, 0.6, 2.5, 3), (42, 50, 0.6, 2.4, 4)]
FUNNELS = [(112.0, 3.6, 9.0, 0.6)]
MASTS = [(22.5, 12.0)]
BOATS = [(d, 3) for d in (58.0, 66.0, 74.0, 82.0, 90.0, 98.0)]
BOAT_X = 0.31
SCREWS = [(-4.6, L - 12.0, 2.6, 2.2), (4.6, L - 12.0, 2.6, 2.2)]
RUDDER_D = L - 4.0
ANCHORS_GONE = ()


def zd(d):
    return L / 2 - d


def t_of(z):
    return (z + L / 2) / L


def w_of(t):                                # hull half-breadth factor along the length (ENTRY forward, RUN aft)
    w = 1.0
    if t > 1 - ENTRY:
        w = math.sin(((1 - t) / ENTRY) * math.pi / 2) ** ENTRY_P
    if t < RUN:
        w = min(w, STERN_W + (1 - STERN_W) * math.sin((t / RUN) * math.pi / 2) ** 0.8)
    return max(w, 0.004)


def yb_of(t):                               # keel: cut up under the stern, rounded forefoot
    y = 0.0
    if t < CUTUP_T:
        y = D * CUTUP_H * ((CUTUP_T - t) / CUTUP_T)
    if t > 0.985:
        y = max(y, D * 0.6 * ((t - 0.985) / 0.015) ** 2)
    return y


def yd_of(t):                               # sheer
    s = (t - 0.5) * 2
    return D + (SHEER_F if s > 0 else SHEER_A) * s * s


def k_of(t):
    return K0 + K1 * w_of(t) ** 4


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


# ── The ship, piece by piece (driven by the description above) ───────────────
#   RAISED: [(d0, d1)] forecastle / islands / poop above the main deck, FC high
#   HOUSES: [(d0, d1, width fraction of B, height, level)] deck houses on the raised decks (level 0, 1, 2…)
#   BRIDGE_FRONT: d of the wheelhouse windows (on the top house) or None
#   FUNNELS: [(d, radius, height, squash)];  MASTS: [(d, height)];  BOATS: [(d, level)]
#   SCREWS: [(x, d, y, radius)];  RUDDER_D
LEVEL_H = 2.5


def ship_part(d0, d1, built, mast_left=None, funnel_state="stump", houses_damage=None):
    """Everything between d0 and d1 from the bow. mast_left: heights left standing (None: stumps of 2 m).
    funnel_state: 'stand', 'stump' or 'gone' for the wreck."""
    inr = lambda a, b=None: (d0 <= a <= d1) if b is None else (a < d1 and b > d0)
    clip = lambda a, b: (max(a, d0), min(b, d1))
    P = [hull(d0, d1)]
    holes = [(a, b, HHW) for a, b in HATCHES]
    wells, last = [], 0.0
    for a, b in RAISED:
        if a > last + 0.5:
            wells.append((last, a))
        last = max(last, b)
    if last < L - 0.5:
        wells.append((last, L))
    for a, b in wells:
        a2, b2 = clip(a, b)
        if b2 - a2 > 0.3:
            P.append(strip(a2, b2, deck_y, half, holes, DECKC))
            P.append(side_strips(a2, b2, lambda d: deck_y(d), lambda d: deck_y(d) + 1.0, HULLC))
            P.append(strip(a2, b2, lambda d: TW, lambda d: hw(zd(d), TW) - 0.05, holes, GREY_D))
            P.append(strip(a2, b2, lambda d: TT, lambda d: hw(zd(d), TT) - 0.05, [], (0.24, 0.22, 0.2)))
    for a, b in RAISED:
        a2, b2 = clip(a, b)
        if b2 - a2 > 0.3:
            P.append(strip(a2, b2, lambda d: deck_y(d) + FC, half, [h for h in holes if h[0] >= a and h[1] <= b], DECKC))
            P.append(side_strips(a2, b2, lambda d: deck_y(d), lambda d: deck_y(d) + FC, HULLC))
            if built:
                P.append(side_strips(a2, b2, lambda d: deck_y(d) + FC, lambda d: deck_y(d) + FC + 1.0, HULLC))
            if a > 0 and inr(a):
                P += face(a, deck_y(a), deck_y(a) + FC, SUPC, doors=(-B * 0.2, B * 0.2), sign=1)
            if b < L and inr(b):
                P += face(b, deck_y(b), deck_y(b) + FC, SUPC, doors=(-B * 0.2, B * 0.2), sign=-1)
            a3, b3 = max(a2, a + 1.2), min(b2, b - 1.2)
            if b3 > a3:
                P += [p.T(0, deck_y((a3 + b3) / 2) + FC * 0.55, 0) for p in portholes(a3, b3, 0, 2.2, lambda d: half(d) + 0.02, r=0.2)]
    for bh in BULKHEADS:
        if d0 + 0.2 < bh < d1 - 0.2:
            P.append(section(bh, deck_y(bh), GREY_D))
    if inr(0, 12):                                                           # forecastle gear
        dw = min(6.0, L * 0.05)
        P += windlass(dw) + chains(dw, built)
        for s in (-1, 1):
            z = zd(3.0); y = deck_y(3.0) - 0.5
            if s not in ANCHORS_GONE or built:
                P.append(anchor().T(s * (hw(z, y) + 0.25), y, z))
        P += bollards((3.5, 9.0), FC if RAISED and RAISED[0][0] == 0 else 0.0)
    for k, (a, b) in enumerate(HATCHES):
        if a >= d0 - 0.1 and b <= d1 + 0.1:
            P += hatch_coaming(a, b, built, wreck_beams=1)
    for k, (d, h) in enumerate(MASTS):
        if inr(d):
            hh = h if built else (2.0 if mast_left is None else mast_left[k])
            if hh > 0:
                P += mast(d, hh, built)
    hs = []
    for (a, b, wf, h, lv) in HOUSES:
        if inr(a, b) and a >= d0 - 0.1 and b <= d1 + 0.1:
            y0 = deck_y((a + b) / 2) + FC + LEVEL_H * lv
            hs.append(box(B * wf, h, b - a, SUPC).T(0, y0, zd((a + b) / 2)))
            for s in (-1, 1):
                hs += [quad(s * B * wf / 2, y0 + h * 0.55, zd(dd), 0.8 if lv < 2 else 1.0, 0.7, "+x" if s > 0 else "-x", DARK) for dd in np.arange(a + 1.0, b - 0.8, 1.6)]
            hs += [quad(x, y0 + h * 0.55, zd(a) + 0.02, 0.8, 0.7, "+z", DARK) for x in np.arange(-B * wf / 2 + 0.8, B * wf / 2 - 0.6, 1.4)]
    if hs:
        hm = merge(hs)
        if not built and houses_damage:
            houses_damage(hm)
        P.append(hm)
    top = max([lv for (_, _, _, _, lv) in HOUSES], default=0)
    for (d, r, h, sx) in FUNNELS:
        if inr(d):
            yb = deck_y(d) + FC + LEVEL_H * (top + 1)
            if built or funnel_state == "stand":
                f = cyl(r, r * 0.97, h, FUNNELC, seg=20, sx=sx).R(-0.07, 0, 0)
                f.c[f.v[:, 1] > h - 1.0] = BLACK
                if not built:
                    crumple(f, 0.2, 0.5, int(d))
                P.append(f.T(0, yb, zd(d)))
            elif funnel_state == "stump":
                st = cyl(r, r * 0.98, 1.0, FUNNELC, seg=20, sx=sx, caps=False)
                crumple(st, 0.3, 0.9, int(d), sstep(0.2, 1.0, st.v[:, 1]))
                P.append(st.T(0, yb, zd(d)))
    for (d, lv) in BOATS:
        if inr(d):
            y0 = deck_y(d) + FC + LEVEL_H * lv
            for s in (-1, 1):
                wf = BOAT_X
                if built:
                    P.append(lifeboat().T(s * (B * wf + 1.1), y0 + 0.9, zd(d)))
                P += [davit(s * (B * wf + 0.2), y0, zd(d + dd), s, bent=0.0 if built else rng.uniform(0.0, 0.6)) for dd in (-2.6, 2.6)]
    for (x, d, y, r) in SCREWS:
        if inr(d):
            P += [m.T(x, 0, 0) for m in propeller(d, y, r)]
    if inr(RUDDER_D):
        P += rudder(RUDDER_D, 0.6)
    return P


# ── Placing the wreck and things about it ────────────────────────────────────
def make_onwreck(roll_fn, xc, lift):
    def onwreck(x, y, d):
        p = roll_fn(Mesh([(x, y, zd(d))], [])).v[0]
        return p + np.array([xc, lift, 0.0])
    return onwreck


def plates(n, cx, cz, spread, seed, cols=None):
    cols = cols or (HULLC, SUPC, DECKC)
    rr = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = box(rr.uniform(1.0, 5.0), 0.14, rr.uniform(0.8, 3.0), cols[rr.integers(len(cols))])
        crumple(p, 0.4, 0.4, seed + i)
        out.append(p.R(rr.uniform(-0.5, 0.5), rr.uniform(0, 6.28), rr.uniform(-0.5, 0.5)).T(cx + rr.normal(0, spread), 0.2, cz + rr.normal(0, spread)))
    return out


CORAL_COLS = [(0.55, 0.48, 0.32), (0.62, 0.56, 0.38), (0.5, 0.42, 0.5), (0.72, 0.5, 0.55), (0.45, 0.5, 0.33), (0.75, 0.68, 0.45),
              (0.8, 0.45, 0.3), (0.42, 0.36, 0.55)]


def corals(m, n, seed, cols=None, size=(0.15, 0.6)):
    import abyss
    cols = cols or CORAL_COLS
    rr = np.random.default_rng(seed)
    out = []
    for k, (p, nn) in enumerate(abyss.sample_surface(m, n, lambda fn: fn[:, 1] > -0.15, rr)):
        rad = rr.uniform(*size) * (1.0 if nn[1] > 0.6 else 0.7)
        c = abyss.blob(rad, cols[rr.integers(len(cols))], seed * 1000 + k, sub=1, amp=0.4)
        c.v[:, 1] *= rr.uniform(0.35, 0.7)
        c.c = np.clip(c.c * (0.8 + 0.4 * fbm(c.v * 6, 1.0, seed + k)[:, None]), 0, 1)
        upv = np.array([0.0, 1.0, 0.0]); ax = np.cross(upv, nn); s_ = np.linalg.norm(ax); co = float(np.dot(upv, nn))
        if s_ > 1e-6:
            ax = ax / s_; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            c.v = c.v @ (np.eye(3) + s_ * K + (1 - co) * K @ K).T
        out.append(c.T(*(p + nn * 0.05)))
    return merge(out)


def built_ship():
    ship = merge(ship_part(0, L, True))
    ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
    return ship.T(0, 0, -zd(L / 2))


def funnel_paint(m, d, r, h, yb):
    """The red band and a gold hammer and sickle on both sides of the funnel, black top."""
    out = [m]
    for s in (-1, 1):
        x = s * (r * 0.6 + 0.08)
        out.append(quad(x, yb + h * 0.62, zd(d), 2 * r * 0.95, 2.0, "+x" if s > 0 else "-x", (0.62, 0.1, 0.08), eps=0.0))
        out.append(quad(x + s * 0.02, yb + h * 0.64, zd(d), 1.1, 1.1, "+x" if s > 0 else "-x", (0.85, 0.68, 0.2), eps=0.0))
    return out


def radar_mast(built):
    P = [cyl(0.3, 0.2, 9.0, SUPC, seg=8), box(3.0, 0.25, 0.4, SUPC).T(0, 6.0, 0)]
    if built:
        P.append(box(4.5, 0.3, 0.6, (0.3, 0.3, 0.3)).T(0, 8.5, 0))
    return merge(P)


yfun = deck_y(112) + FC + LEVEL_H * 5
LIST = math.radians(88)                                                       # on her starboard side (−X)


def roll(m):
    return m.R(0, 0, LIST)


extra = funnel_paint(Mesh(), 112.0, 3.6, 9.0, yfun)[1:] + [radar_mast(False).T(0, deck_y(46) + FC + LEVEL_H * 5, zd(46))]
wreck = merge(ship_part(0, L, False, mast_left=[3.0], funnel_state="stand", houses_damage=lambda m: crumple(m, 0.2, 0.3, 61)) + extra)
crumple(wreck, 0.14, 0.1, 11)
roll(wreck)
LIFT = -np.percentile(wreck.v[:, 1], 0.5) - 1.0
wreck.T(0, LIFT, 0)
XC = -0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max())
wreck.T(XC, 0, 0)
onwreck = make_onwreck(roll, XC, LIFT)
side = onwreck(0, D + 14, 90)[0]
debris = merge(plates(25, side - 5, zd(80), 20, 100) + [lifeboat().R(0.3, 0.6, 0.4).T(side - 8, 0.4, zd(70)), lifeboat().R(-0.2, 2.6, 0.3).T(side - 12, 0.3, zd(95))])

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q1143472" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull_section", wreck), ("debris_field", debris, "loose")], "mud": [], "plain": []}],
            depth=-36, kind="liner", seed=1986)
ship = built_ship()
ship = merge([ship] + funnel_paint(Mesh(), 112.0, 3.6, 9.0, yfun)[1:] + [radar_mast(True).T(0, deck_y(46) + FC + LEVEL_H * 5, zd(46))])
dress_built(out.with_name("model_built.glb"), [("main", ship)])
