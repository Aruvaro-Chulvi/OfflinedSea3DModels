"""
SS Thistlegorm (Q32276) — the wreck of the British armed merchant ship in the Red Sea, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a shelter-deck steamer of the Albyn Line, built in Sunderland in 1940: length 126.5 m, beam 17.7 m,
one triple-expansion engine amidships, one screw. Forecastle, holds 1-3 forward of the midships island (bridge,
engine room, funnel), holds 4-5 aft of it, poop with the stern guns (a low-angle gun and an anti-aircraft gun).
On her last voyage (1941) she carried war supplies for the 8th Army: Bedford lorries, Norton and BSA motorcycles,
Universal (Bren) carriers, aircraft spares, rifles, boots, ammunition, and as deck cargo two LMS Stanier 8F
locomotives with their tenders and water wagons.

The wreck, after the published surveys and dive guides: on 6 October 1941, at anchor at Sha'ab Ali in the
Gulf of Suez, two German bombs hit hold 4; the ammunition there blew up and tore the ship apart aft of the
engine room. She lies at about 30 m, the bow section upright with the deck at about 17 m and holds 1-3 open
with their lorries and motorcycles in the 'tween deck and the lower hold; the main deck has fallen in on the
port side of hold 1. The midships superstructure was largely blown away (the bridge room survives), the
funnel is gone. Aft of the break, where hold 4 was, a field of torn plating, tracked carriers, tyres and
shells; beyond it the stern section, twisted onto its port side, with the guns, the propeller and the rudder.
The two locomotives were thrown off to either side: one off the port side level with the blast, the other
off the starboard side level with hold 2. The water wagons and coal tenders are still on deck forward.
Heading (bow to the north, after dive guides), the length of each piece, the angle of the stern, the exact
place of the cargo and of the debris are approximate.

    python model_sources/Q32276_thistlegorm.py   -> models/item/Q32276/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, sstep, fbm  # noqa: E402
from shipkit import install  # noqa: E402

rng = np.random.default_rng(1941)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B, D = 126.5, 17.7, 9.8                 # length, beam, depth to the main (shelter) deck
FC = 2.4                                   # forecastle, bridge deck and poop above the main deck
TW, TT = 6.0, 1.2                          # 'tween deck and tank top above the keel
WL = 6.2                                   # top of the red bottom paint
GREY, GREY_L, GREY_D = (0.44, 0.46, 0.47), (0.58, 0.6, 0.6), (0.3, 0.31, 0.32)
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.48, 0.38, 0.26)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, KHAKI, KHAKI2, RUBBER = (0.3, 0.29, 0.2), (0.33, 0.34, 0.24), (0.28, 0.3, 0.22), (0.06, 0.06, 0.06)
LOCO, BUFFER = (0.08, 0.08, 0.085), (0.5, 0.12, 0.08)

HHW = 3.6                                   # half width of the hatches
HATCHES = [(16, 27), (32, 45), (48, 55), (78, 89), (93, 104)]
BULKHEADS = [14, 29.5, 46.5, 56.5, 76.5, 91.5, 106]
RAISED = [(0, 13), (56.5, 76.5), (108, L)]  # forecastle, bridge deck (midships island), poop


def w_of(t):                                # a full tramp hull: fine entrance, cruiser stern
    w = 1.0
    if t > 0.8:
        w = math.sin(((1 - t) / 0.2) * math.pi / 2) ** 0.7
    if t < 0.22:
        w = min(w, 0.22 + 0.78 * math.sin((t / 0.22) * math.pi / 2) ** 0.8)
    return max(w, 0.004)


def yb_of(t):                               # keel: cut up under the stern (propeller, rudder), rounded forefoot
    y = 0.0
    if t < 0.1:
        y = D * 0.55 * ((0.1 - t) / 0.1)
    if t > 0.985:
        y = max(y, D * 0.6 * ((t - 0.985) / 0.015) ** 2)
    return y


def yd_of(t):                               # sheer: the deck rises towards the bow and the stern
    s = (t - 0.5) * 2
    return D + (2.0 if s > 0 else 1.0) * s * s


def k_of(t):
    return 2.5 + 7.0 * w_of(t) ** 4


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
    m.paint(lambda p: np.where((p[:, 1] < WL)[:, None], RED, GREY))
    return m


# ── Fittings ─────────────────────────────────────────────────────────────────
def stern_gun(d, y):
    """The low-angle gun on its platform at the stern, trained aft."""
    P = [at(cyl(0.7, 0.9, 1.4, GREY_D, seg=12), d, y=y),
         at(cyl(2.4, 2.4, 0.25, GREY_D, seg=20), d, y=y + 1.4)]
    for k in range(8):                                                                       # platform rail
        a = 2 * math.pi * k / 8
        P.append(at(cyl(0.04, 0.04, 1.0, GREY_D, seg=4), d, x=2.3 * math.cos(a), y=y + 1.65).T(0, 0, 2.3 * math.sin(a)))
    gy = y + 1.65
    P += [at(cyl(0.45, 0.5, 0.9, STEEL, seg=12), d, y=gy),
          at(box(0.9, 0.7, 1.8, STEEL), d + 0.2, y=gy + 0.8),
          at(box(2.4, 1.6, 0.1, GREY_D), d - 0.6, y=gy + 0.5),                               # shield
          at(cyl(0.11, 0.09, 5.4, STEEL, seg=8).R(-math.pi / 2 + 0.12, 0, 0), d + 0.5, y=gy + 1.15)]
    return P


def aa_gun(d, y):
    """The anti-aircraft position: an open gun tub with a twin machine gun."""
    P = [at(cyl(1.7, 1.7, 1.2, GREY_D, seg=16, caps=False), d, y=y), at(cyl(1.7, 1.7, 0.05, GREY_D, seg=16), d, y=y)]
    P += [at(cyl(0.2, 0.2, 1.0, STEEL, seg=8), d, y=y),
          at(box(0.6, 0.35, 0.7, STEEL), d, y=y + 1.0)]
    for s in (-1, 1):
        P.append(at(cyl(0.05, 0.05, 1.5, STEEL, seg=5).R(math.pi / 2 - 0.5, 0, 0), d - 0.2, x=s * 0.15, y=y + 1.2))
    return P


# ── Cargo ────────────────────────────────────────────────────────────────────
def lorry(col=KHAKI, crushed=0.0, seed=0):
    """A Bedford lorry (cab, bonnet, chassis, a flat bed with low sides), bow-to-+Z, standing on y = 0."""
    P = [box(1.0, 0.25, 5.8, BLACK).T(0, 0.55, 0),
         box(2.2, 1.55, 1.6, col).T(0, 0.9, 1.05),
         box(1.5, 0.95, 1.4, col).T(0, 0.9, 2.5),
         box(1.9, 0.18, 1.5, col).T(0, 0.85, 2.5),
         quad(0, 1.95, 1.86, 1.8, 0.55, "+z", DARK),
         box(2.3, 0.12, 3.8, col).T(0, 1.15, -1.3)]
    for s in (-1, 1):
        P.append(box(0.08, 0.55, 3.8, col).T(s * 1.12, 1.25, -1.3))
        P.append(quad(s * 1.11, 1.95, 1.2, 0.8, 0.5, "+x" if s > 0 else "-x", DARK))
    P += [box(2.3, 0.55, 0.08, col).T(0, 1.25, 0.6), box(2.3, 0.55, 0.08, col).T(0, 1.25, -3.2)]
    for z in (2.4, -1.4):
        for s in (-1, 1):
            P.append(wheel(0.47, 0.3).T(s * 0.95 - 0.15, 0.47, z))
    m = merge(P)
    if crushed:
        crumple(m, 0.25 * crushed, 0.9, seed)
        m.v[:, 1] *= 1 - 0.35 * crushed * sstep(0.6, 2.5, m.v[:, 1])
    return m


def motorcycle(col=KHAKI2):
    P = [wheel(0.33, 0.1, seg=8).T(-0.05, 0.33, 0.7), wheel(0.33, 0.1, seg=8).T(-0.05, 0.33, -0.7),
         box(0.14, 0.35, 1.1, col).T(0, 0.45, 0), box(0.3, 0.22, 0.5, col).T(0, 0.72, 0.15),
         box(0.25, 0.1, 0.4, BLACK).T(0, 0.82, -0.3), box(0.3, 0.3, 0.4, STEEL).T(0, 0.3, 0.05),
         box(0.7, 0.05, 0.05, BLACK).T(0, 1.0, 0.62), box(0.18, 0.08, 0.6, col).T(0, 0.66, -0.75)]
    return merge(P)


def carrier(col=KHAKI):
    """A Universal (Bren) carrier: low armoured hull on tracks."""
    hull_ = box(1.9, 0.75, 3.6, col).T(0, 0.35, 0)
    hull_.v[:, 1] -= 0.25 * sstep(1.0, 1.8, hull_.v[:, 2]) * (hull_.v[:, 1] > 0.6)    # sloped front plate
    P = [hull_]
    for s in (-1, 1):
        P.append(box(0.32, 0.6, 3.7, DARK).T(s * 1.05, 0.05, 0))
        for z in (-1.1, 0, 1.1):
            P.append(wheel(0.28, 0.1, col=BLACK).T(s * 1.05 - 0.05, 0.33, z))
    return merge(P)


def shell_pile(n, seed, r=0.08, ln=0.5):
    rr = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = cyl(r, r * 0.55, ln, (0.42, 0.36, 0.2), seg=6).R(math.pi / 2, 0, 0)
        out.append(s.R(0, rr.uniform(0, 6.28), rr.uniform(-0.2, 0.2)).T(rr.normal(0, 0.7), r + rr.uniform(0, 0.25), rr.normal(0, 0.7)))
    return merge(out)


def boots(n, seed):
    rr = np.random.default_rng(seed)
    return merge([box(0.11, 0.2, 0.28, (0.12, 0.1, 0.08)).R(rr.uniform(-1.5, 1.5), rr.uniform(0, 6.28), 0)
                  .T(rr.normal(0, 0.8), rr.uniform(0, 0.4), rr.normal(0, 0.9)) for _ in range(n)])


def locomotive():
    """An LMS Stanier 8F 2-8-0 (without its tender), funnel to +Z, rails at y = 0."""
    P = [box(2.0, 0.9, 11.2, LOCO).T(0, 0.5, 0.4),
         box(2.7, 0.08, 10.6, LOCO).T(0, 1.55, 0.6),
         cyl(0.86, 0.86, 7.4, LOCO, seg=16).R(math.pi / 2, 0, 0).T(0, 2.6, -2.4),
         cyl(0.92, 0.92, 1.6, LOCO, seg=16).R(math.pi / 2, 0, 0).T(0, 2.6, 5.0),
         cyl(0.6, 0.6, 0.06, (0.12, 0.12, 0.12), seg=14).R(math.pi / 2, 0, 0).T(0, 2.6, 6.6),
         cyl(0.3, 0.26, 0.6, LOCO, seg=10).T(0, 3.4, 5.8),
         cyl(0.42, 0.36, 0.5, LOCO, seg=10).T(0, 3.35, 1.3),
         box(2.0, 1.9, 2.4, LOCO).T(0, 1.6, -3.6),
         box(2.7, 2.4, 2.3, LOCO).T(0, 1.6, -5.5), box(2.9, 0.1, 2.7, LOCO).T(0, 4.0, -5.5),
         box(2.6, 0.7, 0.2, BUFFER).T(0, 0.9, 6.9)]
    for s in (-1, 1):
        P.append(quad(s * 1.36, 3.1, -5.2, 0.8, 0.7, "+x" if s > 0 else "-x", DARK))
        P.append(cyl(0.36, 0.36, 1.9, LOCO, seg=10).R(math.pi / 2, 0, 0).T(s * 1.05, 1.2, 4.0))       # cylinders
        P.append(cyl(0.18, 0.18, 0.3, BLACK, seg=8).R(math.pi / 2, 0, 0).T(s * 0.85, 0.9, 7.0))      # buffers
        for z in (-2.5, -0.85, 0.8, 2.45):                                                          # eight coupled wheels
            P.append(wheel(0.72, 0.16, col=LOCO, seg=14).T(s * 0.75 - (0.16 if s < 0 else 0), 0.72, z))
        P.append(wheel(0.46, 0.14, col=LOCO, seg=10).T(s * 0.75 - (0.14 if s < 0 else 0), 0.46, 5.4))
        P.append(box(0.08, 0.12, 5.2, STEEL).T(s * 0.95, 0.66, 0))                                  # coupling rods
    return merge(P)


def coal_tender():
    P = [box(2.0, 0.6, 7.0, LOCO).T(0, 0.45, 0), box(2.6, 2.2, 6.6, LOCO).T(0, 1.0, 0),
         box(2.3, 0.4, 3.6, (0.05, 0.05, 0.05)).T(0, 3.1, 1.2)]
    for s in (-1, 1):
        for z in (-2.2, 0, 2.2):
            P.append(wheel(0.55, 0.14, col=LOCO, seg=10).T(s * 0.75 - (0.14 if s < 0 else 0), 0.55, z))
    return merge(P)


def water_wagon():
    P = [box(2.0, 0.55, 6.6, LOCO).T(0, 0.45, 0), cyl(1.15, 1.15, 5.8, LOCO, seg=14).R(math.pi / 2, 0, 0).T(0, 2.1, -2.9),
         cyl(0.35, 0.35, 0.4, LOCO, seg=8).T(0, 3.2, 0)]
    for s in (-1, 1):
        for z in (-2.0, 2.0):
            P.append(wheel(0.5, 0.14, col=LOCO, seg=10).T(s * 0.75 - (0.14 if s < 0 else 0), 0.5, z))
    return merge(P)


install(globals(), "zd t_of hw deck_y half at stations strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place")


def hold_cargo(wreck):
    """What is in holds 1-3 (lower hold and 'tween deck), after the surveys; positions approximate."""
    j = (lambda s: rng.normal(0, s)) if wreck else (lambda s: 0.0)
    P = []
    # hold 1: lorries in the lower hold and the 'tween deck, motorcycles, piles of boots, rifle crates
    for d in (17.6, 24.2):
        for x in (-1.4, 1.4):
            P.append(place(lorry(seed=int(d * 10 + x)), d, x + j(0.1), TT, j(0.05)))
    for d in (17.5, 24.0):                                   # port 'tween deck: lorries (crushed where the deck fell in)
        P.append(place(lorry(KHAKI2, 1.0 if wreck else 0.0, int(d)), d, 6.0, TW, j(0.04), (0, j(0.03))))
    for k, d in enumerate(np.arange(15.0, 27.5, 0.8)):       # starboard 'tween deck: rows of motorcycles
        P.append(place(motorcycle(), d, -5.6 + j(0.05), TW, math.pi / 2 + j(0.15)))
    P.append(boots(70, 5).T(0, TT, zd(28.0)))
    P += [crate(1.2, 0.5, 0.4).T(x, TT, zd(14.9)) for x in (-1.6, -0.3, 1.0)]
    # hold 2: lorries below, Universal carriers, rows of motorcycles in the 'tween deck, aircraft parts
    for d in (32.4, 39.0):
        for x in (-1.4, 1.4):
            P.append(place(lorry(), d, x + j(0.1), TT, j(0.05)))
    for x in (-1.3, 1.3):
        P.append(place(carrier(), 44.3, x, TT, j(0.05)))
    for d in (32.6, 39.4):
        P.append(place(lorry(KHAKI2), d, 6.0, TW, j(0.03)))
    for d in np.arange(30.2, 46.0, 0.8):
        P.append(place(motorcycle(), d, -6.1 + j(0.06), TW, math.pi / 2 + j(0.12)))
    for d in (30.3, 45.8):
        P.append(place(box(3.4, 0.3, 1.2, (0.5, 0.5, 0.45)), d, 6.0, TW + 0.6, 0, (0.4, 0)))       # aircraft wing pieces leaning
    # hold 3: lorries in both levels
    for x in (-1.4, 1.4):
        P.append(place(lorry(), 51.5, x, TT, j(0.05)))
    for s in (-1, 1):
        P.append(place(lorry(KHAKI2), 51.3, s * 6.0, TW, j(0.04)))
    return P


def engine_room():
    """Two Scotch boilers and the three-cylinder triple-expansion engine, seen through the break."""
    P = []
    for x in (-2.4, 2.4):
        P.append(cyl(2.1, 2.1, 3.6, GREY_D, seg=16).R(math.pi / 2, 0, 0).T(x, TT + 2.2, zd(64.5)))
    P.append(box(3.6, 3.2, 5.8, GREY_D).T(0, TT, zd(71.5)))
    for k, d in enumerate((69.5, 71.5, 73.5)):
        P.append(cyl(0.6 + 0.25 * k, 0.6 + 0.25 * k, 1.6, GREY_D, seg=12).T(0, TT + 3.2, zd(d)))
    P.append(cyl(0.4, 0.4, 1.0, STEEL, seg=8).R(math.pi / 2, 0, 0).T(0, TT + 0.9, zd(76.5)))
    return P


# ── The ship, piece by piece ─────────────────────────────────────────────────
def ship_part(d0, d1, built):
    """Everything of the ship between d0 and d1 (distance from the bow)."""
    P = [hull(d0, d1)]
    inr = lambda a, b=None: (d0 <= a <= d1) if b is None else (a < d1 and b > d0)
    clip = lambda a, b: (max(a, d0), min(b, d1))
    holes = [(a, b, HHW) for a, b in HATCHES]
    # main deck in the wells (between the raised parts), and the decks of the raised parts
    for a, b in ((13, 56.5), (76.5, 108)):
        a2, b2 = clip(a, b)
        if b2 - a2 > 0.3:
            P.append(strip(a2, b2, deck_y, half, holes, DECKC))
            P.append(side_strips(a2, b2, lambda d: deck_y(d), lambda d: deck_y(d) + 1.0, GREY, inset=0.0))        # bulwarks
    for a, b in RAISED:
        a2, b2 = clip(a, b)
        if b2 - a2 > 0.3:
            P.append(strip(a2, b2, lambda d: deck_y(d) + FC, half, [], WOOD if a == 56.5 else DECKC))
            P.append(side_strips(a2, b2, lambda d: deck_y(d), lambda d: deck_y(d) + FC, GREY))
            P.append(side_strips(a2, b2, lambda d: deck_y(d) + FC, lambda d: deck_y(d) + FC + 1.0, GREY))
            if a > 0 and inr(a):
                P += face(a, deck_y(a), deck_y(a) + FC, GREY_L, doors=(-3.5, 3.5), sign=1)
            if b < L and inr(b):
                P += face(b, deck_y(b), deck_y(b) + FC, GREY_L, doors=(-3.5, 3.5), sign=-1)
    for a, b in RAISED:
        a2, b2 = clip(a + 1, b - 1)
        if b2 > a2:
            P += [p.T(0, deck_y((a2 + b2) / 2) + FC * 0.55, 0) for p in portholes(a2, b2, 0, 2.6, lambda d: half(d) + 0.02)]
    # 'tween decks, tank tops and bulkheads
    for a, b in ((14, 56.5), (76.5, 106)):
        a2, b2 = clip(a, b)
        if b2 - a2 > 0.3:
            P.append(strip(a2, b2, lambda d: TW, lambda d: hw(zd(d), TW) - 0.05, holes, GREY_D))
            P.append(strip(a2, b2, lambda d: TT, lambda d: hw(zd(d), TT) - 0.05, [], (0.24, 0.22, 0.2)))
    a2, b2 = clip(56.5, 76.5)
    if b2 - a2 > 0.3:
        P.append(strip(a2, b2, lambda d: TT, lambda d: hw(zd(d), TT) - 0.05, [], (0.24, 0.22, 0.2)))
    for bh in BULKHEADS:
        if d0 + 0.2 < bh < d1 - 0.2 and (built or bh != 76.5):     # the engine room's after bulkhead went with the blast
            P.append(section(bh, deck_y(bh), GREY_D))
    # forecastle: windlass, chains, anchors in their hawse pipes, bollards, a short jack staff
    if inr(0, 13):
        if inr(6):
            P += windlass(6) + chains(6, built)
        for s in (-1, 1):
            if inr(3):
                z = zd(3.0); y = deck_y(3.0) - 0.4
                P.append(anchor().T(s * (hw(z, y) + 0.25), y, z))
        P += [p for p in bollards((4.0, 10.0), FC) if d0 <= L / 2 - p.v[:, 2].mean() <= d1]
        if not built and inr(11):
            P.append(at(ventilator(1.4), 11, x=2.5, y=deck_y(11) + FC))
    # hatches, masts, derricks
    for k, (a, b) in enumerate(HATCHES):
        if inr(a, b) and a >= d0 - 0.1 and b <= d1 + 0.1:
            P += hatch_coaming(a, b, built, wreck_beams=(2 if k in (0, 2) else 0))
    if inr(29.5):
        P += mast(29.5, 17.0 if built else 6.5, built)
    if built and inr(91.5):
        P += mast(91.5, 15.0, built)
    # the midships island: boat deck house, bridge house, wheelhouse, funnel, boats, ventilators
    if inr(56.5, 76.5) and d0 <= 57:
        yb = deck_y(66) + FC
        house = [box(B * 0.7, 2.5, 15.0, GREY_L).T(0, yb, zd(66.5))]
        house += [quad(s * B * 0.35, yb + 1.4, zd(d), 0.45, 0.45, "+x" if s > 0 else "-x", DARK) for s in (-1, 1) for d in np.arange(60, 74, 1.8)]
        bridge = [box(B * 0.86, 2.5, 6.5, GREY_L).T(0, yb + 2.5, zd(61.2)),
                  box(B * 1.0, 0.25, 3.0, GREY_L).T(0, yb + 5.0, zd(59.8))]                     # bridge wings
        bridge += [quad(x, yb + 3.9, zd(57.95), 0.9, 0.8, "+z", DARK) for x in np.arange(-6, 6.1, 1.5)]
        wheel_h = [box(7.0, 2.3, 3.4, GREY_L).T(0, yb + 5.0, zd(60.3))]
        wheel_h += [quad(x, yb + 6.4, zd(58.58), 0.8, 0.8, "+z", DARK) for x in np.arange(-3, 3.1, 1.0)]
        skylight = box(3.2, 1.1, 4.5, GREY_L).T(0, yb + 2.5, zd(70.5))
        if built:
            P += house + bridge + wheel_h + [skylight]
            fun = cyl(2.05, 2.0, 8.6, GREY, seg=20, sx=0.72).R(-0.08, 0, 0)
            fun.c[fun.v[:, 1] > 7.4] = BLACK
            P.append(fun.T(0, yb + 2.5, zd(67.2)))
            for d in (64.5, 72.0):
                for s in (-1, 1):
                    P.append(lifeboat().T(s * (B * 0.35 + 1.1), yb + 3.3, zd(d)))
                    P += [davit(s * (B * 0.35 + 0.2), yb + 2.5, zd(d + dd), s) for dd in (-2.6, 2.6)]
            for d, x in ((63, 3.2), (63, -3.2), (74.5, 3.0), (74.5, -3.0)):
                P.append(at(ventilator(2.6), d, x=x, y=yb + 2.5))
        else:                                                    # largely blown away: the house torn, the bridge level crumpled, no funnel
            hm = merge(house + [skylight]); crumple(hm, 0.5, 0.35, 61)
            keep = hm.v[hm.f][:, :, 2].mean(1) > zd(72.5)          # the after end of the house is gone
            hm.f = hm.f[keep]
            bm = merge(bridge); crumple(bm, 0.9, 0.3, 62); bm.R(0, 0, 0.05).T(-0.3, -0.5, 0)
            P += [hm, bm]
            stump = cyl(2.05, 2.0, 1.2, GREY, seg=20, sx=0.72, caps=False)
            crumple(stump, 0.4, 0.8, 63, sstep(0.2, 1.2, stump.v[:, 1]))
            P.append(stump.T(0, yb + 2.5, zd(67.2)))
            for d in (64.5, 72.0):
                for s in (-1, 1):
                    P += [davit(s * (B * 0.35 + 0.2), yb + 2.5, zd(d + dd), s, bent=rng.uniform(0.2, 1.4)) for dd in (-2.6, 2.6)]
            P.append(at(ventilator(1.6), 63, x=3.2, y=yb + 2.5))
    if inr(56.5, 76.5) and d0 <= 57:
        P += engine_room()
    # poop: steering house, the AA tub, the stern gun platform, bollards
    if inr(108, L) and d0 <= 109:
        yp = lambda d: deck_y(d) + FC
        P.append(at(box(5.0, 2.2, 4.0, GREY_L), 114.5, y=yp(114.5)))
        P += aa_gun(110.5, yp(110.5))
        P += stern_gun(121.0, yp(121.0))
        P += bollards((117.5, 123.5), FC, inset=1.1)
    if inr(L - 3):
        P += propeller(L - 4.0, 2.0) + rudder(L - 1.0, 0.5)
    return P


def deck_cargo(built):
    """The railway deck cargo still forward: two water wagons by hatch 1, two coal tenders by hatch 2."""
    P = []
    for s in (-1, 1):
        P.append(place(water_wagon(), 21.5, s * 6.05, deck_y(21.5)))
        P.append(place(coal_tender(), 38.5, s * 6.05, deck_y(38.5)))
    if built:                                                # the two locomotives were on deck by hatch 4
        for s in (-1, 1):
            P.append(place(locomotive(), 83.5, s * 5.95, deck_y(83.5), math.pi if s < 0 else 0.0))
    return P


# ── The bow section: from the bow to the break, upright ─────────────────────
BREAK = 78.0
bow = merge(ship_part(0, BREAK, False) + deck_cargo(False))
dfb = L / 2 - bow.v[:, 2]
crumple(bow, 0.18, 0.12, 11)                                              # dents all over
crumple(bow, 1.6, 0.35, 12, sstep(71, BREAK, dfb))                        # torn plating at the break
side = bow.v[:, 0]
w_hold1 = sstep(15, 18, dfb) * sstep(28, 25, dfb) * sstep(HHW - 0.3, HHW + 1.5, side) * sstep(TW + 0.5, D - 0.3, bow.v[:, 1])
bow.v[:, 1] -= w_hold1 * (1.6 + 0.8 * np.sin(dfb * 0.9))                  # the main deck fallen in, port side of hold 1
w_up = sstep(73, BREAK, dfb) * sstep(D - 3, D + 1, bow.v[:, 1])
bow.v[:, 1] += w_up * 0.9; bow.v[:, 0] *= 1 + 0.08 * w_up                  # the deck at the break curled up and out
cargo = merge(hold_cargo(True))
bow_site = merge([bow, cargo])
bow_site.T(0, -0.6, 0)                                                    # settled a little into the sand

# ── Where hold 4 was: the bottom plating splayed open, debris all round ─────
blast = hull(BREAK, 96.0)
hy = blast.v[:, 1]
blast.v[:, 0] *= 1 + 0.55 * sstep(1.0, 9.0, hy)                           # the sides blown outwards…
blast.v[:, 1] = np.where(hy > 1.0, 1.0 + (hy - 1.0) * 0.28, hy)           # …and flattened down
crumple(blast, 1.6, 0.25, 21)
crumple(blast, 0.5, 0.9, 22)
blast.f = blast.f[fbm(blast.v[blast.f].mean(1), 0.18, 23) < 0.62]          # holes torn in it
blast.T(0, -0.7, 0)

r = np.random.default_rng(78)
deb = []
for i in range(70):                                                       # torn plates and bent frames
    d, x = r.uniform(BREAK - 2, 100), r.normal(0, 13)
    if r.random() < 0.7:
        p = box(r.uniform(1.5, 6.5), 0.14, r.uniform(1.0, 4.0), GREY if r.random() < 0.6 else DECKC)
        crumple(p, 0.5, 0.35, 300 + i)
        deb.append(p.R(r.uniform(-0.5, 0.5), r.uniform(0, 6.28), r.uniform(-0.5, 0.5)).T(x, 0.25, zd(d)))
    else:
        p = tube([(0, 0, 0), (r.uniform(-1, 1), r.uniform(1.0, 4.0), r.uniform(-1, 1)), (r.uniform(-2, 2), r.uniform(0.5, 3.0), r.uniform(-3, 3))], 0.11, GREY_D, seg=4)
        deb.append(p.T(x, 0, zd(d)))
for k, (d, x, up) in enumerate(((86, -8.5, True), (90, 11.0, True), (83, 13.5, False))):   # carriers thrown out, two upside down
    c = carrier()
    if up:
        c.R(0, 0, math.pi).T(0, 1.15, 0)
    deb.append(c.R(0, r.uniform(0, 6.28), 0).T(x, -0.1, zd(d)))
for k, (d, x) in enumerate(((81, -12), (88, 7), (94, -5), (85, 18))):     # lorries from hold 4, wrecked
    deb.append(lorry(KHAKI2, 1.3, 400 + k).R(r.uniform(-0.3, 0.3), r.uniform(0, 6.28), r.uniform(-0.6, 0.6)).T(x, -0.2, zd(d)))
for k, (d, x) in enumerate(((80, 4), (87, -3), (92, 9), (97, -10), (84, -15))):
    deb.append(shell_pile(int(r.integers(18, 34)), 500 + k).T(x, -0.05, zd(d)))
for k in range(10):
    deb.append(crate(r.uniform(0.6, 1.4), r.uniform(0.3, 0.6), r.uniform(0.4, 1.0)).R(0, r.uniform(0, 6.28), r.uniform(-0.3, 0.3)).T(r.normal(0, 9), -0.05, zd(r.uniform(79, 98))))
funnel = cyl(2.05, 2.0, 7.0, GREY, seg=20, sx=0.72, caps=False)          # the funnel, lying flattened off the port side
crumple(funnel, 0.6, 0.3, 77)
funnel.v[:, 0] *= 0.6
deb.append(funnel.R(0, 0, math.pi / 2).R(0, 0.4, 0).T(16.5, 0.8, zd(70)))
wh = box(7.0, 2.3, 3.4, GREY_L)                                           # what is left of the wheelhouse, on the sand
crumple(wh, 0.6, 0.4, 78)
wh.v[:, 1] *= 0.55
deb.append(wh.R(0.2, 0.9, 0.1).T(13.0, -0.3, zd(60.5)))
debris = merge(deb)

tyres = merge([tyre().R(r.uniform(-1.6, 1.6), r.uniform(0, 6.28), 0).T(r.normal(2, 10), 0.05, zd(r.uniform(79, 99))) for _ in range(26)])

# ── The two locomotives, thrown off either side ─────────────────────────────
loco_p = locomotive(); crumple(loco_p, 0.12, 0.5, 81)
loco_p.R(0, 0, -0.12).R(0, 0.35, 0).T(29.0, -0.2, zd(86))                 # port side, level with the blast
loco_s = locomotive(); crumple(loco_s, 0.12, 0.5, 82)
loco_s.R(0, 0, 0.18).R(0, math.pi - 0.25, 0).T(-30.0, -0.25, zd(38))       # starboard side, level with hold 2

# ── The stern section: from aft of hold 4 to the stern, twisted onto its port side ──
STERN0 = 96.0
stern = merge(ship_part(STERN0, L, False)
              + [crate(1.0, 0.6, 0.7).T(x, TT + 0.6 * k, zd(d)) for k in range(2) for d in (98.5, 101.5, 104.5) for x in (-2.4, -1.2, 0, 1.2, 2.4)]
              + [shell_pile(20, 600 + k).T(x, TT + 1.2, zd(d)) for k, (d, x) in enumerate(((99, -1.5), (103, 1.5)))])
dfs = L / 2 - stern.v[:, 2]
crumple(stern, 0.18, 0.12, 31)
crumple(stern, 1.5, 0.35, 32, sstep(101, STERN0, dfs))                    # torn plating at the forward end
stern.T(0, 0, -zd((STERN0 + L) / 2))
stern.R(0, 0, -0.85)                                                      # rolled onto her port side (port = +X)
stern.T(0, -np.percentile(stern.v[:, 1], 0.5) - 1.0, 0)                   # bedded into the sand
stern.R(0, 0.12, 0).T(1.5, 0, zd((STERN0 + L) / 2) - 6.0)                 # a little skewed, a few metres aft of the debris

# ── Old sand: scour ridges along the hull ────────────────────────────────────
def sand_ridges():
    def h(X, Z):
        d_ = L / 2 - Z
        along = sstep(-6, 4, d_) * sstep(BREAK + 4, BREAK - 4, d_)
        side_ = 0.9 * np.exp(-((np.abs(X) - B / 2 - 2.0) / 3.0) ** 2) * along
        return side_ + 0.15 * (fbm(np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1), 0.2, 7).reshape(X.shape) - 0.5) - 0.4
    g = grid(-B / 2 - 9, B / 2 + 9, zd(BREAK + 6), zd(-6), 24, 70, h, (0.72, 0.67, 0.56))
    g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > 0.12]
    return g


# ── Coral: the Red Sea wreck is crusted with hard corals and hung with soft ones ──
import abyss  # noqa: E402

CORAL_COLS = [(0.55, 0.48, 0.32), (0.62, 0.56, 0.38), (0.5, 0.42, 0.5), (0.72, 0.5, 0.55), (0.45, 0.5, 0.33), (0.75, 0.68, 0.45)]


def corals(m, n, seed):
    """Lumpy coral heads on the upward and side faces of a wreck piece, flattened against the steel."""
    rr = np.random.default_rng(seed)
    out = []
    for k, (p, nn) in enumerate(abyss.sample_surface(m, n, lambda fn: fn[:, 1] > -0.15, rr)):
        rad = rr.uniform(0.15, 0.55) * (1.0 if nn[1] > 0.6 else 0.7)
        c = abyss.blob(rad, CORAL_COLS[rr.integers(len(CORAL_COLS))], seed * 1000 + k, sub=1, amp=0.4)
        c.v[:, 1] *= rr.uniform(0.35, 0.7)
        c.c = np.clip(c.c * (0.8 + 0.4 * fbm(c.v * 6, 1.0, seed + k)[:, None]), 0, 1)
        up = np.array([0.0, 1.0, 0.0]); ax = np.cross(up, nn); s_ = np.linalg.norm(ax); co = float(np.dot(up, nn))
        if s_ > 1e-6:                                                    # turn the flattened head to face along the normal
            ax = ax / s_; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rm = np.eye(3) + s_ * K + (1 - co) * K @ K
            c.v = c.v @ Rm.T
        out.append(c.T(*(p + nn * 0.05)))
    return merge(out)


coral = merge([corals(bow_site, 320, 7), corals(stern, 110, 8), corals(loco_p, 14, 9), corals(loco_s, 14, 10)])

# ── The ship as built ────────────────────────────────────────────────────────
ship = merge(ship_part(0, L, True) + deck_cargo(True))
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
# Origin under the middle of the bow section (the main piece).
Z0 = zd(BREAK / 2)
for m in (bow_site, blast, debris, tyres, loco_p, loco_s, stern, coral, ship):
    m.T(0, 0, -Z0)
ridges = sand_ridges().T(0, 0, -Z0)

from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q32276" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
sites = [{"steel": [("bow_section", bow_site), ("stern_section", stern), ("hold4_bottom", blast), ("debris_field", debris, "loose"),
                    ("locomotive_port", loco_p), ("locomotive_starboard", loco_s)],
          "mud": [ridges],
          "plain": [("tyres", tyres, "artefact"), ("corals", coral, "life")]}]
dress_wreck(out, sites, depth=-30, kind="cargo", seed=1941, life="tropical")
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
