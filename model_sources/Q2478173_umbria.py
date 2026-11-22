"""
SS Umbria (Q2478173) — the wreck of the Italian liner scuttled off Port Sudan, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: built in Hamburg (Reiherstieg yard) in 1911-1912 as the emigrant liner and cargo ship Bahía Blanca,
length 155 m, beam 18.1 m, two masts, one funnel, twin screws; a long central superstructure with the
passenger decks (first-class promenade and dining room), forecastle, poop and five holds. Argentine from 1918,
Italian from 1935 as Umbria (troops and supplies for East Africa).

The wreck, after the published descriptions and dive guides: on 10 June 1940, held by the British at Wingate
Reef off Port Sudan, her captain Lorenzo Muiesan scuttled her as Italy entered the war, with some 6,000 t of
bombs, detonators, 2,000 t of cement, wine and three Fiat 1100 'Lunga' cars aboard. She lies whole on her port
side on the slope of the reef, her starboard side a few metres under the surface with the boat davits
standing out of it, the bottom at about 30-38 m. The funnel has fallen and lies on the sand, two of her
lifeboats lie on the bottom near her, the starboard propeller is at about 18 m, the port one half buried in
the sand; the port anchor chain runs out of its hawse to the bottom. The holds are open: bombs stacked in the
after holds and forward, cement and wine in the second hold, the three cars in the hold aft of the
superstructure, on the starboard side. The angle of the list, the heading (unknown: left at 0), the colours,
the place of the cargo, of the masts and of the boats are approximate.

    python model_sources/Q2478173_umbria.py   -> models/item/Q2478173/model.glb (+ model_built.glb)
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
import abyss  # noqa: E402

rng = np.random.default_rng(1940)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B, D = 155.0, 18.1, 11.0                # length, beam, depth to the shelter deck
FC = 2.5                                   # forecastle, the central island and the poop above the shelter deck
TW, TT = 6.6, 1.3                          # 'tween deck and tank top above the keel
WL = 7.0                                   # top of the red bottom paint
HULLC = (0.09, 0.09, 0.1)                  # black topsides
GREY, GREY_L, GREY_D = (0.62, 0.55, 0.42), (0.82, 0.81, 0.77), (0.3, 0.3, 0.31)   # buff masts and vents, white houses
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06)
FUNNEL, BOMB, CEMENT = (0.1, 0.1, 0.1), (0.36, 0.38, 0.3), (0.7, 0.68, 0.62)

HHW = 3.8                                   # half width of the hatches
HATCHES = [(18, 28), (33, 46), (103, 113), (118, 128), (131, 140)]
BULKHEADS = [15, 30.5, 48, 101, 115.5, 129.5, 142]
RAISED = [(0, 15), (48, 101), (142, L)]     # forecastle, central island (passenger decks), poop
ISLAND = (48, 101)


def w_of(t):                                # a liner of 1912: fine entrance, long run to a counter stern
    w = 1.0
    if t > 0.78:
        w = math.sin(((1 - t) / 0.22) * math.pi / 2) ** 0.65
    if t < 0.25:
        w = min(w, 0.2 + 0.8 * math.sin((t / 0.25) * math.pi / 2) ** 0.8)
    return max(w, 0.004)


def yb_of(t):                               # keel cut up under the counter stern (twin screws, rudder), rounded forefoot
    y = 0.0
    if t < 0.1:
        y = D * 0.55 * ((0.1 - t) / 0.1) ** 1.2
    if t > 0.985:
        y = max(y, D * 0.6 * ((t - 0.985) / 0.015) ** 2)
    return y


def yd_of(t):                               # sheer
    s = (t - 0.5) * 2
    return D + (1.8 if s > 0 else 1.0) * s * s


def k_of(t):
    return 2.5 + 7.0 * w_of(t) ** 4


# ── Fittings ─────────────────────────────────────────────────────────────────


# ── Cargo ────────────────────────────────────────────────────────────────────


install(globals(), "zd t_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place")


# ── Cargo ────────────────────────────────────────────────────────────────────
def bomb(ln=1.25, r=0.17):
    """An aircraft bomb: a streamlined body, nose to +Z, with a box tail."""
    prof = [(0.03, -ln / 2), (r * 0.75, -ln / 2 + 0.1), (r, -ln * 0.15), (r, ln * 0.2), (r * 0.7, ln * 0.42), (0.03, ln / 2)]
    body = lathe(prof, BOMB, seg=6, axis="z")
    fins = [box(r * 2.6, 0.03, 0.32, BOMB).T(0, 0, -ln / 2 - 0.05), box(0.03, r * 2.6, 0.32, BOMB, bottom=False).T(0, 0, -ln / 2 - 0.05)]
    return merge([body] + fins)


def bomb_stack(d, x, y, nx, ny, nz, jitter):
    """Bombs stowed in tiers, each one along the ship, nx across, ny high, nz along."""
    out = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                b = bomb()
                b.R(jitter(0.05), jitter(0.08), 0).T(x + (i - (nx - 1) / 2) * 0.38, y + 0.17 + j * 0.34, zd(d + k * 1.45))
                out.append(b)
    return out


def car(col=(0.16, 0.16, 0.18)):
    """A Fiat 1100 'Lunga' (long-wheelbase saloon of the late 1930s), bow to +Z, standing on y = 0."""
    lower = box(1.55, 0.7, 4.5, col).T(0, 0.32, 0)
    lower.v[:, 1] -= 0.18 * sstep(1.6, 2.25, lower.v[:, 2]) * (lower.v[:, 1] > 0.8)     # the bonnet falls to the grille
    cabin = box(1.42, 0.62, 2.3, col).T(0, 1.02, -0.45)
    cabin.v[:, 2] += 0.25 * (cabin.v[:, 1] > 1.4) * np.sign(-cabin.v[:, 2] - 0.45)       # sloping front and back
    P = [lower, cabin, quad(0, 1.33, 0.72, 1.2, 0.45, "+z", DARK)]
    for s in (-1, 1):
        P += [quad(s * 0.72, 1.33, z, 0.8, 0.4, "+x" if s > 0 else "-x", DARK) for z in (-0.95, 0.0)]
        for z in (1.45, -1.55):
            P.append(box(0.3, 0.38, 1.0, col).T(s * 0.72, 0.25, z))                      # mudguards
            P.append(wheel(0.33, 0.18).T(s * 0.72 - 0.09, 0.33, z))
    return merge(P)


def cement_stack(d, x, y, n, seed):
    rr = np.random.default_rng(seed)
    out = []
    for i in range(n):
        b = box(1.2, 0.8, 0.9, CEMENT)
        crumple(b, 0.08, 1.5, seed + i)
        out.append(b.R(0, rr.normal(0, 0.06), 0).T(x + (i % 3 - 1) * 1.25, y + (i // 9) * 0.8, zd(d + (i // 3) % 3 * 0.95)))
    return out


def wine_crates(d, x, y, n, seed):
    rr = np.random.default_rng(seed)
    return [crate(0.6, 0.35, 0.45, (0.38, 0.3, 0.2)).R(0, rr.normal(0, 0.1), 0).T(x + (i % 4) * 0.62, y + (i // 8) * 0.36, zd(d + (i // 4) % 2 * 0.47)) for i in range(n)]


def wire_coil(r=0.55):
    prof = [(r + 0.16 * math.cos(a), 0.18 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 7)]
    return lathe(prof, (0.45, 0.28, 0.14), seg=10)


def hold_cargo(wreck):
    """What is in the five holds, after the dive descriptions; the stowage itself is approximate."""
    j = (lambda s: rng.normal(0, s)) if wreck else (lambda s: 0.0)
    P = []
    # hold 1: aircraft tyres and coils of electric wire
    for k in range(18):
        P.append(tyre(0.5, 0.15).R(math.pi / 2, 0, 0).R(0, 0, 0).T((k % 3 - 1) * 1.15, TT + 0.5 + (k // 9) * 1.0, zd(19.5 + (k // 3) % 3 * 1.0)))
    for k in range(10):
        P.append(wire_coil().T((k % 2) * 1.4 - 0.7 + 5.2 * (1 if k < 5 else -1), TW + 0.18 + (k % 3) * 0.36, zd(21 + (k % 5) * 1.3)))
    P += bomb_stack(24.5, 0.0, TT, 6, 4, 2, j)
    # hold 2: cement and wine, aircraft bombs
    P += cement_stack(34.5, 0, TT, 27, 21)
    P += wine_crates(40.5, -1.2, TT, 24, 22)
    P += bomb_stack(43.0, 0.0, TT, 6, 3, 2, j)
    for s in (-1, 1):
        P += cement_stack(36.0, s * 5.6, TW, 18, 23 + s)
    # hold 3: building material and the three Fiat cars, in the 'tween deck on the starboard side
    for k, d in enumerate((103.6, 108.4, 113.2)):
        P.append(place(car(), d, -5.6 + j(0.08), TW, j(0.05)))
    P += [box(0.25, 0.25, 6.0, (0.45, 0.4, 0.33)).T((k % 6 - 2.5) * 0.3, TT + (k // 6) * 0.25, zd(108)) for k in range(36)]   # timber, steel bars
    P += cement_stack(105.0, 0, TT, 18, 31)
    # holds 4 and 5: bombs, tier upon tier, in the lower hold and the 'tween deck
    for d0 in (118.5, 131.5):
        P += bomb_stack(d0, 0.0, TT, 6, 4, 5, j)
        for s in (-1, 1):
            P += bomb_stack(d0 + 1.0, s * 5.6, TW, 4, 2, 4, j)
    return P


# ── The ship, piece by piece ─────────────────────────────────────────────────
def superstructure(built):
    """The passenger decks on the central island: promenade house, boat-deck house, bridge, funnel, boats."""
    a, b = ISLAND
    y0 = deck_y((a + b) / 2) + FC
    P = []
    A = box(B * 0.84, 2.6, 49.0, GREY_L).T(0, y0, zd(74.5))                    # promenade and saloons
    P.append(A)
    for s in (-1, 1):
        P += [quad(s * B * 0.42, y0 + 1.5, zd(d), 1.1, 1.0, "+x" if s > 0 else "-x", DARK) for d in np.arange(51.5, 98.5, 1.7)]
    yB = y0 + 2.6
    P.append(box(B * 0.66, 2.5, 40.0, GREY_L).T(0, yB, zd(73.0)))           # boat-deck house
    for s in (-1, 1):
        P += [quad(s * B * 0.33, yB + 1.4, zd(d), 0.45, 0.45, "+x" if s > 0 else "-x", DARK) for d in np.arange(54, 92.5, 1.6)]
    P += [quad(x, yB + 1.4, zd(53.0) + 0.02, 0.9, 0.8, "+z", DARK) for x in np.arange(-5, 5.1, 1.4)]
    yC = yB + 2.5
    P.append(box(B * 0.5, 2.4, 4.5, GREY_L).T(0, yC, zd(54.5)))             # wheelhouse and chart room
    P += [quad(x, yC + 1.5, zd(52.25) + 0.02, 0.8, 0.8, "+z", DARK) for x in np.arange(-3.6, 3.7, 1.2)]
    P.append(box(B * 1.0, 0.25, 2.6, GREY_L).T(0, yC, zd(53.4)))            # bridge wings
    P.append(box(3.0, 1.0, 5.0, GREY_L).T(0, yC, zd(82.0)))                 # engine-room skylight
    for d, x in ((64, 3.6), (64, -3.6), (86, 3.4), (86, -3.4), (59, 4.4), (59, -4.4)):
        P.append(at(ventilator(2.6, 0.45), d, x=x, y=yC))
    for d in (62.0, 68.0, 80.0, 86.0):                                       # eight boats, four a side
        for s in (-1, 1):
            if built:
                P.append(lifeboat().T(s * (B * 0.33 + 1.2), yC + 0.9, zd(d)))
            P += [davit(s * (B * 0.33 + 0.25), yC, zd(d + dd), s, bent=0.0 if built else rng.uniform(0.0, 0.5)) for dd in (-2.8, 2.8)]
    if built:
        fun = cyl(2.6, 2.5, 12.0, FUNNEL, seg=22, sx=0.74).R(-0.09, 0, 0)
        P.append(fun.T(0, yC, zd(74.0)))
    else:                                                                     # the funnel fell; a torn stump is left
        stump = cyl(2.6, 2.55, 1.0, FUNNEL, seg=22, sx=0.74, caps=False)
        crumple(stump, 0.35, 0.9, 41, sstep(0.2, 1.0, stump.v[:, 1]))
        P.append(stump.T(0, yC, zd(74.0)))
    return P


def ship_part(built):
    P = [hull(0, L)]
    holes = [(a, b, HHW) for a, b in HATCHES]
    for a, b in ((15, 48), (101, 142)):                                      # shelter deck in the wells
        P.append(strip(a, b, deck_y, half, holes, DECKC))
        P.append(side_strips(a, b, lambda d: deck_y(d), lambda d: deck_y(d) + 1.1, HULLC))
    for a, b in RAISED:
        P.append(strip(a, b, lambda d: deck_y(d) + FC, half, [], WOOD if a == 48 else DECKC))
        P.append(side_strips(a, b, lambda d: deck_y(d), lambda d: deck_y(d) + FC, HULLC))
        P.append(side_strips(a, b, lambda d: deck_y(d) + FC, lambda d: deck_y(d) + FC + 1.0, GREY_L if a == 48 else HULLC))
        if a > 0:
            P += face(a, deck_y(a), deck_y(a) + FC, GREY_L, doors=(-3.5, 3.5), sign=1)
        if b < L:
            P += face(b, deck_y(b), deck_y(b) + FC, GREY_L, doors=(-3.5, 3.5), sign=-1)
        a2, b2 = a + 1.5, b - 1.5
        P += [p.T(0, deck_y((a2 + b2) / 2) + FC * 0.55, 0) for p in portholes(a2, b2, 0, 2.4, lambda d: half(d) + 0.02)]
    P += [p.T(0, deck_y(74) - 1.6, 0) for p in portholes(20, 140, 0, 2.0, lambda d: half(d) + 0.02, r=0.2)]    # 'tween-deck ports (emigrant quarters)
    for a, b in ((15, 48), (101, 142)):
        P.append(strip(a, b, lambda d: TW, lambda d: hw(zd(d), TW) - 0.05, holes, GREY_D))
        P.append(strip(a, b, lambda d: TT, lambda d: hw(zd(d), TT) - 0.05, [], (0.24, 0.22, 0.2)))
    for bh in BULKHEADS:
        P.append(section(bh, deck_y(bh), GREY_D))
    # forecastle
    P += windlass(6) + chains(6, built)
    for s in (-1, 1):
        z = zd(3.2); y = deck_y(3.2) - 0.5
        if built or s < 0:                                                     # the port anchor is down
            P.append(anchor().T(s * (hw(z, y) + 0.25), y, z))
    P += bollards((4.0, 11.0), FC)
    # hatches and masts
    for k, (a, b) in enumerate(HATCHES):
        P += hatch_coaming(a, b, built, wreck_beams=(2 if k in (0, 3) else 1))
    P += mast(31.0, 26.0 if built else 2.5, built)
    P += mast(116.0, 24.0 if built else 2.5, built)
    P += superstructure(built)
    # poop: steering-gear house, bollards, ensign staff
    yp = deck_y(148) + FC
    P.append(at(box(6.0, 2.3, 5.0, GREY_L), 147.0, y=yp))
    P += bollards((144.5, 152.0), FC, inset=1.1)
    P.append(at(cyl(0.08, 0.06, 6.0, GREY, seg=6), L - 0.8, y=deck_y(L - 0.8) + FC))
    # twin screws and the rudder
    for x in (-3.1, 3.1):
        P += [m.T(x, 0, 0) for m in propeller(L - 9.0, 2.4, 1.8, x=x)]
    P += rudder(L - 2.2, 0.8)
    return P


# ── The wreck: whole, lying on her port side ─────────────────────────────────
LIST = math.radians(75)
wreck = merge(ship_part(False) + hold_cargo(True))
crumple(wreck, 0.15, 0.12, 11)                                              # dents all over
up = wreck.v[:, 1] > D + FC + 0.5
crumple(wreck, 0.35, 0.25, 12, up.astype(float))                            # the houses sagged a little


def roll(m):
    """The same move for the wreck and for points on it: roll onto the port side (+X) about the keel."""
    return m.R(0, 0, -LIST)


roll(wreck)
LIFT = -np.percentile(wreck.v[:, 1], 0.4) - 1.2                             # bedded into the sand
wreck.T(0, LIFT, 0)
XC = -0.5 * (wreck.v[:, 0].min() + wreck.v[:, 0].max())
wreck.T(XC, 0, 0)


def onwreck(x, y, d):
    """Where a point of the upright ship ends up on the wreck."""
    p = roll(Mesh([(x, y, zd(d))], [])).v[0]
    return p + np.array([XC, LIFT, 0.0])


# the port anchor chain, out of its hawse and down to the sand ahead
hp = onwreck(half(3.2) - 0.2, deck_y(3.2) - 0.6, 3.2)
chain = tube([hp, hp + np.array([1.5, -hp[1] * 0.6, 4.0]), np.array([hp[0] + 2.5, 0.15, hp[2] + 14.0]), np.array([hp[0] + 1.0, 0.15, hp[2] + 24.0])], 0.14, BLACK, seg=5)

# the funnel and the two masts lying on the sand, on the side the deck faces
deck_side = onwreck(0, deck_y(74) + FC + 7.6, 74)[0]
fun = cyl(2.6, 2.5, 11.0, FUNNEL, seg=22, sx=0.74, caps=False)
crumple(fun, 0.5, 0.25, 51)
fun.v[:, 0] *= 0.82
fun.R(0, 0, -math.pi / 2).R(0, 0.25, 0).T(deck_side + 1.5, 1.6, zd(75.5))
masts = []
for k, (d, ln) in enumerate(((31.0, 21.0), (116.0, 19.0))):
    base = onwreck(0, deck_y(d) + 4.7, d)
    m = merge([cyl(0.42, 0.24, ln, GREY, seg=10, caps=False), box(3.4, 0.25, 0.4, GREY).T(0, ln * 0.6, 0)])
    crumple(m, 0.12, 0.4, 60 + k)
    masts.append(m.R(0, 0, -math.pi / 2 + 0.05).R(0, 0.2 * (1 - 2 * k), 0).T(base[0] + 2.0, 0.35, base[2]))
boats = []
for k, (d, dx, yaw) in enumerate(((122.0, 9.0, 0.5), (131.0, 14.0, -0.3))):  # two of her lifeboats, on the bottom near the stern
    bt = lifeboat(); crumple(bt, 0.2, 0.6, 70 + k)
    boats.append(bt.R(0, 0, 0.35 * (1 - 2 * k)).R(0, yaw, 0).T(deck_side + dx, 0.3, zd(d)))
r = np.random.default_rng(71)
spill = []                                                                    # bombs, crates and bottles fallen out of the open holds
for k in range(40):
    d = r.choice([22.0, 40.0, 106.0, 123.0, 135.0]) + r.normal(0, 3)
    b = bomb()
    spill.append(b.R(r.uniform(-0.2, 0.2), r.uniform(0, 6.28), r.uniform(0, 6.28)).T(deck_side + r.uniform(1.0, 9.0), 0.12, zd(d)))
for k in range(14):
    spill.append(crate(r.uniform(0.5, 1.2), r.uniform(0.3, 0.6), r.uniform(0.4, 0.9)).R(0, r.uniform(0, 6.28), r.uniform(-0.3, 0.3)).T(deck_side + r.uniform(1, 12), 0.0, zd(r.uniform(15, 145))))
for k in range(16):
    p = box(r.uniform(1.0, 3.5), 0.12, r.uniform(0.8, 2.5), GREY_L if r.random() < 0.5 else HULLC)
    crumple(p, 0.3, 0.4, 80 + k)
    spill.append(p.R(r.uniform(-0.4, 0.4), r.uniform(0, 6.28), r.uniform(-0.4, 0.4)).T(deck_side + r.uniform(0, 14), 0.2, zd(r.uniform(50, 100))))
debris = merge(spill + [chain])
bottles = []
for k in range(36):
    bt = abyss.bottle(False)
    bottles.append(bt.R(math.pi / 2, r.uniform(0, 6.28), 0).T(deck_side + r.normal(5, 2), 0.05, zd(40.0 + r.normal(0, 2.5))))
bottles = merge(bottles)

# ── Coral: hard corals crust her, soft corals hang under every edge ─────────
CORAL_COLS = [(0.55, 0.48, 0.32), (0.62, 0.56, 0.38), (0.5, 0.42, 0.5), (0.72, 0.5, 0.55), (0.45, 0.5, 0.33), (0.75, 0.68, 0.45)]


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


coral = merge([corals(wreck, 520, 7), corals(fun, 18, 8)] + [corals(m, 8, 9 + k) for k, m in enumerate(masts)])

# ── The ship as built ────────────────────────────────────────────────────────
ship = merge(ship_part(True))
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q2478173" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
sites = [{"steel": [("hull_section", wreck), ("funnel", fun), ("masts", merge(masts), "loose"), ("debris_field", debris, "loose"),
                    ("lifeboats", merge(boats), "loose")],
          "mud": [],
          "plain": [("bottles", bottles, "glass"), ("corals", coral, "life")]}]
dress_wreck(out, sites, depth=-32, kind="cargo", seed=1940, life="tropical")
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[WOOD])
