"""
Kormoran (Q708046) — the wreck of the German auxiliary cruiser off Western Australia, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the HAPAG motor ship Steiermark (1938), converted into the raider Kormoran: length 164 m, beam
20.2 m; raised forecastle, six holds, the bridge and accommodation amidships with one funnel; six 15 cm guns
hidden behind flaps (two under the forecastle, two in the holds on lifts…), torpedo tubes above and below the
water, anti-aircraft guns on raised mounts, some 320 mines in the after holds. On 19 November 1941 she was
disguised as the Dutch freighter Straat Malakka.

The wreck, after the Finding Sydney Foundation's final report (2008) and its search diary: scuttled after the
fight with HMAS Sydney, she was torn apart when her mines blew up. At about 2,560 m, the forward part of the
ship — some 90 m from the stem to just aft of the bridge — sits upright, oriented SSW/NNE with the bow to the
north, up to 20 m high: the raised, flared forecastle with the starboard 15 cm gun behind its flap, the opening
of hold 1, a 15 cm gun in hold 2 trained 135° to starboard, the front of the bridge superstructure, a raised
2 cm gun with its flaps on the port side, the torpedo flaps. About 1.4 km to the north-north-east lies debris
field #1, nearly 1 km across (twisted plates and frames, wire, the starboard anchor in a piece of the ship's
side), around a large piece of superstructure about 50 × 20 × 15 m; some 2.1 km north, debris field #2 with a
few pieces, the largest about 35 × 15 × 15 m. What those two large pieces are is not known (perhaps the engine
room and the stern): here the first is the after hull with the engine casing, the second the stern on its
side. The heading, the shape of the after pieces and the spread of the debris are approximate. A war grave.

    python model_sources/Q708046_kormoran.py   -> models/item/Q708046/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, quad, tube, crumple, sstep, fbm, yaw_quat  # noqa: E402
from shipkit import install  # noqa: E402

rng = np.random.default_rng(1941)

# ── The ship ─────────────────────────────────────────────────────────────────
L, B, D = 164.0, 20.2, 12.0                # length, beam, depth
FC = 2.6
TW, TT = 7.2, 1.3
WL = 7.5
HULLC = (0.09, 0.09, 0.1)                  # black hull (disguised as a Dutch freighter)
GREY, GREY_L, GREY_D = (0.6, 0.53, 0.4), (0.7, 0.66, 0.55), (0.3, 0.3, 0.31)   # buff masts, light superstructure
RED, DECKC, WOOD = (0.42, 0.13, 0.09), (0.33, 0.3, 0.27), (0.5, 0.4, 0.27)
BLACK, DARK, STEEL, BRONZE = (0.07, 0.07, 0.08), (0.03, 0.03, 0.035), (0.4, 0.4, 0.4), (0.55, 0.4, 0.2)
TARP, RUBBER = (0.3, 0.29, 0.2), (0.06, 0.06, 0.06)
GUNC = (0.42, 0.44, 0.45)

HHW = 4.2
HATCHES = [(17, 28), (31, 44), (47, 57), (86, 97), (100, 111), (114, 125)]
BULKHEADS = [15, 29.5, 45.5, 58.5, 84, 98.5, 112.5, 126.5]
RAISED = [(0, 15), (58.5, 84), (138, L)]
ISLAND = (58.5, 84)


def w_of(t):                                # a fast cargo motor ship: fine entrance, cruiser stern
    w = 1.0
    if t > 0.76:
        w = math.sin(((1 - t) / 0.24) * math.pi / 2) ** 0.7
    if t < 0.24:
        w = min(w, 0.22 + 0.78 * math.sin((t / 0.24) * math.pi / 2) ** 0.8)
    return max(w, 0.004)


def yb_of(t):
    y = 0.0
    if t < 0.1:
        y = D * 0.5 * ((0.1 - t) / 0.1)
    if t > 0.985:
        y = max(y, D * 0.6 * ((t - 0.985) / 0.015) ** 2)
    return y


def yd_of(t):
    s = (t - 0.5) * 2
    return D + (2.4 if s > 0 else 1.0) * s * s


def k_of(t):
    return 2.5 + 7.0 * w_of(t) ** 4


# ── Fittings ─────────────────────────────────────────────────────────────────


# ── Cargo ────────────────────────────────────────────────────────────────────


install(globals(), "zd t_of hw deck_y half at stations hull strip side_strips section face portholes anchor windlass chains bollards ventilator hatch_coaming mast lifeboat davit propeller rudder wheel crate tyre place")


# ── Guns ─────────────────────────────────────────────────────────────────────
def gun15(train=0.0):
    """A 15 cm (5.9-inch) gun on a pedestal with its shield; barrel to +Z before training."""
    P = [cyl(0.6, 0.75, 1.1, GUNC, seg=12), box(1.0, 0.8, 2.0, GUNC).T(0, 1.1, 0.2),
         box(2.6, 1.8, 0.12, GUNC).T(0, 0.9, 1.4), cyl(0.12, 0.1, 6.5, STEEL, seg=8).R(math.pi / 2 - 0.05, 0, 0).T(0, 1.55, 1.0)]
    return merge(P).R(0, train, 0)


def aa20(raised=True):
    """A 2 cm anti-aircraft gun on its raised mount, the concealment flaps folded open."""
    P = [cyl(0.9, 0.9, 0.15, GUNC, seg=10).T(0, 2.0 if raised else 0, 0), cyl(0.25, 0.25, 2.0 if raised else 0.3, GUNC, seg=8),
         box(0.5, 0.4, 0.8, GUNC).T(0, 2.2, 0), cyl(0.04, 0.04, 2.2, STEEL, seg=5).R(math.pi / 2 - 0.6, 0, 0).T(0, 2.45, 0.2)]
    for s in (-1, 1):
        P.append(box(0.06, 1.6, 2.2, GREY_L).R(0, 0, s * 1.2).T(s * 1.6, 1.3, 0))
    return merge(P)


# ── The ship, piece by piece ─────────────────────────────────────────────────
def ship_part(d0, d1, built):
    inr = lambda a, b=None: (d0 <= a <= d1) if b is None else (a < d1 and b > d0)
    clip = lambda a, b: (max(a, d0), min(b, d1))
    P = [hull(d0, d1)]
    holes = [(a, b, HHW) for a, b in HATCHES]
    for a, b in ((15, 58.5), (84, 138)):
        a2, b2 = clip(a, b)
        if b2 - a2 > 0.3:
            P.append(strip(a2, b2, deck_y, half, holes, DECKC))
            P.append(side_strips(a2, b2, lambda d: deck_y(d), lambda d: deck_y(d) + 1.1, HULLC))
    for a, b in RAISED:
        a2, b2 = clip(a, b)
        if b2 - a2 > 0.3:
            P.append(strip(a2, b2, lambda d: deck_y(d) + FC, half, [], DECKC))
            P.append(side_strips(a2, b2, lambda d: deck_y(d), lambda d: deck_y(d) + FC, HULLC))
            P.append(side_strips(a2, b2, lambda d: deck_y(d) + FC, lambda d: deck_y(d) + FC + 1.1, HULLC))
            if a > 0 and inr(a):
                P += face(a, deck_y(a), deck_y(a) + FC, GREY_L, doors=(-3.5, 3.5), sign=1)
            if b < L and inr(b):
                P += face(b, deck_y(b), deck_y(b) + FC, GREY_L, doors=(-3.5, 3.5), sign=-1)
    for a, b in ((15, 58.5), (84, 126.5)):
        a2, b2 = clip(a, b)
        if b2 - a2 > 0.3:
            P.append(strip(a2, b2, lambda d: TW, lambda d: hw(zd(d), TW) - 0.05, holes, GREY_D))
            P.append(strip(a2, b2, lambda d: TT, lambda d: hw(zd(d), TT) - 0.05, [], (0.24, 0.22, 0.2)))
    for bh in BULKHEADS:
        if d0 + 0.2 < bh < d1 - 0.2:
            P.append(section(bh, deck_y(bh), GREY_D))
    # forecastle: windlass, anchors, the two 15 cm guns behind flaps in the bulwark
    if inr(0, 15):
        P += windlass(6) + chains(6, built)
        for s in (-1, 1):
            z = zd(3.4); y = deck_y(3.4) - 0.5
            if built or s > 0:                                       # the starboard anchor lies in debris field #1
                P.append(anchor().T(s * (hw(z, y) + 0.25), y, z))
        y = deck_y(10) + FC
        for s in (-1, 1):
            open_ = (not built) and s < 0                            # the starboard gun with its flap down
            P.append(at(gun15(-math.pi / 2 if open_ else 0.0), 10, x=s * 5.2, y=y))
            flap = box(0.12, 1.1, 4.0, HULLC).T(0, 0, 0)
            if open_:
                flap.R(0, 0, -1.4)
            P.append(at(flap, 10, x=s * (half(10) + (0.6 if open_ else 0.0)), y=y + (0.0 if open_ else 0.0)))
    # hatches; the gun on its lift in hold 2
    for k, (a, b) in enumerate(HATCHES):
        if a >= d0 - 0.1 and b <= d1 + 0.1:
            P += hatch_coaming(a, b, built, wreck_beams=(0 if k < 2 else 1))
    if inr(37.5) and not built:
        P.append(at(box(5.0, 0.3, 5.0, GUNC), 37.5, y=TW + 2.6))                            # lift platform
        P.append(at(gun15(-math.radians(135)), 37.5, y=TW + 2.9))
    if inr(29.5):
        P += mast(29.5, 22.0 if built else 3.0, built)
    if built and inr(112.5):
        P += mast(112.5, 20.0, built)
    # the island: accommodation, bridge, funnel, boats; the raised 2 cm gun on the port side
    if inr(58.5, 84) and d0 <= 59:
        y0 = deck_y(70) + FC
        A = box(B * 0.8, 2.6, 24.0, GREY_L).T(0, y0, zd(71.5))
        Bh = box(B * 0.6, 2.5, 18.0, GREY_L).T(0, y0 + 2.6, zd(70.5))
        C = box(B * 0.78, 2.5, 4.5, GREY_L).T(0, y0 + 5.1, zd(62.0))
        win = [quad(x, y0 + 6.5, zd(59.75) + 0.02, 0.9, 0.8, "+z", DARK) for x in np.arange(-6.5, 6.6, 1.3)]
        win += [quad(s * B * 0.4, y0 + 1.4, zd(d), 0.5, 0.5, "+x" if s > 0 else "-x", DARK) for s in (-1, 1) for d in np.arange(61, 83, 1.6)]
        wings = box(B * 1.0, 0.25, 2.6, GREY_L).T(0, y0 + 7.6, zd(61.0))
        if built:
            P += [A, Bh, C, wings] + win
            fun = cyl(2.9, 2.8, 8.0, GREY, seg=20, sx=0.7).R(-0.06, 0, 0)
            fun.c[fun.v[:, 1] > 6.6] = BLACK
            P.append(fun.T(0, y0 + 5.1, zd(75.0)))
            for d in (68.0, 78.0):
                for s in (-1, 1):
                    P.append(lifeboat().T(s * (B * 0.3 + 1.2), y0 + 5.9, zd(d)))
                    P += [davit(s * (B * 0.3 + 0.25), y0 + 5.1, zd(d + dd), s) for dd in (-2.6, 2.6)]
        else:                                                         # the front of the bridge intact, torn aft
            hm = merge([A, Bh, C, wings] + win)
            crumple(hm, 0.25, 0.35, 61)
            crumple(hm, 1.2, 0.3, 62, sstep(zd(72), zd(80), hm.v[:, 2]))
            hm.f = hm.f[hm.v[hm.f][:, :, 2].mean(1) > zd(81)]
            P.append(hm)
        P.append(at(aa20(), 64.0, x=B * 0.39, y=y0 + 5.1))
    # torpedo flaps: underwater tube on the starboard side, above-water tubes on the port side
    if inr(50):
        P.append(quad(-(hw(zd(50), 3.5) + 0.04), 3.5, zd(50), 1.4, 1.2, "-x", DARK))
    if inr(82):
        P.append(quad(half(82) + 0.04, deck_y(82) - 1.4, zd(82), 2.2, 1.4, "+x", DARK))
    # after accommodation, poop, screw
    if inr(127, 138) and d0 <= 128:
        P.append(at(box(B * 0.66, 2.5, 10.0, GREY_L), 133, y=deck_y(133)))
    if inr(138, L) and d0 <= 139:
        P.append(at(box(6.0, 2.3, 5.0, GREY_L), 150, y=deck_y(150) + FC))
        P += bollards((143.0, 157.0), FC, inset=1.2)
    if inr(L - 4):
        P += propeller(L - 5.0, 2.6, 2.4) + rudder(L - 1.5, 0.8)
    return P


# ── Where things are ─────────────────────────────────────────────────────────
HEAD = 20.0                                                                  # SSW/NNE, bow to the north (approx.)
M_LAT, M_LON = 30.87, 30.87 * math.cos(math.radians(26.1))


def loc(dn_s, de_s):
    n, e = dn_s * M_LAT, de_s * M_LON
    h = math.radians(HEAD)
    return n * math.sin(h) - e * math.cos(h), n * math.cos(h) + e * math.sin(h)


def plates(n, cx, cz, spread, seed, cols=(HULLC, GREY_L, DECKC)):
    rr = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = box(rr.uniform(1.0, 6.0), 0.14, rr.uniform(0.8, 3.5), cols[rr.integers(len(cols))])
        crumple(p, 0.5, 0.35, seed + i)
        out.append(p.R(rr.uniform(-0.6, 0.6), rr.uniform(0, 6.28), rr.uniform(-0.6, 0.6)).T(cx + rr.normal(0, spread), 0.25, cz + rr.normal(0, spread)))
        if rr.random() < 0.25:                                                # a bent frame
            out.append(tube([(0, 0, 0), (rr.uniform(-1, 1), rr.uniform(1, 3.5), rr.uniform(-1, 1)), (rr.uniform(-2, 2), rr.uniform(0.3, 2.5), rr.uniform(-3, 3))], 0.12, GREY_D, seg=4)
                       .T(cx + rr.normal(0, spread), 0, cz + rr.normal(0, spread)))
    return out


# ── The forward part: stem to just aft of the bridge, upright ───────────────
BREAK = 90.0
fwd = merge(ship_part(0, BREAK, False))
dfb = L / 2 - fwd.v[:, 2]
crumple(fwd, 0.15, 0.12, 11)
crumple(fwd, 1.8, 0.3, 12, sstep(82, BREAK, dfb))
fwd.T(0, 0, -zd(BREAK / 2)).T(0, -0.8, 0)
near = plates(40, 0, -BREAK / 2 - 25, 14, 100)
debris_fwd = merge(near)

# ── Debris field #1: the after hull with the engine casing, plates, wire, the starboard anchor ──
D1_N, D1_E = 46, 12                                                          # arc-seconds from the forward part's centre
aft = merge(ship_part(BREAK + 4, 132, False) + [box(B * 0.55, 6.0, 22.0, GREY_L).T(0, D, zd(108))])   # the engine casing standing up
dfa = L / 2 - aft.v[:, 2]
crumple(aft, 0.3, 0.15, 21)
crumple(aft, 2.2, 0.25, 22, np.maximum(sstep(BREAK + 12, BREAK + 4, dfa), sstep(124, 132, dfa)))
aft.v[:, 1] *= 0.82
aft.T(0, 0, -zd((BREAK + 4 + 132) / 2)).R(0, 0.5, 0.06).T(0, -1.0, 0)
f1 = plates(150, 0, 0, 70, 200) + plates(60, 30, -60, 110, 201)
wire = tube([(math.cos(a) * (3 + a * 0.3), 0.2 + 0.15 * math.sin(a * 3), math.sin(a) * (3 + a * 0.3)) for a in np.linspace(0, 18, 60)], 0.05, BLACK, seg=4)
f1.append(wire.T(18, 0, 25))
sideplate = box(7.0, 0.2, 5.0, HULLC); crumple(sideplate, 0.4, 0.3, 210)
f1.append(merge([sideplate, anchor().R(math.pi / 2, 0, 0).T(0, 0.35, 0)]).R(0.15, 1.0, 0).T(-22, 0.2, 14))
debris1 = merge(f1)

# ── Debris field #2: the stern, on its side ─────────────────────────────────
D2_N, D2_E = 69, -6
stern = merge(ship_part(132, L, False))
crumple(stern, 0.25, 0.15, 31)
crumple(stern, 1.8, 0.3, 32, sstep(138, 132, L / 2 - stern.v[:, 2]))
stern.T(0, 0, -zd((132 + L) / 2)).R(0, 0, 1.45)
stern.T(0, -np.percentile(stern.v[:, 1], 0.5) - 1.0, 0).R(0, -0.7, 0)
debris2 = merge(plates(30, 0, 0, 30, 300))

# ── The ship as built ────────────────────────────────────────────────────────
ship = merge(ship_part(0, L, True))
ship.c = np.clip(ship.c * (0.94 + 0.06 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)
ship.T(0, 0, -zd(L / 2))

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q708046" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
x1, z1 = loc(D1_N, D1_E)
x2, z2 = loc(D2_N, D2_E)
sites = [{"steel": [("forward_section", fwd), ("debris_field", debris_fwd, "loose")], "mud": [], "plain": []},
         {"steel": [("after_hull", aft), ("debris_1", debris1, "loose")], "mud": [], "plain": [],
          "piece": ("debris_field_1", {"translation": [x1, 0, z1]})},
         {"steel": [("stern", stern), ("debris_2", debris2, "loose")], "mud": [], "plain": [],
          "piece": ("debris_field_2", {"translation": [x2, 0, z2]})}]
dress_wreck(out, sites, depth=-2560, kind="warship", seed=1941, memorial=True, drift=0.35)
dress_built(out.with_name("model_built.glb"), [("main", ship)])
