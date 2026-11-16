"""
Endurance (Q911858) — Shackleton's ship and her wreck in the Weddell Sea, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: a three-masted barquentine with an auxiliary steam engine, built in 1912 at Framnæs (Sandefjord,
Norway) as Polaris and renamed Endurance for Ernest Shackleton's Imperial Trans-Antarctic Expedition; about
44 m long and 7.6 m in the beam, her oak and greenheart hull made for the ice; square sails on the foremast,
fore-and-aft sails on the main and mizzen, a funnel abaft the mainmast, deckhouses on the deck, the ship's boats,
a black hull with her name in gold letters arched across the counter stern, over a five-pointed star (the
Polaris star). The hull lines, the deckhouses and the rig follow the photographs of Frank Hurley; approximate.
She was beset in the ice of the Weddell Sea in January 1915, crushed by it and abandoned in October, and sank on
21 November 1915; all her men survived. The Endurance22 expedition found the wreck on 5 March 2022 at 3,008 m:
upright, intact and well kept (no wood-eating animals live there); the masts are down and broken, as in
Hurley's last photographs, with the spars, booms and rigging lying over the deck and beside her; the bow is
damaged by the ice; the aft deckhouse (the wardroom) was crushed by the fallen mast; the ship's wheel still
stands on the after deck; ENDURANCE is plain to see across the stern; the rudder lies in the mud beneath it.
The heading and the places of the fallen spars are approximate.

    python model_sources/Q911858_endurance.py      -> models/item/Q911858/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, tube, crumple, fbm, sstep  # noqa: E402
from sailkit import Ship, Mast, solidify, frames, lettering, star, wheel  # noqa: E402

rng = np.random.default_rng(1915)
BLACK = (0.07, 0.07, 0.075)
GOLD = (0.78, 0.6, 0.22)
HOUSE = (0.62, 0.5, 0.35)
SPAR = (0.55, 0.42, 0.26)


def ship():
    return Ship(L=43.9, B=7.6, D=5.6, draft=4.4, fore=(6.5, 1.6), aft=(9.0, 1.3), sheer=1.3, tumble=0.04, castle_tumble=0.0,
                bluff=0.35, transom=0.5, bowsprit=10.0, spritsail=False, jibs=3, yard=1.9,
                masts=[Mast(0.78, 30, "square", 4), Mast(0.5, 31, "gaff"), Mast(0.25, 26, "gaff")],
                hull_col=BLACK, upper_col=BLACK, wale_col=(0.6, 0.48, 0.2), deck_col=(0.62, 0.55, 0.42),
                spar_col=SPAR, sail_col=(0.84, 0.8, 0.7),
                deckhouses=[(0.56, 0.66, 3.6, 2.2, HOUSE), (0.12, 0.22, 4.2, 2.0, HOUSE)],
                funnel=(0.43, 0.45, 3.4, (0.66, 0.48, 0.24)))


S = ship()
L = S.L


def stern_marks():
    """ENDURANCE in gold letters arched across the stern, the star below."""
    y = S.top(0.0) - 1.1
    z = S.zt(0.0) - 0.02
    name = lettering("ENDURANCE", 0.36, GOLD, arch=0.18).T(0, y - 0.45, z)
    st = star(0.32, GOLD).T(0, y - 1.05, z)
    return merge([name, st])


def boat(L_=6.9, B_=1.9):
    b = Ship(L=L_, B=B_, D=0.9, draft=0.5, sheer=0.3, tumble=0.0, castle_tumble=0.0, bluff=0.3, transom=0.05, rails=False,
             hull_col=(0.78, 0.76, 0.7), upper_col=(0.78, 0.76, 0.7))
    h = b._hull()
    seat = merge([box(B_ * 0.8, 0.05, 0.25, SPAR).T(0, 0.6, z) for z in (-1.5, 0.0, 1.5)])
    return merge([h, seat])


def boats():
    out = []
    for s, t in ((-1, 0.34), (1, 0.34), (1, 0.61)):
        y = S.deck(t)
        out.append(boat().T(s * 1.9 if t < 0.5 else 0.0, y + 2.25 if t > 0.5 else y + 0.35, S.zt(t)))
    return out


def the_wheel():
    t = 0.07
    return wheel().T(0, S.deck(t) + S.castle(t), S.zt(t))


# ── The wreck ────────────────────────────────────────────────────────────────
hull = S._hull()
det, _ = S._details(False)
deck = S._decks()
stumps, _, _ = S._rig(False)
ribs = frames(S, np.arange(0.84, 0.97, 0.6 / L), col=(0.3, 0.22, 0.15), size=0.3, inset=0.3, upto=0.95)

# the rudder was torn off: it lies in the mud under the stern
fcd = det.v[det.f].mean(1)
ytl = S.deck(0.0) + 0.5
is_tiller = (np.abs(fcd[:, 0]) < 0.2) & (np.abs(fcd[:, 1] - (ytl - 0.05)) < 0.2) & (fcd[:, 2] < S.zt(0) + 2.2)
is_rud = ((fcd[:, 2] < S.zt(0) - 0.05) & (np.abs(fcd[:, 0]) < 0.4)) | is_tiller
u_, inv_ = np.unique(det.f[is_rud], return_inverse=True)
rudder = Mesh(det.v[u_], inv_.reshape(-1, 3), det.c[u_])
det = Mesh(det.v, det.f[~is_rud], det.c)
# the ice stove in the bow: planking gone from part of the starboard bow, the ribs showing
fch = hull.v[hull.f].mean(1)
th = S.tz(fch[:, 2])
hole = (th > 0.86) & (th < 0.95) & (fch[:, 0] < -0.2) & (np.abs(fch[:, 1] - 5.0) < 1.3 + 0.6 * (fbm(fch, 0.6, 3, 2) - 0.5))
hull = Mesh(hull.v, hull.f[~hole], hull.c)
hull = solidify(hull, 0.25, (0.3, 0.22, 0.15))
ribs.f = ribs.f[ribs.v[ribs.f].mean(1)[:, 0] < 0]             # only the starboard ones, behind the hole
# the aft deckhouse (the wardroom), crushed by the fallen mizzen
fdd = det.v[det.f].mean(1)
house = (S.tz(fdd[:, 2]) > 0.11) & (S.tz(fdd[:, 2]) < 0.23) & (fdd[:, 1] > S.deck(0.17) + S.castle(0.17) - 0.05) & (np.abs(fdd[:, 0]) < 2.4)
hv = np.unique(det.f[house])
y0 = S.deck(0.17) + S.castle(0.17)
det.v[hv, 1] = y0 + (det.v[hv, 1] - y0) * 0.35
wreck = merge([hull, det, deck, stumps, ribs, stern_marks(), the_wheel()])
dfb = L / 2 - wreck.v[:, 2]
crumple(wreck, 0.35, 0.5, 11, sstep(8, 0, dfb))               # the bow pushed about by the ice
crumple(wreck, 0.05, 1.2, 12)
crumple(wreck, 0.3, 0.6, 13, (np.abs(wreck.v[:, 2] - S.zt(0.17)) < 3.5) * (wreck.v[:, 1] > y0 - 0.2))
# the paint: much of the black is still there, worn through to the wood here and there
lum = wreck.c.mean(1)
worn = sstep(0.55, 0.75, fbm(wreck.v, 0.4, 21, 3)) * (lum < 0.12)
wreck.c = wreck.c * (1 - worn[:, None]) + np.array([0.3, 0.24, 0.17]) * worn[:, None]
wreck.R(0, 0, math.radians(-2.5)).R(math.radians(-1.2), 0, 0)
wreck.T(0, -2.3, 0)                                           # sitting upright, sunk in the mud to below her waterline

# ── The masts and spars, down over the deck and beside her ──
sp = []


def spar(length, r, a, b_, broken=True):
    """A spar of that length lying from point a towards b_, its far end splintered."""
    a, b_ = np.array(a, float), np.array(b_, float)
    d = (b_ - a) / np.linalg.norm(b_ - a)
    pts = [a + d * length * k / 6 for k in range(7)]
    m = tube([tuple(p) for p in pts], r, SPAR, seg=8)
    if broken:
        crumple(m, 0.06, 2.0, int(length * 10))
    return m


yd = S.deck(0.5) - 2.3
sp.append(spar(17, 0.32, (0, yd + 2.6, S.zt(0.78)), (-6, -0.2, S.zt(0.78) + 13)))     # the foremast, over the starboard bow
sp.append(spar(11, 0.22, (-3.8, yd + 1.6, S.zt(0.82)), (-12, 0.2, S.zt(0.8))))        # its topmast, on the bottom
sp.append(spar(18, 0.3, (0, yd + 2.5, S.zt(0.5)), (2.5, yd + 1.2, S.zt(0.5) - 16)))   # the mainmast, aft along the deck
sp.append(spar(15, 0.27, (0, yd + 2.4, S.zt(0.25)), (5.5, -0.1, S.zt(0.25) - 10)))    # the mizzen, across the wardroom and over the port quarter
for i, (x, z, a, ln) in enumerate(((-6.5, 2, 0.2, 14), (-7.5, -6, -0.1, 12), (-5.5, 9, 0.4, 10.5), (-8.5, 14, 0.1, 8.5),
                                   (6.0, -8, 0.3, 9), (6.8, 4, -0.2, 7))):
    c = np.array([x, 0.2, z])
    d = np.array([math.sin(a), 0, math.cos(a)])
    sp.append(spar(ln, 0.17, c - d * ln / 2, c + d * ln / 2))                      # yards, booms and gaffs
for i in range(9):                                                                  # the rigging, slack over the rail and the mud
    x0 = rng.choice([-1, 1]) * S.B * 0.48
    z0 = rng.uniform(S.zt(0.15), S.zt(0.9))
    pts = [(x0, yd + 0.9, z0)]
    for k in range(1, 7):
        pts.append((x0 + np.sign(x0) * k * 1.1, max(0.06, yd + 0.9 - k * 1.5), z0 + 0.35 * k * k * (1 if i % 2 else -1)))
    sp.append(tube(pts, 0.04, (0.22, 0.18, 0.13), seg=3))
rudder.v = rudder.v - rudder.v.mean(0)
rudder.R(0, 0, math.pi / 2 - 0.12).R(0, 0.5, 0).T(-1.0, 0.25, S.zt(0) - 3.2)
spars = merge(sp + [rudder])
spars.c = np.clip(spars.c * 0.75, 0, 1)

# ── Endurance as she sailed (model_built.glb) ────────────────────────────────
P = S.build(built=True)
marks = merge([stern_marks(), the_wheel()] + boats())

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q911858" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull", wreck), ("debris_field", spars)]}], depth=-3008, kind="expedition", seed=1915, wood=True)
dress_built(out.with_name("model_built.glb"),
            [("hull", merge([P["hull"], P["deck"], marks]), "wood"), ("rig", P["spars"], "wood"), ("rigging", P["ropes"], "rope"),
             ("sails", P["sails"], "cloth")], deck_colors=[S.deck_col])
