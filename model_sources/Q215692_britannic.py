"""
HMHS Britannic (Q215692) — the wreck of the hospital ship in the Kea Channel (Aegean Sea), built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The ship: the third Olympic-class liner, the Titanic's younger sister (length 269 m, beam 28.65 m — wider than
her sisters, with a double hull along the engine rooms), finished in 1915 as a hospital ship; she carried
large gantry davits that could launch boats from either side. As she sailed as His Majesty's Hospital Ship:
a white hull with a green band and three red crosses on each side, buff funnels with black tops (the fourth
was a dummy). The hull, decks and superstructure come from the Titanic's model (Q25173_titanic.py), with her
own beam, gantry davits and livery.
She struck a mine on 21 November 1916 and sank in 55 minutes. The wreck lies at about 120 m on her starboard
side, nearly intact: she was longer than the water is deep, so her bow hit the bottom while her stern was still
at the surface and it is crumpled; a large hole opens beneath the forward well deck; the funnels came off as she
rolled and lie apart, the nearest about 50 m from the hull; the gantry davits are still clear to see, those of
the port side (now uppermost) standing, the starboard ones partly crushed into the sediment.
The heading and the places of the funnels are approximate.

    python model_sources/Q215692_britannic.py   -> models/item/Q215692/model.glb (+ model_built.glb)
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, tube, crumple, weather, sstep, fbm  # noqa: E402

spec = importlib.util.spec_from_file_location("olympic", Path(__file__).with_name("Q25173_titanic.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
T.B = 28.65                                                 # beamier than the Titanic (the double hull)
T.rng = np.random.default_rng(1916)
L, B, HA, HB, WL = T.L, T.B, T.HA, T.HB, T.WL
zd = T.zd
GANTRIES = (66.0, 84.0, 172.0, 192.0)                       # gantry davits on each side (distance from the bow)


def gantries(col, crushed_side=None):
    """The big gantry davits: a tall frame at the deck edge with an arm that could swing a boat over either side."""
    P = []
    for d in GANTRIES:
        for s in (-1, 1):
            xi, xo, y0, ht = s * B * 0.30, s * B * 0.47, HA, 11.0
            g = merge([tube([(xi, y0, zd(d) - 1.0), (s * B * 0.40, y0 + ht, zd(d))], 0.28, col, seg=6),
                       tube([(xo, y0, zd(d) + 1.0), (s * B * 0.40, y0 + ht, zd(d))], 0.28, col, seg=6),
                       tube([(xi, y0, zd(d) + 1.0), (s * B * 0.40, y0 + ht, zd(d))], 0.2, col, seg=6),
                       tube([(s * B * 0.40, y0 + ht, zd(d)), (s * B * 0.56, y0 + ht - 1.2, zd(d))], 0.25, col, seg=6),
                       box(1.4, 0.6, 2.6, col).T(s * B * 0.40, y0 + ht - 0.3, zd(d))])
            if crushed_side is not None and s == crushed_side:
                crumple(g, 1.6, 0.4, int(d) + (5 if s > 0 else 0))
                g.v[:, 0] = s * B * 0.30 + (g.v[:, 0] - s * B * 0.30) * 0.6
            P.append(g)
    return P


# ── The wreck ────────────────────────────────────────────────────────────────
P = T.ship_part(0, L, built=False) + [T.rudder_props()] + gantries((0.55, 0.52, 0.46), crushed_side=-1)
wreck = merge(P)
dfb = L / 2 - wreck.v[:, 2]
# the hole beneath the forward well deck, on the port side (the side now facing up)
fv = wreck.v[wreck.f]
fd = L / 2 - fv[:, :, 2].mean(1); fy = fv[:, :, 1].mean(1); fx = fv[:, :, 0].mean(1)
hole = (((fd - 47) / 7.0) ** 2 + ((fy - 11.0) / 4.5) ** 2 < 1 + 0.4 * (fbm(fv.mean(1), 0.35, 7, 2) - 0.5)) & (fx > 0)
wreck.f = wreck.f[~hole]
wreck = merge([wreck, box(3.0, 8.0, 12.0, T.DARK, bottom=False).T(T.hw(zd(47), 11) - 1.6, 11, zd(47))])
dfb = L / 2 - wreck.v[:, 2]
# the bow hit the bottom first: crumpled, shortened and bent towards the keel
k = sstep(42, 0, dfb)
wreck.v[:, 2] -= k * k * 6.0
wreck.v[:, 1] -= k * k * 4.5
crumple(wreck, 2.4, 0.14, 11, k)
crumple(wreck, 0.9, 0.4, 12, sstep(20, 0, dfb))
crumple(wreck, 0.22, 0.08, 13)
# torn funnel bases (the stubs ship_part leaves) and plates peeled around them stay; on her starboard side:
wreck.R(0, 0, math.radians(88))                             # starboard (−x) down
wreck.T(0, -wreck.v[:, 1].min() - 5.0, 0)                   # half her beam's worth of hull in the sediment
weather(wreck, rust=0.45, sediment=0.6, streaks=0.7, seed=21)

# ── The funnels, torn off as she rolled, lying flattened around her (the nearest ~50 m away) ──
r = np.random.default_rng(1915)
fun = []
for k_, (x, z, a) in enumerate(((-52, 70, 0.4), (-68, 25, 1.2), (-85, -35, 2.0), (-60, -80, 2.6))):   # on the deck side (−x)
    f = cyl(3.65, 3.55, 16.0, T.BUFF, seg=24, sx=0.79, caps=False)
    top = cyl(3.55, 3.5, 4.4, T.BLACK, seg=24, sx=0.79, caps=False).T(0, 16.0, 0)
    f = merge([f, top])
    crumple(f, 1.2, 0.18, 300 + k_)
    f.v[:, 0] *= 1.35; f.v[:, 2] *= 0.75                     # flattened on the bottom
    fun.append(f.R(math.pi / 2, a, 0).T(x, 2.2, z))
deb = [merge(fun)]
for i in range(70):                                          # plates, deck fittings, a few lifeboat remains
    x, z = r.normal(-38, 28), r.normal(0, 70)
    m = box(r.uniform(1.5, 6), 0.14, r.uniform(1.2, 4), (0.86, 0.85, 0.8) if r.random() < 0.5 else T.BLACK)
    crumple(m, 0.4, 0.4, 400 + i)
    deb.append(m.R(r.uniform(-0.3, 0.3), r.uniform(0, 6.28), r.uniform(-0.3, 0.3)).T(x, 0.1, z))
debris = merge(deb)
weather(debris, rust=0.5, sediment=0.6, streaks=0.4, seed=61)

# ── The hospital ship as she sailed (model_built.glb) ────────────────────────
WHITE_H, GREEN, CROSS = (0.88, 0.87, 0.83), (0.12, 0.48, 0.24), (0.72, 0.08, 0.07)
T.WHITE = WHITE_H
T.rng = np.random.default_rng(1915)
ship = merge(T.ship_part(0, L, built=True) + [T.rudder_props()] + gantries(WHITE_H))
blk = (np.abs(ship.c - np.array(T.BLACK)).sum(1) < 0.02) & (ship.v[:, 1] < HB + 0.6)      # the black hull -> white
ship.c[blk] = WHITE_H
marks = []
for s in (-1, 1):
    pts = [(s * (T.hw(zd(d), 13.5) + 0.06), 13.5, zd(d)) for d in np.arange(12, 258, 2.0)]   # the green band
    marks.append(tube(pts, 0.55, GREEN, seg=4))
    for d in (70.0, 135.0, 200.0):                                                          # three red crosses a side
        x = s * (T.hw(zd(d), 13.5) + 0.12)
        marks.append(box(0.15, 8.0, 2.6, CROSS, bottom=False).T(x, 14.5, zd(d)))
        marks.append(box(0.15, 2.6, 8.0, CROSS, bottom=False).T(x, 14.5, zd(d)))
ship = merge([ship] + marks)
ship.c = np.clip(ship.c * (0.95 + 0.05 * fbm(ship.v, 0.05, 3)[:, None]), 0, 1)

# ── Write ────────────────────────────────────────────────────────────────────
from dresskit import dress_wreck, dress_built  # noqa: E402
out = ROOT / "models" / "item" / "Q215692" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("hull", wreck), ("debris_field", debris)]}], depth=-120, kind="liner", seed=1916)
dress_built(out.with_name("model_built.glb"), [("main", ship)], deck_colors=[T.DECK])
