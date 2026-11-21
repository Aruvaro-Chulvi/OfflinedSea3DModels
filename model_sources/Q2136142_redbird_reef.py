"""
Redbird Reef (Q2136142) — the artificial reef of New York subway cars off Delaware, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Delaware DNREC, Wikipedia, CBC, Scientific American): reef site 11 of
the Delaware artificial reef system, about 16 miles off the coast in some 80 ft (24 m) of water, covering about
1.3 square miles. From 2001 to 2007 997 retired New York City "Redbird" subway cars, stripped of their wheels,
doors, windows and fittings, were dropped from barges, with 86 old Army tanks, tugboats and barges, a fishing
trawler, a Chesapeake Bay cruise ship and, in 2024, a tug and the Mayor J. Harold Grady; lying where they fell,
some on their sides. On the flat sandy bottom where there was little to cling to, the steel is now crusted with
blue mussels, sponges and anemones, and black sea bass and tautog crowd about them. Here a patch about 400 m across
is drawn with a sample of the cars, a few tanks and a tug; the layout is approximate.

    python model_sources/Q2136142_redbird_reef.py   -> models/item/Q2136142/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, quad, tube, crumple, sstep, fbm  # noqa: E402
import abyss  # noqa: E402

rng = np.random.default_rng(2001)
RED, DARK, GREY, OLIVE = (0.55, 0.12, 0.1), (0.03, 0.03, 0.035), (0.35, 0.36, 0.37), (0.3, 0.32, 0.22)


def subway_car(seed):
    """A Redbird car stripped for the reef: a box body with its window and door openings, no wheels."""
    rr = np.random.default_rng(seed)
    body = box(3.0, 3.5, 15.6, RED).T(0, 0.3, 0)
    body.v[:, 1] += 0.15 * (body.v[:, 1] > 3.6) * (1 - (body.v[:, 0] / 1.5) ** 2)
    P = [body, box(2.6, 0.3, 15.0, GREY).T(0, 0.0, 0)]
    for s in (-1, 1):
        P += [quad(s * 1.51, 2.3, z, 1.1, 0.9, "+x" if s > 0 else "-x", DARK) for z in np.arange(-6.6, 6.7, 1.6)]
        P += [quad(s * 1.51, 1.6, z, 1.3, 2.1, "+x" if s > 0 else "-x", DARK) for z in (-5.2, -1.7, 1.7, 5.2)]
    P += [quad(0, 2.2, 7.81, 1.2, 1.4, "+z", DARK), quad(0, 2.2, -7.81, 1.2, 1.4, "-z", DARK)]
    m = merge(P)
    crumple(m, 0.12, 0.4, seed)
    m.c = np.clip(m.c * (0.75 + 0.35 * fbm(m.v, 0.3, seed + 1)[:, None]), 0, 1)
    return m


def tank(seed):
    """An M60 tank hull and turret, stripped."""
    P = [box(3.6, 1.4, 6.9, OLIVE).T(0, 0.5, 0), box(3.4, 0.9, 2.2, OLIVE).T(0, 1.9, -0.4),
         cyl(0.12, 0.1, 6.0, OLIVE, seg=6).R(math.pi / 2, 0, 0).T(0, 2.4, 0.8)]
    for s in (-1, 1):
        P.append(box(0.6, 1.0, 6.6, DARK).T(s * 1.75, 0.0, 0))
    m = merge(P); crumple(m, 0.06, 0.6, seed)
    return m


def tug():
    hull = lathe([(0.05, -11.5), (2.4, -10.0), (3.6, -5.0), (3.8, 0.0), (3.6, 5.0), (2.4, 9.0), (0.05, 11.0)], (0.15, 0.15, 0.17), seg=12, axis="z")
    hull.v[:, 1] = np.minimum(hull.v[:, 1], 0.4) + 1.8
    P = [hull, box(5.5, 2.6, 7.0, (0.8, 0.78, 0.72)).T(0, 2.2, 1.0), box(4.0, 2.2, 3.5, (0.8, 0.78, 0.72)).T(0, 4.8, 2.5),
         cyl(0.7, 0.6, 3.0, (0.6, 0.15, 0.1), seg=10).T(0, 4.8, -1.5)]
    m = merge(P); crumple(m, 0.08, 0.4, 5)
    return m


r = np.random.default_rng(7)
cars, tanks = [], []
for k in range(110):                                                          # dropped from barges in loose rows, some on their sides
    row = k // 12
    x = -180 + row * 38 + r.normal(0, 6)
    z = -150 + (k % 12) * 26 + r.normal(0, 7)
    c = subway_car(100 + k)
    roll = math.pi / 2 if r.random() < 0.18 else r.normal(0, 0.08)
    c.R(r.normal(0, 0.03), r.uniform(-0.5, 0.5) + (math.pi / 2 if row % 2 else 0), roll)
    c.T(x, 1.6 if abs(roll) > 1 else -0.3, z)
    cars.append(c)
for k in range(8):
    t = tank(300 + k)
    tanks.append(t.R(0, r.uniform(0, 6.28), 0).T(r.uniform(-60, 60) + 140, -0.2, r.uniform(-80, 80)))
tugboat = tug().R(0, 0, math.radians(18)).R(0, 0.7, 0).T(160, -0.6, 130)
cars_m = merge(cars)
tanks_m = merge(tanks)
mussels = abyss.blob(0.25, (0.12, 0.13, 0.2), 9, sub=1, amp=0.5)
mus = []
for k, (p, n) in enumerate(abyss.sample_surface(cars_m, 2500, lambda fn: fn[:, 1] > -0.3, r)):
    if k > 1600:
        break
    m = mussels.copy(); m.v[:, 1] *= 0.4
    up = np.array([0, 1.0, 0]); ax = np.cross(up, n); s_ = np.linalg.norm(ax); co = float(np.dot(up, n))
    if s_ > 1e-6:
        ax = ax / s_; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        m.v = m.v @ (np.eye(3) + s_ * K + (1 - co) * K @ K).T
    mus.append(m.S(r.uniform(0.6, 1.6)).T(*p))
mussel_beds = merge(mus)

from dresskit import dress_wreck  # noqa: E402
out = ROOT / "models" / "item" / "Q2136142" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
dress_wreck(out, [{"steel": [("subway_cars", cars_m, "loose"), ("tanks", tanks_m, "loose"), ("tug_section", tugboat)], "mud": [],
                   "plain": [("mussels", mussel_beds, "life")]}], depth=-24, kind="cargo", seed=2001, drift=0.4)
