"""
Osborne Reef (Q1272680) — the failed tyre reef off Fort Lauderdale, Florida, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions and photographs (Florida DEP, Nova Southeastern University, Wikipedia,
Oceana): an artificial reef first built of concrete "jacks", then in 1972 enlarged with one to two million scrap tyres,
tied in bundles and dropped about 1.3 miles off Fort Lauderdale in 60-70 ft (18-21 m). The bundles came apart as their
fastenings corroded; the loose tyres, which corals cannot settle on, were moved about by currents and storms and lie
heaped on one another, in places so thick that the sand is not seen, smothering the natural reef nearby. They cover
more than 34 acres; clean-ups have removed several hundred thousand and some 700,000 remain. Here about 150 m of the
field is drawn: wide heaps of loose tyres piled several deep, bundles still tied lying on their side, tyres strewn on
the sand between the heaps and the old concrete jacks. Every tyre rests on the sand or on the tyres under it (they
are dropped one by one on a height grid), none hangs in the water. The layout and the number of tyres (a sample of
the real count) are approximate.

    python model_sources/Q1272680_osborne_reef.py   -> models/item/Q1272680/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import merge, box, lathe, crumple  # noqa: E402
from scenekit import material, instances, write_model  # noqa: E402

r = np.random.default_rng(1972)
RUBBER = (0.075, 0.075, 0.08)
T = 0.22                                     # thickness of a tyre lying flat (m)


def tyre(R=0.33, t=0.11, seg=9):
    prof = [(R + t * math.cos(a), t * 0.95 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 6)]
    return lathe(prof, RUBBER, seg=seg)


def jack():
    """A concrete 'jack': three square bars crossed at right angles (resting on two of them)."""
    c = (0.62, 0.6, 0.55)
    m = merge([box(3.0, 0.4, 0.4, c, bottom=False), box(0.4, 3.0, 0.4, c, bottom=False), box(0.4, 0.4, 3.0, c, bottom=False)])
    crumple(m, 0.04, 1.5, 3)
    return m


# ── the height grid: what lies on the sand so far ──
CELL, HALF = 0.12, 90.0
N = int(2 * HALF / CELL)
H = np.zeros((N, N), np.float32)


def ij(x, z):
    return int((x + HALF) / CELL), int((z + HALF) / CELL)


def disc(x, z, rad):
    i0, j0 = ij(x - rad, z - rad); i1, j1 = ij(x + rad, z + rad)
    i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1 + 1, N), min(j1 + 1, N)
    I, J = np.mgrid[i0:i1, j0:j1]
    X, Z = I * CELL - HALF + CELL / 2, J * CELL - HALF + CELL / 2
    return (slice(i0, i1), slice(j0, j1)), (X - x) ** 2 + (Z - z) ** 2 <= rad * rad


def h_at(x, z):
    i, j = ij(x, z)
    return float(H[min(max(i, 0), N - 1), min(max(j, 0), N - 1)])


RIM = 0.38
ANG = np.linspace(0, 2 * math.pi, 8, endpoint=False)


def drop(x, z, kind):
    """Drop a tyre at (x, z): it settles on what is under its rim, tilted with it. Returns (pos, up) and raises H."""
    hr = np.array([h_at(x + RIM * math.cos(a), z + RIM * math.sin(a)) for a in ANG])
    hc = h_at(x, z)
    gx = (hr * np.cos(ANG)).sum() / (4 * RIM); gz = (hr * np.sin(ANG)).sum() / (4 * RIM)   # slope of what is under it
    n = np.array([-gx, 1.0, -gz]); n /= np.linalg.norm(n)
    if n[1] < 0.8:                                                       # never steeper than about 37°
        hxz = math.hypot(n[0], n[2]); n = np.array([n[0] / hxz * 0.6, 0.8, n[2] / hxz * 0.6])
    a = r.uniform(0, 6.283); jit = r.uniform(0, 0.12)                    # a little tossed
    n = n + np.array([math.cos(a) * jit, 0, math.sin(a) * jit]); n /= np.linalg.norm(n)
    base = max(0.7 * hr.max() + 0.3 * hr.mean(), hc - 0.05)              # it rests on the highest points under its rim
    y = base + T / 2 * n[1]
    sl, m = disc(x, z, RIM + 0.06)
    H[sl][m] = np.maximum(H[sl][m], y + T / 2 - 0.03)
    return (x, y, z), tuple(n)


pos, ups, yaws, kinds = [], [], [], []

# the concrete jacks at the heart of the reef (first: the tyres fall around and on them)
JP = []
for k in range(28):
    x, z = r.normal(8, 14), r.normal(-4, 12)
    JP.append((x, 0.2, z))
    for dx, dz, rad in ((0, 0, 0.35), (-1.3, 0, 0.3), (1.3, 0, 0.3), (0, -1.3, 0.3), (0, 1.3, 0.3)):
        sl, m = disc(x + dx, z + dz, rad); H[sl][m] = np.maximum(H[sl][m], 0.4)
    sl, m = disc(x, z, 0.3); H[sl][m] = np.maximum(H[sl][m], 1.7)

# bundles still tied: rows of 8-12 tyres standing on their tread, side by side like a roll of coins
for b in range(70):
    bx, bz = r.uniform(-60, 60), r.uniform(-60, 60)
    if math.hypot(bx, bz) > 68:
        continue
    ax = r.uniform(0, math.pi); cx, cz = math.cos(ax), math.sin(ax)
    nb = int(r.integers(8, 13))
    under = max(h_at(bx + cx * (k - nb / 2) * T, bz + cz * (k - nb / 2) * T) for k in range(nb))
    for k in range(nb):
        x, z = bx + cx * (k - nb / 2 + 0.5) * T, bz + cz * (k - nb / 2 + 0.5) * T
        pos.append((x, under + 0.44, z)); ups.append((cx, 0.0, cz)); yaws.append(0.0); kinds.append(1)
        sl, m = disc(x, z, 0.3); H[sl][m] = np.maximum(H[sl][m], under + 0.86)

# the loose tyres: heaps, thick in their middle, and a thinner scatter between them
heaps = []
while len(heaps) < 34:
    x, z = r.uniform(-62, 62), r.uniform(-62, 62)
    if math.hypot(x, z) < 64:
        heaps.append((x, z, r.uniform(3, 8), r.uniform(0.5, 1.6)))
w = np.array([h[3] * h[2] ** 2 for h in heaps]); w /= w.sum()
NT = 17000
for k in range(NT):
    if r.random() < 0.06:                                                # strewn between the heaps
        x, z = r.uniform(-70, 70), r.uniform(-70, 70)
        if math.hypot(x, z) > 72 * (0.85 + 0.15 * math.sin(math.atan2(z, x) * 3)):
            continue
    else:
        hx, hz, sg, _ = heaps[r.choice(len(heaps), p=w)]
        x, z = hx + r.normal(0, sg) * 0.8, hz + r.normal(0, sg) * 0.6
    p, n = drop(x, z, 0)
    pos.append(p); ups.append(n); yaws.append(r.uniform(0, 6.283)); kinds.append(0)

pos, ups, yaws = np.array(pos), np.array(ups), np.array(yaws)
tyres = [tyre(0.30 + 0.035 * k) for k in range(3)]
pick = r.integers(len(tyres), size=len(pos))
INS = []
for k, t in enumerate(tyres):
    s = pick == k
    INS.append((f"tyres_{k}", t, "rubber", instances(pos[s], ups[s], yaw=yaws[s])))
JP = np.array(JP)
INS.append(("concrete_jacks", jack(), "concrete", instances(JP, yaw=r.uniform(0, 6.28, len(JP)))))

out = ROOT / "models" / "item" / "Q1272680" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
mats = {"rubber": material(rough=0.8), "concrete": material(rough=0.95)}
tris, size = write_model(out, [], mats, instanced=INS)
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB, {len(pos)} tyres, highest heap {H.max():.2f} m")
