"""
Lost City Hydrothermal Field (Q521459) — the white carbonate towers on the Atlantis Massif, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Kelley et al. 2001, 2005, and the expeditions since): on the top of
the Atlantis Massif, a mountain of mantle rock beside the Mid-Atlantic Ridge, at about 750-900 m. Unlike black
smokers, its vents are driven by the reaction of sea water with the mantle rock (serpentinisation): warm (40-90 °C),
alkaline fluids rich in hydrogen and methane build chimneys of white carbonate and brucite. Some thirty large
chimneys and many small ones stand on a terrace about 400 m across; the largest, "Poseidon", rises about 60 m
from a massive base with several spires; others are slender pinnacles; delicate flanges grow out from their sides,
with shimmering warm fluid pooling under them. The towers are white to cream where active, grey where extinct.
Life: snails, amphipods, polychaetes and crabs, and a few fish; bacterial films on the active surfaces. Here the
field is drawn about 300 m across; the place of each tower is approximate.

    python model_sources/Q521459_lost_city.py   -> models/item/Q521459/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, cyl, lathe, grid, crumple, sstep, fbm  # noqa: E402
from scenekit import material, rust_textures, seabed_textures, bake_ao, instances, write_model  # noqa: E402
import abyss  # noqa: E402

rng = np.random.default_rng(2000)
WHITE, CREAM, GREY = np.array([0.9, 0.89, 0.84]), np.array([0.84, 0.8, 0.68]), np.array([0.52, 0.52, 0.5])


def chimney(h, r, seed, active=True, spires=0):
    """A carbonate chimney: a knobbly tapering column with flanges, white where active."""
    rr = np.random.default_rng(seed)
    n = max(6, int(h / 1.5))
    prof = []
    for k in range(n + 1):
        t = k / n
        rad = r * (1 - 0.82 * t ** 0.8) * (0.85 + 0.3 * rr.random())
        prof.append((max(rad, 0.15), h * t))
    prof.append((0.05, h + 0.3))
    m = lathe(prof, WHITE, seg=14)
    crumple(m, r * 0.18, 0.4 / max(r, 0.5), seed)
    P = [m]
    for k in range(int(h / 6)):                                               # flanges: ledges growing out of the sides
        y = rr.uniform(0.15, 0.85) * h
        ro = r * (1 - 0.8 * (y / h) ** 0.8)
        fl = cyl(ro + rr.uniform(0.6, 2.2), ro * 0.9, 0.35, WHITE, seg=12).T(0, y, 0)
        crumple(fl, 0.25, 0.8, seed + 50 + k)
        P.append(fl.T(rr.normal(0, 0.3), 0, rr.normal(0, 0.3)))
    for k in range(spires):                                                   # spires on a big base
        a = rr.uniform(0, 6.28); d = r * rr.uniform(0.3, 0.6)
        P.append(chimney(h * rr.uniform(0.35, 0.6), r * 0.35, seed + 100 + k, active).T(math.cos(a) * d, h * rr.uniform(0.25, 0.45), math.sin(a) * d))
    out = merge(P)
    tone = fbm(out.v, 0.25, seed + 7)
    base = (CREAM if active else GREY)
    out.c = np.clip(base[None] * (0.85 + 0.3 * tone[:, None]) + (WHITE - base)[None] * sstep(0.5, 0.9, out.v[:, 1] / max(h, 1))[:, None] * (1 if active else 0), 0, 1)
    return out


def terrace():
    def h(X, Z):
        P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
        dome = 6.0 * np.exp(-(X ** 2 + Z ** 2) / (2 * 70.0 ** 2))
        return (dome + 2.5 * (fbm(P3, 0.02, 3, 4).reshape(X.shape) - 0.5) + 0.8 * (fbm(P3, 0.12, 4, 3).reshape(X.shape) - 0.5)) * sstep(170, 120, np.hypot(X, Z)) - 0.4
    g = grid(-170, 170, -170, 170, 120, 120, h, (0.62, 0.6, 0.55))
    t = fbm(g.v, 0.05, 9)
    g.c = np.clip(np.array([0.6, 0.58, 0.53])[None] * (0.85 + 0.25 * t[:, None]), 0, 1)
    g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > -0.2]
    return g


ground = terrace()


def ground_y(x, z):
    d = np.hypot(x, z)
    return 6.0 * math.exp(-(x * x + z * z) / (2 * 70.0 ** 2)) * float(sstep(170, 120, d)) - 0.6


towers = [chimney(60.0, 9.0, 1, True, spires=5).T(0, ground_y(0, 0), 0)]           # Poseidon
for k in range(34):
    a, d = rng.uniform(0, 6.28), 20 + 120 * math.sqrt(rng.random())
    x, z = math.cos(a) * d, math.sin(a) * d
    big = rng.random() < 0.3
    h = rng.uniform(15, 35) if big else rng.uniform(3, 14)
    towers.append(chimney(h, h * rng.uniform(0.12, 0.2), 10 + k, rng.random() < 0.6, spires=2 if big else 0).T(x, ground_y(x, z), z))
rock = merge(towers)
bake_ao([rock, ground], occluders=[rock, ground], ground=None, cell=1.0, rays=18, reach=12, strength=0.7)

# life: snails and amphipods on the active white surfaces, bacterial films, a few fish
r = np.random.default_rng(7)
INS = []
snail = abyss.blob(0.05, (0.55, 0.5, 0.42), 3, sub=1, amp=0.3)
pts = abyss.sample_surface(rock, 900, lambda fn: fn[:, 1] > -0.4, r)
if pts:
    INS.append(("snails", snail, "life", instances(np.array([p for p, _ in pts]), np.array([n for _, n in pts]), scale=r.uniform(0.6, 1.6, len(pts)))))
mat = abyss.blob(1.0, (0.9, 0.88, 0.8), 5, sub=1, amp=0.2)
mat.v[:, 1] *= 0.04
pts = abyss.sample_surface(ground, 120, lambda fn: fn[:, 1] > 0.8, r)
if pts:
    INS.append(("bacterial_mats", mat, "life", instances(np.array([p for p, _ in pts]) + [0, 0.02, 0], scale=r.uniform(0.8, 3.0, len(pts)))))
fish = abyss.fish(0.5, (0.45, 0.3, 0.25))
fp = np.c_[r.normal(0, 70, 25), r.uniform(4, 30, 25), r.normal(0, 70, 25)]
INS.append(("fish", fish, "life", instances(fp, yaw=r.uniform(0, 6.28, 25), scale=r.uniform(0.8, 1.6, 25))))

out = ROOT / "models" / "item" / "Q521459" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
mats = {"rock": material(rough=0.95),
        "ground": material(rough=1.0), "life": material(rough=0.6)}
tris, size = write_model(out, [("chimneys", rock, "rock"), ("terrace", ground, "ground")], mats, instanced=INS)
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB")
