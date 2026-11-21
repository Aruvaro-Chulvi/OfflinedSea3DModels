"""
Loki's Castle (Q6668666) — the black-smoker vent field on the Arctic Mid-Ocean Ridge, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Pedersen et al. 2010, Nature Communications, and later work): found
in July 2008 at about 73°30' N, 8° E, where the Mohns and Knipovich ridges meet, at about 2,350 m. Two large mounds
of sulphide (western and eastern), 150-250 m across at the base, carry four active black-smoker chimneys up to
13 m tall in two vent sites about 150 m apart, venting fluid at 310-320 °C. The mound is crusted with dense fields
of thin siboglinid tubeworms (Sclerolinum contortum), with crusts of tube-dwelling polychaetes at the diffuse
vents and white bacterial mats on the chimneys. Named for its castle-like shape and for the trickster god.
Here one mound is drawn, about 200 m across; the shape of the mound and of each chimney is approximate.

    python model_sources/Q6668666_lokis_castle.py   -> models/item/Q6668666/model.glb
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

rng = np.random.default_rng(2008)
WHITE, CREAM, GREY = np.array([0.82, 0.8, 0.74]), np.array([0.2, 0.17, 0.15]), np.array([0.32, 0.25, 0.2])   # bacterial white, sulphide black, rust-brown


def chimney(h, r, seed, active=True, spires=0):
    """A sulphide chimney: a knobbly black column with beehive ledges, dusted white by bacteria near the top."""
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
    out.c = np.clip(base[None] * (0.8 + 0.4 * tone[:, None]) + (WHITE - base)[None] * (sstep(0.6, 0.85, fbm(out.v, 0.6, seed + 9)) * 0.7)[:, None], 0, 1)
    return out


def terrace():
    """The sulphide mound: a broad hill about 200 m across and some 25 m high, flat-topped, its flanks rubbly."""
    def h(X, Z):
        P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
        d = np.hypot(X, Z * 1.15)
        dome = 25.0 * sstep(105, 40, d)
        return dome + 3.0 * (fbm(P3, 0.03, 3, 4).reshape(X.shape) - 0.5) + 1.2 * (fbm(P3, 0.15, 4, 3).reshape(X.shape) - 0.5) - 0.4
    g = grid(-130, 130, -130, 130, 110, 110, h, (0.3, 0.26, 0.22))
    t = fbm(g.v, 0.05, 9)
    rusty = sstep(0.5, 0.75, fbm(g.v, 0.03, 10))
    g.c = np.clip(np.array([0.25, 0.22, 0.19])[None] * (0.8 + 0.4 * t[:, None]) * (1 - rusty[:, None]) + np.array([0.45, 0.28, 0.15])[None] * rusty[:, None], 0, 1)
    g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > -0.2]
    return g


ground = terrace()


def ground_y(x, z):
    d = math.hypot(x, z * 1.15)
    return 25.0 * float(sstep(105, 40, d)) - 0.8


towers = []
for site, (cx, cz) in enumerate(((-30.0, -20.0), (40.0, 25.0))):            # two vent sites about 150 m apart... here on one mound
    for k in range(2):                                                        # two big active black smokers at each
        x, z = cx + rng.normal(0, 6), cz + rng.normal(0, 6)
        towers.append(chimney(rng.uniform(9, 13), rng.uniform(1.6, 2.4), 10 * site + k, True, spires=2).T(x, ground_y(x, z), z))
    for k in range(10):                                                       # and smaller, extinct or diffuse ones about them
        x, z = cx + rng.normal(0, 18), cz + rng.normal(0, 18)
        towers.append(chimney(rng.uniform(1.5, 6), rng.uniform(0.4, 1.0), 100 + 20 * site + k, rng.random() < 0.4).T(x, ground_y(x, z), z))
rock = merge(towers)
bake_ao([rock, ground], occluders=[rock, ground], ground=None, cell=1.0, rays=18, reach=12, strength=0.7)

# life: snails and amphipods on the active white surfaces, bacterial films, a few fish
r = np.random.default_rng(7)
INS = []
tuft = merge([cyl(0.006, 0.004, 0.25 + 0.05 * k, (0.75, 0.68, 0.55), seg=3, caps=False).R(0.15 * math.cos(k), 0, 0.15 * math.sin(k)).T(0.03 * math.cos(2 * k), 0, 0.03 * math.sin(2 * k)) for k in range(9)])
pts = abyss.sample_surface(ground, 2500, lambda fn: fn[:, 1] > 0.6, r)
pts = [p for p in pts if p[0][1] > 15.0]                                    # the tubeworm fields on top of the mound
if pts:
    INS.append(("tubeworms", tuft, "life", instances(np.array([p for p, _ in pts]), np.array([n for _, n in pts]), scale=r.uniform(0.8, 1.8, len(pts)))))
mat = abyss.blob(1.0, (0.9, 0.88, 0.8), 5, sub=1, amp=0.2)
mat.v[:, 1] *= 0.04
pts = abyss.sample_surface(ground, 3000, lambda fn: fn[:, 1] > 0.6, r)
pts = [p for p in pts if min(math.hypot(p[0][0] + 30, p[0][2] + 20), math.hypot(p[0][0] - 40, p[0][2] - 25)) < 22][:140]   # mats about the vents
if pts:
    INS.append(("bacterial_mats", mat, "life", instances(np.array([p for p, _ in pts]) + [0, 0.02, 0], np.array([n for _, n in pts]), scale=r.uniform(0.6, 2.2, len(pts)))))
fish = abyss.fish(0.5, (0.45, 0.3, 0.25))
fp = np.c_[r.normal(0, 60, 6), r.uniform(28, 40, 6), r.normal(0, 60, 6)]
INS.append(("fish", fish, "life", instances(fp, yaw=r.uniform(0, 6.28, 6), scale=r.uniform(0.8, 1.4, 6))))

out = ROOT / "models" / "item" / "Q6668666" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
mats = {"rock": material(rough=0.95),
        "ground": material(rough=1.0), "life": material(rough=0.6)}
tris, size = write_model(out, [("chimneys", rock, "rock"), ("mound", ground, "ground")], mats, instanced=INS)
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB")
