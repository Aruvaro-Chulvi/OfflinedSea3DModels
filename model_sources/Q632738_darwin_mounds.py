"""
Darwin Mounds (Q632738) — cold-water coral mounds in the northern Rockall Trough, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Masson et al. 2003; Wheeler et al.; JNCC): discovered in 1998 on the
sea floor north-west of Scotland at about 950-1,050 m, an area of some 100 km² holding over 300 small sandy mounds,
up to about 75 m across and 5 m high, formed by fluid escaping through a thin sandy contourite shaped by the
Shelf Edge Current. Each mound is capped with thickets of the cold-water corals Lophelia pertusa (Desmophyllum
pertusum) and Madrepora oculata, home to brittle stars, brisingid starfish, echiuran worms and sponges; downstream
of most mounds a long, low "tail" of sediment is strewn with giant single-celled xenophyophores. Bottom trawling
broke up the coral on some mounds before they were protected (closed to trawling in 2003; a Special Area of
Conservation). Here a group of fourteen mounds about 600 m across is drawn; their layout, the tails' direction and
the trawl damage are approximate.

    python model_sources/Q632738_darwin_mounds.py   -> models/item/Q632738/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, cyl, grid, tube, crumple, sstep, fbm  # noqa: E402
from scenekit import material, bake_ao, instances, write_model  # noqa: E402
import abyss  # noqa: E402

rng = np.random.default_rng(1998)
TAIL = np.array([0.7, 0.0, -0.7])                                             # downstream: to the south-west (+X is west here)
mounds = []
for k in range(14):
    for _ in range(50):
        x, z = rng.uniform(-260, 260), rng.uniform(-200, 260)
        if all(math.hypot(x - a, z - b) > 75 for a, b, _, _ in mounds):
            break
    mounds.append((x, z, rng.uniform(22, 37), rng.uniform(2.5, 5.0)))
damaged = set(rng.choice(14, 4, replace=False).tolist())                     # mounds where trawls broke up the coral


def height(X, Z):
    H = np.zeros_like(X)
    for (x, z, r, h) in mounds:
        d = np.hypot(X - x, Z - z)
        H = np.maximum(H, h * np.clip(1 - (d / r) ** 2, 0, 1) ** 0.8)
        along = (X - x) * TAIL[0] + (Z - z) * TAIL[2]
        across = (X - x) * TAIL[2] * -1 + (Z - z) * TAIL[0]
        tail = 0.6 * np.exp(-(across / (r * 0.45)) ** 2) * sstep(0, r, along) * sstep(r * 8, r * 2, along)
        H = np.maximum(H, tail)
    P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    return H + 0.25 * (fbm(P3, 0.05, 3, 3).reshape(X.shape) - 0.5) - 0.2


ground = grid(-330, 330, -330, 330, 170, 170, height, (0.66, 0.62, 0.54))
t = fbm(ground.v, 0.04, 9)
ground.c = np.clip(np.array([0.66, 0.62, 0.54])[None] * (0.88 + 0.2 * t[:, None]), 0, 1)
ground.f = ground.f[ground.v[ground.f][:, :, 1].max(axis=1) > 0.05]
bake_ao([ground], occluders=[ground], ground=None, cell=3.0, rays=12, reach=20, strength=0.5)


def coral_bush(seed, size=1.0, col=(0.9, 0.86, 0.8)):
    """A Lophelia thicket: a branching bush of thin white-pink stems."""
    rr = np.random.default_rng(seed)
    P = []

    def branch(p, d, ln, r, depth):
        q = p + d * ln
        P.append(tube([tuple(p), tuple(q)], r, col, seg=4))
        if depth > 0:
            for _ in range(2):
                nd = d + rr.normal(0, 0.55, 3); nd[1] = abs(nd[1]) + 0.2; nd /= np.linalg.norm(nd)
                branch(q, nd, ln * 0.72, r * 0.75, depth - 1)
    for k in range(4):
        d = np.array([rr.normal(0, 0.5), 1.0, rr.normal(0, 0.5)]); d /= np.linalg.norm(d)
        branch(np.zeros(3), d, 0.35 * size, 0.035 * size, 3)
    return merge(P)


r = np.random.default_rng(7)
INS = []
bushes = [coral_bush(k, 1.0, c) for k, c in enumerate(((0.92, 0.88, 0.82), (0.95, 0.82, 0.78), (0.9, 0.9, 0.86)))]
rubble = abyss.blob(0.6, (0.8, 0.76, 0.7), 11, sub=1, amp=0.5); rubble.v[:, 1] *= 0.25
pos_c, pos_r = [], []
for k, (x, z, rad, h) in enumerate(mounds):
    n = 70 if k not in damaged else 12
    for _ in range(n):
        a, d = r.uniform(0, 6.28), rad * 0.75 * math.sqrt(r.random())
        px, pz = x + math.cos(a) * d, z + math.sin(a) * d
        pos_c.append((px, float(height(np.array([[px]]), np.array([[pz]]))[0, 0]), pz))
    if k in damaged:                                                          # broken coral rubble spread by the trawls
        for _ in range(60):
            a, d = r.uniform(0, 6.28), rad * 1.1 * math.sqrt(r.random())
            px, pz = x + math.cos(a) * d, z + math.sin(a) * d
            pos_r.append((px, float(height(np.array([[px]]), np.array([[pz]]))[0, 0]), pz))
pos_c = np.array(pos_c); pick = r.integers(len(bushes), size=len(pos_c))
for k, b in enumerate(bushes):
    s = pick == k
    INS.append((f"lophelia_{k}", b, "life", instances(pos_c[s], yaw=r.uniform(0, 6.28, s.sum()), scale=r.uniform(1.0, 3.0, s.sum()))))
if pos_r:
    pos_r = np.array(pos_r)
    INS.append(("coral_rubble", rubble, "life", instances(pos_r, yaw=r.uniform(0, 6.28, len(pos_r)), scale=r.uniform(0.5, 1.8, len(pos_r)))))
# xenophyophores on the tails, brittle stars and brisingid starfish about the corals, sponges, a few grenadiers
xen = abyss.xeno(0.08)
pts = abyss.sample_surface(ground, 6000, lambda fn: fn[:, 1] > 0.9, r)
xp = [p for p, _ in pts if 0.15 < p[1] < 0.9]
if xp:
    INS.append(("xenophyophores", xen, "life", instances(np.array(xp[:2500]), scale=r.uniform(0.6, 1.6, min(len(xp), 2500)))))
star = abyss.brittle_star(0.1, (0.6, 0.45, 0.4))
sp = pos_c[r.choice(len(pos_c), 300)] + np.c_[r.normal(0, 0.8, 300), np.full(300, 0.02), r.normal(0, 0.8, 300)]
INS.append(("brittle_stars", star, "life", instances(sp, yaw=r.uniform(0, 6.28, 300), scale=r.uniform(0.7, 1.4, 300))))
bris = abyss.starfish(0.25, (0.9, 0.4, 0.15))
bp = pos_c[r.choice(len(pos_c), 80)] + np.array([0, 0.5, 0])
INS.append(("brisingids", bris, "life", instances(bp, yaw=r.uniform(0, 6.28, 80), scale=r.uniform(0.8, 1.6, 80))))
spg = abyss.glass_sponge(0.4)
gp = pos_c[r.choice(len(pos_c), 60)]
INS.append(("sponges", spg, "life", instances(gp, yaw=r.uniform(0, 6.28, 60), scale=r.uniform(0.6, 1.4, 60))))
fish = abyss.rattail(0.8)
fp = np.c_[r.uniform(-250, 250, 14), r.uniform(1.0, 4.0, 14), r.uniform(-250, 250, 14)]
INS.append(("grenadiers", fish, "life", instances(fp, yaw=r.uniform(0, 6.28, 14))))

out = ROOT / "models" / "item" / "Q632738" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
tris, size = write_model(out, [("sediment", ground, "ground")], {"ground": material(rough=1.0), "life": material(rough=0.6)}, instanced=INS)
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB")
