"""
Osborne Reef (Q1272680) — the failed tyre reef off Fort Lauderdale, Florida, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.

The place, after the published descriptions (Florida DEP, Nova Southeastern University, Wikipedia): an artificial
reef first built of concrete "jacks", then in 1972 enlarged with one to two million scrap tyres, tied in bundles
and dropped about 1.3 miles off Fort Lauderdale in 60-70 ft (18-21 m). The bundles came apart as their fastenings
corroded; the loose tyres, which corals cannot settle on, were moved about by currents and storms, smothering the
natural reef nearby. They now cover more than 34 acres of the sea floor. Clean-ups by the US forces and the State
of Florida have removed several hundred thousand; some 700,000 remain. Here about 250 m of the field is drawn:
the old concrete jacks, a few bundles still tied, drifts of loose tyres heaped against the edge of the natural
reef, and cleared patches; the layout and the number of tyres drawn (a sample of the real count) are approximate.

    python model_sources/Q1272680_osborne_reef.py   -> models/item/Q1272680/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, grid, crumple, sstep, fbm  # noqa: E402
from scenekit import material, bake_ao, instances, write_model  # noqa: E402
import abyss  # noqa: E402

rng = np.random.default_rng(1972)
RUBBER = (0.08, 0.08, 0.085)


def tyre(r=0.33, t=0.11, seg=8):
    prof = [(r + t * math.cos(a), t * 1.1 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 6)]
    return lathe(prof, RUBBER, seg=seg)


def jack():
    """A concrete 'jack': three square bars crossed at right angles."""
    c = (0.62, 0.6, 0.55)
    P = [box(3.0, 0.4, 0.4, c, bottom=False), box(0.4, 3.0, 0.4, c, bottom=False), box(0.4, 0.4, 3.0, c, bottom=False)]
    m = merge(P); crumple(m, 0.05, 1.5, 3)
    return m


def reef_edge():
    """The edge of the natural reef to the east: a low rocky ridge, with sand in front of it."""
    def h(X, Z):
        P3 = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
        ridge = 2.2 * sstep(-115, -95, X) * (0.7 + 0.6 * fbm(P3, 0.05, 5, 3).reshape(X.shape))
        return ridge + 0.5 * (fbm(P3, 0.2, 6, 3).reshape(X.shape) - 0.5) - 0.3
    g = grid(-150, -80, -140, 140, 30, 110, h, (0.55, 0.52, 0.45))
    g.f = g.f[g.v[g.f][:, :, 1].max(axis=1) > 0.1]
    return g


reef = reef_edge()
bake_ao([reef], occluders=[reef], ground=None, cell=1.0, rays=12, reach=8, strength=0.6)
r = np.random.default_rng(7)
INS = []
tyres = [tyre(0.3 + 0.04 * k) for k in range(3)]
pos, nrm = [], []
cleared = [(rng.uniform(-60, 80), rng.uniform(-90, 90), rng.uniform(12, 25)) for _ in range(4)]


def is_cleared(x, z):
    return any(math.hypot(x - a, z - b) < rr for a, b, rr in cleared)


for k in range(9000):                                                         # loose tyres, thicker towards the reef edge where they heap up
    x = -95 + 210 * r.random() ** 1.6
    z = r.uniform(-125, 125)
    if is_cleared(x, z) and r.random() < 0.9:
        continue
    pile = max(0.0, (x + 95) < 25) * r.uniform(0, 1.6)
    y = 0.1 + pile * r.random()
    pos.append((x, y, z)); a = r.uniform(0, 6.28); tl = r.uniform(0, 1.2) if pile else r.uniform(0, 0.25)
    nrm.append((math.sin(tl) * math.cos(a), math.cos(tl), math.sin(tl) * math.sin(a)))
for b in range(30):                                                           # bundles still tied: stacks of tyres
    bx, bz = rng.uniform(-60, 100), rng.uniform(-110, 110)
    lie = rng.random() < 0.6
    for k in range(int(rng.integers(6, 12))):
        if lie:
            pos.append((bx + k * 0.24, 0.35, bz)); nrm.append((1.0, 0.0, 0.0))
        else:
            pos.append((bx, 0.12 + k * 0.24, bz)); nrm.append((0.0, 1.0, 0.0))
pos, nrm = np.array(pos), np.array(nrm)
pick = r.integers(len(tyres), size=len(pos))
for k, t in enumerate(tyres):
    s = pick == k
    INS.append((f"tyres_{k}", t, "rubber", instances(pos[s], nrm[s], yaw=r.uniform(0, 6.28, s.sum()))))
jp = np.c_[r.normal(40, 18, 40), np.full(40, 1.2), r.normal(0, 18, 40)]
INS.append(("concrete_jacks", jack(), "concrete", instances(jp, np.c_[r.normal(0, 0.4, 40), np.ones(40), r.normal(0, 0.4, 40)], yaw=r.uniform(0, 6.28, 40))))
# a few corals and sponges hold on to the jacks and the natural reef, not to the tyres; fish
spg = abyss.glass_sponge(0.6); spg.c = np.clip(spg.c * np.array([1.0, 0.7, 0.4]), 0, 1)
sp = abyss.sample_surface(reef, 160, lambda fn: fn[:, 1] > 0.5, r)
INS.append(("sponges", spg, "life", instances(np.array([p for p, _ in sp]), scale=r.uniform(0.6, 1.6, len(sp)))))
cor = abyss.blob(0.4, (0.6, 0.55, 0.35), 21, sub=1, amp=0.4); cor.v[:, 1] *= 0.6
cp = abyss.sample_surface(reef, 220, lambda fn: fn[:, 1] > 0.3, r)
INS.append(("corals", cor, "life", instances(np.array([p for p, _ in cp]), np.array([n for _, n in cp]), scale=r.uniform(0.5, 1.6, len(cp)))))
fish = abyss.fish(0.35, (0.8, 0.65, 0.3))
fp = np.c_[r.normal(-60, 40, 60), r.uniform(1.5, 8, 60), r.normal(0, 60, 60)]
INS.append(("fish", fish, "life", instances(fp, yaw=r.uniform(0, 6.28, 60), scale=r.uniform(0.6, 1.5, 60))))

out = ROOT / "models" / "item" / "Q1272680" / "model.glb"
out.parent.mkdir(parents=True, exist_ok=True)
mats = {"reef": material(rough=0.95), "rubber": material(rough=0.75), "concrete": material(rough=0.95), "life": material(rough=0.6)}
tris, size = write_model(out, [("natural_reef", reef, "reef")], mats, instanced=INS)
print(f"{out}: {tris:,} triangles, {size / 2**20:.2f} MB, {len(pos)} tyres")
