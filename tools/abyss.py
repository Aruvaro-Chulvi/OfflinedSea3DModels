"""
abyss — the deep sea, by code (MIT): animals that live on wrecks and on the abyssal plain, rust, and things
from a ship that last on the bottom. Every function returns a meshkit Mesh in metres, built «up = +y» around
the origin, so it can be stored once and copied with instancing (scenekit.instances / write_model).

Only what is real far from the light (below ~1,000 m): no reefs, no algae, no colourful fish.
    abyss.seed(7); m = abyss.anemone(0.06, (0.95, 0.9, 0.85))
"""
from __future__ import annotations

import math

import numpy as np

from meshkit import Mesh, merge, box, cyl, lathe, tube, crumple, sstep, fbm, normals

rng = np.random.default_rng(1)


def seed(n):
    """Reseed the shapes' random details (curl of the arms, number of polyps…)."""
    global rng
    rng = np.random.default_rng(n)


LEATHER = (0.17, 0.10, 0.06)


def ttube(points, radii, c, seg=6, cap=True):
    """A tube along a polyline whose radius changes from point to point (arms, tentacles, stalks)."""
    pts = np.asarray(points, float); rr = np.broadcast_to(np.asarray(radii, float), (len(pts),))
    v, f = [], []
    for i, p in enumerate(pts):
        t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        t = t / (np.linalg.norm(t) or 1)
        a = np.cross(t, [0, 1, 0]) if abs(t[1]) < 0.9 else np.cross(t, [1, 0, 0])
        a /= np.linalg.norm(a); b = np.cross(t, a)
        for k in range(seg):
            th = 2 * math.pi * k / seg
            v.append(p + rr[i] * (math.cos(th) * a + math.sin(th) * b))
    for i in range(len(pts) - 1):
        for k in range(seg):
            j = (k + 1) % seg
            p0, p1 = i * seg + k, i * seg + j
            f += [(p0, p0 + seg, p1 + seg), (p0, p1 + seg, p1)]
    if cap:
        o = len(v); v.append(pts[-1])
        n0 = (len(pts) - 1) * seg
        f += [(n0 + k, n0 + (k + 1) % seg, o) for k in range(seg)]
    return Mesh(v, f, c)


def blob(r, c, seed, sub=1, amp=0.25):
    """A lumpy ball (stones, xenophyophores): an icosphere pushed by noise."""
    t = (1 + 5 ** 0.5) / 2
    V = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = list(V / np.linalg.norm(V[0]))
    for _ in range(sub):
        mid, nf = {}, []
        for a, b, cc in F:
            ab, bc, ca = [], [], []
            for (i, j), out in (((a, b), ab), ((b, cc), bc), ((cc, a), ca)):
                k = (min(i, j), max(i, j))
                if k not in mid:
                    m = (V[i] + V[j]) / 2; V.append(m / np.linalg.norm(m)); mid[k] = len(V) - 1
                out.append(mid[k])
            nf += [(a, ab[0], ca[0]), (b, bc[0], ab[0]), (cc, ca[0], bc[0]), (ab[0], bc[0], ca[0])]
        F = nf
    V = np.array(V)
    k = 1 + amp * (fbm(V * 1.3, 1.0, seed, 3) - 0.5) * 2
    return Mesh(V * k[:, None] * r, F, c)


def sample_surface(m, n, cond, r, nrm=None):
    """n random points on the faces of m that pass cond(face_normal) -> [(point, normal)] (area-weighted)."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(b - a, c - a); ar = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(ar[:, None], 1e-12)
    ok = cond(fn) & (ar > 1e-4)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    pick = r.choice(idx, size=n, p=ar[idx] / ar[idx].sum())
    r1 = np.sqrt(r.random(n)); r2 = r.random(n)
    p = (1 - r1)[:, None] * a[pick] + (r1 * (1 - r2))[:, None] * b[pick] + (r1 * r2)[:, None] * c[pick]
    return list(zip(p, fn[pick]))


def mix(c1, c2, t):
    return tuple(np.array(c1) * (1 - t) + np.array(c2) * t)


def rusticle(length, r0, seed):
    rr_ = np.random.default_rng(seed)
    n = 5
    prof = []
    ph = rr_.uniform(0, 6.28)
    for i in range(n + 1):
        t = i / n
        rad = r0 * (1 - t) ** 0.75 * (1 + 0.22 * math.sin(t * 9 + ph)) + 0.004
        prof.append((rad, -length * t))
    m = lathe(prof, (0.5, 0.25, 0.1), seg=5)
    y = m.v[:, 1] / -length
    m.c = np.array([mix((0.36, 0.14, 0.06), (0.82, 0.5, 0.2), sstep(0.15, 1.0, t) * rr_.uniform(0.6, 1.0)) for t in y])
    bend = rr_.normal(0, 0.06, 2)
    m.v[:, 0] += bend[0] * y ** 2 * length; m.v[:, 2] += bend[1] * y ** 2 * length
    return m


def rust_flow(length, w, seed):
    """A run of rust down a wall: a thin, lumpy drip (built pointing down -y, bulging towards +z), dark at the
    top where it comes out of a seam or a porthole, orange where it is fresh."""
    rr_ = np.random.default_rng(seed)
    prof = [(0.001, 0)]
    for t in np.linspace(0.08, 0.95, 6):
        prof.append((w * (0.55 + 0.45 * math.sin(t * math.pi)) * (1 + 0.35 * rr_.uniform(-1, 1)), -length * t))
    prof.append((0.001, -length))
    m = lathe(prof, (0.5, 0.25, 0.1), seg=5)
    m.v[:, 2] = np.abs(m.v[:, 2]) * 0.45
    hot = rr_.uniform(0.3, 1.0)
    m.c = np.array([mix((0.30, 0.12, 0.05), (0.72, 0.38, 0.15), sstep(0.1, 1.0, -y / length) * hot) for y in m.v[:, 1]])
    crumple(m, w * 0.25, 2.5 / w, seed)
    return m


def brittle_star(sz, c):
    parts = [cyl(sz * 0.22, sz * 0.18, sz * 0.08, c, seg=6)]
    a0 = rng.uniform(0, 6.28)
    for k in range(5):
        a = a0 + k * 2 * math.pi / 5
        curl = rng.uniform(-0.9, 0.9)
        pts, rad = [], []
        for t in np.linspace(0, 1, 5):
            aa = a + curl * t * t
            rr_ = sz * (0.18 + 0.95 * t)
            pts.append((rr_ * math.cos(aa), sz * 0.04 + 0.03 * sz * math.sin(t * 3), rr_ * math.sin(aa)))
            rad.append(sz * (0.06 * (1 - t) + 0.012))
        parts.append(ttube(pts, rad, c, seg=3))
    return merge(parts)


def anemone(sz, c):
    col = mix(c, (0.95, 0.92, 0.88), 0.3)
    parts = [cyl(sz * 0.32, sz * 0.38, sz * 0.9, col, seg=7), cyl(sz * 0.4, sz * 0.4, 0.01, mix(c, (0.3, 0.1, 0.1), 0.3), seg=7).T(0, sz * 0.9, 0)]
    n = 11
    for k in range(n):
        a = 2 * math.pi * k / n + rng.uniform(-0.1, 0.1)
        ln = sz * rng.uniform(0.7, 1.1)
        pts = [(sz * 0.36 * math.cos(a), sz * 0.9, sz * 0.36 * math.sin(a)),
               ((sz * 0.36 + ln * 0.5) * math.cos(a), sz * 0.9 + ln * 0.45, (sz * 0.36 + ln * 0.5) * math.sin(a)),
               ((sz * 0.36 + ln * 0.85) * math.cos(a), sz * 0.9 + ln * 0.5, (sz * 0.36 + ln * 0.85) * math.sin(a))]
        parts.append(ttube(pts, [sz * 0.05, sz * 0.035, sz * 0.012], c, seg=3))
    return merge(parts)


def mushroom_coral(sz):
    red = (0.62, 0.08, 0.08)
    parts = [cyl(sz * 0.3, sz * 0.26, sz * 0.7, (0.75, 0.25, 0.2), seg=8),
             lathe([(0.001, sz * 1.15), (sz * 0.45, sz * 1.08), (sz * 0.7, sz * 0.9), (sz * 0.62, sz * 0.7), (sz * 0.25, sz * 0.66)], red, seg=12)]
    for k in range(7):
        a = rng.uniform(0, 6.28); rr_ = rng.uniform(0, sz * 0.45)
        parts.append(cyl(sz * 0.05, 0.002, sz * 0.18, (0.95, 0.92, 0.88), seg=4).T(rr_ * math.cos(a), sz * 1.08, rr_ * math.sin(a)))
    return merge(parts)


def squat_lobster(sz):
    c = rng.choice([(0.93, 0.9, 0.84), (0.95, 0.62, 0.35), (0.9, 0.82, 0.7)])
    body = lathe([(0.001, -sz * 0.5), (sz * 0.18, -sz * 0.35), (sz * 0.22, 0), (sz * 0.16, sz * 0.35), (0.001, sz * 0.5)], c, seg=8, axis="z").S(1, 0.6, 1)
    parts = [body.T(0, sz * 0.12, 0)]
    for s in (-1, 1):
        parts.append(ttube([(s * sz * 0.12, sz * 0.12, sz * 0.35), (s * sz * 0.5, sz * 0.25, sz * 0.9), (s * sz * 0.55, sz * 0.2, sz * 1.6)],
                           [sz * 0.05, sz * 0.04, sz * 0.02], c, seg=4))
        for k in range(3):
            zz = sz * (0.2 - 0.18 * k)
            parts.append(ttube([(s * sz * 0.15, sz * 0.12, zz), (s * sz * 0.45, sz * 0.25, zz + sz * 0.05), (s * sz * 0.65, 0.0, zz + sz * 0.1)],
                               [sz * 0.03, sz * 0.025, sz * 0.01], c, seg=3))
    return merge(parts)


def sea_cucumber(kind):
    if kind == "psychropotes":                     # yellowish, translucent, with a tall «sail» at the back
        ln, c = rng.uniform(0.35, 0.6), (0.86, 0.72, 0.42)
    else:                                          # Benthodytes: deep purple
        ln, c = rng.uniform(0.25, 0.45), (0.38, 0.12, 0.3)
    body = lathe([(0.001, -ln / 2), (ln * 0.12, -ln * 0.4), (ln * 0.17, -ln * 0.1), (ln * 0.16, ln * 0.25), (ln * 0.1, ln * 0.45), (0.001, ln / 2)],
                 c, seg=10, axis="z")
    body.v[:, 1] = np.maximum(body.v[:, 1], -ln * 0.02)
    body.S(1.25, 0.65, 1).T(0, ln * 0.06, 0)
    parts = [body]
    if kind == "psychropotes":
        sail = merge([box(ln * 0.05, ln * 0.6, ln * 0.18, mix(c, (1, 1, 1), 0.2))]).R(-0.5, 0, 0).T(0, ln * 0.1, -ln * 0.38)
        parts.append(sail)
    for k in range(6):                                                  # little tube feet along the sides
        zz = -ln * 0.35 + ln * 0.7 * k / 5
        for s in (-1, 1):
            parts.append(cyl(ln * 0.018, ln * 0.008, ln * 0.06, c, seg=3, caps=False).R(0, 0, s * 1.1).T(s * ln * 0.2, ln * 0.04, zz))
    return merge(parts)


def rattail(ln):
    c = (0.5, 0.47, 0.42)
    body = lathe([(0.001, ln * 0.0), (ln * 0.07, ln * 0.04), (ln * 0.085, ln * 0.12), (ln * 0.07, ln * 0.25), (ln * 0.035, ln * 0.5), (ln * 0.012, ln * 0.8), (0.001, ln)],
                 c, seg=10, axis="z")
    body.S(0.75, 1, 1)
    body.v[:, 2] = -body.v[:, 2]                                        # head at +z
    body.paint(lambda q: np.array(c) * (0.85 + 0.25 * sstep(-0.02, 0.06, q[:, 1]))[:, None])
    parts = [body]
    for s in (-1, 1):
        parts.append(cyl(ln * 0.026, ln * 0.026, 0.004, (0.03, 0.04, 0.06), seg=8).R(0, 0, s * math.pi / 2).T(s * ln * 0.05, ln * 0.025, -ln * 0.06))
    parts.append(box(0.004, ln * 0.12, ln * 0.12, mix(c, (0.2, 0.2, 0.2), 0.3), bottom=True).R(-0.4, 0, 0).T(0, ln * 0.05, -ln * 0.2))   # first dorsal fin
    parts.append(box(0.003, ln * 0.03, ln * 0.6, mix(c, (0.2, 0.2, 0.2), 0.3), bottom=True).T(0, ln * 0.02, -ln * 0.62))              # long fin to the tail
    parts.append(box(0.003, ln * 0.025, ln * 0.55, mix(c, (0.2, 0.2, 0.2), 0.3), bottom=False).T(0, -ln * 0.03, -ln * 0.6))
    parts.append(tube([(0, -ln * 0.06, ln * 0.06), (0, -ln * 0.1, ln * 0.04)], ln * 0.004, c, seg=3))                                 # chin barbel
    for s in (-1, 1):
        parts.append(box(0.003, ln * 0.04, ln * 0.12, c, bottom=False).R(0, s * 0.5, s * 0.6).T(s * ln * 0.06, -ln * 0.01, ln * 0.13))  # pectorals
    m = merge(parts)
    m.v[:, 2] -= ln * 0.0
    return m


def sea_pen(h):
    c = (0.9, 0.86, 0.72)
    lean = rng.uniform(0.05, 0.25)                                     # leaning with the current (to the north-east)
    pts = [(0, 0, 0)] + [(lean * h * t * t * 0.6, h * t, lean * h * t * t) for t in np.linspace(0.15, 1, 7)]
    parts = [ttube(pts, np.linspace(0.016, 0.008, len(pts)), (0.82, 0.78, 0.62), seg=5)]
    tip = np.array(pts[-1])
    for k in range(rng.integers(5, 10)):
        a = rng.uniform(0, 6.28); ln = rng.uniform(0.06, 0.13)
        e = tip + np.array([math.cos(a) * ln, -ln * rng.uniform(0.2, 0.8), math.sin(a) * ln])
        parts.append(ttube([tip, (tip + e) / 2 + [0, 0.02, 0], e], [0.022, 0.018, 0.006], c, seg=5))
    return merge(parts)


def glass_sponge(h):
    prof = [(0.03 * h, 0), (0.12 * h, 0.1 * h), (0.16 * h, 0.4 * h), (0.22 * h, 0.8 * h), (0.25 * h, h)]
    m = lathe(prof, (0.88, 0.86, 0.78), seg=14)
    m.paint(lambda q: (np.array((0.9, 0.88, 0.8)) * (0.8 + 0.25 * (np.sin(q[:, 1] * 60) * np.sin(np.arctan2(q[:, 2], q[:, 0]) * 14) > 0.3))[:, None]))
    inner = lathe([(r * 0.9, y + 0.01) for r, y in prof], (0.55, 0.52, 0.46), seg=14)
    inner.f = inner.f[:, ::-1]
    return merge([m, inner])


def bamboo_coral(h, seed):
    r = np.random.default_rng(seed)
    parts = []

    def branch(p, d, ln, rad, depth):
        pts = [p]
        for t in range(1, 5):
            d = d + r.normal(0, 0.12, 3); d[1] = abs(d[1]) + 0.2; d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * ln / 4)
        cols = [(0.92, 0.9, 0.8) if k % 2 == 0 else (0.3, 0.2, 0.12) for k in range(len(pts))]
        m = ttube(pts, np.linspace(rad, rad * 0.6, len(pts)), (0.9, 0.88, 0.78), seg=4)
        m.c = np.repeat(np.array(cols), 4, axis=0)[:len(m.v)] if len(m.v) == len(pts) * 4 else m.c
        parts.append(m)
        if depth < 3:
            for _ in range(2):
                branch(pts[r.integers(2, 5)], d + r.normal(0, 0.6, 3), ln * 0.65, rad * 0.6, depth + 1)
    branch(np.zeros(3), np.array([0, 1.0, 0]), h * 0.5, 0.025, 0)
    return merge(parts)


def xeno(r_):
    return blob(r_, (0.84, 0.8, 0.68), int(rng.integers(1e6)), sub=1, amp=0.5).S(1, 0.6, 1)


def mound(r_):
    m = lathe([(r_, 0), (r_ * 0.6, r_ * 0.25), (r_ * 0.22, r_ * 0.38), (r_ * 0.08, r_ * 0.3), (0.001, r_ * 0.1)], (0.68, 0.64, 0.56), seg=7)
    m.paint(lambda q: np.where((np.hypot(q[:, 0], q[:, 2]) < r_ * 0.12)[:, None], (0.32, 0.29, 0.25), (0.66, 0.62, 0.54)))
    return m


def shoe(sz=1.0, boot=False):
    ln, w = 0.27 * sz, 0.095 * sz
    up = lathe([(0.001, -ln / 2), (w * 0.35, -ln * 0.46), (w * 0.5, -ln * 0.3), (w * 0.5, ln * 0.1), (w * 0.45, ln * 0.4), (0.001, ln / 2)],
               LEATHER, seg=10, axis="z")
    up.v[:, 1] = np.maximum(up.v[:, 1], 0) * (0.9 if not boot else 1.0)
    up.S(1, 1.1, 1)
    hi = (ln * 0.55 if boot else ln * 0.28)
    up.v[:, 1] += np.where(up.v[:, 2] < ln * 0.05, sstep(ln * 0.1, -ln * 0.2, up.v[:, 2]) * hi * (up.v[:, 1] > 0.005), 0)
    sole = box(w * 1.05, 0.018 * sz, ln * 1.02, (0.08, 0.06, 0.05)).T(0, -0.012, 0)
    heel = box(w * 0.8, 0.03 * sz, ln * 0.22, (0.08, 0.06, 0.05)).T(0, -0.03, -ln * 0.37)
    hole = cyl(w * 0.32, w * 0.32, 0.004, (0.03, 0.025, 0.02), seg=10, sx=1.3).R(0, math.pi / 2, 0).T(0, hi + 0.035 * sz, -ln * 0.22)
    m = merge([up.T(0, 0.012, 0), sole, heel, hole])
    m.c = m.c * (0.85 + 0.3 * fbm(m.v, 25, 3, 2))[:, None]
    return m


def plate(r=0.13):
    m = lathe([(0.001, 0), (r * 0.6, 0), (r * 0.72, 0.006), (r, 0.018), (r * 0.98, 0.022), (r * 0.7, 0.012), (0.001, 0.01)], (0.93, 0.92, 0.88), seg=20)
    m.paint(lambda q: np.where((np.abs(np.hypot(q[:, 0], q[:, 2]) - r * 0.86) < r * 0.05)[:, None], (0.2, 0.3, 0.6), (0.93, 0.92, 0.88)))
    return m


def cup():
    m = lathe([(0.001, 0), (0.025, 0), (0.035, 0.01), (0.045, 0.06), (0.042, 0.065), (0.038, 0.012), (0.001, 0.01)], (0.94, 0.93, 0.9), seg=16)
    h = tube([(0.044, 0.05, 0), (0.065, 0.045, 0), (0.066, 0.025, 0), (0.042, 0.02, 0)], 0.005, (0.94, 0.93, 0.9), seg=4)
    return merge([m, h])


def bottle(champ=True):
    g = (0.05, 0.14, 0.07) if champ else (0.12, 0.08, 0.04)
    m = lathe([(0.001, 0), (0.044, 0.004), (0.045, 0.2), (0.04, 0.23), (0.016, 0.27), (0.015, 0.31), (0.017, 0.315), (0.013, 0.32), (0.001, 0.32)], g, seg=12)
    cork = cyl(0.012, 0.012, 0.012, (0.4, 0.3, 0.18), seg=8).T(0, 0.318, 0)
    return merge([m, cork])


def bench_end():
    c = (0.13, 0.11, 0.1)
    curve = [(0, 0, 0.3), (0, 0.35, 0.22), (0, 0.45, 0.0), (0, 0.42, -0.25), (0, 0.6, -0.35), (0, 0.85, -0.3), (0, 0.9, -0.15)]
    leg = [(0, 0, -0.3), (0, 0.2, -0.2), (0, 0.42, -0.25)]
    scroll = [(0, 0.45, 0.0), (0, 0.55, 0.12), (0, 0.5, 0.2), (0, 0.42, 0.14)]
    return merge([tube(curve, 0.025, c, seg=5), tube(leg, 0.025, c, seg=5), tube(scroll, 0.018, c, seg=5)])


def pot(r_):
    cu = (0.6, 0.33, 0.18)
    m = lathe([(0.001, 0), (r_ * 0.95, 0), (r_, 0.03), (r_, r_ * 1.1), (r_ * 1.05, r_ * 1.12), (r_ * 0.96, r_ * 1.1), (r_ * 0.9, r_ * 0.1), (0.001, 0.02)], cu, seg=16)
    m.paint(lambda q: np.where((fbm(q, 18, 4, 2) > 0.6)[:, None], (0.3, 0.5, 0.4), cu))
    h = tube([(r_, r_ * 0.9, 0), (r_ * 1.5, r_ * 0.95, 0), (r_ * 2.2, r_ * 1.0, 0)], r_ * 0.06, (0.3, 0.2, 0.12), seg=4)
    return merge([m, h])


def telegraph():
    br = (0.6, 0.45, 0.2)
    col = lathe([(0.16, 0), (0.12, 0.08), (0.08, 0.15), (0.075, 0.9), (0.1, 0.95), (0.001, 0.96)], br, seg=14)
    head = lathe([(0.001, -0.07), (0.24, -0.06), (0.26, 0), (0.24, 0.06), (0.001, 0.07)], br, seg=20).R(math.pi / 2, 0, 0).T(0, 1.15, 0)
    dial = cyl(0.19, 0.19, 0.01, (0.8, 0.78, 0.7), seg=20).R(math.pi / 2, 0, 0).T(0, 1.15, 0.07)
    lever = box(0.04, 0.32, 0.03, br).T(0.08, 0.95, 0.12).R(0, 0, -0.35)
    return merge([col, head, dial, lever])


def safe():
    c = (0.12, 0.1, 0.09)
    body = box(0.95, 1.25, 0.75, c)
    door = box(0.75, 1.0, 0.05, (0.15, 0.12, 0.1)).T(0, 0.12, 0.4)
    knob = cyl(0.06, 0.06, 0.04, (0.6, 0.45, 0.2), seg=12).R(math.pi / 2, 0, 0).T(0.2, 0.65, 0.43)
    handle = tube([(-0.15, 0.62, 0.43), (-0.15, 0.62, 0.5), (-0.05, 0.62, 0.5), (-0.05, 0.62, 0.43)], 0.015, (0.6, 0.45, 0.2), seg=4)
    plate_n = box(0.3, 0.08, 0.01, (0.65, 0.5, 0.22)).T(0, 1.0, 0.43)
    m = merge([body, door, knob, handle, plate_n])
    crumple(m, 0.03, 3, 7)
    return m


def suitcase():
    c = (0.3, 0.18, 0.1)
    m = box(0.7, 0.22, 0.45, c)
    crumple(m, 0.02, 4, 9)
    straps = [box(0.04, 0.23, 0.46, (0.12, 0.08, 0.05)).T(x, 0, 0) for x in (-0.2, 0.2)]
    hand = tube([(-0.08, 0.22, 0.23), (-0.08, 0.27, 0.23), (0.08, 0.27, 0.23), (0.08, 0.22, 0.23)], 0.012, (0.12, 0.08, 0.05), seg=4)
    return merge([m] + straps + [hand])


def dust(m, amt, seed, col=(0.66, 0.62, 0.55)):
    """A film of fine sediment on everything that faces up."""
    n = normals(m)
    k = amt * sstep(0.25, 0.95, n[:, 1]) * (0.6 + 0.4 * fbm(m.v, 6.0, seed, 2))
    m.c = m.c * (1 - k[:, None]) + np.array(col) * k[:, None]
