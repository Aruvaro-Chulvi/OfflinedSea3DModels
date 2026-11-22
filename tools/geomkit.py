"""
geomkit — geometry queries for the model checks and fixes (numpy only, MIT).

    components(V, F)             -> one id per face: pieces made of faces that share vertices (welded at 2 mm)
    Grid(V, F, cell).query(lo, hi) -> faces whose box touches a box
    closest(P, A, B, C)          -> distances (n, m) and closest points (n, m, 3) from points to triangles
    seg_hits(P0, P1, A, B, C)    -> does any segment cross any triangle?
    ray_down(P, A, B, C)         -> for each point, the height of the highest triangle under it (-inf if none)
    thickness(P)                 -> (thinnest extent, middle extent, normal of the thinnest axis) by PCA
    gaps(V, F, cid, solid, …)    -> for each piece, how far it is from everything else (0 when touching)

Used by tools/fixkit.py (puts decals on their surface and loose pieces on what is under them) and
tools/geomcheck.py (the check run on every model in CI).
"""
from __future__ import annotations

import numpy as np

TOUCH = 0.015          # closer than this (metres) counts as touching


def components(V, F, eps=0.002):
    if not len(F):
        return np.zeros(0, np.int64)
    key = np.floor(V / eps).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    fw = inv[F]
    lab = np.arange(inv.max() + 1)
    while True:
        m = lab[fw].min(1)
        old = lab.copy()
        for k in range(3):
            np.minimum.at(lab, fw[:, k], m)
        lab = lab[lab]
        lab = lab[lab]
        if np.array_equal(lab, old):
            break
    _, cid = np.unique(lab[fw[:, 0]], return_inverse=True)
    return cid.ravel()


def closest(P, A, B, C):
    """Ericson's closest point on a triangle, for every point against every triangle: (dist (n,m), Q (n,m,3))."""
    P = P[:, None, :]
    ab, ac = B - A, C - A
    ap = P - A; bp = P - B; cp = P - C
    d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
    d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
    d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    Q = A + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    best = np.where(inside, np.linalg.norm(P - Q, axis=-1), np.inf)
    Q = np.where(inside[..., None], Q, 0.0)

    def edge(O, E, num, dd):
        t = np.clip(num / np.where(np.abs(dd) < 1e-20, 1e-20, dd), 0, 1)
        return O + E * t[..., None]

    for QQ in (edge(A, ab, d1, d1 - d3), edge(A, ac, d2, d2 - d6), edge(B, C - B, d4 - d3, (d4 - d3) + (d5 - d6))):
        d = np.linalg.norm(P - QQ, axis=-1)
        better = d < best
        best = np.where(better, d, best)
        Q = np.where(better[..., None], QQ, Q)
    return best, Q


def seg_hits(P0, P1, A, B, C):
    if not len(P0) or not len(A):
        return False
    D = P1 - P0
    e1, e2 = B - A, C - A
    h = np.cross(D[:, None, :], e2[None])
    a = (e1[None] * h).sum(-1)
    ok = np.abs(a) > 1e-12
    f = 1.0 / np.where(ok, a, 1.0)
    s = P0[:, None, :] - A[None]
    u = f * (s * h).sum(-1)
    q = np.cross(s, e1[None])
    v = f * (D[:, None, :] * q).sum(-1)
    t = f * (e2[None] * q).sum(-1)
    return bool(np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)))


def ray_down(P, A, B, C, above=0.0):
    """Highest y of the triangles straight under each point (allowing `above` metres over it); -inf if none."""
    out = np.full(len(P), -np.inf)
    if not len(A):
        return out
    for s in range(0, len(P), 256):
        p = P[s:s + 256][:, None, :]
        x, z = p[..., 0], p[..., 2]
        # barycentric in xz
        v0x, v0z = (C - A)[:, 0], (C - A)[:, 2]
        v1x, v1z = (B - A)[:, 0], (B - A)[:, 2]
        v2x, v2z = x - A[:, 0], z - A[:, 2]
        d00 = v0x * v0x + v0z * v0z; d01 = v0x * v1x + v0z * v1z; d11 = v1x * v1x + v1z * v1z
        d02 = v0x * v2x + v0z * v2z; d12 = v1x * v2x + v1z * v2z
        den = d00 * d11 - d01 * d01
        ok = np.abs(den) > 1e-14
        inv = 1.0 / np.where(ok, den, 1.0)
        u = (d11 * d02 - d01 * d12) * inv
        v = (d00 * d12 - d01 * d02) * inv
        hit = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
        y = A[:, 1] + u * (C - A)[:, 1] + v * (B - A)[:, 1]
        y = np.where(hit & (y <= p[..., 1] + above), y, -np.inf)
        out[s:s + 256] = y.max(1)
    return out


def thickness(P):
    if len(P) < 3:
        return 0.0, 0.0, np.array([0, 1.0, 0])
    Q = P - P.mean(0)
    try:
        _, _, vt = np.linalg.svd(Q, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, np.array([0, 1.0, 0])
    pr = Q @ vt.T
    rg = pr.max(0) - pr.min(0)
    if len(rg) < 3:
        return 0.0, float(rg[-1]), vt[-1]
    return float(rg[2]), float(rg[1]), vt[2]


class Grid:
    """Faces bucketed in cubic cells (huge faces kept apart and always returned)."""

    def __init__(self, V, F, cell=None):
        T = V[F] if len(F) else np.zeros((0, 3, 3))
        if cell is None:
            ext = (V.max(0) - V.min(0)).max() if len(V) else 1.0
            cell = max(0.25, float(ext) / 400)
        self.c = cell
        if not len(F):
            self.k = np.zeros(0, np.int64); self.t = np.zeros(0, np.int64); self.big = np.zeros(0, np.int64); return
        lo = np.floor(T.min(1) / cell).astype(np.int64); hi = np.floor(T.max(1) / cell).astype(np.int64)
        span = hi - lo + 1
        n = span.prod(1)
        cap = n > 4000
        self.big = np.nonzero(cap)[0]
        idx = np.nonzero(~cap)[0]
        reps = n[idx]
        tri = np.repeat(idx, reps)
        start = np.repeat(np.cumsum(reps) - reps, reps)
        k = np.arange(len(tri)) - start
        sp = span[tri]
        ox = k % sp[:, 0]; oy = (k // sp[:, 0]) % sp[:, 1]; oz = k // (sp[:, 0] * sp[:, 1])
        key = self._key(lo[tri, 0] + ox, lo[tri, 1] + oy, lo[tri, 2] + oz)
        o = np.argsort(key, kind="stable")
        self.k = key[o]; self.t = tri[o]

    @staticmethod
    def _key(x, y, z):
        return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)

    def query(self, lo, hi):
        a = np.floor(lo / self.c).astype(np.int64); b = np.floor(hi / self.c).astype(np.int64)
        n = np.prod(b - a + 1)
        if n > 60000:
            return None
        X, Y, Z = np.meshgrid(*[np.arange(a[i], b[i] + 1) for i in range(3)], indexing="ij")
        keys = self._key(X, Y, Z).ravel()
        i0 = np.searchsorted(self.k, keys, "left"); i1 = np.searchsorted(self.k, keys, "right")
        parts = [self.t[i:j] for i, j in zip(i0, i1) if j > i]
        parts.append(self.big)
        return np.unique(np.concatenate(parts))


def piece_gap(V, F, T, fl, cand, rng, marg=0.35):
    """Distance between the faces fl and the faces cand (0 if they touch or cross); np.inf if none is within marg.
    Also returns the per-vertex distances of the piece (for decals: how far it stands off its surface)."""
    pts = V[np.unique(F[fl])]
    lo, hi = pts.min(0), pts.max(0)
    if len(pts) > 300:
        pts = pts[rng.choice(len(pts), 300, replace=False)]
    d = np.full(len(pts), np.inf)
    if len(cand):
        Tc = T[cand]
        ok = np.all((Tc.max(1) >= lo - marg) & (Tc.min(1) <= hi + marg), axis=1)
        cand = cand[ok]
    for s in range(0, len(cand), 3000):
        cc = cand[s:s + 3000]
        d = np.minimum(d, closest(pts, T[cc, 0], T[cc, 1], T[cc, 2])[0].min(1))
    gap = float(d.min()) if len(d) else np.inf
    if gap > TOUCH and len(cand):
        Cv = V[np.unique(F[cand])]
        Cv = Cv[np.all((Cv >= lo - marg) & (Cv <= hi + marg), axis=1)]
        if len(Cv):
            if len(Cv) > 1500:
                Cv = Cv[rng.choice(len(Cv), 1500, replace=False)]
            own = fl if len(fl) <= 4000 else fl[rng.choice(len(fl), 4000, replace=False)]
            for s in range(0, len(own), 2000):
                oo = own[s:s + 2000]
                gap = min(gap, float(closest(Cv, T[oo, 0], T[oo, 1], T[oo, 2])[0].min()))
                if gap <= TOUCH:
                    break
    if gap > TOUCH and len(cand):
        cc = cand if len(cand) <= 3000 else cand[rng.choice(len(cand), 3000, replace=False)]
        E = _edges(F, fl, rng, 700)
        hit = seg_hits(V[E[:, 0]], V[E[:, 1]], T[cc, 0], T[cc, 1], T[cc, 2])
        if not hit:
            own = fl if len(fl) <= 3000 else fl[rng.choice(len(fl), 3000, replace=False)]
            E2 = _edges(F, cc, rng, 700)
            hit = seg_hits(V[E2[:, 0]], V[E2[:, 1]], T[own, 0], T[own, 1], T[own, 2])
        if hit:
            gap = 0.0
    return gap, d


def _edges(F, idx, rng, cap):
    E = np.vstack([F[idx][:, [0, 1]], F[idx][:, [1, 2]], F[idx][:, [2, 0]]])
    if len(E) > cap:
        E = E[rng.choice(len(E), cap, replace=False)]
    return E
