"""
fixkit — the last pass before a model is written: nothing floats, nothing stands off its surface (MIT).

    fix_meshes(objects, ground, settle=True) -> report

objects: the Meshes of the model (wreck, fittings, debris); ground: the Meshes of the seabed (sediment, rock, mud).
Both are changed in place (objects only; the ground is a support, never moved).

1. Decals: thin flat pieces (windows, portholes, painted stars and crosses, letters, door panels) lying on a
   surface but a little off it, or crossing it, are laid on it, 6 mm out along its normal (layered pieces keep
   their order: a star stays over its disc). Flat pieces on a curved surface are subdivided until they follow it.
2. Settling (wrecks and places, not ships as built): a piece that touches nothing goes straight down until it
   rests on what is under it (another piece, the sediment, or the seabed at y = 0) and sinks 2 cm into sand.

The check is the same as tools/geomcheck.py, so a model that went through this pass also passes the check.
"""
from __future__ import annotations

import numpy as np

from geomkit import TOUCH, Grid, closest, components, piece_gap, ray_down, thickness
from meshkit import Mesh

THIN = 0.05            # a piece thinner than this (and wider than WIDE) is a decal or a plate
WIDE = 0.08
DECAL_REACH = 3.0      # a decal is laid on a surface no farther than this (if it lies along it)
DECAL_AREA = 400.0     # bigger thin pieces are plates, not decals (a flight deck's lift outline is ~180 m²)
OFFSET, LAYER = 0.006, 0.012
DEBUG = __import__('os').environ.get('FIXKIT_DEBUG') == '1'


def _pack(objects, ground):
    ms = list(objects) + list(ground)
    offs = np.cumsum([0] + [len(m.v) for m in ms])
    V = np.vstack([m.v for m in ms]) if ms else np.zeros((0, 3))
    C = np.vstack([m.c for m in ms]) if ms else np.zeros((0, 3))
    F = np.vstack([m.f + offs[k] for k, m in enumerate(ms)]) if ms else np.zeros((0, 3), np.int64)
    fm = np.concatenate([np.full(len(m.f), k) for k, m in enumerate(ms)]) if ms else np.zeros(0, np.int64)
    vm = np.concatenate([np.full(len(m.v), k) for k, m in enumerate(ms)]) if ms else np.zeros(0, np.int64)
    return ms, V, F, C, fm, vm


def _unpack(ms, n_obj, V, F, C, fm, vm):
    for k in range(n_obj):
        vi = np.nonzero(vm == k)[0]
        remap = np.full(len(V), -1)
        remap[vi] = np.arange(len(vi))
        m = ms[k]
        m.v, m.c, m.f = V[vi].copy(), C[vi].copy(), remap[F[fm == k]]


def _face_normals(T):
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def _subdivide(V, F, C, vm, faces):
    """Split the given faces in four (shared edge midpoints). The new faces are appended; the caller marks the old
    ones dead (indices of the other faces never change). Returns V, F, C, vm and the new faces' ids."""
    sub = F[faces]
    E = np.sort(np.vstack([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]]), 1)
    uE, inv = np.unique(E, axis=0, return_inverse=True)
    inv = inv.ravel()
    mid = len(V) + np.arange(len(uE))
    V = np.vstack([V, (V[uE[:, 0]] + V[uE[:, 1]]) / 2])
    C = np.vstack([C, (C[uE[:, 0]] + C[uE[:, 1]]) / 2])
    vm = np.concatenate([vm, vm[uE[:, 0]]])
    n = len(faces)
    m01, m12, m20 = mid[inv[:n]], mid[inv[n:2 * n]], mid[inv[2 * n:]]
    a, b, c = sub[:, 0], sub[:, 1], sub[:, 2]
    new = np.vstack([np.stack([a, m01, m20], 1), np.stack([m01, b, m12], 1), np.stack([m20, m12, c], 1), np.stack([m01, m12, m20], 1)])
    ids = np.arange(len(F), len(F) + len(new))
    return V, np.vstack([F, new]), C, vm, ids


def fix_meshes(objects, ground=(), settle=True, log=None, reach=None):
    """settle: True for a wreck or a place (loose pieces go down onto what is under them, or the short way onto
    what they nearly touch, up to 3 m); "near" for a ship as built (only the short way, up to 2 m: a funnel or a
    deckhouse set a little off its deck goes onto it; nothing falls)."""
    near_reach = reach if reach is not None else (2.0 if settle == "near" else 3.0)
    objects = [m for m in objects if m is not None and len(m.f)]
    ground = [m for m in ground if m is not None and len(m.f)]
    rep = {"decals": 0, "settled": 0, "floating": 0, "max_drop": 0.0, "removed": 0}
    if not objects:
        return rep
    n_obj = len(objects)
    ms, V, F, C, fm, vm = _pack(objects, ground)
    rng = np.random.default_rng(7)
    isg = fm >= n_obj

    # ── 1. decals ──
    cid = components(V, F)
    cid[isg] = -1
    order = np.argsort(cid, kind="stable")
    ids = cid[order]
    starts = np.searchsorted(ids, np.arange(cid.max() + 2))
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    thin = {}
    for c in range(cid.max() + 1):
        fl = order[starts[c]:starts[c + 1]]
        if not len(fl) or len(fl) > 3000 or area[fl].sum() > DECAL_AREA:
            continue
        P = V[np.unique(F[fl])]
        th, mid_ext, nrm = thickness(P)
        if th < THIN and mid_ext > WIDE:
            fn = _face_normals(T[fl])
            face = (fn * area[fl, None]).sum(0)
            if np.linalg.norm(face) > 1e-9 and np.dot(face, nrm) < 0:
                nrm = -nrm
            thin[c] = (fl, nrm, th)
    solid_f = np.ones(len(F), bool)
    for fl, _, _ in thin.values():
        solid_f[fl] = False
    alive = np.ones(len(F), bool)
    attached = []                                   # (decal vertices, face it lies on)
    drop_f = []
    grid = Grid(V, F)
    plan = []                                       # (comp, faces, normal, mean signed distance)
    WATCH = __import__('os').environ.get('FIXKIT_WATCH')
    WATCH = np.array([float(x) for x in WATCH.split(',')]) if WATCH else None
    for c, (fl, nrm, th) in thin.items():
        P = V[np.unique(F[fl])]
        watch = WATCH is not None and np.linalg.norm(P.mean(0) - WATCH) < 1.5
        lo, hi = P.min(0) - DECAL_REACH, P.max(0) + DECAL_REACH
        cand = grid.query(lo, hi)
        if cand is None:
            continue
        cand = cand[solid_f[cand]]
        if not len(cand):
            continue
        Tc = T[cand]
        cand = cand[np.all((Tc.max(1) >= lo) & (Tc.min(1) <= hi), axis=1)]
        if not len(cand):
            continue
        d, Q = closest(P, T[cand, 0], T[cand, 1], T[cand, 2])
        j = d.argmin(1)
        dmin = d[np.arange(len(P)), j]
        sn = _face_normals(T[cand[j]])
        par = np.abs((sn * nrm).sum(1))
        far = np.percentile(dmin, 90) > 1.5
        qn = Q[np.arange(len(P)), j]
        if watch:
            print("  WATCH", np.round(P.mean(0), 2), "th", round(th, 3), "dmin", round(float(dmin.min()), 3), "p90", round(float(np.percentile(dmin, 90)), 3),
                  "par", round(float(np.median(par)), 2), "sn.nrm", round(float(np.median((sn * nrm).sum(1))), 2),
                  "front", round(float(np.median(((P - qn) * sn).sum(1))), 3), "dark", round(float((C[np.unique(F[fl])].mean(1) < 0.12).mean()), 2))
        if (th < 0.03 and area[fl].sum() < 1.5 and (C[np.unique(F[fl])].mean(1) < 0.12).mean() >= 0.25
                and np.median(((P - qn) * sn).sum(1)) > 0 and np.median((sn * nrm).sum(1)) < -0.5 and dmin.min() < 0.5):
            drop_f.append(fl)                       # a window facing into the wall it lies on (a squashed house's far side): out
            continue
        close = np.percentile(dmin, 90) < 0.5       # right by a surface (a wall bent by a collapse): lay it on, as it is
        if np.percentile(dmin, 90) > DECAL_REACH or np.median(par) < (0.7 if far else 0.2 if close else 0.5):   # no surface it lies along
            # a small dark one (a window, a porthole) with no plating behind it: a mistake of the layout, out
            if th < 0.03 and area[fl].sum() < 1.5 and (C[np.unique(F[fl])].mean(1) < 0.12).mean() >= 0.25 and dmin.min() > TOUCH:
                drop_f.append(fl)
                if DEBUG and len(drop_f) <= 12:
                    print("  removed", len(fl), "faces at", np.round(P.mean(0), 2), "size", np.round(np.ptp(P, 0), 2),
                          "p90", round(float(np.percentile(dmin, 90)), 2), "par", round(float(np.median(par)), 2), "dmin", round(float(dmin.min()), 3))
            continue
        if dmin.max() < 0.003 and np.median(par) > 0.98:
            continue                                # already on it
        sd = ((P - Q[np.arange(len(P)), j]) * nrm).sum(1).mean()
        plan.append((c, fl, nrm, sd, P.min(0), P.max(0)))
    for fl in drop_f:
        alive[fl] = False
    rep["removed"] = len(drop_f)
    # layers: overlapping decals keep their order (a star stays over its disc): i is over j when its points are
    # on the outer side of j's plane
    layer = {}
    plan = [(c, fl, nrm if sd >= 0 else -nrm, abs(sd), lo, hi) for (c, fl, nrm, sd, lo, hi) in plan]
    cen = {c: V[np.unique(F[fl])].mean(0) for (c, fl, *_r) in plan}
    pts = {c: V[np.unique(F[fl])] for (c, fl, *_r) in plan}
    by_cell = {}
    for k, (c, fl, nrm, sd, lo, hi) in enumerate(plan):
        for key in {tuple(np.floor(x / 2.0).astype(int)) for x in (lo, hi, (lo + hi) / 2)}:
            by_cell.setdefault(key, []).append(k)
    for k, (c, fl, nrm, sd, lo, hi) in enumerate(plan):
        near = {j for key in {tuple(np.floor(x / 2.0).astype(int)) for x in (lo, hi, (lo + hi) / 2)} for j in by_cell[key]}
        n_under = 0
        for j in near:
            cj, flj, nj, sdj, loj, hij = plan[j]
            if j == k or not (np.all(loj <= hi + 0.01) and np.all(hij >= lo - 0.01)):
                continue
            if abs(float(np.dot(nrm, nj))) < 0.7:
                continue
            if float(((pts[c] - cen[cj]) @ nj).mean()) > float(((pts[cj] - cen[c]) @ nrm).mean()) + 0.003:
                n_under += 1
        layer[c] = n_under
    if DEBUG:
        for c, fl, nrm, sd, lo, hi in plan:
            if lo[0] > 9 and hi[0] < 13:
                print("  plan", c, len(fl), "sd", round(float(sd), 4), "layer", layer[c], np.round(lo, 2), np.round(hi, 2))
    for c, fl, nrm, sd, lo, hi in plan:
        off = OFFSET + LAYER * min(layer[c], 1)
        P = V[np.unique(F[fl])]
        lo2, hi2 = P.min(0) - DECAL_REACH, P.max(0) + DECAL_REACH
        cand = grid.query(lo2, hi2)
        if cand is None:
            continue
        cand = cand[solid_f[cand] & alive[cand]]
        Tc = T[cand]
        cand = cand[np.all((Tc.max(1) >= lo2) & (Tc.min(1) <= hi2), axis=1)]
        if not len(cand):
            continue
        A_, B_, C_ = T[cand, 0], T[cand, 1], T[cand, 2]
        facing = np.abs((_face_normals(T[cand]) * nrm).sum(1)) > 0.5
        faces = fl
        for level in range(4):
            vi = np.unique(F[faces])
            P = V[vi]
            d, Q = closest(P, A_, B_, C_)
            front = ((P[:, None, :] - Q) * nrm).sum(-1) > -0.05      # the surface on the decal's own side
            dm = np.where(facing[None] & front, d, np.inf)
            j = np.where(np.isfinite(dm.min(1)), dm.argmin(1), d.argmin(1))
            q = Q[np.arange(len(P)), j]
            sn = _face_normals(T[cand[j]])
            sn = np.where(((sn * nrm).sum(1) < 0)[:, None], -sn, sn)
            V[vi] = q + sn * off
            sup_face = int(np.bincount(cand[j]).argmax())            # the face it is laid on (it moves with it)
            # does it follow the surface? (the middle of each face must stay at the same height over it)
            cen = V[F[faces]].mean(1)
            dc, _ = closest(cen, A_, B_, C_)
            dev = np.abs(np.where(facing[None], dc, np.inf).min(1) - off)
            bad = dev > 0.004
            if not bad.any() or level == 3 or len(faces) > 8 * len(fl) + 200:
                if DEBUG and level:
                    print("  decal subdivided", level, "x:", len(fl), "->", len(faces), "faces at", np.round(V[np.unique(F[faces])].mean(0), 1))
                break
            badf = faces[bad]
            V, F, C, vm, newf = _subdivide(V, F, C, vm, badf)
            fm = np.concatenate([fm, np.repeat(fm[badf], 4)])
            solid_f = np.concatenate([solid_f, np.zeros(len(newf), bool)])
            alive = np.concatenate([alive, np.ones(len(newf), bool)])
            alive[badf] = False
            faces = np.concatenate([faces[~bad], newf])
        # a closed thin box (a disc, raised letters) is flat now: its back and sides would only fight its face
        fnn = _face_normals(V[F[faces]])
        dots = fnn @ nrm
        if (dots > 0.5).any() and (dots < -0.5).any():
            alive[faces[dots <= 0.5]] = False
        elif not (dots > 0.5).any() and (dots < -0.5).any():
            alive[faces[np.abs(dots) <= 0.5]] = False
        else:
            alive[faces[np.abs(dots) <= 0.5]] = False
        attached.append((np.unique(F[faces]), sup_face))
        rep["decals"] += 1

    newidx = np.cumsum(alive) - 1
    F, fm = F[alive], fm[alive]

    # ── 2. settling ──
    if settle:
        T = V[F]
        cid = components(V, F)
        cid[fm >= n_obj] = -1
        order = np.argsort(cid, kind="stable")
        starts = np.searchsorted(cid[order], np.arange(cid.max() + 2))
        comps = [order[starts[c]:starts[c + 1]] for c in range(cid.max() + 1)]
        # decals ride on the piece they lie on: not settled on their own, moved with it
        rides = {}
        vcomp = np.full(len(V), -1)
        for c_, fl_ in enumerate(comps):
            vcomp[F[fl_].ravel()] = c_
        ride_comps = set()
        for verts, sf in attached:
            sc = cid[newidx[sf]] if sf < len(newidx) and newidx[sf] >= 0 else -1
            dc_ = vcomp[verts[0]]
            if sc >= 0 and dc_ >= 0 and sc != dc_:
                rides.setdefault(sc, []).append(verts)
                ride_comps.add(dc_)
        ridev = {sc: np.unique(np.concatenate(vs)) for sc, vs in rides.items()}
        comps = [(c_, fl_) for c_, fl_ in enumerate(comps) if len(fl_) and c_ not in ride_comps]
        comps.sort(key=lambda cf: V[F[cf[1]]][..., 1].min())
        grid = Grid(V, F)
        ybot = float(V[:, 1].min()) - 1.0                  # the lowest thing in the model: nothing to find under it
        moved = np.zeros(0, np.int64)
        gone_f, gone_c = [], set()
        gone_m = np.zeros(len(F), bool)                 # faces taken out: no longer a support
        csize = np.bincount(cid[cid >= 0], minlength=cid.max() + 1)
        solid_c = (cid < 0) | (csize[np.maximum(cid, 0)] >= 24)       # pieces that can hold another (the ground, real pieces)
        for pass_ in range(4):                     # again until nothing moves (a support that moved leaves what
            n_before = rep["settled"]             # rested on it hanging)
            rep["floating"] = 0
            for c_id, fl in comps:
                if c_id in gone_c:
                    continue
                if len(fl) == 0:
                    continue
                vi = np.unique(F[fl])
                P = V[vi]
                lo, hi = P.min(0), P.max(0)
                if lo[1] <= 0.02 and settle is True:
                    continue                                    # on (or in) the seabed
                cand = grid.query(lo - near_reach, hi + near_reach)
                if cand is None:
                    continue
                own = np.zeros(len(F), bool); own[fl] = True
                cand = np.union1d(cand, moved)
                cand = cand[~own[cand] & ~gone_m[cand]]
                gap, _ = piece_gap(V, F, T, fl, cand, rng)
                if WATCH is not None and np.linalg.norm(P.mean(0) - WATCH) < 3:
                    print("  WATCH-settle pass", pass_, np.round(P.mean(0), 2), np.round(np.ptp(P, 0), 2), "gap", gap, "faces", len(fl))
                if gap <= TOUCH:
                    continue
                if len(fl) <= 2:                            # a stray triangle left by a collapse: out
                    gone_f.append(fl); gone_c.add(c_id); gone_m[fl] = True
                    rep["removed"] += 1
                    continue
                if len(fl) <= 400 and area_of(T, fl) < 20.0 and (C[vi].mean(1) < 0.12).mean() >= 0.25 and thickness(P)[0] < 0.03:
                    gone_f.append(fl)                       # a window (or a row of them) with no wall left: out
                    gone_c.add(c_id)
                    gone_m[fl] = True
                    rep["removed"] += 1
                    continue
                if len(cand) and (gap < near_reach or not np.isfinite(gap)):
                    # close to something (beside it, under an overhang): move it the short way until it touches
                    mv = _nearest_move(V, F, T, fl, cand[solid_c[cand]], rng, near_reach)   # onto something solid
                    if mv is not None:
                        V[vi] += mv
                        if c_id in ridev:
                            V[ridev[c_id]] += mv
                        T = V[F]
                        moved = np.union1d(moved, fl)
                        rep["settled"] += 1
                        rep["max_drop"] = max(rep["max_drop"], float(np.linalg.norm(mv)))
                        continue
                if settle == "near":
                    rep["floating"] += 1
                    continue
                # straight down: what is under the piece's vertices, and what pokes up under its faces
                allc = grid.query(np.array([lo[0], ybot, lo[2]]), np.array([hi[0], hi[1], hi[2]]))
                if allc is None:
                    rep["floating"] += 1
                    continue
                allc = np.union1d(allc, moved)
                allc = allc[~own[allc] & ~gone_m[allc]]
                Pq = P if len(P) <= 600 else P[rng.choice(len(P), 600, replace=False)]
                under = ray_down(Pq, T[allc, 0], T[allc, 1], T[allc, 2])
                drop = float(np.min(Pq[:, 1] - np.maximum(under, 0.0)))
                Cv = V[np.unique(F[allc])]
                Cv = Cv[(Cv[:, 0] >= lo[0]) & (Cv[:, 0] <= hi[0]) & (Cv[:, 2] >= lo[2]) & (Cv[:, 2] <= hi[2]) & (Cv[:, 1] < hi[1])]
                if len(Cv):
                    if len(Cv) > 1500:
                        Cv = Cv[rng.choice(len(Cv), 1500, replace=False)]
                    A_, B_, C_ = T[fl, 0], T[fl, 1], T[fl, 2]
                    # the piece's underside straight over each support vertex: flip y and look "down" from below
                    flip = np.array([1, -1, 1.0])
                    over = ray_down(Cv * flip, A_ * flip, B_ * flip, C_ * flip)
                    ok = np.isfinite(over)
                    if ok.any():
                        drop = min(drop, float(np.min(-over[ok] - Cv[ok, 1])))
                if not np.isfinite(drop) or drop <= 0 or drop > 40:
                    rep["floating"] += 1
                    continue
                on_ground = abs((P[:, 1] - drop).min()) < 0.03
                drop += 0.02 if on_ground else 0.002
                V[vi] -= np.array([0, drop, 0])
                if c_id in ridev:
                    V[ridev[c_id]] -= np.array([0, drop, 0])
                if DEBUG and drop > 2.0:
                    print("  dropped", len(fl), "faces", round(drop, 2), "m from", np.round(P.mean(0), 1), "size", np.round(np.ptp(P, 0), 1))
                T = V[F]
                moved = np.union1d(moved, fl)
                rep["settled"] += 1
                rep["max_drop"] = max(rep["max_drop"], drop)
            if rep["settled"] == n_before:
                break
    if settle and gone_f:
        keep = np.ones(len(F), bool)
        keep[np.concatenate(gone_f)] = False
        F, fm = F[keep], fm[keep]
    _unpack(ms, n_obj, V, F, C, fm, vm)
    if log:
        log(f"fixkit: {rep['decals']} decals laid on their surface ({rep['removed']} with nothing behind removed), {rep['settled']} loose pieces settled "
            f"(largest drop {rep['max_drop']:.2f} m), {rep['floating']} left as they were")
    return rep


def _nearest_move(V, F, T, fl, cand, rng, reach=0.6):
    """The shortest move (no longer than reach) that brings the faces fl into contact with the faces cand (2 mm in)."""
    P = V[np.unique(F[fl])]
    if len(P) > 400:
        P = P[rng.choice(len(P), 400, replace=False)]
    lo, hi = P.min(0) - reach, P.max(0) + reach
    Tc = T[cand]
    cand = cand[np.all((Tc.max(1) >= lo) & (Tc.min(1) <= hi), axis=1)]
    if not len(cand):
        return None
    best, mv = np.inf, None
    for s in range(0, len(cand), 3000):
        cc = cand[s:s + 3000]
        d, Q = closest(P, T[cc, 0], T[cc, 1], T[cc, 2])
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < best:
            best, mv = d[i, j], Q[i, j] - P[i]
    Cv = V[np.unique(F[cand])]
    Cv = Cv[np.all((Cv >= lo) & (Cv <= hi), axis=1)]
    if len(Cv):
        if len(Cv) > 1500:
            Cv = Cv[rng.choice(len(Cv), 1500, replace=False)]
        own = fl if len(fl) <= 3000 else fl[rng.choice(len(fl), 3000, replace=False)]
        d, Q = closest(Cv, T[own, 0], T[own, 1], T[own, 2])
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < best:
            best, mv = d[i, j], Cv[i] - Q[i, j]
    if mv is None or not np.isfinite(best) or best < 1e-6 or best > reach:
        return None
    return mv * (1.0 + 0.002 / best)


def area_of(T, fl):
    t = T[fl]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())
