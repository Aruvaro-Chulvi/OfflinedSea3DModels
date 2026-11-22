"""
geomcheck — nothing floats, nothing stands off its surface (needs numpy).

    python tools/geomcheck.py                    -> check every models/*/*/model*.glb
    python tools/geomcheck.py Q25173 L1          -> only those
    python tools/geomcheck.py --verbose …        -> list every problem piece

A model fails when it has
  - a piece that touches nothing and does not rest on the seabed (more than 5 cm from everything), or
  - a decal (a thin flat piece lying along a surface: window, porthole, painted marking, letters) standing more
    than 2 cm off it as a whole, or tilted off it by more than 6 cm on average (a drip of rust is a bump on its
    wall: its own relief is not a gap).
Exceptions that are right as they are (a cable that really hangs free…) go in tools/geomcheck_allow.json:
    {"Q123/model.glb": [{"center": [x, y, z], "why": "…"}]}   (matched within 1 m)

Run in CI by .github/workflows/check.yml; tools/fixkit.py makes models pass it when they are built.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geomkit import TOUCH, Grid, closest, components, piece_gap, thickness  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
NORM = {np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0}
GROUND = ("sediment", "mud", "stones", "reef", "rock", "seabed", "floor", "sand", "ground")
FLOAT_GAP, DECAL_GAP, DECAL_TILT, MASS_INSTANCES = 0.05, 0.02, 0.06, 500


def _read(p):
    b = Path(p).read_bytes()
    jl = struct.unpack_from("<I", b, 12)[0]
    g = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl = struct.unpack_from("<I", b, off)[0]
    return g, b[off + 8:off + 8 + bl]


def _acc(g, binc, i):
    a = g["accessors"][i]; v = g["bufferViews"][a["bufferView"]]
    dt = np.dtype(CT[a["componentType"]]); n = NC[a["type"]]
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride") or dt.itemsize * n
    cnt = a["count"]
    raw = np.frombuffer(binc, np.uint8, count=stride * (cnt - 1) + dt.itemsize * n, offset=off)
    out = np.lib.stride_tricks.as_strided(raw, shape=(cnt, n * dt.itemsize), strides=(stride, 1)).copy().view(dt).reshape(cnt, n).astype(np.float64)
    if a.get("normalized"):
        out /= NORM[CT[a["componentType"]]]
    return out


def _q(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _trs(n):
    if "matrix" in n:
        return np.array(n["matrix"]).reshape(4, 4).T
    M = np.eye(4)
    M[:3, :3] = _q(n.get("rotation", [0, 0, 0, 1])) @ np.diag(n.get("scale", [1, 1, 1]))
    M[:3, 3] = n.get("translation", [0, 0, 0])
    return M


def load(path):
    """-> list of frames; each frame = (V, F, ground mask per face). Pieces placed apart (top nodes with a transform
    of their own, meta.json 'pieces') are their own frame: Offlined puts them on their own seabed."""
    g, binc = _read(path)
    frames = []

    def walk(i, M, ground, acc):
        n = g["nodes"][i]
        nm = (n.get("name") or "").lower()
        ground = ground or nm.startswith(GROUND)
        Mi = M @ _trs(n)
        if "mesh" in n:
            ext = (n.get("extensions") or {}).get("EXT_mesh_gpu_instancing")
            mats = [np.eye(4)]
            gi = ground
            if ext:
                at = ext["attributes"]; cnt = g["accessors"][next(iter(at.values()))]["count"]
                T = _acc(g, binc, at["TRANSLATION"]) if "TRANSLATION" in at else np.zeros((cnt, 3))
                R = _acc(g, binc, at["ROTATION"]) if "ROTATION" in at else np.tile([0, 0, 0, 1.0], (cnt, 1))
                S = _acc(g, binc, at["SCALE"]) if "SCALE" in at else np.ones((cnt, 3))
                gi = ground or cnt > MASS_INSTANCES        # thousands of small things strewn on the bottom
                mats = []
                for k in range(cnt):
                    m = np.eye(4); m[:3, :3] = _q(R[k]) @ np.diag(S[k]); m[:3, 3] = T[k]; mats.append(m)
            for pr in g["meshes"][n["mesh"]]["primitives"]:
                if pr.get("mode", 4) != 4:
                    continue
                mat = g["materials"][pr["material"]].get("name", "") if "material" in pr else ""
                p = _acc(g, binc, pr["attributes"]["POSITION"])
                f = _acc(g, binc, pr["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in pr else np.arange(len(p)).reshape(-1, 3)
                for m in mats:
                    W = Mi @ m
                    acc.append((p @ W[:3, :3].T + W[:3, 3], f, gi or mat in ("sediment", "rock")))
        for c in n.get("children", []):
            walk(c, Mi, ground, acc)

    main = []
    for r in g["scenes"][g.get("scene", 0)]["nodes"]:
        n = g["nodes"][r]
        if any(k in n for k in ("translation", "rotation", "matrix")):
            acc = []
            walk(r, np.eye(4), False, acc)      # in its own frame (its own seabed under it)
            frames.append(acc)
        else:
            walk(r, np.eye(4), False, main)
    frames.insert(0, main)
    out = []
    for acc in frames:
        if not acc:
            continue
        offs = np.cumsum([0] + [len(a[0]) for a in acc])
        V = np.vstack([a[0] for a in acc])
        F = np.vstack([a[1] + offs[k] for k, a in enumerate(acc)])
        G = np.concatenate([np.full(len(a[1]), a[2]) for a in acc])
        out.append((V, F, G))
    return out


def check(path, built=False):
    """-> list of problems: {"kind": "floating"|"decal", "center", "size", "gap"}"""
    probs = []
    rng = np.random.default_rng(1)
    for V, F, G in load(path):
        cid = components(V, F)
        cid[G] = -1
        T = V[F]
        area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        order = np.argsort(cid, kind="stable")
        starts = np.searchsorted(cid[order], np.arange(cid.max() + 2))
        comps = [order[starts[c]:starts[c + 1]] for c in range(cid.max() + 1)]
        if not comps:
            continue
        biggest = int(np.argmax([area[fl].sum() for fl in comps]))
        thin = np.zeros(len(comps), bool)
        for c, fl in enumerate(comps):
            if 0 < len(fl) <= 3000:
                th, mid, _ = thickness(V[np.unique(F[fl])])
                thin[c] = th < 0.04 and mid > 0.12
        grid = Grid(V, F)
        thin_f = np.zeros(len(F), bool)
        for c in np.nonzero(thin)[0]:
            thin_f[comps[c]] = True
        for c, fl in enumerate(comps):
            if not len(fl) or c == biggest:
                continue
            P = V[np.unique(F[fl])]
            lo, hi = P.min(0), P.max(0)
            if lo[1] <= 0.12 and not built:
                continue                                    # on the seabed
            cand = grid.query(lo - 0.35, hi + 0.35)
            if cand is None:
                continue
            cand = cand[cid[cand] != c]
            if thin[c]:
                cand = cand[~thin_f[cand]]
            gap, d = piece_gap(V, F, T, fl, cand, rng)
            along = False
            if thin[c] and np.isfinite(d).mean() > 0.8 and len(cand):
                # a decal lies along its surface; a plate leaning on the hull edgewise is not one
                Pn = P if len(P) <= 200 else P[rng.choice(len(P), 200, replace=False)]
                cc = cand if len(cand) <= 3000 else cand[rng.choice(len(cand), 3000, replace=False)]
                dd, _ = closest(Pn, T[cc, 0], T[cc, 1], T[cc, 2])
                sn = np.cross(T[cc, 1] - T[cc, 0], T[cc, 2] - T[cc, 0])
                sn /= np.maximum(np.linalg.norm(sn, axis=1, keepdims=True), 1e-12)
                _, _, nrm = thickness(P)
                along = np.median(np.abs(sn[dd.argmin(1)] @ nrm)) > 0.75
            box = {"center": [round(float(x), 2) for x in (lo + hi) / 2], "size": [round(float(x), 2) for x in hi - lo]}
            if along:
                df = d[np.isfinite(d)]
                p10, p50 = float(np.percentile(df, 10)), float(np.median(df))
                if p10 > DECAL_GAP or p50 > DECAL_TILT:          # lifted off as a whole, or tilted off its surface
                    probs.append({"kind": "decal", "gap": round(p50, 3), **box})
                    continue
            if gap > FLOAT_GAP and area[fl].sum() > 0.02 and not built:
                probs.append({"kind": "floating", "gap": None if not math.isfinite(gap) else round(gap, 3), **box})
            elif gap > 0.3 and area[fl].sum() > 0.02 and built:
                probs.append({"kind": "floating", "gap": None if not math.isfinite(gap) else round(gap, 3), **box})
    return probs


def main():
    args = sys.argv[1:]
    verbose = "--verbose" in args
    want = {a for a in args if not a.startswith("--")}
    allow = {}
    ap = ROOT / "tools" / "geomcheck_allow.json"
    if ap.exists():
        allow = json.loads(ap.read_text("utf-8"))
    bad = 0
    for p in sorted(ROOT.glob("models/*/*/model*.glb")):
        if want and p.parent.name not in want:
            continue
        key = f"{p.parent.name}/{p.name}"
        probs = check(p, built=p.name == "model_built.glb")
        ok_c = [np.array(a["center"]) for a in allow.get(key, [])]
        probs = [q for q in probs if not any(np.linalg.norm(np.array(q["center"]) - c) < 1.0 for c in ok_c)]
        if probs:
            bad += 1
            fl = sum(1 for q in probs if q["kind"] == "floating"); dc = len(probs) - fl
            print(f"FAIL {key}: {fl} floating piece(s), {dc} decal(s) off their surface")
            for q in probs[: (None if verbose else 3)]:
                print(f"     {q['kind']:8s} at {q['center']} size {q['size']} gap {q['gap']}")
        elif verbose:
            print(f"ok   {key}")
    print(f"{bad} model file(s) with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
