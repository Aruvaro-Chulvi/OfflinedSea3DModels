"""
Read and write .glb files at the level of JSON + binary chunk (standard library only), and report what weighs what.

    python tools/glbkit.py report [models/item/Q25173/model.glb ...]   -> table of the heaviest models
    python tools/glbkit.py report --top 20 --md                         -> same, as a Markdown table

Used by tools/slim.py (which makes models lighter) and handy on its own.
"""
from __future__ import annotations

import gzip
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMP_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path):
    """-> (gltf dict, bin bytes)"""
    data = Path(path).read_bytes()
    magic, ver, total = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and ver == 2, path
    jl, jt = struct.unpack_from("<II", data, 12)
    g = json.loads(data[20:20 + jl])
    off = 20 + jl
    binb = b""
    if off < len(data):
        bl, bt = struct.unpack_from("<II", data, off)
        binb = data[off + 8:off + 8 + bl]
    return g, binb


def glb_bytes(g, binb):
    js = json.dumps(g, separators=(",", ":"), ensure_ascii=False).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    binb = bytes(binb) + b"\0" * ((4 - len(binb) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(binb)
    return (b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binb), 0x004E4942) + binb)


def view_bytes(g, binb, vi):
    v = g["bufferViews"][vi]
    o = v.get("byteOffset", 0)
    return binb[o:o + v["byteLength"]]


def image_size(data):
    """(width, height, format) of a JPEG / PNG / WebP without decoding it."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24]); return w, h, "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        try:
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(data)); return im.width, im.height, "webp"
        except Exception:
            return 0, 0, "webp"
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1; continue
            m = data[i + 1]
            if m in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", data[i + 5:i + 9]); return w, h, "jpeg"
            ln = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + ln
    return 0, 0, "?"


def stats(path):
    """Bytes by what they are: textures, positions, normals, colours, uvs, indices, instances, json."""
    g, binb = read_glb(path)
    s = {"file": Path(path).stat().st_size, "json": 0, "tex": 0, "pos": 0, "nrm": 0, "col": 0, "uv": 0, "idx": 0,
         "inst": 0, "other": 0, "verts": 0, "tris": 0, "textures": [], "white_col": 0}
    s["json"] = s["file"] - len(binb)
    kind = {}                                            # accessor -> category
    for m in g.get("meshes", []):
        for p in m["primitives"]:
            for a, ai in p["attributes"].items():
                kind[ai] = {"POSITION": "pos", "NORMAL": "nrm", "COLOR_0": "col", "TEXCOORD_0": "uv"}.get(a, "other")
            if "indices" in p:
                kind[p["indices"]] = "idx"
                s["tris"] += g["accessors"][p["indices"]]["count"] // 3
            s["verts"] += g["accessors"][p["attributes"]["POSITION"]]["count"]
    for n in g.get("nodes", []):
        inst = n.get("extensions", {}).get("EXT_mesh_gpu_instancing")
        if inst:
            for ai in inst["attributes"].values():
                kind[ai] = "inst"
    seen = set()
    for ai, k in kind.items():
        a = g["accessors"][ai]
        vi = a.get("bufferView")
        if vi is None or vi in seen:
            continue
        seen.add(vi)
        s[k] += g["bufferViews"][vi]["byteLength"]
    for im in g.get("images", []):
        d = view_bytes(g, binb, im["bufferView"])
        w, h, f = image_size(d)
        s["tex"] += len(d)
        s["textures"].append((w, h, f, len(d)))
    s["gz"] = len(gzip.compress(Path(path).read_bytes(), 6))
    return s


def mb(n):
    return f"{n / 2**20:.2f}"


def report(paths, top=20, md=False):
    rows = []
    for p in paths:
        p = Path(p)
        s = stats(p)
        rows.append((p, s))
    rows.sort(key=lambda r: -r[1]["file"])
    cols = ("file", "gz", "tex", "pos", "nrm", "col", "uv", "idx", "inst", "json")
    tot = {c: sum(r[1][c] for r in rows) for c in cols}
    head = ["model", *cols, "verts", "tris", "textures"]
    sep = " | " if md else "  "
    lines = []
    if md:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
    else:
        lines.append(sep.join(f"{h:>8}" for h in head))
    for p, s in rows[:top]:
        name = f"{p.parent.name}/{p.stem.replace('model', '') or 'wreck'}".replace("/_built", "/built")
        texs = ", ".join(sorted({f"{w}{f[0]}" for w, h, f, n in s["textures"]})) + f" ×{len(s['textures'])}" if s["textures"] else "-"
        vals = [name, *(mb(s[c]) for c in cols), str(s["verts"]), str(s["tris"]), texs]
        lines.append(("| " + " | ".join(vals) + " |") if md else sep.join(f"{v:>8}" for v in vals))
    vals = ["TOTAL", *(mb(tot[c]) for c in cols), "", "", ""]
    lines.append(("| " + " | ".join(vals) + " |") if md else sep.join(f"{v:>8}" for v in vals))
    print("\n".join(lines))
    print(f"\n{len(rows)} file(s); MB = 2^20 bytes; gz = gzip -6 of the whole file")
    return rows


def all_glbs():
    return sorted(ROOT.glob("models/*/*/model*.glb"))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "report":
        sys.exit(__doc__)
    args = args[1:]
    top, md = 20, False
    if "--top" in args:
        i = args.index("--top"); top = int(args[i + 1]); del args[i:i + 2]
    if "--md" in args:
        args.remove("--md"); md = True
    report(args or all_glbs(), top, md)
