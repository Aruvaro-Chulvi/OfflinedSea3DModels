"""
Prepare a release: every model as one file (plus "-built.glb", the ship as built, when there is one), each
also gzipped (".glb.gz", listed as "gz": Offlined downloads and keeps that one, and serves it with
Content-Encoding: gzip), and manifest.json (what Offlined downloads).

    python tools/build_release.py models-2026-11-14          -> dist/
    gh release create models-2026-11-14 dist/* --title ... --notes ...

Offlined reads https://github.com/Aruvaro-Chulvi/OfflinedSea3DModels/releases/latest/download/manifest.json
and downloads the files it lists (checking size and SHA-256). Standard library only.

Each model also gets a meshopt-compressed copy (".mo.glb.gz", listed as "mo"; about half the download, same geometry):
tools/meshopt/compress.mjs, which needs Node.js and, once, `npm ci` in tools/meshopt. Offlined 4.x prefers it; older
Offlined ignores "mo" and keeps downloading "gz". Without Node the release is built as before, without "mo".
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/Aruvaro-Chulvi/OfflinedSea3DModels"
META_KEYS = ("name", "state", "heading", "position", "depth", "pieces", "hole", "authors", "license", "triangles", "based_on")


def packed(out, name, data):
    """name.gz next to it (fixed mtime, so the same model always gives the same bytes)"""
    z = gzip.compress(data, 9, mtime=0)
    (out / (name + ".gz")).write_bytes(z)
    return {"file": name + ".gz", "size": len(z), "sha256": hashlib.sha256(z).hexdigest()}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tag = sys.argv[1]
    out = ROOT / "dist"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    models = []
    for kind in ("item", "class", "vehicle"):
        for d in sorted([*(ROOT / "models" / kind).glob("Q*"), *(ROOT / "models" / kind).glob("L*")]):   # L…: own entries
            if not (d / "model.glb").exists() or not (d / "meta.json").exists():
                continue
            meta = json.loads((d / "meta.json").read_text("utf-8"))
            name = f"{kind}-{d.name}.glb"
            shutil.copy2(d / "model.glb", out / name)
            data = (out / name).read_bytes()
            m = {"kind": kind, "id": d.name, "file": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                 "gz": packed(out, name, data)}
            m.update({k: meta[k] for k in META_KEYS if k in meta})
            if (d / "model_built.glb").exists():                     # the ship as built (optional; older Offlined ignores it)
                bname = f"{kind}-{d.name}-built.glb"
                shutil.copy2(d / "model_built.glb", out / bname)
                bdata = (out / bname).read_bytes()
                m["built"] = {"file": bname, "size": len(bdata), "sha256": hashlib.sha256(bdata).hexdigest(),
                              "gz": packed(out, bname, bdata)}
            models.append(m)
    add_meshopt(out, models)
    man = {
        "name": "Offlined Sea 3D Models", "version": tag, "generated": dt.date.today().isoformat(),
        "repository": REPO, "base_url": f"{REPO}/releases/download/{tag}/",
        "license": "Each model: its own licence (CC-BY-4.0 or CC0-1.0), authors in 'authors'.",
        "models": models,
    }
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), "utf-8")
    total = sum(m["size"] + m.get("built", {}).get("size", 0) for m in models)
    gz = sum(m["gz"]["size"] + m.get("built", {}).get("gz", {}).get("size", 0) for m in models)
    mo = sum(m.get("mo", m["gz"])["size"] + m.get("built", {}).get("mo", m.get("built", {}).get("gz", {})).get("size", 0) for m in models)
    print(f"{len(models)} model(s), {total / 2**20:.2f} MB ({gz / 2**20:.2f} MB gzipped, {mo / 2**20:.2f} MB with meshopt) -> {out}")


def add_meshopt(out, models):
    """A meshopt copy of every file (one Node run for all), gzipped and listed as "mo" next to "gz"."""
    tool = ROOT / "tools" / "meshopt" / "compress.mjs"
    node = shutil.which("node")
    if not node or not (tool.parent / "node_modules").is_dir():
        print("meshopt: skipped (needs Node.js and `npm ci` in tools/meshopt)")
        return
    entries = []                                            # (manifest entry, plain .glb name)
    for m in models:
        entries.append((m, m["file"]))
        if "built" in m:
            entries.append((m["built"], m["built"]["file"]))
    args = []
    for _, name in entries:
        args += [str(out / name), str(out / name.replace(".glb", ".mo.glb"))]
    for i in range(0, len(args), 80):                       # (Windows: command lines of limited length)
        subprocess.run([node, str(tool), *args[i:i + 80]], check=False)
    for e, name in entries:
        mname = name.replace(".glb", ".mo.glb")
        f = out / mname
        if f.is_file():                                     # (the tool skips a model whose copy does not match)
            e["mo"] = packed(out, mname, f.read_bytes())
            f.unlink()                                      # only the gzipped one is published


if __name__ == "__main__":
    main()
