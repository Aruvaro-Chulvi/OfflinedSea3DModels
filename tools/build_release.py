"""
Prepare a release: every model as one file (plus "-built.glb", the ship as built, and "-diorama.glb", the
close-up scene, when there are) and manifest.json (what Offlined downloads).

    python tools/build_release.py models-2026-11-14          -> dist/
    gh release create models-2026-11-14 dist/* --title ... --notes ...

Offlined reads https://github.com/Aruvaro-Chulvi/OfflinedSea3DModels/releases/latest/download/manifest.json
and downloads the files it lists (checking size and SHA-256). Standard library only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/Aruvaro-Chulvi/OfflinedSea3DModels"
META_KEYS = ("name", "state", "heading", "position", "depth", "pieces", "authors", "license", "triangles", "based_on")


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
        for d in sorted((ROOT / "models" / kind).glob("Q*")):
            if not (d / "model.glb").exists() or not (d / "meta.json").exists():
                continue
            meta = json.loads((d / "meta.json").read_text("utf-8"))
            name = f"{kind}-{d.name}.glb"
            shutil.copy2(d / "model.glb", out / name)
            data = (out / name).read_bytes()
            m = {"kind": kind, "id": d.name, "file": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            m.update({k: meta[k] for k in META_KEYS if k in meta})
            if (d / "model_built.glb").exists():                     # the ship as built (optional; older Offlined ignores it)
                bname = f"{kind}-{d.name}-built.glb"
                shutil.copy2(d / "model_built.glb", out / bname)
                bdata = (out / bname).read_bytes()
                m["built"] = {"file": bname, "size": len(bdata), "sha256": hashlib.sha256(bdata).hexdigest()}
            if (d / "diorama.glb").exists():                         # the close-up scene (optional; older Offlined ignores it)
                dname = f"{kind}-{d.name}-diorama.glb"
                shutil.copy2(d / "diorama.glb", out / dname)
                ddata = (out / dname).read_bytes()
                m["diorama"] = {"file": dname, "size": len(ddata), "sha256": hashlib.sha256(ddata).hexdigest()}
            models.append(m)
    man = {
        "name": "Offlined Sea 3D Models", "version": tag, "generated": dt.date.today().isoformat(),
        "repository": REPO, "base_url": f"{REPO}/releases/download/{tag}/",
        "license": "Each model: its own licence (CC-BY-4.0 or CC0-1.0), authors in 'authors'.",
        "models": models,
    }
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), "utf-8")
    total = sum(m["size"] + m.get("built", {}).get("size", 0) + m.get("diorama", {}).get("size", 0) for m in models)
    print(f"{len(models)} model(s), {total / 2**20:.2f} MB -> {out}")


if __name__ == "__main__":
    main()
