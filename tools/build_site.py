"""
Assemble the catalog web page (GitHub Pages) in a folder:

    python tools/build_site.py _site

site/* + generator/ + catalog/catalog.json + models/ (glb, diorama, meta.json, previews) + models/index.json.
Standard library only.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "_site").resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)
    shutil.copytree(ROOT / "generator", out / "generator")
    shutil.copy2(ROOT / "catalog" / "catalog.json", out / "catalog.json")
    idx = {"item": {}, "class": {}, "vehicle": {}}
    for kind in idx:
        for d in sorted((ROOT / "models" / kind).glob("Q*")):
            meta = d / "meta.json"
            if not (meta.exists() and (d / "model.glb").exists()):
                continue
            m = json.loads(meta.read_text("utf-8"))
            dst = out / "models" / kind / d.name
            dst.mkdir(parents=True)
            for f in ("model.glb", "model_built.glb", "diorama.glb", "meta.json", "preview.jpg", "diorama.jpg"):
                if (d / f).exists():
                    shutil.copy2(d / f, dst / f)
            idx[kind][d.name] = {k: m.get(k) for k in ("id", "name", "authors", "license", "state", "heading")}
            idx[kind][d.name]["built"] = (d / "model_built.glb").exists()
            idx[kind][d.name]["diorama"] = (d / "diorama.glb").exists()
    (out / "models").mkdir(exist_ok=True)
    (out / "models" / "index.json").write_text(json.dumps(idx, ensure_ascii=False), "utf-8")
    (out / ".nojekyll").write_text("", "utf-8")
    n = sum(len(v) for v in idx.values())
    print(f"site -> {out} ({n} model(s))")


if __name__ == "__main__":
    main()
