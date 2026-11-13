"""
Copy what Offlined's Wiki Submarine needs into its frontend:

    python tools/export_offlined.py "path/to/Offlined/frontend/subsim"

- wrecks.json: the catalog without the fields the game does not use (types, image, coast, pos…)
- sea3d-generator.js: the generic wreck generator
Standard library only.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("q", "name", "fam", "lat", "lon", "depth", "titles", "length", "beam", "built", "sunk", "class", "vehicle", "heading")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dst = Path(sys.argv[1])
    c = json.loads((ROOT / "catalog" / "catalog.json").read_text("utf-8"))
    out = {"source": "https://github.com/Aruvaro-Chulvi/OfflinedSea3DModels", "license": "CC0-1.0",
           "generated": c["generated"], "items": [{k: i[k] for k in KEEP if k in i} for i in c["items"]]}
    (dst / "wrecks.json").write_bytes(json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    shutil.copyfile(ROOT / "generator" / "sea3d-generator.js", dst / "sea3d-generator.js")
    print(f"{len(out['items'])} items -> {dst / 'wrecks.json'}; generator copied")


if __name__ == "__main__":
    main()
