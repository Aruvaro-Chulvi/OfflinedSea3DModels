"""
Check every model folder against docs/MODEL_SPEC.md. Standard library only.

    python tools/validate.py            # all models
    python tools/validate.py models/item/Q25173

Exit code 1 if anything is wrong (errors); warnings do not fail.
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("item", "class", "vehicle")
LIMITS = {"item": (8 * 2**20, 200_000), "class": (5 * 2**20, 120_000), "vehicle": (5 * 2**20, 120_000)}
EXT_OK = {"KHR_mesh_quantization", "KHR_texture_transform", "EXT_mesh_gpu_instancing",
          "KHR_materials_emissive_strength", "EXT_texture_webp"}
LICENSES = {"CC-BY-4.0", "CC0-1.0"}
STATES = {"intact", "broken", "scattered", "capsized", "buried"}
QID = re.compile(r"^Q[1-9][0-9]*$")
FILES_OK = {"model.glb", "meta.json", "preview.jpg", "model_built.glb"}


def image_size(b: bytes):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", b[16:24])
    if b[:2] == b"\xff\xd8":
        i = 2
        while i < len(b) - 9:
            if b[i] != 0xFF:
                i += 1; continue
            m = b[i + 1]
            if m in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", b[i + 5:i + 9]); return w, h
            i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        if b[12:16] == b"VP8 ":
            w, h = struct.unpack("<HH", b[26:30]); return w & 0x3FFF, h & 0x3FFF
        if b[12:16] == b"VP8L":
            v = int.from_bytes(b[21:25], "little"); return (v & 0x3FFF) + 1, ((v >> 14) & 0x3FFF) + 1
        if b[12:16] == b"VP8X":
            return int.from_bytes(b[24:27], "little") + 1, int.from_bytes(b[27:30], "little") + 1
    return None


def check_glb(p: Path, kind: str, err, warn):
    b = p.read_bytes()
    max_size, max_tris = LIMITS[kind]
    if len(b) > max_size:
        err(f"{p.name}: {len(b) / 2**20:.1f} MB, more than {max_size // 2**20} MB")
    if len(b) < 20 or b[:4] != b"glTF":
        err(f"{p.name}: not a binary glTF (.glb) file"); return None
    ver, total = struct.unpack("<II", b[4:12])
    if ver != 2:
        err(f"{p.name}: glTF version {ver}, must be 2"); return None
    clen, ctype = struct.unpack("<II", b[12:20])
    if ctype != 0x4E4F534A:
        err(f"{p.name}: first chunk is not JSON"); return None
    g = json.loads(b[20:20 + clen])
    binc = None
    off = 20 + clen
    if off + 8 <= len(b):
        blen, btype = struct.unpack("<II", b[off:off + 8])
        if btype == 0x004E4942:
            binc = b[off + 8:off + 8 + blen]
    for ext in set(g.get("extensionsUsed", [])) | set(g.get("extensionsRequired", [])):
        if ext not in EXT_OK:
            err(f"{p.name}: extension {ext} is not allowed (see MODEL_SPEC §2)")
    for buf in g.get("buffers", []):
        if "uri" in buf:
            err(f"{p.name}: external buffer '{buf['uri'][:40]}': everything must be inside the .glb")
    acc = g.get("accessors", [])
    tris = 0
    for m in g.get("meshes", []):
        for pr in m.get("primitives", []):
            mode = pr.get("mode", 4)
            n = acc[pr["indices"]]["count"] if "indices" in pr else acc[pr["attributes"]["POSITION"]]["count"]
            tris += n // 3 if mode == 4 else max(0, n - 2) if mode in (5, 6) else 0
    if tris > max_tris:
        err(f"{p.name}: {tris:,} triangles, more than {max_tris:,}")
    for k in ("animations", "cameras"):
        if g.get(k):
            warn(f"{p.name}: has {k}; they are ignored")
    if (g.get("extensions") or {}).get("KHR_lights_punctual"):
        warn(f"{p.name}: has lights; they are ignored")
    views = g.get("bufferViews", [])
    for i, im in enumerate(g.get("images", [])):
        if "uri" in im:
            err(f"{p.name}: image {i} is an external file; textures must be inside the .glb"); continue
        if binc is None or "bufferView" not in im:
            continue
        v = views[im["bufferView"]]
        data = binc[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
        sz = image_size(data)
        if sz and max(sz) > 2048:
            err(f"{p.name}: texture {i} is {sz[0]}×{sz[1]} px, more than 2048")
    # size of the whole thing, from the POSITION bounds and each node's own translation/scale (16-bit positions
    # are undone that way); rotations and parents are ignored: a rough check
    lo, hi = [1e18] * 3, [-1e18] * 3
    for nd in g.get("nodes", []):
        if "mesh" not in nd or "EXT_mesh_gpu_instancing" in (nd.get("extensions") or {}):
            continue
        t, sc = nd.get("translation", [0, 0, 0]), nd.get("scale", [1, 1, 1])
        for pr in g["meshes"][nd["mesh"]].get("primitives", []):
            a = acc[pr["attributes"]["POSITION"]]
            if "min" in a and "max" in a:
                lo = [min(x, tt + ss * y) for x, tt, ss, y in zip(lo, t, sc, a["min"])]
                hi = [max(x, tt + ss * y) for x, tt, ss, y in zip(hi, t, sc, a["max"])]
    ext = [h - l for l, h in zip(lo, hi)] if hi[0] > -1e17 else None
    return {"triangles": tris, "extent": ext, "nodes": {n.get("name") for n in g.get("nodes", [])}}


def check_folder(d: Path, catalog: dict, errors: list, warnings: list):
    rel = d.relative_to(ROOT).as_posix()
    err = lambda m: errors.append(f"{rel}: {m}")
    warn = lambda m: warnings.append(f"{rel}: {m}")
    kind, qid = d.parent.name, d.name
    if kind not in KINDS:
        err(f"must be inside models/item, models/class or models/vehicle"); return
    if not QID.match(qid):
        err(f"folder name must be a Wikidata id like Q25173"); return
    for f in d.iterdir():
        if f.name not in FILES_OK:
            err(f"unexpected file {f.name} (allowed: {', '.join(sorted(FILES_OK))})")
    if not (d / "model.glb").exists():
        err("model.glb is missing")
    if not (d / "meta.json").exists():
        err("meta.json is missing"); return
    try:
        m = json.loads((d / "meta.json").read_text("utf-8"))
    except Exception as e:                                  # noqa: BLE001
        err(f"meta.json is not valid JSON: {e}"); return
    if m.get("id") != qid:
        err(f"meta.json id is {m.get('id')!r}, the folder is {qid}")
    if m.get("kind") != kind:
        err(f"meta.json kind is {m.get('kind')!r}, the folder is {kind}")
    if not isinstance(m.get("name"), str) or not m["name"].strip():
        err("meta.json needs a name")
    if m.get("license") not in LICENSES:
        err(f"license must be one of {sorted(LICENSES)}")
    au = m.get("authors")
    if not au or not all(isinstance(a, dict) and str(a.get("name", "")).strip() for a in au):
        err("authors: a list of {name, url?}, at least one")
    so = m.get("sources")
    if not so or not all(isinstance(s, dict) and (s.get("title") or s.get("url")) for s in so):
        err("sources: a list of {title, url}, at least one")
    if "based_on" in m:
        b = m["based_on"]
        if not isinstance(b, dict) or b.get("license") not in LICENSES or not b.get("author"):
            err("based_on needs author, url and a CC-BY-4.0 or CC0-1.0 license")
    if kind == "item":
        if m.get("state") not in STATES:
            err(f"state must be one of {sorted(STATES)}")
        h = m.get("heading")
        if not isinstance(h, (int, float)) or not 0 <= h < 360:
            err("heading: the bow direction in degrees from north, 0–359")
        if qid not in catalog:
            warn("not in catalog/catalog.json (fine if the catalog is older than the model)")
    if "position" in m:
        p = m["position"]
        if not (isinstance(p, dict) and -90 <= p.get("lat", 999) <= 90 and -180 <= p.get("lon", 999) <= 180 and p.get("source")):
            err("position needs lat, lon and source")
    if "depth" in m and not (isinstance(m["depth"], (int, float)) and m["depth"] < 0):
        err("depth is in metres below sea level, negative")
    if "pieces" in m:
        pcs = m["pieces"]
        if not isinstance(pcs, dict) or not pcs:
            err("pieces: an object {node name: {lat, lon, heading?, depth?, source}}")
        else:
            for nm, pc in pcs.items():
                if not (isinstance(pc, dict) and -90 <= pc.get("lat", 999) <= 90 and -180 <= pc.get("lon", 999) <= 180 and pc.get("source")):
                    err(f"pieces.{nm} needs lat, lon and source")
                if "heading" in pc and not (isinstance(pc["heading"], (int, float)) and 0 <= pc["heading"] < 360):
                    err(f"pieces.{nm}.heading must be 0-359")
    for f in ("model.glb", "model_built.glb"):
        if (d / f).exists():
            info = check_glb(d / f, kind, err, warn)
            if info and f == "model_built.glb":
                # the ship as built is whole: its length must match the real one (catches models of something else)
                c = catalog.get(qid)
                if info["extent"] and c and c.get("length"):
                    L = max(info["extent"][0], info["extent"][2])
                    if not 0.5 < L / c["length"] < 2:
                        err(f"model_built.glb is about {L:.0f} m long, Wikidata says {c['length']:.0f} m: is it the right ship, in metres?")
                    elif not 0.85 < L / c["length"] < 1.2:
                        warn(f"model_built.glb is about {L:.0f} m long, Wikidata says {c['length']:.0f} m")
            if info and f == "model.glb":
                c = catalog.get(qid)
                if info["extent"] and c and c.get("length") and m.get("state") not in ("broken", "scattered"):
                    L = max(info["extent"][0], info["extent"][2])
                    if not 0.7 < L / c["length"] < 1.4:
                        warn(f"model is about {L:.0f} m long, Wikidata says {c['length']:.0f} m: is it in metres?")
                if "triangles" not in m:
                    warn(f"{info['triangles']:,} triangles (you can add \"triangles\" to meta.json)")
                for nm in (m.get("pieces") or {}):
                    if nm not in info["nodes"]:
                        err(f"pieces.{nm}: there is no node with that name in model.glb")
    pv = d / "preview.jpg"
    if pv.exists():
        sz = image_size(pv.read_bytes()[:65536])
        if not sz:
            err("preview.jpg is not a JPEG")
        elif sz[0] > 1920:
            warn(f"preview.jpg is {sz[0]}×{sz[1]}; 1280×720 is enough")
    else:
        warn("no preview.jpg (recommended)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cat = ROOT / "catalog" / "catalog.json"
    catalog = {i["q"]: i for i in json.loads(cat.read_text("utf-8"))["items"]} if cat.exists() else {}
    if len(sys.argv) > 1:
        folders = [Path(a).resolve() for a in sys.argv[1:]]
    else:
        folders = [d for k in KINDS for d in sorted((ROOT / "models" / k).glob("*")) if d.is_dir()]
    errors, warnings = [], []
    for d in folders:
        check_folder(d, catalog, errors, warnings)
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("ERROR:  ", e)
    print(f"{len(folders)} model folder(s): {len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
