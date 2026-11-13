"""
Build catalog/catalog.json: everything under the sea that has a Wikipedia article.

1. Wikidata (SPARQL): shipwrecks, ships with a shipwrecking, submarines, aviation accidents,
   reefs, hydrothermal vents, underwater archaeological sites and museums that have coordinates
   AND an article in at least one of the Wikipedias Offlined ships (ar de en es fr hi it pt ru zh).
2. Wikidata (API): names, article titles, sizes, dates, class, image… for each of them.
3. GEBCO: the depth at that point. Only points below sea level are kept (museum ships in a
   harbour, accidents over land… fall out; a few coastal ones stay and are marked "coast").
4. catalog/overrides.json: hand corrections from the community (exclude, move, rename…).

    python tools/build_catalog.py --gebco "path/to/seafloor.pmtiles"

Needs Python 3.10+ and Pillow. Downloads are cached in tools/.cache (delete it to refresh).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / ".cache"
UA = "OfflinedSea3DModels-catalog/1.0 (https://github.com/Aruvaro-Chulvi/OfflinedSea3DModels)"
LANGS = ["en", "es", "de", "fr", "it", "pt", "ru", "zh", "ar", "hi"]
WIKIS = " ".join(f"<https://{l}.wikipedia.org/>" for l in LANGS)

# Where the candidates come from (a Wikidata item can come from several)
SOURCES = {
    "wreck":   "?i wdt:P31/wdt:P279* wd:Q852190 .",          # shipwreck
    "sunk":    "?i wdt:P793 wd:Q906512 .",                   # significant event: shipwrecking
    "ship":    "?i wdt:P31/wdt:P279* wd:Q11446 .",           # ship (coordinates may be the wreck or a harbour)
    "sub":     "?i wdt:P31/wdt:P279* wd:Q2811 .",            # submarine
    "air":     "?i wdt:P31/wdt:P279* wd:Q744913 .",          # aviation accident
    "reef":    "?i wdt:P31/wdt:P279* wd:Q11292 .",           # coral reef
    "areef":   "?i wdt:P31/wdt:P279* wd:Q985843 .",          # artificial reef
    "vent":    "?i wdt:P31/wdt:P279* wd:Q867648 .",          # hydrothermal vent
    "site":    "?i wdt:P31/wdt:P279* wd:Q11558926 .",        # underwater archaeological site
    "museum":  "?i wdt:P31/wdt:P279* wd:Q12192226 .",        # underwater museum
}

# Families: which generic model the generator draws. First match wins.
FAMILIES = [
    ("submarine", ["Q2811"]),
    ("carrier",   ["Q17205"]),
    ("sail",      ["Q170483", "Q181469", "Q207452"]),        # sailing ship, galleon, ship of the line
    ("warship",   ["Q2031121", "Q3114762"]),                 # warship (two items on Wikidata)
    ("liner",     ["Q697196", "Q2055880", "Q25653"]),        # ocean liner, passenger vessel, ferry
    ("tanker",    ["Q14928"]),
    ("cargo",     ["Q105999", "Q17210", "Q473932"]),         # cargo, container, ro-ro
    ("fishing",   ["Q1420024"]),
    ("steamer",   ["Q12859788"]),
    ("aircraft",  ["Q11436"]),
    ("reef",      ["Q11292", "Q985843"]),
    ("vent",      ["Q867648"]),
    ("site",      ["Q11558926"]),
    ("museum",    ["Q12192226"]),
    ("ship",      ["Q1229765"]),                             # any other watercraft
]
SOURCE_FAMILY = {"air": "aircraft", "reef": "reef", "areef": "reef","vent": "vent", "site": "site", "museum": "museum"}
FOOT = "http://www.wikidata.org/entity/Q3710"


def http_json(url: str, tries: int = 4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json, application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except Exception as e:                               # noqa: BLE001 (network: retry)
            if k == tries - 1:
                raise
            print(f"  retry ({e})", flush=True)
            time.sleep(5 * (k + 1))


def sparql(q: str):
    return http_json("https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q}))["results"]["bindings"]


def cached(name: str, fn):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / name
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    v = fn()
    p.write_text(json.dumps(v, ensure_ascii=False), "utf-8")
    return v


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


# ── 1. Candidates ────────────────────────────────────────────────────────────
def candidates() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, where in SOURCES.items():
        def run(where=where):
            q = f"""SELECT DISTINCT ?i WHERE {{ {where} ?i wdt:P625 ?c .
                    ?a schema:about ?i ; schema:isPartOf ?w . VALUES ?w {{ {WIKIS} }} }}"""
            return sorted({qid(b["i"]["value"]) for b in sparql(q)})
        ids = cached(f"src_{name}.json", run)
        print(f"{name:7s} {len(ids):6d} with an article", flush=True)
        for i in ids:
            out.setdefault(i, []).append(name)
    return out


# ── 2. Details ───────────────────────────────────────────────────────────────
def entities(ids: list[str]) -> dict:
    """Details of each item; the cache only downloads the ones it does not have yet."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / "entities.json"
    got = json.loads(p.read_text("utf-8")) if p.exists() else {}
    todo = [i for i in ids if i not in got]
    sites = "|".join(f"{l}wiki" for l in LANGS)
    for k in range(0, len(todo), 50):
        part = todo[k:k + 50]
        r = http_json("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
            "action": "wbgetentities", "ids": "|".join(part), "props": "labels|sitelinks|claims",
            "languages": "|".join(LANGS), "sitefilter": sites, "format": "json", "maxlag": 5}))
        got.update(r.get("entities", {}))
        if k // 50 % 20 == 0:
            print(f"  details {k + len(part)}/{len(todo)}", flush=True)
        time.sleep(0.2)
    if todo:
        p.write_text(json.dumps(got, ensure_ascii=False), "utf-8")
    return got


def claims(e, p):
    return [c for c in e.get("claims", {}).get(p, []) if c.get("rank") != "deprecated" and c["mainsnak"].get("snaktype") == "value"]


def val(c):
    return c["mainsnak"]["datavalue"]["value"]


def item_ids(e, p):
    return [val(c)["id"] for c in claims(e, p)]


def metres(e, p):
    for c in claims(e, p):
        v = val(c)
        try:
            a = float(v["amount"])
        except (KeyError, ValueError):
            continue
        return round(a * 0.3048 if v.get("unit") == FOOT else a, 2)
    return None


def year(t: dict | None):
    if not t:
        return None
    try:
        y = int(t["time"][:5].replace("+", ""))
        return y if t["time"][0] == "+" else -y
    except (KeyError, ValueError):
        return None


def first_year(e, props):
    for p in props:
        for c in claims(e, p):
            y = year(val(c))
            if y is not None:
                return y
    return None


def position(e):
    """The wreck site if Wikidata states it (P625 on the shipwrecking event), else the preferred P625."""
    for c in claims(e, "P793"):
        if val(c).get("id") == "Q906512":
            for q in c.get("qualifiers", {}).get("P625", []):
                if q.get("snaktype") == "value":
                    v = q["datavalue"]["value"]
                    return v["latitude"], v["longitude"], "event"
    cs = claims(e, "P625")
    cs = [c for c in cs if c.get("rank") == "preferred"] or cs
    for c in cs:
        v = val(c)
        if v.get("globe", "").endswith("Q2"):
            return v["latitude"], v["longitude"], "item"
    return None


def sunk_year(e):
    for c in claims(e, "P793"):
        if val(c).get("id") == "Q906512":
            for q in c.get("qualifiers", {}).get("P585", []):
                if q.get("snaktype") == "value":
                    return year(q["datavalue"]["value"])
    return first_year(e, ["P585", "P730", "P576"])


# ── 3. Families ──────────────────────────────────────────────────────────────
def families(types: set[str]) -> dict[str, set[str]]:
    """For each type (P31 value or vessel class), the family roots it belongs to."""
    roots = " ".join(f"wd:{q}" for _, qs in FAMILIES for q in qs)
    tl = sorted(types)

    def run():
        got: dict[str, list[str]] = {}
        for k in range(0, len(tl), 150):
            vals = " ".join(f"wd:{t}" for t in tl[k:k + 150])
            q = f"SELECT ?t ?r WHERE {{ VALUES ?t {{ {vals} }} VALUES ?r {{ {roots} }} ?t wdt:P279* ?r . }}"
            for b in sparql(q):
                got.setdefault(qid(b["t"]["value"]), []).append(qid(b["r"]["value"]))
        return got
    key = hashlib.md5((roots + " " + " ".join(tl)).encode()).hexdigest()[:10]
    return {k: set(v) for k, v in cached(f"families_{key}.json", run).items()}


def family_of(types: list[str], fam_map, built: int | None, srcs: list[str]) -> str:
    for s in srcs:
        if s in SOURCE_FAMILY and not any(r in ("Q1229765", "Q11446") for t in types for r in fam_map.get(t, ())):
            return SOURCE_FAMILY[s]
    roots = set().union(*(fam_map.get(t, set()) for t in types)) if types else set()
    for name, qs in FAMILIES:
        if roots & set(qs):
            if name in ("warship", "ship", "steamer") and built is not None and built < 1850:
                return "sail"                              # before steam, a warship or a ship was a sailing ship
            return name
    return "ship"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gebco", required=True, help="GEBCO Terrarium PMTiles (Offlined: Maps/Seafloor/seafloor.pmtiles)")
    ap.add_argument("--out", default=str(ROOT / "catalog" / "catalog.json"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gebco_depth import Gebco

    ov = json.loads((ROOT / "catalog" / "overrides.json").read_text("utf-8")).get("items", {})
    cand = candidates()
    for q, o in ov.items():                                # added by hand (Wikidata classes them otherwise)
        if o.get("include"):
            cand.setdefault(q, []).append("extra")
    print(f"{len(cand)} distinct candidates", flush=True)
    ents = entities(sorted(cand))
    types = set()
    for e in ents.values():
        types.update(item_ids(e, "P31")); types.update(item_ids(e, "P289"))
    fam_map = families(types)
    g = Gebco(a.gebco)

    items, dropped = [], Counter()
    for q in sorted(cand, key=lambda s: int(s[1:])):
        e = ents.get(q)
        if not e or "missing" in e:
            dropped["missing"] += 1; continue
        o = ov.get(q, {})
        if o.get("exclude"):
            dropped["excluded"] += 1; continue
        pos = position(e)
        if "lat" in o:
            pos = (o["lat"], o["lon"], "override")
        if not pos:
            dropped["no position"] += 1; continue
        lat, lon, how = pos
        d = g.depth(lat, lon)
        if d is None or (d >= 0 and "lat" not in o and not o.get("include")):
            dropped["on land (GEBCO)"] += 1; continue
        titles = {l: e["sitelinks"][f"{l}wiki"]["title"] for l in LANGS if f"{l}wiki" in e.get("sitelinks", {})}
        if not titles:
            dropped["no article"] += 1; continue
        labels = e.get("labels", {})
        name = next((labels[l]["value"] for l in LANGS if l in labels), q)
        p31, cls = item_ids(e, "P31"), item_ids(e, "P289")
        built = first_year(e, ["P729", "P571", "P1619"])
        it = {
            "q": q, "name": name, "fam": o.get("fam") or family_of(p31 + cls, fam_map, built, cand[q]),
            "lat": round(lat, 5), "lon": round(lon, 5), "pos": how, "depth": round(d),
            "titles": titles, "types": p31,
        }
        if d > -15: it["coast"] = True                    # GEBCO cells are ~1 km: may still be in a harbour
        for k, v in (("class", cls[0] if cls else None), ("vehicle", (item_ids(e, "P121") or [None])[0]),
                     ("length", metres(e, "P2043")), ("beam", metres(e, "P2261")), ("draft", metres(e, "P2262")),
                     ("gt", metres(e, "P1093")), ("built", built), ("sunk", sunk_year(e)),
                     ("image", (val(claims(e, "P18")[0]) if claims(e, "P18") else None))):
            if v is not None: it[k] = v
        for k in ("depth", "heading", "note"):
            if k in o: it[k] = o[k]
        items.append(it)

    fams = Counter(i["fam"] for i in items)
    langs = Counter(l for i in items for l in i["titles"])
    meta = {
        "name": "Offlined Sea 3D Models — catalog",
        "generated": dt.date.today().isoformat(),
        "license": "CC0-1.0 (data from Wikidata, CC0; depths from the GEBCO Grid)",
        "count": len(items), "families": dict(fams.most_common()), "languages": dict(langs.most_common()),
    }
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:          # one item per line: small, readable diffs
        f.write("{\n" + "\n".join(f'  "{k}": {json.dumps(v, ensure_ascii=False)},' for k, v in meta.items()) + '\n  "items": [\n')
        f.write(",\n".join("    " + json.dumps(i, ensure_ascii=False, separators=(",", ":")) for i in items))
        f.write("\n  ]\n}\n")
    print(f"\n{len(items)} items under the sea → {out}")
    print("dropped:", dict(dropped))
    print("families:", dict(fams.most_common()))
    print("articles per language:", dict(langs.most_common()))


if __name__ == "__main__":
    main()
