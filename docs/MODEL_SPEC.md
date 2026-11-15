# Model specification

*[Leer en español](MODEL_SPEC.es.md)*

Every model in this repository follows the same rules, so that any of them can be dropped on the seabed in
[Offlined](https://www.offlined.org) (Wiki Submarine, Seas and Oceans) — or in any other program — at the right
place, size and heading, without manual tweaks.

## 1. Where a model goes

```
models/
  item/<Wikidata id>/      one particular wreck or site            e.g. models/item/Q25173/  (RMS Titanic)
  class/<Wikidata id>/     every ship of a vessel class            e.g. models/class/Q10374571/ (Type VIIC U-boat)
  vehicle/<Wikidata id>/   every aircraft of a type                e.g. models/vehicle/Q6478/   (Airbus A330)
```

The Wikidata id (`Q…`) identifies the thing without depending on any language. You can look it up on
[wikidata.org](https://www.wikidata.org) or in [`catalog/catalog.json`](../catalog/catalog.json).

Offlined picks, for each wreck in the catalog, the first that exists:
`item/<its id>` → `class/<its vessel class>` → `vehicle/<its aircraft type>` → a **generic model** drawn by
code for its family (liner, warship, submarine, cargo ship, sailing ship, aircraft…).

Each folder holds:

| File | Required | What |
|---|---|---|
| `model.glb` | yes | The model **as it lies on the seabed today** (broken, rusted, half buried…). |
| `meta.json` | yes | Who made it, licence, sources, position and heading. See §4. |
| `preview.jpg` | recommended | A picture of the model, 1280 × 720, for the catalog page. |
| `model_built.glb` | optional | The ship or aircraft **as it was when it sailed or flew**, for a before/after view. |
| `diorama.glb` | optional (item) | The wreck **on its own piece of seabed**, with the life and the objects around it, to look at closely. See §7. |
| `diorama.jpg` | optional | A picture of the diorama, 1280 × 720. |

## 2. Format

- **glTF 2.0 binary** (`.glb`), one file with everything inside (no external `.bin` or textures).
- Extensions allowed: `KHR_mesh_quantization`, `KHR_texture_transform`, `KHR_materials_emissive_strength`,
  `EXT_texture_webp`. **Not allowed:** Draco, meshopt and KTX2/Basis compression (they need extra decoders that
  Offlined does not ship; the release packs are compressed as a whole instead).
- Textures: JPEG, PNG or WebP, **2048 × 2048 px at most**, power-of-two sizes preferred.
- Materials: PBR metallic-roughness (the glTF default). No lights, cameras or animations.

## 3. Scale, axes and origin

- **Real size, in metres** (1 glTF unit = 1 m). The Titanic is 269 m long, not 2.69 m.
- **+Y is up.** **The bow (front) points to +Z** — the glTF convention for "forward". Starboard is −X.
- **Origin = where the wreck touches the seabed**, under the middle of the hull (or of the main piece).
  `y = 0` is the seabed line: what is buried goes below 0. Do not model the seabed itself — Offlined draws
  it from the GEBCO data; a little sediment heaped against the hull is fine.
- Several pieces (the Titanic's bow and stern, 600 m apart; a debris field) go in the same file, each at its
  real place relative to the origin, within 1 km.

## 4. `meta.json`

```json
{
  "id": "Q25173",
  "kind": "item",
  "name": "RMS Titanic",
  "state": "broken",
  "heading": 33,
  "position": { "lat": 41.7325, "lon": -49.9469, "source": "https://…" },
  "depth": -3800,
  "authors": [{ "name": "Your Name", "url": "https://github.com/you" }],
  "license": "CC-BY-4.0",
  "sources": [{ "title": "Wreck survey 2010 (plan of the debris field)", "url": "https://…" }],
  "triangles": 148000,
  "notes": "Bow section only detailed; stern simplified."
}
```

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | The Wikidata id, the same as the folder name. |
| `kind` | yes | `item`, `class` or `vehicle` — the same as the parent folder. |
| `name` | yes | Name in English (as in the catalog). |
| `state` | yes (item) | `intact`, `broken` (in pieces), `scattered` (debris field), `capsized` (upside down) or `buried`. |
| `heading` | yes (item) | Where the **bow** points, in degrees clockwise from north (0–359). |
| `position` | no | Only if the wreck site differs from the catalog's: `lat`, `lon` and a `source`. |
| `depth` | no | Real depth of the wreck in metres (negative), if known better than GEBCO's ~1 km grid. |
| `pieces` | no | For a wreck broken into pieces lying far apart (more than 1 km, or at different depths): `{ "node name": { lat, lon, heading, depth, source } }`. Each of those nodes is modelled around its own origin (in the file it can sit at its offset from the main piece, with +Z north and +X west); programs that know the seabed place it on its own spot. |
| `authors` | yes | Everyone who made it. `url` is optional. |
| `license` | yes | `CC-BY-4.0` or `CC0-1.0`. Nothing else is accepted (see [NOTICE](../NOTICE.md)). |
| `sources` | yes | The plans, photos, surveys or papers you used. At least one. |
| `based_on` | no | If the model adapts someone else's work: `{ "title", "author", "url", "license" }` — that licence must be CC-BY-4.0 or CC0-1.0 too. |
| `triangles` | no | Filled in by the checker if missing. |
| `notes` | no | Anything worth knowing. |

## 5. Size limits

| | `item` | `class` / `vehicle` |
|---|---|---|
| File size | 8 MB | 5 MB |
| Triangles | 200,000 | 120,000 |

The checker in [`tools/validate.py`](../tools/validate.py) runs on every pull request and tells you what is
wrong. You can run it yourself: `python tools/validate.py`.

## 6. Good practice

- Model what can be **seen today**: the wreck, not the ship. If you also want the ship as built, add
  `model_built.glb`.
- Use real plans and survey data, and cite them. Do not guess the heading — say so in `notes` if unknown
  (then leave it as the catalog's, or 0).
- Keep it light: the seabed around is drawn at 1:100 in Offlined, so the camera rarely gets closer than a few
  metres. Detail you cannot see from 10 m away is weight for nothing.
- Do **not** copy models from other websites unless their licence is CC-BY-4.0 or CC0-1.0 and you name the
  author in `based_on`. Many sites (Sketchfab included) host models under *NonCommercial* or custom licences:
  those cannot come here.

## 7. The diorama (`diorama.glb`)

`model.glb` is light because Offlined draws it in a 1:100 world you fly through. The diorama is the
opposite: a scene to **stop, turn around and zoom into**, opened from the wreck's card («see it up close»).
It is optional and only for `item` folders. Rules 2 and 3 apply, with these differences:

- **It includes the seabed**: a piece of floor big enough to hold every part of the wreck (for the Titanic,
  bow, stern and the debris field between them), cut out like a museum diorama. Keep the same origin and
  axes as `model.glb`, so both line up.
- **Textures are welcome** (2048 px at most): colour and normal maps that tile, for the mud, the steel, the
  rust. Bake the light that does not move (ambient occlusion) into the vertex colours or a texture: the
  viewer adds only its own lights (ROV lamps or a soft museum light) and the marine snow.
- **Only what is real at that depth.** Below about 1,000 m there is no light: no reefs, no algae, no colourful
  fish. Find out which animals live there (photos from the surveys are the best source) and keep them few
  and small, as they are. Name them in `meta.json` → `sources`.
- **Points of interest**: empty nodes named `poi_<id>` whose `extras` are
  `{ "poi": { "id", "label": { "en", "es", … }, "text": { "en", "es", … }, "kind", "view" } }` —
  `label` is a few words, `text` one or two sentences, `kind` is `wreck`, `life` or `memorial`, `view` how
  far (m) the camera stands when it flies there. English is required; add any other languages you can.
- Optionally, the scene's `extras` can hold `{ "diorama": { "depth", "bounds", "start": { "target", "distance" } } }`
  for where the camera starts.
- **Respect**: many wrecks are graves. Show what is there with care; no human remains, ever.

| | `diorama.glb` |
|---|---|
| File size | 40 MB |
| Triangles | 1,500,000 |

The Titanic's diorama is built by code: [`model_sources/Q25173_titanic_diorama.py`](../model_sources/Q25173_titanic_diorama.py),
with [`tools/scenekit.py`](../tools/scenekit.py) (tiling textures, triplanar UVs, baked AO, points of interest).
