# Contributing

*[Leer en español](CONTRIBUTING.es.md)*

Thank you! There are many ways to help, and most of them need no 3D skills.

## Without modelling

- **Correct a position, a depth or a heading.** Many Wikipedia coordinates are approximate, or point to a
  museum instead of the wreck. Open an issue with *Correct a wreck* and give a source; or edit
  [`catalog/overrides.json`](catalog/overrides.json) in a pull request.
- **Remove something that is not under the sea** (a museum ship in a harbour, a raised wreck…): same way,
  with `"exclude": true` and the reason.
- **Add a famous wreck that has no Wikipedia article** (a well-known dive site, like the B-17 «Black Jack»): add it to
  [`catalog/own_entries.json`](catalog/own_entries.json) with the next id (L2, L3…), a short text in English (and
  other languages if you can) and its sources, then run `python tools/build_catalog.py --own-only`. Offlined shows it
  as a pin with that text instead of an article.
- **Ask for a model** of a wreck you care about: issue *Model request*. Plans, photos and surveys help whoever
  makes it.

## With a model

1. Find the thing's Wikidata id (`Q…`) in [`catalog/catalog.json`](catalog/catalog.json) or on
   [wikidata.org](https://www.wikidata.org). Check there is no folder for it yet in `models/` and no open
   pull request.
2. Make the model following [docs/MODEL_SPEC.md](docs/MODEL_SPEC.md): `.glb`, metres, +Y up, bow to +Z,
   origin where it touches the seabed, under the size limits.
   Free tools that export `.glb`: [Blender](https://www.blender.org) (File › Export › glTF 2.0, format
   *glTF Binary*, "+Y Up" ticked), [gltf.report](https://gltf.report) to inspect it.
   Tip: on the [catalog page](https://aruvaro-chulvi.github.io/OfflinedSea3DModels/) every wreck has a
   *Download .glb* button with its generic model, already at the right size, axes and origin — import it in
   Blender and build on it.
3. Add `models/<item|class|vehicle>/<Q…>/model.glb`, `meta.json` and, if you can, `preview.jpg`.
4. Run `python tools/validate.py models/item/Q…` (Python 3.10+, nothing to install) and
   `python tools/geomcheck.py Q…` (needs numpy: nothing floating, no decal off its surface), and fix what they say.
5. Open a pull request. The same check runs automatically; a maintainer then looks at the model in a viewer.

By opening the pull request you confirm the points in [NOTICE.md](NOTICE.md#contributing-a-model): your own
work (or based only on CC BY 4.0 / CC0 works you credit), under **CC BY 4.0** or **CC0 1.0**.

### Which one first?

`class` models give the most for the effort: one Type VII U-boat model covers dozens of wrecks. Then the
famous wrecks people visit in Offlined. The catalog page will show which have a model and which still use
the generic one.

### Improving a model that already exists

Anyone can make a better Titanic. Fork the repository and change `models/item/Q25173/` in your fork:

- **Built by code** (a `model_sources/Q…_*.py` script is there): improve the script, run it, and commit the
  script together with the `.glb` it writes.
- **Built in Blender or another program**: replace `model.glb` (and `model_built.glb`, `preview.jpg` if you
  redo them).
- In `meta.json`, **keep the authors who are already there and add yourself**, and add your sources. If you
  start from someone else's model, that is what `based_on` is for.
- In the pull request, say what is better and why (a new survey, a photo, a plan…).

## How contributions are reviewed

Nothing reaches Offlined without three filters:

1. **The automatic check** (GitHub Actions, the same `tools/validate.py` and `tools/geomcheck.py`): nothing
   floating and no decal off its surface, file format, size and triangle
   limits, no external files or scripts, licence, authors and sources in `meta.json`, a real Wikidata id, and
   sizes that match the real ship (a `model_built.glb` that is not about as long as the ship fails).
2. **A maintainer looks at it** in the viewer before merging, with this list: it is the right wreck; it
   matches its sources; the size, depth and heading are believable; the licence and authorship are credible
   (not copied from a paid model or a site with another licence); there is nothing offensive or unrelated.
   Only maintainers can merge — a pull request on its own changes nothing.
3. **A release.** Offlined only downloads what is in a published release (its `manifest.json`, with the size
   and SHA-256 of every file). A merged model waits there until the maintainer publishes the next release.

Models that turn out to be wrong are removed in the next release; the previous release stays available.

## Code

The catalog builder and the checker are in [`tools/`](tools/) (Python, MIT). To rebuild the catalog you need
the GEBCO Terrarium file that Offlined downloads (`Offlined Data/Maps/Seafloor/seafloor.pmtiles`):

```bash
pip install -r tools/requirements.txt
python tools/build_catalog.py --gebco "path/to/seafloor.pmtiles"
```
