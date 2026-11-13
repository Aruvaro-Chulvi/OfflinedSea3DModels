# Contributing

*[Leer en español](CONTRIBUTING.es.md)*

Thank you! There are many ways to help, and most of them need no 3D skills.

## Without modelling

- **Correct a position, a depth or a heading.** Many Wikipedia coordinates are approximate, or point to a
  museum instead of the wreck. Open an issue with *Correct a wreck* and give a source; or edit
  [`catalog/overrides.json`](catalog/overrides.json) in a pull request.
- **Remove something that is not under the sea** (a museum ship in a harbour, a raised wreck…): same way,
  with `"exclude": true` and the reason.
- **Ask for a model** of a wreck you care about: issue *Model request*. Plans, photos and surveys help whoever
  makes it.

## With a model

1. Find the thing's Wikidata id (`Q…`) in [`catalog/catalog.json`](catalog/catalog.json) or on
   [wikidata.org](https://www.wikidata.org). Check there is no folder for it yet in `models/` and no open
   pull request.
2. Make the model following [docs/MODEL_SPEC.md](docs/MODEL_SPEC.md): `.glb`, metres, +Y up, bow to +Z,
   origin where it touches the seabed, under the size limits.
   Free tools that export `.glb`: [Blender](https://www.blender.org) (File › Export › glTF 2.0, format
   *glTF Binary*, "+Y Up" ticked), [gltf.report](https://gltf.report) to inspect and compress (meshopt).
   Tip: on the [catalog page](https://aruvaro-chulvi.github.io/OfflinedSea3DModels/) every wreck has a
   *Download .glb* button with its generic model, already at the right size, axes and origin — import it in
   Blender and build on it.
3. Add `models/<item|class|vehicle>/<Q…>/model.glb`, `meta.json` and, if you can, `preview.jpg`.
4. Run `python tools/validate.py models/item/Q…` (Python 3.10+, nothing to install) and fix what it says.
5. Open a pull request. The same check runs automatically; a maintainer then looks at the model in a viewer.

By opening the pull request you confirm the points in [NOTICE.md](NOTICE.md#contributing-a-model): your own
work (or based only on CC BY 4.0 / CC0 works you credit), under **CC BY 4.0** or **CC0 1.0**.

### Which one first?

`class` models give the most for the effort: one Type VII U-boat model covers dozens of wrecks. Then the
famous wrecks people visit in Offlined. The catalog page will show which have a model and which still use
the generic one.

## Code

The catalog builder and the checker are in [`tools/`](tools/) (Python, MIT). To rebuild the catalog you need
the GEBCO Terrarium file that Offlined downloads (`Offlined Data/Maps/Seafloor/seafloor.pmtiles`):

```bash
pip install -r tools/requirements.txt
python tools/build_catalog.py --gebco "path/to/seafloor.pmtiles"
```
