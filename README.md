# Offlined Sea 3D Models

*[Leer en español](README.es.md)*

![Offlined Sea 3D Models: shipwrecks, reefs and vents at real scale — the Titanic, Bismarck, Prestige, Lusitania, Andrea Doria and Yamato](docs/social-preview.png)

**Open 3D models of what lies on the seabed — shipwrecks, sunken aircraft, reefs, underwater sites — each one
linked to its Wikipedia article and placed where it really is, at real size.**

A project of [Offlined](https://www.offlined.org), the offline library with Wikipedia, maps and the seafloor
of the whole planet. In Offlined's *Wiki Submarine* you fly over the GEBCO seabed, go down to a place and read
its article; with these models you will also **see the wreck lying there**: the Titanic broken in two at
3,800 m, the Thistlegorm with its cargo at 30 m, a U-boat on the North Sea bed — and, with one button, the ship
as she was floating just above her wreck, to compare the two.

**[▶ Browse the catalog in 3D](https://aruvaro-chulvi.github.io/OfflinedSea3DModels/)** — search any wreck, look at it next to a diver
for scale, open its Wikipedia article, and download its generic model as a starting point for a better one.

## What is here

| | |
|---|---|
| [`catalog/catalog.json`](catalog/catalog.json) | **4,537 things under the sea that have a Wikipedia article** (in at least one of 10 languages), with their Wikidata id, position, GEBCO depth, size, dates and family. Built automatically from Wikidata + GEBCO. |
| [`models/`](models/) | The 3D models, one folder per Wikidata id. Own models so far: **[RMS Titanic](models/item/Q25173/)**, **[MV Prestige](models/item/Q1977186/)**, **[Bismarck](models/item/Q150857/)**, **[RMS Lusitania](models/item/Q187425/)**, **[SS Andrea Doria](models/item/Q493798/)**, **[Yamato](models/item/Q215010/)**, **[HMHS Britannic](models/item/Q215692/)**, **[Musashi](models/item/Q1127368/)**, **[HMS Hood](models/item/Q220239/)**, the warship **[Mars](models/item/Q31357/)** (1564) Shackleton's **[Endurance](models/item/Q911858/)** (wooden ships, built with `tools/sailkit.py`), the airliner **[Kaleva](models/item/Q1464299/)** (a Ju 52 of 1940, built with `tools/airkit.py`) the submarine **[USS Scorpion](models/item/Q1757204/)** and the bomber **[B-17 «Black Jack»](models/item/L1/)** (an own catalog entry: it has no Wikipedia article) and the Red Sea cargo ship **[SS Thistlegorm](models/item/Q32276/)**, with her lorries, motorcycles and locomotives, and the Italian liner **[SS Umbria](models/item/Q2478173/)**, scuttled with her bombs off Port Sudan, and the passenger steamer **[SS Yongala](models/item/Q1424289/)** on the Great Barrier Reef, and the two ships of one battle off Western Australia, the cruiser **[HMAS Sydney](models/item/Q1031986/)** and the raider **[Kormoran](models/item/Q708046/)**, at 2,500 m; then wrecks from around the world — **[Fujikawa Maru](models/item/Q1473175/)**, **[USAT Liberty](models/item/Q2468064/)**, **[MS Antilla](models/item/Q576959/)**, **[Giannis D](models/item/Q20181784/)**, **[SS Carnatic](models/item/Q1043906/)**, **[MS Mikhail Lermontov](models/item/Q1143472/)**, **[RMS Empress of Ireland](models/item/Q1339472/)**, **[RMS Rhone](models/item/Q1664018/)**, **[MV Bianca C](models/item/Q3273207/)**, **[MS Zenobia](models/item/Q2603065/)**, **[Breadalbane](models/item/Q4959197/)**, **[HMS Investigator](models/item/Q1564906/)**, **[HMS Hermes](models/item/Q596282/)**, **[USS Saratoga](models/item/Q852981/)**, **[USS Oriskany](models/item/Q562053/)**, **[HMS Prince of Wales](models/item/Q724919/)**, **[Nagato](models/item/Q1291512/)**, **[SMS Markgraf](models/item/Q481485/)**, **[SMS Szent István](models/item/Q642429/)**, **[Admiral Graf Spee](models/item/Q154115/)**, **[USS Monitor](models/item/Q857891/)**, **[USS Grunion](models/item/Q6147254/)**, **[HMS Royal Oak](models/item/Q620472/)**, **[SMS Kronprinz Wilhelm](models/item/Q470400/)**, **[HMS Hampshire](models/item/Q1060785/)**, **[Blücher](models/item/Q161346/)**, **[Scharnhorst](models/item/Q155222/)**, **[K-278 Komsomolets](models/item/Q499298/)**, **[MV Wilhelm Gustloff](models/item/Q155014/)**, **[Vrouw Maria](models/item/Q54887/)**, **[Kronan](models/item/Q1789811/)**, **[HMS Ark Royal](models/item/Q1333909/)**, **[Armenia](models/item/Q683174/)**, **[Maldive Victory](models/item/Q1255918/)**, **[Shinkoku Maru](models/item/Q2279126/)**, **[Kaga](models/item/Q726146/)**, **[USS Yorktown](models/item/Q311245/)**, **[USS Hornet](models/item/Q838125/)**, **[USS Indianapolis](models/item/Q600180/)**, **[USS Johnston](models/item/Q5229080/)**, **[Rainbow Warrior](models/item/Q627265/)**, **[U-352](models/item/Q32804/)**, **[USS Spiegel Grove](models/item/Q7874298/)**, **[USS Kittiwake](models/item/Q7870825/)**, **[SMS Scharnhorst](models/item/Q533999/)**, **[San José](models/item/Q5874775/)**, **[RMS Carpathia](models/item/Q504096/)**, **[SS Mendi](models/item/Q1920341/)**, **[RMS Lancastria](models/item/Q701654/)**, **[MV Goya](models/item/Q700168/)**, **[SS General von Steuben](models/item/Q703586/)**, **[SS Gairsoppa](models/item/Q1041233/)**, **[SS Baron Gautsch](models/item/Q686128/)**, **[MV Salem Express](models/item/Q1417751/)**, **[Hilma Hooker](models/item/Q1618842/)**, **[SS City of Cairo](models/item/Q1094070/)**, **[AHS Centaur](models/item/Q686392/)**, **[SS Sirio](models/item/Q654430/)**, **[HMS Victoria](models/item/Q1502183/)**, **[Prinz Eugen](models/item/Q155430/)**, **[Yamashiro](models/item/Q598221/)**, **[USS Juneau (CL-52)](models/item/Q2003934/)**, **[USS Atlanta (CL-51)](models/item/Q281442/)**, **[USS Lexington (CV-2)](models/item/Q838148/)**, **[Akagi](models/item/Q320967/)**, **[USS Thresher (SSN-593)](models/item/Q2164015/)**, **[ARA San Juan](models/item/Q5653859/)**, **[U-85](models/item/Q571310/)**, **[HMS Victory (1737)](models/item/Q839283/)**, **[Queen Anne's Revenge](models/item/Q247567/)** — and places that are not wrecks: the drowned towns and ports of **[Pavlopetri](models/item/Q966038/)**, **[Baiae](models/item/Q290097/)**, **[Thonis-Heracleion](models/item/Q1607758/)**, **[Canopus](models/item/Q1145180/)**, **[Antirhodos](models/item/Q583052/)**, the fallen **[Lighthouse of Alexandria](models/item/Q43244/)**, **[Port Royal](models/item/Q685357/)**, **[Atlit Yam](models/item/Q2896085/)** and **[Bouldnor Cliff](models/item/Q4949619/)**; the undersea habitats **[Aquarius](models/item/Q623183/)** and Cousteau's **[Conshelf II](models/item/Q3408831/)**, the **[Aqaba Underwater Military Museum](models/item/L2/)**; the blue holes of **[Belize](models/item/Q86486/)**, **[Dean's](models/item/Q1181185/)** and **[Dahab](models/item/Q885734/)** (holes in the seabed you can go down into: meta.json `hole`), and the volcanoes **[Kolumbo](models/item/Q1513696/)** and **[Kick 'em Jenny](models/item/Q1740807/)**, the vent fields **[Lost City](models/item/Q521459/)** and **[Loki's Castle](models/item/Q6668666/)**, the coral **[Darwin Mounds](models/item/Q632738/)**, and two reefs made of what people threw away, **[Osborne Reef](models/item/Q1272680/)** (tyres) and **[Redbird Reef](models/item/Q2136142/)** (subway cars), the Roman cargo of marble columns at **[Mahdia](models/item/Q3589083/)** and the site of the **[Antikythera wreck](models/item/Q2616711/)** — the wrecks as they lie today, and the ships as built. |
| [`model_sources/`](model_sources/) | Models built by code (Python + [`tools/meshkit.py`](tools/meshkit.py)): run the script, get the `.glb`. |
| [`docs/MODEL_SPEC.md`](docs/MODEL_SPEC.md) | The rules every model follows (glTF `.glb`, metres, axes, origin, limits, `meta.json`). |
| [`generator/`](generator/) | The generic wreck generator (JavaScript, three.js): any catalog item → a model at real size, with paint, rust and sediment. Offlined uses the same file. |
| [`site/`](site/) | The catalog page published on GitHub Pages. |
| [`tools/`](tools/) | Catalog builder, GEBCO depth reader and the model checker that runs on every pull request. |

### The catalog in numbers

| Family | Items |
|---|---:|
| Submarines | 1,206 |
| Other ships | 1,095 |
| Warships | 911 |
| Sailing ships | 267 |
| Steamships | 253 |
| Aircraft | 211 |
| Liners and ferries | 201 |
| Cargo ships | 186 |
| Aircraft carriers | 79 |
| Reefs | 67 |
| Tankers | 44 |
| Fishing vessels | 11 |
| Hydrothermal vents | 3 |
| Archaeological sites and statues | 2 |
| Underwater museums | 1 |

Articles per language: English 4,288 · Français 2,381 · Deutsch 2,053 · Русский 1,258 · العربية 1,072 · Español 1,026 · Italiano 1,005 · Português 1,001 · 中文 853 · हिन्दी 9.

**2,178** of them belong to a known vessel class — a single class model covers many: the Type VIIC U-boat alone, **338** wrecks.

Not yet included: wrecks in lakes (the GEBCO grid treats lakes as land), and places whose Wikidata entry has no coordinates.

## How a wreck gets its model

Nobody can model 4,537 objects by hand, so every wreck gets the best model available, in this order:

1. **Its own model** — `models/item/<id>/` — made by someone for that wreck.
2. **Its class model** — `models/class/<id>/` — one model for every ship of the same design (all the Type VII
   U-boats, all the Liberty ships…).
3. **Its aircraft type** — `models/vehicle/<id>/` — for sunken aircraft.
4. **A generic model** drawn by code from its family and its real length and beam (liner, warship, submarine,
   cargo ship, tanker, sailing ship, aircraft…), lying on the seabed with a list, half buried and rusted —
   always the same for the same wreck. Reefs, vents, archaeological sites and underwater museums get a
   representative patch.

So the seabed is full from day one, and gets better with every contribution.

## Contributing

You can help **without any 3D skills**: correct a position, a depth or a heading, or point out something that
is not really under the sea. And if you model, a single class model can cover dozens of wrecks.
Anyone can also improve an existing model (a better Titanic, say) with a pull request.

Every contribution goes through the automatic check, a review by the maintainer in the viewer, and only then into
a release — Offlined downloads nothing that has not been published there. See [CONTRIBUTING.md](CONTRIBUTING.md#how-contributions-are-reviewed).

## Licences

- **Models:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (or [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) if the author prefers) — free for any use, also commercial, crediting the authors.
- **Catalog and `meta.json`:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
- **Code:** [MIT](LICENSE).

Details, sources and how to credit: [NOTICE.md](NOTICE.md).

Data: [Wikidata](https://www.wikidata.org) (CC0) · [GEBCO Grid](https://www.gebco.net) (GEBCO Compilation Group).
