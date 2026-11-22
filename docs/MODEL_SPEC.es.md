# Especificación de los modelos

*[Read in English](MODEL_SPEC.md)*

Todos los modelos de este repositorio siguen las mismas reglas, para que cualquiera de ellos se pueda poner
en el fondo del mar en [Offlined](https://www.offlined.org) (Wiki Submarino, Mares y Océanos) —o en cualquier
otro programa— en su sitio, a su tamaño y con su rumbo, sin retoques a mano.

## 1. Dónde va cada modelo

```
models/
  item/<id de Wikidata>/      un pecio o un lugar concreto             p. ej. models/item/Q25173/  (RMS Titanic)
  class/<id de Wikidata>/     todos los barcos de una clase            p. ej. models/class/Q10374571/ (U-Boot tipo VIIC)
  vehicle/<id de Wikidata>/   todos los aviones de un modelo           p. ej. models/vehicle/Q6478/   (Airbus A330)
```

El id de Wikidata (`Q…`) identifica la cosa sin depender del idioma. Lo encuentras en
[wikidata.org](https://www.wikidata.org) o en [`catalog/catalog.json`](../catalog/catalog.json).

Para cada pecio del catálogo, Offlined usa el primero que exista:
`item/<su id>` → `class/<su clase de barco>` → `vehicle/<su modelo de avión>` → un **modelo genérico**
dibujado por código según su familia (transatlántico, buque de guerra, submarino, carguero, velero, avión…).

Cada carpeta lleva:

| Archivo | Obligatorio | Qué es |
|---|---|---|
| `model.glb` | sí | El modelo **tal como está hoy en el fondo** (partido, oxidado, medio enterrado…). |
| `meta.json` | sí | Quién lo hizo, licencia, fuentes, posición y rumbo. Ver §4. |
| `preview.jpg` | recomendado | Una imagen del modelo, 1280 × 720, para la página del catálogo. |
| `model_built.glb` | opcional | El barco o avión **tal como era** cuando navegaba o volaba, para verlo «antes y ahora». |

## 2. Formato

- **glTF 2.0 binario** (`.glb`), un único archivo con todo dentro (sin `.bin` ni texturas aparte).
- Extensiones permitidas: `KHR_mesh_quantization`, `KHR_texture_transform`, `EXT_mesh_gpu_instancing`,
  `KHR_materials_emissive_strength`, `EXT_texture_webp`. **No se admiten** las compresiones Draco, meshopt ni
  KTX2/Basis (necesitan decodificadores que Offlined no incluye). Cada versión publica además cada modelo comprimido
  con gzip (`.glb.gz`): Offlined descarga y guarda ese.
- **Que pese poco**: posiciones y coordenadas de textura de 16 bits y normales de 8 bits (`KHR_mesh_quantization`),
  e **instancias** (`EXT_mesh_gpu_instancing`) para todo lo que se repite —carámbanos de óxido, anémonas, platos,
  remaches…—: la forma se guarda una vez, más dónde va cada copia. [`tools/scenekit.py`](../tools/scenekit.py)
  `write_model()` hace las dos cosas.
- Texturas: JPEG, PNG o WebP, **como mucho 2048 × 2048 px**, mejor en potencias de dos. Los modelos de este
  repositorio llevan WebP (`EXT_texture_webp`, obligatoria) de **1024 px como mucho**: a las distancias a las que
  Offlined enseña un pecio no se nota la diferencia con 2048 px, y el modelo carga antes.
  [`tools/slim.py`](../tools/slim.py) lo hace (y quita materiales sin usar y une bloques y copias que se pueden unir);
  `write_model()` lo ejecuta solo.
- Materiales: PBR metálico-rugosidad (el normal de glTF). Sin luces, cámaras ni animaciones.

## 3. Escala, ejes y origen

- **Tamaño real, en metros** (1 unidad de glTF = 1 m). El Titanic mide 269 m, no 2,69 m.
- **+Y hacia arriba.** **La proa apunta a +Z**, el «hacia delante» de glTF. Estribor queda en −X.
- **Origen = donde el pecio toca el fondo**, bajo el centro del casco (o de la pieza principal). `y = 0` es la
  línea del fondo: lo enterrado va por debajo de 0. No modeles el fondo: Offlined lo dibuja con los datos de
  GEBCO; un poco de sedimento amontonado contra el casco sí vale.
- Si hay varias piezas (la proa y la popa del Titanic, a 600 m; un campo de restos), van en el mismo archivo,
  cada una en su sitio real respecto al origen, a menos de 1 km.

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
  "authors": [{ "name": "Tu Nombre", "url": "https://github.com/tu-usuario" }],
  "license": "CC-BY-4.0",
  "sources": [{ "title": "Estudio del pecio de 2010 (plano del campo de restos)", "url": "https://…" }],
  "triangles": 148000,
  "notes": "Solo la proa en detalle; la popa, simplificada."
}
```

| Clave | Obligatoria | Significado |
|---|---|---|
| `id` | sí | El id de Wikidata, igual que el nombre de la carpeta. |
| `kind` | sí | `item`, `class` o `vehicle`, igual que la carpeta de encima. |
| `name` | sí | Nombre en inglés (como en el catálogo). |
| `state` | sí (item) | `intact` (entero), `broken` (en trozos), `scattered` (restos esparcidos), `capsized` (boca abajo) o `buried` (enterrado). |
| `heading` | sí (item) | Hacia dónde apunta la **proa**, en grados desde el norte en el sentido de las agujas del reloj (0–359). |
| `position` | no | Solo si el pecio no está donde dice el catálogo: `lat`, `lon` y una `source`. |
| `depth` | no | Profundidad real del pecio en metros (negativa), si se sabe mejor que con la malla de ~1 km de GEBCO. |
| `pieces` | no | Para un pecio partido en trozos muy separados (más de 1 km, o a distintas profundidades): `{ "nombre del nodo": { lat, lon, heading, depth, source } }`. Cada uno de esos nodos se modela alrededor de su propio origen (en el archivo puede estar a su distancia de la pieza principal, con +Z al norte y +X al oeste); los programas que conocen el fondo lo colocan en su sitio. |
| `hole` | no | Para un lugar que es un hoyo en el fondo (un agujero azul, un cráter): `{ "r": radio en m, "floor": profundidad de su suelo bajo el origen en m (negativa) }`, centrado en el origen. El modelo pone las paredes, el suelo y una tapa de fondo alrededor de la boca (el origen está al nivel del fondo de alrededor); los programas que dibujan el fondo lo abren hasta el suelo dentro de r para poder bajar dentro. |
| `authors` | sí | Todas las personas que lo han hecho. `url` es opcional. |
| `license` | sí | `CC-BY-4.0` o `CC0-1.0`. No se acepta otra (ver [NOTICE](../NOTICE.md)). |
| `sources` | sí | Los planos, fotos, estudios o artículos que has usado. Al menos uno. |
| `based_on` | no | Si el modelo adapta el trabajo de otra persona: `{ "title", "author", "url", "license" }`; esa licencia también debe ser CC-BY-4.0 o CC0-1.0. |
| `triangles` | no | Lo rellena el comprobador si falta. |
| `notes` | no | Cualquier cosa que convenga saber. |

## 5. Límites de tamaño

| | `item` | `class` / `vehicle` |
|---|---|---|
| Tamaño del archivo | 8 MB | 5 MB |
| Triángulos | 200.000 | 120.000 |

El comprobador [`tools/validate.py`](../tools/validate.py) se ejecuta en cada *pull request* y te dice qué
falla. Puedes lanzarlo tú: `python tools/validate.py`.

## 5b. Nada flota, nada se despega de su superficie

Cada pieza de un modelo tiene que tocar algo: el casco al que pertenece, otra pieza o el fondo. Las chapas sueltas
de un pecio, una chimenea caída o una hélice arrancada descansan **sobre** el fondo o sobre el pecio, nunca unos
centímetros por encima. Las marcas pintadas, ventanas, ojos de buey y letras van **sobre** su superficie (unos
milímetros por fuera, para que no parpadeen) y siguen su curva: no vale un disco plano sobre un ala o un fuselaje
curvos.

[`tools/geomcheck.py`](../tools/geomcheck.py) comprueba las dos cosas (necesita numpy: `pip install numpy`) y se
ejecuta en cada *pull request*. Suspende un modelo con una pieza a más de 5 cm de todo (y que no esté en el fondo)
o una calcomanía a más de 2 cm de su superficie. Si algo cuelga libre de verdad (un cable hacia la superficie…),
añádelo a `tools/geomcheck_allow.json` con el motivo.

Los modelos hechos por código aquí lo tienen gratis: [`tools/fixkit.py`](../tools/fixkit.py) se ejecuta dentro de
`scenekit.write_model()`, pega las calcomanías a su superficie y asienta las piezas sueltas sobre lo que tienen
debajo.

## 6. Buenas prácticas

- Modela lo que **se ve hoy**: el pecio, no el barco. Si además quieres el barco como era, añade
  `model_built.glb`.
- Usa planos y estudios reales, y cítalos. No adivines el rumbo: si no se sabe, dilo en `notes`.
- Que merezca verse de cerca: en Offlined puedes acercarte a un par de metros de cualquier parte del pecio. El
  óxido, los carámbanos, los animales que viven allí de verdad (solo esos: por debajo de unos 1.000 m no hay luz,
  así que ni arrecifes ni peces de colores) y objetos reconocibles en el campo de restos hacen interesante un
  pecio; [`tools/abyss.py`](../tools/abyss.py) los tiene. Muchos pecios son tumbas: nunca restos humanos.
- **No copies** modelos de otras webs salvo que su licencia sea CC-BY-4.0 o CC0-1.0 y nombres al autor en
  `based_on`. Muchas webs (Sketchfab incluida) tienen modelos con licencias *NoComercial* o propias: esos no
  pueden venir aquí.
