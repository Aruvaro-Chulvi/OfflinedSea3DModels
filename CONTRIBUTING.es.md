# Cómo colaborar

*[Read in English](CONTRIBUTING.md)*

¡Gracias! Hay muchas formas de ayudar, y la mayoría no necesitan saber modelar en 3D.

## Sin modelar

- **Corregir una posición, una profundidad o un rumbo.** Muchas coordenadas de Wikipedia son aproximadas, o
  señalan un museo en vez del pecio. Abre un aviso (*issue*) con *Correct a wreck* y da una fuente; o edita
  [`catalog/overrides.json`](catalog/overrides.json) en un *pull request*.
- **Quitar algo que no está bajo el mar** (un barco museo en un puerto, un pecio que se reflotó…): igual, con
  `"exclude": true` y el motivo.
- **Añadir un pecio famoso que no tiene artículo en Wikipedia** (un punto de buceo conocido, como el B-17 «Black
  Jack»): añádelo a [`catalog/own_entries.json`](catalog/own_entries.json) con el siguiente id (L2, L3…), un texto
  corto en inglés (y en otros idiomas si puedes) y sus fuentes, y ejecuta `python tools/build_catalog.py --own-only`.
  Offlined lo enseña como una chincheta con ese texto en vez de un artículo.
- **Pedir un modelo** de un pecio que te interese: aviso *Model request*. Los planos, fotos y estudios ayudan a
  quien lo haga.

## Con un modelo

1. Busca el id de Wikidata (`Q…`) en [`catalog/catalog.json`](catalog/catalog.json) o en
   [wikidata.org](https://www.wikidata.org). Comprueba que no hay ya una carpeta en `models/` ni un *pull
   request* abierto.
2. Haz el modelo según [docs/MODEL_SPEC.es.md](docs/MODEL_SPEC.es.md): `.glb`, metros, +Y arriba, proa hacia
   +Z, origen donde toca el fondo, dentro de los límites de tamaño.
   Programas gratuitos que exportan `.glb`: [Blender](https://www.blender.org) (Archivo › Exportar › glTF 2.0,
   formato *glTF Binary*, con «+Y Up» marcado) y [gltf.report](https://gltf.report) para revisarlo.
   Truco: en la [página del catálogo](https://aruvaro-chulvi.github.io/OfflinedSea3DModels/) cada pecio tiene un
   botón *Descargar .glb* con su modelo genérico, ya con el tamaño, los ejes y el origen correctos: impórtalo en
   Blender y trabaja sobre él.
3. Añade `models/<item|class|vehicle>/<Q…>/model.glb`, `meta.json` y, si puedes, `preview.jpg`.
4. Ejecuta `python tools/validate.py models/item/Q…` (Python 3.10+, no hay que instalar nada) y arregla lo que
   te diga.
5. Abre un *pull request*. La misma comprobación se ejecuta sola; después alguien revisa el modelo en un visor.

Al abrir el *pull request* confirmas lo que dice [NOTICE.md](NOTICE.md#contributing-a-model): es trabajo tuyo
(o basado solo en obras CC BY 4.0 / CC0 que citas), con licencia **CC BY 4.0** o **CC0 1.0**.

### ¿Por cuál empiezo?

Los modelos de `class` son los que más rinden: un solo modelo de U-Boot tipo VII cubre decenas de pecios.
Después, los pecios famosos que más se visitan en Offlined. La página del catálogo enseñará cuáles tienen
modelo y cuáles siguen con el genérico.

### Mejorar un modelo que ya existe

Cualquiera puede hacer un Titanic mejor. Haz un *fork* del repositorio y cambia `models/item/Q25173/` en tu copia:

- **Hecho por código** (hay un script `model_sources/Q…_*.py`): mejora el script, ejecútalo y sube el script
  junto con el `.glb` que escribe.
- **Hecho en Blender u otro programa**: sustituye `model.glb` (y `model_built.glb`, `preview.jpg` si los rehaces).
- En `meta.json`, **deja los autores que ya están y añádete**, y añade tus fuentes. Si partes del modelo de
  otra persona, para eso está `based_on`.
- En el *pull request*, cuenta qué mejora y por qué (un estudio nuevo, una foto, un plano…).

## Cómo se revisan las aportaciones

Nada llega a Offlined sin pasar tres filtros:

1. **La comprobación automática** (GitHub Actions, el mismo `tools/validate.py`): formato, límites de tamaño
   y de triángulos, nada de archivos externos ni scripts, licencia, autores y fuentes en `meta.json`, un id
   de Wikidata real, y medidas que cuadran con el barco real (un `model_built.glb` que no mide más o menos lo
   que el barco no pasa).
2. **Una persona que mantiene el proyecto lo mira** en el visor antes de aceptarlo, con esta lista: es el
   pecio correcto; cuadra con sus fuentes; el tamaño, la profundidad y el rumbo son creíbles; la licencia y la
   autoría son creíbles (no está copiado de un modelo de pago ni de una web con otra licencia); no hay nada
   ofensivo ni ajeno. Solo quien mantiene el proyecto puede aceptar: un *pull request* por sí solo no cambia nada.
3. **Una versión publicada.** Offlined solo descarga lo que está en una versión publicada (su `manifest.json`,
   con el tamaño y el SHA-256 de cada archivo). Un modelo aceptado espera ahí hasta que se publica la siguiente.

Si un modelo resulta estar mal, se quita en la versión siguiente; la anterior sigue disponible.

## Código

El generador del catálogo y el comprobador están en [`tools/`](tools/) (Python, MIT). Para rehacer el catálogo
hace falta el archivo GEBCO Terrarium que descarga Offlined (`Offlined Data/Maps/Seafloor/seafloor.pmtiles`):

```bash
pip install -r tools/requirements.txt
python tools/build_catalog.py --gebco "ruta/a/seafloor.pmtiles"
```
