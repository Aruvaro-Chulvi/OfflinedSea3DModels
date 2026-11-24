// Release step: a meshopt-compressed copy of each model (EXT_meshopt_compression).
//
//     cd tools/meshopt && npm ci          (once)
//     node tools/meshopt/compress.mjs in.glb out.glb [in2.glb out2.glb ...]
//
// Lossless: the vertex and index data already in the model (KHR_mesh_quantization, as tools/slim.py leaves it) is only
// reordered for the cache and compressed; nothing is requantized and no node, name, extra or transform changes, so the
// pieces Offlined places on their own (meta.json "pieces") keep working. Textures are left as they are (WebP).
// Offlined 4.x reads it; older Offlined ignores these files ("mo" in the manifest) and keeps using the plain ones.
// After writing, each file is read back and its meshes compared with the original (vertex and index counts per mesh).
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression } from '@gltf-transform/extensions';
import { reorder } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import fs from 'fs';

await MeshoptEncoder.ready; await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });

/** Per mesh (by node path): vertex count and index count of each primitive */
function shape(doc) {
  const out = [];
  const walk = (node, path) => {
    const p = path + '/' + node.getName();
    const mesh = node.getMesh();
    if (mesh) for (const prim of mesh.listPrimitives())
      out.push(p + ':' + prim.getAttribute('POSITION').getCount() + ':' + (prim.getIndices() ? prim.getIndices().getCount() : 0));
    for (const c of node.listChildren()) walk(c, p);
  };
  for (const s of doc.getRoot().listScenes()) for (const n of s.listChildren()) walk(n, '');
  return out.sort().join('\n');
}

const args = process.argv.slice(2);
if (!args.length || args.length % 2) { console.error('usage: node compress.mjs in.glb out.glb [...]'); process.exit(2); }
let bad = 0;
for (let i = 0; i < args.length; i += 2) {
  const [inp, out] = [args[i], args[i + 1]];
  const doc = await io.read(inp);
  const before = shape(doc);
  await doc.transform(reorder({ encoder: MeshoptEncoder, target: 'size' }));
  doc.createExtension(EXTMeshoptCompression).setRequired(true)
    .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
  const buf = await io.writeBinary(doc);
  const after = shape(await io.readBinary(buf));
  if (after !== before) { console.error(`${inp}: the compressed copy does not match (skipped)`); bad++; continue; }
  fs.writeFileSync(out, buf);
  console.log(`${inp.split(/[\\/]/).pop()}: ${(fs.statSync(inp).size / 1e6).toFixed(2)} -> ${(buf.length / 1e6).toFixed(2)} MB`);
}
process.exit(bad ? 1 : 0);
