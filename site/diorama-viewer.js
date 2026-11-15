// Offlined Sea 3D Models — close-up diorama viewer (MIT)
// One module for the catalog site and for Offlined (Wiki Submarine): it gets THREE, OrbitControls and a loader
// from the page, so each one can use its own copy of three.js.
//
//   const v = await openDiorama(container, { THREE, OrbitControls, load: url => Promise<Object3D>, url, title, lang,
//                                            onClose, credits, texts })
// texts: the interface in another language ({ rov, museum, … } as in TXT below); the labels of the points of
// interest come from the .glb in whatever languages it has (English when the page's is missing).
//   v.close()
//
// The .glb (docs/MODEL_SPEC.md, «diorama.glb») carries its points of interest as empty nodes named "poi_<id>"
// whose extras are { poi: { id, label: {en, es…}, text: {en, es…}, kind } }, and the scene extras
// { diorama: { bounds, start: { target, distance } } }.

const TXT = {
  en: { rov: "ROV lights", museum: "Museum light", labels: "Labels", list: "Places", close: "Close", reset: "Overview",
        help: "Drag to turn · right-drag to move · wheel to zoom where the cursor points · double-click to fly there",
        loading: "Loading the diorama…", depth: "{d} m deep", kinds: { wreck: "The wreck", memorial: "Memorial", life: "Life and seabed" },
        snow: "Marine snow", fail: "The diorama could not be loaded." },
  es: { rov: "Focos del ROV", museum: "Luz de museo", labels: "Etiquetas", list: "Lugares", close: "Cerrar", reset: "Vista general",
        help: "Arrastra para girar · botón derecho para mover · rueda para acercarte adonde apunta el cursor · doble clic para ir allí",
        loading: "Cargando el diorama…", depth: "{d} m de profundidad", kinds: { wreck: "El pecio", memorial: "Memoria", life: "Vida y fondo" },
        snow: "Nieve marina", fail: "No se ha podido cargar el diorama." },
};

const CSS = `
.dio{position:absolute;inset:0;overflow:hidden;background:#01060b;color:#e6eef4;font:14px/1.45 system-ui,"Segoe UI",sans-serif;user-select:none}
.dio canvas{position:absolute;inset:0;width:100%;height:100%;display:block;outline:none}
.dio .dio-top{position:absolute;left:16px;right:16px;top:12px;display:flex;gap:10px;align-items:center;pointer-events:none}
.dio .dio-top h2{margin:0;font-size:18px;font-weight:600;text-shadow:0 1px 3px #000}
.dio .dio-top .dio-sub{color:#9fb6c6;font-size:13px}
.dio .dio-top .dio-sp{flex:1}
.dio .dio-top button,.dio .dio-bar button{pointer-events:auto;border:1px solid #2a4458;background:rgba(8,22,33,.85);color:#e6eef4;border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit;font-size:13px}
.dio .dio-top button.on,.dio .dio-bar button.on{background:#3fb6d8;color:#04121d;border-color:#3fb6d8}
.dio .dio-bar{position:absolute;left:16px;bottom:14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.dio .dio-help{position:absolute;right:16px;bottom:16px;color:#7f98aa;font-size:12px;text-align:right;max-width:46%;pointer-events:none}
.dio .dio-pl{position:absolute;left:16px;top:56px;width:250px;max-height:calc(100% - 130px);overflow:auto;background:rgba(6,18,28,.9);border:1px solid #1d3447;border-radius:12px;padding:8px 0}
.dio .dio-pl h4{margin:8px 14px 4px;font-size:12px;color:#8fa6b8;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.dio .dio-pl div{padding:5px 14px;cursor:pointer;font-size:13px}
.dio .dio-pl div:hover,.dio .dio-pl div.on{background:#15405a}
.dio .dio-lb{position:absolute;left:0;top:0;transform:translate(-50%,-100%);pointer-events:auto;cursor:pointer;white-space:nowrap;font-size:12px;padding:3px 8px;border-radius:12px;background:rgba(6,18,28,.78);border:1px solid rgba(120,190,220,.45);color:#e6eef4}
.dio .dio-lb::after{content:"";position:absolute;left:50%;top:100%;width:1px;height:14px;background:rgba(160,210,230,.6)}
.dio .dio-lb.life{border-color:rgba(150,220,160,.5)}
.dio .dio-lb.memorial{border-color:rgba(230,200,120,.6)}
.dio .dio-lb.on{background:#3fb6d8;color:#04121d}
.dio .dio-card{position:absolute;right:16px;top:56px;width:320px;max-width:calc(100% - 32px);background:rgba(6,18,28,.92);border:1px solid #1d3447;border-radius:12px;padding:12px 14px}
.dio .dio-card h3{margin:0 0 6px;font-size:16px}
.dio .dio-card p{margin:0;color:#c9d7e1;font-size:13px}
.dio .dio-card .dio-x{float:right;background:none;border:0;color:#8fa6b8;cursor:pointer;font-size:16px}
.dio .dio-msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#9fb6c6;font-size:15px}
.dio .dio-cr{position:absolute;right:16px;top:44px;color:#6f889a;font-size:11px;text-align:right;pointer-events:none}
@media (max-width:760px){.dio .dio-pl{width:auto;right:16px;max-height:40%}.dio .dio-help{display:none}.dio .dio-card{top:auto;bottom:112px}}
`;

export async function openDiorama(container, opt) {
  const { THREE, OrbitControls } = opt;
  const lang = TXT[opt.lang] ? opt.lang : "en";
  const UI = Object.assign({}, TXT.en, TXT[lang], opt.texts || {});
  const tx = (k, v = {}) => String(UI[k] ?? k).replace(/\{(\w+)\}/g, (_, x) => v[x] ?? "");
  const pick = o => (o && (o[opt.lang] || o[lang] || o.en || Object.values(o)[0])) || "";
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  if (!document.getElementById("dio-css")) {
    const st = document.createElement("style"); st.id = "dio-css"; st.textContent = CSS; document.head.appendChild(st);
  }
  const root = document.createElement("div"); root.className = "dio";
  root.innerHTML = `<canvas tabindex="0"></canvas><div class="dio-msg">${tx("loading")}</div>
    <div class="dio-top"><div><h2>${esc(opt.title || "")}</h2><div class="dio-sub"></div></div><div class="dio-sp"></div>
      <button data-a="list">${tx("list")}</button>${opt.onClose ? `<button data-a="close">✕ ${tx("close")}</button>` : ""}</div>
    <div class="dio-cr">${opt.credits || ""}</div>
    <div class="dio-pl" hidden></div><div class="dio-card" hidden></div>
    <div class="dio-bar"><button data-a="rov">${tx("rov")}</button><button data-a="museum">${tx("museum")}</button>
      <button data-a="labels" class="on">${tx("labels")}</button><button data-a="snow" class="on">${tx("snow")}</button><button data-a="reset">${tx("reset")}</button></div>
    <div class="dio-help">${tx("help")}</div>`;
  container.appendChild(root);
  const $ = s => root.querySelector(s);
  const canvas = $("canvas");

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.03, 4000);
  scene.add(camera);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.08; controls.zoomToCursor = true;
  controls.maxPolarAngle = Math.PI * 0.495; controls.minDistance = 0.25; controls.maxDistance = 1600;
  controls.screenSpacePanning = true;

  // ── lights ──
  const hemi = new THREE.HemisphereLight(0x9cc6de, 0x3a3226, 1.0);
  const sun = new THREE.DirectionalLight(0xfff2dc, 1.6); sun.position.set(-0.4, 1, 0.55);
  const fill = new THREE.DirectionalLight(0x6fa8d0, 0.35); fill.position.set(0.6, 0.4, -0.7);
  const spots = [];
  for (const sx of [-1, 1]) {
    const s = new THREE.SpotLight(0xfff1dc, 520, 200, 0.55, 0.85, 1.5);
    s.position.set(sx * 0.9, 0.5, 0.4); s.target.position.set(sx * 0.15, -0.6, -6);
    camera.add(s, s.target); spots.push(s);
  }
  const rovAmb = new THREE.HemisphereLight(0x2c5674, 0x0b0b0b, 0.06);
  scene.add(hemi, sun, fill, rovAmb);

  // ── marine snow: particles drifting around the camera ──
  const SN = 4000, snowBox = 60;
  const sg = new THREE.BufferGeometry(), sp = new Float32Array(SN * 3), sv = new Float32Array(SN);
  for (let i = 0; i < SN; i++) { sp[i * 3] = (Math.random() - 0.5) * snowBox; sp[i * 3 + 1] = (Math.random() - 0.5) * snowBox; sp[i * 3 + 2] = (Math.random() - 0.5) * snowBox; sv[i] = 0.02 + Math.random() * 0.05; }
  sg.setAttribute("position", new THREE.BufferAttribute(sp, 3));
  const sc = document.createElement("canvas"); sc.width = sc.height = 32;
  { const g = sc.getContext("2d"), gr = g.createRadialGradient(16, 16, 0, 16, 16, 16); gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.4, "rgba(220,235,240,.5)"); gr.addColorStop(1, "rgba(200,220,230,0)"); g.fillStyle = gr; g.fillRect(0, 0, 32, 32); }
  const snowTex = new THREE.CanvasTexture(sc);
  const snow = new THREE.Points(sg, new THREE.PointsMaterial({ size: 0.09, map: snowTex, transparent: true, depthWrite: false, opacity: 0.75, sizeAttenuation: true, color: 0xd8e6ec }));
  snow.frustumCulled = false; scene.add(snow);

  let mode = "museum";
  function setMode(m) {
    mode = m;
    const rov = m === "rov";
    hemi.visible = sun.visible = fill.visible = !rov; rovAmb.visible = rov; spots.forEach(s => (s.visible = rov));
    scene.background = new THREE.Color(rov ? 0x000308 : 0x06192a);
    scene.fog = rov ? new THREE.FogExp2(0x000610, 0.010) : new THREE.FogExp2(0x06192a, 0.0011);
    renderer.toneMappingExposure = rov ? 1.15 : 1.0;
    snow.material.color.set(rov ? 0xe8f2f6 : 0x9fb8c6); snow.material.opacity = rov ? 0.85 : 0.35;
    root.querySelector('[data-a="rov"]').classList.toggle("on", rov);
    root.querySelector('[data-a="museum"]').classList.toggle("on", !rov);
  }
  setMode("museum");

  // ── the model ──
  let obj;
  try { obj = await opt.load(opt.url); }
  catch (e) { console.error(e); $(".dio-msg").textContent = tx("fail"); return { close: () => destroy() }; }
  $(".dio-msg").remove();
  scene.add(obj);
  obj.updateMatrixWorld(true);
  const info = (obj.userData && obj.userData.diorama) || findExtras(obj) || {};
  const pois = [];
  obj.traverse(o => {
    const p = o.userData && o.userData.poi;
    if (p || /^poi[:_]/.test(o.name)) pois.push({ id: (p && p.id) || o.name.replace(/^poi[:_]?/, ""), p: p || {}, pos: o.getWorldPosition(new THREE.Vector3()) });
    if (o.isMesh) {
      for (const m of [].concat(o.material)) {
        if (m.map) m.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
        if (m.normalMap) m.normalMap.anisotropy = 4;
      }
    }
  });
  function findExtras(o) { let r = null; o.traverse(x => { if (!r && x.userData && x.userData.diorama) r = x.userData.diorama; }); return r; }
  if (info.depth) $(".dio-top .dio-sub").textContent = tx("depth", { d: Math.abs(info.depth).toLocaleString(opt.lang || "en") });

  const bb = new THREE.Box3().setFromObject(obj), size = bb.getSize(new THREE.Vector3()), mid = bb.getCenter(new THREE.Vector3());
  function overview(instant) {
    const t = info.start && info.start.target ? new THREE.Vector3().fromArray(info.start.target) : mid.clone();
    const d = (info.start && info.start.distance) || Math.max(size.x, size.z) * 0.6;
    const p = t.clone().add(new THREE.Vector3(d * 0.75, d * 0.5, d * 0.65));
    fly(p, t, instant ? 0 : 1.6);
  }

  // ── labels and the list ──
  const lbl = [];
  for (const q of pois) {
    const e = document.createElement("div"); e.className = "dio-lb " + (q.p.kind || "wreck"); e.textContent = pick(q.p.label) || q.id;
    e.onclick = ev => { ev.stopPropagation(); go(q); };
    root.appendChild(e); lbl.push([q, e]);
  }
  const pl = $(".dio-pl");
  const kinds = ["wreck", "memorial", "life"];
  pl.innerHTML = kinds.map(k => {
    const qs = pois.filter(q => (q.p.kind || "wreck") === k);
    return qs.length ? `<h4>${esc((UI.kinds || {})[k] || TXT.en.kinds[k] || k)}</h4>` + qs.map(q => `<div data-id="${esc(q.id)}">${esc(pick(q.p.label) || q.id)}</div>`).join("") : "";
  }).join("");
  pl.onclick = e => { const d = e.target.closest("[data-id]"); if (d) go(pois.find(q => q.id === d.dataset.id)); };
  let cur = null;
  function go(q) {
    cur = q;
    const t = q.pos.clone(), dir = camera.position.clone().sub(controls.target).normalize();
    const dist = q.p.view || (q.p.kind === "life" ? 4 : q.p.kind === "memorial" ? 5 : 18);
    if (dir.y < 0.25) { dir.y = 0.35; dir.normalize(); }
    fly(t.clone().add(dir.multiplyScalar(dist)), t, 1.8);
    const c = $(".dio-card");
    c.innerHTML = `<button class="dio-x">✕</button><h3>${esc(pick(q.p.label))}</h3><p>${esc(pick(q.p.text))}</p>`;
    c.hidden = false; c.querySelector(".dio-x").onclick = () => { c.hidden = true; cur = null; mark(); };
    mark();
  }
  function mark() {
    for (const [q, e] of lbl) e.classList.toggle("on", q === cur);
    pl.querySelectorAll("[data-id]").forEach(d => d.classList.toggle("on", !!cur && d.dataset.id === cur.id));
  }

  // ── camera flights ──
  let flight = null;
  function fly(pos, target, secs) {
    if (!secs) { camera.position.copy(pos); controls.target.copy(target); controls.update(); return; }
    flight = { p0: camera.position.clone(), t0: controls.target.clone(), p1: pos, t1: target, s: 0, d: secs };
  }
  const ray = new THREE.Raycaster();
  canvas.addEventListener("dblclick", e => {
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const hit = ray.intersectObject(obj, true)[0];
    if (!hit) return;
    const dir = camera.position.clone().sub(hit.point).normalize();
    const dist = Math.max(1.5, Math.min(camera.position.distanceTo(hit.point) * 0.35, 30));
    fly(hit.point.clone().add(dir.multiplyScalar(dist)), hit.point, 1.2);
  });

  // ── buttons ──
  let showLabels = true;
  root.querySelector(".dio-top").addEventListener("click", e => {
    const a = e.target.dataset && e.target.dataset.a;
    if (a === "list") { pl.hidden = !pl.hidden; e.target.classList.toggle("on", !pl.hidden); }
    if (a === "close") opt.onClose();
  });
  root.querySelector(".dio-bar").addEventListener("click", e => {
    const a = e.target.dataset && e.target.dataset.a;
    if (a === "rov" || a === "museum") setMode(a);
    if (a === "labels") { showLabels = !showLabels; e.target.classList.toggle("on", showLabels); }
    if (a === "snow") { snow.visible = !snow.visible; e.target.classList.toggle("on", snow.visible); }
    if (a === "reset") { overview(false); $(".dio-card").hidden = true; cur = null; mark(); }
  });
  const onKey = e => {
    if (e.key === "Escape" && opt.onClose) opt.onClose();
    if (e.key === "l" || e.key === "L") root.querySelector('[data-a="labels"]').click();
    if (e.key === "m" || e.key === "M") setMode(mode === "rov" ? "museum" : "rov");
  };
  addEventListener("keydown", onKey);

  function resize() {
    const w = root.clientWidth, h = Math.max(1, root.clientHeight);
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize); ro.observe(root); resize();
  overview(true);

  // ── loop ──
  const v = new THREE.Vector3(), clock = new THREE.Clock();
  let alive = true;
  renderer.setAnimationLoop(() => {
    if (!alive) return;
    const dt = Math.min(clock.getDelta(), 0.1);
    if (flight) {
      flight.s += dt / flight.d;
      const k = flight.s >= 1 ? 1 : 1 - Math.pow(1 - flight.s, 3);
      camera.position.lerpVectors(flight.p0, flight.p1, k); controls.target.lerpVectors(flight.t0, flight.t1, k);
      if (flight.s >= 1) flight = null;
    }
    controls.update();
    // the ROV lights adapt to how far away you look, like the camera's own exposure would
    if (mode === "rov") {
      const d = Math.max(1.2, camera.position.distanceTo(controls.target));
      for (const s of spots) { s.intensity = 2.6 * Math.pow(d, 1.5) + 2; s.distance = Math.max(30, d * 4); }
    }
    // marine snow: falls slowly and wraps around the camera
    if (snow.visible) {
      const a = sg.attributes.position.array, c = camera.position, hb = snowBox / 2;
      for (let i = 0; i < SN; i++) {
        let y = a[i * 3 + 1] - sv[i] * dt; a[i * 3] += Math.sin((i + clock.elapsedTime * 0.3) * 0.7) * 0.004 * dt * 10;
        a[i * 3 + 1] = y;
        for (let k = 0; k < 3; k++) { const d = a[i * 3 + k] - c.getComponent(k); if (d > hb) a[i * 3 + k] -= snowBox; else if (d < -hb) a[i * 3 + k] += snowBox; }
      }
      sg.attributes.position.needsUpdate = true;
      snow.material.size = Math.max(0.05, Math.min(0.12, camera.position.distanceTo(controls.target) * 0.004));
    }
    renderer.render(scene, camera);
    // labels follow their points; the nearest win when two would overlap
    const w = root.clientWidth, h = root.clientHeight, cand = [];
    for (const [q, e] of lbl) {
      v.copy(q.pos).project(camera);
      const dist = camera.position.distanceTo(q.pos);
      const ok = showLabels && v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05 && (dist < (q.p.kind === "wreck" ? 700 : 70) || q === cur);
      if (ok) cand.push({ e, q, x: (v.x * 0.5 + 0.5) * w, y: (-v.y * 0.5 + 0.5) * h - 14, d: q === cur ? -1 : dist });
      else e.style.display = "none";
    }
    cand.sort((a, b) => a.d - b.d);
    const placed = [];
    for (const c of cand) {
      const bw = c.e._w || (c.e._w = Math.max(60, c.e.textContent.length * 6.6 + 18)), bh = 24;
      const r = [c.x - bw / 2, c.y - bh, c.x + bw / 2, c.y];
      const hit = placed.some(p => r[0] < p[2] && r[2] > p[0] && r[1] < p[3] && r[3] > p[1]);
      c.e.style.display = hit ? "none" : "";
      if (!hit) { placed.push(r); c.e.style.transform = `translate(${c.x}px, ${c.y}px) translate(-50%,-100%)`; }
    }
  });

  function destroy() {
    alive = false; renderer.setAnimationLoop(null); removeEventListener("keydown", onKey); ro.disconnect();
    obj && obj.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) [].concat(o.material).forEach(m => { for (const k in m) if (m[k] && m[k].isTexture) m[k].dispose(); m.dispose(); }); });
    sg.dispose(); snowTex.dispose(); snow.material.dispose(); controls.dispose(); renderer.dispose();
    root.remove();
  }
  return { close: destroy, scene, camera, controls, renderer, setMode, go, pois };
}
