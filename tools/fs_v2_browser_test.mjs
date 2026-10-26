// Headless-browser check for ?fs=v2: the page loads with no console errors, the v2 fingerspelling
// model runs, and the browser's CTC decoding matches Python's on the same landmark sequences.
//
// Cases file (JSON list of {T, x: [T*63 floats], python: "decoded text"}) comes from
//   python training/train_fingerspell.py cases --npz ... --csv ... --work ... --cases cases.json
// (held-out test signers; not committed, it is derived from the Kaggle data).
//
// CDN files are served from a local npm folder with the same package versions, so this works offline:
//   npm i onnxruntime-web@1.19.2 @mediapipe/tasks-vision@0.10.14 mermaid@10.9.1   (in NPM_DIR)
//   NPM_DIR/tasks/{hand_landmarker,pose_landmarker_lite}.task   (from storage.googleapis.com)
// Run from the repo root:  node tools/fs_v2_browser_test.mjs NPM_DIR cases.json
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");

const [npmDir, casesPath] = process.argv.slice(2);
const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".task": "application/octet-stream" };
const serve = (route, file) => fs.existsSync(file)
  ? route.fulfill({ status: 200, body: fs.readFileSync(file), headers: { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "access-control-allow-origin": "*" } })
  : route.fulfill({ status: 404, body: "not found: " + file });

const browser = await chromium.launch({ args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
const context = await browser.newContext({ permissions: ["camera"] });
const page = await context.newPage();
const errors = [];
page.on("console", m => { if (m.type() === "error") errors.push(m.text()); else console.log("  [page]", m.text()); });
page.on("pageerror", e => errors.push(String(e)));
await page.route("http://localhost:8123/**", r => serve(r, path.join(".", new URL(r.request().url()).pathname)));
await page.route("https://cdn.jsdelivr.net/npm/**", r => {
  // /npm/<pkg>@<version>/<file>  →  NPM_DIR/node_modules/<pkg>/<file>
  const m = new URL(r.request().url()).pathname.match(/^\/npm\/((?:@[^/]+\/)?[^@/]+)@[^/]+\/(.*)$/);
  return m ? serve(r, path.join(npmDir, "node_modules", m[1], m[2])) : r.abort();
});
await page.route("https://storage.googleapis.com/**", r => serve(r, path.join(npmDir, "tasks", path.basename(new URL(r.request().url()).pathname))));

await page.goto("http://localhost:8123/index.html?fs=v2");
await page.click("#startBtn");
await page.waitForFunction(() => /ok/.test(document.getElementById("onnxStatus").textContent) || /failed/.test(document.getElementById("onnxStatus").textContent), null, { timeout: 120000 });
const status = await page.textContent("#onnxStatus");
console.log("onnx status:", status);

let same = 0;
for (const c of cases) {
  const got = await page.evaluate(([x, T]) => window.statTest.decodeV2(x, T), [c.x, c.T]);
  const ok = got === c.python;
  same += ok;
  console.log(`${ok ? "same" : "DIFF"}  browser "${got}"  python "${c.python}"  (true: "${c.ref}")`);
}
// whole line through processLine (forced spell mode): frames at 15 fps built from the first case
const line = await page.evaluate(async ([x, T]) => {
  const frames = [];
  for (let i = 0; i < T; i++) {
    const h = new Float32Array(x.slice(i * 63, i * 63 + 63));
    frames.push({ t: i * 1000 / 15, rawVec: new Float32Array(150), handsUp: 1, domHand63: h, domWrist: [h[0] * 480, h[1] * 640], scale: 200 });
  }
  window.statTest.setMode("spell");
  await window.statTest.processLine(frames);
  window.statTest.setMode("auto");
  await window.statTest.processLine(frames);
  return window.statTest.transcript().map(e => `${e.kind}: ${e.english}  [${e.raw}]`);
}, [cases[0].x, cases[0].T]);
console.log("processLine (spell, then auto):\n  " + line.join("\n  "));
if (!line.length) errors.push("processLine produced no line");
await page.waitForTimeout(3000); // let the camera loop run a little
console.log(`\n${same}/${cases.length} decodes identical; console errors: ${errors.length}`);
errors.forEach(e => console.log("  ERROR", e));
await browser.close();
process.exit(same === cases.length && errors.length === 0 && /v2/.test(status) ? 0 : 1);
