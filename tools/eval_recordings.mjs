// Score capture-page recordings by replaying them through the real app in headless Chromium.
//
// Every take (capture.html "Save zip", format stat-capture-v1) goes through the same path as the
// camera: analyzeFrame -> line detection -> processLine. The report separates the parts that
// can fail: hand tracking, how the signing was cut into lines, and what was recognized.
//
//   node tools/eval_recordings.mjs TAKES [--params "model=v2&fs=v2"] [--out report] [--cdn DIR]
//
// TAKES: a capture zip, or a folder of take .json files.
// --params: app URL settings (default "model=v2&fs=v2"); debug=1 is always added.
// --out:    writes OUT.json and OUT.md (default: eval_report in the current folder).
// --cdn:    a node_modules folder holding onnxruntime-web@1.19.2, @mediapipe/tasks-vision@0.10.14
//           and mermaid@10.9.1, served in place of jsDelivr (only for networks that block it).
// Needs: Node, Playwright with Chromium, Python 3 (serves the app locally). Run from the repo root.
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn, execFileSync } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : def; };
const src = args.find(a => !a.startsWith("--") && args[args.indexOf(a) - 1]?.startsWith("--") !== true);
if (!src) { console.error("usage: node tools/eval_recordings.mjs TAKES [--params ...] [--out ...] [--cdn DIR]"); process.exit(1); }
const appParams = opt("params", "model=v2&fs=v2");
const outBase = opt("out", "eval_report");
const cdnDir = opt("cdn", null);

// ---- load takes ----
function loadTakes(p) {
  let dir = p;
  if (p.endsWith(".zip")) {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "stat-takes-"));
    execFileSync("python3", ["-m", "zipfile", "-e", p, dir]);
  }
  return fs.readdirSync(dir).filter(f => f.endsWith(".json") && f !== "manifest.json").sort()
    .map(f => ({ file: f, ...JSON.parse(fs.readFileSync(path.join(dir, f), "utf8")) }))
    .filter(t => t.format === "stat-capture-v1" && Array.isArray(t.frames));
}
const takes = loadTakes(src);
if (!takes.length) { console.error("no stat-capture-v1 takes found in", src); process.exit(1); }

// ---- scoring helpers ----
const norm = s => String(s).toUpperCase().replace(/[^A-Z]/g, "");
// a phrase label like "TODAY C-A-L-I-B-R-A-T-E": hyphenated single letters are a fingerspelled word
const labelTokens = label => label.trim().split(/\s+/).map(t =>
  /^([A-Za-z]-)+[A-Za-z]$/.test(t) ? { word: t.replace(/-/g, "").toLowerCase(), spelled: true } : { word: norm(t), spelled: false });
function matchTokens(expected, got) {
  const pool = got.map(norm);
  let hit = 0;
  for (const e of expected) {
    const i = pool.indexOf(norm(e.word));
    if (i >= 0) { hit++; pool.splice(i, 1); }
  }
  return hit;
}
const v2Read = lines => lines.flatMap(l => l.details).map(d => (d.match(/^v2 read: "(.*)"$/) || [])[1]).filter(x => x !== undefined).join(" ");

// ---- serve the app and open it ----
const port = 8800 + Math.floor(Math.random() * 100);
const server = spawn("python3", ["-m", "http.server", String(port)], { stdio: "ignore" });
await new Promise(r => setTimeout(r, 1000));
const { chromium } = require(process.env.PLAYWRIGHT_PATH || "playwright");
const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] });
const page = await (await browser.newContext({ permissions: ["camera"] })).newPage();
const pageErrors = [];
page.on("pageerror", e => pageErrors.push(e.message));
if (cdnDir) {
  await page.route("https://cdn.jsdelivr.net/npm/**", async r => {
    const rest = new URL(r.request().url()).pathname.replace("/npm/", "");
    const file = path.join(cdnDir, rest.replace(/@(?=\d)[^/]+/, ""));
    if (!fs.existsSync(file)) return r.abort();
    r.fulfill({ status: 200, body: fs.readFileSync(file), headers: { "content-type": file.endsWith(".wasm") ? "application/wasm" : "text/javascript", "access-control-allow-origin": "*" } });
  });
}
const results = [];
try {
  await page.goto(`http://localhost:${port}/index.html?debug=1&${appParams}`);
  await page.click("#startBtn");
  await page.waitForFunction(() => document.getElementById("status").classList.contains("hidden") || document.getElementById("err").textContent, null, { timeout: 180000 });
  const err = await page.evaluate(() => document.getElementById("err").textContent);
  if (err) throw new Error("app failed to start: " + err);
  await page.waitForFunction(() => window.__stat?.replay, null, { timeout: 30000 });

  for (const t of takes) {
    const mode = t.type === "letter" ? "spell" : "auto"; // a held letter is one letter, never a line of signs
    const r = await page.evaluate(({ take, mode }) => window.__stat.replay(take, { mode }), { take: t, mode });
    const tokens = r.entries.flatMap(e => e.tokens).filter(x => x !== "?");
    const row = { file: t.file, type: t.type, label: t.label, signer: t.signer, handFrames: +r.handFrames.toFixed(2),
      lines: r.lines.length, kinds: r.lines.map(l => l.kind).join(","), english: r.entries.map(e => e.english).join(" / "),
      raw: r.entries.map(e => e.raw).join(" / "), read: v2Read(r.lines) };
    if (t.type === "letter") {
      const letters = row.read.replace(/[^a-z]/gi, "").toUpperCase();
      row.correct = letters.length > 0 && [...new Set(letters)].join("") === norm(t.label);
      row.score = letters.includes(norm(t.label)) ? 1 : 0;
    } else {
      const exp = t.type === "sign" ? [{ word: norm(t.label), spelled: false }] : labelTokens(t.label);
      row.expected = exp.length;
      row.matched = matchTokens(exp, tokens);
      row.extra = Math.max(0, tokens.length - row.matched);
      row.correct = row.matched === exp.length && row.extra === 0;
      row.score = row.matched / exp.length;
    }
    results.push(row);
    process.stdout.write(`${row.correct ? "✓" : "✗"} ${t.type.padEnd(6)} ${t.label.padEnd(28)} -> ${row.raw || "(nothing)"}\n`);
  }
} finally {
  await browser.close();
  server.kill();
}

// ---- summary ----
const by = k => results.filter(r => r.type === k);
const pct = x => `${(100 * x).toFixed(0)}%`;
const mean = (a, f) => a.length ? a.reduce((s, r) => s + f(r), 0) / a.length : 0;
const summary = {};
for (const k of ["letter", "sign", "phrase"]) {
  const rs = by(k);
  if (!rs.length) continue;
  const good = rs.filter(r => r.handFrames >= 0.6), poor = rs.filter(r => r.handFrames < 0.6);
  summary[k] = {
    takes: rs.length, exact: mean(rs, r => +r.correct), tokenScore: mean(rs, r => r.score),
    handFrames: mean(rs, r => r.handFrames),
    oneLine: k === "letter" ? undefined : mean(rs, r => +(r.lines === (k === "sign" ? 1 : r.expected))),
    wellTracked: { takes: good.length, exact: mean(good, r => +r.correct) },
    poorlyTracked: { takes: poor.length, exact: mean(poor, r => +r.correct) },
  };
}
const report = { app: appParams, when: new Date().toISOString(), summary, pageErrors, results };
fs.writeFileSync(`${outBase}.json`, JSON.stringify(report, null, 1));
const md = [`# STAT evaluation (${appParams})`, "", `${results.length} takes from ${src}`, ""];
md.push("| Type | Takes | Exactly right | Words/signs found | Hand found in frames | Right line count |", "|---|---|---|---|---|---|");
for (const [k, s] of Object.entries(summary)) {
  md.push(`| ${k} | ${s.takes} | ${pct(s.exact)} | ${pct(s.tokenScore)} | ${pct(s.handFrames)} | ${s.oneLine === undefined ? "n/a" : pct(s.oneLine)} |`);
}
md.push("", "Split by hand tracking (hand found in at least 60% of frames):", "");
for (const [k, s] of Object.entries(summary)) {
  md.push(`- ${k}: well tracked ${s.wellTracked.takes} takes, ${pct(s.wellTracked.exact)} right; poorly tracked ${s.poorlyTracked.takes} takes, ${pct(s.poorlyTracked.exact)} right`);
}
md.push("", "| | Type | Label | Output | Letters read | Hands | Lines |", "|---|---|---|---|---|---|---|");
for (const r of results) md.push(`| ${r.correct ? "✓" : "✗"} | ${r.type} | ${r.label} | ${r.raw || "(nothing)"} | ${r.read || ""} | ${pct(r.handFrames)} | ${r.lines} (${r.kinds}) |`);
if (pageErrors.length) md.push("", "Page errors:", ...pageErrors.map(e => `- ${e}`));
fs.writeFileSync(`${outBase}.md`, md.join("\n") + "\n");
console.log("\n" + md.slice(4, 4 + 2 + Object.keys(summary).length).join("\n"));
console.log(`\nfull report: ${outBase}.md and ${outBase}.json`);
