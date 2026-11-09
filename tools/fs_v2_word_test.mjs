// Word accuracy of fingerspelling v2 on held-out signers' English sentences, three ways:
//   raw letters | letters repaired by inferWord (top-3 guesses per letter) | whole-word scoring (scoreWord)
// The scoring code is pulled straight out of index.html, so this tests what the app runs.
//
// Input (not committed: derived from the Kaggle data):
//   python training/train_fingerspell.py lp --npz DIR --csv train.csv,supplemental_metadata.csv --cases lp.json
// Run from the repo root:
//   node tools/fs_v2_word_test.mjs lp.json            report on the test half
//   node tools/fs_v2_word_test.mjs lp.json --grid     try settings on the tuning half
// Sentences alternate between a tuning half (settings are chosen on it) and a test half (reported).
import fs from "node:fs";
const args = process.argv.slice(2);
const file = args.find(a => !a.startsWith("--"));
const flag = n => args.includes(`--${n}`);
const num = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? Number(args[i + 1]) : d; };

const html = fs.readFileSync("index.html", "utf8");
const start = html.indexOf("// ---- word correction"), end = html.indexOf("// ---- end word correction");
const eng = JSON.parse(fs.readFileSync("models/english_dictionary.json", "utf8"));
const stem = JSON.parse(fs.readFileSync("models/stem_dictionary.json", "utf8"));
const domain = JSON.parse(fs.readFileSync("models/domain_words.json", "utf8"));
const chars = JSON.parse(fs.readFileSync("models/fingerspell_v2.json", "utf8")).chars;
const mergeSrc = html.match(/const merged = \{ \.\.\.eng \};[\s\S]*?dictionary = [^\n]*\n/)[0];
const dictionary = new Function("eng", "stem", "domain", `let dictionary; ${mergeSrc}; return dictionary;`)(eng, stem, domain);
const { inferWord, scoreWord, ctcWordSpans, WW, ctcBeamLine, BEAM } = new Function("dictionary",
  `${html.slice(start, end)}; return { inferWord, scoreWord, ctcWordSpans, WW, ctcBeamLine, BEAM };`)(dictionary);

const lev = (a, b) => { const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) d[i][j] = Math.min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (a[i-1] === b[j-1] ? 0 : 1));
  return d[a.length][b.length]; };

const { cases, test_signers } = JSON.parse(fs.readFileSync(file, "utf8"));
const bin = fs.readFileSync(file + ".bin");
const all = new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4);
const lpOf = c => all.subarray(c.offset, c.offset + c.S * c.C);

// the app's old path: top-3 letter guesses at each emitted letter, then inferWord
function top3Words(data, S, C) {
  const isLetter = c => c > 0 && /[a-z]/.test(chars[c - 1]);
  const words = [[]]; let prev = 0;
  for (let s = 0; s < S; s++) {
    const row = data.subarray(s * C, (s + 1) * C);
    let best = 0;
    for (let c = 1; c < C; c++) if (row[c] > row[best]) best = c;
    if (best !== 0 && best !== prev) {
      if (chars[best - 1] === " ") words.push([]);
      else if (isLetter(best)) words[words.length - 1].push([...row.keys()].filter(isLetter).sort((x, y) => row[y] - row[x]).slice(0, 3).map(c => [chars[c - 1].toUpperCase(), Math.exp(row[c])]));
    }
    prev = best;
  }
  return words.filter(w => w.length);
}

function evaluate(set, withOld) {
  const groups = { "all clips": () => true, "hand found in 60%+ of frames": c => c.present >= 0.6, "hand found in 90%+ of frames": c => c.present >= 0.9 };
  const out = {};
  for (const [name, sel] of Object.entries(groups)) {
    let N = 0, dRaw = 0, dOld = 0, dNew = 0, dBeam = 0, dBoth = 0, n = 0;
    for (const c of set.filter(sel)) {
      const ref = c.ref.split(" ").filter(Boolean), data = lpOf(c);
      const spans = ctcWordSpans(data, c.S, c.C, chars).words;
      N += ref.length; n++;
      dRaw += lev(spans.map(w => w.raw), ref);
      dNew += lev(spans.map(w => scoreWord(data, c.C, w, chars).word), ref);
      const beam = ctcBeamLine(data, c.S, c.C, chars);
      dBeam += lev(beam, ref);
      // whole line, then snap any word the dictionary doesn't know to the closest dictionary word
      dBoth += lev(beam.map(w => dictionary[w] || !/^[a-z]+$/.test(w) ? w : inferWord([...w].map(ch => [[ch.toUpperCase(), 0.6]])).word), ref);
      if (withOld) dOld += lev(top3Words(data, c.S, c.C).map(w => inferWord(w).word), ref);
    }
    out[name] = { sentences: n, words: N, raw_letters: +(1 - dRaw / N).toFixed(3), top3_then_dictionary: withOld ? +(1 - dOld / N).toFixed(3) : undefined, whole_word: +(1 - dNew / N).toFixed(3),
      whole_line: +(1 - dBeam / N).toFixed(3), whole_line_then_snap: +(1 - dBoth / N).toFixed(3) };
  }
  return out;
}

const tune = cases.filter((_, i) => i % 2 === 0), test = cases.filter((_, i) => i % 2 === 1);
if (flag("grid")) {
  const good = tune.filter(c => c.present >= 0.6);
  console.log(`tuning half: ${good.length} sentences with the hand found in 60%+ of frames`);
  for (const prior of [0.1, 0.2, 0.3, 0.45, 0.6]) for (const oov of [1, 3, 5, 8]) {
    WW.prior = BEAM.prior = prior; WW.oov = BEAM.oov = oov;
    const t0 = Date.now(), r = evaluate(good, false)["all clips"];
    console.log(`prior ${prior} oov ${oov}: whole-word ${r.whole_word}, whole-line ${r.whole_line}, line+snap ${r.whole_line_then_snap} (raw ${r.raw_letters}) ${((Date.now() - t0) / r.words).toFixed(0)} ms/word`);
  }
} else {
  WW.prior = num("prior", WW.prior); WW.oov = num("oov", WW.oov);
  BEAM.prior = num("bprior", BEAM.prior); BEAM.oov = num("boov", BEAM.oov);
  const t0 = Date.now(), res = evaluate(test, true);
  console.log(JSON.stringify({ test_signers, settings: WW, word_accuracy: res }, null, 1));
  console.log(`about ${((Date.now() - t0) / res["all clips"].words / 2).toFixed(0)} ms per word`);
  for (const c of test.filter(c => c.present >= 0.6).slice(0, 12)) {
    const data = lpOf(c), spans = ctcWordSpans(data, c.S, c.C, chars).words;
    console.log(`  "${c.ref}"\n    read:  "${spans.map(w => w.raw).join(" ")}"\n    old:   "${top3Words(data, c.S, c.C).map(w => inferWord(w).word).join(" ")}"\n    word:  "${spans.map(w => scoreWord(data, c.C, w, chars).word).join(" ")}"\n    line:  "${ctcBeamLine(data, c.S, c.C, chars).join(" ")}"`);
  }
}
