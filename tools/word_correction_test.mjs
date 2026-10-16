// Spelling-mistake test for fingerspelling word correction.
//
// Takes real words, simulates the letter sequence the app would commit for them
// (confusing similar handshapes, dropping or adding letters, double letters coming
// out single), and checks whether inferWord() recovers the original word.
//
// inferWord() and levenshteinWeighted() are pulled straight out of index.html, so
// this always tests the code the app actually runs.
//
// The letter confusions below are a hand-written model of which ASL handshapes look
// alike, NOT measured from the letter model. Treat the numbers as a way to compare
// versions of the correction code, not as real-world accuracy.
//
// Run from the repo root:  node tools/word_correction_test.mjs [--words 200] [--seed 1] [--show 10]

import fs from "node:fs";

const args = Object.fromEntries(
  process.argv.slice(2).join(" ").split("--").filter(Boolean).map(s => s.trim().split(/\s+/))
);
const N_WORDS = Number(args.words ?? 200); // per word list
const SEED = Number(args.seed ?? 1);
const SHOW = Number(args.show ?? 8);       // example failures to print per level

// ---- load the app's word-correction code and dictionary exactly as index.html does ----
const html = fs.readFileSync("index.html", "utf8");
const start = html.indexOf("// ---- word correction");
const end = html.indexOf("// ---- end word correction");
if (start < 0 || end < 0) throw new Error("couldn't find inferWord code in index.html");
const eng = JSON.parse(fs.readFileSync("models/english_dictionary.json", "utf8"));
const stem = JSON.parse(fs.readFileSync("models/stem_dictionary.json", "utf8"));
const mergeSrc = html.match(/const merged = \{ \.\.\.eng \};[\s\S]*?dictionary = [^\n]*\n/)[0];
const dictionary = new Function("eng", "stem", `let dictionary; ${mergeSrc}; return dictionary;`)(eng, stem);
const inferWord = new Function("dictionary", `${html.slice(start, end)}; return inferWord;`)(dictionary);

// ---- seeded RNG so runs are repeatable ----
let s = SEED >>> 0;
const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const uniform = (a, b) => a + (b - a) * rand();
const pick = arr => arr[Math.floor(rand() * arr.length)];
function sample(arr, n) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a.slice(0, n);
}

// Letters whose handshapes look alike (to a person, and roughly to landmarks).
const GROUPS = ["aestmn", "uvrk", "gh", "pk", "gq", "co", "fw", "yi", "ij", "dz", "xd", "bf"];
const LETTERS = "abcdefghijklmnopqrstuvwxyz";
function confusable(ch) {
  const near = [...new Set(GROUPS.filter(g => g.includes(ch)).join("").replace(new RegExp(ch, "g"), ""))];
  return near.length ? pick(near) : pick([...LETTERS.replace(ch, "")]);
}
// N, M and T self-occlude, and MediaPipe often loses the hand entirely on them.
const HARD = new Set(["m", "n", "t"]);

// One committed letter = the app's top-3 list: [[LETTER, prob], ...], top-1 first.
function letterEntry(top, topConf, others) {
  let rest = 1 - topConf;
  const entry = [[top.toUpperCase(), topConf]];
  for (const [ch, share] of others) { entry.push([ch.toUpperCase(), rest * share]); }
  return entry;
}

function simulate(word, lvl) {
  const out = [];
  for (const ch of word) {
    if (rand() < lvl.drop * (HARD.has(ch) ? 2 : 1)) continue;
    if (rand() < lvl.sub) {
      // misread: a look-alike wins; the true letter is in the top 3 most of the time
      const wrong = confusable(ch);
      const second = rand() < 0.7 ? ch : confusable(ch);
      out.push(letterEntry(wrong, uniform(0.4, 0.9), [[second, 0.7], [confusable(wrong), 0.2]]));
    } else {
      out.push(letterEntry(ch, uniform(0.6, 0.99), [[confusable(ch), 0.6], [confusable(ch), 0.3]]));
    }
    if (rand() < lvl.ins) {
      // a stray letter from the hand passing through another shape between letters
      const stray = confusable(ch);
      out.push(letterEntry(stray, uniform(0.4, 0.8), [[confusable(stray), 0.6], [ch, 0.3]]));
    }
  }
  // the app never commits the same letter twice in a row
  return out.filter((e, i) => i === 0 || e[0][0] !== out[i - 1][0][0]);
}

const LEVELS = [
  { name: "clean (doubles only)", sub: 0, drop: 0, ins: 0 },
  { name: "light noise", sub: 0.1, drop: 0.03, ins: 0.03 },
  { name: "medium noise", sub: 0.2, drop: 0.06, ins: 0.06 },
  { name: "heavy noise", sub: 0.35, drop: 0.1, ins: 0.1 },
];

// ---- test words: common STEM Wiki words and common English words ----
const isWord = w => /^[a-z]{3,12}$/.test(w);
const topWords = (d, n) => Object.entries(d).filter(([w]) => isWord(w)).sort((a, b) => b[1] - a[1]).slice(0, n).map(([w]) => w);
const LISTS = {
  STEM: sample(topWords(stem, 3000), N_WORDS),
  English: sample(topWords(eng, 10000), N_WORDS),
};

console.log(`dictionary: ${Object.keys(dictionary).length} words | ${N_WORDS} words per list | seed ${SEED}\n`);
const t0 = Date.now();
let calls = 0;
for (const lvl of LEVELS) {
  console.log(`== ${lvl.name} (misread ${lvl.sub * 100}%, dropped ${lvl.drop * 100}%, extra ${lvl.ins * 100}%)`);
  const fails = [];
  for (const [listName, words] of Object.entries(LISTS)) {
    let rawOk = 0, fixedOk = 0;
    for (const w of words) {
      const cands = simulate(w, lvl);
      if (!cands.length) continue;
      const raw = cands.map(c => c[0][0].toLowerCase()).join("");
      const { word } = inferWord(cands);
      calls++;
      if (raw === w) rawOk++;
      if (word === w) fixedOk++; else fails.push(`${w} -> "${raw}" -> ${word}`);
    }
    const pct = n => `${((100 * n) / words.length).toFixed(1)}%`.padStart(6);
    console.log(`  ${listName.padEnd(8)} raw letters ${pct(rawOk)}   after correction ${pct(fixedOk)}`);
  }
  if (SHOW) console.log("  e.g. " + fails.slice(0, SHOW).join(" | "));
  console.log();
}
console.log(`inferWord: ${((Date.now() - t0) / calls).toFixed(0)} ms per word`);
