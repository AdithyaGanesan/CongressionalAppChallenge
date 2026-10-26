// Word accuracy of fingerspelling v2 on held-out English sentences: raw letters vs after index.html's
// inferWord (the dictionary correction the app runs). Input from
//   python training/train_fingerspell.py words --npz ... --csv ... --work ... --cases words.json
// (not committed: derived from the Kaggle data). Run from the repo root: node tools/fs_v2_word_test.mjs words.json
import fs from "node:fs";
const html = fs.readFileSync("index.html", "utf8");
const start = html.indexOf("// ---- word correction"), end = html.indexOf("// ---- end word correction");
const eng = JSON.parse(fs.readFileSync("models/english_dictionary.json", "utf8"));
const stem = JSON.parse(fs.readFileSync("models/stem_dictionary.json", "utf8"));
const domain = JSON.parse(fs.readFileSync("models/domain_words.json", "utf8"));
const mergeSrc = html.match(/const merged = \{ \.\.\.eng \};[\s\S]*?dictionary = [^\n]*\n/)[0];
const dictionary = new Function("eng", "stem", "domain", `let dictionary; ${mergeSrc}; return dictionary;`)(eng, stem, domain);
const { inferWord } = new Function("dictionary", `${html.slice(start, end)}; return { inferWord };`)(dictionary);
const lev = (a, b) => { const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) d[i][j] = Math.min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (a[i-1] === b[j-1] ? 0 : 1));
  return d[a.length][b.length]; };
const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const res = {};
for (const [name, sel] of [["all", c => true], ["hand_found_60pct+", c => c.present >= 0.6], ["hand_found_90pct+", c => c.present >= 0.9]]) {
  let N = 0, Draw = 0, Dfix = 0, n = 0;
  for (const c of cases.filter(sel)) {
    const ref = c.ref.split(" ").filter(Boolean);
    const raw = c.words.map(w => w.map(l => l[0][0].toLowerCase()).join(""));
    const fixed = c.words.map(w => inferWord(w).word);
    N += ref.length; Draw += lev(raw, ref); Dfix += lev(fixed, ref); n++;
  }
  res[name] = { sentences: n, words: N, word_accuracy_raw_letters: +(1 - Draw / N).toFixed(3), word_accuracy_after_dictionary: +(1 - Dfix / N).toFixed(3) };
}
console.log(JSON.stringify(res, null, 1));
for (const c of cases.filter(c => c.present >= 0.6).slice(0, 10)) console.log(`  "${c.ref}"\n    raw: "${c.raw}"\n    app: "${c.words.map(w => inferWord(w).word).join(" ")}"`);
