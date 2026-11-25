// Checks sentence_diagram.js on signed-style (ASL word order) and English sentences.
// Usage: node tools/diagram_test.mjs [--show]   (--show prints the Mermaid source of each diagram)
import { parseSentence, DiagramSession, toMermaid } from "../sentence_diagram.js";

const show = process.argv.includes("--show");
// [input, expected kind or null, expected node texts (in order, optional), expected edge labels (optional)]
const cases = [
  ["IF RAIN THEN GAME CANCEL", "decision", ["Rain?", "Game cancel"], ["Yes"]],
  ["if it rains, the game is canceled", "decision", ["Rains?", "Game canceled"]],
  ["IF WATER HOT THEN STOP OTHERWISE WAIT", "decision", ["Water hot?", "Stop", "Wait"], ["Yes", "No"]],
  ["force equals mass times acceleration", "formula", ["Force", "×", "Mass", "Acceleration"]],
  ["ENERGY EQUAL WORK", "formula", ["Energy", "Work"], ["equals"]],
  ["SUN CAUSE WATER WARM", "cause", ["Sun", "Water warm"], ["causes"]],
  ["the ice melted because the sun was hot", "cause", ["Sun hot", "Ice melted"], ["causes"]],
  ["GAME CANCEL WHY RAIN", "cause", ["Rain", "Game cancel"]],
  ["because it rained, the game was canceled", "cause", ["Rained", "Game canceled"]],
  ["heat makes ice become water", "cause", ["Heat", "Ice become water"], ["makes"]],
  ["the sun is bigger than the earth", "compare", ["Sun", "Earth"], ["bigger than"]],
  ["SUN BIG EARTH SMALL", "compare", ["Sun: big", "Earth: small"]],
  ["FIRST MIX THEN HEAT THEN COOL", "steps", ["1. Mix", "2. Heat", "3. Cool"]],
  ["first warm up the sensor, then zero it, finally record", "steps", ["1. Warm up sensor", "2. Zero", "3. Record"]],
  ["EAT FINISH SLEEP", "steps", ["1. Eat", "2. Sleep"]],
  ["wash your hands before you eat", "steps", ["1. Wash your hands", "2. You eat"]],
  ["eat after you wash", "steps", ["1. You wash", "2. Eat"]],
  ["PLANT HAVE ROOT STEM LEAF", "parts", ["Plant", "Root", "Stem", "Leaf"]],
  ["a cell has a nucleus, a membrane and cytoplasm", "parts", ["Cell", "Nucleus", "Membrane", "Cytoplasm"]],
  ["VELOCITY MEAN SPEED DIRECTION", "define", ["Velocity", "Speed direction"], ["means"]],
  ["What does force mean?", null],
  ["HELLO", null],
  ["Thank you.", null],
  ["I LIKE WATER", null],
  // injection attempt: only safe characters reach Mermaid
  ['IF x"] --> evil["click THEN y', "decision"],
];

let pass = 0, fail = 0;
for (const [input, kind, nodes, edges] of cases) {
  const g = parseSentence(input);
  const got = g?.kind ?? null;
  let ok = got === kind;
  if (ok && nodes) ok = JSON.stringify(g.nodes.map(n => n.text)) === JSON.stringify(nodes);
  if (ok && edges) ok = JSON.stringify(g.edges.map(e => e.text).filter(Boolean)) === JSON.stringify(edges);
  if (g) { const m = toMermaid(g); if (/["\]]\s*-->\s*evil|click /i.test(m.replace(/"[^"]*"/g, ""))) ok = false; }
  ok ? pass++ : fail++;
  console.log(`${ok ? "ok  " : "FAIL"} ${input}  ->  ${got}${g ? "  " + JSON.stringify(g.nodes.map(n => n.text)) : ""}`);
  if (show && g) console.log(toMermaid(g).replace(/^/gm, "      "));
}

// lines joined over time
const s = new DiagramSession(20000);
const seq = [["FIRST MIX", 0], ["NEXT HEAT", 3000], ["FINISH COOL", 6000]].map(([l, t]) => s.add(l, t));
const okSeq = seq[0] === null && seq[2]?.nodes.map(n => n.text).join("|") === "1. Mix|2. Heat|3. Cool";
okSeq ? pass++ : fail++;
console.log(`${okSeq ? "ok  " : "FAIL"} session: FIRST MIX / NEXT HEAT / FINISH COOL -> ${seq[2]?.nodes.map(n => n.text).join(" → ")}`);
const s2 = new DiagramSession(20000);
s2.add("IF RAIN", 0); const d = s2.add("GAME CANCEL", 2000);
const okIf = d?.kind === "decision" && d.nodes.map(n => n.text).join("|") === "Rain?|Game cancel";
okIf ? pass++ : fail++;
console.log(`${okIf ? "ok  " : "FAIL"} session: IF RAIN / GAME CANCEL -> ${d?.nodes.map(n => n.text).join(" → ")}`);
const s3 = new DiagramSession(20000);
s3.add("FIRST MIX", 0); const late = s3.add("NEXT HEAT", 60000);
const okLate = late === null; // too long a gap: not joined
okLate ? pass++ : fail++;
console.log(`${okLate ? "ok  " : "FAIL"} session: a step 60 s later starts over`);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
