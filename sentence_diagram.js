// STAT: turn a translated sentence into a diagram (flowchart / concept map), in the browser, no AI.
//
// Input is the list of words of one line: recognized signs (ASL order, e.g. "rain why game cancel")
// or plain English ("if it rains, the game is canceled"). The parser looks for the word patterns that
// carry structure, in this order:
//   if / then / otherwise          -> decision (diamond with Yes / No)
//   equal / same / plus / times    -> formula
//   because / cause / why / result -> cause and effect
//   more / less / bigger / than    -> comparison
//   first / then / next / finish   -> steps
//   have / has / include           -> parts of a whole
//   mean / is                      -> definition
// DiagramSession joins lines: "FIRST MIX" then "NEXT HEAT" then "FINISH COOL" grows one flowchart.
// toMermaid() writes Mermaid flowchart source. Recognized text only reaches it through label(), which
// keeps letters, digits, spaces and a few safe marks, so nothing a signer says can inject Mermaid code.

const STOP = new Set(["the", "a", "an", "it", "its", "of", "is", "are", "was", "be", "will", "to"]);
const SEQ_START = new Set(["first", "start", "begin", "step"]);
const SEQ = new Set(["then", "next", "after", "afterward", "later", "finally", "finish", "last", "second", "third", "end"]);
const CAUSE_FWD = ["cause", "causes", "caused", "result", "results", "effect", "so", "therefore", "lead", "leads", "make", "makes", "become", "becomes"];
const CAUSE_BACK = ["because", "since", "why"]; // "X because Y" and ASL "X WHY? Y": Y causes X
const EQUAL = ["equal", "equals", "same", "="];
const OPS = { plus: "+", add: "+", minus: "−", subtract: "−", times: "×", multiply: "×", multiplied: "×", divide: "÷", divided: "÷", over: "÷" };
const MORE = ["more", "bigger", "larger", "greater", "faster", "hotter", "higher", "big", "heavier", "stronger"];
const LESS = ["less", "smaller", "fewer", "slower", "colder", "lower", "small", "lighter", "weaker"];
const HAS = ["have", "has", "include", "includes", "contain", "contains"];
const MEANS = ["mean", "means"];
const ELSE = ["otherwise", "else"];

/** Normalize a line: lower case, hyphenated glosses to words, punctuation to separators (","). */
export function words(input) {
  const s = Array.isArray(input) ? input.filter(t => t !== "?").join(" ") : String(input);
  return s.toLowerCase().replace(/[;:!?]/g, " , ").replace(/,/g, " , ").replace(/\./g, " , ")
    .replace(/(\w)-(\w)/g, "$1 $2").split(/\s+/).filter(Boolean);
}

/** A safe, short node label: letters, digits, spaces and + − × ÷ = ' only; first letter upper case. */
export function label(ws) {
  let s = ws.filter(w => w !== "," && !STOP.has(w)).join(" ");
  s = s.replace(/[^a-z0-9 +−×÷=']/gi, "").replace(/\s+/g, " ").trim();
  if (s.length > 40) s = s.slice(0, 39).replace(/\s\S*$/, "") + "…";
  return s ? s[0].toUpperCase() + s.slice(1) : "";
}

const clean = ws => ws.filter(w => w !== ",");
const idxOf = (ws, list) => ws.findIndex(w => list.includes(w));
function splitAt(ws, isSep) { // split a word list at separator words, dropping the separators
  const parts = [[]];
  for (const w of ws) { if (isSep(w)) parts.push([]); else parts[parts.length - 1].push(w); }
  return parts.map(p => label(p)).filter(Boolean);
}

// ---------- one-line patterns; each returns a graph {kind, title, nodes:[{id,text,shape}], edges:[{from,to,text}]} ----------
function graph(kind, title) {
  const g = { kind, title, nodes: [], edges: [] };
  g.add = (text, shape = "box") => { const id = `n${g.nodes.length}`; g.nodes.push({ id, text, shape }); return id; };
  g.link = (from, to, text = "") => g.edges.push({ from, to, text });
  return g;
}

function decision(ws) {
  const i = ws.indexOf("if");
  if (i < 0) return null;
  const rest = ws.slice(i + 1);
  let j = rest.indexOf("then");
  if (j < 0) j = rest.indexOf(","); // "if it rains, the game is canceled"
  const e = idxOf(rest, ELSE);
  const cond = label(j < 0 ? (e < 0 ? rest : rest.slice(0, e)) : rest.slice(0, j));
  if (!cond) return null;
  const yes = j < 0 ? "" : label(rest.slice(j + 1, e < 0 ? undefined : e));
  const no = e < 0 ? "" : label(rest.slice(e + 1));
  const g = graph("decision", "If … then");
  const before = label(ws.slice(0, i)); // "cancel game if rain": the result came first
  const q = g.add(`${cond}?`, "diamond");
  const y = yes || before;
  if (y) g.link(q, g.add(y), "Yes");
  if (no) g.link(q, g.add(no), "No");
  g.open = !y; // the result may come on the next line
  return g;
}

function formula(ws) {
  const e = idxOf(ws, EQUAL);
  if (e < 1) return null;
  const left = label(ws.slice(0, e)), right = clean(ws.slice(e + 1));
  if (!left || !right.length) return null;
  const terms = [[]], ops = [];
  for (const w of right) { if (OPS[w]) { ops.push(OPS[w]); terms.push([]); } else if (w !== "by") terms[terms.length - 1].push(w); }
  const labels = terms.map(label);
  if (labels.some(t => !t)) return null;
  const g = graph("formula", "Formula");
  const top = g.add(left);
  if (!ops.length) { g.link(top, g.add(labels[0]), "equals"); return g; }
  const expr = labels.map((t, k) => (k ? `${ops[k - 1]} ` : "") + t).join(" ");
  g.title = `Formula: ${left} = ${expr}`;
  const op = g.add(ops.every(o => o === ops[0]) ? ops[0] : "=", "circle");
  g.link(top, op, "equals");
  labels.forEach(t => g.link(op, g.add(t)));
  return g;
}

function causeEffect(ws) {
  let k = idxOf(ws, CAUSE_FWD), back = false;
  const kb = idxOf(ws, CAUSE_BACK);
  if (kb >= 0 && (k < 0 || kb < k)) { k = kb; back = true; }
  if (k < 0) return null;
  const a = ws.slice(0, k);
  let b = ws.slice(k + 1);
  if (["to", "in"].includes(b[0])) b = b.slice(1); // "lead to", "result in"
  const A = label(a), B = label(b);
  if (back && !A && B) { // "because X, Y"
    const parts = splitAt(b, w => w === ",");
    if (parts.length < 2) return null;
    const g = graph("cause", "Cause and effect");
    g.link(g.add(parts[0]), g.add(parts.slice(1).join(" ")), "causes");
    return g;
  }
  if (!A || !B) return null;
  const g = graph("cause", "Cause and effect");
  const verb = ["become", "becomes"].includes(ws[k]) ? "becomes" : ["make", "makes"].includes(ws[k]) ? "makes" : "causes";
  if (back) g.link(g.add(B), g.add(A), "causes");
  else g.link(g.add(A), g.add(B), verb);
  return g;
}

function comparison(ws) {
  const t = ws.indexOf("than");
  if (t > 0) { // "x is bigger than y" / "x more energy than y"
    const before = ws.slice(0, t), m = before.findLastIndex(w => MORE.includes(w) || LESS.includes(w) || w.endsWith("er"));
    if (m < 1) return null;
    const A = label(before.slice(0, m)), B = label(ws.slice(t + 1)), rel = label(before.slice(m)).toLowerCase();
    if (!A || !B) return null;
    const g = graph("compare", "Comparison");
    g.link(g.add(A), g.add(B), `${rel} than`);
    return g;
  }
  // ASL: "SUN BIG EARTH SMALL" — two things, each followed by a size word
  const marks = ws.map((w, i) => (MORE.includes(w) || LESS.includes(w) ? i : -1)).filter(i => i >= 0);
  if (marks.length === 2 && marks[0] > 0 && marks[1] > marks[0] + 1) {
    const A = label(ws.slice(0, marks[0])), B = label(ws.slice(marks[0] + 1, marks[1]));
    if (!A || !B) return null;
    const g = graph("compare", "Comparison");
    g.add(`${A}: ${ws[marks[0]]}`); g.add(`${B}: ${ws[marks[1]]}`);
    g.link("n0", "n1", "compared with");
    return g;
  }
  return null;
}

function steps(ws) {
  const markers = ws.filter(w => SEQ.has(w) || SEQ_START.has(w)).length;
  // "after X, Y": X comes first; "X before Y" keeps order; "X after Y": Y first
  const a = ws.indexOf("after"), b = ws.indexOf("before");
  if (b > 0 && markers === 0) {
    const parts = [label(ws.slice(0, b)), label(ws.slice(b + 1))];
    if (parts.every(Boolean)) return chain(parts);
  }
  if (markers === 0) return null;
  if (a > 0 && !ws.slice(0, a).some(w => SEQ.has(w) || SEQ_START.has(w))) {
    const parts = [label(ws.slice(a + 1)), label(ws.slice(0, a))];
    if (parts.every(Boolean)) return chain(parts);
  }
  // ASL "EAT FINISH" means "after eating": FINISH ends a step, like a comma
  const parts = splitAt(ws, w => SEQ.has(w) || SEQ_START.has(w) || w === "," || w === "and");
  const g = chain(parts);
  if (g) g.startsNew = SEQ_START.has(ws[0]);
  else if (parts.length === 1) return { kind: "step", step: parts[0], startsNew: SEQ_START.has(ws[0]) };
  return g;
}
function chain(parts) {
  if (parts.length < 2) return null;
  const g = graph("steps", "Steps");
  let prev = null;
  parts.forEach((p, i) => { const id = g.add(`${i + 1}. ${p}`); if (prev) g.link(prev, id); prev = id; });
  return g;
}

function parts(ws) {
  const h = idxOf(ws, HAS);
  if (h < 1) return null;
  const whole = label(ws.slice(0, h));
  const items = splitAt(ws.slice(h + 1), w => w === "," || w === "and");
  if (!whole || !items.length) return null;
  // signed lists have no "and": "PLANT HAVE ROOT STEM LEAF" — one part per word
  const list = items.length === 1 && ws.slice(h + 1).filter(w => !STOP.has(w)).length > 1
    ? ws.slice(h + 1).filter(w => !STOP.has(w) && w !== ",").map(w => label([w])) : items;
  if (list.length < 2) return null;
  const g = graph("parts", `Parts of ${whole.toLowerCase()}`);
  const top = g.add(whole);
  list.forEach(p => g.link(top, g.add(p), "has"));
  return g;
}

function definition(ws) {
  const m = idxOf(ws, MEANS);
  if (m < 1 || ws.includes("what")) return null; // "what does x mean" is a question, not a definition
  const A = label(ws.slice(0, m)), B = label(ws.slice(m + 1));
  if (!A || !B) return null;
  const g = graph("define", `What ${A.toLowerCase()} means`);
  g.link(g.add(A), g.add(B), "means");
  return g;
}

/** Parse one line into a graph, or null when it has no structure worth drawing. */
export function parseSentence(input) {
  const ws = words(input);
  if (ws.length < 2) return null;
  for (const f of [decision, formula, causeEffect, comparison, steps, parts, definition]) {
    const g = f(ws);
    if (g) return g;
  }
  return null;
}

/** Joins lines over time: a steps diagram grows with each THEN/NEXT/FINISH line; an IF line waits for its result. */
export class DiagramSession {
  constructor(joinMs = 20000) { this.joinMs = joinMs; this.g = null; this.t = -Infinity; }
  reset() { this.g = null; this.t = -Infinity; }
  /** Returns the graph to show now (new or grown), or null. */
  add(input, now = Date.now()) {
    const ws = words(input);
    const recent = now - this.t < this.joinMs;
    const p = parseSentence(ws);
    const cur = recent ? this.g : null;
    this.t = now;
    // the result of an open IF: "IF RAIN" / "GAME CANCEL"
    if (cur?.kind === "decision" && cur.open && !p) {
      const text = label(ws);
      if (text) { cur.link("n0", cur.add(text), "Yes"); cur.open = false; return cur; }
    }
    const isStep = p?.kind === "step" || (p?.kind === "steps" && !p.startsNew);
    if (cur?.kind === "steps" && isStep) { // grow the flowchart
      const add = p.kind === "step" ? [p.step] : p.nodes.map(n => n.text.replace(/^\d+\.\s/, ""));
      for (const s of add) {
        const last = cur.nodes[cur.nodes.length - 1].id;
        cur.link(last, cur.add(`${cur.nodes.length + 1}. ${s}`));
      }
      return cur;
    }
    if (p?.kind === "step") { // a lone "FIRST MIX": start a flowchart and wait for the next step
      const g = chain([p.step, "…"]);
      g.nodes.pop(); g.edges.pop();
      this.g = g;
      return null;
    }
    if (p) { this.g = p; return p; }
    return null;
  }
}

const SHAPES = { box: ["[\"", "\"]"], diamond: ["{\"", "\"}"], circle: ["((\"", "\"))"] };
/** Mermaid flowchart source for a graph. Labels already passed through label(); quotes are stripped again here. */
export function toMermaid(g) {
  const safe = s => String(s).replace(/["<>&#`\\[\]{}()|]/g, "");
  // left to right fits the short panel; long step lists and part trees go top to bottom
  const dir = (g.kind === "steps" && g.nodes.length > 4) || g.kind === "parts" ? "TD" : "LR";
  const lines = [`flowchart ${dir}`];
  for (const n of g.nodes) { const [a, b] = SHAPES[n.shape] || SHAPES.box; lines.push(`  ${n.id}${a}${safe(n.text)}${b}`); }
  for (const e of g.edges) lines.push(e.text ? `  ${e.from} -->|"${safe(e.text)}"| ${e.to}` : `  ${e.from} --> ${e.to}`);
  return lines.join("\n");
}
