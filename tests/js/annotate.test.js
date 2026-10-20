// Tests of inst/www/emotion-annotator/logic.js, the annotation page's logic.
// Run from the package root: node --test tests/js/*.test.js
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const L = require(process.env.TE_ANNOTATE_LOGIC ||
  path.join(__dirname, "..", "..", "inst", "www", "emotion-annotator", "logic.js"));

const opts = (...v) => v.map((x) => ({ value: x, text: x }));
const task = (blind = true) => ({
  schema: "transforEmotion.annotation_task", schema_version: "1.0.0", task_id: "annotate-1", blind,
  classes: ["angry", "sad"], score_classes: ["angry", "sad"],
  has: { label: true, prediction: true, scores: true },
  questions: [
    { id: "emotion", prompt: "Which emotion?", options: opts("angry", "sad", "other"), reveal: false },
    { id: "why", prompt: "Why?", options: opts("model", "label"), reveal: true },
    { id: "note", prompt: "Note", options: null, reveal: true }
  ],
  items: [{ id: "a", label: "angry", prediction: "sad", scores: { angry: 0.3, sad: 0.7 } },
    { id: "b", label: "sad", prediction: "sad", scores: null }]
});

test("a blind task shows the label and the questions about it only after the coder's answer", () => {
  const t = task();
  let s = L.stage(t, {});
  assert.equal(s.revealed, false);
  assert.deepEqual(s.before.map((q) => q.id), ["emotion"]);
  assert.deepEqual(s.after, []);
  s = L.stage(t, { emotion: "sad" });
  assert.equal(s.revealed, true);
  assert.deepEqual(s.after.map((q) => q.id), ["why", "note"]);
  // Not blind: everything at once
  assert.equal(L.stage(task(false), {}).revealed, true);
});

test("an item is coded when every question with options is answered", () => {
  const t = task();
  assert.equal(L.isCoded(t, { emotion: "sad" }), false);
  assert.equal(L.isCoded(t, { emotion: "sad", why: "model" }), true);
  assert.equal(L.isCoded(t, { emotion: "", why: "model" }), false);
  assert.deepEqual(L.progress(t, { a: { values: { emotion: "sad", why: "model" } } }), { coded: 1, total: 2 });
  assert.equal(L.currentQuestion(t, {}).id, "emotion");
  assert.equal(L.currentQuestion(t, { emotion: "sad" }).id, "why");
  assert.equal(L.currentQuestion(t, { emotion: "sad", why: "label" }), null);
});

test("scores come in class order with the top one marked", () => {
  const t = task();
  assert.deepEqual(L.scoreList(t, t.items[0]), [{ cls: "angry", p: 0.3, top: false }, { cls: "sad", p: 0.7, top: true }]);
  assert.deepEqual(L.scoreList(t, t.items[1]), []);
});

test("the answers document and the notebook code hold the answers given", () => {
  const t = task();
  const answers = { a: { values: { emotion: "angry", why: "model", note: "čaša" }, seconds: 12.34, answered_at: "x" },
    b: { values: {}, seconds: 3 } };
  const doc = L.answersDocument(t, answers, "AT", "2026-10-20T10:00:00Z");
  assert.equal(doc.schema, "transforEmotion.annotations");
  assert.deepEqual(doc.question_ids, ["emotion", "why", "note"]);
  assert.deepEqual(doc.answers, [{ id: "a", emotion: "angry", why: "model", note: "čaša", seconds: 12.3, answered_at: "x" }]);
  const code = L.answersCode(t, answers, "AT", "2026-10-20T10:00:00Z", "errors");
  const m = code.match(/^answers <- read_annotations\("te-annotations:([A-Za-z0-9+/=]+)", errors\)$/);
  assert.ok(m, code);
  assert.deepEqual(JSON.parse(Buffer.from(m[1], "base64").toString("utf8")), doc);
});

test("tasks the page cannot show are explained", () => {
  assert.match(L.checkTask(null), /No task data/);
  assert.match(L.checkTask({ schema: "x" }), /not hold an annotation task/);
  const t = task();
  t.items.push({ id: "a" });
  assert.match(L.checkTask(t), /same id/);
  assert.equal(L.checkTask(task()), null);
});
