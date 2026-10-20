// Tests of the annotate-mode functions in inst/www/face-annotator/logic.js.
// Run from the package root: node --test tests/js/*.test.js
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const L = require(process.env.TE_FACE_LOGIC ||
  path.join(__dirname, "..", "..", "inst", "www", "face-annotator", "logic.js"));

// A task as annotation_task() writes it: three items, two coders
const task = () => ({
  schema: "transforEmotion.annotation_task", schema_version: "1.0.0", task_id: "t1", mode: "annotate",
  classes: ["happy", "sad"],
  questions: [
    { id: "face_check", type: "single", options: ["ok", "not_face"], labels: { ok: "Yes, a face" }, required: true },
    { id: "emotion", type: "single", options_from: "classes", show_if: { face_check: "ok" }, required: true },
    { id: "intensity", type: "likert", min: 1, max: 3, show_if: { face_check: "ok" }, required: false }
  ],
  items: ["i1", "i2", "i3"].map((id) => ({ item_id: id, media: "media/p001.jpg", width: 400, height: 300,
    box: { x: 1, y: 2, w: 3, h: 4 } })),
  assignments: [{ coder_id: "C01", items: ["i3", "i1", "i2"] }, { coder_id: "C02", items: ["i2"] }]
});

test("options come from the question, the task's classes, or the Likert range", () => {
  const [check, emotion, intensity] = task().questions;
  assert.deepEqual(L.optionsOf(check, task()), [{ value: "ok", label: "Yes, a face" }, { value: "not_face", label: "Not face" }]);
  assert.deepEqual(L.optionsOf(emotion, task()).map((o) => o.value), ["happy", "sad"]);
  assert.deepEqual(L.optionsOf(intensity, task()).map((o) => o.value), [1, 2, 3]);
});

test("questions after the face check are asked only for a face", () => {
  const qs = task().questions;
  assert.equal(L.isShown(qs[1], {}), false);
  assert.equal(L.isShown(qs[1], { face_check: "ok" }), true);
  assert.equal(L.isShown({ show_if: { face_check: ["ok", "unclear"] } }, { face_check: "unclear" }), true);
  assert.equal(L.activeQuestion(qs, {}).id, "face_check");
  assert.equal(L.activeQuestion(qs, { face_check: "ok" }).id, "emotion");
  assert.equal(L.missingQuestion(qs, { face_check: "ok" }).id, "emotion");
  // intensity is optional, and "not a face" needs nothing more
  assert.equal(L.missingQuestion(qs, { face_check: "ok", emotion: "sad" }), undefined);
  assert.equal(L.missingQuestion(qs, { face_check: "not_face" }), undefined);
  // answers to questions no longer shown are dropped
  assert.deepEqual(L.cleanResponses(qs, { face_check: "not_face", emotion: "sad" }, true),
    { face_check: "not_face", emotion: null, intensity: null });
  assert.deepEqual(L.cleanResponses(qs, { face_check: "ok" }, false), { face_check: null, emotion: null, intensity: null });
});

test("each coder gets their own list, and continues at the first open item", () => {
  assert.deepEqual(L.orderFor(task(), "C01"), ["i3", "i1", "i2"]);
  assert.deepEqual(L.orderFor(task(), "C02"), ["i2"]);
  assert.deepEqual(L.orderFor(task(), "nobody"), []);
  const t = task(), order = L.orderFor(t, "C01"), records = [];
  const add = (item, status, responses) => records.push(L.makeRecord(t, "C01", "s1", records, t.items.find((x) => x.item_id === item),
    status, responses, "2026-10-11T10:00:00.000Z", "2026-10-11T10:00:02.000Z", 2000.4));
  assert.equal(L.firstOpen(order, records), 0);
  add("i3", "answered", { face_check: "ok", emotion: "happy" });
  add("i1", "skipped", { face_check: "ok" });
  assert.equal(L.firstOpen(order, records), 2);
  add("i3", "answered", { face_check: "ok", emotion: "sad", intensity: 2 });
  assert.deepEqual(records.map((r) => [r.seq, r.revision]), [[1, 1], [2, 1], [3, 2]]);
  assert.equal(records[1].responses.face_check, null); // a skip records no answers
  assert.equal(records[0].rt_ms, 2000);
  assert.equal(L.latestRecord(records, "i3").responses.emotion, "sad");
  assert.deepEqual(L.progressOf(order, records), { total: 3, done: 2, answered: 1, skipped: 1 });
  const doc = L.annotationsDocument(t, "C01", order, records, "2026-10-11T10:01:00.000Z");
  assert.equal(doc.schema, "transforEmotion.annotations");
  assert.equal(doc.records.length, 3);
  assert.notEqual(doc.records, records); // a copy
});

test("checkTask accepts annotation tasks and explains broken ones", () => {
  assert.equal(L.checkTask(task()), null);
  const noItems = Object.assign(task(), { items: [] });
  assert.match(L.checkTask(noItems), /no items/);
  const badLikert = task();
  badLikert.questions[2].max = 1;
  assert.match(L.checkTask(badLikert), /min < max/);
  const stray = task();
  stray.assignments[1].items = ["i9"];
  assert.match(L.checkTask(stray), /C02 has an item that is not in the task: i9/);
  const noOptions = task();
  noOptions.classes = [];
  assert.match(L.checkTask(noOptions), /emotion has no options/);
  assert.match(L.checkTask(Object.assign(task(), { mode: "view" })), /Expected annotate/);
});

test("a notebook coding page gives the answers as a line of R code", () => {
  const t = task();
  const records = [L.makeRecord(t, "C01", "s", [], t.items[0], "answered", { face_check: "ok", emotion: "sad" },
    "a", "b", 10)];
  const code = L.annotationsCode(t, "C01", ["i3", "i1", "i2"], records, "2026-10-20T10:00:00Z", '"round-1"');
  const m = code.match(/^answers <- read_annotations\("round-1", "te-annotations:([A-Za-z0-9+/=]+)"\)$/);
  assert.ok(m, code);
  assert.deepEqual(JSON.parse(Buffer.from(m[1], "base64").toString("utf8")),
    L.annotationsDocument(t, "C01", ["i3", "i1", "i2"], records, "2026-10-20T10:00:00Z"));
  assert.match(L.annotationsCode(t, "C01", [], [], "x", null), /^answers <- read_annotations\(task, /);
});

test("questions with reveal wait for the coder's own answer in a blind item task", () => {
  const t = {
    schema: "transforEmotion.annotation_task", schema_version: "1.0.0", task_id: "r1", mode: "annotate", unit: "item",
    classes: ["angry", "sad"], score_classes: ["angry", "sad"], settings: { blind: true },
    questions: [
      { id: "emotion", type: "single", options_from: "classes", required: true },
      { id: "why", type: "single", options: ["model", "label"], reveal: true, required: true }
    ],
    items: [{ item_id: "i1", kind: "text", text: "x", reference: "sad", prediction: "angry", scores: { angry: 0.6, sad: 0.4 } }],
    assignments: [{ coder_id: "C01", items: ["i1"] }]
  };
  const [emotion, why] = t.questions;
  assert.equal(L.isRevealed(t, {}), false);
  assert.equal(L.isShown(why, {}, false), false);
  assert.equal(L.isShown(why, {}, undefined), true);
  assert.equal(L.missingQuestion(t.questions, {}, false), emotion);
  assert.equal(L.isRevealed(t, { emotion: "sad" }), true);
  assert.equal(L.activeQuestion(t.questions, { emotion: "sad" }, true), why);
  assert.equal(L.missingQuestion(t.questions, { emotion: "sad", why: "label" }, true), undefined);
  // A why given before the reveal is not kept; after it, it is
  const rec = (responses) => L.makeRecord(t, "C01", "s", [], t.items[0], "answered", responses, "a", "b", 1).responses;
  assert.deepEqual(rec({ why: "model" }), { emotion: null, why: null });
  assert.deepEqual(rec({ emotion: "sad", why: "model" }), { emotion: "sad", why: "model" });
  // Not blind: shown at once
  assert.equal(L.isRevealed(Object.assign({}, t, { settings: { blind: false } }), {}), true);
  assert.deepEqual(L.scoreList(t, t.items[0]), [{ cls: "angry", p: 0.6, top: true }, { cls: "sad", p: 0.4, top: false }]);
  assert.equal(L.checkTask(t), null);
});
