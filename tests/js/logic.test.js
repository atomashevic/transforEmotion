// Tests of inst/www/face-annotator/logic.js, the face viewer's logic.
// Run from the package root: node --test tests/js/*.test.js
// (tests/testthat/test_face_page_js.R runs them with the installed logic.js,
// named by TE_FACE_LOGIC)
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const L = require(process.env.TE_FACE_LOGIC ||
  path.join(__dirname, "..", "..", "inst", "www", "face-annotator", "logic.js"));

// Two images (three faces, two people) as view_faces() writes them
const face = (id, extra) => Object.assign({ face_id: id, media_id: id.split("#")[0], x: 10, y: 20, w: 50, h: 60,
  confidence: 0.9, track_id: null, person_id: null }, extra);
const imageTask = () => ({
  schema: "transforEmotion.face_task", schema_version: "1.0.0", task_id: "t", mode: "select",
  media: [{ media_id: "m01", media_type: "image", label: "a.jpg", key: "0cc175b9c0f1b6a831c399e269772661" },
    { media_id: "m02", media_type: "image", label: "b.jpg" }],
  faces: [face("m01#0#1", { person_id: "p1" }), face("m01#0#2", { person_id: "p2" }), face("m02#0#1", { person_id: "p1" })]
});
// A video: tracks 1 and 2 are person p1, track 3 is p2
const videoFaces = () => [
  face("m01#0#1", { track_id: 1, person_id: "p1", frame: 0 }), face("m01#0#2", { track_id: 3, person_id: "p2", frame: 0 }),
  face("m01#15#1", { track_id: 1, person_id: "p1", frame: 1 }), face("m01#15#2", { track_id: 3, person_id: "p2", frame: 1 }),
  face("m01#210#1", { track_id: 2, person_id: "p1", frame: 14 })
];

test("face numbers are read from the right of the face_id", () => {
  assert.equal(L.faceNumber("m01#0#2"), 2);
  assert.equal(L.faceNumber("we#ird#15#3"), 3);
});

test("people and tracks sort by number", () => {
  const faces = ["p10", "p2", "p1"].map((p, i) => face(`m01#0#${i + 1}`, { person_id: p }));
  assert.deepEqual(L.personsOf(faces), ["p1", "p2", "p10"]);
  assert.deepEqual(L.tracksOf(videoFaces(), "m01"), [1, 2, 3]);
  assert.deepEqual(L.tracksOf(videoFaces(), "m02"), []);
});

test("colours follow the person, then the track, then the face number", () => {
  const faces = videoFaces();
  assert.equal(L.colourNumber(faces[0], faces), 1);
  assert.equal(L.colourNumber(faces[4], faces), 1); // p1 in another track
  assert.equal(L.colourNumber(faces[1], faces), 2);
  const tracks = faces.map((f) => Object.assign({}, f, { person_id: null }));
  assert.equal(L.colourNumber(tracks[4], tracks), 2);
  assert.equal(L.colourNumber(face("m01#0#4"), [face("m01#0#4")]), 4);
});

test("labels name the person and the track", () => {
  const faces = videoFaces();
  assert.equal(L.boxLabel(faces[4]), "P1 · T2");
  assert.equal(L.boxLabel(imageTask().faces[1]), "P2");
  assert.equal(L.boxLabel(face("m01#0#3")), "3");
  assert.match(L.describe(faces[4]), /^person P1, track T2, frame 14, confidence 0\.90, 50 × 60 px$/);
  assert.equal(L.describe(face("m01#0#3", { confidence: null })), "face 3, 50 × 60 px");
});

test("a group is selected, then deselected, as a whole", () => {
  const faces = videoFaces(), chosen = new Set(["m01#0#1"]);
  const track1 = L.clickGroup(faces[0], faces);
  assert.deepEqual(track1.map((f) => f.face_id), ["m01#0#1", "m01#15#1"]);
  assert.equal(L.groupState(chosen, track1), "mixed");
  assert.equal(L.toggleGroup(chosen, track1), true);
  assert.equal(L.groupState(chosen, track1), "true");
  assert.equal(L.toggleGroup(chosen, track1), false);
  assert.equal(chosen.size, 0);
  // A face in a still image is its own group
  const still = imageTask().faces[0];
  assert.deepEqual(L.clickGroup(still, imageTask().faces), [still]);
});

test("the exported selection lists faces, tracks and whole people", () => {
  const task = imageTask();
  const doc = L.selectionDocument(task, new Set(["m02#0#1", "m01#0#1", "x#0#9"]), "2026-10-11T10:00:00Z");
  assert.equal(doc.schema, "transforEmotion.face_selection");
  assert.deepEqual(doc.selected_face_ids, ["m01#0#1", "m02#0#1"]); // table order; unknown ids dropped
  assert.deepEqual(doc.selected_person_ids, ["p1"]);
  assert.deepEqual(doc.media, [{ media_id: "m01", label: "a.jpg", key: "0cc175b9c0f1b6a831c399e269772661" },
    { media_id: "m02", label: "b.jpg", key: null }]);

  const video = Object.assign(imageTask(), { faces: videoFaces() });
  const t = L.selectionDocument(video, new Set(["m01#0#1", "m01#15#1", "m01#0#2"]), "now");
  assert.deepEqual(t.selected_tracks, [{ media_id: "m01", track_id: 1 }, { media_id: "m01", track_id: 3 }]);
  assert.deepEqual(t.selected_person_ids, []); // p1's track 2 is not selected, p2's face in frame 15 neither
});

test("tasks the page cannot show are explained", () => {
  assert.equal(L.checkTask(imageTask()), null);
  assert.match(L.checkTask(undefined), /No task data found/);
  assert.match(L.checkTask(Object.assign(imageTask(), { schema_version: "2.0.0" })), /version 2\.0\.0/);
  assert.match(L.checkTask(Object.assign(imageTask(), { mode: "annotate" })), /Unknown mode "annotate"/);
  const stray = imageTask();
  stray.faces.push(face("m09#0#1"));
  assert.match(L.checkTask(stray), /m09#0#1 belongs to no image/);
});

test("a notebook page gives the selection as a line of R code", () => {
  const task = imageTask();
  const chosen = new Set(["m01#0#2"]);
  const code = L.selectionCode(task, chosen, "2026-10-20T10:00:00Z", "my_faces");
  const m = code.match(/^chosen <- read_face_selection\("te-selection:([A-Za-z0-9+/=]+)", my_faces\)$/);
  assert.ok(m, code);
  const doc = JSON.parse(Buffer.from(m[1], "base64").toString("utf8"));
  assert.deepEqual(doc, L.selectionDocument(task, chosen, "2026-10-20T10:00:00Z"));
  // Non-ASCII file names survive the round trip
  task.media[0].label = "čaša ☕.jpg";
  const again = L.selectionCode(task, chosen, "t", null);
  assert.match(again, /, faces\)$/);
  const label = JSON.parse(Buffer.from(again.match(/te-selection:([^"]+)/)[1], "base64").toString("utf8")).media[0].label;
  assert.equal(label, "čaša ☕.jpg");
});
