/* transforEmotion face viewer: the logic, without the page.
 *
 * Pure functions on the task that view_faces() or annotation_task() writes
 * to data.js (see README.md in this folder). app.js draws the page with
 * them; the tests in the package's tests/js folder run them in Node. No DOM,
 * no network.
 */
(function (root) {
  "use strict";

  // face_id is "<media_id>#<frame_index>#<face>"; the face number is read from
  // the right, so a media_id that contains "#" still works
  function faceNumber(faceId) {
    return Number(String(faceId).split("#").pop());
  }

  // "p2" before "p10"
  function byNumber(a, b) {
    return String(a).localeCompare(String(b), undefined, { numeric: true });
  }

  function unique(values) {
    return Array.from(new Set(values));
  }

  // Person ids span every image and video of the task
  function personsOf(faces) {
    return unique(faces.filter((f) => f.person_id != null).map((f) => f.person_id)).sort(byNumber);
  }

  // Track ids are numbered within one video
  function tracksOf(faces, mediaId) {
    return unique(faces.filter((f) => f.media_id === mediaId && f.track_id != null)
      .map((f) => f.track_id)).sort(byNumber);
  }

  function trackLabel(t) {
    return "T" + t;
  }

  function personLabel(p) {
    return String(p).toUpperCase();
  }

  // Colour number (1, 2, ...) of a face: by person, else by track, else by
  // face number, so the same person has the same colour everywhere
  function colourNumber(face, faces) {
    if (face.person_id != null) return personsOf(faces).indexOf(face.person_id) + 1;
    if (face.track_id != null) return tracksOf(faces, face.media_id).indexOf(face.track_id) + 1;
    return faceNumber(face.face_id);
  }

  // Short label drawn above a box: "P1 · T2" in a video, "P1" or "3" in an image
  function boxLabel(face) {
    const parts = [];
    if (face.person_id != null) parts.push(personLabel(face.person_id));
    if (face.track_id != null) parts.push(trackLabel(face.track_id));
    else if (face.person_id == null) parts.push(String(faceNumber(face.face_id)));
    return parts.join(" · ");
  }

  // Longer description, for screen readers and tooltips
  function describe(face) {
    const who = [];
    if (face.person_id != null) who.push("person " + personLabel(face.person_id));
    who.push(face.track_id != null
      ? "track " + trackLabel(face.track_id) + ", frame " + face.frame
      : "face " + faceNumber(face.face_id));
    const score = face.confidence == null ? "" : ", confidence " + Number(face.confidence).toFixed(2);
    return who.join(", ") + score + ", " + Math.round(face.w) + " × " + Math.round(face.h) + " px";
  }

  /* Selection: a Set of face_ids */

  // Selecting a group (a track, a person) selects all its faces, unless they
  // are all selected already, in which case it deselects them. Returns whether
  // the group is now selected.
  function toggleGroup(chosen, group) {
    const on = !group.every((f) => chosen.has(f.face_id));
    group.forEach((f) => (on ? chosen.add(f.face_id) : chosen.delete(f.face_id)));
    return on;
  }

  // "true", "false" or "mixed" (some faces of the group selected)
  function groupState(chosen, group) {
    const k = group.filter((f) => chosen.has(f.face_id)).length;
    return k === 0 ? "false" : k === group.length ? "true" : "mixed";
  }

  // Clicking a face in a video selects its whole track
  function clickGroup(face, faces) {
    if (face.track_id == null) return [face];
    return faces.filter((f) => f.media_id === face.media_id && f.track_id === face.track_id);
  }

  function selectedIds(faces, chosen) {
    return faces.filter((f) => chosen.has(f.face_id)).map((f) => f.face_id);
  }

  // The file read_face_selection() reads
  function selectionDocument(task, chosen, exportedAt) {
    const faces = task.faces || [];
    const tracks = new Map();
    faces.filter((f) => f.track_id != null && chosen.has(f.face_id))
      .forEach((f) => tracks.set(f.media_id + "#" + f.track_id, { media_id: f.media_id, track_id: f.track_id }));
    const persons = personsOf(faces).filter((p) =>
      faces.filter((f) => f.person_id === p).every((f) => chosen.has(f.face_id)));
    return {
      schema: "transforEmotion.face_selection",
      schema_version: task.schema_version,
      task_id: task.task_id,
      exported_at: exportedAt,
      // lets R check that the selection is read with the same images or video
      // (key: the file's MD5, when R could read it)
      media: (task.media || []).map((m) => ({ media_id: m.media_id, label: m.label, key: m.key == null ? null : m.key })),
      selected_face_ids: selectedIds(faces, chosen),
      selected_tracks: Array.from(tracks.values()),
      selected_person_ids: persons
    };
  }

  // Base64 of the UTF-8 bytes of text, in the browser and in Node
  function base64(text) {
    let bin = "";
    new TextEncoder().encode(text).forEach((b) => { bin += String.fromCharCode(b); });
    return btoa(bin);
  }

  // The line of R code a page in a notebook gives instead of a file: the
  // selection document, base64-encoded so it needs no quoting.
  // read_face_selection() reads the "te-selection:" code.
  function selectionCode(task, chosen, exportedAt, facesName) {
    const doc = selectionDocument(task, chosen, exportedAt);
    return 'chosen <- read_face_selection("te-selection:' + base64(JSON.stringify(doc)) + '", ' +
      (facesName || "faces") + ")";
  }

  /* Annotation: the task annotation_task() writes, and the coder's answers */

  // "not_face" -> "Not face"
  function nice(s) {
    s = String(s).replace(/_/g, " ");
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  // The answers a question offers: [{value, label}]
  function optionsOf(q, task) {
    if (q.type === "likert") {
      const out = [];
      for (let v = q.min; v <= q.max; v++) out.push({ value: v, label: String(v) });
      return out;
    }
    const raw = q.options_from ? (task && task[q.options_from]) || [] : q.options || [];
    return raw.map((o) => ({ value: o, label: (q.labels && q.labels[o]) || nice(o) }));
  }

  // show_if {question: value, or [values]}: shown only after one of those
  // answers. reveal: shown only once the item's label and model answer are
  // (revealed false hides it; undefined shows it)
  function isShown(q, responses, revealed) {
    if (q.reveal && revealed === false) return false;
    return !q.show_if || Object.entries(q.show_if).every(([k, v]) =>
      (Array.isArray(v) ? v.includes(responses[k]) : responses[k] === v));
  }

  // Whether an item's label, model answer and scores are shown: at once
  // unless the task is blind, else once every shown, required question
  // without reveal is answered
  function isRevealed(task, responses) {
    if (!task.settings || task.settings.blind === false) return true;
    return (task.questions || []).filter((q) => !q.reveal && q.required !== false && isShown(q, responses))
      .every((q) => responses[q.id] != null);
  }

  // The question that keys 1-9 answer: the first shown one without an answer
  function activeQuestion(questions, responses, revealed) {
    return questions.find((q) => isShown(q, responses, revealed) && responses[q.id] == null);
  }

  // A shown, required question without an answer (blocks Next), or undefined
  function missingQuestion(questions, responses, revealed) {
    return questions.find((q) => isShown(q, responses, revealed) && q.required !== false && responses[q.id] == null);
  }

  // Every question, with null for questions not shown, not answered, or skipped
  function cleanResponses(questions, responses, answered, revealed) {
    const out = {};
    for (const q of questions) {
      const v = responses[q.id];
      out[q.id] = answered && isShown(q, responses, revealed) && v != null ? v : null;
    }
    return out;
  }

  // An item's model scores in the order of the task's classes, the top one marked
  function scoreList(task, item) {
    if (!item || !item.scores) return [];
    const rows = (task.score_classes || []).filter((c) => item.scores[c] != null)
      .map((c) => ({ cls: c, p: Number(item.scores[c]) }));
    const top = rows.reduce((a, b) => (b.p > a.p ? b : a), rows[0] || null);
    return rows.map((r) => Object.assign(r, { top: top != null && r.cls === top.cls }));
  }

  // The items a coder codes, in the order annotation_task() chose for them
  function orderFor(task, coderId) {
    const known = new Set((task.items || []).map((it) => it.item_id));
    const a = (task.assignments || []).find((x) => x.coder_id === coderId);
    return a ? [].concat(a.items).filter((id) => known.has(id)) : [];
  }

  // Records are append-only: the last one for an item is the coder's answer
  function latestRecord(records, itemId) {
    for (let i = records.length - 1; i >= 0; i--) if (records[i].item_id === itemId) return records[i];
    return undefined;
  }

  // Where a coder continues: the first item without a record
  function firstOpen(order, records) {
    const i = order.findIndex((id) => !latestRecord(records, id));
    return i < 0 ? order.length : i;
  }

  function progressOf(order, records) {
    let answered = 0, skipped = 0;
    for (const id of order) {
      const r = latestRecord(records, id);
      if (r && r.status === "answered") answered++;
      else if (r) skipped++;
    }
    return { total: order.length, done: answered + skipped, answered, skipped };
  }

  // One record: an answered or skipped item. rt_ms is the time the item was shown.
  function makeRecord(task, coderId, sessionId, records, item, status, responses, shownAt, submittedAt, rtMs) {
    return {
      task_id: task.task_id, item_id: item.item_id, coder_id: coderId, session_id: sessionId,
      seq: records.length + 1,
      revision: records.filter((r) => r.item_id === item.item_id).length + 1,
      status,
      responses: cleanResponses(task.questions || [], responses, status === "answered", isRevealed(task, responses)),
      shown_at: shownAt, submitted_at: submittedAt, rt_ms: Math.max(0, Math.round(rtMs))
    };
  }

  // The file read_annotations() reads
  function annotationsDocument(task, coderId, order, records, exportedAt) {
    return {
      schema: "transforEmotion.annotations",
      schema_version: task.schema_version,
      task_id: task.task_id,
      coder_id: coderId,
      exported_at: exportedAt,
      n_items: order.length,
      records: records.slice()
    };
  }

  // The line of R code a coding page in a notebook gives instead of a file
  function annotationsCode(task, coderId, order, records, exportedAt, taskCode) {
    const doc = annotationsDocument(task, coderId, order, records, exportedAt);
    return 'answers <- read_annotations(' + (taskCode || "task") + ', "te-annotations:' +
      base64(JSON.stringify(doc)) + '")';
  }

  // A message for an annotation task this page cannot run, or null
  function checkAnnotationTask(task) {
    if (task.mode !== "annotate") return "Unknown mode \"" + task.mode + "\" for an annotation task. Expected annotate.";
    if (!Array.isArray(task.items) || !task.items.length) return "The task has no items to code.";
    if (!Array.isArray(task.questions) || !task.questions.length) return "The task has no questions.";
    if (!Array.isArray(task.assignments) || !task.assignments.length) return "The task names no coders.";
    for (const q of task.questions) {
      if (q.type === "likert") {
        if (!Number.isInteger(q.min) || !Number.isInteger(q.max) || q.min >= q.max) {
          return "Question " + q.id + " needs whole numbers min < max.";
        }
      } else if (q.type === "single") {
        if (!optionsOf(q, task).length) return "Question " + q.id + " has no options.";
      } else {
        return "Question " + q.id + " has an unknown type \"" + q.type + "\".";
      }
    }
    const known = new Set(task.items.map((it) => it.item_id));
    for (const a of task.assignments) {
      const stray = [].concat(a.items).find((id) => !known.has(id));
      if (stray) return "Coder " + a.coder_id + " has an item that is not in the task: " + stray + ".";
    }
    return null;
  }

  // A message for a task this page cannot show, or null
  function checkTask(task) {
    if (!task || typeof task !== "object") {
      return "No task data found. index.html needs data.js, which sets window.TE_TASK, in the same folder.";
    }
    const annotation = task.schema === "transforEmotion.annotation_task";
    if (task.schema !== "transforEmotion.face_task" && !annotation) return "data.js does not hold a face task.";
    if (String(task.schema_version || "").split(".")[0] !== "1") {
      return "This task uses schema version " + task.schema_version + "; this page reads version 1.";
    }
    if (annotation) return checkAnnotationTask(task);
    if (task.mode !== "view" && task.mode !== "select") {
      return "Unknown mode \"" + task.mode + "\". Expected view or select.";
    }
    if (!Array.isArray(task.media) || !Array.isArray(task.faces)) return "The task has no media or faces list.";
    const media = new Set(task.media.map((m) => m.media_id));
    const stray = task.faces.find((f) => !media.has(f.media_id));
    if (stray) return "Face " + stray.face_id + " belongs to no image or video of the task.";
    return null;
  }

  const api = {
    faceNumber, byNumber, personsOf, tracksOf, trackLabel, personLabel,
    colourNumber, boxLabel, describe, toggleGroup, groupState, clickGroup,
    selectedIds, selectionDocument, selectionCode, checkTask,
    nice, optionsOf, isShown, isRevealed, activeQuestion, missingQuestion, cleanResponses, scoreList, orderFor,
    latestRecord, firstOpen, progressOf, makeRecord, annotationsDocument, annotationsCode
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TEFaceLogic = api;
})(typeof window !== "undefined" ? window : this);
