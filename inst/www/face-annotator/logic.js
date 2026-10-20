/* transforEmotion face viewer: the logic, without the page.
 *
 * Pure functions on the task that view_faces() writes to data.js (see
 * README.md in this folder). app.js draws the page with them; the tests in
 * the package's tests/js folder run them in Node. No DOM, no network.
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

  // A message for a task this page cannot show, or null
  function checkTask(task) {
    if (!task || typeof task !== "object") {
      return "No task data found. index.html needs data.js, which sets window.TE_TASK, in the same folder.";
    }
    if (task.schema !== "transforEmotion.face_task") return "data.js does not hold a face task.";
    if (String(task.schema_version || "").split(".")[0] !== "1") {
      return "This task uses schema version " + task.schema_version + "; this page reads version 1.";
    }
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
    selectedIds, selectionDocument, selectionCode, checkTask
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TEFaceLogic = api;
})(typeof window !== "undefined" ? window : this);
