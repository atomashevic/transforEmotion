/* transforEmotion annotation page: the logic, without the page.
 *
 * Pure functions on the task that annotate_emotions() writes to data.js (see
 * README.md in this folder). app.js draws the page with them; the tests in
 * the package's tests/js folder run them in Node. No DOM, no network.
 */
(function (root) {
  "use strict";

  const choiceQuestions = (task) => task.questions.filter((q) => Array.isArray(q.options) && q.options.length);
  const answered = (values, q) => values != null && values[q.id] != null && values[q.id] !== "";

  // The questions to show for one item, given its answers so far. With a
  // blind task, the label, prediction and scores, and the questions asked
  // about them (reveal: true), appear once the questions before them are
  // answered, so the coder's own label is given first.
  function stage(task, values) {
    const before = task.questions.filter((q) => !q.reveal);
    const after = task.questions.filter((q) => q.reveal);
    const ready = before.filter((q) => Array.isArray(q.options) && q.options.length)
      .every((q) => answered(values, q));
    const revealed = !task.blind || ready;
    return { before, after: revealed ? after : [], revealed };
  }

  // An item is coded when every question with options is answered
  function isCoded(task, values) {
    return choiceQuestions(task).every((q) => answered(values, q));
  }

  function progress(task, answers) {
    const coded = task.items.filter((it) => isCoded(task, (answers[it.id] || {}).values)).length;
    return { coded, total: task.items.length };
  }

  // The first choice question still open among those shown, for the number keys
  function currentQuestion(task, values) {
    const s = stage(task, values);
    return s.before.concat(s.after).find((q) => Array.isArray(q.options) && q.options.length && !answered(values, q)) || null;
  }

  // Scores in the order of the classes, the top one marked
  function scoreList(task, item) {
    if (!item.scores) return [];
    const rows = task.score_classes.filter((c) => item.scores[c] != null).map((c) => ({ cls: c, p: Number(item.scores[c]) }));
    const top = rows.reduce((a, b) => (b.p > a.p ? b : a), rows[0] || null);
    return rows.map((r) => Object.assign(r, { top: top != null && r.cls === top.cls }));
  }

  // The file read_annotations() reads: one entry per item with any answer
  function answersDocument(task, answers, coder, exportedAt) {
    const ids = task.questions.map((q) => q.id);
    const rows = task.items.filter((it) => answers[it.id] && Object.keys(answers[it.id].values || {}).length)
      .map((it) => {
        const a = answers[it.id];
        const row = { id: it.id };
        ids.forEach((q) => { row[q] = a.values[q] == null || a.values[q] === "" ? null : a.values[q]; });
        row.seconds = Math.round((a.seconds || 0) * 10) / 10;
        row.answered_at = a.answered_at || null;
        return row;
      });
    return {
      schema: "transforEmotion.annotations",
      schema_version: task.schema_version,
      task_id: task.task_id,
      coder: coder || "",
      exported_at: exportedAt,
      question_ids: ids,
      answers: rows
    };
  }

  // Base64 of the UTF-8 bytes of text, in the browser and in Node
  function base64(text) {
    let bin = "";
    new TextEncoder().encode(text).forEach((b) => { bin += String.fromCharCode(b); });
    return btoa(bin);
  }

  // The line of R code a page in a notebook gives instead of a file
  function answersCode(task, answers, coder, exportedAt, itemsName) {
    const doc = answersDocument(task, answers, coder, exportedAt);
    return 'answers <- read_annotations("te-annotations:' + base64(JSON.stringify(doc)) + '", ' +
      (itemsName || "items") + ")";
  }

  // A message for a task this page cannot show, or null
  function checkTask(task) {
    if (!task || typeof task !== "object") {
      return "No task data found. index.html needs data.js, which sets window.TE_TASK, in the same folder.";
    }
    if (task.schema !== "transforEmotion.annotation_task") return "data.js does not hold an annotation task.";
    if (String(task.schema_version || "").split(".")[0] !== "1") {
      return "This task uses schema version " + task.schema_version + "; this page reads version 1.";
    }
    if (!Array.isArray(task.items) || !task.items.length) return "The task has no items.";
    if (!Array.isArray(task.questions) || !task.questions.length) return "The task has no questions.";
    const ids = new Set(task.items.map((it) => it.id));
    if (ids.size !== task.items.length) return "Two items have the same id.";
    return null;
  }

  const api = { stage, isCoded, progress, currentQuestion, scoreList, answersDocument, answersCode, base64, checkTask };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TEAnnotateLogic = api;
})(typeof window !== "undefined" ? window : this);
