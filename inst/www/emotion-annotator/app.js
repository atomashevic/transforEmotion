/* transforEmotion annotation page: the page.
 *
 * Shows one item at a time (frames or an image with face boxes, a clip, a
 * text), asks the task's questions and keeps the answers. When R serves the
 * page (window.TE_API) every answer goes to R as the coder goes; in a
 * notebook (window.TE_NOTEBOOK) the answers leave as a line of R code;
 * opened from disk, as a downloaded file. The logic lives in logic.js.
 */
(function () {
  "use strict";
  const L = window.TEAnnotateLogic;
  const SVGNS = "http://www.w3.org/2000/svg";
  const $ = (sel, root) => (root || document).querySelector(sel);

  function el(tag, attrs, ...kids) {
    const svg = tag.startsWith("svg:");
    const node = svg ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (k === "text") node.textContent = v;
      else if (k === "style") for (const [p, val] of Object.entries(v)) node.style.setProperty(p, String(val));
      else node.setAttribute(k, v === true ? "" : String(v));
    }
    for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) node.append(kid);
    return node;
  }
  const put = (node, ...kids) => node.replaceChildren(...kids.flat(Infinity).filter((k) => k != null && k !== false && k !== ""));
  const fmt = (v, d) => (v == null ? "NA" : Number(v).toFixed(d));

  function say(msg) {
    $("#status").textContent = msg;
    clearTimeout(say.timer);
    say.timer = setTimeout(() => { $("#status").textContent = ""; }, 6000);
  }
  function fatal(msg) {
    put($("#main"), el("div", { class: "card narrow", role: "alert" }, el("h2", { text: "Cannot open this task" }), el("p", { text: msg })));
  }

  const api = window.TE_API || null;
  const notebook = window.TE_NOTEBOOK || null;
  async function postJson(url, data) {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data), cache: "no-store" });
    let out = {};
    try { out = await res.json(); } catch (e) { /* not JSON */ }
    if (!res.ok) throw new Error(out.error || `HTTP ${res.status}`);
    return out;
  }
  function download(filename, text) {
    try {
      const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
      const a = el("a", { href: url, download: filename, hidden: true });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (e) { /* blocked: the text below still works */ }
  }
  function sheet(heading, note, text, copyLabel) {
    $("#export-h").textContent = heading;
    put($("#export-note"), note);
    $("#export-text").value = text;
    $("#export-copy").textContent = copyLabel;
    $("#export").hidden = false;
    $("#export-copy").focus();
  }
  $("#export-close").addEventListener("click", () => { $("#export").hidden = true; });
  $("#export-copy").addEventListener("click", async () => {
    const ta = $("#export-text");
    try {
      await navigator.clipboard.writeText(ta.value);
      say("Copied to the clipboard.");
    } catch (e) {
      ta.focus();
      ta.select();
      say("Copying is blocked here. The text is selected: press Ctrl+C (Cmd+C on a Mac).");
    }
  });
  $("#viewer-close").addEventListener("click", () => $("#viewer").close());

  // An image or frame with its face boxes; boxes are in original pixels, the
  // overlay's viewBox is the original size
  function picture(fr, maxh, name) {
    const box = el("span", { class: "stage", style: { "--ar": (fr.w / fr.h).toFixed(5), "--maxh": maxh } });
    const img = el("img", { src: fr.src, alt: "", width: fr.w, height: fr.h, draggable: "false" });
    img.addEventListener("error", () => box.classList.add("broken"));
    const svg = el("svg:svg", { viewBox: `0 0 ${fr.w} ${fr.h}`, preserveAspectRatio: "none", role: "img", "aria-label": name });
    (fr.faces || []).forEach((f) => svg.append(el("svg:rect", { class: "face" + (f.scored ? " scored" : ""), x: f.x, y: f.y, width: f.w, height: f.h })));
    box.append(img, svg);
    return box;
  }

  function annotator(task) {
    const STORE = "te-annotate-" + task.task_id;
    let answers = {};
    try { answers = JSON.parse(localStorage.getItem(STORE) || "{}") || {}; } catch (e) { answers = {}; }
    let coder = task.coder || "";
    try { coder = coder || localStorage.getItem(STORE + "-coder") || ""; } catch (e) { /* no storage */ }
    let current = 0, shownAt = Date.now(), finishing = false, done = false;
    const items = task.items;
    const entry = (it) => (answers[it.id] = answers[it.id] || { values: {}, seconds: 0, answered_at: null });

    function persist() {
      try { localStorage.setItem(STORE, JSON.stringify(answers)); localStorage.setItem(STORE + "-coder", coder); } catch (e) { /* no storage */ }
      if (api && api.answers && !done) {
        clearTimeout(persist.timer);
        persist.timer = setTimeout(() => postJson(api.answers, documentNow())
          .then(() => { $("#saved").textContent = "Saved in R"; })
          .catch((e) => { $("#saved").textContent = "Not saved in R: " + e.message; }), 300);
      }
    }
    const documentNow = () => L.answersDocument(task, answers, coder, new Date().toISOString());

    function clock() { // time spent on the item shown
      const it = items[current], now = Date.now();
      entry(it).seconds += (now - shownAt) / 1000;
      shownAt = now;
    }
    function go(i) {
      if (i < 0 || i >= items.length) return;
      clock();
      current = i;
      render();
      $("#item").focus();
    }
    function answer(q, value) {
      const it = items[current], a = entry(it);
      const before = L.stage(task, a.values).revealed;
      if (value === "" || value == null) delete a.values[q.id];
      else a.values[q.id] = value;
      a.answered_at = new Date().toISOString();
      persist();
      const after = L.stage(task, a.values).revealed;
      renderItem(); renderList(); renderProgress();
      if (!before && after) say("The label and the model's answer are shown below.");
    }

    function questionBlock(q, values) {
      if (!Array.isArray(q.options) || !q.options.length) {
        const ta = el("textarea", { rows: 2, "aria-label": q.prompt, placeholder: "Type here" });
        ta.value = values[q.id] || "";
        ta.addEventListener("change", () => answer(q, ta.value.trim()));
        return el("fieldset", { class: "q" }, el("legend", { text: q.prompt }), ta);
      }
      return el("fieldset", { class: "q", "data-q": q.id }, el("legend", { text: q.prompt }),
        el("div", { class: "options", role: "group", "aria-label": q.prompt }, q.options.map((o, k) =>
          el("button", { type: "button", class: "opt" + (values[q.id] === o.value ? " on" : ""), "aria-pressed": String(values[q.id] === o.value),
            "data-value": o.value, onclick: () => answer(q, values[q.id] === o.value ? null : o.value) },
          k < 9 && el("kbd", { text: String(k + 1) }), o.text))));
    }
    function reveal(it) {
      const parts = [];
      const row = (k, v) => el("div", { class: "kv" }, el("span", { class: "muted", text: k }), el("strong", { text: v == null ? "–" : v }));
      if (task.has.label) parts.push(row("Label", it.label));
      if (task.has.prediction) parts.push(row("Model", it.prediction));
      const scores = L.scoreList(task, it);
      if (scores.length) {
        parts.push(el("div", { class: "bars", role: "list", "aria-label": "Model scores" }, scores.map((s) =>
          el("div", { class: "bar-row" + (s.top ? " top" : ""), role: "listitem" },
            el("span", { text: s.cls }), el("span", { class: "track" }, el("span", { class: "fill", style: { width: (100 * s.p).toFixed(1) + "%" } })),
            el("span", { class: "num", text: fmt(s.p, 2) })))));
      }
      return parts.length ? el("section", { class: "reveal", "aria-label": "Label and model" }, parts) : null;
    }
    function renderItem() {
      const it = items[current], values = entry(it).values;
      const s = L.stage(task, values);
      const media = [];
      if (it.video) {
        media.push(el("video", { src: it.video, controls: true, preload: "metadata", class: "clip" }));
      }
      if (it.frames && it.frames.length) {
        media.push(el("div", { class: "strip", role: "list", "aria-label": it.media_type === "video" ? "Frames" : "Image" },
          it.frames.map((fr, k) => el("button", { type: "button", class: "frame", role: "listitem",
            "aria-label": `Enlarge ${it.media_type === "video" ? "frame " + (k + 1) : "image"}`, onclick: () => enlarge(it, k) },
          picture(fr, it.frames.length > 1 ? "180px" : "360px", ""),
          fr.time != null && el("span", { class: "muted", text: fmt(fr.time, 1) + " s" })))));
        if (it.frames.some((fr) => fr.faces && fr.faces.length)) {
          media.push(el("p", { class: "muted small", text: "Solid box: the face the model scores by default (the largest). Dashed: other faces found." }));
        }
      }
      const pending = task.blind && (task.has.label || task.has.prediction || task.has.scores) && !s.revealed;
      put($("#item"),
        el("div", { class: "head" }, el("h2", { text: `Item ${current + 1} of ${items.length}` }),
          el("span", { class: "muted", text: it.id + (it.label_file ? " · " + it.label_file : "") })),
        media.length && el("div", { class: "media" }, media),
        it.text && el("blockquote", { class: "text", text: it.text }),
        s.before.map((q) => questionBlock(q, values)),
        pending && el("p", { class: "muted", text: "The label and the model's answer appear once you have answered." }),
        s.revealed && reveal(it),
        s.after.map((q) => questionBlock(q, values)));
      $("#prev").disabled = current === 0;
      $("#next").disabled = current === items.length - 1;
    }
    function renderList() {
      put($("#list"), items.map((it, i) => el("button", { type: "button", class: "dot" + (i === current ? " here" : "") +
          (L.isCoded(task, (answers[it.id] || {}).values) ? " done" : ""), "aria-label": `Item ${i + 1}${L.isCoded(task, (answers[it.id] || {}).values) ? ", coded" : ""}`,
        "aria-current": i === current ? "true" : null, onclick: () => go(i), text: String(i + 1) })));
    }
    function renderProgress() {
      const p = L.progress(task, answers);
      $("#progress").textContent = `${p.coded} of ${p.total} coded`;
      if (finishing && p.coded === p.total) finishing = false;
      const fin = $("#finish");
      if (fin) fin.textContent = finishing ? `Finish with ${p.total - p.coded} not coded?` : fin.getAttribute("data-label");
    }
    function enlarge(it, k) {
      const fr = it.frames[k];
      $("#viewer-title").textContent = it.media_type === "video" ? `${it.id} · ${fmt(fr.time, 1)} s` : it.id;
      put($("#viewer-body"), el("div", { class: "big" }, picture(fr, "min(70vh, 900px)", "")),
        it.frames.length > 1 && el("div", { class: "row" },
          el("button", { type: "button", disabled: k === 0, onclick: () => enlarge(it, k - 1), text: "← Previous frame" }),
          el("span", { class: "muted", text: `${k + 1} / ${it.frames.length}` }),
          el("button", { type: "button", disabled: k === it.frames.length - 1, onclick: () => enlarge(it, k + 1), text: "Next frame →" })));
      if (!$("#viewer").open) $("#viewer").showModal();
    }

    function ended(msg) {
      done = true;
      $("#foot").hidden = true;
      put($("#tools"));
      put($("#main"), el("div", { class: "card narrow", role: "status" }, el("h2", { text: msg }),
        el("p", { class: "muted", text: "You can close this tab." })));
    }
    async function finish(btn) {
      const p = L.progress(task, answers);
      if (p.coded < p.total && !finishing) { finishing = true; renderProgress(); return; }
      clock();
      btn.disabled = true;
      try {
        const out = await postJson(api.finish, documentNow());
        try { localStorage.removeItem(STORE); } catch (e) { /* no storage */ }
        ended(`Sent the answers for ${out.saved} items to R.`);
      } catch (e) {
        btn.disabled = false;
        say("R did not take the answers: " + e.message);
        exportFile();
      }
    }
    async function cancel() {
      try { await postJson(api.cancel, { task_id: task.task_id }); } catch (e) { /* R stopped waiting */ }
      ended("Cancelled. The answers sent so far are saved in R's file.");
    }
    function exportFile() {
      clock();
      const name = `answers-${task.task_id}.json`, text = JSON.stringify(documentNow(), null, 2);
      download(name, text);
      sheet("Exported answers", [`Downloaded as ${name}. In R, read it with `, el("code", { text: `read_annotations("${name}", items)` }),
        `, giving the path of the download. If no file appeared, copy the text below and save it as ${name}.`], text, "Copy JSON");
    }
    function copyCode() {
      clock();
      sheet("R code for these answers", ["Paste this line into a new cell and run it: ", el("code", { text: "read_annotations()" }),
        " returns the answers. Copy it again after more answers: it holds the answers given so far."],
      L.answersCode(task, answers, coder, new Date().toISOString(), notebook.items), "Copy code");
    }

    // Header: coder and progress. Footer: moving and leaving
    const who = el("input", { type: "text", id: "coder", placeholder: "Your name or initials", "aria-label": "Coder" });
    who.value = coder;
    who.addEventListener("change", () => { coder = who.value.trim(); persist(); });
    put($("#tools"), el("label", { class: "coder" }, "Coder ", who), el("span", { id: "progress", class: "chip", role: "status" }),
      api && el("span", { id: "saved", class: "muted small" }));
    const leave = api ? [el("button", { type: "button", id: "cancel", onclick: cancel, text: "Cancel" }),
      el("button", { type: "button", class: "primary", id: "finish", "data-label": "Finish and send to R", text: "Finish and send to R", onclick: (e) => finish(e.currentTarget) })]
      : notebook ? el("button", { type: "button", class: "primary", id: "copy-code", text: "Copy R code", onclick: copyCode })
        : el("button", { type: "button", class: "primary", id: "export-answers", text: "Export answers", onclick: exportFile });
    put($("#foot"), el("div", { class: "row" },
      el("button", { type: "button", id: "prev", onclick: () => go(current - 1), text: "← Previous" }),
      el("button", { type: "button", id: "next", onclick: () => go(current + 1), text: "Next →" })),
    el("div", { class: "row" }, leave));
    $("#foot").hidden = false;
    put($("#main"), task.instructions && el("p", { class: "lead", text: task.instructions }),
      el("nav", { id: "list", class: "list", "aria-label": "Items" }),
      el("article", { id: "item", class: "card item", tabindex: "-1" }));

    document.addEventListener("keydown", (e) => {
      const typing = e.target && e.target.closest && e.target.closest("input, textarea");
      if ($("#viewer").open || typing || e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === "ArrowRight") { go(current + 1); e.preventDefault(); return; }
      if (e.key === "ArrowLeft") { go(current - 1); e.preventDefault(); return; }
      if (/^[1-9]$/.test(e.key)) {
        const q = L.currentQuestion(task, entry(items[current]).values);
        const o = q && q.options[Number(e.key) - 1];
        if (o) { answer(q, o.value); e.preventDefault(); }
      }
    });
    render();
    function render() { renderItem(); renderList(); renderProgress(); }
  }

  const task = window.TE_TASK;
  const problem = L.checkTask(task);
  if (problem) { fatal(problem); return; }
  $("#title").textContent = task.title || "Annotate emotions";
  document.title = task.title || "Annotate emotions";
  annotator(task);
  if (notebook && window.parent !== window) {
    const report = () => window.parent.postMessage({ teFacesHeight: document.documentElement.scrollHeight }, "*");
    if (typeof ResizeObserver === "function") new ResizeObserver(report).observe(document.body);
    report();
  }
})();
