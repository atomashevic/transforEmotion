/* transforEmotion face viewer: the page.
 *
 * view mode shows every image and sampled video frame with its face boxes,
 * coloured by person. select mode makes faces, tracks and people clickable
 * and sends the chosen face_ids to R, when R serves the page, or exports
 * them for read_face_selection() when the page is opened from disk.
 * annotate mode shows one face at a time with the task's questions and
 * saves each answer in R (annotate_faces()), or in the browser and an
 * exported file when the page is opened from disk.
 * The logic lives in logic.js; this file only builds the page.
 */
(function () {
  "use strict";
  const L = window.TEFaceLogic;
  const SVGNS = "http://www.w3.org/2000/svg";
  const N_COLOURS = 6;
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // el("div", {class: "x", onclick: fn, style: {"--ar": 1.5}}, ...children); "svg:rect" makes SVG
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
  const colour = (n) => `var(--c${((Math.max(1, n) - 1) % N_COLOURS) + 1})`;
  const pct = (v, total) => (100 * v / total).toFixed(3) + "%";
  const fmt = (v, d) => (v == null ? "NA" : Number(v).toFixed(d));
  const plural = (n, word, many) => `${n} ${n === 1 ? word : many || word + "s"}`;
  const ui = { main: $("#main"), tools: $("#tools"), foot: $("#foot"), title: $("#title"), chip: $("#mode-chip") };

  function say(msg) { // polite live region, cleared after a few seconds
    $("#status").textContent = msg;
    clearTimeout(say.timer);
    say.timer = setTimeout(() => { $("#status").textContent = ""; }, 6000);
  }
  function fatal(msg) {
    put(ui.main, el("div", { class: "card narrow", role: "alert" }, el("h2", { text: "Cannot open these faces" }), el("p", { text: msg })));
  }

  /* Talking to R, when R serves the page (window.TE_API, set in data.js) */
  const api = window.TE_API || null;
  // In a notebook (window.TE_NOTEBOOK, set in data.js) the page lives in the
  // cell's output and gives the selection as a line of R code
  const notebook = window.TE_NOTEBOOK || null;
  async function postJson(url, data) {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data), cache: "no-store" });
    let out = {};
    try { out = await res.json(); } catch (e) { /* not JSON */ }
    if (!res.ok) throw new Error(out.error || `HTTP ${res.status}`);
    return out;
  }

  /* Leaving the page: a download, and the same JSON shown in the page */
  function download(filename, text) { // may be blocked (sandboxed frames): the in-page copy still works
    try {
      const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
      const a = el("a", { href: url, download: filename, hidden: true });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (e) { /* ignore */ }
  }
  function showCode(code, heading, reader, what) {
    $("#export-h").textContent = heading || "R code for this selection";
    $("#export-text").value = code;
    $("#export-text").setAttribute("aria-label", "R code");
    $("#export-text").rows = 4;
    $("#export-copy").textContent = "Copy code";
    put($("#export-note"), "Paste this line into a new cell and run it: ",
      el("code", { text: reader || "read_face_selection()" }), ` returns ${what || "the selected faces"}.`);
    $("#export").hidden = false;
    $("#export-copy").focus();
  }
  // hint: what to do with the file in R, as [text, code, text]
  function deliver(filename, data, hint) {
    const text = JSON.stringify(data, null, 2);
    download(filename, text);
    $("#export-text").value = text;
    $("#export-h").textContent = hint.title;
    put($("#export-note"), `Downloaded as ${filename}. ${hint.before}`, el("code", { text: hint.code }),
      `${hint.after} If no file appeared, copy the text below and save it as ${filename}.`);
    $("#export").hidden = false;
    $("#export-copy").focus();
  }
  function warn(msg) { // stays until the problem is gone
    const b = $("#banner");
    b.textContent = msg || "";
    b.hidden = !msg;
  }
  const store = { // localStorage, which may be blocked; never required
    get(k) { try { return JSON.parse(window.localStorage.getItem(k)); } catch (e) { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; } }
  };
  const nowIso = () => new Date().toISOString();
  $("#export-close").addEventListener("click", () => { $("#export").hidden = true; });
  $("#export-copy").addEventListener("click", async () => {
    const ta = $("#export-text");
    try {
      await navigator.clipboard.writeText(ta.value);
      say("Copied to the clipboard.");
    } catch (e) { // clipboard blocked: leave the text selected instead
      ta.focus();
      ta.select();
      say("Copying is blocked here. The text is selected: press Ctrl+C (Cmd+C on a Mac).");
    }
  });
  $("#viewer-close").addEventListener("click", () => $("#viewer").close());

  /* Pictures with boxes */
  // The overlay's viewBox is the ORIGINAL pixel grid, so boxes stored in
  // original pixels line up with the smaller display copy at any size
  function stage(src, W, H, maxh, name) {
    const box = el("span", { class: "stage", style: { "--ar": (W / H).toFixed(5), "--maxh": maxh } });
    const img = el("img", { src, alt: "", width: W, height: H, draggable: "false", loading: "lazy" });
    img.addEventListener("error", () => box.classList.add("broken"));
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "group", "aria-label": name });
    box.append(img, svg);
    return { box, svg, W, H };
  }
  function drawBox(st, f, o) {
    const g = el("svg:g", { class: "face", "data-face": f.face_id, style: { "--c": o.colour },
      role: o.onToggle ? "button" : "img", "aria-label": o.title, tabindex: o.onToggle ? "0" : null,
      "aria-pressed": o.onToggle ? "false" : null },
    el("svg:title", { text: o.title }),
    el("svg:rect", { class: "halo", x: f.x, y: f.y, width: f.w, height: f.h }),
    el("svg:rect", { class: "line", x: f.x, y: f.y, width: f.w, height: f.h }));
    if (o.onToggle) {
      g.addEventListener("click", o.onToggle);
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); o.onToggle(); }
      });
    }
    st.svg.append(g);
    if (o.badge) st.box.append(el("span", { class: "badge" + (f.y / st.H < 0.07 ? " inside" : ""), "aria-hidden": "true",
      "data-face": f.face_id, text: o.badge, style: { left: pct(f.x, st.W), top: pct(f.y, st.H), "--c": o.colour } }));
  }
  // A square close-up made by pointing an SVG viewBox at the box: no canvas
  function crop(src, W, H, f, pad) {
    const side = Math.max(f.w, f.h) * (1 + 2 * pad), cx = f.x + f.w / 2, cy = f.y + f.h / 2;
    return el("svg:svg", { class: "crop", viewBox: `${cx - side / 2} ${cy - side / 2} ${side} ${side}`, "aria-hidden": "true" },
      el("svg:image", { href: src, x: 0, y: 0, width: W, height: H, preserveAspectRatio: "none" }));
  }

  function browser(task) {
    const selecting = task.mode === "select";
    const faces = task.faces;
    const chosen = new Set(faces.filter((f) => f.selected).map((f) => f.face_id));
    const initial = new Set(chosen);
    let showScores = false, viewing = null;
    const mediaById = new Map(task.media.map((m) => [m.media_id, m]));
    const nameOf = (m) => m.label || m.media_id;
    const persons = L.personsOf(faces);
    const personFaces = (p) => faces.filter((f) => f.person_id === p);
    const trackFaces = (mid, t) => faces.filter((f) => f.media_id === mid && f.track_id === t);
    const facesOf = (mid, fi) => faces.filter((f) => f.media_id === mid && (fi === undefined || f.frame_index === fi));
    const tint = (f) => colour(L.colourNumber(f, faces));
    const srcOf = (f) => {
      const m = mediaById.get(f.media_id);
      return m.media_type === "video" ? m.frames.find((x) => x.frame_index === f.frame_index).src : m.src;
    };
    const best = (fs) => fs.reduce((a, b) => ((b.confidence || 0) > (a.confidence || 0) ? b : a));
    // "in 2 images", or "in 13 frames" for a video
    const isVideo = task.media.some((m) => m.media_type === "video");
    const where = (fs) => isVideo
      ? plural(new Set(fs.map((f) => f.media_id + "#" + f.frame_index)).size, "frame")
      : plural(new Set(fs.map((f) => f.media_id)).size, "image");

    function toggle(group, what) {
      const on = L.toggleGroup(chosen, group);
      sync();
      say(`${on ? "Selected" : "Deselected"} ${what} (${plural(group.length, "face")}).`);
    }
    function toggleFace(f) {
      const what = f.track_id != null ? `track ${L.trackLabel(f.track_id)} in ${nameOf(mediaById.get(f.media_id))}`
        : `face ${L.faceNumber(f.face_id)} in ${nameOf(mediaById.get(f.media_id))}`;
      toggle(L.clickGroup(f, faces), what);
    }
    function paint(st, f) {
      drawBox(st, f, {
        colour: tint(f), title: L.describe(f),
        badge: L.boxLabel(f) + (showScores && f.confidence != null ? " · " + fmt(f.confidence, 2) : ""),
        onToggle: selecting ? () => toggleFace(f) : null
      });
    }
    // A person or track thumbnail: a toggle button in select mode, a plain chip in view mode
    function chip(fs, title, sub, what, attrs) {
      const b = best(fs), style = { "--c": tint(fs[0]) };
      const m = mediaById.get(b.media_id);
      const inner = [crop(srcOf(b), m.img_w, m.img_h, b, 0.2),
        el("span", {}, el("strong", { text: title }), sub.map((t) => el("small", { text: t })))];
      if (!selecting) return el("div", Object.assign({ class: "track", style }, attrs), inner);
      const btn = el("button", Object.assign({ type: "button", class: "track", style, "aria-pressed": "false",
        "aria-label": `${what}: ${sub.join(", ")}`, onclick: () => toggle(fs, what) }, attrs), inner);
      btn.teFaces = fs; // read by sync()
      return btn;
    }
    function peoplePanel() {
      if (!persons.length) return null;
      return el("section", { class: "card people", "aria-labelledby": "people-h" },
        el("h2", { id: "people-h", text: selecting ? "People: choose a person to select all their faces"
          : `People recognised across ${isVideo ? "the video" : "all images"}` }),
        el("div", { class: "tracks" }, persons.map((p) => {
          const fs = personFaces(p), tracks = new Map();
          fs.filter((f) => f.track_id != null).forEach((f) =>
            tracks.set(f.media_id, new Set(tracks.get(f.media_id)).add(L.trackLabel(f.track_id))));
          const sub = [`${plural(fs.length, "face")} in ${where(fs)}`]
            .concat([...tracks].map(([mid, ts]) => `${nameOf(mediaById.get(mid))}: ${[...ts].join(" + ")}`));
          return chip(fs, L.personLabel(p), sub, `person ${L.personLabel(p)}`, { "data-person": p });
        })));
    }
    function imageCard(m) {
      const fs = facesOf(m.media_id);
      const st = stage(m.src, m.img_w, m.img_h, "220px", `${nameOf(m)}: ${plural(fs.length, "face")}`);
      fs.forEach((f) => paint(st, f));
      const open = () => openViewer(m, 0);
      return el("article", { class: "card", "data-media": m.media_id },
        selecting ? el("div", { class: "thumb" }, st.box)
          : el("button", { type: "button", class: "thumb", "aria-label": `Open ${nameOf(m)} large`, onclick: open }, st.box),
        el("div", { class: "meta" }, el("span", {}, el("strong", { text: nameOf(m) }), " · " + plural(fs.length, "face")),
          selecting && el("button", { type: "button", class: "small", "aria-label": `Enlarge ${nameOf(m)}`, onclick: open, text: "Enlarge" })));
    }
    function videoCard(m) {
      const tracks = L.tracksOf(faces, m.media_id), groups = new Map(); // person -> tracks
      tracks.forEach((t) => {
        const p = trackFaces(m.media_id, t)[0].person_id;
        groups.set(p, (groups.get(p) || []).concat([t]));
      });
      const people = [...groups].map(([p, ts]) => el("div", { class: "pgroup", role: "group",
        "aria-label": p != null ? `Person ${L.personLabel(p)}` : "Not recognised" },
      el("span", { class: "plabel", text: p != null ? L.personLabel(p) : "?" }),
      ts.map((t) => {
        const fs = trackFaces(m.media_id, t);
        return chip(fs, L.trackLabel(t), [`in ${fs.length} of ${m.frames.length} frames`],
          `track ${L.trackLabel(t)}`, { "data-track": `${m.media_id}#${t}` });
      })));
      const strip = el("div", { class: "strip", role: "list", "aria-label": `Frames of ${nameOf(m)}` });
      m.frames.forEach((fr, i) => {
        const fs = facesOf(m.media_id, fr.frame_index);
        const st = stage(fr.src, m.img_w, m.img_h, "150px", `Frame ${fr.frame} at ${fmt(fr.timestamp_s, 1)} s: ${plural(fs.length, "face")}`);
        fs.forEach((f) => paint(st, f));
        strip.append(el("div", { class: "frame", role: "listitem", "data-frame": fr.frame_index },
          selecting ? st.box : el("button", { type: "button", class: "thumb", "aria-label": `Open frame ${fr.frame} large`,
            onclick: () => openViewer(m, i) }, st.box),
          el("span", { text: `${fmt(fr.timestamp_s, 1)} s · ${plural(fs.length, "face")}` })));
      });
      return el("article", { class: "card wide", "data-media": m.media_id },
        el("div", { class: "meta" }, el("span", {}, el("strong", { text: nameOf(m) }),
          ` · video, ${plural(m.frames.length, "sampled frame")} · ${plural(tracks.length, "track")}, ${plural(persons.filter((p) => groups.has(p)).length, "person", "people")}`),
        el("button", { type: "button", class: "small", onclick: () => openViewer(m, 0), text: "Enlarge" })),
        people.length > 0 && el("div", { class: "tracks", role: "group", "aria-label": "People and their tracks in this video" }, people),
        strip);
    }
    function sync() {
      for (const n of $$("[data-face]")) {
        const on = chosen.has(n.getAttribute("data-face"));
        if (n.hasAttribute("aria-pressed")) n.setAttribute("aria-pressed", String(on));
        n.classList.toggle("on", on);
      }
      for (const n of $$("button[data-track], button[data-person]")) n.setAttribute("aria-pressed", L.groupState(chosen, n.teFaces));
      if (selecting) summary();
    }
    const sumText = el("strong"), sumIds = el("code", { class: "ids" });
    function summary() {
      const ids = L.selectedIds(faces, chosen);
      sumText.textContent = `${plural(ids.length, "face")} selected in ${where(faces.filter((f) => chosen.has(f.face_id)))}`;
      sumIds.textContent = ids.join("  ") || "none";
    }
    function render() {
      put(ui.main, task.instructions && el("p", { class: "lead", text: task.instructions }), peoplePanel(),
        el("div", { class: "grid" }, task.media.map((m) => (m.media_type === "video" ? videoCard(m) : imageCard(m)))));
      sync();
    }
    function openViewer(m, i) {
      viewing = { m, i };
      drawViewer();
      if (!$("#viewer").open) $("#viewer").showModal();
    }
    function drawViewer() {
      const { m, i } = viewing, video = m.media_type === "video", fr = video ? m.frames[i] : null;
      const fs = facesOf(m.media_id, video ? fr.frame_index : undefined);
      const st = stage(video ? fr.src : m.src, m.img_w, m.img_h, "min(64vh, 900px)", `${nameOf(m)}: ${plural(fs.length, "face")}`);
      fs.forEach((f) => paint(st, f));
      $("#viewer-title").textContent = video ? `${nameOf(m)} · frame ${fr.frame} (${fmt(fr.timestamp_s, 1)} s)` : nameOf(m);
      const step = (d) => { viewing.i = Math.max(0, Math.min(m.frames.length - 1, i + d)); drawViewer(); };
      put($("#viewer-body"), el("div", { class: "big" }, st.box),
        video && el("div", { class: "row" },
          el("button", { type: "button", disabled: i === 0, onclick: () => step(-1), text: "← Previous frame" }),
          el("span", { class: "muted", text: `${i + 1} / ${m.frames.length}` }),
          el("button", { type: "button", disabled: i === m.frames.length - 1, onclick: () => step(1), text: "Next frame →" })),
        fs.length > 0 && el("div", { class: "tablewrap" }, el("table", {}, el("caption", { class: "sr", text: "Detected faces" }),
          el("thead", {}, el("tr", {}, [video ? "Track" : "Face", "Person", "Confidence", "Box x, y, width, height (pixels)"]
            .map((h) => el("th", { scope: "col", text: h })))),
          el("tbody", {}, fs.map((f) => el("tr", {},
            el("td", { text: f.track_id != null ? L.trackLabel(f.track_id) : String(L.faceNumber(f.face_id)) }),
            el("td", { text: f.person_id != null ? L.personLabel(f.person_id) : "–" }),
            el("td", { text: fmt(f.confidence, 2) }),
            el("td", { text: [f.x, f.y, f.w, f.h].map((v) => Math.round(v)).join(", ") })))))));
      viewing.step = step;
      sync();
    }
    document.addEventListener("keydown", (e) => {
      if (!$("#viewer").open || !viewing || viewing.m.media_type !== "video") return;
      if (e.key === "ArrowLeft") viewing.step(-1);
      if (e.key === "ArrowRight") viewing.step(1);
    });

    ui.tools.append(el("label", { class: "check" },
      el("input", { type: "checkbox", id: "show-scores", onchange: (e) => { showScores = e.target.checked; render(); } }), "Show confidence"));
    const selection = () => L.selectionDocument(task, chosen, new Date().toISOString());
    const exportFile = () => deliver(`selection-${task.task_id}.json`, selection(), { title: "Exported selection",
      before: "In R, read it with ", code: `read_face_selection("selection-${task.task_id}.json", faces)`, after: ", giving the path of the download." });
    const copyCode = () => showCode(L.selectionCode(task, chosen, new Date().toISOString(), notebook.faces));
    // Served by R: the choice goes straight back to view_faces()
    async function sendToR(btn) {
      btn.disabled = true;
      try {
        const doc = selection();
        await postJson(api.selection, doc);
        finished(`Sent ${plural(doc.selected_face_ids.length, "face")} to R.`);
      } catch (e) {
        btn.disabled = false;
        say("R did not take the selection: " + e.message);
        exportFile();
      }
    }
    async function cancel() {
      try { await postJson(api.cancel, { task_id: task.task_id }); } catch (e) { /* R stopped waiting already */ }
      finished("Cancelled. R got no faces.");
    }
    function finished(msg) {
      ui.foot.hidden = true;
      put(ui.tools);
      put(ui.main, el("div", { class: "card narrow", role: "status" }, el("h2", { text: msg }),
        el("p", { class: "muted", text: "You can close this tab." })));
    }
    if (selecting) {
      ui.foot.hidden = false;
      const send = api && api.selection && el("button", { type: "button", class: "primary", id: "send-selection",
        text: "Send to R", onclick: (e) => sendToR(e.currentTarget) });
      put(ui.foot, el("div", { role: "status" }, sumText),
        el("details", {}, el("summary", { text: "Show face_ids" }), sumIds),
        el("div", { class: "row" },
          el("button", { type: "button", onclick: () => { chosen.clear(); initial.forEach((id) => chosen.add(id)); sync(); }, text: "Reset" }),
          el("button", { type: "button", onclick: () => { chosen.clear(); sync(); }, text: "Clear" }),
          send ? [api.cancel && el("button", { type: "button", id: "cancel-selection", onclick: cancel, text: "Cancel" }), send]
            : notebook ? el("button", { type: "button", class: "primary", id: "copy-code", text: "Copy R code", onclick: copyCode })
              : el("button", { type: "button", class: "primary", id: "export-selection", text: "Export selection", onclick: exportFile })));
    }
    render();
  }

  /* Annotate mode: one face at a time, questions answered by click or key */
  function annotator(task) {
    const items = task.items, byId = new Map(items.map((it) => [it.item_id, it]));
    const questions = task.questions;
    const settings = Object.assign({ allow_back: true, allow_skip: true, show_progress: true, context_padding: 0.25 }, task.settings);
    const coders = task.assignments.map((a) => a.coder_id);
    const sessionId = (window.crypto && crypto.randomUUID) ? crypto.randomUUID()
      : "s-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
    const lastKey = `te-annot:${task.task_id}:last-coder`;
    let coder = null, order = [], pos = 0, records = [], unsent = [], cur = null, saving = Promise.resolve();
    const storeKey = () => `te-annot:${task.task_id}:${coder}`; // pages opened from disk share one storage
    const options = new Map(questions.map((q) => [q.id, L.optionsOf(q, task)]));
    const progress = el("span", { class: "progress", role: "progressbar", "aria-label": "Faces done", "aria-valuemin": "0" }, el("span"));
    const progressText = el("span", { class: "muted", id: "progress-text" });

    function chooseCoder() {
      cur = null;
      put(ui.tools);
      const wanted = /[?&]coder=([A-Za-z0-9_.-]{1,40})(?:&|$)/.exec(location.search);
      if (wanted && coders.includes(wanted[1])) return begin(wanted[1]);
      if (notebook && notebook.coder && coders.includes(notebook.coder)) return begin(notebook.coder);
      if (coders.length === 1) return begin(coders[0]);
      const last = store.get(lastKey);
      put(ui.main, el("section", { class: "card narrow", "aria-labelledby": "coder-h" },
        el("h2", { id: "coder-h", text: "Who is coding?" }),
        el("p", { class: "muted", text: "Choose the coder ID you were given. Use the same ID to continue later." }),
        el("div", { class: "row coders" }, coders.map((id) => el("button", { type: "button", "data-coder": id,
          class: id === last ? "primary" : null, onclick: () => begin(id), text: id })))));
      const first = $(".coders button.primary") || $(".coders button");
      if (first) first.focus();
    }

    async function begin(id) {
      coder = id;
      store.set(lastKey, id);
      order = L.orderFor(task, id);
      const saved = store.get(storeKey());
      records = saved && Array.isArray(saved.records) ? saved.records : [];
      if (api && api.load) {
        try {
          const fromR = (await postJson(api.load, { coder_id: id })).records || [];
          // R has what was saved; anything only in this browser is sent again
          const inR = new Set(fromR.map((r) => r.session_id + "#" + r.seq));
          unsent = records.filter((r) => !inR.has(r.session_id + "#" + r.seq));
          records = fromR.concat(unsent);
          if (unsent.length) sync();
        } catch (e) {
          warn("Could not reach R: answers are kept in this browser for now. " + e.message);
        }
      }
      persist();
      pos = L.firstOpen(order, records);
      put(ui.tools, settings.show_progress !== false && progress, settings.show_progress !== false && progressText,
        el("span", { class: "chip", id: "coder-chip", text: "Coder " + id }),
        coders.length > 1 && el("button", { type: "button", class: "small", onclick: chooseCoder, text: "Switch coder" }),
        el("button", { type: "button", class: "small", id: "export-annotations", onclick: exportAll,
          text: notebook ? "Copy R code" : "Export annotations" }));
      const p = L.progressOf(order, records);
      if (p.done) say(`Welcome back, ${id}: ${p.done} of ${p.total} done.`);
      show();
    }
    function persist() {
      const ok = store.set(storeKey(), { coder_id: coder, records });
      if (!ok && !api) warn(notebook ? "This page cannot keep your progress. Copy R code before you leave the notebook."
        : "This browser cannot keep your progress. Export your annotations before closing this tab.");
    }
    // Sends the records R has not confirmed yet, one batch at a time
    function sync() {
      if (!api || !api.save || !unsent.length) return saving;
      saving = saving.then(async () => {
        if (!unsent.length) return;
        const batch = unsent.slice();
        try {
          await postJson(api.save, { records: batch });
          unsent = unsent.filter((r) => !batch.includes(r));
          if (!unsent.length) warn(null);
        } catch (e) {
          warn(`Not saved in R yet (${plural(unsent.length, "answer")}): ${e.message}. They are kept in this browser and sent with the next answer.`);
        }
      });
      return saving;
    }
    function updateProgress() {
      const p = L.progressOf(order, records);
      progress.setAttribute("aria-valuemax", String(order.length));
      progress.setAttribute("aria-valuenow", String(p.done));
      progress.firstChild.style.width = pct(p.done, Math.max(1, order.length));
      progressText.textContent = `${p.done} / ${order.length} done`;
    }

    function show() {
      updateProgress();
      if (pos >= order.length) return finish();
      const item = byId.get(order[pos]), prev = L.latestRecord(records, item.item_id);
      cur = { item, responses: {}, shownAt: nowIso(), t0: performance.now(), dirty: false };
      if (prev) for (const [k, v] of Object.entries(prev.responses || {})) if (v != null) cur.responses[k] = v;
      const isItem = task.unit === "item";
      const media = isItem ? itemMedia(item) : faceMedia(item);

      const views = questions.map((q) => {
        const opts = options.get(q.id), keys = Math.min(9, opts.length);
        const buttons = opts.map((o, i) => el("button", { type: "button", class: "opt", "aria-pressed": "false",
          "data-value": String(o.value),
          "aria-label": q.type === "likert" && q.anchors ? `${o.label}${i === 0 ? ", " + q.anchors.min : i === opts.length - 1 ? ", " + q.anchors.max : ""}` : null,
          onclick: (e) => answer(q, o.value, e) }, i < 9 && el("kbd", { "aria-hidden": "true", text: String(i + 1) }), el("span", { text: o.label })));
        const fs = el("fieldset", { class: "q" + (q.type === "likert" ? " likert" : ""), "data-q": q.id },
          el("legend", {}, q.prompt || L.nice(q.id), " ", el("span", { class: "keyhint", text: `keys 1–${keys}` })),
          el("div", { class: "opts" }, buttons),
          q.anchors && el("div", { class: "anchors", "aria-hidden": "true" }, el("span", { text: q.anchors.min }), el("span", { text: q.anchors.max })));
        return { q, opts, buttons, fs };
      });
      const nextBtn = el("button", { type: "button", class: "primary", id: "next", onclick: next, text: "Next →" });
      const reveal = isItem ? revealBlock(item) : null;
      const panel = el("section", { class: "panel", tabindex: "-1", "aria-labelledby": "item-h" },
        el("div", { class: "itemhead" }, el("h2", { id: "item-h", text: `${isItem ? "Item" : "Face"} ${pos + 1} of ${order.length}` }),
          el("span", { class: "muted", text: isItem ? { clip: "Video clip", image: "Image", text: "Text" }[item.kind] || ""
            : item.media_type === "frame" ? `Video frame at ${fmt(item.timestamp_s, 1)} s` : "Still image" })),
        isItem && item.text && el("blockquote", { class: "itemtext", text: item.text }),
        views.filter((v) => !v.q.reveal).map((v) => v.fs),
        reveal && reveal.node,
        views.filter((v) => v.q.reveal).map((v) => v.fs),
        el("div", { class: "row nav" },
          settings.allow_back && el("button", { type: "button", id: "back", disabled: pos === 0, onclick: back, text: "← Back" }),
          settings.allow_skip && el("button", { type: "button", id: "skip", onclick: skip, text: "Skip" }), nextBtn),
        el("p", { class: "keys muted", text: "Keys: 1–9 answer · Enter next · Backspace undo · ← back · S skip" }));
      put(ui.main,
        task.instructions && el("details", { class: "instructions", open: pos === 0 && !L.progressOf(order, records).done },
          el("summary", { text: "Instructions" }), el("p", { text: task.instructions })),
        el("div", { class: "item" + (isItem ? " review" : ""), "data-item": item.item_id }, media, panel));

      function update() {
        const revealed = L.isRevealed(task, cur.responses);
        if (reveal) {
          reveal.panel.hidden = !revealed || !reveal.any;
          reveal.waiting.hidden = revealed || !reveal.any;
        }
        for (const v of views) {
          const shown = L.isShown(v.q, cur.responses, revealed);
          v.fs.hidden = !shown;
          if (!shown) delete cur.responses[v.q.id];
          v.buttons.forEach((btn, i) => btn.setAttribute("aria-pressed", String(v.opts[i].value === cur.responses[v.q.id])));
        }
        const active = L.activeQuestion(questions, cur.responses, revealed);
        views.forEach((v) => v.fs.classList.toggle("active", v.q === active));
        nextBtn.disabled = !!L.missingQuestion(questions, cur.responses, revealed);
      }
      function answer(q, value, e) {
        cur.responses[q.id] = value;
        cur.dirty = true;
        update();
        if (e && e.detail > 0) panel.focus({ preventScroll: true }); // keys keep working after a click
      }
      cur.update = update;
      cur.answer = answer;
      update();
      panel.focus({ preventScroll: true });
      const nx = byId.get(order[pos + 1]);
      if (nx) [].concat(nx.media || [], (nx.frames || []).map((f) => f.src)).forEach((src) => { new Image().src = src; }); // load the next pictures early
    }

    // A face item: the face outlined in its picture, and a close-up
    function faceMedia(item) {
      const W = item.width, H = item.height, b = item.box;
      const target = { face_id: item.item_id, x: b.x, y: b.y, w: b.w, h: b.h };
      const st = stage(item.media, W, H, "var(--ctx-h)", "Picture with the face to code outlined");
      st.svg.append(el("svg:path", { class: "scrim", "fill-rule": "evenodd", d: `M0 0H${W}V${H}H0Z M${b.x} ${b.y}v${b.h}h${b.w}v${-b.h}Z` }));
      drawBox(st, target, { colour: colour(1), title: "Face to code", badge: "" });
      const zoom = crop(item.media, W, H, target, settings.context_padding);
      zoom.setAttribute("role", "img");
      zoom.setAttribute("aria-label", "Close-up of the face to code");
      zoom.append(el("svg:rect", { class: "cropbox", x: b.x, y: b.y, width: b.w, height: b.h }));
      return el("div", { class: "views" }, el("figure", { class: "ctx" }, st.box),
        el("figure", { class: "zoom" }, zoom, el("figcaption", { text: "Close-up" })));
    }
    // A clip, image or text item: the clip, its frames with face boxes (the
    // scored face solid), nothing for a text
    function framePicture(fr, maxh) {
      const st = stage(fr.src, fr.width, fr.height, maxh, fr.time != null ? `Frame at ${fmt(fr.time, 1)} s` : "Image");
      (fr.faces || []).forEach((f) => st.svg.append(el("svg:rect", { class: "itembox" + (f.scored ? " scored" : ""),
        x: f.x, y: f.y, width: f.w, height: f.h })));
      return st.box;
    }
    function enlargeFrame(item, k) {
      const fr = item.frames[k];
      $("#viewer-title").textContent = fr.time != null ? `Frame at ${fmt(fr.time, 1)} s` : "Image";
      put($("#viewer-body"), el("div", { class: "big" }, framePicture(fr, "min(64vh, 900px)")),
        item.frames.length > 1 && el("div", { class: "row" },
          el("button", { type: "button", disabled: k === 0, onclick: () => enlargeFrame(item, k - 1), text: "← Previous frame" }),
          el("span", { class: "muted", text: `${k + 1} / ${item.frames.length}` }),
          el("button", { type: "button", disabled: k === item.frames.length - 1, onclick: () => enlargeFrame(item, k + 1), text: "Next frame →" })));
      if (!$("#viewer").open) $("#viewer").showModal();
    }
    function itemMedia(item) {
      const frames = item.frames || [];
      if (!item.video && !frames.length) return el("div", { class: "views itemmedia empty" });
      return el("div", { class: "views itemmedia" },
        item.video && el("video", { class: "clip", src: item.video, controls: true, preload: "metadata" }),
        frames.length > 0 && el("div", { class: "strip", role: "list", "aria-label": item.kind === "clip" ? "Frames of the clip" : "Image" },
          frames.map((fr, k) => el("button", { type: "button", class: "frame", role: "listitem", onclick: () => enlargeFrame(item, k),
            "aria-label": fr.time != null ? `Enlarge the frame at ${fmt(fr.time, 1)} s` : "Enlarge the image" },
          framePicture(fr, frames.length > 1 ? "var(--frame-h)" : "var(--ctx-h)"),
          fr.time != null && el("span", { class: "muted", text: `${fmt(fr.time, 1)} s` })))),
        frames.some((fr) => fr.faces && fr.faces.length) &&
          el("p", { class: "muted small", text: "Solid box: the face the model scores (the largest). Dashed: other faces found." }));
    }
    // The item's label, the model's answer and its scores, shown once revealed
    function revealBlock(item) {
      const row = (k, v) => v != null && el("div", { class: "kv" }, el("span", { class: "muted", text: k }), el("strong", { text: v }));
      const scores = L.scoreList(task, item);
      const parts = [row("Label", item.reference), row("Model", item.prediction),
        scores.length > 0 && el("div", { class: "bars", role: "list", "aria-label": "The model's scores" }, scores.map((sc) =>
          el("div", { class: "bar-row" + (sc.top ? " top" : ""), role: "listitem" }, el("span", { text: sc.cls }),
            el("span", { class: "bar-track" }, el("span", { class: "fill", style: { width: (100 * sc.p).toFixed(1) + "%" } })),
            el("span", { class: "num", text: fmt(sc.p, 2) }))))].filter(Boolean);
      const waiting = el("p", { class: "muted reveal-wait", text: "The label and the model's answer appear once you have answered." });
      const panel = el("section", { class: "reveal", "aria-label": "Label and model" }, parts);
      return { node: el("div", { class: "reveal-slot" }, waiting, panel), panel, waiting, any: parts.length > 0 };
    }

    function record(status) {
      const rec = L.makeRecord(task, coder, sessionId, records, cur.item, status, cur.responses,
        cur.shownAt, nowIso(), performance.now() - cur.t0);
      records.push(rec); // append-only: the last record of an item is the answer
      unsent.push(rec);
      persist();
      sync();
    }
    function next() { // an unchanged answer is not recorded again
      if (!cur || L.missingQuestion(questions, cur.responses, L.isRevealed(task, cur.responses))) return;
      if (cur.dirty || !L.latestRecord(records, cur.item.item_id)) record("answered");
      pos++;
      show();
    }
    function skip() {
      if (!cur || !settings.allow_skip) return;
      if (!L.latestRecord(records, cur.item.item_id)) record("skipped");
      pos++;
      show();
    }
    function back() {
      if (settings.allow_back && pos > 0) {
        pos--;
        show();
      }
    }
    function undo() {
      const revealed = L.isRevealed(task, cur.responses);
      const answered = questions.filter((q) => L.isShown(q, cur.responses, revealed) && cur.responses[q.id] != null);
      if (!answered.length) return;
      delete cur.responses[answered.pop().id];
      cur.dirty = true;
      cur.update();
    }
    function exportAll() {
      if (notebook) {
        return showCode(L.annotationsCode(task, coder, order, records, nowIso(), notebook.task), "R code for these answers",
          "read_annotations()", `the answers given so far (${plural(L.progressOf(order, records).done, "face")} done); copy it again after more coding`);
      }
      deliver(`annotations-${task.task_id}-${coder}.json`, L.annotationsDocument(task, coder, order, records, nowIso()), {
        title: "Exported annotations", before: "Put it in the task's results folder; in R, ",
        code: "read_annotations(task)", after: " reads it." });
    }
    async function finish() {
      cur = null;
      const p = L.progressOf(order, records);
      const card = el("section", { class: "card narrow", "aria-labelledby": "done-h" },
        el("h2", { id: "done-h", text: "All faces done" }),
        el("p", { text: `You answered ${p.answered} of ${p.total}${p.skipped ? ` and skipped ${p.skipped}` : ""}. Thank you!` }),
        el("p", { class: "muted", id: "done-note", text: api ? "Saving your answers in R…"
          : notebook ? "Copy R code and run the line in a new cell." : "Export your annotations and send the file to the researcher." }),
        el("div", { class: "row" },
          settings.allow_back && order.length > 0 && el("button", { type: "button", onclick: back, text: "← Review the last face" }),
          el("button", { type: "button", class: api ? null : "primary", onclick: exportAll, text: notebook ? "Copy R code" : "Export annotations" })));
      put(ui.main, card);
      if (api) {
        await sync();
        const note = $("#done-note");
        if (note) note.textContent = unsent.length ? `${plural(unsent.length, "answer")} could not be saved in R yet. Export your annotations to keep them.`
          : "Your answers are saved in R. You can close this tab.";
      }
    }

    document.addEventListener("keydown", (e) => {
      if (!cur || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target;
      if (t.closest && t.closest("input, textarea, select, .sheet")) return;
      if (/^[1-9]$/.test(e.key)) {
        const q = L.activeQuestion(questions, cur.responses, L.isRevealed(task, cur.responses));
        const o = q && options.get(q.id)[Number(e.key) - 1];
        if (o) { e.preventDefault(); cur.answer(q, o.value); }
      } else if (e.key === "Enter" && t.tagName !== "BUTTON") { e.preventDefault(); next(); }
      else if (e.key === "Backspace") { e.preventDefault(); undo(); }
      else if (e.key === "ArrowLeft" && t.tagName !== "BUTTON") { e.preventDefault(); back(); }
      else if (e.key === "s" || e.key === "S") { e.preventDefault(); skip(); }
    });
    window.addEventListener("beforeunload", (e) => { // answers R has not confirmed
      if (unsent.length) { e.preventDefault(); e.returnValue = ""; }
    });
    chooseCoder();
  }

  const task = window.TE_TASK;
  const problem = L.checkTask(task);
  if (problem) {
    fatal(problem);
    return;
  }
  document.body.setAttribute("data-mode", task.mode);
  ui.title.textContent = task.title || "Faces";
  ui.chip.textContent = L.nice(task.mode);
  document.title = `${task.title || "Faces"}: ${task.mode}`;
  if (task.mode === "annotate") annotator(task);
  else browser(task);
  // In a notebook the page is an iframe in the cell's output: tell the output
  // how tall the page is, so it needs no scroll bar of its own
  if (notebook && window.parent !== window) {
    document.body.setAttribute("data-notebook", "");
    const report = () => window.parent.postMessage({ teFacesHeight: document.documentElement.scrollHeight }, "*");
    if (typeof ResizeObserver === "function") new ResizeObserver(report).observe(document.body);
    report();
  }
})();
