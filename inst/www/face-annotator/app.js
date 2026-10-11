/* transforEmotion face viewer: the page.
 *
 * view mode shows every image and sampled video frame with its face boxes,
 * coloured by person. select mode makes faces, tracks and people clickable
 * and sends the chosen face_ids to R, when R serves the page, or exports
 * them for read_face_selection() when the page is opened from disk.
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
  function deliver(filename, data) {
    const text = JSON.stringify(data, null, 2);
    download(filename, text);
    $("#export-text").value = text;
    put($("#export-note"), `Downloaded as ${filename}. In R, read it with `,
      el("code", { text: `read_face_selection("${filename}", faces)` }),
      `, giving the path of the download. If no file appeared, copy the text below and save it as ${filename}.`);
    $("#export").hidden = false;
    $("#export-copy").focus();
  }
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
    st.box.append(el("span", { class: "badge" + (f.y / st.H < 0.07 ? " inside" : ""), "aria-hidden": "true",
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
    const exportFile = () => deliver(`selection-${task.task_id}.json`, selection());
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
            : el("button", { type: "button", class: "primary", id: "export-selection", text: "Export selection", onclick: exportFile })));
    }
    render();
  }

  const task = window.TE_TASK;
  const problem = L.checkTask(task);
  if (problem) {
    fatal(problem);
    return;
  }
  document.body.setAttribute("data-mode", task.mode);
  ui.title.textContent = task.title || "Faces";
  ui.chip.textContent = task.mode === "select" ? "Select" : "View";
  document.title = `${task.title || "Faces"}: ${task.mode}`;
  browser(task);
})();
