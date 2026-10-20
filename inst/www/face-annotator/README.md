# Face pages

The web page that `view_faces()` and `annotation_task()` write. It shows the
faces of a `detect_faces()` table (`view` mode), lets the user choose faces to
analyse (`select` mode), and asks coders questions about one face at a time
(`annotate` mode). Plain JavaScript with no framework, no build step and no
network access: the files here are the source and are copied as they are.
`inst/schema/face-task-v1.json` describes every file the page reads or
writes, as JSON Schema.

| File | Role |
|---|---|
| `index.html` | The page. Its Content-Security-Policy allows only files in its own folder and requests to the R session that serves it. |
| `style.css` | Layout and colours (light and dark). |
| `logic.js` | Pure functions: labels, colours, selection, questions, answer records, the exported files. Tested in Node by `tests/js/`. |
| `app.js` | Builds the page from `window.TE_TASK` with `logic.js`. |

`view_faces()` copies these files into a new folder, next to `data.js` and
`media/`, and serves the folder from R with httpuv (`R/face_server.R`) on
127.0.0.1, under a random path. With `live = FALSE` it only writes the folder,
which then also works when opened from disk (`file://`). That is why
`data.js` is a plain script that sets `window.TE_TASK`: browsers block
`fetch()` on such pages.

## Talking to R

When R serves the page in select mode, `data.js` also sets
`window.TE_API = {"selection": "api/selection", "cancel": "api/cancel"}`
(relative to the page). **Send to R** POSTs the selection below as JSON to
`api/selection`, and `view_faces()` returns the chosen rows; **Cancel** POSTs
to `api/cancel`. If R does not answer (it stopped waiting), the page offers
the selection as a file instead. Without `window.TE_API` the page shows
**Export selection**. The server answers only POSTs of JSON to these paths,
with the server's own address as the `Host`; any other address outside the
page is a 404 that never reaches R.

## In a notebook

In a Jupyter notebook (Google Colab, JupyterLab, VS Code), `view_faces()`
shows the page in the cell's output (`R/face_notebook.R`). It writes the
folder as usual, then makes one self-contained `faces.html` from it: the
stylesheet and scripts inline, the images as `data:` URIs in `data.js`, and a
Content-Security-Policy that allows only those. The notebook gets an iframe
whose `srcdoc` is that page; the page posts its height to the iframe's parent
so the output needs no scroll bar of its own.

`data.js` then also sets `window.TE_NOTEBOOK = {"faces": "<R name>"}`. In
select mode the page shows **Copy R code** instead of Export selection: a
line `chosen <- read_face_selection("te-selection:<base64>", <R name>)`,
where the base64 is the exported selection below (UTF-8 JSON), so it needs no
quoting. R reads the code like the file.

## data.js (`schema: "transforEmotion.face_task"`, version 1.0.0)

- `task_id`, `mode` (`"view"` or `"select"`), `title`, `instructions`.
- `media`: one entry per image, `{media_id, media_type: "image", label, key, src, img_w, img_h}`,
  or one video, `{media_id, media_type: "video", label, key, img_w, img_h, frames: [{frame, frame_index, timestamp_s, src}]}`.
  `label` is the file name; `key` its MD5; `src` is a smaller copy under `media/`;
  `img_w` and `img_h` are the original size in pixels.
- `faces`: `{face_id, media_id, media_type, frame, frame_index, timestamp_s, img_w, img_h,
  x, y, w, h, confidence, detector, track_id, person_id, selected}`.
  Boxes are in original pixels. `face_id` is `"<media_id>#<frame_index>#<face>"`,
  with frame 0 for still images; R computes the same ids to read a selection.

Boxes are drawn in an SVG whose `viewBox` is the original pixel size, so they
line up with the smaller copies without any arithmetic.

## The exported selection (`schema: "transforEmotion.face_selection"`)

`{schema, schema_version, task_id, exported_at, media: [{media_id, label}],
selected_face_ids, selected_tracks: [{media_id, track_id}], selected_person_ids}`.
`media` also carries each file's MD5 as `key` (null for URLs).
`read_face_selection()` keeps the rows of the faces table whose ids are in
`selected_face_ids`, after checking that `media` names the same files, by
name and, where it can still read them, by content.
`selected_person_ids` lists only people whose faces are all selected.

## Annotate mode (`schema: "transforEmotion.annotation_task"`)

`annotation_task()` writes the task folder: `page/` (these files, `data.js`
and `media/p001.jpg` and so on), `task.json` (the same task as `data.js`),
`items.csv` (which face each item is) and an empty `results/`. The page gets
no file names: items carry only an opaque `item_id`, the picture, its size and
the face box.

- `questions`: `{id, type: "single" | "likert", prompt, options | options_from: "classes",
  labels, min, max, anchors, show_if, required}`. A question with `show_if`
  (for example `{"face_check": "ok"}`) is asked only after one of those answers.
- `items`: `{item_id, media, media_type, frame_index, timestamp_s, width, height, box: {x, y, w, h}}`.
- `assignments`: `{coder_id, items}`, each coder's items in their own order.

Each answer is a record:
`{task_id, item_id, coder_id, session_id, seq, revision, status: "answered" | "skipped",
responses: {<question id>: value or null}, shown_at, submitted_at, rt_ms}`.
`seq` counts a browser session's records; `revision` counts a coder's records
for one item. Records are only ever added: `read_annotations()` keeps the
last one per item and coder.

`annotate_faces()` serves `page/` and adds
`window.TE_API = {"save": "api/save", "load": "api/load"}` to `data.js`.
The page POSTs `{coder_id}` to `api/load` to get the coder's earlier records,
and `{records: [...]}` to `api/save` after every answer; R checks each record
against the task and appends it to `results/<coder_id>.jsonl`. Records R has
not confirmed stay in the browser (localStorage) and are sent again with the
next answer, or when the coder comes back. Without `window.TE_API` the page
keeps answers in the browser only and offers **Export annotations**, a
`transforEmotion.annotations` file with the coder's records, which
`read_annotations()` reads like the `.jsonl` files.

## Tests

`node --test tests/js/*.test.js` from the package root.
`tests/testthat/test_face_page_js.R` runs the same tests, and checks the
syntax of the page's scripts, whenever Node 18 or later is installed, so
R CMD check on CI runs them too.
