# Annotation page

The web page that `annotate_emotions()` writes. It shows one item at a time
(an image or the frames of a clip with face boxes, the clip itself, a text)
and asks the task's questions. Plain JavaScript with no framework, no build
step and no network access, like the face viewer in `../face-annotator`,
whose server (`R/face_server.R`) and notebook mode (`R/face_notebook.R`) it
shares.

| File | Role |
|---|---|
| `index.html` | The page. Its Content-Security-Policy allows only files in its own folder and requests to the R session that serves it. |
| `style.css` | Layout and colours (light and dark). |
| `logic.js` | Pure functions: what to show, progress, the answers document and code. Tested in Node by `tests/js/annotate.test.js`. |
| `app.js` | Builds the page from `window.TE_TASK` with `logic.js`. |

## How the answers leave

- **Served by R** (`window.TE_API`): every change POSTs the whole answers
  document to `api/answers`, and R writes it to the `save` file; **Finish**
  posts to `api/finish` and `annotate_emotions()` returns the answers;
  **Cancel** posts to `api/cancel`.
- **In a notebook** (`window.TE_NOTEBOOK = {"items": "<R name>"}`): **Copy R
  code** gives `answers <- read_annotations("te-annotations:<base64>", <R name>)`.
- **Opened from disk**: **Export answers** downloads `answers-<task>.json`.

The page also keeps the answers in the browser's `localStorage` (when it may),
so a reload does not lose them.

## data.js (`schema: "transforEmotion.annotation_task"`, version 1.0.0)

- `task_id`, `title`, `instructions`, `coder`, `blind`, `classes`,
  `score_classes` (the classes with a score column), `has` (`label`,
  `prediction`, `scores`: whether the items have them).
- `questions`: `{id, prompt, options: [{value, text}] or null for free text, reveal}`.
  With `blind`, questions with `reveal: true`, and the label, prediction and
  scores, appear once every question without it is answered.
- `items`: `{id, text, label, prediction, scores: {class: p}, media_type
  ("video", "image" or null), label_file, frames: [{src, w, h, time, faces:
  [{x, y, w, h, scored}]}], video}`. Boxes are in the original pixels of the
  frame, `w` and `h` its original size; `scored` marks the largest face, the
  one the scoring functions use by default. `video` is the clip's copy in
  `media/`, when the browser can play it.

## The answers (`schema: "transforEmotion.annotations"`)

`{schema, schema_version, task_id, coder, exported_at, question_ids,
answers: [{id, <question ids>..., seconds, answered_at}]}`, one entry per
item with any answer. `read_annotations()` turns it into a data frame and
adds the items' label, prediction and scores.
