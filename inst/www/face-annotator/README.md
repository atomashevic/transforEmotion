# Face viewer

The web page that `view_faces()` writes. It shows the faces of a
`detect_faces()` table (`view` mode) and lets the user choose faces to analyse
(`select` mode). Plain JavaScript with no framework, no build step and no
network access: the files here are the source and are copied as they are.

| File | Role |
|---|---|
| `index.html` | The page. Its Content-Security-Policy allows only files in its own folder. |
| `style.css` | Layout and colours (light and dark). |
| `logic.js` | Pure functions: labels, colours, selection, the exported file. Tested in Node by `tests/js/`. |
| `app.js` | Builds the page from `window.TE_TASK` with `logic.js`. |

`view_faces()` copies these files into a new folder, next to `data.js` and
`media/`. It works when opened from disk (`file://`), so `data.js` is a plain
script that sets `window.TE_TASK`: browsers block `fetch()` on such pages.

## data.js (`schema: "transforEmotion.face_task"`, version 1.0.0)

- `task_id`, `mode` (`"view"` or `"select"`), `title`, `instructions`.
- `media`: one entry per image, `{media_id, media_type: "image", label, src, img_w, img_h}`,
  or one video, `{media_id, media_type: "video", label, img_w, img_h, frames: [{frame, frame_index, timestamp_s, src}]}`.
  `label` is the file name; `src` is a smaller copy under `media/`;
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
`read_face_selection()` keeps the rows of the faces table whose ids are in
`selected_face_ids`, after checking that `media` names the same files.
`selected_person_ids` lists only people whose faces are all selected.

## Tests

`node --test tests/js/*.test.js` from the package root.
`tests/testthat/test_face_page_js.R` runs the same tests, and checks the
syntax of the page's scripts, whenever Node 18 or later is installed, so
R CMD check on CI runs them too.
