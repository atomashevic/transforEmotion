"""Face detection, tracking and same-person recognition for transforEmotion.

Used by detect_faces(), face_scores(), image_scores() and video_scores(), and by
image.py and video.py. Needs only OpenCV, NumPy and Pillow, so it loads
without PyTorch.

detect()            boxes, confidence and 5 landmarks per face (YuNet), or
                    boxes only (the Haar cascade crop_face() has always used)
select()            the largest/left/right rules of crop_face(), "all", or ids
crop()              a face crop with padding relative to the box size
track()             stable track ids across sampled video frames (IoU)
embed()             SFace embeddings, kept in memory only: they are biometric
                    templates and are never returned to R or written to disk
assign_person_ids() same-person ids across tracks and still images
extract_frames()    the frames video_scores() samples from a video
faces_table()       all of the above for a list of images or video frames
draw()              face boxes and labels drawn on an image, for checking
display_copies()    smaller JPEG copies of images and frames for view_faces()
"""
import math
import os

import numpy as np

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
LANDMARKS = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]

# Detector and recognizer objects, keyed by (kind, model path)
_models = {}


def default_yunet_path():
    """The YuNet model shipped in the package's extdata folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Installed package: <pkg>/python and <pkg>/extdata; source: inst/...
    path = os.path.normpath(os.path.join(here, os.pardir, "extdata", YUNET_FILE))
    if not os.path.exists(path):
        raise FileNotFoundError("YuNet face detection model not found: " + path)
    return path


def load_image(image):
    """Path, URL or PIL image -> RGB PIL image, with EXIF orientation applied."""
    from PIL import Image, ImageOps
    if isinstance(image, str):
        if image.startswith("http"):
            import requests
            image = Image.open(requests.get(image, stream=True).raw)
        else:
            if not os.path.exists(image):
                raise ValueError("Image file does not exist: " + image)
            image = Image.open(image)
    return ImageOps.exif_transpose(image).convert("RGB")


def _yunet(model_path, min_confidence):
    import cv2
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("OpenCV " + cv2.__version__ + " has no YuNet face "
                           "detector (needs 4.8 or later); use face_detector = 'haar'")
    key = ("yunet", model_path)
    if key not in _models:
        # The input size is set per image with setInputSize()
        _models[key] = cv2.FaceDetectorYN.create(model_path, "", (320, 320),
                                                 float(min_confidence), 0.3, 5000)
    det = _models[key]
    det.setScoreThreshold(float(min_confidence))
    return det


def _haar():
    import cv2
    if ("haar", None) not in _models:
        _models[("haar", None)] = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _models[("haar", None)]


def detect(rgb, detector="yunet", min_confidence=0.6, model_path=None,
           max_side=1280, min_size=30):
    """Detect faces in an RGB array; return a list of dicts ordered left to right.

    Coordinates are pixels of `rgb`. Each dict has face (1, 2, ... from left to
    right), x, y, width, height, confidence and landmarks (10 numbers: x and y
    of the right eye, left eye, nose tip and right and left mouth corners, or
    None for Haar).
    """
    import cv2
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    faces = []
    if detector == "yunet":
        # YuNet finds small faces better on a moderately sized input
        scale = min(1.0, float(max_side) / max(h, w))
        small = cv2.resize(rgb, (round(w * scale), round(h * scale))) if scale < 1 else rgb
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        det = _yunet(model_path or default_yunet_path(), min_confidence)
        det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, out = det.detect(bgr)
        for row in (out if out is not None else []):
            x, y, bw, bh = (row[:4] / scale).tolist()
            faces.append(dict(x=x, y=y, width=bw, height=bh, confidence=float(row[14]),
                              landmarks=(row[4:14] / scale).tolist()))
    elif detector == "haar":
        # The same call as crop_face(), on the full-size image
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rects = _haar().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(min_size, min_size))
        for (x, y, bw, bh) in (rects if len(rects) else []):
            faces.append(dict(x=float(x), y=float(y), width=float(bw), height=float(bh),
                              confidence=None, landmarks=None))
    else:
        raise ValueError("face_detector must be 'yunet' or 'haar'")

    # Keep boxes inside the image and at least min_size pixels
    for f in faces:
        x0, y0 = max(0.0, f["x"]), max(0.0, f["y"])
        f["width"] = min(float(w), f["x"] + f["width"]) - x0
        f["height"] = min(float(h), f["y"] + f["height"]) - y0
        f["x"], f["y"] = x0, y0
    faces = [f for f in faces if min(f["width"], f["height"]) >= min_size]
    faces.sort(key=lambda f: f["x"] + f["width"] / 2)
    for i, f in enumerate(faces, start=1):
        f["face"] = i
    return faces


def select(faces, rule="largest"):
    """'largest' | 'left' | 'right' | 'all' | face number(s) -> list of faces.

    largest, left and right pick one face with the rules crop_face() uses.
    """
    if not faces:
        return []
    if rule == "all":
        return list(faces)
    if rule == "largest":
        return [max(faces, key=lambda f: f["width"] * f["height"])]
    if rule == "left":
        return [min(faces, key=lambda f: f["x"])]
    if rule == "right":
        return [max(faces, key=lambda f: f["x"])]
    ids = {int(rule)} if np.isscalar(rule) else {int(i) for i in rule}
    return [f for f in faces if f["face"] in ids]


def crop(rgb, box, padding=0.25):
    """Crop a box from an RGB array, padded by a share of the box size."""
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    x, y, bw, bh = (float(v) for v in box)
    px, py = bw * padding, bh * padding
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(w, int(math.ceil(x + bw + px))), min(h, int(math.ceil(y + bh + py)))
    return rgb[y0:y1, x0:x1]


def draw(image, boxes, labels, colors, out_path, preview_side=800):
    """Draw face boxes with labels on an image and save it to out_path.

    boxes: [x, y, width, height] per face, in pixels of the EXIF-rotated image
    (as faces_table() reports them); labels: one string per face (may be
    empty); colors: one (red, green, blue) triple per face. Returns a copy no
    larger than preview_side pixels on its longer side, as an RGB array, for
    plotting.
    """
    import cv2
    bgr = cv2.cvtColor(np.asarray(load_image(image)), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    # Lines and text grow with the image, so they stay readable when shrunk
    thick = max(2, round(max(h, w) / 400))
    scale = max(0.5, max(h, w) / 1200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, color in zip(boxes, labels, colors):
        x, y, bw, bh = (int(round(float(v))) for v in box)
        bgr_color = tuple(int(c) for c in reversed(list(color)))
        cv2.rectangle(bgr, (x, y), (x + bw, y + bh), bgr_color, thick)
        if not label:
            continue
        (tw, th), base = cv2.getTextSize(label, font, scale, max(1, thick - 1))
        pad = thick
        # Above the box, or inside its top edge when there is no room
        top = y - th - base - 2 * pad if y - th - base - 2 * pad >= 0 else y
        cv2.rectangle(bgr, (x, top), (x + tw + 2 * pad, top + th + base + 2 * pad),
                      bgr_color, -1)
        cv2.putText(bgr, label, (x + pad, top + pad + th), font, scale, (255, 255, 255),
                    max(1, thick - 1), cv2.LINE_AA)
    if not cv2.imwrite(out_path, bgr):
        raise ValueError("Cannot write " + out_path)
    shrink = min(1.0, float(preview_side) / max(h, w))
    if shrink < 1:
        bgr = cv2.resize(bgr, (round(w * shrink), round(h * shrink)),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def display_copies(images, out_paths, max_side=1280, quality=85):
    """Save a JPEG copy of each image, at most max_side pixels on its longer side.

    The copies are EXIF-rotated like the images faces were detected in, so the
    boxes of detect_faces() line up with them once scaled. Returns the original
    (width, height) of each image.
    """
    from PIL import Image
    sizes = []
    for image, out_path in zip(images, out_paths):
        img = load_image(image)
        sizes.append(list(img.size))
        if max(img.size) > int(max_side):
            img.thumbnail((int(max_side), int(max_side)), Image.LANCZOS)
        img.save(out_path, "JPEG", quality=int(quality))
    return sizes


def iou(a, b):
    """Intersection over union of two face dicts."""
    ax1, ay1 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"] + b["width"], b["y"] + b["height"]
    iw = max(0.0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def track(frames, iou_threshold=0.3, max_gap=5):
    """Give every face of a sequence of frames a track_id (1, 2, ...).

    frames: one list of face dicts per sampled frame. Faces are matched greedily
    on IoU (highest first) to the last box of each track; a track not seen for
    more than max_gap sampled frames ends. Tracks are numbered from left to
    right by their mean centre, as faces are within an image.
    """
    tracks = []
    next_id = 1
    for t, faces in enumerate(frames):
        live = [tr for tr in tracks if t - tr["last_frame"] <= max_gap]
        pairs = sorted(((iou(tr["last"], f), ti, fi)
                        for ti, tr in enumerate(live) for fi, f in enumerate(faces)),
                       key=lambda p: p[0], reverse=True)
        used_t, used_f = set(), set()
        for score, ti, fi in pairs:
            if score < iou_threshold:
                break
            if ti in used_t or fi in used_f:
                continue
            used_t.add(ti)
            used_f.add(fi)
            faces[fi]["track_id"] = live[ti]["id"]
            live[ti].update(last=faces[fi], last_frame=t)
        for fi, f in enumerate(faces):
            if fi not in used_f:
                f["track_id"] = next_id
                tracks.append(dict(id=next_id, last=f, last_frame=t))
                next_id += 1
    centres = {}
    for faces in frames:
        for f in faces:
            centres.setdefault(f["track_id"], []).append(f["x"] + f["width"] / 2)
    order = sorted(centres, key=lambda k: np.mean(centres[k]))
    remap = {old: new for new, old in enumerate(order, start=1)}
    for faces in frames:
        for f in faces:
            f["track_id"] = remap[f["track_id"]]
    return frames


def _sface(model_path):
    import cv2
    key = ("sface", model_path)
    if key not in _models:
        _models[key] = cv2.FaceRecognizerSF.create(model_path, "")
    return _models[key]


def embed(rgb, faces, model_path):
    """One L2-normalised SFace embedding per face (None without landmarks).

    The 5 YuNet landmarks align a 112 x 112 crop (FaceRecognizerSF.alignCrop).
    The list runs parallel to `faces` and is never stored with them.
    """
    import cv2
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    rec = _sface(model_path)
    out = []
    for f in faces:
        lm = f.get("landmarks")
        if not lm or len(lm) != 10:
            out.append(None)
            continue
        row = np.array([f["x"], f["y"], f["width"], f["height"], *lm,
                        f.get("confidence") or 1.0], dtype=np.float32)
        v = rec.feature(rec.alignCrop(bgr, row)).ravel().astype(np.float64)
        n = np.linalg.norm(v)
        out.append(v / n if n > 0 else None)
    return out


def assign_person_ids(units, threshold=0.363):
    """Group units into people; return "p1", "p2", ... (or None) per unit.

    units: in media order, each a list of embeddings (one still-image face, or
    all faces of one video track; None entries are ignored). A unit is the
    mean of its embeddings. Clusters merge by average linkage on cosine
    similarity while the best pair is at least `threshold` (0.363 is the
    threshold SFace's authors suggest). Two faces in one frame may be the same
    person (mirrors, screens, posters, composites). Ids follow first appearance.
    """
    vecs, idx = [], []
    for i, unit in enumerate(units):
        members = [np.asarray(v, dtype=np.float64) for v in unit if v is not None]
        members = [v / np.linalg.norm(v) for v in members if np.linalg.norm(v) > 0]
        if not members:
            continue
        m = np.mean(members, axis=0)
        if np.linalg.norm(m) > 0:
            vecs.append(m / np.linalg.norm(m))
            idx.append(i)
    ids = [None] * len(units)
    if not vecs:
        return ids
    x = np.vstack(vecs)
    sim = x @ x.T
    size = np.ones(len(vecs))
    label = np.arange(len(vecs))
    alive = np.ones(len(vecs), dtype=bool)
    np.fill_diagonal(sim, -np.inf)
    while alive.sum() > 1:
        s = np.where(np.outer(alive, alive), sim, -np.inf)
        a, b = np.unravel_index(np.argmax(s), s.shape)
        if s[a, b] < threshold - 1e-9:
            break
        # Lance-Williams update for average linkage, then fold b into a
        sim[a, :] = (size[a] * sim[a, :] + size[b] * sim[b, :]) / (size[a] + size[b])
        sim[:, a] = sim[a, :]
        sim[a, a] = -np.inf
        size[a] += size[b]
        alive[b] = False
        label[label == b] = a
    names = {}
    for k, i in enumerate(idx):
        ids[i] = names.setdefault(label[k], "p%d" % (len(names) + 1))
    return ids


def _height(stream):
    """The stream's height in pixels from its resolution ("720p"), or None."""
    digits = "".join(ch for ch in (stream.resolution or "").split("p")[0] if ch.isdigit())
    return int(digits) if digits else None


def pick_youtube_stream(streams, max_height=720):
    """The stream download_youtube() saves, or None when there is none.

    The highest progressive MP4 (video with audio) if YouTube offers one.
    Otherwise the highest H.264 video-only MP4 no taller than max_height, or
    the lowest one if all are taller: only the frames are scored, so no audio
    is needed, and OpenCV may not decode the AV1 streams YouTube also offers.
    """
    streams = [s for s in streams if s.subtype == "mp4" and _height(s) is not None]
    progressive = [s for s in streams if s.is_progressive]
    if progressive:
        return max(progressive, key=_height)
    h264 = [s for s in streams if not s.is_progressive and s.includes_video_track
            and not s.includes_audio_track and (s.video_codec or "").startswith("avc1")]
    if not h264:
        return None
    low = [s for s in h264 if _height(s) <= max_height]
    return max(low, key=_height) if low else min(h264, key=_height)


def download_youtube(url, frame_dir, video_name):
    """Download a YouTube video to <frame_dir>/<video_name>.mp4; return its path.

    The file has no audio track when YouTube offers no progressive stream
    (see pick_youtube_stream()).
    """
    from pytubefix import YouTube  # optional dependency, added by the R functions
    yt = YouTube(url)
    video = None
    for _ in range(3):  # three attempts to get the video stream
        try:
            video = pick_youtube_stream(yt.streams)
            break
        except Exception:
            print("Error occurred while getting video stream. Retrying ... ")
    if video is None:
        raise ValueError("Failed to get an H.264 MP4 video stream for: " + url)
    video_path = os.path.join(frame_dir, video_name + ".mp4")
    print(f"Downloading video to {video_path}")
    for _ in range(3):
        try:
            video.download(output_path=frame_dir, filename="%s.mp4" % video_name)
            return video_path
        except Exception:
            print("Error occurred while downloading video. Retrying...")
    raise ValueError("Failed to download video after 3 attempts.")


def extract_frames(video_path, nframes, start=0, end=-1, uniform=False, ff=15,
                   frame_dir="temp/", video_name="temp"):
    """Save the sampled frames as <video_name>-frame-<i>.jpg.

    Returns one dict per saved frame: frame (the sample number i), frame_index
    (the frame's index in the video), time (seconds) and path. Frames that
    cannot be read are left out, so the list may be shorter than nframes.
    """
    import cv2
    nframes = int(nframes)
    if end == -1 and start == 0:
        cut = False
    else:
        if end <= start:
            raise ValueError("End time must be greater than start time.")
        if end < 0 or start < 0:
            raise ValueError("Start and end times must be positive.")
        cut = True
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: " + video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cut:
        first = int(start * fps)
        last = min(int(end * fps), total) if total > 0 else int(end * fps)
    else:
        first, last = 0, total
    num_frames = last - first
    if uniform:
        # Spread the frames evenly over the video (or the cut)
        n = min(nframes, num_frames)
        indices = [first + (i * num_frames) // n for i in range(n)] if n > 0 else []
    elif cut:
        step = 1 if num_frames < nframes else int(ff)
        indices = [i for i in range(first, last, step)][:nframes]
    else:
        if int(ff) * nframes > num_frames:
            raise ValueError("Video is too short for requested number of frames.")
        indices = [i * int(ff) for i in range(nframes)]
    print(f"Extracting {len(indices)} frames from {num_frames} total frames")
    frames = []
    for i, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {i}")
            continue
        path = os.path.join(frame_dir, f"{video_name}-frame-{i}.jpg")
        cv2.imwrite(path, frame)
        frames.append(dict(frame=i, frame_index=int(frame_index),
                           time=frame_index / fps if fps > 0 else float("nan"),
                           path=path))
    cap.release()
    print(f"Total number of saved frames: {len(frames)}")
    return frames


def faces_table(paths, detector="yunet", min_confidence=0.6, yunet_model=None,
                frames=None, iou_threshold=0.3, max_gap=5, sface_model=None,
                identity_threshold=0.363):
    """Detect faces in images, or in the frames of one video, as columns.

    paths: image paths or URLs, or the frame paths of one video when `frames`
    (the dicts extract_frames() returns) is given; faces are then tracked
    across frames. With sface_model, faces of the same person share a
    person_id across all images, or across all tracks of the video.

    Returns (faces, media): two dicts of equal-length lists. faces has one
    entry per face; media has one entry per image or frame (with the number of
    faces found). Missing numbers are NaN and missing strings are "".
    """
    per_image, sizes, embeddings = [], [], []
    for path in paths:
        rgb = np.asarray(load_image(path))
        faces = detect(rgb, detector, min_confidence, yunet_model)
        per_image.append(faces)
        sizes.append(rgb.shape[:2])
        embeddings.append(embed(rgb, faces, sface_model) if sface_model and faces else
                          [None] * len(faces))
    is_video = frames is not None
    if is_video:
        track(per_image, iou_threshold, max_gap)

    person = {}
    if sface_model:
        if is_video:
            ids = sorted({f["track_id"] for faces in per_image for f in faces})
            units = [[e for faces, embs in zip(per_image, embeddings)
                      for f, e in zip(faces, embs) if f["track_id"] == t] for t in ids]
            for t, p in zip(ids, assign_person_ids(units, identity_threshold)):
                person.update({("t", t): p})
        else:
            keys = [(i, f["face"]) for i, faces in enumerate(per_image) for f in faces]
            units = [[e] for embs in embeddings for e in embs]
            person.update(zip(keys, assign_person_ids(units, identity_threshold)))
    del embeddings  # biometric templates: never returned

    nan = float("nan")
    cols = ["item", "frame", "frame_index", "time", "face", "track_id", "person_id",
            "x", "y", "width", "height", "confidence", "image_width", "image_height"]
    cols += [f"{p}_{a}" for p in LANDMARKS for a in ("x", "y")]
    out = {c: [] for c in cols}
    media = {c: [] for c in ("item", "frame", "frame_index", "time",
                             "image_width", "image_height", "n_faces")}
    for i, (faces, (h, w)) in enumerate(zip(per_image, sizes)):
        meta = frames[i] if is_video else {}
        for key, val in (("item", i + 1), ("frame", meta.get("frame", nan)),
                         ("frame_index", meta.get("frame_index", nan)),
                         ("time", meta.get("time", nan)), ("image_width", w),
                         ("image_height", h), ("n_faces", len(faces))):
            media[key].append(val)
        for f in faces:
            pid = person.get(("t", f["track_id"]) if is_video else (i, f["face"]))
            row = dict(item=i + 1, frame=media["frame"][-1],
                       frame_index=media["frame_index"][-1], time=media["time"][-1],
                       face=f["face"], track_id=f.get("track_id", nan),
                       person_id=pid or "", x=f["x"], y=f["y"], width=f["width"],
                       height=f["height"],
                       confidence=nan if f["confidence"] is None else f["confidence"],
                       image_width=w, image_height=h)
            lm = f["landmarks"] or [nan] * 10
            for k, p in enumerate(LANDMARKS):
                row[f"{p}_x"], row[f"{p}_y"] = lm[2 * k], lm[2 * k + 1]
            for c in cols:
                out[c].append(row[c])
    return out, media
