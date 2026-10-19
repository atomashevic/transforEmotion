"""Speech emotion scores with a CLAP audio-text model.

Used by speech_scores() in R. The audio track of an audio or video file is
decoded with PyAV (which bundles FFmpeg), split into segments and each
segment is scored against the user's classes with CLAP: the softmax over the
classes of the scaled cosine similarity between the audio embedding and each
class's text embedding.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

# CLAP models loaded in this session, by (model id, local path)
_models = {}


def load_audio(path, start=0.0, stop=None, sampling_rate=48000):
    """Decode the first audio stream of a file to mono float32 samples.

    Returns the samples from `start` to `stop` seconds (stop=None: to the end
    of the audio). Decoding stops once `stop` is reached.
    """
    import av
    first = int(round(start * sampling_rate))
    last = None if stop is None else int(round(stop * sampling_rate))
    chunks = []
    seen = 0  # samples decoded so far

    def keep(frames):
        nonlocal seen
        for out in frames:
            samples = out.to_ndarray().reshape(-1)
            lo, hi = seen, seen + len(samples)
            seen = hi
            a = max(first, lo) - lo
            b = (hi if last is None else min(last, hi)) - lo
            if b > a:
                chunks.append(samples[a:b])
        return last is not None and seen >= last

    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream in: " + path)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sampling_rate)
        done = False
        for frame in container.decode(stream):
            if keep(resampler.resample(frame)):
                done = True
                break
        if not done:
            keep(resampler.resample(None))  # flush the resampler
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def download_youtube_audio(url, out_dir, name):
    """Download the audio of a YouTube video to out_dir; return its path."""
    from pytubefix import YouTube  # optional dependency, added by the R function
    error = None
    for _ in range(3):  # three attempts
        try:
            stream = YouTube(url).streams.get_audio_only()
            if stream is None:
                raise ValueError("No audio stream for: " + url)
            return stream.download(output_path=out_dir,
                                   filename="%s.%s" % (name, stream.subtype))
        except Exception as e:
            error = e
    raise ValueError("Failed to download the audio after 3 attempts: %s" % error)


def plan_segments(duration, nsegments, segment_length, uniform=False, min_length=1.0):
    """Segment (start, end) times in seconds within audio of `duration` seconds.

    uniform=False: consecutive segments from the start, at most nsegments; a
    last segment shorter than min_length seconds is dropped unless it is the
    only one. uniform=True: at most nsegments segments of segment_length
    spread evenly over the audio, without overlap. Audio no longer than
    segment_length is one segment.
    """
    n = int(nsegments)
    length = float(segment_length)
    if duration <= 0 or n < 1:
        return []
    if duration <= length:
        return [(0.0, float(duration))]
    if uniform:
        k = min(n, int(duration // length))
        if k <= 1:
            return [(0.0, length)]
        step = (duration - length) / (k - 1)
        return [(i * step, i * step + length) for i in range(k)]
    segments = []
    for i in range(n):
        s = i * length
        if s >= duration:
            break
        e = min(s + length, duration)
        if e - s < min_length and segments:
            break
        segments.append((s, e))
    return segments


def _clap(model_name, local_model_path=None):
    import torch
    from transformers import ClapModel, ClapProcessor
    key = (model_name, local_model_path)
    if key not in _models:
        source = local_model_path or model_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = ClapProcessor.from_pretrained(source)
        model = ClapModel.from_pretrained(source).to(device).eval()
        _models[key] = (model, processor, device)
    return _models[key]


def score_segments(segments, labels, model_name, local_model_path=None, batch_size=8):
    """Class probabilities for each segment (an array of 48 kHz samples).

    Returns an array with one row per segment and one column per label.
    """
    import torch
    import torch.nn.functional as F
    model, processor, device = _clap(model_name, local_model_path)
    sampling_rate = processor.feature_extractor.sampling_rate
    rows = []
    with torch.no_grad():
        text = processor.tokenizer(list(labels), padding=True, return_tensors="pt")
        text = {k: v.to(device) for k, v in text.items()}
        text_embeds = F.normalize(model.get_text_features(**text), dim=-1)
        scale = model.logit_scale_a.exp()
        for i in range(0, len(segments), int(batch_size)):
            batch = [np.asarray(s, dtype=np.float32) for s in segments[i:i + int(batch_size)]]
            features = processor.feature_extractor(batch, sampling_rate=sampling_rate,
                                                   return_tensors="pt")
            features = {k: v.to(device) for k, v in features.items()}
            audio_embeds = F.normalize(model.get_audio_features(**features), dim=-1)
            logits = scale * audio_embeds @ text_embeds.T
            rows.append(logits.softmax(dim=-1).cpu().numpy())
    if not rows:
        return np.zeros((0, len(labels)))
    return np.concatenate(rows).astype(np.float64)


def speech_analyze(path, labels, nsegments=100, segment_length=5.0, start=0.0, end=-1,
                   uniform=False, model_name="laion/larger_clap_music_and_speech",
                   local_model_path=None, batch_size=8):
    """Score segments of the audio track of a file against labels.

    end=-1 means the end of the audio. Returns a dict with the segment start
    and end times (seconds in the file) and the scores (one row per segment).
    """
    _, processor, _ = _clap(model_name, local_model_path)
    sampling_rate = processor.feature_extractor.sampling_rate
    start = float(start)
    stop = None if end == -1 else float(end)
    if not uniform:
        # Consecutive segments need only the first nsegments * segment_length seconds
        needed = start + int(nsegments) * float(segment_length)
        stop = needed if stop is None else min(stop, needed)
    samples = load_audio(path, start, stop, sampling_rate)
    duration = len(samples) / sampling_rate
    if duration <= 0:
        raise ValueError("No audio between %g s and %s." %
                         (start, "the end" if end == -1 else "%g s" % float(end)))
    times = plan_segments(duration, nsegments, segment_length, uniform)
    segments = [samples[int(round(s * sampling_rate)):int(round(e * sampling_rate))]
                for s, e in times]
    scores = score_segments(segments, labels, model_name, local_model_path, batch_size)
    return dict(start=[start + s for s, _ in times], end=[start + e for _, e in times],
                scores=scores)
