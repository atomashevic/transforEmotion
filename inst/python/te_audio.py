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


def _mono_chunks(path, sampling_rate):
    """Yield the first audio stream of a file as mono float32 sample chunks."""
    import av
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream in: " + path)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sampling_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                yield out.to_ndarray().reshape(-1)
        for out in resampler.resample(None):  # flush the resampler
            yield out.to_ndarray().reshape(-1)


def load_windows(path, windows, sampling_rate=48000):
    """Decode the samples of each (start, stop) window, in seconds.

    The windows must be sorted and must not overlap; stop=None means the end
    of the audio. The file is decoded once, only the samples in the windows
    are kept, and decoding stops at the end of the last window. Returns one
    mono float32 array per window.
    """
    bounds = [(int(round(s * sampling_rate)),
               None if e is None else int(round(e * sampling_rate)))
              for s, e in windows]
    last = None if any(e is None for _, e in bounds) else max([e for _, e in bounds] + [0])
    kept = [[] for _ in bounds]
    seen = 0  # samples decoded so far
    i = 0     # first window that does not end before the current chunk
    if last != 0:
        for samples in _mono_chunks(path, sampling_rate):
            lo, hi = seen, seen + len(samples)
            seen = hi
            while i < len(bounds) and bounds[i][1] is not None and bounds[i][1] <= lo:
                i += 1
            j = i
            while j < len(bounds) and bounds[j][0] < hi:
                first, stop = bounds[j]
                a = max(first, lo) - lo
                b = (hi if stop is None else min(stop, hi)) - lo
                if b > a:
                    kept[j].append(samples[a:b])
                j += 1
            if last is not None and seen >= last:
                break
    return [np.concatenate(k).astype(np.float32, copy=False) if k
            else np.zeros(0, dtype=np.float32) for k in kept]


def load_audio(path, start=0.0, stop=None, sampling_rate=48000):
    """Decode the first audio stream of a file to mono float32 samples.

    Returns the samples from `start` to `stop` seconds (stop=None: to the end
    of the audio). Decoding stops once `stop` is reached.
    """
    return load_windows(path, [(start, stop)], sampling_rate)[0]


def audio_duration(path, sampling_rate=48000):
    """Length of the first audio stream of a file, in seconds.

    Taken from the file's metadata; files without it are decoded once and
    their samples counted, without keeping them.
    """
    import av
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream in: " + path)
        stream = container.streams.audio[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        if container.duration:
            return container.duration / av.time_base
    return sum(len(c) for c in _mono_chunks(path, sampling_rate)) / sampling_rate


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
    if uniform:
        # Plan the segments from the length of the audio, then decode only them,
        # so a long recording is never held in memory
        total = audio_duration(path, sampling_rate)
        duration = (total if stop is None else min(stop, total)) - start
        times = plan_segments(duration, nsegments, segment_length, uniform)
        segments = load_windows(path, [(start + s, start + e) for s, e in times],
                                sampling_rate)
        # The metadata length can be slightly off: cut segments at the real end
        kept = [((s, s + len(x) / sampling_rate), x)
                for (s, _), x in zip(times, segments) if len(x)]
        times = [t for t, _ in kept]
        segments = [x for _, x in kept]
    else:
        # Consecutive segments need only the first nsegments * segment_length seconds
        needed = start + int(nsegments) * float(segment_length)
        stop = needed if stop is None else min(stop, needed)
        samples = load_audio(path, start, stop, sampling_rate)
        times = plan_segments(len(samples) / sampling_rate, nsegments, segment_length)
        segments = [samples[int(round(s * sampling_rate)):int(round(e * sampling_rate))]
                    for s, e in times]
    if not times:
        raise ValueError("No audio between %g s and %s." %
                         (start, "the end" if end == -1 else "%g s" % float(end)))
    scores = score_segments(segments, labels, model_name, local_model_path, batch_size)
    return dict(start=[start + s for s, _ in times], end=[start + e for _, e in times],
                scores=scores)
