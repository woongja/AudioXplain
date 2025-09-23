import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances

# HuggingFace Whisper (word-level segmentation용)
from transformers import pipeline

# --------------------------
# Segmentation
# --------------------------
def segment_waveform_frames(waveform, sr=16000, frame_ms=100):
    """
    waveform을 frame_ms(ms) 단위로 segment 분할
    return: [(start_sample, end_sample), ...], seg_len
    """
    if isinstance(waveform, torch.Tensor):
        num_samples = waveform.shape[-1]
    else:
        num_samples = len(waveform)

    seg_len = int(sr * frame_ms / 1000)
    segments = []
    for start in range(0, num_samples, seg_len):
        end = min(start + seg_len, num_samples)
        segments.append((start, end))
    return segments, seg_len


def segment_waveform_words(audio_path, sr=16000, model_name="openai/whisper-small", device="cuda"):
    """
    Whisper 기반 단어 단위 segmentation
    return: [(start_sample, end_sample), ...], words(list)
    """
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=0 if device.startswith("cuda") else -1,
        return_timestamps="word",
        generate_kwargs={"task": "transcribe"}
    )
    result = asr(audio_path)

    segments, words = [], []
    for c in result.get("chunks", []):
        ts = c.get("timestamp", None)
        if not ts or ts[0] is None or ts[1] is None:
            continue
        start_s, end_s = float(ts[0]), float(ts[1])
        start = max(0, int(start_s * sr))
        end = max(start, int(end_s * sr))
        segments.append((start, end))
        words.append(c.get("text", "").strip())

    return segments, words


# --------------------------
# Masking
# --------------------------
def mask_waveform(waveform, segments, mask_idx):
    masked = waveform.clone()
    for idx in mask_idx:
        start, end = segments[idx]
        masked[0, start:end] = 0.0
    return masked


# --------------------------
# LIME 샘플 생성
# --------------------------
def generate_lime_samples(model, waveform, segments, num_samples=200, device="cpu", use_logit=True, p_mask=0.5):
    model.eval()
    with torch.no_grad():
        output = model(waveform.to(device))  # [1,2]
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        base_score = output[0, pred].item() if use_logit else prob[0, pred].item()

    print(f"[Baseline] pred={pred} ({'real' if pred==0 else 'spoof'}), score={base_score:.6f}")

    X, y = [], []
    S = len(segments)

    for _ in range(num_samples):
        mask_idx = [i for i in range(S) if np.random.rand() < p_mask]
        x_vec = np.ones(S, dtype=np.float32)
        if mask_idx:
            x_vec[mask_idx] = 0.0

        masked_wave = mask_waveform(waveform, segments, mask_idx).to(device)

        with torch.no_grad():
            out = model(masked_wave)
            score = out[0, pred].item() if use_logit else torch.softmax(out, dim=1)[0, pred].item()

        X.append(x_vec)
        y.append(score)

    return np.array(X), np.array(y), base_score, pred


# --------------------------
# Surrogate 학습
# --------------------------
def fit_weighted_surrogate(X, y, base_vec, sigma=0.25):
    D = cosine_distances(X, base_vec.reshape(1, -1)).ravel()
    W = np.exp(-(D**2) / (sigma**2))

    surrogate = Ridge(alpha=1.0, fit_intercept=True)
    surrogate.fit(X, y, sample_weight=W)
    return surrogate, W


# --------------------------
# Visualization
# --------------------------
def plot_waveform_with_importance(
    waveform, importances, sr=16000, segments=None, top_ratio=0.1, words=None, title="LIME Importance"
):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze().cpu().numpy()

    T = len(waveform)
    times = np.arange(T) / sr
    num_segments = len(importances)

    k = max(1, int(num_segments * top_ratio))
    top_idx = np.argsort(np.abs(importances))[-k:]

    plt.figure(figsize=(14, 4))
    plt.plot(times, waveform, color="black", linewidth=0.8)
    ymin, ymax = plt.ylim()

    if segments is not None:
        for idx in range(num_segments):
            start, end = segments[idx]
            s_t, e_t = start / sr, end / sr

            if idx in top_idx:
                plt.axvspan(s_t, e_t, color="red", alpha=0.3)

            if words and 0 <= idx < len(words):
                mid_t = (s_t + e_t) / 2
                plt.text(mid_t, ymin - 0.05 * (ymax - ymin), words[idx],
                         ha="center", va="top", fontsize=9)

    plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


# --------------------------
# Main pipeline
# --------------------------
def lime_explain(model, waveform, audio_path=None, mode="frame", sr=16000,
                 frame_ms=100, num_samples=200, device="cpu", top_ratio=0.1):
    """
    mode = "frame" | "word"
    """
    if mode == "frame":
        segments, seg_len = segment_waveform_frames(waveform, sr=sr, frame_ms=frame_ms)
        words = None
    elif mode == "word":
        if audio_path is None:
            raise ValueError("audio_path is required for word-level segmentation")
        segments, words = segment_waveform_words(audio_path, sr=sr, device=str(device))
        seg_len = None
    else:
        raise ValueError("mode must be 'frame' or 'word'")

    print(f"Total segments: {len(segments)}")

    X, y, base_score, pred = generate_lime_samples(
        model, waveform, segments, num_samples=num_samples, device=device
    )

    base_vec = np.ones(X.shape[1], dtype=np.float32)
    surrogate, W = fit_weighted_surrogate(X, y, base_vec, sigma=0.25)
    importances = surrogate.coef_

    plot_waveform_with_importance(
        waveform, importances, sr=sr, segments=segments,
        top_ratio=top_ratio, words=words,
        title=f"LIME Importance ({mode}-level)"
    )

    return importances, segments, (words if mode == "word" else None)
