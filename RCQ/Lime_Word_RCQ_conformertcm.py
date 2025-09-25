import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm
from transformers import pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
import logging
import sys
sys.path.append('/home/woongjae/AudioXplain')
from tcm_add.model import Model
from tqdm import tqdm

def setup_logger(log_path="run_lime_rcq.log"):
    logger = logging.getLogger("LIME_RCQ")
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Stream handler (console)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # 중복 방지
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger

# ------------------------------
# 1. Whisper 기반 Word Segmentation
# ------------------------------
def segment_waveform_words(audio_path, sr=16000, model_name="openai/whisper-small", device="cuda"):
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


# ------------------------------
# 2. LIME Utility
# ------------------------------
def mask_waveform(waveform, segments, mask_idx):
    masked = waveform.clone()
    for idx in mask_idx:
        start, end = segments[idx]
        masked[0, start:end] = 0.0
    return masked


def generate_lime_samples(model, waveform, segments, num_samples=200, device="cpu",
                          use_logit=True, p_mask=0.5, batch_size=32):
    """
    LIME 샘플 생성 (배치 처리 가능)
    """
    model.eval()
    with torch.no_grad():
        output = model(waveform.to(device))
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        base_score = output[0, pred].item() if use_logit else prob[0, pred].item()

    X, y = [], []
    S = len(segments)

    for start_idx in range(0, num_samples, batch_size):
        cur_bs = min(batch_size, num_samples - start_idx)
        batch_masks, batch_vecs = [], []

        for _ in range(cur_bs):
            mask_idx = [i for i in range(S) if np.random.rand() < p_mask]
            x_vec = np.ones(S, dtype=np.float32)
            if mask_idx:
                x_vec[mask_idx] = 0.0
            batch_vecs.append(x_vec)

            masked_wave = mask_waveform(waveform.clone(), segments, mask_idx)
            batch_masks.append(masked_wave)

        batch_tensor = torch.cat(batch_masks, dim=0).to(device)  # [B, 1, T]

        with torch.no_grad():
            out = model(batch_tensor)  # [B, num_classes]
            if use_logit:
                scores = out[:, pred].cpu().numpy()
            else:
                scores = torch.softmax(out, dim=1)[:, pred].cpu().numpy()

        X.extend(batch_vecs)
        y.extend(scores.tolist())

    return np.array(X), np.array(y), base_score, pred


def fit_weighted_surrogate(X, y, base_vec, sigma=0.25):
    D = cosine_distances(X, base_vec.reshape(1, -1)).ravel()
    W = np.exp(-(D**2) / (sigma**2))
    surrogate = Ridge(alpha=1.0, fit_intercept=True)
    surrogate.fit(X, y, sample_weight=W)
    return surrogate


def lime_explain(model, waveform, audio_path, sr=16000,
                 num_samples=200, batch_size=32, device="cpu"):
    segments, words = segment_waveform_words(audio_path, sr=sr, device=str(device))

    X, y, base_score, pred = generate_lime_samples(
        model, waveform, segments,
        num_samples=num_samples, batch_size=batch_size, device=device
    )

    base_vec = np.ones(X.shape[1], dtype=np.float32)
    surrogate = fit_weighted_surrogate(X, y, base_vec, sigma=0.25)
    importances = surrogate.coef_
    return importances, segments, words, pred


# ------------------------------
# 3. GT Interval Matching
# ------------------------------
def assign_label(word_start, word_end, gt_intervals):
    overlaps = []
    for label, start, end in gt_intervals:
        overlap = max(0, min(word_end, end) - max(word_start, start))
        overlaps.append((label, overlap))
    if not overlaps:
        return None
    return max(overlaps, key=lambda x: x[1])[0]


# ------------------------------
# 4. Main Dataset Runner
# ------------------------------
def run_dataset(model, protocol_path, audio_root, device="cuda",
                num_samples=200, batch_size=32, logger=None):

    dataset_spoof_scores = []
    dataset_bonafide_scores = []

    # 프로토콜 로드
    gt_dict = {}
    with open(protocol_path, "r") as f:
        for line in f:
            fname, subset, label = line.strip().split()
            audio_path = os.path.join(audio_root, fname)
            if label == "spoof":
                gt_dict[fname] = [("spoof", 0.0, 9999.0)]
            else:
                gt_dict[fname] = [("bonafide", 0.0, 9999.0)]

    # tqdm으로 진행률 표시
    for fname, intervals in tqdm(gt_dict.items(), desc="Processing files", unit="file"):
        audio_path = os.path.join(audio_root, fname)
        if not os.path.exists(audio_path):
            if logger:
                logger.warning(f"File not found: {audio_path}")
            continue

        waveform, sr = librosa.load(audio_path, sr=16000)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)

        importances, segments, words, pred = lime_explain(
            model, waveform, audio_path=audio_path,
            num_samples=num_samples, batch_size=batch_size, device=device
        )

        if pred != 1:  # spoof로 탐지한 경우만
            if logger:
                logger.info(f"[SKIP] {fname} → pred=bonafide")
            continue

        for idx, (start, end) in enumerate(segments):
            word_start, word_end = start/sr, end/sr
            label = assign_label(word_start, word_end, intervals)
            if label == "spoof":
                dataset_spoof_scores.append(importances[idx])
            elif label == "bonafide":
                dataset_bonafide_scores.append(importances[idx])

        if logger:
            logger.info(f"[DONE] {fname} (segments={len(segments)}, pred=spoof)")

    # 평균/RCQ 계산
    mean_spoof = np.mean(dataset_spoof_scores) if dataset_spoof_scores else 0
    mean_bonafide = np.mean(dataset_bonafide_scores) if dataset_bonafide_scores else 0
    all_scores = dataset_spoof_scores + dataset_bonafide_scores
    overall_mean = np.mean(all_scores) if all_scores else 0

    rcq_spoof = (mean_spoof - overall_mean) / abs(overall_mean) * 100 if overall_mean != 0 else 0
    rcq_bonafide = (mean_bonafide - overall_mean) / abs(overall_mean) * 100 if overall_mean != 0 else 0

    results = {
        "mean_spoof": mean_spoof,
        "mean_bonafide": mean_bonafide,
        "rcq_spoof": rcq_spoof,
        "rcq_bonafide": rcq_bonafide,
        "count_spoof_words": len(dataset_spoof_scores),
        "count_bonafide_words": len(dataset_bonafide_scores)
    }

    if logger:
        logger.info("===== Dataset Results =====")
        for k, v in results.items():
            logger.info(f"{k}: {v}")

    return results


# ------------------------------
# 5. CLI Entry
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, required=True)
    parser.add_argument("--audio_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)   # 추가됨
    parser.add_argument("--log_path", type=str, default="run_lime_rcq.log")
    args = parser.parse_args()

    # Logger 설정
    logger = setup_logger(args.log_path)
    logger.info("===== LIME RCQ Experiment Started =====")
    logger.info(f"Protocol: {args.protocol}")
    logger.info(f"Audio Root: {args.audio_root}")
    logger.info(f"Model: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(None, device)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    logger.info("✅ Model loaded")

    results = run_dataset(
        model,
        protocol_path=args.protocol,
        audio_root=args.audio_root,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        logger=logger
    )