import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
import logging
import sys
sys.path.append('/home/woongjae/AudioXplain')
from tcm_add.model import Model
import json

# ------------------------------
# Logger
# ------------------------------
def setup_logger(log_path="run_lime_rcq.log"):
    logger = logging.getLogger("LIME_RCQ")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

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
    model.eval()
    with torch.no_grad():
        out = model(waveform.to(device))
        if isinstance(out, tuple):  # (output, embedding) 구조일 수도 있음
            output = out[0]
        else:
            output = out
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

        batch_tensor = torch.cat(batch_masks, dim=0).to(device)

        with torch.no_grad():
            out = model(batch_tensor)
            if isinstance(out, tuple):
                out = out[0]
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
# 3. VAD 기반 GT Interval Matching
# ------------------------------
label_mapping = {
    0: 'BN', 1: 'BS',
    2: 'SS', 3: 'SS', 4: 'SS', 5: 'SS', 6: 'SS', 7: 'SS', 8: 'SS', 9: 'SS', 10: 'SS',
    11: 'SS', 12: 'SS', 13: 'SS', 14: 'SS', 15: 'SS', 16: 'SS', 17: 'SS', 18: 'SN',
    19: 'SS', 20: 'SS',
    100: 'TR',
    101: 'BN',
    102: 'SN', 103: 'SN', 104: 'SN', 105: 'SN', 106: 'SN', 107: 'SN', 108: 'SN', 
    109: 'SN', 110: 'SN', 111: 'SN', 112: 'SN', 113: 'SN', 114: 'SN', 115: 'SN', 
    116: 'SN', 117: 'SN', 118: 'SN', 119: 'SN', 120: 'SN'
}

def load_vad_labels(vad_file_path):
    vad_labels = []
    with open(vad_file_path, 'r') as f:
        for line in f:
            start_time, end_time, label_id = line.strip().split()
            label_id = int(label_id)
            label_name = label_mapping.get(label_id, None)
            if label_name:
                vad_labels.append((float(start_time), float(end_time), label_name))
    return vad_labels

def assign_label_from_vad(word_start, word_end, vad_labels):
    overlaps = {}
    for (start, end, label_name) in vad_labels:
        overlap = max(0, min(word_end, end) - max(word_start, start))
        if overlap > 0:
            overlaps[label_name] = overlaps.get(label_name, 0) + overlap
    if not overlaps:
        return None
    return max(overlaps, key=overlaps.get)


# ------------------------------
# 4. Main Dataset Runner
# ------------------------------
def run_dataset(model, protocol_path, audio_root, vad_dir, device="cuda",
                num_samples=200, batch_size=32, logger=None, save_csv="per_file_results.csv", save_json="dataset_results.json"):

    dataset_spoof_scores = []
    dataset_bonafide_scores = []
    dataset_tr_scores = []
    results_per_file = []

    with open(protocol_path, "r") as f:
        file_list = [line.strip().split()[0] for line in f]

    for fname in tqdm(file_list, desc="Processing files", unit="file"):
        audio_path = os.path.join(audio_root, fname)
        vad_path = os.path.join(vad_dir, f"{os.path.splitext(fname)[0]}.vad")

        if not os.path.exists(audio_path) or not os.path.exists(vad_path):
            if logger:
                logger.warning(f"Missing file: {audio_path} or {vad_path}")
            continue

        waveform, sr = librosa.load(audio_path, sr=16000)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)

        importances, segments, words, pred = lime_explain(
            model, waveform, audio_path=audio_path,
            num_samples=num_samples, batch_size=batch_size, device=device
        )

        if pred != 1:
            if logger:
                logger.info(f"[SKIP] {fname} → pred=bonafide")
            continue

        vad_labels = load_vad_labels(vad_path)

        spoof_scores, bonafide_scores, tr_scores = [], [], []

        for idx, (start, end) in enumerate(segments):
            word_start, word_end = start/sr, end/sr
            assigned_label = assign_label_from_vad(word_start, word_end, vad_labels)
            if not assigned_label:
                continue

            if assigned_label in ["SS", "SN"]:
                spoof_scores.append(importances[idx])
                dataset_spoof_scores.append(importances[idx])
            elif assigned_label in ["BS", "BN"]:
                bonafide_scores.append(importances[idx])
                dataset_bonafide_scores.append(importances[idx])
            elif assigned_label == "TR":
                tr_scores.append(importances[idx])
                dataset_tr_scores.append(importances[idx])

        mean_spoof_sample = np.mean(spoof_scores) if spoof_scores else 0
        mean_bonafide_sample = np.mean(bonafide_scores) if bonafide_scores else 0
        mean_tr_sample = np.mean(tr_scores) if tr_scores else 0

        results_per_file.append({
            "file": fname,
            "segments": len(segments),
            "mean_spoof": mean_spoof_sample,
            "mean_bonafide": mean_bonafide_sample,
            "mean_tr": mean_tr_sample
        })

        if logger:
            logger.info(
                f"[DONE] {fname} (segments={len(segments)}, pred=spoof, "
                f"mean_spoof={mean_spoof_sample:.4f}, "
                f"mean_bonafide={mean_bonafide_sample:.4f}, "
                f"mean_tr={mean_tr_sample:.4f})"
            )

    mean_spoof = np.mean(dataset_spoof_scores) if dataset_spoof_scores else 0
    mean_bonafide = np.mean(dataset_bonafide_scores) if dataset_bonafide_scores else 0
    mean_tr = np.mean(dataset_tr_scores) if dataset_tr_scores else 0

    all_scores = dataset_spoof_scores + dataset_bonafide_scores + dataset_tr_scores
    overall_mean = np.mean(all_scores) if all_scores else 0

    rcq_spoof = (mean_spoof - overall_mean) / abs(overall_mean) * 100 if overall_mean != 0 else 0
    rcq_bonafide = (mean_bonafide - overall_mean) / abs(overall_mean) * 100 if overall_mean != 0 else 0
    rcq_tr = (mean_tr - overall_mean) / abs(overall_mean) * 100 if overall_mean != 0 else 0

    results = {
        "mean_spoof": mean_spoof,
        "mean_bonafide": mean_bonafide,
        "mean_tr": mean_tr,
        "rcq_spoof": rcq_spoof,
        "rcq_bonafide": rcq_bonafide,
        "rcq_tr": rcq_tr,
        "count_spoof_words": len(dataset_spoof_scores),
        "count_bonafide_words": len(dataset_bonafide_scores),
        "count_tr_words": len(dataset_tr_scores)
    }

    if save_csv:
        df = pd.DataFrame(results_per_file)
        df.to_csv(save_csv, index=False)
        if logger:
            logger.info(f"Per-file results saved to {save_csv}")

    if save_json:
        with open(save_json, "w") as f:
            json.dump(results, f, indent=4)
        if logger:
            logger.info(f"Dataset-level results saved to {save_json}")

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
    parser.add_argument("--vad_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_path", type=str, default="run_lime_rcq.log")
    parser.add_argument("--save_csv", type=str, default="per_file_results.csv")
    parser.add_argument("--save_json", type=str, default="dataset_results.json")

    # === 여기를 args = parser.parse_args() "바로 위"에 추가 ===
    parser.add_argument("--emb_size", type=int, default=144, help="embedding size for Conformer/W2V head")
    parser.add_argument("--num_encoders", type=int, default=4, help="number of Conformer encoder blocks")
    parser.add_argument("--heads", type=int, default=4, help="multi-head attention heads")
    parser.add_argument("--kernel_size", type=int, default=31, help="Conv module kernel size in Conformer")


    args = parser.parse_args()

    logger = setup_logger(args.log_path)
    logger.info("===== LIME RCQ Experiment Started =====")
    logger.info(f"Protocol: {args.protocol}")
    logger.info(f"Audio Root: {args.audio_root}")
    logger.info(f"VAD Dir: {args.vad_dir}")
    logger.info(f"Model: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(args, device).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("✅ Model loaded")

    results = run_dataset(
        model,
        protocol_path=args.protocol,
        audio_root=args.audio_root,
        vad_dir=args.vad_dir,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        logger=logger,
        save_csv=args.save_csv
    )
    logger.info("===== Experiment Finished =====")