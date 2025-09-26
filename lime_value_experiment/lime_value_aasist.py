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
import json

sys.path.append('/home/woongjae/XAI')
from SSL_aasist.model import Model  # 🔥 AASIST 모델
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ------------------------------
# Logger
# ------------------------------
def setup_logger(log_path="run_lime.log"):
    logger = logging.getLogger("LIME")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 파일 로그만 남김
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 기존 handler 제거 후 다시 추가
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)

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
        if isinstance(out, tuple):
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
                num_samples=200, batch_size=32, logger=None,
                save_csv="per_file_results.csv", save_json="dataset_results.json"):

    # CSV, JSON 초기화
    if save_csv:
        with open(save_csv, "w") as f:
            f.write("file,segments,mean_spoof,mean_bonafide,mean_tr\n")
    if save_json:
        with open(save_json, "w") as f:
            json.dump([], f)

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

        vad_labels = load_vad_labels(vad_path)
        spoof_scores, bonafide_scores, tr_scores = [], [], []

        for idx, (start, end) in enumerate(segments):
            word_start, word_end = start/sr, end/sr
            assigned_label = assign_label_from_vad(word_start, word_end, vad_labels)
            if not assigned_label:
                continue
            if assigned_label in ["SS", "SN"]:
                spoof_scores.append(importances[idx])
            elif assigned_label in ["BS", "BN"]:
                bonafide_scores.append(importances[idx])
            elif assigned_label == "TR":
                tr_scores.append(importances[idx])

        file_result = {
            "file": fname,
            "segments": len(segments),
            "mean_spoof": float(np.mean(spoof_scores)) if spoof_scores else 0,
            "mean_bonafide": float(np.mean(bonafide_scores)) if bonafide_scores else 0,
            "mean_tr": float(np.mean(tr_scores)) if tr_scores else 0,
            "scores": {
                "spoof": [float(x) for x in spoof_scores],
                "bonafide": [float(x) for x in bonafide_scores],
                "tr": [float(x) for x in tr_scores]
            }
        }

        # ---- 실시간 저장 ----
        if save_csv:
            with open(save_csv, "a") as f:
                f.write(f"{file_result['file']},{file_result['segments']},"
                        f"{file_result['mean_spoof']},{file_result['mean_bonafide']},"
                        f"{file_result['mean_tr']}\n")

        if save_json:
            with open(save_json, "r+") as f:
                data = json.load(f)
                data.append(file_result)
                f.seek(0)
                json.dump(data, f, indent=4)

        if logger:
            logger.info(f"[DONE] {fname} → "
                        f"mean_spoof={file_result['mean_spoof']:.4f}, "
                        f"mean_bonafide={file_result['mean_bonafide']:.4f}, "
                        f"mean_tr={file_result['mean_tr']:.4f}")


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
    parser.add_argument("--log_path", type=str, default="run_lime.log")
    parser.add_argument("--save_csv", type=str, default="per_file_results.csv")
    parser.add_argument("--save_json", type=str, default="dataset_results.json")

    args = parser.parse_args()

    logger = setup_logger(args.log_path)
    logger.info("===== LIME Experiment Started (AASIST) =====")
    logger.info(f"Protocol: {args.protocol}")
    logger.info(f"Audio Root: {args.audio_root}")
    logger.info(f"VAD Dir: {args.vad_dir}")
    logger.info(f"Model: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- SSL-AASIST 모델 로드 ---
    model = Model(None, device)
    model = nn.DataParallel(model).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("✅ SSL-AASIST Model loaded")

    run_dataset(
        model,
        protocol_path=args.protocol,
        audio_root=args.audio_root,
        vad_dir=args.vad_dir,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        logger=logger,
        save_csv=args.save_csv,
        save_json=args.save_json
    )