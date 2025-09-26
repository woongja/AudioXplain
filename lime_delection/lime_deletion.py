import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import auc

# HuggingFace Whisper (word-level segmentation용)
from transformers import pipeline
from sklearn.metrics import auc

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm

import sys
sys.path.append('/home/woongjae/XAI')
from SSL_aasist.model import Model

import logging

def setup_logger(log_path="deletion_experiment.log"):
    logger = logging.getLogger("DeletionMetric")
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

def deletion_metric(model, waveform, segments, importances, target_class=None,
                    device="cpu", use_logit=False, plot=True):
    """
    Deletion metric: 중요도가 높은 순서대로 segment를 제거하며 모델 confidence 변화 추적
    Args:
        model: PyTorch 모델
        waveform: [1, T] torch waveform
        segments: [(start, end), ...]
        importances: LIME importances (길이 = len(segments))
        target_class: 분석할 class index (None이면 모델 baseline 예측 사용)
        device: cpu or cuda
        use_logit: True면 logit score 사용, False면 softmax prob 사용
        plot: 그래프 출력 여부
    Return:
        ratios: 제거 비율 리스트
        confidences: 각 단계 confidence
        deletion_auc: 면적 값 (낮을수록 설명력이 좋음)
    """
    model.eval()
    waveform = waveform.to(device)

    # baseline prediction
    with torch.no_grad():
        out = model(waveform)
        prob = torch.softmax(out, dim=1)
        if target_class is None:
            target_class = torch.argmax(prob, dim=1).item()
        base_score = out[0, target_class].item() if use_logit else prob[0, target_class].item()

    print(f"[Deletion] target_class={target_class}, baseline={base_score:.4f}")

    # 중요도 순 정렬 (내림차순)
    sorted_idx = np.argsort(np.abs(importances))[::-1]

    confidences = [base_score]
    ratios = [0.0]

    modified = waveform.clone()

    for step, idx in enumerate(sorted_idx, start=1):
        start, end = segments[idx]
        modified[0, start:end] = 0.0  # silence masking

        with torch.no_grad():
            out = model(modified)
            score = out[0, target_class].item() if use_logit else torch.softmax(out, dim=1)[0, target_class].item()

        confidences.append(score)
        ratios.append(step / len(sorted_idx))

    # AUC 계산
    deletion_auc = auc(ratios, confidences)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.fill_between(ratios, confidences, alpha=0.3)
        plt.plot(ratios, confidences, marker="o", linewidth=1)
        plt.xlabel("Segments removed")
        plt.ylabel(f"P[class={target_class}]")
        plt.title(f"Deletion AUC = {deletion_auc:.4f}")
        plt.tight_layout()
        plt.show()

    return ratios, confidences, deletion_auc

def deletion_metric(model, waveform, segments, importances,
                           target_class=None, device="cpu", use_logit=False,
                           plot=True, logger=None):
    """
    Deletion metric (binary classification, class flip 기준, logging 기록)
    """
    if logger is None:
        import logging
        logger = logging.getLogger("DeletionMetric")

    model.eval()
    waveform = waveform.to(device)

    # baseline prediction
    with torch.no_grad():
        out = model(waveform)
        prob = torch.softmax(out, dim=1)
        if target_class is None:
            target_class = torch.argmax(prob, dim=1).item()
        base_score = prob[0, target_class].item()
        logger.info(f"[Baseline] pred={target_class}, score={base_score:.4f}")

    sorted_idx = np.argsort(np.abs(importances))[::-1]

    confidences = [base_score]
    ratios = [0.0]

    modified = waveform.clone()
    stop_idx = len(sorted_idx)
    flipped_to = None

    for step, idx in enumerate(sorted_idx, start=1):
        start, end = segments[idx]
        modified[0, start:end] = 0.0

        with torch.no_grad():
            out = model(modified)
            prob_new = torch.softmax(out, dim=1)
            score = prob_new[0, target_class].item()
            pred_new = torch.argmax(prob_new, dim=1).item()

        confidences.append(score)
        ratios.append(step / len(sorted_idx))

        # 🔥 클래스 flip 감지
        if pred_new != target_class:
            stop_idx = step
            flipped_to = pred_new
            logger.info(
                f"[Flip Detected] step={step} ({ratios[-1]*100:.1f}% 제거) "
                f"원래={target_class} → 바뀐={flipped_to}, "
                f"confidence={score:.4f}"
            )
            break

    # cut-off까지 AUC 계산
    ratios_cut = ratios[:stop_idx+1]
    confidences_cut = confidences[:stop_idx+1]
    deletion_auc = auc(ratios_cut, confidences_cut)

    logger.info(
        f"[Result] AUC={deletion_auc:.4f}, stopped at {stop_idx}/{len(segments)}, "
        f"원래={target_class}, 바뀐={flipped_to}"
    )

    if plot:
        plt.figure(figsize=(6, 4))
        plt.fill_between(ratios_cut, confidences_cut, alpha=0.3)
        plt.plot(ratios_cut, confidences_cut, marker="o", linewidth=1, color="red")
        plt.xlabel("Segments removed")
        plt.ylabel(f"P[class={target_class}]")
        plt.title(f"Deletion AUC (cut-off) = {deletion_auc:.4f}")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    return ratios_cut, confidences_cut, deletion_auc, stop_idx, flipped_to
    

def run_dataset(model, dataset_path, protocol_path,
                           batch_size=1, sr=16000, frame_ms=500,
                           num_samples=200, device="cuda",
                           output_csv="deletion_results.csv",
                           log_file="deletion_experiment.log",
                           summary_plot="dataset_summary.png",
                           logger=None):
    """
    데이터셋 단위 Deletion Metric 실험 (logging + tqdm + summary plot)
    """
    if logger is None:
        import logging
        logger = logging.getLogger("DeletionMetric")

    model.eval()
    df = pd.read_csv(protocol_path, sep=" ", header=None, names=["file_path", "label"])
    results = []

    # tqdm으로 전체 파일 진행 표시
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        file_path = os.path.join(dataset_path, row["file_path"])
        label = row["label"]

        try:
            wav, _ = librosa.load(file_path, sr=sr)
            wav_tensor = torch.tensor(wav).float().unsqueeze(0).to(device)

            # Segmentation
            segments, _ = segment_waveform_frames(wav_tensor, sr=sr, frame_ms=frame_ms)

            # LIME importances
            importances, _, _ = lime_explain(
                model, wav_tensor, audio_path=file_path,
                mode="frame", sr=sr, frame_ms=frame_ms,
                num_samples=num_samples, device=device, top_ratio=0.1
            )

            # Deletion metric 실행
            ratios, confs, auc_val, stop_idx, flipped_to = deletion_metric(
                model, wav_tensor, segments, importances,
                device=device, use_logit=False, plot=False, logger=logger
            )

            stop_ratio = stop_idx / len(segments)

            results.append({
                "file": file_path,
                "label": label,
                "auc": auc_val,
                "stop_idx": stop_idx,
                "stop_ratio": stop_ratio,
                "flipped_to": flipped_to
            })

            logger.info(
                f"[Sample Done] file={file_path}, label={label}, "
                f"AUC={auc_val:.4f}, stop_idx={stop_idx}, stop_ratio={stop_ratio:.3f}, "
                f"flipped_to={flipped_to}"
            )

        except Exception as e:
            logger.error(f"[Error] file={file_path}: {e}")

    # CSV 저장
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)

    # 전체 통계 로그
    logger.info("\n=== Dataset Summary ===")
    logger.info(f"총 샘플 수: {len(res_df)}")
    logger.info(f"평균 AUC: {res_df['auc'].mean():.4f}")
    logger.info(f"평균 stop_ratio: {res_df['stop_ratio'].mean():.4f}")
    logger.info(f"\n{res_df.describe()}")

    # 최종 summary plot 저장
    plt.figure(figsize=(8, 5))
    plt.hist(res_df["stop_ratio"], bins=20, alpha=0.7, label="Stop Ratio", color="blue")
    plt.xlabel("Stop Ratio (제거 비율)")
    plt.ylabel("Count")
    plt.title("Distribution of Stop Ratios across Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(summary_plot)
    plt.close()

    logger.info(f"[Summary Plot Saved] {summary_plot}")

    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--protocol_path", type=str, required=True, help="Protocol file path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--frame_ms", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_csv", type=str, default="deletion_results.csv")
    parser.add_argument("--log_file", type=str, default="deletion_experiment.log")
    parser.add_argument("--summary_plot", type=str, default="dataset_summary.png")

    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "/home/woongjae/XAI/SSL_aasist/Best_LA_model_for_DF.pth"
    model = Model(None, device)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    run_dataset(
        model=model,
        dataset_path=args.dataset_path,
        protocol_path=args.protocol_path,
        batch_size=args.batch_size,
        sr=args.sr,
        frame_ms=args.frame_ms,
        num_samples=args.num_samples,
        device=device,
        output_csv=args.output_csv,
        summary_plot=args.summary_plot,
        log_file=args.log_file,
        logger=logger
    )