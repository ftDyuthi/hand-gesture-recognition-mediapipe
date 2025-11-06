
import argparse
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Paths (edit if you move files)
# -----------------------------
MODEL_PATH = r"model.pth"
LABELS_TXT = r"top25_gloss_list.txt"

# -----------------------------
# Model (matches your VM arch)
# -----------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.dropout(self.relu(self.norm1(self.conv1(x))))
        out = self.dropout(self.relu(self.norm2(self.conv2(out))))
        return self.relu(out + (x if self.downsample is None else self.downsample(x)))


class SignLanguageModel(nn.Module):
    def __init__(self, input_size=477, hidden_size=2048, num_classes=25, dropout=0.2):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.temp_conv1 = TemporalBlock(hidden_size, hidden_size, dilation=1, dropout=dropout)
        self.temp_conv2 = TemporalBlock(hidden_size, hidden_size, dilation=2, dropout=dropout)
        self.temp_conv3 = TemporalBlock(hidden_size, hidden_size, dilation=4, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)

        self.gru = nn.GRU(
            hidden_size, hidden_size // 2,
            num_layers=2, batch_first=True,
            dropout=dropout, bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [B, T, D]
        x = self.input_proj(x)         # [B, T, H]
        x = x.transpose(1, 2)          # [B, H, T]
        x = self.temp_conv1(x)
        x = self.temp_conv2(x)
        x = self.temp_conv3(x)
        x = x.transpose(1, 2)          # [B, T, H]

        x = self.transformer(x)        # [B, T, H]
        x_gru, _ = self.gru(x)         # [B, T, H]

        x_attn, _ = self.attention(x_gru, x_gru, x_gru)
        x_attn = self.norm(x_attn + x_gru)

        avg = torch.mean(x_attn, dim=1)     # [B, H]
        mx, _ = torch.max(x_attn, dim=1)    # [B, H]
        pooled = torch.cat([avg, mx], dim=1)
        return self.classifier(pooled)       # [B, C]


# -----------------------------
# MediaPipe feature extraction
# -----------------------------
def build_holistic():
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True
    )
    return holistic

def landmarks_to_array(landmarks, expected_count):
    """Convert a list of normalized landmarks to a flat np.array of length expected_count*3,
       pad with zeros if missing."""
    arr = np.zeros((expected_count, 3), dtype=np.float32)
    if landmarks is None:
        return arr
    n = min(expected_count, len(landmarks))
    for i in range(n):
        lm = landmarks[i]
        arr[i] = [lm.x, lm.y, lm.z if hasattr(lm, "z") else 0.0]
    return arr

def extract_frame_features(holistic, frame_bgr, want_D):
    """
    Build a single-frame feature vector length 'want_D'.
    We concatenate:
      - Pose: 33
      - Left hand: 21
      - Right hand: 21
      - Face subset: 84 (first 84 landmarks)
    => total points = 33 + 21 + 21 + 84 = 159 points -> 159*3 = 477 dims (matches common setup)
    If your model input_dim != 477, we will pad/truncate to match.
    """
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)

    pose = landmarks_to_array(res.pose_landmarks.landmark if res.pose_landmarks else None, 33)
    lh   = landmarks_to_array(res.left_hand_landmarks.landmark if res.left_hand_landmarks else None, 21)
    rh   = landmarks_to_array(res.right_hand_landmarks.landmark if res.right_hand_landmarks else None, 21)

    # Face mesh 468 -> take first 84 for compactness (training likely used a subset to reach 477 dims)
    face_full = res.face_landmarks.landmark if res.face_landmarks else None
    if face_full is not None:
        face = landmarks_to_array(face_full, 84)
    else:
        face = np.zeros((84, 3), dtype=np.float32)

    feat = np.concatenate([pose, lh, rh, face], axis=0).reshape(-1)  # (159*3,) = 477 by default

    # Conform to model expected D
    if feat.shape[0] < want_D:
        feat = np.pad(feat, (0, want_D - feat.shape[0]))
    elif feat.shape[0] > want_D:
        feat = feat[:want_D]

    return feat  # (D,)

def normalize_sequence(seq):
    """Per-sequence z-score normalization: seq [T, D] -> [T, D]"""
    mu = seq.mean(axis=0, keepdims=True)
    sigma = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mu) / sigma

def uniform_resample(seq, target_len=64):
    """seq [T, D] -> [target_len, D] by uniform index sampling."""
    T = seq.shape[0]
    if T == target_len:
        return seq
    idx = np.linspace(0, max(T - 1, 1), num=target_len)
    idx = np.round(idx).astype(int)
    return seq[idx]


# -----------------------------
# Utilities
# -----------------------------
def load_labels_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def infer_dims_from_state_dict(state):
    """Return (input_dim, num_classes) using weight shapes in state_dict."""
    # input_proj.0.weight : [H, D]
    in_w = state.get("input_proj.0.weight", None)
    if in_w is None:
        # torch.save(model.state_dict()) should use these names. Try alternative:
        for k in state.keys():
            if k.endswith("input_proj.0.weight"):
                in_w = state[k]
                break
    if in_w is None:
        raise RuntimeError("Cannot find input projection weights in state_dict to infer input_dim.")
    input_dim = in_w.shape[1]

    # classifier[-1].weight : [C, 512]
    cls_w = None
    for tail in ["classifier.12.weight", "classifier.8.weight", "classifier.15.weight", "classifier.9.weight", "classifier.13.weight"]:
        if tail in state:
            cls_w = state[tail]
            break
    if cls_w is None:
        # fallback: find any key that matches pattern 'classifier.*.weight' with 2D tensor and take out_features
        for k, v in state.items():
            if k.startswith("classifier.") and v.ndim == 2:
                cls_w = v
    if cls_w is None:
        raise RuntimeError("Cannot find classifier final layer weights in state_dict to infer num_classes.")
    num_classes = cls_w.shape[0]

    return input_dim, num_classes


# -----------------------------
# Main realtime loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--win_len", type=int, default=64, help="Temporal window length")
    parser.add_argument("--step", type=int, default=2, help="Run inference every N frames")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence_smooth", type=int, default=10, help="N-best smoothing window")
    args = parser.parse_args()

    # Load labels
    labels = load_labels_from_txt(LABELS_TXT)
    # Load model weights
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Infer dims
    input_dim, num_classes = infer_dims_from_state_dict(state)
    if num_classes != len(labels):
        print(f"[!] Warning: num_classes ({num_classes}) != labels count ({len(labels)}). "
              f"Label order/file must match training mapping.")

    # Build model & load weights
    model = SignLanguageModel(input_size=input_dim, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[!] State dict keys adjusted:")
        if missing:    print("    Missing:", missing)
        if unexpected: print("    Unexpected:", unexpected)
    model.eval().to(args.device)

    # MediaPipe
    holistic = build_holistic()

    # Video
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Buffers
    seq = []  # list of frame feature vectors
    preds_hist = deque(maxlen=args.confidence_smooth)

    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_idx = 0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Extract features for this frame
            feat = extract_frame_features(holistic, frame, want_D=input_dim)  # (D,)
            seq.append(feat)
            if len(seq) > args.win_len:
                seq.pop(0)

            pred_label = ""
            pred_conf = 0.0

            # Inference every N frames when we have enough history
            if len(seq) == args.win_len and (frame_idx % args.step == 0):
                arr = np.stack(seq, axis=0).astype(np.float32)    # [T, D]
                arr = normalize_sequence(arr)
                arr = arr[None, ...]  # [1, T, D]
                tens = torch.from_numpy(arr).to(args.device)

                logits = model(tens)           # [1, C]
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                top = int(probs.argmax())
                pred_label = labels[top] if top < len(labels) else f"class_{top}"
                pred_conf = float(probs[top])

                preds_hist.append((top, pred_conf))

            # Smoothed prediction
            if len(preds_hist) > 0:
                counts = {}
                confs = {}
                for k, c in preds_hist:
                    counts[k] = counts.get(k, 0) + 1
                    confs[k] = max(confs.get(k, 0.0), c)
                top_sm = max(counts, key=lambda k: (counts[k], confs[k]))
                pred_label = labels[top_sm] if top_sm < len(labels) else f"class_{top_sm}"
                pred_conf = confs[top_sm]

            # UI overlay
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
            alpha = 0.35
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            txt = f"{pred_label}  ({pred_conf*100:.1f}%)" if pred_label else "…"
            cv2.putText(frame, "SQUIRL Realtime Sign", (18, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (18, 60), font, 0.9, (60, 240, 60), 2, cv2.LINE_AA)

            cv2.imshow("Realtime Sign Recognition", frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
