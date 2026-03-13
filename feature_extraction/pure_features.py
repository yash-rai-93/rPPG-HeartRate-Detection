import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import entropy
import joblib
from glob import glob
import gc
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. CONFIGURATION
# ==========================================
PATH_PURE = "/content/drive/MyDrive/PURE"
LOG_FILE = "/content/drive/MyDrive/output-pure-extracted.txt"
SAVE_DIR = "/content/drive/MyDrive/PRISM_RESULTS_PURE_FINAL/"
CHECKPOINT_FILE = os.path.join(SAVE_DIR, "prism_features_pure_v3.csv")

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Kalman Filter for stabilized ROI
class FaceKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 4)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
    def update(self, x, y, w, h):
        measured = np.array([[np.float32(x)],[np.float32(y)],[np.float32(w)],[np.float32(h)]])
        self.kf.correct(measured)
        return self.kf.predict().flatten().astype(int)

# ==========================================
# 2. FEATURE EXTRACTOR
# ==========================================
def process_single_frame_v3(frame, prev_gray_full, kf):
    h_f, w_f, _ = frame.shape
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        raw_x, raw_y, raw_w, raw_h = w_f//4, h_f//4, w_f//2, h_f//2
    else:
        lms = result.face_landmarks[0]
        pts_x = [int(lm.x * w_f) for lm in lms]
        pts_y = [int(lm.y * h_f) for lm in lms]
        raw_x, raw_y = max(0, min(pts_x)), max(0, min(pts_y))
        raw_w, raw_h = min(w_f-raw_x, max(pts_x)-raw_x), min(h_f-raw_y, max(pts_y)-raw_y)

    x, y, w, h = kf.update(raw_x, raw_y, raw_w, raw_h)
    x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)

    face_roi = frame[y:y+h, x:x+w]
    face_g = face_roi[:, :, 1]
    face_gray = gray_full[y:y+h, x:x+w]

    phi = np.mean(face_g) / 255.0
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    mu = np.mean(ycrcb[:,:,0][mask > 0]) / 255.0 if np.sum(mask) > 0 else 0.5
    sigma = np.std(face_g[mask > 0]) / 50.0 if np.sum(mask) > 0 else np.std(face_g) / 50.0
    chi = min(cv2.Laplacian(face_gray, cv2.CV_64F).var(), 1000.0) / 1000.0
    hist, _ = np.histogram(face_g.flatten(), bins=20, density=True)
    H = entropy(hist + 1e-10) / 5.0
    clip = (np.sum(face_g > 250) + np.sum(face_g < 5)) / face_g.size

    m_val = 0.0
    if prev_gray_full is not None:
        try:
            prev_face = prev_gray_full[y:y+h, x:x+w]
            flow = cv2.calcOpticalFlowFarneback(prev_face, face_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            m_val = np.mean(np.linalg.norm(flow, axis=2))
        except: pass

    res = (w * h) / (w_f * h_f)
    f_fft = np.fft.fft2(gray_full)
    mag = 20 * np.log(np.abs(np.fft.fftshift(f_fft)) + 1e-10)
    cy, cx = h_f//2, w_f//2
    ghost = np.mean(mag[cy-10:cy+10, cx-10:cx+10]) / (np.mean(mag) + 1e-5)

    return [phi, sigma, mu, chi, H, clip, m_val, res, np.mean(face_g), ghost], gray_full

def extract_prism_features_stream(path):
    kf = FaceKalman()
    accumulators = {i: [] for i in range(10)}
    prev_gray = None
    images = sorted(glob(os.path.join(path, "*.png")))
    if not images:
        sub = os.path.basename(path)
        images = sorted(glob(os.path.join(path, sub, "*.png")))
    if not images: return np.zeros(11)

    step = max(1, len(images) // 300)
    for i in range(0, min(len(images), step*300), step):
        frame = cv2.imread(images[i])
        if frame is None: continue
        stats, prev_gray = process_single_frame_v3(frame, prev_gray, kf)
        for idx, val in enumerate(stats): accumulators[idx].append(val)

    phi, sigma, mu, chi, H, clip = [np.mean(accumulators[i]) for i in range(6)]
    M_norm = min(np.percentile(accumulators[6], 95), 5.0) / 5.0
    res = np.mean(accumulators[7])
    means = np.array(accumulators[8])
    flicker = np.std(means) / (np.mean(means) + 1e-6)
    tau = 0.0
    if np.std(means) > 1e-5:
        norm = (means - np.mean(means)) / np.std(means)
        if len(norm) > 1: tau = max(0.0, np.corrcoef(norm[:-1], norm[1:])[0,1])
    ghost = min(np.mean(accumulators[9]), 5.0) / 5.0
    return np.array([phi, sigma, mu, chi, H, clip, M_norm, res, flicker, tau, ghost], dtype=np.float32)

# ==========================================
# 3. TRAINING CORE
# ==========================================
class HeartGoldMicroMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 16), nn.ReLU(), nn.BatchNorm1d(16),
            nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, num_classes)
        )
    def forward(self, x): return self.net(x)

def get_winner_data(log_path):
    df = pd.read_csv(log_path, index_col=0)
    df.columns = [c.replace('.pure', '').replace('.rlap', '') for c in df.columns]
    df.fillna(df.max().max() + 10.0, inplace=True)
    return {sub: {'winner': row.idxmin(), 'errors': row.to_dict()} for sub, row in df.iterrows()}

# ==========================================
# 4. EXECUTION BLOCK (COMPLETE)
# ==========================================
if __name__ == "__main__":
    print("\n🏗️ Building PURE v3 Dataset...")
    labels = get_winner_data(LOG_FILE)
    subjects = list(labels.keys())
    final_data = []

    # Resume/Build logic
    for idx, sub_id in enumerate(subjects):
        target_path = os.path.join(PATH_PURE, sub_id)
        if not os.path.exists(target_path): continue
        print(f" Processing {sub_id} ({idx+1}/{len(subjects)})")
        feats = extract_prism_features_stream(target_path)
        if np.sum(feats) > 0:
            final_data.append({"id": sub_id, "features": feats, "label": labels[sub_id]['winner'], "errors": labels[sub_id]['errors']})
            # Checkpoint saving
            row = {"id": sub_id, "label": labels[sub_id]['winner'], "errors": str(labels[sub_id]['errors'])}
            for f_i, f_val in enumerate(feats): row[f"feat_{f_i}"] = f_val
            pd.DataFrame([row]).to_csv(CHECKPOINT_FILE, mode='a', header=not os.path.exists(CHECKPOINT_FILE), index=False)
        gc.collect()

    df = pd.DataFrame(final_data)
    if len(df) > 5:
        X = np.stack(df['features'].values)
        model_list = list(df.iloc[0]['errors'].keys())
        y_errors = np.array([[row['errors'][m] for m in model_list] for _, row in df.iterrows()])
        y_class = np.argmin(y_errors, axis=1)

        print(f"\n⚡ Starting 5-Fold Cross-Validation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test_err = y_class[train_idx], y_errors[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = HeartGoldMicroMLP(len(model_list))
            opt = optim.Adam(model.parameters(), lr=0.01)
            for _ in range(150):
                opt.zero_grad()
                nn.CrossEntropyLoss()(model(torch.tensor(X_train_s, dtype=torch.float32)), torch.tensor(y_train, dtype=torch.long)).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                preds = torch.argmax(model(torch.tensor(X_test_s, dtype=torch.float32)), dim=1).numpy()
            prism_mae = np.mean([y_test_err[i, preds[i]] for i in range(len(preds))])
            print(f" Fold Results - PRISM MAE: {prism_mae:.2f} BPM")

        # Save Final Model
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "prism_pure_v3.pth"))
        joblib.dump(scaler, os.path.join(SAVE_DIR, "prism_pure_scaler_v3.pkl"))
        print(f"\n💾 Model saved to {SAVE_DIR}")
    else:
        print("⚠️ Not enough data.")
