# 1. INSTALL STABLE MEDIAPIPE
# We specifically use 0.10.2.1 to avoid the 'audio_classifier' bug in newer versions


import os
import sys
import cv2
import numpy as np
import pandas as pd
import ast
import gc
import warnings
from scipy.stats import entropy
from glob import glob

# 2. IMPORTS (Standard Syntax works fine with 0.10.2.1)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
# PATHS TO YOUR UBFC DATASETS
PATH_UBFC_1 = "/content/drive/MyDrive/DATASET_1"
PATH_UBFC_2 = "/content/drive/MyDrive/UBFC/DATASET_2"

# LABELS FILE (UBFC MAE LOG)
# Ensure this file exists from your previous benchmarking run
LOG_FILE = "/content/drive/MyDrive/UBFC/output-ubfc_ppg_derived.txt"

# RESULTS DIR
SAVE_DIR = "/content/drive/MyDrive/PRISM_RESULTS_UBFC_FINAL22/"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

CHECKPOINT_FILE = os.path.join(SAVE_DIR, "prism_features_ubfc_v3.csv")

# 11 FEATURES
FEATURES_LIST = [
    'Phi', 'Sigma', 'Mu', 'Chi', 'H', 'Clip', 'M',
    'Res', 'Flicker', 'Tau', 'Ghost'
]

# ==========================================
# 2. MEDIAPIPE SETUP (PURE STYLE)
# ==========================================
# Using the exact same setup as your working PURE code
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)
print("✅ MediaPipe (0.10.2.1) initialized successfully.")

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
# 3. FEATURE EXTRACTOR (EXACT COPY OF PURE v3)
# ==========================================
def process_single_frame_v3(frame, prev_gray_full, kf):
    h_f, w_f, _ = frame.shape
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MediaPipe Detection
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

    # Kalman Stabilization
    x, y, w, h = kf.update(raw_x, raw_y, raw_w, raw_h)
    x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)

    # ROI Extraction
    face_roi = frame[y:y+h, x:x+w]
    if face_roi.size == 0: return np.zeros(10), gray_full

    face_g = face_roi[:, :, 1]
    face_gray = gray_full[y:y+h, x:x+w]

    # --- PHYSICS METRICS ---
    phi = np.mean(face_g) / 255.0

    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    if np.sum(mask) > 0:
        mu = np.mean(ycrcb[:,:,0][mask > 0]) / 255.0
        sigma = np.std(face_g[mask > 0]) / 50.0
    else:
        mu = 0.5
        sigma = np.std(face_g) / 50.0

    chi = min(cv2.Laplacian(face_gray, cv2.CV_64F).var(), 1000.0) / 1000.0
    hist, _ = np.histogram(face_g.flatten(), bins=20, density=True)
    H = entropy(hist + 1e-10) / 5.0
    clip = (np.sum(face_g > 250) + np.sum(face_g < 5)) / face_g.size

    m_val = 0.0
    if prev_gray_full is not None:
        try:
            prev_face = prev_gray_full[y:y+h, x:x+w]
            if prev_face.shape == face_gray.shape:
                flow = cv2.calcOpticalFlowFarneback(prev_face, face_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                m_val = np.mean(np.linalg.norm(flow, axis=2))
        except: pass

    res = (w * h) / (w_f * h_f)
    mean_val = np.mean(face_g)

    # Global Ghost
    f_fft = np.fft.fft2(gray_full)
    mag = 20 * np.log(np.abs(np.fft.fftshift(f_fft)) + 1e-10)
    cy, cx = h_f//2, w_f//2
    ghost = np.mean(mag[cy-10:cy+10, cx-10:cx+10]) / (np.mean(mag) + 1e-5)

    return [phi, sigma, mu, chi, H, clip, m_val, res, mean_val, ghost], gray_full

def extract_prism_features_from_video(video_path):
    """
    UBFC Adaptation: Reads video frames directly instead of reading images from a folder.
    Logic is identical to PURE otherwise.
    """
    kf = FaceKalman()
    accumulators = {i: [] for i in range(10)}
    prev_gray = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return np.zeros(11)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 300)

    curr_frame_idx = 0
    frames_processed = 0

    while frames_processed < 300:
        ret, frame = cap.read()
        if not ret: break

        if curr_frame_idx % step == 0:
            stats, prev_gray = process_single_frame_v3(frame, prev_gray, kf)
            if np.sum(stats) > 0:
                for idx, val in enumerate(stats): accumulators[idx].append(val)
                frames_processed += 1
        curr_frame_idx += 1

    cap.release()

    if frames_processed < 10: return np.zeros(11)

    # Aggregation (Identical to PURE)
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
# 4. DATA LOADING
# ==========================================
def load_labels_ubfc(log_path):
    print(f"📖 Loading Labels from: {os.path.basename(log_path)}")
    if not os.path.exists(log_path): return {}
    try:
        df = pd.read_csv(log_path, index_col=0)
        data = {}
        for sub, row in df.iterrows():
            # UBFC logs usually have columns like 'PhysNet.rlap', etc.
            errors = row.to_dict()
            winner = min(errors, key=errors.get)
            data[str(sub)] = {'winner': winner, 'errors': errors}
        return data
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return {}

def build_dataset_ubfc_v3():
    print("\n🏗️ Building UBFC Dataset v3 (Video -> Features)...")

    labels = load_labels_ubfc(LOG_FILE)
    subjects = list(labels.keys())

    if not subjects:
        print("❌ No labels found.")
        return pd.DataFrame()

    # Resume Checkpoint
    completed_feats = {}
    if os.path.exists(CHECKPOINT_FILE):
        try:
            cp_df = pd.read_csv(CHECKPOINT_FILE)
            for _, row in cp_df.iterrows():
                if 'feat_10' in row:
                    cols = [f"feat_{i}" for i in range(11)]
                    completed_feats[str(row['id'])] = row[cols].values.astype(float)
            print(f"   ✅ Resumed {len(completed_feats)} subjects.")
        except: pass

    final_data = []

    for idx, sub_id in enumerate(subjects):
        if sub_id in completed_feats:
            final_data.append({
                "id": sub_id,
                "features": completed_feats[sub_id],
                "label": labels[sub_id]['winner'],
                "errors": labels[sub_id]['errors']
            })
            continue

        # FIND VIDEO (UBFC Logic)
        # ID is usually "DATASET_2_subject1" -> clean="subject1"
        clean_name = sub_id.split('_')[-1]
        if "subject" not in clean_name and clean_name.isdigit():
            clean_name = "subject" + clean_name

        path_opt1 = os.path.join(PATH_UBFC_1, clean_name, "vid.avi")
        path_opt2 = os.path.join(PATH_UBFC_2, clean_name, "vid.avi")

        target_path = ""
        if os.path.exists(path_opt1): target_path = path_opt1
        elif os.path.exists(path_opt2): target_path = path_opt2
        else:
            # Fallback for exact match
            path_opt3 = os.path.join(PATH_UBFC_1, sub_id, "vid.avi")
            path_opt4 = os.path.join(PATH_UBFC_2, sub_id, "vid.avi")
            if os.path.exists(path_opt3): target_path = path_opt3
            elif os.path.exists(path_opt4): target_path = path_opt4

        if target_path == "":
            print(f"   ⚠️ Could not find video for {sub_id}")
            continue

        print(f"   ...Processing {sub_id} ({idx+1}/{len(subjects)})")

        # EXTRACT FROM VIDEO
        try:
            feats = extract_prism_features_from_video(target_path)

            if np.sum(feats) > 0:
                row = {
                    "id": sub_id,
                    "label": labels[sub_id]['winner'],
                    "errors": str(labels[sub_id]['errors'])
                }
                for f_i, f_val in enumerate(feats): row[f"feat_{f_i}"] = f_val

                pd.DataFrame([row]).to_csv(CHECKPOINT_FILE, mode='a', header=not os.path.exists(CHECKPOINT_FILE), index=False)

                final_data.append({
                    "id": sub_id,
                    "features": feats,
                    "label": labels[sub_id]['winner'],
                    "errors": labels[sub_id]['errors']
                })
        except Exception as e:
            print(f"   ❌ Error extracting {sub_id}: {e}")

        gc.collect()

    return pd.DataFrame(final_data)

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    df = build_dataset_ubfc_v3()

    if len(df) > 0:
        print(f"\n✅ UBFC v3 Extraction Complete. Saved to {CHECKPOINT_FILE}")
