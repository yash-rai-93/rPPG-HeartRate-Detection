import sys
import os
import json
import numpy as np
import pandas as pd
import cv2
import glob
import torch
import time

# 1. Setup Path & GPU Check
sys.path.append(os.getcwd())

print("="*40)
if torch.cuda.is_available():
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print("🚀 Inference will run on T4 GPU.")
else:
    print("⚠️ GPU NOT DETECTED. Processing full videos on CPU will be slow.")
print("="*40 + "\n")

# Setup Open-rPPG
if not os.path.exists("open-rppg"):
    os.system("git clone https://github.com/KegangWangCCNU/open-rppg.git")
    os.system("pip install -r open-rppg/requirements.txt")
sys.path.append(os.path.join(os.getcwd(), "open-rppg"))

try:
    from rppg.main import Model
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "open-rppg"))
    from rppg.main import Model

# 2. Define Models
PURE_MODELS = [
    'PhysNet.rlap',
    'PhysMamba.rlap',
    'RhythmMamba.rlap',
    'PhysFormer.rlap',
    'ME-flow.rlap'
]

# ==========================================
# HELPER: CONVERT IMAGES TO VIDEO (FULL)
# ==========================================
def images_to_video(image_folder, output_path, fps=30.0):
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        sub_name = os.path.basename(image_folder)
        nested_path = os.path.join(image_folder, sub_name)
        if os.path.exists(nested_path):
            images = sorted(glob.glob(os.path.join(nested_path, "*.png")))

    if not images:
        return False

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(image))
    video.release()
    return True

# ==========================================
# HELPER: PARSE FULL GROUND TRUTH
# ==========================================
def get_pure_ground_truth(json_path):
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if "/FullPackage" in data:
            package = data["/FullPackage"]
        elif "FullPackage" in data:
            package = data["FullPackage"]
        elif "Reference" in data and "FullPackage" in data["Reference"]:
            package = data["Reference"]["FullPackage"]
        else:
            return None
        hr_values = [entry['Value']['pulseRate'] for entry in package if 'Value' in entry and 'pulseRate' in entry['Value']]
        return np.mean(hr_values) if hr_values else None
    except Exception:
        return None

# ==========================================
# MAIN BENCHMARK LOOP (WITH RESUME & SAVING)
# ==========================================
def run_optimized_benchmark(dataset_root):
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset not found at: {dataset_root}")
        return

    subjects = sorted([s for s in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, s))])
    print(f"📂 Found {len(subjects)} subjects. Starting PURE Full-Video Benchmark...\n")

    results = {sub: {} for sub in subjects}
    temp_video_path = "/content/temp_pure_full.avi"
    checkpoint_file = "/content/drive/MyDrive/output-pure.txt"

    for i, sub in enumerate(subjects):
        # --- RESUME LOGIC ---
        if os.path.isfile(checkpoint_file):
            existing_df = pd.read_csv(checkpoint_file, index_col=0)
            if sub in existing_df.index:
                print(f"⏩ [{i+1}/{len(subjects)}] Skipping {sub} (Already in checkpoint)")
                continue

        subj_path = os.path.join(dataset_root, sub)
        gt_path = os.path.join(subj_path, f"{sub}.json")

        print(f"[{i+1}/{len(subjects)}] Processing Subject: {sub}...")

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        has_video = images_to_video(subj_path, temp_video_path)
        if not has_video:
            print(f"   ⚠️ Skipping {sub} (No images found)")
            continue

        gt_bpm = get_pure_ground_truth(gt_path)
        if gt_bpm is None:
             print(f"   ⚠️ Skipping {sub} (No GT found)")
             continue

        for model_name in PURE_MODELS:
            try:
                wrapper = Model(model=model_name)
                result = wrapper.process_video(temp_video_path)
                est_bpm = result.get('hr') if isinstance(result, dict) else result

                if est_bpm is not None:
                    results[sub][model_name] = abs(est_bpm - gt_bpm)
                else:
                    results[sub][model_name] = np.nan
            except Exception:
                results[sub][model_name] = np.nan

        # --- SAVING CHECKPOINT ---
        checkpoint_df = pd.DataFrame([results[sub]], index=[sub])
        if not os.path.isfile(checkpoint_file):
            checkpoint_df.to_csv(checkpoint_file, mode='w', header=True)
        else:
            checkpoint_df.to_csv(checkpoint_file, mode='a', header=False)
        print(f"   💾 Saved results for {sub}")

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    return results

# --- RUN ---
DATASET_ROOT = "/content/drive/MyDrive/PURE"
stats = run_optimized_benchmark(DATASET_ROOT)

# --- FINAL SUMMARY ---
if os.path.exists("/content/drive/MyDrive/output-pure.txt"):
    print("\n📊 FINAL SUMMARY:")
    df = pd.read_csv("/content/drive/MyDrive/output-pure.txt", index_col=0)
    print("\n🏆 LEADERBOARD (Average MAE):")
    print(df.mean().sort_values().to_string())