import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings
import scipy.signal as signal
from scipy.signal import find_peaks

# 1. Setup Path & GPU Check
sys.path.append(os.getcwd())

print("="*40)
if torch.cuda.is_available():
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print("🚀 Inference will run on T4 GPU.")
else:
    print("⚠️ GPU NOT DETECTED. Processing might be slow.")
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
RLAP_MODELS = [
    'TSCAN.rlap',
    'PhysNet.rlap',
    'EfficientPhys.rlap',
    'PhysMamba.rlap',
    'RhythmMamba.rlap',
    'PhysFormer.rlap',
    'ME-chunk.rlap',
    'ME-flow.rlap'
]

# ==========================================
# HELPER: SIGNAL PROCESSING ENGINE
# ==========================================
def calculate_hr_from_contact_ppg(ppg_signal, time_vector):
    """
    Derives the Ground Truth Heart Rate from the Contact PPG signal.
    """
    # 1. Estimate Sampling Rate
    if len(time_vector) < 2: return None
    fs = 1.0 / np.mean(np.diff(time_vector))

    # 2. Bandpass Filter (0.7 Hz - 4.0 Hz) -> (42 - 240 BPM)
    # This removes baseline wander and high freq noise from the finger clip
    nyquist = 0.5 * fs
    low = 0.7 / nyquist
    high = 4.0 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_ppg = signal.filtfilt(b, a, ppg_signal)

    # 3. Peak Detection
    # Min distance = 0.4s (Assuming max HR of 150 BPM to avoid double counting)
    min_dist = int(fs * 0.4)
    peaks, _ = find_peaks(filtered_ppg, distance=min_dist, prominence=0.5)

    if len(peaks) < 2:
        return None # Signal too noisy

    # 4. Calculate BPM
    peak_times = time_vector[peaks]
    ibi = np.diff(peak_times) # Inter-beat intervals
    bpm_values = 60.0 / ibi

    # Return the average BPM of the whole session
    return np.mean(bpm_values)

# ==========================================
# HELPER: SMART GROUND TRUTH PARSER
# ==========================================
def get_ground_truth(subj_path):
    """
    Reads Contact PPG -> Calculates HR.
    Falls back to Sensor HR if calculation fails.
    """
    gt_hr = None

    # --- CHECK FOR DATASET 2 (ground_truth.txt) ---
    gt2_path = os.path.join(subj_path, "ground_truth.txt")
    if os.path.exists(gt2_path):
        try:
            # Line 1: PPG, Line 2: HR, Line 3: Time
            data = np.loadtxt(gt2_path)
            ppg_trace = data[0, :]
            sensor_hr = np.mean(data[1, :]) # Backup
            time_trace = data[2, :]

            # Calculate from Signal
            calc_hr = calculate_hr_from_contact_ppg(ppg_trace, time_trace)

            if calc_hr is not None: return calc_hr
            else: return sensor_hr # Fallback

        except Exception as e:
            # print(f"⚠️ GT2 Parsing Error: {e}")
            pass

    # --- CHECK FOR DATASET 1 (gtdump.xmp) ---
    gt1_path = os.path.join(subj_path, "gtdump.xmp")
    if os.path.exists(gt1_path):
        try:
            # Col 1: Time(ms), Col 2: HR, Col 4: PPG
            df = pd.read_csv(gt1_path, header=None)
            data = df.values

            time_trace = data[:, 0] / 1000.0 # Convert ms to seconds
            sensor_hr = np.mean(data[:, 1]) # Backup
            ppg_trace = data[:, 3]

            # Calculate from Signal
            calc_hr = calculate_hr_from_contact_ppg(ppg_trace, time_trace)

            if calc_hr is not None: return calc_hr
            else: return sensor_hr # Fallback

        except Exception as e:
            # print(f"⚠️ GT1 Parsing Error: {e}")
            pass

    return None

# ==========================================
# MAIN BENCHMARK LOOP
# ==========================================
def run_universal_benchmark(dataset_folders):
    # Checkpoint File (New name for PPG-derived results)
    checkpoint_file = "/content/drive/MyDrive/UBFC/output-ubfc_ppg_derived.txt"
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    # 1. GATHER ALL SUBJECTS
    all_subject_paths = []
    for root_folder in dataset_folders:
        if not os.path.exists(root_folder):
            print(f"❌ Warning: Folder not found {root_folder}")
            continue

        subs = sorted([s for s in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, s))])
        for s in subs:
            origin_name = os.path.basename(root_folder)
            unique_id = f"{origin_name}_{s}"
            all_subject_paths.append( (unique_id, os.path.join(root_folder, s)) )

    print(f"📂 Found total {len(all_subject_paths)} subjects. Using Contact PPG for Ground Truth.")

    # 2. PROCESSING LOOP
    for i, (unique_id, subj_path) in enumerate(all_subject_paths):

        # SKIP IF DONE
        if os.path.isfile(checkpoint_file):
            try:
                existing_df = pd.read_csv(checkpoint_file, index_col=0)
                if unique_id in existing_df.index:
                    print(f"⏩ [{i+1}/{len(all_subject_paths)}] Skipping {unique_id} (Done)")
                    continue
            except: pass

        print(f"[{i+1}/{len(all_subject_paths)}] Processing: {unique_id}...")

        vid_path = os.path.join(subj_path, "vid.avi")
        if not os.path.exists(vid_path):
            print(f"   ⚠️ Video missing")
            continue

        # GET GROUND TRUTH (Now derived from PPG)
        gt_bpm = get_ground_truth(subj_path)
        if gt_bpm is None or gt_bpm == 0:
            print(f"   ⚠️ GT missing/invalid")
            continue

        # RUN MODELS
        subject_results = {}
        for model_name in RLAP_MODELS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wrapper = Model(model=model_name)

                    # Run Inference
                    result = wrapper.process_video(vid_path)

                    est_bpm = None
                    if isinstance(result, dict): est_bpm = result.get('hr')
                    elif isinstance(result, (float, int)): est_bpm = result

                    if est_bpm is not None:
                        err = abs(est_bpm - gt_bpm)
                        subject_results[model_name] = err
                    else:
                        subject_results[model_name] = np.nan
            except Exception as e:
                subject_results[model_name] = np.nan

        # SAVE
        current_df = pd.DataFrame([subject_results], index=[unique_id])
        if not os.path.isfile(checkpoint_file):
            current_df.to_csv(checkpoint_file, mode='w', header=True)
        else:
            current_df.to_csv(checkpoint_file, mode='a', header=False)

        print(f"   💾 Saved {unique_id} (GT: {gt_bpm:.2f} BPM)")

# ==========================================
# EXECUTION
# ==========================================
DATASET_PATHS = [
    "/content/drive/MyDrive/dataset/DATASET_2",
    "/content/drive/MyDrive/DATASET_1"
]

run_universal_benchmark(DATASET_PATHS)

# --- FINAL SUMMARY ---
output_file = "/content/drive/MyDrive/UBFC/output-ubfc_ppg_derived.txt"
if os.path.exists(output_file):
    print("\n📊 FINAL SUMMARY (PPG-Derived GT):")
    try:
        df = pd.read_csv(output_file, index_col=0)
        print(df.describe())
        print("\n🏆 LEADERBOARD (MAE):")
        print(df.mean().sort_values())
    except:
        print("Could not read final summary.")
