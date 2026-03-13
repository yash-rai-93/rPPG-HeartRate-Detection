import pandas as pd
import numpy as np
import ast
import warnings
import re
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# ==========================================
# 1. PHYSICS FEATURE ENGINE
# ==========================================
class PrismFeatureEngine:
    def __init__(self):
        self.idx_map = {'phi': 0, 'sigma': 1, 'mu': 2, 'chi': 3, 'motion': 6, 'ghost': 10}

    def compute_features(self, raw_features: np.ndarray) -> np.ndarray:
        phi = raw_features[:, self.idx_map['phi']]
        sigma = raw_features[:, self.idx_map['sigma']] + 1e-6
        mu = raw_features[:, self.idx_map['mu']]
        chi = raw_features[:, self.idx_map['chi']]
        motion = raw_features[:, self.idx_map['motion']]
        ghost = raw_features[:, self.idx_map['ghost']]

        snr = phi / sigma
        stability = mu / (sigma + 1e-3)
        motion_delta = np.gradient(motion) if len(motion) > 1 else np.zeros_like(motion)
        purity_score = (phi * chi) / (sigma * ghost + 1e-6)

        return np.column_stack((
            snr, stability, motion, motion_delta, purity_score,
            phi, sigma, chi, ghost, np.log1p(np.abs(phi))
        ))

# ==========================================
# 2. PRISM-NET: ERROR-RANKING REGRESSOR
# ==========================================
class PrismNet:
    def __init__(self, hidden_layers=(512, 256, 128), alpha=0.01):
        self.scaler = RobustScaler()
        self.feature_engine = PrismFeatureEngine()
        self.network = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='tanh',
            alpha=alpha,
            learning_rate_init=0.0005,
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=100,
            random_state=42
        )
        self.models_in_play = []
        self.safe_model_idx = -1

    def load_data(self, filepath: str) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError: return None, None, None

        X_list, Y_dict_list, sub_list = [], [], []
        feature_cols = [f'feat_{i}' for i in range(11)]
        for _, row in df.iterrows():
            try:
                err_dict = ast.literal_eval(row['errors']) if isinstance(row['errors'], str) else row['errors']
                clean_errs = {k.replace('.rlap', '').strip(): v for k, v in err_dict.items()}
                X_list.append(row[feature_cols].values.astype(float))
                Y_dict_list.append(clean_errs)
                sub_list.append(str(row['id']))
            except: continue

        X_prism = self.feature_engine.compute_features(np.array(X_list))
        return X_prism, Y_dict_list, np.array(sub_list)

    def fit(self, X_train, Y_train_maps):
        self.models_in_play = sorted(list(set().union(*[d.keys() for d in Y_train_maps])))
        Y_matrix = np.array([[d.get(m, 100.0) for m in self.models_in_play] for d in Y_train_maps])
        self.safe_model_idx = np.argmin(np.mean(Y_matrix, axis=0))

        Y_target = np.log1p(Y_matrix)

        X_scaled = self.scaler.fit_transform(X_train)
        self.network.fit(X_scaled, Y_target)

    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        predicted_log_errors = self.network.predict(X_scaled)

        final_choices = np.argmin(predicted_log_errors, axis=1)
        confidence = np.min(predicted_log_errors, axis=1) # Log-error confidence
        switches = np.sum(final_choices != self.safe_model_idx)

        return final_choices, switches, confidence

# ==========================================
# 3. METRICS & PIPELINE
# ==========================================
def calculate_metrics(y_true, y_oracle):
    return {
        "MAE": np.mean(y_true),
        "RMSE": np.sqrt(np.mean(y_true**2)),
        "95%ile": np.percentile(y_true, 95),
        "Max Err": np.max(y_true),
        "Regret": np.mean(y_true - y_oracle)
    }

def run_experiment(name, prism_model, data, sub_ids_test):
    X_tr, Y_tr, X_te, Y_te = data
    prism_model.fit(X_tr, Y_tr)
    preds, sw, conf = prism_model.predict(X_te)

    models = prism_model.models_in_play
    Y_matrix = np.array([[d.get(m, 100.0) for m in models] for d in Y_te])
    oracle = np.min(Y_matrix, axis=1)

    # 1. Print Leaderboard
    table_data = []
    for idx, m_name in enumerate(models):
        table_data.append((m_name, calculate_metrics(Y_matrix[:, idx], oracle)))

    prism_errs = np.array([Y_matrix[i, preds[i]] for i in range(len(preds))])
    table_data.append(("PRISM-Net", calculate_metrics(prism_errs, oracle)))

    print(f"\n🚀 SCENARIO: {name} (Switches: {sw}/{len(preds)})")
    print("=" * 110)
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'95%ile':<10} | {'Max Err':<10} | {'Regret':<10}")
    print("-" * 110)
    for n, m in sorted(table_data, key=lambda x: x[1]['MAE']):
        prefix = "👉 " if n == "PRISM-Net" else "    "
        print(f"{prefix}{n:<17} | {m['MAE']:<10.4f} | {m['RMSE']:<10.4f} | {m['95%ile']:<10.4f} | {m['Max Err']:<10.4f} | {m['Regret']:<10.4f}")

    # 2. SAVE ANALYSIS TO CSV
    analysis_records = []
    safe_model_name = models[prism_model.safe_model_idx]

    for i in range(len(preds)):
        sel_idx = preds[i]
        opt_idx = np.argmin(Y_matrix[i])

        record = {
            "Video_ID": sub_ids_test[i],
            "Selected_Model": models[sel_idx],
            "Selected_Error": round(Y_matrix[i, sel_idx], 4),
            "Titan_Model": safe_model_name,
            "Titan_Error": round(Y_matrix[i, prism_model.safe_model_idx], 4),
            "Optimal_Model": models[opt_idx],
            "Optimal_Error": round(Y_matrix[i, opt_idx], 4),
            "Regret": round(Y_matrix[i, sel_idx] - Y_matrix[i, opt_idx], 4),
            "Did_Switch": sel_idx != prism_model.safe_model_idx,
            "Confidence": round(conf[i], 4)
        }
        # Add individual model errors for full context
        for m_idx, m_name in enumerate(models):
            record[m_name] = round(Y_matrix[i, m_idx], 4)

        analysis_records.append(record)

    safe_filename = re.sub(r'[\\/*?:"<>|]', '_', name).replace(' ', '_')
    csv_path = f"PRISM_Analysis_{safe_filename}.csv"
    pd.DataFrame(analysis_records).to_csv(csv_path, index=False)
    print(f"💾 Detailed analysis saved to: {csv_path}")

def main():
    prism = PrismNet()
    X_p, Y_p, S_p = prism.load_data("prism_features_pure_vnew.csv")
    X_u, Y_u, S_u = prism.load_data("prism_features_ubfc_v3.csv")

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

    if X_u is not None:
        tr, te = next(gss.split(X_u, groups=S_u))
        run_experiment("Intra-UBFC", prism,
                       (X_u[tr], [Y_u[i] for i in tr], X_u[te], [Y_u[i] for i in te]),
                       S_u[te]) # Pass test IDs

    if X_p is not None:
        tr, te = next(gss.split(X_p, groups=S_p))
        run_experiment("Intra-PURE", prism,
                       (X_p[tr], [Y_p[i] for i in tr], X_p[te], [Y_p[i] for i in te]),
                       S_p[te]) # Pass test IDs

    if X_u is not None and X_p is not None:
        X_g = np.concatenate([X_u, X_p])
        Y_g = Y_u + Y_p
        # Prefix IDs to avoid collisions between datasets
        S_u_prefixed = np.array([f"U{s}" for s in S_u])
        S_p_prefixed = np.array([f"P{s}" for s in S_p])
        S_g = np.concatenate([S_u_prefixed, S_p_prefixed])

        strat_labels = np.concatenate([np.zeros(len(X_u)), np.ones(len(X_p))])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(sss.split(X_g, strat_labels))

        run_experiment("Global Mix (80 20)", prism,
                       (X_g[tr_idx], [Y_g[i] for i in tr_idx], X_g[te_idx], [Y_g[i] for i in te_idx]),
                       S_g[te_idx]) # Pass test IDs

if __name__ == "__main__":
    main()
