# ============================================================
# PS S6E3 — Churn Prediction | Grandmaster Full Pipeline v2
# LightGBM + XGBoost + CatBoost + Ensemble
# Target: 0.925+ OOF AUC
# ============================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('submissions', exist_ok=True)
os.makedirs('oof', exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("=" * 60)
print("  PS S6E3 Churn Prediction — Full Pipeline v2")
print("=" * 60)
print("\n[1/6] Loading data...")

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
sub   = pd.read_csv('data/sample_submission.csv')

print(f"  Train: {train.shape} | Test: {test.shape}")
print(f"  Churn rate: {(train['Churn'] == 'Yes').mean():.3f}")

# ── 2. COMBINE FOR FEATURE ENGINEERING ───────────────────────
train['is_train'] = 1
test['is_train']  = 0
test['Churn']     = 'No'

df = pd.concat([train, test], axis=0).reset_index(drop=True)

# ── 3. FEATURE ENGINEERING ───────────────────────────────────
print("\n[2/6] Engineering features...")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

# --- Core ratio features ---
df['charges_per_tenure']     = df['MonthlyCharges'] / (df['tenure'] + 1)
df['total_vs_expected']      = df['TotalCharges'] / ((df['MonthlyCharges'] * df['tenure']) + 1)
df['avg_monthly_from_total'] = df['TotalCharges'] / (df['tenure'] + 1)
df['charge_consistency']     = df['avg_monthly_from_total'] - df['MonthlyCharges']
df['charge_gap']             = df['MonthlyCharges'] - df['avg_monthly_from_total']

# --- Log transforms ---
df['log_tenure']       = np.log1p(df['tenure'])
df['log_monthly']      = np.log1p(df['MonthlyCharges'])
df['log_total']        = np.log1p(df['TotalCharges'])
df['monthly_sq']       = df['MonthlyCharges'] ** 2
df['tenure_sq']        = df['tenure'] ** 2

# --- Service count features ---
service_cols = ['PhoneService','MultipleLines','OnlineSecurity',
                'OnlineBackup','DeviceProtection','TechSupport',
                'StreamingTV','StreamingMovies']

for col in service_cols:
    df[col + '_bin'] = (df[col] == 'Yes').astype(int)

df['num_services']       = df[[c+'_bin' for c in service_cols]].sum(axis=1)
df['revenue_per_service']= df['MonthlyCharges'] / (df['num_services'] + 1)

# Streaming services count
df['num_streaming']      = df['StreamingTV_bin'] + df['StreamingMovies_bin']

# Security/protection services count
df['num_security']       = (df['OnlineSecurity_bin'] + df['OnlineBackup_bin'] +
                            df['DeviceProtection_bin'] + df['TechSupport_bin'])

# --- High-risk flags ---
df['is_month_to_month']    = (df['Contract'] == 'Month-to-month').astype(int)
df['is_fiber']             = (df['InternetService'] == 'Fiber optic').astype(int)
df['is_paperless']         = (df['PaperlessBilling'] == 'Yes').astype(int)
df['is_electronic_check']  = (df['PaymentMethod'] == 'Electronic check').astype(int)
df['is_no_internet']       = (df['InternetService'] == 'No').astype(int)

# Highest-risk combination: month-to-month + fiber + paperless + e-check
df['risk_combo_2']  = df['is_month_to_month'] & df['is_fiber']
df['risk_combo_3']  = df['risk_combo_2'] & df['is_paperless']
df['risk_combo_4']  = df['risk_combo_3'] & df['is_electronic_check']

# No protection services + fiber = risky
df['no_security_fiber']   = ((df['num_security'] == 0) & df['is_fiber']).astype(int)

# --- Loyalty/tenure segments ---
df['is_new_customer']   = (df['tenure'] <= 3).astype(int)
df['is_established']    = ((df['tenure'] > 3) & (df['tenure'] <= 12)).astype(int)
df['is_loyal']          = (df['tenure'] >= 24).astype(int)
df['is_very_loyal']     = (df['tenure'] >= 48).astype(int)
df['tenure_bucket']     = pd.cut(df['tenure'],
                                  bins=[0, 3, 6, 12, 24, 48, 72],
                                  labels=[0, 1, 2, 3, 4, 5]).astype(float)

# --- Demographics ---
df['senior_no_support'] = ((df['SeniorCitizen'] == 1) &
                           (df['TechSupport'] == 'No')).astype(int)
df['solo_no_contract']  = ((df['Partner'] == 'No') &
                           (df['Dependents'] == 'No') &
                           df['is_month_to_month']).astype(int)

# --- Interaction: price × contract type ---
df['monthly_x_contract'] = df['MonthlyCharges'] * df['is_month_to_month']
df['tenure_x_contract']  = df['tenure'] * (1 - df['is_month_to_month'])

print(f"  Features engineered successfully")

# ── 4. ENCODE CATEGORICALS ───────────────────────────────────
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ── 5. SPLIT BACK & DEFINE FEATURES ─────────────────────────
train_df = df[df['is_train'] == 1].drop(['is_train'], axis=1).copy()
test_df  = df[df['is_train'] == 0].drop(['is_train', 'Churn'], axis=1).copy()

train_df['Churn'] = (train_df['Churn'] == 'Yes').astype(int)

drop_cols   = ['id', 'Churn'] + [c+'_bin' for c in service_cols]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X      = train_df[feature_cols]
y      = train_df['Churn']
X_test = test_df[feature_cols]

print(f"  Total features: {len(feature_cols)}")
print(f"  Features: {feature_cols}")

# ── 6. CV SETUP ──────────────────────────────────────────────
N_FOLDS    = 5
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Storage for OOF and test predictions
oof_lgb  = np.zeros(len(X))
oof_xgb  = np.zeros(len(X))
oof_cat  = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

# ── 7. LIGHTGBM ──────────────────────────────────────────────
print("\n[3/6] Training LightGBM...")
print("-" * 50)

lgb_params = {
    'objective':         'binary',
    'metric':            'auc',
    'learning_rate':     0.03,
    'num_leaves':        127,
    'max_depth':         -1,
    'min_child_samples': 50,
    'subsample':         0.8,
    'subsample_freq':    1,
    'colsample_bytree':  0.8,
    'reg_alpha':         0.1,
    'reg_lambda':        1.0,
    'n_estimators':      3000,
    'random_state':      RANDOM_STATE,
    'n_jobs':            -1,
    'verbose':           -1,
}

lgb_fold_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(200, verbose=False),
                         lgb.log_evaluation(500)])

    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_lgb        += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_lgb[val_idx])
    lgb_fold_scores.append(fold_auc)
    print(f"  Fold {fold+1}: AUC={fold_auc:.5f} | iters={model.best_iteration_}")

lgb_oof_auc = roc_auc_score(y, oof_lgb)
print(f"\n  LightGBM OOF AUC: {lgb_oof_auc:.5f} ± {np.std(lgb_fold_scores):.5f}")

# ── 8. XGBOOST ───────────────────────────────────────────────
print("\n[4/6] Training XGBoost...")
print("-" * 50)

xgb_params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'learning_rate':    0.03,
    'max_depth':        6,
    'min_child_weight': 50,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'n_estimators':     3000,
    'random_state':     RANDOM_STATE,
    'n_jobs':           -1,
    'verbosity':        0,
    'tree_method':      'hist',
}

xgb_fold_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False,
              early_stopping_rounds=200)

    oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_xgb        += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    xgb_fold_scores.append(fold_auc)
    print(f"  Fold {fold+1}: AUC={fold_auc:.5f} | iters={model.best_iteration}")

xgb_oof_auc = roc_auc_score(y, oof_xgb)
print(f"\n  XGBoost OOF AUC: {xgb_oof_auc:.5f} ± {np.std(xgb_fold_scores):.5f}")

# ── 9. CATBOOST ──────────────────────────────────────────────
print("\n[5/6] Training CatBoost...")
print("-" * 50)

cat_params = {
    'iterations':        3000,
    'learning_rate':     0.03,
    'depth':             6,
    'l2_leaf_reg':       3,
    'min_data_in_leaf':  50,
    'subsample':         0.8,
    'colsample_bylevel': 0.8,
    'eval_metric':       'AUC',
    'random_seed':       RANDOM_STATE,
    'verbose':           False,
    'early_stopping_rounds': 200,
}

cat_fold_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = cb.CatBoostClassifier(**cat_params)
    model.fit(X_tr, y_tr,
              eval_set=(X_val, y_val),
              use_best_model=True)

    oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]
    test_cat        += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_cat[val_idx])
    cat_fold_scores.append(fold_auc)
    print(f"  Fold {fold+1}: AUC={fold_auc:.5f}")

cat_oof_auc = roc_auc_score(y, oof_cat)
print(f"\n  CatBoost OOF AUC:  {cat_oof_auc:.5f} ± {np.std(cat_fold_scores):.5f}")

# ── 10. ENSEMBLE ─────────────────────────────────────────────
print("\n[6/6] Building ensemble...")
print("-" * 50)

# Save OOF predictions for debugging
np.save('oof/oof_lgb.npy', oof_lgb)
np.save('oof/oof_xgb.npy', oof_xgb)
np.save('oof/oof_cat.npy', oof_cat)

# --- Method 1: Simple average ---
oof_simple  = (oof_lgb + oof_xgb + oof_cat) / 3
test_simple = (test_lgb + test_xgb + test_cat) / 3
simple_auc  = roc_auc_score(y, oof_simple)

# --- Method 2: Rank averaging (more robust) ---
oof_rank  = (rankdata(oof_lgb) + rankdata(oof_xgb) + rankdata(oof_cat)) / 3
test_rank = (rankdata(test_lgb) + rankdata(test_xgb) + rankdata(test_cat)) / 3
rank_auc  = roc_auc_score(y, oof_rank)

# --- Method 3: Weighted average (weight by OOF AUC) ---
total_auc   = lgb_oof_auc + xgb_oof_auc + cat_oof_auc
w_lgb = lgb_oof_auc / total_auc
w_xgb = xgb_oof_auc / total_auc
w_cat = cat_oof_auc / total_auc

oof_weighted  = w_lgb*oof_lgb  + w_xgb*oof_xgb  + w_cat*oof_cat
test_weighted = w_lgb*test_lgb + w_xgb*test_xgb + w_cat*test_cat
weighted_auc  = roc_auc_score(y, oof_weighted)

print(f"\n  Model scores:")
print(f"    LightGBM  OOF AUC : {lgb_oof_auc:.5f}  (weight: {w_lgb:.3f})")
print(f"    XGBoost   OOF AUC : {xgb_oof_auc:.5f}  (weight: {w_xgb:.3f})")
print(f"    CatBoost  OOF AUC : {cat_oof_auc:.5f}  (weight: {w_cat:.3f})")
print(f"\n  Ensemble scores:")
print(f"    Simple average : {simple_auc:.5f}")
print(f"    Rank average   : {rank_auc:.5f}")
print(f"    Weighted avg   : {weighted_auc:.5f}")

# Pick the best ensemble method
best_oof_scores = {
    'simple':   simple_auc,
    'rank':     rank_auc,
    'weighted': weighted_auc,
}
best_method = max(best_oof_scores, key=best_oof_scores.get)
best_auc    = best_oof_scores[best_method]

test_preds_map = {
    'simple':   test_simple,
    'rank':     test_rank / test_rank.max(),  # normalize rank to [0,1]
    'weighted': test_weighted,
}

print(f"\n  Best method: '{best_method}' with OOF AUC = {best_auc:.5f}")

# ── 11. SAVE SUBMISSIONS ─────────────────────────────────────
# Save all 4 submissions (3 individual + 1 best ensemble)
for name, preds in [('lgb', test_lgb), ('xgb', test_xgb),
                    ('cat', test_cat),
                    (f'ensemble_{best_method}', test_preds_map[best_method])]:
    out = sub.copy()
    out['Churn'] = preds
    path = f'submissions/sub_{name}.csv'
    out.to_csv(path, index=False)
    print(f"  Saved → {path}")

print(f"""
{'='*60}
  FINAL RESULTS SUMMARY
{'='*60}
  LightGBM  : {lgb_oof_auc:.5f}
  XGBoost   : {xgb_oof_auc:.5f}
  CatBoost  : {cat_oof_auc:.5f}
  Best ensemble ({best_method}): {best_auc:.5f}
{'='*60}
  → Submit:  submissions/sub_ensemble_{best_method}.csv
  → This is your strongest submission!
{'='*60}
""")