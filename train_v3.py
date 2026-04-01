# ============================================================
# PS S6E3 — Churn Prediction | Grandmaster Pipeline v3
# LightGBM (GBDT+DART) + XGBoost + CatBoost + Optuna Ensemble
# Optimized for speed: ~30-45 min total
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
import optuna
import warnings, os, time, gc

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

os.makedirs('submissions', exist_ok=True)
os.makedirs('oof', exist_ok=True)

RANDOM_STATE  = 42
N_FOLDS       = 5
OPTUNA_TRIALS = 300

t0 = time.time()
def elapsed():
    return f"{time.time()-t0:.0f}s"

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. LOAD DATA                                               ║
# ╚══════════════════════════════════════════════════════════════╝
print("=" * 65)
print("  PS S6E3 Churn — Grandmaster Pipeline v3 (Speed Optimized)")
print("=" * 65)
print(f"\n[1/9] Loading data... ({elapsed()})")

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
sub   = pd.read_csv('data/sample_submission.csv')

print(f"  Train: {train.shape} | Test: {test.shape}")
churn_rate = (train['Churn'] == 'Yes').mean()
print(f"  Churn rate: {churn_rate:.4f}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  2. FEATURE ENGINEERING                                     ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n[2/9] Feature engineering... ({elapsed()})")

train['is_train'] = 1
test['is_train']  = 0
test['Churn']     = 'No'

df = pd.concat([train, test], axis=0).reset_index(drop=True)

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# Binary services
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']

for col in service_cols:
    df[col + '_bin'] = (df[col] == 'Yes').astype(np.int8)

# ── Core financial ratios ──
df['charges_per_tenure']     = df['MonthlyCharges'] / (df['tenure'] + 1)
df['expected_total']         = df['MonthlyCharges'] * df['tenure']
df['charges_deviation']      = df['TotalCharges'] - df['expected_total']
df['total_vs_expected']      = df['TotalCharges'] / (df['expected_total'] + 1)
df['avg_monthly_from_total'] = df['TotalCharges'] / (df['tenure'] + 1)
df['charge_consistency']     = df['avg_monthly_from_total'] - df['MonthlyCharges']
df['charge_gap']             = df['MonthlyCharges'] - df['avg_monthly_from_total']
df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

# ── Log / power transforms ──
df['log_tenure']    = np.log1p(df['tenure'])
df['log_monthly']   = np.log1p(df['MonthlyCharges'])
df['log_total']     = np.log1p(df['TotalCharges'])
df['monthly_sq']    = df['MonthlyCharges'] ** 2
df['tenure_sq']     = df['tenure'] ** 2

# ── Service aggregations ──
df['num_services']        = df[[c + '_bin' for c in service_cols]].sum(axis=1)
df['revenue_per_service'] = df['MonthlyCharges'] / (df['num_services'] + 1)
df['num_streaming']       = df['StreamingTV_bin'] + df['StreamingMovies_bin']
df['num_security']        = (df['OnlineSecurity_bin'] + df['OnlineBackup_bin'] +
                             df['DeviceProtection_bin'] + df['TechSupport_bin'])
df['cost_per_security']   = df['MonthlyCharges'] / (df['num_security'] + 1)

# ── High-risk flags ──
df['is_mtm']     = (df['Contract'] == 'Month-to-month').astype(np.int8)
df['is_fiber']   = (df['InternetService'] == 'Fiber optic').astype(np.int8)
df['is_paper']   = (df['PaperlessBilling'] == 'Yes').astype(np.int8)
df['is_echeck']  = (df['PaymentMethod'] == 'Electronic check').astype(np.int8)
df['is_no_inet'] = (df['InternetService'] == 'No').astype(np.int8)

df['risk_2'] = (df['is_mtm'] & df['is_fiber']).astype(np.int8)
df['risk_3'] = (df['risk_2'] & df['is_paper']).astype(np.int8)
df['risk_4'] = (df['risk_3'] & df['is_echeck']).astype(np.int8)
df['no_sec_fiber'] = ((df['num_security'] == 0) & (df['is_fiber'] == 1)).astype(np.int8)

# ── Loyalty / tenure segments ──
df['is_new']       = (df['tenure'] <= 3).astype(np.int8)
df['is_est']       = ((df['tenure'] > 3) & (df['tenure'] <= 12)).astype(np.int8)
df['is_loyal']     = (df['tenure'] >= 24).astype(np.int8)
df['is_very_loyal'] = (df['tenure'] >= 48).astype(np.int8)
df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-1, 3, 6, 12, 24, 48, 72],
                              labels=[0,1,2,3,4,5]).astype(float)

# ── Demographic / behavioral ──
df['senior_no_sup']    = ((df['SeniorCitizen'] == 1) & (df['TechSupport'] == 'No')).astype(np.int8)
df['solo_no_contract'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No') & (df['is_mtm'] == 1)).astype(np.int8)
df['full_protect']     = ((df['OnlineSecurity'] == 'Yes') & (df['OnlineBackup'] == 'Yes') &
                          (df['DeviceProtection'] == 'Yes') & (df['TechSupport'] == 'Yes')).astype(np.int8)
df['ent_bundle']       = ((df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')).astype(np.int8)
df['bare_min']         = ((df['InternetService'] == 'No') & (df['PhoneService'] == 'Yes')).astype(np.int8)
df['premium']          = ((df['tenure'] >= 36) & (df['num_services'] >= 5)).astype(np.int8)
df['at_risk']          = ((df['tenure'] <= 6) & (df['MonthlyCharges'] > 70) & (df['num_security'] == 0)).astype(np.int8)

# ── Interactions ──
df['mc_x_mtm']         = df['MonthlyCharges'] * df['is_mtm']
df['tenure_x_lock']    = df['tenure'] * (1 - df['is_mtm'])
df['tenure_x_svc']     = df['tenure'] * df['num_services']
df['mc_x_sec']         = df['MonthlyCharges'] * df['num_security']
df['tenure_to_mc']     = df['tenure'] / (df['MonthlyCharges'] + 1)
df['mc_to_tc']         = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df['contract_mo']      = df['Contract'].map(contract_map)
df['contract_util']    = df['tenure'] / (df['contract_mo'] + 1)

# ── Synthetic digit features ──
df['tenure_mod12']   = df['tenure'] % 12
df['tenure_mod6']    = df['tenure'] % 6
df['tenure_dig1']    = (df['tenure'] // 10).astype(np.int8)
df['mc_dig1']        = (df['MonthlyCharges'] // 10).astype(np.int8)
df['mc_decimal']     = ((df['MonthlyCharges'] * 100) % 100).astype(int)
df['tc_dig1']        = (df['TotalCharges'] // 1000).astype(np.int8)

# ── Group statistics ──
for grp_col in ['Contract', 'InternetService', 'PaymentMethod']:
    for num_col in ['MonthlyCharges', 'tenure']:
        gm  = df.groupby(grp_col)[num_col].transform('mean')
        gs  = df.groupby(grp_col)[num_col].transform('std').fillna(0)
        df[f'{grp_col}_{num_col}_diff'] = df[num_col] - gm
        df[f'{grp_col}_{num_col}_z']    = (df[num_col] - gm) / (gs + 1e-6)
        df[f'{grp_col}_{num_col}_pct']  = df.groupby(grp_col)[num_col].rank(pct=True)

print(f"  Columns before encoding: {df.shape[1]}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  3. ENCODE CATEGORICALS                                      ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n[3/9] Encoding... ({elapsed()})")

cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']

# Label encode
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# OOF Target Encoding
train_idx = df[df['is_train'] == 1].index
test_idx  = df[df['is_train'] == 0].index
y_full    = (train['Churn'] == 'Yes').astype(np.int8)
g_mean    = y_full.mean()

# Create pairwise combos
pairwise = [('Contract','InternetService'), ('Contract','PaymentMethod'),
            ('InternetService','PaymentMethod'), ('Contract','PaperlessBilling'),
            ('InternetService','OnlineSecurity')]

for c1, c2 in pairwise:
    df[f'{c1}_x_{c2}'] = df[c1].astype(str) + '_' + df[c2].astype(str)

te_cols = [f'{c1}_x_{c2}' for c1, c2 in pairwise] + cat_cols
skf_te  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SMOOTH  = 20

print(f"  OOF target-encoding {len(te_cols)} cols...")
for col in te_cols:
    te_name = f'{col}_te'
    df[te_name] = np.nan

    for tr_idx, val_idx in skf_te.split(train_idx, y_full):
        tr_rows  = train_idx[tr_idx]
        val_rows = train_idx[val_idx]
        s = pd.DataFrame({'c': df.loc[tr_rows, col], 't': y_full.iloc[tr_idx].values})
        st = s.groupby('c')['t'].agg(['mean','count'])
        sm = (st['count'] * st['mean'] + SMOOTH * g_mean) / (st['count'] + SMOOTH)
        df.loc[val_rows, te_name] = df.loc[val_rows, col].map(sm)

    df.loc[train_idx, te_name] = df.loc[train_idx, te_name].fillna(g_mean)

    s = pd.DataFrame({'c': df.loc[train_idx, col], 't': y_full.values})
    st = s.groupby('c')['t'].agg(['mean','count'])
    sm = (st['count'] * st['mean'] + SMOOTH * g_mean) / (st['count'] + SMOOTH)
    df.loc[test_idx, te_name] = df.loc[test_idx, col].map(sm).fillna(g_mean)

# WOE for key cols
woe_cols = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport']
print(f"  WOE encoding {len(woe_cols)} cols...")
for col in woe_cols:
    wn = f'{col}_woe'
    df[wn] = np.nan
    for tr_idx, val_idx in skf_te.split(train_idx, y_full):
        tr_rows  = train_idx[tr_idx]
        val_rows = train_idx[val_idx]
        y_tr = y_full.iloc[tr_idx]
        s = pd.DataFrame({'c': df.loc[tr_rows, col], 't': y_tr.values})
        g = s.groupby('c')['t'].agg(['sum','count'])
        g['ne'] = g['count'] - g['sum']
        te, tne, sm, nc = y_tr.sum(), len(y_tr)-y_tr.sum(), 10, len(g)
        g['de'] = (g['sum']+sm)/(te+sm*nc)
        g['dn'] = (g['ne']+sm)/(tne+sm*nc)
        g['woe'] = np.log(g['dn']/g['de'])
        df.loc[val_rows, wn] = df.loc[val_rows, col].map(g['woe'])
    df.loc[train_idx, wn] = df.loc[train_idx, wn].fillna(0)

    y_tr = y_full
    s = pd.DataFrame({'c': df.loc[train_idx, col], 't': y_tr.values})
    g = s.groupby('c')['t'].agg(['sum','count'])
    g['ne'] = g['count'] - g['sum']
    te, tne, sm, nc = y_tr.sum(), len(y_tr)-y_tr.sum(), 10, len(g)
    g['de'] = (g['sum']+sm)/(te+sm*nc)
    g['dn'] = (g['ne']+sm)/(tne+sm*nc)
    g['woe'] = np.log(g['dn']/g['de'])
    df.loc[test_idx, wn] = df.loc[test_idx, col].map(g['woe']).fillna(0)

print(f"  Total cols: {df.shape[1]}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  4. SPLIT & PREP                                            ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n[4/9] Prep... ({elapsed()})")

train_df = df[df['is_train'] == 1].copy()
test_df  = df[df['is_train'] == 0].copy()

train_df['Churn'] = (train_df['Churn'].astype(str) == 'Yes').astype(int)

drop_cols = (['id', 'Churn', 'is_train', 'expected_total'] +
             [c+'_bin' for c in service_cols] +
             [f'{c1}_x_{c2}' for c1,c2 in pairwise])
drop_cols = [c for c in drop_cols if c in train_df.columns]

feature_cols = [c for c in train_df.columns if c not in drop_cols]
X      = train_df[feature_cols].copy()
y      = train_df['Churn'].copy()
X_test = test_df[[c for c in feature_cols if c in test_df.columns]].copy()
for c in [cc for cc in X.columns if cc not in X_test.columns]:
    X_test[c] = 0
X_test = X_test[X.columns]
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

print(f"  Features: {len(feature_cols)}")
print(f"  X={X.shape} | X_test={X_test.shape}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  5. LightGBM GBDT (fast, reliable)                         ║
# ╚══════════════════════════════════════════════════════════════╝
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
all_oof  = {}
all_test = {}

print(f"\n[5/9] LightGBM GBDT... ({elapsed()})")
print("-" * 55)

lgb_params = {
    'objective': 'binary', 'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'num_leaves': 47, 'max_depth': 5,
    'min_child_samples': 80, 'subsample': 0.75, 'subsample_freq': 1,
    'colsample_bytree': 0.75, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
    'n_estimators': 3000, 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
}

oof_lgb = np.zeros(len(X)); test_lgb = np.zeros(len(X_test))
for fold, (tr, vl) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[vl], y.iloc[vl])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[vl] = m.predict_proba(X.iloc[vl])[:,1]
    test_lgb += m.predict_proba(X_test)[:,1] / N_FOLDS
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y.iloc[vl], oof_lgb[vl]):.5f} iters={m.best_iteration_}")

auc_lgb = roc_auc_score(y, oof_lgb)
print(f"  ★ LGB GBDT OOF: {auc_lgb:.5f}")
all_oof['lgb_gbdt'] = oof_lgb.copy()
all_test['lgb_gbdt'] = test_lgb.copy()
gc.collect()

# ╔══════════════════════════════════════════════════════════════╗
# ║  6. LightGBM DART (fewer trees for speed)                  ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n[6/9] LightGBM DART... ({elapsed()})")
print("-" * 55)

dart_params = {
    'objective': 'binary', 'metric': 'auc',
    'boosting_type': 'dart',
    'learning_rate': 0.05, 'num_leaves': 47, 'max_depth': 5,
    'min_child_samples': 80, 'subsample': 0.75, 'subsample_freq': 1,
    'colsample_bytree': 0.75, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
    'drop_rate': 0.05, 'skip_drop': 0.5,
    'n_estimators': 500,   # DART: fixed trees, no early stopping
    'random_state': 42, 'n_jobs': -1, 'verbose': -1,
}

oof_dart = np.zeros(len(X)); test_dart = np.zeros(len(X_test))
for fold, (tr, vl) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**dart_params)
    m.fit(X.iloc[tr], y.iloc[tr],
          callbacks=[lgb.log_evaluation(0)])
    oof_dart[vl] = m.predict_proba(X.iloc[vl])[:,1]
    test_dart += m.predict_proba(X_test)[:,1] / N_FOLDS
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y.iloc[vl], oof_dart[vl]):.5f}")

auc_dart = roc_auc_score(y, oof_dart)
print(f"  ★ LGB DART OOF: {auc_dart:.5f}")
all_oof['lgb_dart'] = oof_dart.copy()
all_test['lgb_dart'] = test_dart.copy()
gc.collect()

# ╔══════════════════════════════════════════════════════════════╗
# ║  7. XGBoost                                                  ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n[7/9] XGBoost... ({elapsed()})")
print("-" * 55)

xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'learning_rate': 0.03, 'max_depth': 4, 'min_child_weight': 80,
    'subsample': 0.75, 'colsample_bytree': 0.75,
    'reg_alpha': 5.0, 'reg_lambda': 3.0,
    'n_estimators': 3000, 'random_state': 42,
    'n_jobs': -1, 'verbosity': 0, 'tree_method': 'hist',
    'early_stopping_rounds': 200,
}

oof_xgb = np.zeros(len(X)); test_xgb = np.zeros(len(X_test))
for fold, (tr, vl) in enumerate(skf.split(X, y)):
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[vl], y.iloc[vl])],
          verbose=False)
    oof_xgb[vl] = m.predict_proba(X.iloc[vl])[:,1]
    test_xgb += m.predict_proba(X_test)[:,1] / N_FOLDS
    try:
        best_it = m.best_iteration
    except AttributeError:
        best_it = m.n_estimators
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y.iloc[vl], oof_xgb[vl]):.5f} iters={best_it}")

auc_xgb = roc_auc_score(y, oof_xgb)
print(f"  ★ XGB OOF: {auc_xgb:.5f}")
all_oof['xgb'] = oof_xgb.copy()
all_test['xgb'] = test_xgb.copy()
gc.collect()

# ╔══════════════════════════════════════════════════════════════╗
# ║  8. CatBoost                                                ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n[8/9] CatBoost... ({elapsed()})")
print("-" * 55)

cat_params = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 4, 'l2_leaf_reg': 0.05, 'min_data_in_leaf': 80,
    'subsample': 0.75, 'colsample_bylevel': 0.75,
    'eval_metric': 'AUC', 'random_seed': 42,
    'verbose': False, 'early_stopping_rounds': 300,
}

oof_cat = np.zeros(len(X)); test_cat = np.zeros(len(X_test))
for fold, (tr, vl) in enumerate(skf.split(X, y)):
    m = cb.CatBoostClassifier(**cat_params)
    m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[vl], y.iloc[vl]), use_best_model=True)
    oof_cat[vl] = m.predict_proba(X.iloc[vl])[:,1]
    test_cat += m.predict_proba(X_test)[:,1] / N_FOLDS
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y.iloc[vl], oof_cat[vl]):.5f}")

auc_cat = roc_auc_score(y, oof_cat)
print(f"  ★ CatBoost OOF: {auc_cat:.5f}")
all_oof['cat'] = oof_cat.copy()
all_test['cat'] = test_cat.copy()
gc.collect()

# ╔══════════════════════════════════════════════════════════════╗
# ║  9. ENSEMBLE OPTIMIZATION                                   ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n[9/9] Ensemble... ({elapsed()})")
print("-" * 55)

# Save OOF
for n in all_oof:
    np.save(f'oof/oof_{n}.npy', all_oof[n])

# Correlation check
print("\n  Correlation matrix:")
cdf = pd.DataFrame(all_oof)
cm = cdf.corr()
for c in cm.columns:
    print(f"    {c:>10}: " + " | ".join(f"{cm.loc[c,c2]:.4f}" for c2 in cm.columns))

model_names = list(all_oof.keys())
oof_arrs   = [all_oof[n] for n in model_names]
test_arrs  = [all_test[n] for n in model_names]

# Simple avg
oof_simple = sum(oof_arrs) / len(oof_arrs)
test_simple = sum(test_arrs) / len(test_arrs)
auc_simple = roc_auc_score(y, oof_simple)

# Rank avg
oof_rank = sum(rankdata(v) for v in oof_arrs) / len(oof_arrs)
test_rank = sum(rankdata(v) for v in test_arrs) / len(test_arrs)
auc_rank = roc_auc_score(y, oof_rank)

# Optuna weighted
print(f"\n  Optuna: optimizing weights ({OPTUNA_TRIALS} trials)...")
def obj_w(trial):
    ws = [trial.suggest_float(f'w_{n}', 0, 1) for n in model_names]
    t = sum(ws)
    if t < 1e-8: return 0.5
    ws = [w/t for w in ws]
    return roc_auc_score(y, sum(w*p for w,p in zip(ws, oof_arrs)))

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(obj_w, n_trials=OPTUNA_TRIALS)

bw = [study.best_params[f'w_{n}'] for n in model_names]
bt = sum(bw)
bw = [w/bt for w in bw]
oof_optuna = sum(w*p for w,p in zip(bw, oof_arrs))
test_optuna = sum(w*p for w,p in zip(bw, test_arrs))
auc_optuna = roc_auc_score(y, oof_optuna)

# Optuna rank blend
print(f"  Optuna: optimizing rank-blend ({OPTUNA_TRIALS} trials)...")
def obj_r(trial):
    ws = [trial.suggest_float(f'w_{n}', 0, 1) for n in model_names]
    t = sum(ws)
    if t < 1e-8: return 0.5
    ws = [w/t for w in ws]
    ranked = [rankdata(p) for p in oof_arrs]
    return roc_auc_score(y, sum(w*r for w,r in zip(ws, ranked)))

s2 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
s2.optimize(obj_r, n_trials=OPTUNA_TRIALS)

rw = [s2.best_params[f'w_{n}'] for n in model_names]
rt = sum(rw)
rw = [w/rt for w in rw]
oof_orank = sum(w*rankdata(p) for w,p in zip(rw, oof_arrs))
test_orank = sum(w*rankdata(p) for w,p in zip(rw, test_arrs))
auc_orank = roc_auc_score(y, oof_orank)

# ── Results ──
print(f"\n  {'='*55}")
print(f"  INDIVIDUAL MODEL SCORES:")
for n in model_names:
    print(f"    {n:>12}: {roc_auc_score(y, all_oof[n]):.5f}")

print(f"\n  ENSEMBLE SCORES:")
print(f"    Simple avg     : {auc_simple:.5f}")
print(f"    Rank avg       : {auc_rank:.5f}")
print(f"    Optuna weights : {auc_optuna:.5f}  ← {dict(zip(model_names, [f'{w:.3f}' for w in bw]))}")
print(f"    Optuna rank    : {auc_orank:.5f}  ← {dict(zip(model_names, [f'{w:.3f}' for w in rw]))}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  SUBMISSIONS                                                ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n  Generating submissions...")

subs = {
    'lgb_gbdt':    test_lgb,
    'lgb_dart':    test_dart,
    'xgb':         test_xgb,
    'cat':         test_cat,
    'ens_simple':  test_simple,
    'ens_rank':    test_rank / test_rank.max(),
    'ens_optuna':  test_optuna,
    'ens_optrank': test_orank / test_orank.max(),
}

scores = {
    'lgb_gbdt': auc_lgb, 'lgb_dart': auc_dart,
    'xgb': auc_xgb, 'cat': auc_cat,
    'ens_simple': auc_simple, 'ens_rank': auc_rank,
    'ens_optuna': auc_optuna, 'ens_optrank': auc_orank,
}

best = max(scores, key=scores.get)

for name, preds in subs.items():
    out = sub.copy()
    out['Churn'] = preds
    tag = '_BEST' if name == best else ''
    path = f'submissions/sub_v3_{name}{tag}.csv'
    out.to_csv(path, index=False)
    print(f"    {path}  (OOF: {scores.get(name,0):.5f}){' ★ SUBMIT THIS' if name == best else ''}")

print(f"""
{'='*65}
  FINAL -- Pipeline v3
{'='*65}
  LGB GBDT : {auc_lgb:.5f}
  LGB DART : {auc_dart:.5f}
  XGBoost  : {auc_xgb:.5f}
  CatBoost : {auc_cat:.5f}
  -------------------
  Simple   : {auc_simple:.5f}
  Rank     : {auc_rank:.5f}
  Optuna W : {auc_optuna:.5f}
  Optuna R : {auc_orank:.5f}
  -------------------
  [*] BEST: {best} -> {scores[best]:.5f}
{'='*65}
  -> SUBMIT: submissions/sub_v3_{best}_BEST.csv
  -> Time: {elapsed()}
{'='*65}
""")
