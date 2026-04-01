# ============================================================
# final_push.py — Close the 0.001 gap to top 10
# Strategy: Get REAL diversity via different feature subsets
# + aggressive seed averaging on best model (GBDT)
# Expected: 0.917+ OOF AUC
# Run time: ~45-60 min
# ============================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('submissions', exist_ok=True)
os.makedirs('oof', exist_ok=True)

print("=" * 60)
print("  FINAL PUSH — Diversity + Seed Averaging")
print("=" * 60)

# ── LOAD & ENGINEER ───────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
sub   = pd.read_csv('data/sample_submission.csv')

train['is_train'] = 1
test['is_train']  = 0
test['Churn']     = 'No'
df = pd.concat([train, test], axis=0).reset_index(drop=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for c in service_cols:
    df[c+'_bin'] = (df[c]=='Yes').astype(int)

# All features
df['charges_per_tenure']     = df['MonthlyCharges'] / (df['tenure'] + 1)
df['charges_deviation']      = df['TotalCharges'] - df['MonthlyCharges'] * df['tenure']
df['total_vs_expected']      = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'] + 1)
df['avg_monthly_from_total'] = df['TotalCharges'] / (df['tenure'] + 1)
df['charge_gap']             = df['MonthlyCharges'] - df['avg_monthly_from_total']
df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
df['log_tenure']             = np.log1p(df['tenure'])
df['log_monthly']            = np.log1p(df['MonthlyCharges'])
df['log_total']              = np.log1p(df['TotalCharges'])
df['monthly_sq']             = df['MonthlyCharges'] ** 2
df['tenure_sq']              = df['tenure'] ** 2
df['tenure_x_monthly']       = df['tenure'] * df['MonthlyCharges']
df['num_services']           = df[[c+'_bin' for c in service_cols]].sum(axis=1)
df['num_security']           = (df['OnlineSecurity_bin']+df['OnlineBackup_bin']+
                                df['DeviceProtection_bin']+df['TechSupport_bin'])
df['num_streaming']          = df['StreamingTV_bin'] + df['StreamingMovies_bin']
df['revenue_per_service']    = df['MonthlyCharges'] / (df['num_services'] + 1)
df['is_mtm']                 = (df['Contract']=='Month-to-month').astype(int)
df['is_fiber']               = (df['InternetService']=='Fiber optic').astype(int)
df['is_paper']               = (df['PaperlessBilling']=='Yes').astype(int)
df['is_echeck']              = (df['PaymentMethod']=='Electronic check').astype(int)
df['is_autopay']             = df['PaymentMethod'].isin(
                               ['Bank transfer (automatic)',
                                'Credit card (automatic)']).astype(int)
df['risk_2']                 = (df['is_mtm'] & df['is_fiber']).astype(int)
df['risk_3']                 = (df['risk_2'] & df['is_paper']).astype(int)
df['risk_4']                 = (df['risk_3'] & df['is_echeck']).astype(int)
df['safe_combo']             = (((df['Contract']=='Two year') &
                                 df['is_autopay'])).astype(int)
df['is_new']                 = (df['tenure'] <= 3).astype(int)
df['is_loyal']               = (df['tenure'] >= 24).astype(int)
df['has_family']             = ((df['Partner']=='Yes')|(df['Dependents']=='Yes')).astype(int)
df['senior_no_sup']          = ((df['SeniorCitizen']==1)&(df['TechSupport']=='No')).astype(int)
df['full_protect']           = (df['num_security']==4).astype(int)
df['ent_bundle']             = (df['StreamingTV_bin']&df['StreamingMovies_bin']).astype(int)
df['at_risk']                = ((df['tenure']<=6)&(df['MonthlyCharges']>70)&
                                (df['num_security']==0)).astype(int)
df['tenure_mod12']           = df['tenure'] % 12
df['tenure_mod6']            = df['tenure'] % 6
df['mc_dig1']                = (df['MonthlyCharges']//10).astype(int)
df['mc_decimal']             = ((df['MonthlyCharges']*100)%100).astype(int)
df['tenure_dig1']            = (df['tenure']//10).astype(int)
df['mc_x_mtm']               = df['MonthlyCharges'] * df['is_mtm']
df['tenure_x_lock']          = df['tenure'] * (1 - df['is_mtm'])
df['tenure_to_mc']           = df['tenure'] / (df['MonthlyCharges'] + 1)

contract_map = {'Month-to-month':1,'One year':12,'Two year':24}
df['contract_mo']            = df['Contract'].map(contract_map)
df['contract_util']          = df['tenure'] / (df['contract_mo'] + 1)

for grp in ['Contract','InternetService','PaymentMethod']:
    gm = df.groupby(grp)['MonthlyCharges'].transform('mean')
    gs = df.groupby(grp)['MonthlyCharges'].transform('std').fillna(0)
    df[f'{grp}_mc_diff'] = df['MonthlyCharges'] - gm
    df[f'{grp}_mc_z']    = (df['MonthlyCharges'] - gm) / (gs + 1e-6)
    df[f'{grp}_mc_pct']  = df.groupby(grp)['MonthlyCharges'].rank(pct=True)

# Encode
cat_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
            'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies','Contract',
            'PaperlessBilling','PaymentMethod']
le = LabelEncoder()
for c in cat_cols:
    df[c] = le.fit_transform(df[c].astype(str))

# OOF Target Encoding — pairwise
train_idx = df[df['is_train']==1].index
test_idx  = df[df['is_train']==0].index
y_full    = (train['Churn']=='Yes').astype(int)
g_mean    = y_full.mean()
skf_te    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SMOOTH    = 20

pairwise = [('Contract','InternetService'),('Contract','PaymentMethod'),
            ('InternetService','PaymentMethod'),('Contract','PaperlessBilling')]
for c1,c2 in pairwise:
    df[f'{c1}_x_{c2}'] = df[c1].astype(str)+'_'+df[c2].astype(str)

te_cols = cat_cols + [f'{c1}_x_{c2}' for c1,c2 in pairwise]
for col in te_cols:
    te_name = f'{col}_te'
    df[te_name] = np.nan
    for tr_i, val_i in skf_te.split(train_idx, y_full):
        tr_r, val_r = train_idx[tr_i], train_idx[val_i]
        s = pd.DataFrame({'c':df.loc[tr_r,col],'t':y_full.iloc[tr_i].values})
        st = s.groupby('c')['t'].agg(['mean','count'])
        sm = (st['count']*st['mean']+SMOOTH*g_mean)/(st['count']+SMOOTH)
        df.loc[val_r,te_name] = df.loc[val_r,col].map(sm)
    df.loc[train_idx,te_name] = df.loc[train_idx,te_name].fillna(g_mean)
    s = pd.DataFrame({'c':df.loc[train_idx,col],'t':y_full.values})
    st = s.groupby('c')['t'].agg(['mean','count'])
    sm = (st['count']*st['mean']+SMOOTH*g_mean)/(st['count']+SMOOTH)
    df.loc[test_idx,te_name] = df.loc[test_idx,col].map(sm).fillna(g_mean)

# Split
train_df = df[df['is_train']==1].copy()
test_df  = df[df['is_train']==0].copy()
train_df['Churn'] = (train_df['Churn'].astype(str)=='Yes').astype(int)

drop = (['id','Churn','is_train'] + [c+'_bin' for c in service_cols] +
        [f'{c1}_x_{c2}' for c1,c2 in pairwise])
fcols = [c for c in train_df.columns if c not in drop]
X     = train_df[fcols].replace([np.inf,-np.inf], np.nan)
y     = train_df['Churn']
Xt    = test_df[[c for c in fcols if c in test_df.columns]].copy()
for c in [c for c in X.columns if c not in Xt.columns]: Xt[c]=0
Xt    = Xt[X.columns].replace([np.inf,-np.inf], np.nan)
print(f"Features: {X.shape[1]}")

# ── DEFINE 3 DIFFERENT FEATURE SUBSETS ───────────────────────
# This is the KEY to diversity without changing architecture
all_cols  = X.columns.tolist()

# Subset A: Financial + Risk focus (no digit features)
cols_A = [c for c in all_cols if not any(x in c for x in
          ['mod','dig1','decimal','_te','_woe'])]

# Subset B: Encoding-heavy (TE + WOE only, minimal raw)
cols_B = [c for c in all_cols if any(x in c for x in
          ['_te','log_','charges','tenure','num_','is_','risk',
           'safe','mc_','contract_','senior','at_risk'])]

# Subset C: All features (full)
cols_C = all_cols

print(f"Subset A (financial+risk): {len(cols_A)} features")
print(f"Subset B (encoding-heavy): {len(cols_B)} features")
print(f"Subset C (all):            {len(cols_C)} features")

# ── LGBM PARAMS ───────────────────────────────────────────────
base_params = {
    'objective':'binary', 'metric':'auc', 'boosting_type':'gbdt',
    'learning_rate':0.03, 'num_leaves':47, 'max_depth':5,
    'min_child_samples':80, 'subsample':0.75, 'subsample_freq':1,
    'colsample_bytree':0.75, 'reg_alpha':0.5, 'reg_lambda':2.0,
    'n_estimators':3000, 'n_jobs':-1, 'verbose':-1,
}

# ── TRAIN ALL COMBINATIONS ───────────────────────────────────
# 3 subsets × 5 seeds = 15 models
SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

results = {}

for subset_name, cols in [('A', cols_A), ('B', cols_B), ('C', cols_C)]:
    print(f"\n{'='*55}")
    print(f"  Subset {subset_name} — {len(cols)} features")
    print(f"{'='*55}")

    Xs  = X[cols]
    Xts = Xt[cols]
    oof_seeds  = []
    test_seeds = []

    for seed in SEEDS:
        oof_s  = np.zeros(len(Xs))
        test_s = np.zeros(len(Xts))
        fold_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_i, val_i) in enumerate(fold_skf.split(Xs, y)):
            m = lgb.LGBMClassifier(**{**base_params, 'random_state':seed})
            m.fit(Xs.iloc[tr_i], y.iloc[tr_i],
                  eval_set=[(Xs.iloc[val_i], y.iloc[val_i])],
                  callbacks=[lgb.early_stopping(200,verbose=False),
                             lgb.log_evaluation(-1)])
            oof_s[val_i]  = m.predict_proba(Xs.iloc[val_i])[:,1]
            test_s       += m.predict_proba(Xts)[:,1] / N_FOLDS

        s_auc = roc_auc_score(y, oof_s)
        print(f"  Seed {seed}: {s_auc:.5f}")
        oof_seeds.append(oof_s)
        test_seeds.append(test_s)

    oof_avg  = np.mean(oof_seeds, axis=0)
    test_avg = np.mean(test_seeds, axis=0)
    avg_auc  = roc_auc_score(y, oof_avg)
    print(f"  ★ Subset {subset_name} seed-avg AUC: {avg_auc:.5f}")
    results[subset_name] = {'oof': oof_avg, 'test': test_avg, 'auc': avg_auc}

# ── LOAD EXISTING OOF from v3 ─────────────────────────────────
print(f"\n{'='*55}")
print("  Loading OOF from v3 run...")
print(f"{'='*55}")

try:
    oof_v3_gbdt = np.load('oof/oof_lgb_gbdt.npy')
    oof_v3_cat  = np.load('oof/oof_cat.npy')
    # Reconstruct v3 best blend: gbdt=0.589, cat=0.352, xgb=0.059
    oof_v3_xgb  = np.load('oof/oof_xgb.npy')
    oof_v3_best = (0.589*oof_v3_gbdt + 0.352*oof_v3_cat + 0.059*oof_v3_xgb)
    test_v3_gbdt = np.load('oof/oof_lgb_gbdt.npy')  # use oof as proxy
    print(f"  v3 GBDT OOF: {roc_auc_score(y, oof_v3_gbdt):.5f}")
    print(f"  v3 CAT  OOF: {roc_auc_score(y, oof_v3_cat):.5f}")
    has_v3 = True
except:
    print("  OOF files not found — using new results only")
    has_v3 = False

# ── MEGA ENSEMBLE ─────────────────────────────────────────────
print(f"\n{'='*55}")
print("  Mega ensemble — all predictions combined")
print(f"{'='*55}")

oof_all = [results['A']['oof'], results['B']['oof'], results['C']['oof']]
test_all = [results['A']['test'], results['B']['test'], results['C']['test']]

if has_v3:
    oof_all  += [oof_v3_gbdt, oof_v3_cat]
    # For test predictions, reload from submission files
    try:
        v3_gbdt_sub  = pd.read_csv('submissions/sub_v3_lgb_gbdt.csv')
        v3_cat_sub   = pd.read_csv('submissions/sub_v3_cat.csv')
        test_all += [v3_gbdt_sub['Churn'].values, v3_cat_sub['Churn'].values]
        print("  Loaded v3 test predictions")
    except:
        oof_all  = oof_all[:3]
        test_all = test_all[:3]
        has_v3   = False

# Rank average of everything
oof_mega  = sum(rankdata(o) for o in oof_all) / len(oof_all)
test_mega = sum(rankdata(t) for t in test_all) / len(test_all)
mega_auc  = roc_auc_score(y, oof_mega)

# Normalize test
test_mega_norm = (test_mega - test_mega.min())/(test_mega.max()-test_mega.min())

print(f"\n  Subset A seed-avg : {results['A']['auc']:.5f}")
print(f"  Subset B seed-avg : {results['B']['auc']:.5f}")
print(f"  Subset C seed-avg : {results['C']['auc']:.5f}")
print(f"  Mega rank blend   : {mega_auc:.5f}")

# ── SAVE SUBMISSIONS ──────────────────────────────────────────
print(f"\n  Saving submissions...")

best_single = max(results, key=lambda k: results[k]['auc'])

# Individual subsets
for name in ['A','B','C']:
    out = sub.copy()
    out['Churn'] = results[name]['test']
    out.to_csv(f'submissions/sub_subset_{name}.csv', index=False)

# Mega ensemble
out = sub.copy()
out['Churn'] = test_mega_norm
out.to_csv('submissions/MEGA_ENSEMBLE.csv', index=False)
print(f"  Saved MEGA_ENSEMBLE.csv")

# Best single subset + seed avg
out = sub.copy()
out['Churn'] = results[best_single]['test']
out.to_csv(f'submissions/BEST_SEED_AVG.csv', index=False)

print(f"""
{'='*60}
  FINAL PUSH RESULTS
{'='*60}
  Subset A (financial): {results['A']['auc']:.5f}
  Subset B (encodings): {results['B']['auc']:.5f}
  Subset C (all feats): {results['C']['auc']:.5f}
  Mega rank ensemble  : {mega_auc:.5f}
  ─────────────────────────────────────────
  v3 best (reference) : 0.91665
  ─────────────────────────────────────────
  YOUR 2 SUBMISSION SLOTS:
  Slot 1 → MEGA_ENSEMBLE.csv    (most robust)
  Slot 2 → BEST_SEED_AVG.csv    (best single CV)
{'='*60}
""")