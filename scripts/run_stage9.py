import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(f'{base}/features_clean_v1.csv', dtype={'billing.zip.code': str})
folds = pd.read_csv(f'{base}/train_folds.csv')
train = df[df['split'] == 'train'].merge(folds, on=['account.id', 'label'])
ALL_CAT = ['billing.city', 'billing.zip.code', 'ticket_last_season', 'sub_last_season']
ALL_NUM = [c for c in df.columns if c not in ALL_CAT + ['account.id', 'split', 'label']]
y = train['label'].astype(int).values

TICKET = ['n_ticket_rows', 'ticket_total_seats', 'ticket_n_seasons',
          'ticket_last_season', 'ticket_n_2013_14', 'ticket_n_sets']
SUB = ['n_sub_rows', 'sub_total_seats', 'sub_n_seasons',
       'sub_last_season', 'sub_n_2013_14', 'sub_n_packages']
DONATION = ['amt_donated_2013', 'donor_before_2014',
            'first_donated_year_pre2014', 'donor_years_to_2014']
GEO = ['billing.city', 'billing.zip.code']

DEFAULT = dict(max_depth=6, learning_rate=0.3, n_estimators=100, random_state=42)
EXPERIMENTS = [
    ('xgb_default_recheck', {}, None),
    ('depth2', dict(max_depth=2), None),
    ('depth3', dict(max_depth=3), None),
    ('depth4', dict(max_depth=4), None),
    ('lr0.1_n300', dict(learning_rate=0.1, n_estimators=300), None),          
    ('lr0.05_n500', dict(learning_rate=0.05, n_estimators=500), None),          
    ('reg_lambda10', dict(reg_lambda=10), None),
    ('reg_alpha1', dict(reg_alpha=1), None),
    ('min_child_weight5', dict(min_child_weight=5), None),
    ('min_child_weight10', dict(min_child_weight=10), None),
    ('subsample_colsample0.8', dict(subsample=0.8, colsample_bytree=0.8), None),          
    ('combined_depth3_lr0.1_n300_reg5_mcw5_sample0.8',
     dict(max_depth=3, learning_rate=0.1, n_estimators=300, reg_lambda=5,
          min_child_weight=5, subsample=0.8, colsample_bytree=0.8), None),
    ('ablate_geo', {}, GEO),
    ('ablate_ticket', {}, TICKET),
    ('ablate_sub', {}, SUB),
    ('ablate_donation', {}, DONATION),
    ('only_ever_subscribed_and_donor', {}, None),                             
]

rows, oof = [], pd.DataFrame({'account.id': train['account.id'], 'label': y, 'fold': train['fold']})
for name, overrides, drop in EXPERIMENTS:
    if name == 'only_ever_subscribed_and_donor':
        cols = ['ever_subscribed', 'donor_before_2014']
        cats, nums = [], cols
    else:
        cols = [c for c in ALL_CAT + ALL_NUM if not (drop and c in drop)]
        cats = [c for c in ALL_CAT if c in cols]
        nums = [c for c in ALL_NUM if c in cols]
    X = train[cols]
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cats),
                             ('num', 'passthrough', nums)])
    params = {**DEFAULT, **overrides}
    model = Pipeline([('pre', pre), ('clf', XGBClassifier(**params))])
    preds = np.zeros(len(train))
    fold_aucs = []
    for f in range(5):
        tr = train['fold'] != f
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[~tr])[:, 1]
        preds[~tr] = p
        fold_aucs.append(roc_auc_score(y[~tr], p))
    pooled = roc_auc_score(y, preds)
    oof[name] = preds
    rows.append({'experiment': name, 'changed_from_default': str(overrides) or 'none',
                 'dropped': ','.join(drop) if drop else ('all-but-2' if name.startswith('only_') else 'none'),
                 'fold_aucs': ','.join(f'{a:.4f}' for a in fold_aucs),
                 'mean': round(float(np.mean(fold_aucs)), 4),
                 'std': round(float(np.std(fold_aucs)), 4),
                 'pooled_oof': round(float(pooled), 4)})
    print(f'{name}: folds {[round(a, 4) for a in fold_aucs]}, '
          f'mean {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}, pooled OOF {pooled:.4f}', flush=True)

res = pd.DataFrame(rows)
res.to_csv(f'{base}/stage9_results.csv', index=False)
oof.to_csv(f'{base}/stage9_oof.csv', index=False)
print(res[['experiment', 'mean', 'std', 'pooled_oof']].to_string(index=False))
print('LR baseline reference (stage 8): pooled OOF 0.9623, mean 0.9624')
