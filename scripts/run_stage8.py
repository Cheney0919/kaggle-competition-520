import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(f'{base}/features_clean_v1.csv', dtype={'billing.zip.code': str})
folds = pd.read_csv(f'{base}/train_folds.csv')
train = df[df['split'] == 'train'].merge(folds, on=['account.id', 'label'])
CAT = ['billing.city', 'billing.zip.code', 'ticket_last_season', 'sub_last_season']
NUM = [c for c in df.columns if c not in CAT + ['account.id', 'split', 'label']]
X, y = train[CAT + NUM], train['label'].astype(int).values

def make_pre(scale):
    trans = [('cat', OneHotEncoder(handle_unknown='ignore'), CAT)]
    trans.append(('num', StandardScaler() if scale else 'passthrough', NUM))
    return ColumnTransformer(trans)

models = {
    'logistic_baseline': Pipeline([('pre', make_pre(True)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'xgboost_default': Pipeline([('pre', make_pre(False)),
        ('clf', XGBClassifier(random_state=42))]),
}
oof = pd.DataFrame({'account.id': train['account.id'], 'label': y, 'fold': train['fold']})
for name, model in models.items():
    preds = np.zeros(len(train))
    fold_aucs = []
    for f in range(5):
        tr = train['fold'] != f
        te = ~tr
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        preds[te] = p
        fold_aucs.append(roc_auc_score(y[te], p))
    oof[name] = preds
    print(f'{name}: fold AUROCs = {[round(a, 4) for a in fold_aucs]}, '
          f'mean = {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}, '
          f'pooled OOF = {roc_auc_score(y, preds):.4f}')
oof.to_csv(f'{base}/stage8_oof.csv', index=False)
                                                                    
n_feat = make_pre(True).fit(X).transform(X[:5]).shape[1]
print(f'features after one-hot: {n_feat} (cats one-hot + {len(NUM)} numerics)')
print(f'rows: {len(train)}, positives: {y.sum()} ({y.mean():.3f})')
