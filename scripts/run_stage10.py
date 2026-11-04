import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(f'{base}/features_clean_v1.csv', dtype={'billing.zip.code': str})
ss = pd.read_csv(f'{base}/data/for_students/sample_submission.csv')
train = df[df['split'] == 'train']
test = df[df['split'] == 'test']
assert len(train) == 6941 and len(test) == 2975
assert list(ss['ID']) == list(test['account.id']), 'sample order != test order!'

CAT = ['billing.city', 'billing.zip.code', 'ticket_last_season', 'sub_last_season']
NUM = [c for c in df.columns if c not in CAT + ['account.id', 'split', 'label']]
pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), CAT),
    ('num', StandardScaler(), NUM),
])
model = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
model.fit(train[CAT + NUM], train['label'].astype(int).values)
proba = model.predict_proba(test[CAT + NUM])[:, 1]

sub = pd.DataFrame({'ID': ss['ID'], 'Predicted': proba})
assert len(sub) == 2975 and sub['Predicted'].notna().all()
assert sub['Predicted'].between(0, 1).all()
sub.to_csv(f'{base}/submission_lr_v1.csv', index=False)
print(f'submission rows: {len(sub)}, ID order matches sample_submission: True')
print(f'Predicted: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}, '
      f'median={np.median(proba):.4f}')
print(f'predicted-positive@0.5: {(proba >= 0.5).sum()} / 2975')
print(f'Predicted>0.1: {(proba > 0.1).sum()}, >0.5: {(proba >= 0.5).sum()}, >0.9: {(proba > 0.9).sum()}')
