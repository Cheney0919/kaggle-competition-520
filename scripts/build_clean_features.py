import csv, os, random, re
from collections import Counter, defaultdict

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base, 'features_strict_v1.csv'), newline='', encoding='utf-8') as fh:
    rows = list(csv.DictReader(fh))
train = [r for r in rows if r['split'] == 'train']
assert len(rows) == 9916 and len(train) == 6941

def clean_zip(z):
    z = (z or '').strip()
    if not z:
        return 'Unknown'
    if re.fullmatch(r'\d{5}-\d{4}', z):
        return z[:5]
    if z.isdigit():
        return z[:5] if len(z) >= 5 else z.rjust(5, '0')
    return 'Unknown'

                            
city_counts = Counter(r['billing.city'] for r in train if r['billing.city'])
zip_counts = Counter(clean_zip(r['billing.zip.code']) for r in train
                     if clean_zip(r['billing.zip.code']) != 'Unknown')
city_keep = {c for c, k in city_counts.items() if k >= 20}
zip_keep = {z for z, k in zip_counts.items() if k >= 20}

out = []
for r in rows:
    o = dict(r)
    o.pop('relationship', None)
    city = r['billing.city']
    o['billing.city'] = 'Unknown' if not city else (city if city in city_keep else 'Other')
    zc = clean_zip(r['billing.zip.code'])
    o['billing.zip.code'] = 'Unknown' if zc == 'Unknown' else (zc if zc in zip_keep else 'Other')
    o['ticket_last_season'] = r['ticket_last_season'] or 'None'
    o['sub_last_season'] = r['sub_last_season'] or 'None'
    o['first_donated_year_pre2014'] = r['first_donated_year_pre2014'] or '0'
    o['donor_years_to_2014'] = r['donor_years_to_2014'] or '0'
    out.append(o)

fields = [f for f in out[0].keys()]
assert all(all(o[f] != '' for f in fields if not (o['split'] == 'test' and f == 'label')) for o in out), 'blanks remain!'                              
with open(os.path.join(base, 'features_clean_v1.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader(); w.writerows(out)

                                              
rng = random.Random(42)
by_label = defaultdict(list)
for r in train:
    by_label[r['label']].append(r['account.id'])
fold_of = {}
for label, ids in by_label.items():
    rng.shuffle(ids)
    for i, aid in enumerate(ids):
        fold_of[aid] = i % 5
with open(os.path.join(base, 'train_folds.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(['account.id', 'label', 'fold'])
    for r in train:
        w.writerow([r['account.id'], r['label'], fold_of[r['account.id']]])

print(f'clean rows: {len(out)}, cols: {len(fields)} (relationship dropped)')
print(f'city: 838 raw -> {len(set(o["billing.city"] for o in out))} values '
      f'({len(city_keep)} kept + Other + Unknown)')
print(f'zip: 1359 raw -> {len(set(o["billing.zip.code"] for o in out))} values '
      f'({len(zip_keep)} kept + Other + Unknown)')
print('fold table (train only):')
for f in range(5):
    ids = [r for r in train if fold_of[r['account.id']] == f]
    pos = sum(int(r['label']) for r in ids)
    print(f'  fold {f}: n={len(ids)}, pos={pos}, rate={pos/len(ids):.3f}')
print('columns:', ', '.join(fields))
