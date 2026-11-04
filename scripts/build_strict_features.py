import csv, os
from collections import defaultdict

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'for_students')
out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'features_strict_v1.csv')

def read(name):
    with open(os.path.join(base, name), newline='', encoding='utf-8-sig', errors='replace') as fh:
        return list(csv.DictReader(fh))

account = {r['account.id']: r for r in read('account.csv')}
train = read('train.csv')
test = read('test.csv')
tickets = read('tickets_all.csv')
subs = read('subscriptions.csv')

                                         
assert max(r['season'] for r in tickets) == '2013-2014', 'ticket season beyond 2013-2014!'
assert max(r['season'] for r in subs) == '2013-2014', 'sub season beyond 2013-2014!'
assert not any(r['season'] == '2014-2015' for r in tickets + subs)

                             
t_agg = defaultdict(lambda: {'n': 0, 'seats': 0, 'seasons': set(), 'n_1314': 0, 'sets': set()})
for r in tickets:
    a = t_agg[r['account.id']]
    a['n'] += 1
    a['seats'] += int(r['no.seats'] or 0)
    a['seasons'].add(r['season'])
    a['sets'].add(r['set'])
    if r['season'] == '2013-2014':
        a['n_1314'] += 1

s_agg = defaultdict(lambda: {'n': 0, 'seats': 0, 'seasons': set(), 'n_1314': 0, 'pkgs': set()})
for r in subs:
    a = s_agg[r['account.id']]
    a['n'] += 1
    a['seats'] += int(r['no.seats'] or 0)
    a['seasons'].add(r['season'])
    a['pkgs'].add(r['package'])
    if r['season'] == '2013-2014':
        a['n_1314'] += 1

                                                         
rows = []
for r in train:
    rows.append((r['account.id'], 'train', r['label']))
for r in test:
    rows.append((r['ID'], 'test', ''))

ids = [x[0] for x in rows]
assert len(ids) == len(set(ids)) == 9916, f'coverage problem: {len(ids)} / {len(set(ids))}'
assert all(i in account for i in ids), 'some train/test id missing from account.csv'

out = []
for aid, split, label in rows:
    ac = account[aid]
    fd = ac['first.donated'].strip()
    donor_pre14 = 1 if (fd and fd < '2014-01-01') else 0
    fd_year = int(fd[:4]) if donor_pre14 else ''
    t = t_agg.get(aid, {'n': 0, 'seats': 0, 'seasons': set(), 'n_1314': 0, 'sets': set()})
    s = s_agg.get(aid, {'n': 0, 'seats': 0, 'seasons': set(), 'n_1314': 0, 'pkgs': set()})
    out.append({
        'account.id': aid, 'split': split, 'label': label,
        'billing.city': ac['billing.city'], 'billing.zip.code': ac['billing.zip.code'],
        'relationship': ac['relationship'],
        'amt_donated_2013': ac['amount.donated.2013'],
        'donor_before_2014': donor_pre14,
        'first_donated_year_pre2014': fd_year,
        'donor_years_to_2014': (2014 - fd_year) if donor_pre14 else '',
        'n_ticket_rows': t['n'], 'ticket_total_seats': t['seats'],
        'ticket_n_seasons': len(t['seasons']),
        'ticket_last_season': max(t['seasons']) if t['seasons'] else '',
        'ticket_n_2013_14': t['n_1314'], 'ticket_n_sets': len(t['sets']),
        'n_sub_rows': s['n'], 'sub_total_seats': s['seats'],
        'sub_n_seasons': len(s['seasons']),
        'sub_last_season': max(s['seasons']) if s['seasons'] else '',
        'sub_n_2013_14': s['n_1314'], 'sub_n_packages': len(s['pkgs']),
        'ever_subscribed': 1 if s['n'] > 0 else 0,
    })

fields = list(out[0].keys())
with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(out)

                              
n_no_ticket = sum(1 for o in out if o['n_ticket_rows'] == 0)
n_no_sub = sum(1 for o in out if o['n_sub_rows'] == 0)
n_donor_pre14 = sum(1 for o in out if o['donor_before_2014'] == 1)
print(f'rows: {len(out)} (train {sum(1 for o in out if o["split"]=="train")}, '
      f'test {sum(1 for o in out if o["split"]=="test")}), cols: {len(fields)}')
print(f'no lifetime columns: {"amount.donated.lifetime" not in fields and "no.donations.lifetime" not in fields}')
print(f'donor_before_2014=1: {n_donor_pre14}; masked 2014 first-donors: 126 in account, 0 kept')
print(f'no tickets: {n_no_ticket}; no subs: {n_no_sub}')
print(f'ticket_last_season max: {max(o["ticket_last_season"] for o in out if o["ticket_last_season"])}')
print(f'sub_last_season max: {max(o["sub_last_season"] for o in out if o["sub_last_season"])}')
print('columns:', ', '.join(fields))
