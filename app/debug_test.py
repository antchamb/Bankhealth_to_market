# -*- coding: utf-8 -*-
"""
Bank-health shock × industry × size, 2010-Q1 … 2019-Q4
"""

# ------------------------------------------------------------------
# 0.  Imports & raw data
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from fredapi import Fred

panel      = pd.read_csv('data/final/fdic_2014_2019.csv', low_memory=False)
bank_info  = pd.read_csv('data/final/bank_info.csv', low_memory=False)      # not used below
snp_df     = pd.read_csv('data/final/snp.csv', low_memory=False)
mkt_value  = pd.read_csv('data/final/market_value.csv', low_memory=False)

# ------------------------------------------------------------------
# 1.  Market-value & SIC pre-processing  (*** new lines here ***)
# ------------------------------------------------------------------
mkt_value.rename(columns={'LPERMNO':'PERMNO', 'datadate':'date'}, inplace=True)
mkt_value = mkt_value[['PERMNO','date','mkvaltq']]

snp_df.sort_values(['date','PERMNO'], inplace=True)
snp_df['PERMNO'] = snp_df['PERMNO'].astype(int)

snp_df = snp_df.merge(mkt_value, on=['PERMNO','date'], how='inner')

# ------------ add SIC-based industry bucket -----------------------
snp_df['sic4']     = snp_df['SICCD'].astype(str).str.zfill(4)
snp_df['sector10'] = snp_df['sic4'].str[0].astype('category')   # '0'–'9'

# ------------------------------------------------------------------
# 2.  Build systemic “bank-health” factor
# ------------------------------------------------------------------
metrics_pos = ['RBC1AAJ','IDT1RWAJR','ROA','NIMY']   # good ↑
metrics_neg = ['NCLNLSR','EEFFR']                   # good ↓

for v in metrics_pos + metrics_neg:
    panel[v] = pd.to_numeric(panel[v], errors='coerce')

for v in metrics_pos + metrics_neg:
    panel[v + '_z'] = (panel[v] - panel[v].mean()) / panel[v].std(ddof=0)
for v in metrics_neg:
    panel[v + '_z'] *= -1

panel['health_z'] = panel[[f'{v}_z' for v in metrics_pos + metrics_neg]].mean(axis=1)

sys_health_q = (panel
    .assign(w=panel['ASSET'] / panel.groupby('REPDTE')['ASSET'].transform('sum'))
    .groupby('REPDTE')
    .apply(lambda g: (g['health_z'] * g['w']).sum(), include_groups=False)
    .to_frame('bank_health'))

sys_health_q.index.name = 'date'
sys_health_q.index      = pd.to_datetime(sys_health_q.index).to_period('Q').to_timestamp('Q')
sys_health_q            = sys_health_q.sort_index()
sys_health_q['d_bh']    = sys_health_q['bank_health'].diff()

# ------------------------------------------------------------------
# 3.  CRSP → quarterly return, lagged size, momentum --------------
# ------------------------------------------------------------------
snp_df['PRC'] = snp_df['PRC'].abs()
snp_df = snp_df.sort_values(['PERMNO','date'])
snp_df['ret_m'] = (snp_df['PRC'] + snp_df['DIVAMT']) / snp_df.groupby('PERMNO')['PRC'].shift() - 1
snp_df          = snp_df.dropna(subset=['ret_m'])
snp_df['date']  = pd.to_datetime(snp_df['date'])

# --- total return per quarter (calendar Q-ends) -------------------
qret = (snp_df.set_index('date')
              .groupby('PERMNO')['ret_m']
              .apply(lambda x: (1+x).resample('QE-DEC').prod() - 1)
              .dropna()
              .reset_index()
              .rename(columns={'ret_m':'ret_q'}))

# --- lagged log-size (quarter-end) --------------------------------
snp_df['mkt_cap'] = snp_df['mkvaltq'] * 1000           # $ to $1,000
size = (snp_df.set_index('date')
                .groupby('PERMNO')['mkt_cap']
                .resample('QE-DEC').last()
                .shift()
                .pipe(np.log)
                .rename('size_lag')
                .reset_index())

# add sector10 (static per PERMNO; take first obs)
sector_map = (snp_df[['PERMNO','sector10']]
              .drop_duplicates(subset='PERMNO')
              .set_index('PERMNO')['sector10'])

# ------------------------------------------------------------------
# 4.  Macro controls (lagged) --------------------------------------
# ------------------------------------------------------------------
fred   = Fred(api_key='db2aa0d51078fff05263d27cd2efa2ef')
start, end = '2009-01-01','2019-12-31'
tbill  = fred.get_series('TB3MS',  start, end)
tsprd  = fred.get_series('T10Y3M', start, end)
vix    = fred.get_series('VIXCLS', start, end)

macro_q = (pd.concat([tbill, tsprd, vix], axis=1)
             .rename(columns={'0':'tbill',
                              '1':'term_spread',
                              '2':'vix'})
             .resample('QE-DEC').mean()   # quarter-end average
             .shift(1))

macro_q.index.name = 'date'

macro_q = macro_q.reset_index()          # <-- so merge finds the column

# macro_q.dropna(inplace=True)
macro_q.columns = ['date', 'tbill', 'term_spread', 'vix']

# ------------------------------------------------------------------
# 5.  Merge everything
# ------------------------------------------------------------------
df = (qret
      .merge(size,     on=['PERMNO','date'])
      .merge(sys_health_q[['d_bh']],        on='date', how='left')
      .merge(macro_q,                       on='date', how='left'))

# add industry bucket
df['sector10'] = df['PERMNO'].map(sector_map)
df = df.dropna(subset=['sector10'])           # keep obs we can classify

# keep only complete rows
req = ['ret_q','d_bh','size_lag','tbill','term_spread','vix']
df  = df.dropna(subset=req)

# lagged systemic shock
df['d_bh_lag1'] = df.groupby('PERMNO')['d_bh'].shift(1)
df = df.dropna(subset=['d_bh_lag1'])

# ------------------------------------------------------------------
# 6.  *** NEW *** small-cap dummy & interaction terms --------------
# ------------------------------------------------------------------
df['small'] = (df.groupby('date')['size_lag']
                 .transform(lambda s: s < s.median())).astype(int)

# sector × shock interactions
for s in df['sector10'].cat.categories:
    df[f'd_bh_sec{s}'] = df['d_bh_lag1'] * (df['sector10'] == s).astype(int)

# small-cap interaction
df['d_bh_small'] = df['d_bh_lag1'] * df['small']

# ------------------------------------------------------------------
# 7.  Panel regression with entity FE & 2-way clustered SE ---------
# ------------------------------------------------------------------
Xcols = ['d_bh_lag1', 'd_bh_small'] + \
        [f'd_bh_sec{s}' for s in df['sector10'].cat.categories] + \
        ['size_lag','tbill','term_spread','vix']

Y = df.set_index(['PERMNO', 'date'])['ret_q']
X = sm.add_constant(df.set_index(['PERMNO', 'date'])[Xcols])

# Check if indices are aligned
if not Y.index.equals(X.index):
    raise ValueError("Indices of Y and X are not aligned. Ensure they have the same multi-index.")
# Fit the PanelOLS model with drop_absorbed=True
res = PanelOLS(Y, X, entity_effects=True, check_rank=False, drop_absorbed=True).fit(
    cov_type='clustered', cluster_entity=True, cluster_time=True
)

print(res.summary)



# ------------------------------------------------------------------
# 8.  Joint F-test for H2 (all industry deltas = 0) ----------------
hypo = ' = '.join([f'd_bh_sec{s}' for s in df['sector10'].cat.categories]) + ' = 0'
print("\nJoint test of sector heterogeneity (H2):")
print(res.f_statistic(hypo))

# ------------------------------------------------------------------
# 9.  Optional visualisations --------------------------------------
coef = res.params.to_frame('coef')
ci   = res.conf_int()
coef[['ci_low','ci_high']] = ci
print("\nPoint estimates & 95% CI:\n", coef.round(4))

# --- plot systemic index ------------------------------------------
fig, ax = plt.subplots()
sys_health_q['bank_health'].plot(ax=ax, marker='o')
ax.axhline(0, lw=.8)
ax.set(title='Systemic Bank-Health Index (2010-2019)',
       ylabel='Asset-weighted z-score')
plt.tight_layout(); plt.show()

# --- coefficient bar with error bars ------------------------------
fig, ax = plt.subplots()
coef.reset_index(inplace=True)
ax.errorbar(coef['index'], coef['coef'],
            yerr=[coef['coef']-coef['ci_low'],
                  coef['ci_high']-coef['coef']],
            fmt='o', capsize=4)
ax.set_xticklabels(coef['index'], rotation=45, ha='right')
ax.set(title='Estimated coefficients with 95% CI', ylabel='β')
plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
# 10.  (Optional) Re-run by size quartiles & plot beta profile -----
# ------------------------------------------------------------------
df['size_q'] = pd.qcut(df['size_lag'], 4, labels=False) + 1
betas = []
for q, sub in df.groupby('size_q'):
    y = sub.set_index(['PERMNO','date'])['ret_q']
    x = sm.add_constant(sub.set_index(['PERMNO','date'])[['d_bh_lag1']])
    b = PanelOLS(y, x, entity_effects=True).fit()
    betas.append(b.params['d_bh_lag1'])

plt.plot([1,2,3,4], betas, marker='o')
plt.title('β(d_bh_lag1) across size quartiles')
plt.xlabel('Size quartile (1 = small)')
plt.ylabel('Coefficient')
plt.tight_layout(); plt.show()
