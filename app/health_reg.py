# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:25:11 2025

@author: dell
"""
import statsmodels.api as sm
import pandas as pd
from fredapi import Fred


panel = pd.read_csv(r'data/final/fdic_2014_2019.csv')
bank_info_df = pd.read_csv(r'data/final/bank_info.csv')
snp_df = pd.read_csv(r'data/final/snp.csv')
mkt_value = pd.read_csv(r'data/final/market_value.csv')

mkt_value.rename(columns={'LPERMNO': 'PERMNO', 'datadate': 'date'}, inplace=True)
snp_df.sort_values(['date', 'PERMNO'], inplace=True)
snp_df['PERMNO'] = snp_df['PERMNO'].astype(int)
mkt_value = mkt_value[['PERMNO', 'date', 'mkvaltq']]
mkt_value.sort_values(['date', 'PERMNO'], inplace=True)
snp_df = snp_df.merge(mkt_value, on=['date', 'PERMNO'], how='inner')
# ------------------------------------------------------------------
# 0.  Imports
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# ------------------------------------------------------------------
# 1.  Build a quarterly BANK-HEALTH factor 2010-Q1 … 2019-Q4
# ------------------------------------------------------------------
metrics_pos = ['RBC1AAJ', 'IDT1RWAJR', 'ROA', 'NIMY']       # “higher is good”
metrics_neg = ['NCLNLSR', 'EEFFR']                       # “lower is good”

for v in metrics_pos + metrics_neg:
    panel[v] = pd.to_numeric(panel[v], errors='coerce')

# z-scores
for v in metrics_pos + metrics_neg:
    panel[v + '_z'] = (panel[v] - panel[v].mean()) / panel[v].std(ddof=0)
for v in metrics_neg:
    panel[v + '_z'] *= -1                                 # flip the bad ones

panel['health_z'] = panel[[f'{v}_z' for v in metrics_pos + metrics_neg]].mean(axis=1)

# one value per quarter, asset-weighted across all banks
sys_health_q = (panel
                .assign(w = panel['ASSET'] /
                             panel.groupby('REPDTE')['ASSET'].transform('sum'))
                .groupby('REPDTE')
                .apply(lambda g: (g['health_z'] * g['w']).sum())
                .to_frame('bank_health'))

sys_health_q.index.name = 'date'          # call the index ‘date’
sys_health_q.index = pd.to_datetime(sys_health_q.index)
sys_health_q = sys_health_q.sort_index()

# --- 2. CRSP → QUARTERLY TOTAL RETURN ---------------------------------
snp_df['PRC'] = snp_df['PRC'].abs()
snp_df = snp_df.sort_values(['PERMNO', 'date'])
snp_df['ret_m'] = (snp_df['PRC'] + snp_df['DIVAMT']) / snp_df.groupby('PERMNO')['PRC'].shift() - 1
snp_df = snp_df.dropna(subset=['ret_m'])

snp_df['date'] = pd.to_datetime(snp_df['date'])
# quarter-end cumulative return
qret = (snp_df
        .set_index('date')
        .groupby('PERMNO')['ret_m']
        .apply(lambda x: (1 + x).resample('QE-DEC').prod() - 1)
        .dropna()
        .reset_index()
        .rename(columns={'ret_m': 'ret_q'}))

# --- 3. Firm controls --------------------------------------------------
# lagged log-size
snp_df['mkt_cap'] = snp_df['mkvaltq'] * 1000
size = (snp_df.set_index('date')
                .groupby('PERMNO')['mkt_cap']
                .resample('QE-DEC').last()
                .shift()
                .pipe(np.log)
                .rename('size_lag')
                .reset_index())



# Compute rolling 12-month momentum by PERMNO
mom_raw = (snp_df
           .sort_values(['PERMNO', 'date'])
           .groupby('PERMNO')
           .apply(lambda g: pd.DataFrame({
               'PERMNO': g['PERMNO'].values,  # explicitly re-add PERMNO
               'date': g['date'].values,
               'mom': (1 + g['ret_m']).rolling(12).apply(np.prod, raw=True) - 1
           }))
           .reset_index(drop=True))  # drop groupby index


# Now resample to quarterly frequency
mom = (mom_raw
       .set_index('date')
       .groupby('PERMNO')['mom']
       .resample('Q')  # or 'QE-DEC' for calendar quarters ending Dec, Mar, Jun, Sep
       .last()
       .shift()  # lag by one quarter
       .rename('mom_12_1')
       .reset_index())

# qret = (qret.merge(size, on=['PERMNO', 'date'])
#             .merge(mom,  on=['PERMNO', 'date']))

# # --- 4. Δ BankHealth (surprise) ---------------------------------
sys_health_q = sys_health_q.asfreq('QE-DEC')              # ensure quarterly index
sys_health_q['d_bh'] = sys_health_q['bank_health'].diff()

# --- after building size_lag -------------------------------------
qret = qret.merge(size, on=['PERMNO','date'])

fred = Fred(api_key='db2aa0d51078fff05263d27cd2efa2ef')      # replace if needed

start, end = '2009-01-01', '2019-12-31'
tbill        = fred.get_series('TB3MS',   start, end)   # 3-month T-bill rate (% p.a.)
term_spread  = fred.get_series('T10Y3M',  start, end)   # 10Y – 3M (%)
vix          = fred.get_series('VIXCLS',  start, end)   # VIX level

macro_q = (pd.concat([tbill, term_spread, vix], axis=1)
             .rename(columns={'0':'tbill',
                              '1':'term_spread',
                              '2':'vix'})
             .resample('QE-DEC').mean()   # quarter-end average
             .shift(1))                   # lag one quarter

macro_q.index.name = 'date'

macro_q = macro_q.reset_index()          # <-- so merge finds the column

# macro_q.dropna(inplace=True)
macro_q.columns = ['date', 'tbill', 'term_spread', 'vix']

# ---------------------------------------------------------------
# 3-C  MERGE EVERYTHING ONCE  -----------------------------------
# ---------------------------------------------------------------

# qret ALREADY contains size_lag; do NOT merge `size` again!
df = (qret
      .merge(sys_health_q[['d_bh']], on='date', how='left')
      .merge(macro_q,                on='date', how='left'))

# ---------  keep only rows with all required vars  -------------
needed = ['ret_q', 'd_bh', 'size_lag', 'tbill', 'term_spread', 'vix']
df = df.dropna(subset=needed)

# ---------  create lagged systemic surprise  -------------------
df['d_bh_lag1'] = df.groupby('PERMNO')['d_bh'].shift(1)
df = df.dropna(subset=['d_bh_lag1'])



Y = df.set_index(['PERMNO','date'])['ret_q']
X = sm.add_constant(df.set_index(['PERMNO','date'])
                    [['d_bh_lag1','size_lag',
                      'tbill','term_spread','vix']])


res = PanelOLS(Y, X, entity_effects=True).fit(
        cov_type='clustered', cluster_entity=True, cluster_time=True)
print(res.summary)


import matplotlib.pyplot as plt


coef = res.params.to_frame('coef')
ci   = res.conf_int()
coef[['ci_low','ci_high']] = ci
print(coef.round(4))

fig, ax = plt.subplots()
sys_health_q['bank_health'].plot(ax=ax, marker='o')
ax.axhline(0, lw=.8)
ax.set(title='Systemic Bank-Health Index (2010-2019)',
       ylabel='Asset-weighted z-score')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
coef.reset_index(inplace=True)
ax.errorbar(coef['index'], coef['coef'],
            yerr=[coef['coef']-coef['ci_low'],
                  coef['ci_high']-coef['coef']],
            fmt='o', capsize=4)
ax.set_xticklabels(coef['index'], rotation=45, ha='right')
ax.set(title='Estimated coefficients with 95% CI', ylabel='β')
plt.tight_layout()
plt.show()


df['size_q'] = pd.qcut(df['size_lag'], 4, labels=False)+1
g = df.groupby('size_q')
betas = []
for q, sub in g:
    y = sub.set_index(['PERMNO','date'])['ret_q']
    x = sm.add_constant(sub.set_index(['PERMNO','date'])
                        [['d_bh_lag1']])
    b = PanelOLS(y, x, entity_effects=True).fit()
    betas.append(b.params['d_bh_lag1'])
plt.plot([1,2,3,4], betas, marker='o')
plt.title('β(d_bh_lag1) across size quartiles')
plt.xlabel('Size quartile (1=small)')
plt.ylabel('Coefficient')
plt.tight_layout()
plt.show()


snp_df['sic4'] = snp_df['SICCD'].astype(str).str.zfill(4)
snp_df['sector10'] = snp_df['sic4'].str[0]
snp_df['sector10'] = snp_df['sector10'].astype('category')
# snp_df['small'] = (snp_df.groupby('date')['size_lag'])