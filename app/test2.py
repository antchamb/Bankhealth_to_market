import pandas as pd, numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from pathlib import Path

# ➊  Régler l’option d’affichage
pd.set_option('display.float_format', '{:.4f}'.format)

# ---------------- FICHIERS ----------------
fdic   = pd.read_csv('data/final/fdic_2014_2019.csv',  low_memory=False)
crsp   = pd.read_csv('data/final/snp.csv',             low_memory=False)
mkv    = pd.read_csv('data/final/market_value.csv',    low_memory=False)

# -------- Banque : indicateur de santé agrégé --------
pos = ['RBC1AAJ','IDT1RWAJR','ROA','NIMY']
neg = ['NCLNLSR','EEFFR']
for v in pos + neg:
    fdic[v] = pd.to_numeric(fdic[v], errors='coerce')
    fdic[v + '_z'] = (fdic[v] - fdic[v].mean()) / fdic[v].std(ddof=0)
for v in neg:
    fdic[v + '_z'] *= -1
fdic['health_z'] = fdic[[v + '_z' for v in pos+neg]].mean(axis=1)

sys_bh = (fdic.assign(w = fdic['ASSET'] /
                           fdic.groupby('REPDTE')['ASSET'].transform('sum'))
                 .groupby('REPDTE')
                 .apply(lambda g: (g['health_z']*g['w']).sum(), include_groups=False)
                 .to_frame('bank_health'))
sys_bh.index = pd.to_datetime(sys_bh.index).to_period('Q').to_timestamp('Q')
sys_bh['d_bh'] = sys_bh['bank_health'].diff()

# -------- CRSP : rendement trimestriel & taille retardée --------
mkv.rename(columns={'LPERMNO':'PERMNO','datadate':'date'}, inplace=True)
crsp = (crsp.merge(mkv[['PERMNO','date','mkvaltq']], on=['PERMNO','date'])
             .assign(PERMNO = lambda x: x['PERMNO'].astype(int)))
crsp['PRC']     = crsp['PRC'].abs()
crsp['ret_m']   = (crsp['PRC'] + crsp['DIVAMT']) / crsp.groupby('PERMNO')['PRC'].shift() - 1
crsp            = crsp.dropna(subset=['ret_m'])
crsp['date']    = pd.to_datetime(crsp['date'])

qret = (crsp.set_index('date').groupby('PERMNO')['ret_m']
              .apply(lambda x: (1+x).resample('QE').prod() - 1)
              .dropna().rename('ret_q').reset_index())

crsp['mkt_cap'] = crsp['mkvaltq'] * 1_000
size = (crsp.set_index('date').groupby('PERMNO')['mkt_cap']
              .resample('QE').last().shift().pipe(np.log)
              .rename('size_lag').reset_index())

# -------- Secteur (premier digit SIC) ----------------
crsp['sector'] = crsp['SICCD'].astype(str).str.zfill(4).str[0]
sector_map = crsp.drop_duplicates('PERMNO').set_index('PERMNO')['sector']

# -------- Macros + Fama-French -----------------------
ff = pd.read_csv('data/final/ff_data.csv', parse_dates=['date'])        # Mkt-RF, SMB, HML, MOM, RF
ff['date'] = pd.to_datetime(ff['date'])
mom = pd.read_csv('data/final/mom_data.csv', parse_dates=['date'])
mom['date'] = pd.to_datetime(mom['date'])
ff = ff.merge(mom, on='date', how='left')  # Add MOM to Fama-French data


# -------- Merge principal ----------------------------
# Reset the index of sys_bh and ensure "d_bh" is included
sys_bh = sys_bh.reset_index().rename(columns={'REPDTE': 'date'})

# Ensure 'date' is properly formatted as datetime
sys_bh['date'] = pd.to_datetime(sys_bh['date'])

# Merge principal
base = (qret.merge(size, on=['PERMNO', 'date'])
             .merge(sys_bh[['date', 'd_bh']], on='date')  # Ensure 'd_bh' is included
             .merge(ff, on='date')
             .assign(sector=lambda d: d['PERMNO'].map(sector_map))
             .dropna())

# rendements excédentaires
base['rx_q'] = base['ret_q'] - base['RF']

# décaler le choc bancaire
base['d_bh_lag1'] = base.groupby('PERMNO')['d_bh'].shift(1)
base = base.dropna(subset=['d_bh_lag1'])

def dk_panel(y, X, entity=True, time=False, bw=4):
    """Estimateur FE + erreurs Driscoll–Kraay."""
    mod = PanelOLS(y, X, entity_effects=entity, time_effects=time)
    return mod.fit(cov_type='driscoll-kraay',
                   kernel='bartlett', bandwidth=bw)

def prep_xy(df, xcols):
    y = df.set_index(['PERMNO','date'])['rx_q']
    X = sm.add_constant(df.set_index(['PERMNO','date'])[xcols])
    return y, X

x1 = ['d_bh_lag1','Mkt-RF','SMB','HML','MOM','size_lag']
y1, X1 = prep_xy(base, x1)
res1 = dk_panel(y1, X1)
print(res1.summary)

base['small']      = (base.groupby('date')['size_lag']
                          .transform(lambda s: s < s.median())).astype(int)
base['d_bh_small'] = base['d_bh_lag1'] * base['small']

x2 = ['d_bh_lag1','d_bh_small','Mkt-RF','SMB','HML','MOM','size_lag']
res2 = dk_panel(*prep_xy(base, x2))
print(res2.summary)

def prep_xy_for_h2(df, xcols):
    y = df.set_index(['PERMNO', 'date'])['rx_q']
    X = sm.add_constant(df.set_index(['PERMNO', 'date'])[xcols])

    # Drop linearly dependent columns
    # Compute the rank of X and identify independent columns
    independent_cols = np.linalg.matrix_rank(X.values)
    if independent_cols < X.shape[1]:
        # Drop columns with perfect multicollinearity
        X = X.loc[:, ~X.columns.duplicated()]

    return y, X

    return y, X
def dk_panel_for_h2(y, X, entity=True, time=False, bw=4):
    """Estimateur FE + erreurs Driscoll–Kraay."""
    # Ensure proper alignment and no missing data
    y, X = y.align(X, join='inner', axis=0)

    # Debugging information
    print("y shape:", y.shape)
    print("X shape:", X.shape)
    print("y index levels:", y.index.names)
    print("X index levels:", X.index.names)

    # Fit the model
    mod = PanelOLS(y, X, entity_effects=entity, time_effects=time)
    return mod.fit(cov_type='driscoll-kraay', kernel='bartlett', bandwidth=bw)
# ------------------ INTERACTIONS SECTEUR (colinéarité corrigée) -------------
# 1) Interactions secteur (SECTEUR '0' = référence)
ref = '0'
sector_dummies = []
for s in sorted(base['sector'].unique()):
    if s != ref and base['sector'].eq(s).any():          # au moins une obs.
        var = f'd_bh_sec{s}'
        base[var] = base['d_bh_lag1'] * (base['sector'] == s).astype(int)
        sector_dummies.append(var)

# 2) Matrice des régressors : ***PAS*** d_bh_lag1, ***PAS*** de constante
x3 = sector_dummies + ['Mkt-RF','SMB','HML','MOM','size_lag']

y3 = base.set_index(['PERMNO','date'])['rx_q']
X3 = base.set_index(['PERMNO','date'])[x3]

# 1 ) construire le modèle en précisant drop_absorbed ici
mod3 = PanelOLS(y3, X3, entity_effects=True, drop_absorbed=True)

# 2 ) appeler fit sans ce mot-clé
res3 = mod3.fit(cov_type='driscoll-kraay',
                kernel='bartlett', bandwidth=4)


print(res3.summary)

import numpy as np

# Verify parameter names
print("Model parameters:", res3.params.index)

# Construct the hypothesis matrix
num_dummies = len(sector_dummies)
restriction_matrix = np.zeros((num_dummies, len(res3.params)))
for i, var in enumerate(sector_dummies):
    restriction_matrix[i, res3.params.index.get_loc(var)] = 1

# Perform the Wald test
f_test = res3.wald_test(restriction_matrix)
print("\nTest conjoint H2: Statistic =", f_test.stat, ", P-value =", f_test.pval)