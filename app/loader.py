from pathlib import Path
import pandas as pd
import csv

# =============================================================================
#                   FDIC DATA
# =============================================================================

ROOT   = Path(r"data/raw/sdi_zips")
YEARS  = range(2010, 2020)


def sniff(path, n=65536) -> str:
    """Return ',' or '|' – use '|' when comma would yield one big column."""
    with path.open('rb') as fh:
        sample = fh.read(n).decode('latin-1', errors='ignore')
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=[',', '|']).delimiter
    except csv.Error:
        delim = '|'
    if delim == ',' and sample.count(',') == 0 and sample.count('|') > 0:
        delim = '|'
    return delim

def norm(col: str) -> str:
    return col.strip().upper()

def load_one(csv_path: Path) -> pd.DataFrame:
    """Read a single file (pipe or comma, latin-1) and normalise headers."""
    df = pd.read_csv(csv_path,
                     sep=sniff(csv_path),
                     encoding='latin-1',
                     low_memory=False)
    df.columns = [norm(c) for c in df.columns]
    return df


qdirs = []
for y in YEARS:
    year_dir = ROOT / str(y)
    qdirs.extend(sorted(year_dir.glob("id-sdi-*")))

assert qdirs, "No quarter folders found – check extraction path."


frames = []
for qdir in qdirs:

    try:
        al_csv  = next(qdir.glob("*Assets and Liabilities.csv"))
        pcr_csv = next(qdir.glob("*Performance and Condition Ratios.csv"))
    except StopIteration:
        raise FileNotFoundError(f"  Missing A&L or PCR file in {qdir}")

    al  = load_one(al_csv)
    pcr = load_one(pcr_csv)

    # make sure CERT & REPDTE exist in both
    key = ['CERT', 'REPDTE']
    if not all(k in al.columns for k in key) or not all(k in pcr.columns for k in key):
        raise KeyError(f"CERT/REPDTE missing in {qdir.name}")

    merged = pd.merge(al, pcr, on=key, how='outer', suffixes=('_AL', '_PCR'))
    frames.append(merged)


panel = pd.concat(frames, ignore_index=True)

# panel.dropna(subset='IDT1CER', inplace = True)
# -->  start at 2014


bank_info = ['CERT', 'REPDTE', 'NAME_AL', 'CITY_AL', 'STALP_AL', 'ZIP_AL', 'NAMEHCR_AL', 'COUNTY_AL']
bank_info = panel[bank_info]

TARGET = ['CERT','REPDTE','ASSET','IDT1CER','RBC1AAJ','ROA','ROE','NCLNLSR', 'FREPO', 'DEP', 'NIMY', 'EEFFR', 'IDT1RWAJR']
panel = panel[TARGET]

panel['REPDTE'] = pd.to_datetime(panel['REPDTE'])


# panel already exists from the merge step
key_cols = {'CERT', 'REPDTE'}



def cast_numeric(series: pd.Series) -> pd.Series:
    """Try to downcast to integer; fall back to float if needed."""
   
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().all():
        return series
    
    if (out.dropna() % 1 == 0).all():
        return pd.to_numeric(out, downcast="integer").astype("Int64")
    return out.astype("float64")

for col in panel.columns.difference(key_cols):
    panel[col] = cast_numeric(panel[col])



# =============================================================================
#               SNP 500 DATA
# =============================================================================
import datetime as dt

snp_df = pd.read_csv(r'D:/Bureau/advance_emp_corpo/Projet/app/data/raw/crsp/snp500.csv')
snp_df['date'] = pd.to_datetime(snp_df['date'])
eom_mask = snp_df['date'].dt.is_month_end
snp_df = snp_df.loc[eom_mask].copy()

