import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 3000
dates = pd.date_range("2025-01-01", periods=n, freq="3h")
hour = dates.hour
month = dates.month

temp = 25 + 7*np.sin(2*np.pi*(dates.dayofyear.to_numpy()/365.25)) + 3*np.sin(2*np.pi*hour/24) + rng.normal(0, 1.7, n)
rh = np.clip(68 - 0.75*(temp-25) + 8*np.sin(2*np.pi*(hour+4)/24) + rng.normal(0, 5, n), 25, 98)
ah = np.clip(0.012 + (rh/100)*0.009 + rng.normal(0, 0.0012, n), 0.004, 0.030)
rush = np.where(((hour>=7)&(hour<=10)) | ((hour>=17)&(hour<=21)), 1, 0)
season = 1 + 0.18*np.cos(2*np.pi*(month-1)/12)
base_pollution = 35*season + 18*rush + rng.normal(0, 7, n)
nox = np.clip(base_pollution + 1.5*(30-temp) + rng.normal(0, 15, n), 5, None)
no2 = np.clip(0.62*nox + 8*rush + rng.normal(0, 10, n), 5, None)
benzene = np.clip(0.045*nox + 0.8*rush + rng.normal(0, 1.2, n), 0.2, None)
co = np.clip(0.012*nox + 0.12*rush + 0.18*(rh/100) + rng.normal(0, 0.10, n), 0.2, None)
pt08_s1 = np.clip(800 + 120*co + rng.normal(0, 70, n), 300, 2000)
pt08_s2 = np.clip(650 + 95*benzene + rng.normal(0, 55, n), 250, 1800)
pt08_s3 = np.clip(1400 - 5*nox + rng.normal(0, 80, n), 200, 2000)
pt08_s4 = np.clip(900 + 4*no2 + rng.normal(0, 70, n), 250, 2000)
pt08_s5 = np.clip(700 + 3*nox + rng.normal(0, 70, n), 200, 2200)
nmhc = np.clip(80 + 18*benzene + rng.normal(0, 15, n), 10, 400)

df = pd.DataFrame({
    "Date": dates.strftime("%d/%m/%Y"),
    "Time": dates.strftime("%H.%M.%S"),
    "CO(GT)": np.round(co, 2),
    "PT08.S1(CO)": np.round(pt08_s1, 1),
    "NMHC(GT)": np.round(nmhc, 1),
    "C6H6(GT)": np.round(benzene, 2),
    "PT08.S2(NMHC)": np.round(pt08_s2, 1),
    "NOx(GT)": np.round(nox, 1),
    "PT08.S3(NOx)": np.round(pt08_s3, 1),
    "NO2(GT)": np.round(no2, 1),
    "PT08.S4(NO2)": np.round(pt08_s4, 1),
    "PT08.S5(O3)": np.round(pt08_s5, 1),
    "T": np.round(temp, 1),
    "RH": np.round(rh, 1),
    "AH": np.round(ah, 4),
})
for col in ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)", "T", "RH", "AH"]:
    idx = rng.choice(n, size=round(n*0.012), replace=False)
    df.loc[idx, col] = -200

out = Path("data/air_quality_data.csv")
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Created {len(df):,} records at {out}")
