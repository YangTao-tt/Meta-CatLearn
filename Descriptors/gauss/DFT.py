


import os
import sys
import pandas as pd
import numpy as np
from gaussian_extractor import GaussianExtractor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


log_dir = r"./out"
output_file = "DFT_descriptors.csv"
expected_fields = [
    "filename", "E_scf", "ZPE", "E_corr", "H_corr", "G_corr",
    "HOMO", "LUMO", "Gap", "IE", "EA", "GlobalHardness", "ChemPot", "GEI",
    "Zr_13_mul", "C_14_mul", "C_16_mul", "C_15_mul",
    "Zr_13_nbo", "C_14_nbo", "C_16_nbo", "C_15_nbo",
    "Zr_13_apt", "C_14_apt", "C_16_apt", "C_15_apt",
    "Freq_mean", "Freq_max", "Freq_min",
    "C14_M13_C15", "M13_C15_C16", "C15_C16_C14", "M13_C14_C16",
    "dC14_M13_C15", "dM13_C15_C16", "dC15_C16_C14", "dM13_C14_C16",
    "Zr13_Me14", "C14_C16", "M13_C16", "M13_C15", "C15_C16",
    "dZr13_Me14", "dC14_C16", "dM13_C16", "dM13_C15", "dC15_C16",
    "Dipole", "Isotropic", "Anisotropic",
    "Cone_angle_1", "MC_Dist_1", "L_1", "B1_min_1", "B5_max_1",
    "Cone_angle_2", "MC_Dist_2", "L_2", "B1_min_2", "B5_max_2",
    "%VolBur_TS", "SASA_TS", "Estab", "BDE", "delErxn", "delCat"
]


results = []

log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.out')], key=lambda x: int(x.split('.')[0]))

for filename in log_files:
    full_path = os.path.join(log_dir, filename)
    try:
        extractor = GaussianExtractor(full_path)
        data = extractor.extract()
        data["filename"] = os.path.splitext(filename)[0]
        results.append(data)
        print(f"[OK] {filename}")
    except Exception as e:
        print(f"[FAIL] {filename}: {e}")
        # NaN
        empty = {key: np.nan for key in expected_fields}
        empty["filename"] = os.path.splitext(filename)[0]
        results.append(empty)


df = pd.DataFrame(results)


for col in expected_fields:
    if col not in df.columns:
        df[col] = np.nan
df = df[expected_fields]

df.to_csv(output_file, index=False, encoding='utf-8')
print(f"{output_file}")
