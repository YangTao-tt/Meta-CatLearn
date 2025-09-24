import os
import pandas as pd
import numpy as np

log_dir = "./out"
output_csv = "HomoLumo.csv"

HARTREE_TO_EV = 27.2114

def extract_homo_lumo(filepath):
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    homo, lumo = None, None
    occ_energies, virt_energies = [], []

    for line in lines:
        if "Alpha  occ. eigenvalues" in line:
            occ_energies += [float(x) for x in line.split('--')[-1].split()]
        elif "Alpha virt. eigenvalues" in line:
            virt_energies += [float(x) for x in line.split('--')[-1].split()]

    if occ_energies and virt_energies:
        homo = occ_energies[-1]
        lumo = virt_energies[0]
        return homo, lumo
    else:
        return np.nan, np.nan


results = []
log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".out")], key=lambda x: int(x.split(".")[0]))

for filename in log_files:
    filepath = os.path.join(log_dir, filename)
    try:
        homo, lumo = extract_homo_lumo(filepath)

        gap = lumo - homo
        IE = -homo
        EA = -lumo
        global_hardness = (lumo - homo) / 2
        chem_pot = (lumo + homo) / 2
        gei = (chem_pot ** 2) / (2 * global_hardness)

        # eV
        results.append({
            "filename": os.path.splitext(filename)[0],
            "HOMO (Ha)": homo,
            "LUMO (Ha)": lumo,
            "GAP (eV)": gap * HARTREE_TO_EV,
            "IE (eV)": IE * HARTREE_TO_EV,
            "EA (eV)": EA * HARTREE_TO_EV,
            "GlobalHardness (eV)": global_hardness * HARTREE_TO_EV,
            "ChemPot (eV)": chem_pot * HARTREE_TO_EV,
            "GEI (eV)": gei * HARTREE_TO_EV
        })
    except Exception as e:
        results.append({
            "filename": os.path.splitext(filename)[0],
            "HOMO (Ha)": np.nan,
            "LUMO (Ha)": np.nan,
            "GAP (eV)": np.nan,
            "IE (eV)": np.nan,
            "EA (eV)": np.nan,
            "GlobalHardness (eV)": np.nan,
            "ChemPot (eV)": np.nan,
            "GEI (eV)": np.nan
        })

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding='utf-8')
