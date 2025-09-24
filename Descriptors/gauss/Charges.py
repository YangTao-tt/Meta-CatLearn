import os
import re
import pandas as pd
import numpy as np

log_dir = "./out"
output_csv = "Charges_pi.csv"


target_atoms = {
    12: "Ti12",
    13: "C13",
    14: "C14",
    15: "C15"
}

def extract_charge_block(lines, header):
    charges = {}
    capture = False
    for line in lines:
        if header in line:
            capture = True
            continue
        if capture:
            if line.strip() == "" or "Sum of" in line or "with hydrogens" in line or "charges:" in line:
                break
            match = re.match(r"\s*(\d+)\s+([A-Za-z]+)\s+([-+]?\d*\.\d+)", line)
            if match:
                atom_idx = int(match.group(1))
                value = float(match.group(3))
                if atom_idx in target_atoms:
                    charges[target_atoms[atom_idx]] = value
    return charges

def extract_charges(filepath):
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()


    mul = extract_charge_block(lines, "Mulliken charges:")


    apt = extract_charge_block(lines, "APT charges:")

    data = {}
    for idx, label in target_atoms.items():
        data[f"Mul_{label}"] = mul.get(label, np.nan)
        data[f"APT_{label}"] = apt.get(label, np.nan)
    return data


results = []
log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".out")], key=lambda x: int(x.split(".")[0]))

for filename in log_files:
    filepath = os.path.join(log_dir, filename)
    charges = extract_charges(filepath)
    charges["filename"] = os.path.splitext(filename)[0]
    results.append(charges)


df = pd.DataFrame(results)
df = df[["filename"] + sorted([col for col in df.columns if col != "filename"])]
df.to_csv(output_csv, index=False, encoding='utf-8')
