import os
import csv
from morfeus import SASA, read_xyz


xyz_dir = r"./Hf-TS_XYZs"
output_csv = "SASA_ts.csv"


header = ["filename", "atom1_sasa", "sasa_area", "sasa_volume"]

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for filename in os.listdir(xyz_dir):
        if filename.endswith(".xyz"):
            filepath = os.path.join(xyz_dir, filename)
            try:
                elements, coordinates = read_xyz(filepath)
                sasa = SASA(elements, coordinates)

                atom1_sasa = sasa.atom_areas[1]
                total_area = sasa.area
                total_volume = sasa.volume

                writer.writerow([filename, atom1_sasa, total_area, total_volume])
            except Exception as e:
                print(f"Error in {filename}: {e}")
                writer.writerow([filename, "NaN", "NaN", "NaN"])
