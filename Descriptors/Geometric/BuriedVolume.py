

import os
import csv
from morfeus import read_xyz, BuriedVolume

xyz_dir = r"./Hf-TS_XYZs"

output_dir = r"./Geometric"
output_csv = os.path.join(output_dir, "BuriedVolume_ts.csv")


center_index = 14

xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith(".xyz")]

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "%VBur"])

    for filename in sorted(xyz_files):
        filepath = os.path.join(xyz_dir, filename)
        try:
            elements, coordinates = read_xyz(filepath)

            bv = BuriedVolume(elements, coordinates, center_index, radius=3.5, include_hs=True)

            percent = bv.fraction_buried_volume * 100
            writer.writerow([filename, f"{percent:.4f}"])

            print(f"{filename}: %VBur = {percent:.4f}")
        except Exception as e:
            print(f"Failed：{filename}，Error：{str(e)}")