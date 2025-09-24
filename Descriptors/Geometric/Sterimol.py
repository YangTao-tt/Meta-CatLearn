import os
import pandas as pd
from morfeus import Sterimol, read_xyz


atom_Hf = 13
lig1_start = 28
lig2_start = 10


xyz_dir = r"./Hf-TS_XYZs"
output_file = "Sterimol_ts.csv"


results = []


for file in os.listdir(xyz_dir):
    if file.endswith(".xyz"):
        path = os.path.join(xyz_dir, file)
        try:
            elements, coordinates = read_xyz(path)

            sterimol1 = Sterimol(elements, coordinates, atom_Hf, lig1_start)

            sterimol2 = Sterimol(elements, coordinates, atom_Hf, lig2_start)

            results.append({
                "filename": file,
                "L_1": round(sterimol1.L_value, 3),
                "B1_1": round(sterimol1.B_1_value, 3),
                "B5_1": round(sterimol1.B_5_value, 3),
                "L_2": round(sterimol2.L_value, 3),
                "B1_2": round(sterimol2.B_1_value, 3),
                "B5_2": round(sterimol2.B_5_value, 3),
            })
        except Exception as e:
            results.append({
                "filename": file,
                "L_1": None, "B1_1": None, "B5_1": None,
                "L_2": None, "B1_2": None, "B5_2": None,
                "error": str(e)
            })


df = pd.DataFrame(results)
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Done: {output_file}")
