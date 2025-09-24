import os
import csv
from morfeus import ConeAngle, read_xyz

xyz_dir = r"./Hf-TS-complex_XYZs"


center_index = 12


results = []


for filename in os.listdir(xyz_dir):
    if filename.endswith(".Geometric"):
        file_path = os.path.join(xyz_dir, filename)
        try:
            elements, coordinates = read_xyz(file_path)
            cone_angle = ConeAngle(elements, coordinates, center_index)
            angle = round(cone_angle.cone_angle, 2)
            tangent_atoms = ";".join(str(i) for i in cone_angle.tangent_atoms)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            angle = float('nan')
            tangent_atoms = "NaN"
        results.append([filename, angle, tangent_atoms])


output_csv = os.path.join(xyz_dir, "ConeAngle.csv")
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "cone_angle_deg", "tangent_atoms"])
    writer.writerows(results)

print(f" {output_csv}")

