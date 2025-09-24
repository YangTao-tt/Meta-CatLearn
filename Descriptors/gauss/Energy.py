import os
import csv
import numpy as np


ts_folder = './ts/out'
pi_folder = './pi/out'



def extract_energy(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Sum of electronic and thermal Free Energies" in line:

                    energy = float(line.split('=')[-1].strip())
                    return energy
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return np.nan



def get_out_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.out')]



def write_energies_to_csv(ts_folder, pi_folder, output_csv='energy.csv'):
    ts_files = get_out_files(ts_folder)
    pi_files = get_out_files(pi_folder)

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Catalyst', 'TS Energy', 'PI Energy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()


        for i, (ts_file, pi_file) in enumerate(zip(ts_files, pi_files), start=1):
            ts_energy = extract_energy(os.path.join(ts_folder, ts_file))
            pi_energy = extract_energy(os.path.join(pi_folder, pi_file))

            writer.writerow({'Catalyst': i, 'TS Energy': ts_energy, 'PI Energy': pi_energy})

        print(f"Energy data has been written to {output_csv}")



write_energies_to_csv(ts_folder, pi_folder)
