import re
import numpy as np

class GaussianExtractor:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, 'r', encoding='latin-1') as f:
            self.log = f.read()
        self.lines = self.log.splitlines()

    def extract(self):
        data = {
            "E_scf": self._parse_scf_energy(),
            "ZPE": self._parse_zpe(),
            "H_corr": self._parse_thermal_corr("Enthalpy"),
            "G_corr": self._parse_thermal_corr("Gibbs"),
            "E_corr": self._parse_thermal_corr("Thermal"),
            "Mulliken_mean": None,
            "Mulliken_max": None,
            "Mulliken_min": None,
            "Freq_mean": None,
            "Freq_max": None,
            "Freq_min": None,
            "Dipole": self._parse_dipole()
        }



        mulliken = self._parse_mulliken_charges()
        if mulliken:
            charges = np.array(mulliken)
            data["Mulliken_mean"] = np.mean(charges)
            data["Mulliken_max"] = np.max(charges)
            data["Mulliken_min"] = np.min(charges)

        freqs = self._parse_frequencies()
        if freqs:
            farr = np.array(freqs)
            data["Freq_mean"] = np.mean(farr)
            data["Freq_max"] = np.max(farr)
            data["Freq_min"] = np.min(farr)

        return data

    def _parse_scf_energy(self):
        m = re.search(r'SCF Done:.*?=\s*(-?\d+\.\d+)', self.log)
        return float(m.group(1)) if m else None

    def _parse_zpe(self):
        m = re.search(r'Zero-point correction=\s+([-\d.]+)', self.log)
        return float(m.group(1)) if m else None

    def _parse_thermal_corr(self, keyword):
        patterns = {
            "Thermal": r'Thermal correction to Energy=\s+([-\d.]+)',
            "Enthalpy": r'Thermal correction to Enthalpy=\s+([-\d.]+)',
            "Gibbs": r'Thermal correction to Gibbs Free Energy=\s+([-\d.]+)'
        }
        m = re.search(patterns[keyword], self.log)
        return float(m.group(1)) if m else None

    def _parse_homo_lumo(self):
        lines = self.lines
        occ_energies = []
        virt_energies = []
        start = False
        for line in lines:
            if "Alpha  occ. eigenvalues" in line:
                occ_energies += [float(x) for x in line.split('--')[-1].split()]
                start = True
            elif start and "Alpha virt. eigenvalues" in line:
                virt_energies += [float(x) for x in line.split('--')[-1].split()]
        if occ_energies and virt_energies:
            return occ_energies[-1], virt_energies[0]
        return None, None

    def _parse_mulliken_charges(self):
        charges = []
        pattern = r'Mulliken charges:\s*\n((?:\s*\d+\s+\w+\s+-?\d+\.\d+\s*\n)+)'
        match = re.search(pattern, self.log)
        if match:
            block = match.group(1)
            for line in block.strip().splitlines():
                parts = line.strip().split()
                charges.append(float(parts[2]))
        return charges if charges else None

    def _parse_frequencies(self):
        freqs = []
        for line in self.lines:
            if "Frequencies --" in line:
                freqs += [float(f) for f in line.split("--")[-1].split()]
        return freqs if freqs else None
