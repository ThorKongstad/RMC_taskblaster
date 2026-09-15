import os
import sys
import pathlib
import shutil
import re
from pathlib import Path
import subprocess
from copy import deepcopy

#from gxtb.ase import GXTB  # placeholder — replace with your actual GxTB binding
from tblite.ase import TBLite
#from dftd4.ase import DFTD4
#from ase.calculators.mixing import SumCalculator

#sys.path.insert(0, str(pathlib.Path(__file__).parent))
#from RMC_taskblaster import folder_exist

import numpy as np
from ase.db import connect
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

#def calculation_setter(atoms, calc_params, dftd4_bool, txt=None):
#    calc_params_copy = deepcopy(calc_params)
#    if txt is None:
#        calc_params_copy.pop('txt', None)
#    else:
#        calc_params_copy.update({'txt': txt})
#    if dftd4_bool:
#        calc = SumCalculator([DFTD4(method=calc_params_copy['method']), GXTB(**calc_params_copy)])
#    atoms.calc = calc


def parse_xtb_energy(log_path: Path) -> float:
    """Extract total energy (Hartree) from xtb log file."""
    text = log_path.read_text()
    for line in reversed(text.splitlines()):
        if "TOTAL ENERGY" in line:
            m = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"Could not parse energy from {log_path}")


def parse_xtb_forces(run_dir: Path, n_atoms: int) -> np.ndarray:
    """
    Parse Turbomole $grad file written by xtb --grad.
    Coordinate lines: x y z symbol  (4 tokens, last is non-float)
    Gradient lines:   gx gy gz      (3 tokens, all float)
    Returns forces in eV/Å (force = -gradient).
    """
    gradient_file = run_dir / "gradient"
    if not gradient_file.exists():
        raise FileNotFoundError(f"gradient file not found in {run_dir}")

    grad_lines = []
    for line in gradient_file.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            try:
                grad_lines.append([float(p) for p in parts])
            except ValueError:
                pass

    if len(grad_lines) != n_atoms:
        raise RuntimeError(
            f"Expected {n_atoms} gradient lines in {gradient_file}, "
            f"got {len(grad_lines)}"
        )

    forces = -np.array(grad_lines) * (Hartree / Bohr)  # eV/Å
    return forces


def parse_xtb_charges(run_dir: Path) -> list:
    """Parse per-atom charges from xtb 'charges' file (one float per line)."""
    charges_file = run_dir / "charges"
    if not charges_file.exists():
        raise FileNotFoundError(f"charges file not found in {run_dir}")
    return [
        float(line.strip())
        for line in charges_file.read_text().splitlines()
        if line.strip()
    ]


def is_scf_failure(err_msg: str) -> bool:
    return "Could not parse energy" in err_msg or "NOT_CONVERGED" in err_msg or "SCF not converged" in err_msg



class GxTBSubprocess(Calculator):
    """
    ASE Calculator wrapping xtb --gxtb subprocess.
    Each calculate() call runs xtb in its own numbered subdirectory of
    base_dir so trajectory restarts are intact and nothing relies on cwd.
    """
    implemented_properties = ["energy", "forces", "charges"]

    def __init__(self, charge=0, uhf=0, xtb_cmd="xtb", maxiter=250,
                 base_dir=None, write_log=True, keep_files=True, **kwargs):
        super().__init__(**kwargs)
        self.charge = int(charge)
        self.uhf = int(uhf)
        self.xtb_cmd = xtb_cmd
        self.maxiter = int(maxiter)
        self.base_dir = Path(base_dir)
        self.write_log = write_log
        self.keep_files = keep_files
        self._sp_counter = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        sp_dir = self.base_dir / f"sp_{self._sp_counter:04d}"
        self._sp_counter += 1
        sp_dir.mkdir(parents=True, exist_ok=True)

        xyz_path = sp_dir / "struc.xyz"
        write(str(xyz_path), self.atoms)

        cmd = [
            self.xtb_cmd, "struc.xyz",
            "--gxtb",
            "--grad",
            "--chrg", str(self.charge),
            "--uhf",  str(self.uhf),
            "--iterations", str(self.maxiter),
        ]

        log_path = sp_dir / "xtb.out"
        with open(log_path, "w") as fout:
            result = subprocess.run(
                cmd,
                cwd=str(sp_dir),        # xtb writes gradient/charges here
                stdout=fout,
                stderr=subprocess.STDOUT,
            )

        if result.returncode != 0:
            tail = log_path.read_text().splitlines()[-20:]
            raise RuntimeError(
                f"xtb exited {result.returncode} in {sp_dir}\n" + "\n".join(tail)
            )

        energy_ev = parse_xtb_energy(log_path) * Hartree
        forces    = parse_xtb_forces(sp_dir, len(self.atoms))
        charges   = parse_xtb_charges(sp_dir)

        self.results["energy"]  = energy_ev
        self.results["forces"]  = forces
        self.results["charges"] = np.array(charges)

        if not self.keep_files:
            shutil.rmtree(sp_dir)

    def get_charges(self):
        return self.results.get("charges")


def calculation_setter(atoms, calc_params):
    calc_params_copy = deepcopy(calc_params)
    # f_max = 0.01
    #calc = TBLite(**calc_params_copy)
    calc = GxTBSubprocess(**calc_params_copy)
    atoms.calc = calc


