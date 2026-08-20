import os
import sys
from gpaw import GPAW
from dftd4.ase import DFTD4
from ase.calculators.mixing import SumCalculator

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from RMC_taskblaster import folder_exist


def calculation_setter(atoms, calc_params, dftd4_bool):
    if 'txt' in calc_params.keys(): folder_exist(os.path.basename(calc_params['txt']))
    if dftd4_bool: calc = SumCalculator([DFTD4(method=calc_params['xc']), GPAW(**calc_params)])
    else: calc = GPAW(**calc_params)
    atoms.calc = calc
