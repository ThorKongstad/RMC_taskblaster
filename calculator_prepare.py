from gpaw import GPAW
from dftd4.ase import DFTD4
from ase.calculators.mixing import SumCalculator


def calculation_setter(atoms, calc_params, dftd4_bool):
    if dftd4_bool: calc = SumCalculator([DFTD4(method=calc_params['xc']), GPAW(**calc_params)])
    else: calc = GPAW(**calc_params)
    atoms.calc = calc
