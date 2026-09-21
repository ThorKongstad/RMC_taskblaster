import os
import sys
import pathlib
from copy import deepcopy

from gpaw import GPAW
from dftd4.ase import DFTD4
from ase.calculators.mixing import SumCalculator

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from . import folder_exist


def functional_exceptions(functional):
    match functional:
        case 'BEEF-vdW': return {'name': 'BEEF-vdW', 'backend': 'libvdwxc'}
        case _: return functional


def calculation_setter(atoms, calc_params, dftd4_bool, txt=None):
    calc_params_copy = deepcopy(calc_params)
    calc_params_copy['xc'] = functional_exceptions(calc_params_copy['xc'])
    if txt is None:
        if 'txt' in calc_params.keys(): calc_params.pop('txt', None)
    else: calc_params_copy.update({'txt': txt})
    if dftd4_bool: calc = SumCalculator([DFTD4(method=calc_params['xc']), GPAW(**calc_params_copy)])
    else: calc = GPAW(**calc_params_copy)
    atoms.calc = calc
