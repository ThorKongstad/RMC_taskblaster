import os
import pathlib
import sys
from copy import deepcopy
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from RMC_taskblaster import sanitize, folder_exist, update_db, Row_descriptor

import taskblaster as tb
from ase.optimize import BFGS
from ase.parallel import parprint, world, barrier
from ase.units import mol, kJ, kcal, Pascal, m
from gpaw import GPAW
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction
)


@tb.workflow
class calc_solv_RMC_Workflow(Row_descriptor):
    relaxed_atoms = tb.var()
    FD_bool = tb.var(default=False)

    @tb.task(tags={'Calculation'})
    def calc_non_solvation_sp(self):
        return tb.node(calc_non_solvation_sp_func, atoms=self.relaxed_atoms, row_wf=self, FD_bool=self.FD_bool)

    @tb.task(tags={'Calculation'})
    def calc_solvation_sp(self):
        return tb.node(calc_solvation_sp_func, atoms=self.relaxed_atoms, row_wf=self, FD_bool=self.FD_bool)

    @tb.task(tags={'organise'})
    def subtract_solv_corr(self):
        return tb.node(subtract, gas=self.calc_non_solvation_sp, solv=self.calc_solvation_sp)

    @tb.task(tags={'organise'})
    def write_solv_result(self):
        return tb.node(update_db, db_dir=self.db_path, db_update_args=dict(id=self.db_id, solvation_E=self.subtract_solv_corr))


def calc_non_solvation_sp_func(row_wf, atoms, FD_bool):
    row_dc = row_wf.as_dc()
    parprint(f'outstd of non-solvation sp calculation for db entry {row_dc.db_id} with structure: {row_dc.structure_str}, adsorbate: {row_dc.adsorbate_str} and functional: {row_dc.functional}')
    atoms = atoms.copy()
    functional_folder = os.path.dirname(row_dc.db_path) + '/' + sanitize(row_dc.xc) + ('_D4' if row_dc.dftd4 else '')
    if world.rank == 0: folder_exist(functional_folder)

    if FD_bool:
        calc_params = deepcopy(row_dc.calc_params)
        calc_params.update({'mode': 'fd'})
        atoms.calc = GPAW(**calc_params)

    atoms.calc.txt = f'{functional_folder}/sp{'_fd' if FD_bool else ''}_id{row_dc.db_id}_{row_dc.structure_str}_{row_dc.adsorbate_str}.txt'
    return atoms.get_potential_energy()


def calc_solvation_sp_func(row_wf, atoms, FD_bool):
    row_dc = row_wf.as_dc()
    parprint(f'outstd of solvation sp calculation for db entry {row_dc.db_id} with structure: {row_dc.structure_str}, adsorbate: {row_dc.adsorbate_str} and functional: {row_dc.functional}')
    atoms = atoms.copy()
    functional_folder = os.path.dirname(row_dc.db_path) + '/' + sanitize(row_dc.xc) + ('_D4' if row_dc.dftd4 else '')
    if world.rank == 0: folder_exist(functional_folder)

    calc_params = deepcopy(row_dc.calc_params)

    if FD_bool:
        calc_params.update({'mode': 'fd'})

    atomic_radii = {'H': 1.09, 'C': 1.77, 'N': 1.66, 'O': 1.50, 'Co': 2.4, 'Fe': 2.44}

    calc = SolvationGPAW(
        cavity=EffectivePotentialCavity(
            effective_potential=Power12Potential(atomic_radii, 0.18),
            temperature=298.15,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=78.36),
        interactions=[SurfaceInteraction(surface_tension=18.4 * 1e-3 * Pascal * m)],
        **calc_params)
    atoms.calc = calc

    atoms.calc.txt = f'{functional_folder}/solv{'_fd' if FD_bool else ''}_id{row_dc.db_id}_{row_dc.structure_str}_{row_dc.adsorbate_str}.txt'

    return atoms.get_potential_energy()

def subtract(gas, solv):
    return solv - gas

