import os
import pathlib
import sys
from copy import deepcopy

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from RMC_taskblaster import sanitize, folder_exist, update_db
from RMC_taskblaster.workflow import Row_descriptor

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
        return tb.node(lambda gas, solv: solv-gas, gas=self.calc_non_solvation_sp, solv=self.calc_solvation_sp)

    @tb.task(tags={'organise'})
    def write_solv_result(self):
        return tb.node(lambda at: update_db(self.db_path, dict(id=self.db_id, solvation_E=self.subtract_solv_corr)), at=self.relaxed_atoms)


def calc_non_solvation_sp_func(row_wf, atoms, FD_bool):
    parprint(f'outstd of non-solvation sp calculation for db entry {row_wf.db_id} with structure: {row_wf.structure_str}, adsorbate: {row_wf.adsorbate_str} and functional: {row_wf.functional}')
    atoms = atoms.copy()
    functional_folder = os.path.basename(row_wf.db_path) + '/' + sanitize(row_wf.xc) + ('_D4' if row_wf.dftd4 else '')
    if world.rank == 0: folder_exist(functional_folder)

    if FD_bool:
        calc_params = deepcopy(row_wf.calc_params)
        calc_params.update({'mode': 'fd'})
        atoms.calc = GPAW(**calc_params)

    atoms.calc['txt'] = os.path.basename(row_wf.db_path) + '/' + f'{functional_folder}/sp{'_fd' if FD_bool else ''}_id{row_wf.db_id}_{row_wf.structure_str}_{row_wf.adsorbate_str}.txt'
    return atoms.get_potential_energy()


def calc_solvation_sp_func(row_wf, atoms, FD_bool):
    parprint( f'outstd of solvation sp calculation for db entry {row_wf.db_id} with structure: {row_wf.structure_str}, adsorbate: {row_wf.adsorbate_str} and functional: {row_wf.functional}')
    atoms = atoms.copy()
    functional_folder = os.path.basename(row_wf.db_path) + '/' + sanitize(row_wf.xc) + ('_D4' if row_wf.dftd4 else '')
    if world.rank == 0: folder_exist(functional_folder)

    calc_params = deepcopy(row_wf.calc_params)

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

    atoms.calc['txt'] = os.path.basename(row_wf.db_path) + '/' + f'{functional_folder}/solv{'_fd' if FD_bool else ''}_id{row_wf.db_id}_{row_wf.structure_str}_{row_wf.adsorbate_str}.txt'

    return atoms.get_potential_energy()

