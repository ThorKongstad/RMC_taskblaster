import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from RMC_tb_pyscf import sanitize, folder_exist, update_db, Row_descriptor
from RMC_tb_pyscf.calculator_prepare import calculation_setter

import taskblaster as tb
from ase.optimize import BFGS
from ase.parallel import parprint, world, barrier


@tb.workflow
class Opt_RMC_Workflow(Row_descriptor):
    @tb.task
    def run_optimisation(self, fmax: float = 0.03):
        return tb.node(optimise, row_wf=self, fmax=fmax)

    @tb.task
    def write_opt_result(self):
        return tb.node(update_db, db_dir=self.db_path, db_update_args=dict(id=self.db_id, atoms=self.run_optimisation, relaxed=True, vibration=False, vib_en=False))


def optimise(row_wf, fmax: float = 0.03):
    parprint(f'outstd of opt calculation for db entry {row_wf.db_id} with structure: {row_wf.structure_str}, adsorbate: {row_wf.adsorbate_str} and functional: {row_wf.xc}')

    functional_folder = sanitize(row_wf.xc) + ('_D4' if row_wf.dftd4 else '')
    if world.rank == 0: folder_exist(functional_folder)

    row_wf.atoms.calc['txt'] = f'{functional_folder}/opt_id{row_wf.db_id}_{row_wf.structure_str}_{row_wf.adsorbate_str}.txt'

    calculation_setter(row_wf.atoms, row_wf)

    dyn = BFGS(row_wf.atoms, trajectory=None)
    dyn.run(fmax=fmax)

    return row_wf.atoms

