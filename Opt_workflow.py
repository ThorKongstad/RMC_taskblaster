import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from RMC_taskblaster import sanitize, folder_exist, update_db
from RMC_taskblaster.workflow import Row_descriptor

import taskblaster as tb
from ase.optimize import BFGS
from ase.parallel import parprint, world, barrier


@tb.workflow
class Opt_RMC_Workflow(Row_descriptor):
    @tb.task(tags={'calculation'})
    def run_optimisation(self):
        return tb.node(optimise, row_wf=self)

    @tb.task(tags={'organise'})
    def write_opt_result(self):
        return tb.node(lambda at: update_db(self.db_path, dict(id=self.db_id, atoms=at, relaxed=True, vibration=False, vib_en=False)), at=self.run_optimisation)


def optimise(row_wf, fmax: float=0.03):
    parprint(f'outstd of opt calculation for db entry {row_wf.db_id} with structure: {row_wf.structure_str}, adsorbate: {row_wf.adsorbate_str} and functional: {row_wf.xc}')

    functional_folder = sanitize(row_wf.xc) + ('_D4' if row_wf.dftd4 else '')
    if world.rank == 0: folder_exist(os.path.basename(row_wf.db_path) + functional_folder)

    row_wf.atoms.calc['txt'] = f'{functional_folder}/opt_id{row_wf.db_id}_{row_wf.structure_str}_{row_wf.adsorbate_str}.txt'

    dyn = BFGS(row_wf.atoms, trajectory=None)
    dyn.run(fmax=fmax)

    return row_wf.atoms

