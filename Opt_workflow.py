import os
import pathlib
import sys
#from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from RMC_taskblaster import sanitize, folder_exist, update_db, Row_descriptor

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
        return tb.node(update_db, db_dir=self.db_path, db_update_args=dict(id=self.db_id, atoms=self.run_optimisation, relaxed=True, vibration=False, vib_en=False))


def optimise(row_wf, fmax: float=0.03):
    row_dc = row_wf.as_dc()
    parprint(f'outstd of opt calculation for db entry {row_dc.db_id} with structure: {row_dc.structure_str}, adsorbate: {row_dc.adsorbate_str} and functional: {row_dc.xc}')

    functional_folder = sanitize(row_dc.xc) + ('_D4' if row_dc.dftd4 else '')
    if world.rank == 0: folder_exist(os.path.basename(row_dc.db_path) + functional_folder)

    atoms = row_dc.atoms
    atoms.calc.txt = f'{functional_folder}/opt_id{row_dc.db_id}_{row_dc.structure_str}_{row_dc.adsorbate_str}.txt'

    dyn = BFGS(row_dc.atoms, trajectory=None)
    dyn.run(fmax=fmax)

    return atoms

