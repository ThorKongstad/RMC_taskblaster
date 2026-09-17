import os
import pathlib
import sys
from dataclasses import make_dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from . import sanitize, folder_exist, update_db, Row_descriptor
from .calculator_prepare import calculation_setter

import taskblaster as tb
from ase.optimize import BFGS
from ase.parallel import parprint, world, barrier


@tb.workflow
class Opt_RMC_Workflow(Row_descriptor):
    @tb.task(tags={'calculation'})
    def run_optimisation(self):
        return tb.node(optimise, row_describ=self.as_dict())

    @tb.task(tags={'organise'})
    def write_opt_result(self):
        return tb.node(update_db, db_dir=self.db_path, db_update_args=dict(id=self.db_id, atoms=self.run_optimisation, relaxed=True, vibration=False, vib_en=False))


def optimise(row_describ, fmax: float=0.03):
    row_dc = make_dataclass('Row_descriptor_dc', list(row_describ.keys()))(**row_describ)
    parprint(f'outstd of opt calculation for db entry {row_dc.db_id} with structure: {row_dc.structure_str}, adsorbate: {row_dc.adsorbate_str}')

    #functional_folder = os.path.dirname(row_dc.db_path) + '/' + sanitize(row_dc.xc) + ('_D4' if row_dc.dftd4 else '')
    #if world.rank == 0: folder_exist(folder_name=os.path.basename(functional_folder), path=os.path.dirname(functional_folder))

    atoms = row_dc.atoms
    #txt = f'{functional_folder}/opt_id{row_dc.db_id}_{row_dc.structure_str}_{row_dc.adsorbate_str}.txt'

    calculation_setter(atoms=atoms, calc_params=row_dc.calc_params)

    dyn = BFGS(row_dc.atoms, trajectory=None)
    dyn.run(fmax=fmax)

    return atoms

