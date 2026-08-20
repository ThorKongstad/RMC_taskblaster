import argparse
import pickle
import pathlib
import sys
import os
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from RMC_taskblaster import Row_descriptor
from RMC_taskblaster.calculator_prepare import calculation_setter
from RMC_taskblaster.Opt_workflow import Opt_RMC_Workflow
from RMC_taskblaster.Vib_workflow import Vib_RMC_Workflow
from RMC_taskblaster.Solv_workflow import calc_solv_RMC_Workflow

import taskblaster as tb
import ase.db as db


@tb.workflow
class RMC_Row_Workflow(Row_descriptor):
    @tb.subworkflow
    def task1_opt(self, fmax=0.03):
        return Opt_RMC_Workflow(fmax=fmax, **self.as_dict())

    @tb.subworkflow
    def task2_vib(self):
        return Vib_RMC_Workflow(relaxed_atoms=self.task1_opt.run_optimisation, **self.as_dict())

    @tb.subworkflow
    def task3_solv(self):
        return calc_solv_RMC_Workflow(relaxed_atoms=self.task1_opt.run_optimisation, **self.as_dict())

#    @tb.task
#    def task4_ensemble(self):
#        ...


@tb.workflow
class Workflow:
    db_paths = tb.var()

    @tb.dynamical_workflow_generator({'results': '*/*'})
    def rows(self):
        return tb.node(generate_row_workflows, db_paths=self.db_paths)


@tb.dynamical_workflow_generator_task
def generate_row_workflows(db_paths):
    for db_path in db_paths:
        if not os.path.basename(db_path) in os.listdir(db_p if len(db_p := os.path.dirname(db_path)) > 0 else '.'):
            warnings.warn(f"Can't find database: skipping {os.path.basename(db_path)}")
            continue

        conn = db.connect(db_path)
        for row in conn.select():
            atoms = row.toatoms()
            calc_params = pickle.loads(row.data['dft_calc_pickle'])
            calc_params['charge'] = row.get('gpaw_charge')

            calculation_setter(atoms=atoms, calc_params=calc_params, dftd4=row.data['dftd4'])

            wf = RMC_Row_Workflow(
                db_path=db_path,
                db_id=row.get('id'),
                atoms=atoms,
                calc_params=calc_params,
                xc=row.get('xc'),
                dftd4=row.get('dftd4'),
                structure_str=row.get('structure_str'),
                adsorbate_str=row.get('adsorbate_str'),
            )
            name = f'row_{row.id}_{row.get("structure_str")}_{row.get("adsorbate_str")}_{row.get('xc')}' + ('-d4' if row.get('dftd4') else '')
            yield name, wf


def workflow(runner):
    arp = argparse.ArgumentParser()
    arp.add_argument('-db', "--databases", nargs="+")
    args = arp.parse_known_args()

    runner.run_workflow(Workflow(db_paths=args.databases))

