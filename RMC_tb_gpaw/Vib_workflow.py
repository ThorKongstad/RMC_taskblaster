import pathlib
import sys
import os
from copy import deepcopy
from dataclasses import make_dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from . import sanitize, folder_exist, update_db, Row_descriptor
from .calculator_prepare import calculation_setter


import numpy as np
import taskblaster as tb
from ase.parallel import parprint, world, barrier
from ase.constraints import FixAtoms
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo


@tb.workflow
class Vib_RMC_Workflow(Row_descriptor):
    relaxed_atoms = tb.var()
    E_pot = tb.var()

    @tb.task(tags={'calculation'})
    def run_vibration(self):
        return tb.node(calc_vibration, row_describ=self.as_dict(), atoms=self.relaxed_atoms)

    @tb.task(tags={'organise'})
    def write_vib_result(self):
        return tb.node(vib_thermo_to_db, row_describ=self.as_dict(), atoms=self.relaxed_atoms, vib_done=self.run_vibration, db_dir=self.db_path, E_pot=self.E_pot)

def calc_vibration(row_describ, atoms):
    row_dc = make_dataclass('Row_descriptor_dc', list(row_describ.keys()))(**row_describ)
    parprint(f'outstd of vib calculation for db entry {row_dc.db_id} with structure: {row_dc.structure_str}, adsorbate: {row_dc.adsorbate_str} and functional: {row_dc.xc}')

    functional_folder, file_name, indices, locked = vib_setup(row_dc, atoms)
    if world.rank == 0:
        folder_exist(folder_name=os.path.basename(functional_folder), path=os.path.dirname(functional_folder))
    barrier()  # folder must exist before any rank opens txt

    calc_params = deepcopy(row_dc.calc_params)
    calc_params.update({'symmetry': 'off'})
    calculation_setter(atoms=atoms, calc_params=calc_params, dftd4_bool=row_dc.dftd4,
                       txt=f'{functional_folder}/{file_name}.txt')
    atoms.set_constraint(FixAtoms(locked))

    vib = Vibrations(atoms, indices=indices, name=f'{functional_folder}/{file_name}')
    vib.run()

    return dict(id=row_dc.db_id, vib_cache=f'{functional_folder}/{file_name}')  # small, all ranks identical


def vib_thermo_to_db(row_describ, atoms, vib_done, db_dir, E_pot):
    row_dc = make_dataclass('Row_descriptor_dc', list(row_describ.keys()))(**row_describ)
    functional_folder, file_name, indices, _ = vib_setup(row_dc, atoms)

    vib = Vibrations(atoms, indices=indices, name=vib_done['vib_cache'])  # reads cache, no calculator
    vib_energies = vib.get_energies()
    thermo = HarmonicThermo(vib_energies, E_pot, ignore_imag_modes=True)

    summary_file = f'{functional_folder}/{file_name.replace("vib", "vib_en")}'
    vib.summary(log=summary_file)
    with open(summary_file) as fil:
        energy_string = fil.read()

    result = dict(id=row_dc.db_id, vibration=True, vib_en=energy_string,
                  zpe=float(thermo.get_ZPE_correction()),
                  enthalpy=float(thermo.get_internal_energy(300)),
                  entropy=float(thermo.get_entropy(300)),
                  free_E=float(thermo.get_helmholtz_energy(300)))
    update_db(db_dir=db_dir, db_update_args=result)


def vib_setup(row_dc, atoms):
    functional_folder = os.path.dirname(row_dc.db_path) + '/' + sanitize(row_dc.xc) + ('_D4' if row_dc.dftd4 else '')
    file_name = f'vib_id{row_dc.db_id}_{row_dc.structure_str}_{row_dc.adsorbate_str}'

    metal_symbol = ['Co', 'Fe', 'Mn', 'Ni', 'Cu', 'Sn', 'Zn']
    metal_at = [i for i, at in enumerate(atoms) if at.symbol in metal_symbol]
    assert len(metal_at) > 0
    z_cut = np.mean([atoms[i].position[2] for i in metal_at]) + 0.4

    indices = [i for i, at in enumerate(atoms) if at.position[2] > z_cut or at.symbol in metal_symbol]
    locked = [i for i, at in enumerate(atoms) if at.position[2] < z_cut and at.symbol not in metal_symbol]
    return functional_folder, file_name, indices, locked