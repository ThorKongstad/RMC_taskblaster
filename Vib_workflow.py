import pathlib
import sys
import os
from copy import deepcopy
from dataclasses import make_dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from RMC_taskblaster import sanitize, folder_exist, update_db, Row_descriptor
from RMC_taskblaster.calculator_prepare import calculation_setter


import numpy as np
import taskblaster as tb
from ase.parallel import parprint, world, barrier
from ase.constraints import FixAtoms
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo


@tb.workflow
class Vib_RMC_Workflow(Row_descriptor):
    relaxed_atoms = tb.var()

    @tb.task(tags={'calculation'})
    def run_vibration(self):
        return tb.node(calc_vibration, row_describ=self.as_dict(), atoms=self.relaxed_atoms)

    @tb.task(tags={'organise'})
    def write_opt_result(self):
        return tb.node(update_db, db_dir=self.db_path, db_update_args=self.run_vibration)


def calc_vibration(row_describ, atoms):
    row_dc = make_dataclass('Row_descriptor_dc', list(row_describ.keys()))(**row_describ)
    parprint(f'outstd of vib calculation for db entry {row_dc.db_id} with structure: {row_dc.structure_str}, adsorbate: {row_dc.adsorbate_str} and functional: {row_dc.xc}')

    functional_folder = os.path.dirname(row_dc.db_path) + '/' + sanitize(row_dc.xc) + ('_D4' if row_dc.dftd4 else '')
    if world.rank == 0: folder_exist(folder_name=os.path.basename(functional_folder), path=os.path.dirname(functional_folder))

    file_name = f'vib_id{row_dc.db_id}_{row_dc.structure_str}_{row_dc.adsorbate_str}'
    txt = f'{functional_folder}/{file_name}.txt'

    calc_params = deepcopy(row_dc.calc_params)
    calc_params.update({'symmetry': 'off'})
    calculation_setter(atoms=atoms, calc_params=calc_params, dftd4_bool=row_dc.dftd4, txt=txt)

    metal_symbol = ['Co', 'Fe']
    metal_at, not_metal_at = [], []
    for i, at in enumerate(atoms):
        if at.symbol in metal_symbol: metal_at.append(i)
        else: not_metal_at.append(i)
    metal_z_pos = list(pos[2] for pos in atoms[metal_at].get_positions())
    avg_metal_z_pos = np.mean(metal_z_pos)
    atoms_for_vib = list(filter(lambda i: (atoms[i].position[2] > (avg_metal_z_pos + 0.4)) or atoms[i].symbol in metal_symbol, list(range(len(atoms)))))
    locked_metals = list(filter(lambda i: (atoms[i].position[2] < (avg_metal_z_pos + 0.4)) and atoms[i].symbol not in metal_symbol, list(range(len(atoms)))))

    atoms.set_constraint(constraint=FixAtoms(locked_metals))

    vib = Vibrations(atoms, indices=atoms_for_vib, name=os.path.basename(row_dc.db_path) + f'/{functional_folder}/{file_name}')
    vib.run()
    vib_energies = vib.get_energies()
    thermo = HarmonicThermo(vib_energies, atoms.get_potential_energy(), ignore_imag_modes=True)

    if world.rank == 0:
        vib.summary(log=f'{functional_folder}/{file_name.replace("vib", "vib_en")}')

        with open(f'{functional_folder}/{file_name.replace("vib", "vib_en")}', 'r') as fil: energy_string = fil.read()

        return dict(id=row_dc.db_id, vibration=True, zpe=thermo.get_ZPE_correction(), vib_en=energy_string,
                    enthalpy=thermo.get_internal_energy(300), entropy=thermo.get_entropy(300),
                    free_E=thermo.get_helmholtz_energy(300))

