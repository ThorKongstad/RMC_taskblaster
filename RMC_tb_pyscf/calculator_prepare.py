import os
#from gpaw import GPAW
from dftd4.ase import DFTD4
from ase.calculators.mixing import SumCalculator
from pyscf.pbc import dft, gto
from pyscf.pbc.tools.pyscf_ase import PySCF, cell_from_ase, ase_atoms_to_pyscf

def get_slurm_resouces():
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_JOB_CPUS_PER_NODE", 1)))
    if mem_per_node is not None:
        return int(mem_per_node), ncpus

    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if mem_per_cpu is not None:
        return int(mem_per_cpu) * ncpus, ncpus

    raise RuntimeError("Neither SLURM_MEM_PER_NODE nor SLURM_MEM_PER_CPU is set "
                        "— job may not have a memory request, or you're not "
                        "running inside a Slurm allocation.")


def calculation_setter(atoms, row_dc):
    pyscf_cell = cell_from_ase(atoms)
    pyscf_cell.basis = row_dc.calc_params['basis']
    pyscf_cell.spin = row_dc.spin
    pyscf_cell.verbose = 4
    pyscf_cell.ouput = row_dc.calc_params['txt']

    nprocs, mem_per_cpu = get_slurm_resouces()
    pyscf_cell.max_memory = int(nprocs * mem_per_cpu * 0.85)

    pyscf_cell.build()

    method = dft.UKS(pyscf_cell,
                     xc=row_dc.xc,
                     )
    method.chkfile = None

    calc = PySCF(atoms=atoms,
                 method=method, )

    if row_dc.dftd4: calc = SumCalculator([DFTD4(method=row_dc.xc), calc])
    atoms.calc = calc
