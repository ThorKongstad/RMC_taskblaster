import os
#from gpaw import GPAW
from dftd4.ase import DFTD4
from ase.calculators.mixing import SumCalculator
from pyscf.pbc import dft, gto
from pyscf.pbc.tools.pyscf_ase import PySCF, cell_from_ase, ase_atoms_to_pyscf


def get_slurm_resources(safety_factor=0.85):
    """Resources Slurm granted this job: (max_memory_mb, ncpus).

    Falls back to physical node RAM / os.cpu_count() if no Slurm env
    vars are present (e.g. running interactively).
    """
    # --- CPUs ---
    ncpus_raw = os.environ.get("SLURM_CPUS_PER_TASK") \
                or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if ncpus_raw is not None:
        # SLURM_JOB_CPUS_PER_NODE can be "4(x2)" style for multi-node jobs
        ncpus = int(ncpus_raw.split("(")[0])
    else:
        ncpus = os.cpu_count() or 1

    # --- Memory ---
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_per_node is not None:
        total_mb = int(mem_per_node)
    else:
        mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
        if mem_per_cpu is not None:
            total_mb = int(mem_per_cpu) * ncpus
        else:
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            total_mb = kb // 1024

    max_memory_mb = int(total_mb * safety_factor)
    return max_memory_mb, ncpus


def calculation_setter(atoms, row_dc):
    pyscf_cell = cell_from_ase(atoms)
    pyscf_cell.basis = row_dc.calc_params['basis']
    pyscf_cell.spin = row_dc.spin
    pyscf_cell.verbose = 4
    pyscf_cell.ouput = row_dc.calc_params['txt']

    nprocs, mem_per_cpu = get_slurm_resources()
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
