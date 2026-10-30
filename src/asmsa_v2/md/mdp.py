"""GROMACS ``.mdp`` parameter files, generated rather than copied.

v1 shipped four static ``.mdp`` files plus a fifth built by string
interpolation inside a notebook cell. Generating all of them from one place
means the step count, temperature and output stride cannot silently disagree
between equilibration and production, and the whole protocol is visible as
Python rather than scattered across files.

The protocol itself is the standard solvated-protein one (box, solvate,
neutralize, minimize, NVT, NPT, production), unchanged from v1.
"""
from __future__ import annotations

from pathlib import Path

_COMMON_NONBONDED = """\
; Neighbour searching
cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 10
rcoulomb                = 1.0
rvdw                    = 1.0
; Electrostatics
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.16
; Periodic boundary conditions
pbc                     = xyz
DispCorr                = EnerPres
"""

_CONSTRAINTS = """\
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
"""


def minim_mdp(emtol: float = 1000.0, nsteps: int = 50000,
              emstep: float = 0.01) -> str:
    """Steepest-descent energy minimization."""
    return f"""\
; Energy minimization
integrator              = steep
emtol                   = {emtol}
emstep                  = {emstep}
nsteps                  = {nsteps}
nstlist                 = 10
cutoff-scheme           = Verlet
ns_type                 = grid
coulombtype             = PME
rcoulomb                = 1.0
rvdw                    = 1.0
pbc                     = xyz
"""


def ions_mdp() -> str:
    """Minimal parameters, only used to build the ``.tpr`` that ``genion`` needs."""
    return minim_mdp(emtol=1000.0, nsteps=50000)


def nvt_mdp(nsteps: int = 50000, dt: float = 0.002, ref_t: float = 300.0,
            nstout: int = 500, posres: bool = True,
            tc_grps: str = "Protein Non-Protein") -> str:
    """Isothermal-isochoric equilibration with the solute position-restrained."""
    taus = " ".join(["0.1"] * len(tc_grps.split()))
    temps = " ".join([f"{ref_t}"] * len(tc_grps.split()))
    return f"""\
; NVT equilibration
{'define                  = -DPOSRES' if posres else ''}
integrator              = md
nsteps                  = {nsteps}
dt                      = {dt}
; Output
nstxout                 = {nstout}
nstvout                 = {nstout}
nstenergy               = {nstout}
nstlog                  = {nstout}
; Bonds
continuation            = no
{_CONSTRAINTS}{_COMMON_NONBONDED}\
; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = {tc_grps}
tau_t                   = {taus}
ref_t                   = {temps}
; No pressure coupling in NVT
pcoupl                  = no
; Velocity generation
gen_vel                 = yes
gen_temp                = {ref_t}
gen_seed                = -1
"""


def npt_mdp(nsteps: int = 50000, dt: float = 0.002, ref_t: float = 300.0,
            ref_p: float = 1.0, nstout: int = 500, posres: bool = True,
            tc_grps: str = "Protein Non-Protein") -> str:
    """Isothermal-isobaric equilibration; continues from NVT velocities."""
    taus = " ".join(["0.1"] * len(tc_grps.split()))
    temps = " ".join([f"{ref_t}"] * len(tc_grps.split()))
    return f"""\
; NPT equilibration
{'define                  = -DPOSRES' if posres else ''}
integrator              = md
nsteps                  = {nsteps}
dt                      = {dt}
; Output
nstxout                 = {nstout}
nstvout                 = {nstout}
nstenergy               = {nstout}
nstlog                  = {nstout}
; Bonds
continuation            = yes
{_CONSTRAINTS}{_COMMON_NONBONDED}\
; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = {tc_grps}
tau_t                   = {taus}
ref_t                   = {temps}
; Pressure coupling
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = {ref_p}
compressibility         = 4.5e-5
refcoord_scaling        = com
; Velocities come from NVT
gen_vel                 = no
"""


def production_mdp(nsteps: int, dt: float = 0.002, ref_t: float = 300.0,
                   ref_p: float = 1.0, nstxout_compressed: int = 5000,
                   nstenergy: int = 5000,
                   compressed_x_grps: str = "Protein",
                   tc_grps: str = "Protein Non-Protein") -> str:
    """Production MD, to be biased by PLUMED.

    ``compressed_x_grps`` defaults to the solute: writing only the solute
    keeps the ``.xtc`` small enough to re-featurize in the analysis notebook,
    and the solvent is not part of any ASMSA collective variable. Set it to
    ``System`` if you need the solvent.

    Pressure coupling uses ``C-rescale``, the correct-ensemble barostat that
    superseded ``Parrinello-Rahman`` for this purpose in modern GROMACS
    (Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020)); v1 used
    Parrinello-Rahman, which does not sample the exact NPT ensemble.
    """
    taus = " ".join(["0.1"] * len(tc_grps.split()))
    temps = " ".join([f"{ref_t}"] * len(tc_grps.split()))
    return f"""\
; Production MD (biased by PLUMED)
integrator              = md
nsteps                  = {int(nsteps)}
dt                      = {dt}
; Output -- suppress the bulky .trr, keep a compressed trajectory
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = {nstenergy}
nstlog                  = {nstenergy}
nstxout-compressed      = {nstxout_compressed}
compressed-x-grps       = {compressed_x_grps}
; Bonds
continuation            = yes
{_CONSTRAINTS}{_COMMON_NONBONDED}\
; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = {tc_grps}
tau_t                   = {taus}
ref_t                   = {temps}
; Pressure coupling
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = {ref_p}
compressibility         = 4.5e-5
; Velocities come from NPT
gen_vel                 = no
"""


def ns_to_steps(ns: float, dt: float = 0.002) -> int:
    """Convert a simulation length in nanoseconds to a step count.

    v1's notebook asked the user to pick from a commented list of magic step
    counts (``mdsteps=250000000  # 500 ns``), which is exactly the kind of
    thing that silently goes wrong when ``dt`` changes.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    return int(round(ns * 1000.0 / dt))


def steps_to_ns(steps: int, dt: float = 0.002) -> float:
    return steps * dt / 1000.0


def write_mdp_set(
    out_dir: str | Path,
    production_ns: float = 100.0,
    dt: float = 0.002,
    ref_t: float = 300.0,
    ref_p: float = 1.0,
    equil_nvt_ps: float = 100.0,
    equil_npt_ps: float = 100.0,
    compressed_x_grps: str = "Protein",
    tc_grps: str = "Protein Non-Protein",
) -> dict[str, Path]:
    """Write the whole ``.mdp`` set for one protocol, consistently.

    Returns a mapping of stage name to written path. Lengths are given in
    physical time (ps / ns) and converted once, so every stage agrees.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nvt_steps = int(round(equil_nvt_ps / dt))
    npt_steps = int(round(equil_npt_ps / dt))
    files = {
        "ions": ions_mdp(),
        "minim": minim_mdp(),
        "nvt": nvt_mdp(nsteps=nvt_steps, dt=dt, ref_t=ref_t, tc_grps=tc_grps),
        "npt": npt_mdp(nsteps=npt_steps, dt=dt, ref_t=ref_t, ref_p=ref_p,
                       tc_grps=tc_grps),
        "md": production_mdp(ns_to_steps(production_ns, dt), dt=dt, ref_t=ref_t,
                             ref_p=ref_p, compressed_x_grps=compressed_x_grps,
                             tc_grps=tc_grps),
    }
    paths: dict[str, Path] = {}
    for name, text in files.items():
        p = out_dir / f"{name}.mdp"
        p.write_text(text)
        paths[name] = p
    return paths
