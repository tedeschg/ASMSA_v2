"""The solvated-MD preparation protocol, as resumable steps.

Each step is a separate method so a notebook can run them one cell at a time
and inspect the output in between, which is how the equilibration is actually
checked in practice. Every step also skips itself when its output already
exists, so re-running a cell after a kernel restart costs nothing instead of
redoing a minimization — v1's notebooks silently repeated this work and had
cells commented with "run on the first run only".

The protocol is the standard one and matches v1: box, solvate, neutralize,
minimize, NVT, NPT, production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shutil import copyfile

from .gromacs import Gromacs, read_xvg
from .mdp import ns_to_steps, write_mdp_set
from .runner import BackgroundJob


@dataclass
class ProtocolState:
    """Which artifacts exist, so steps can be skipped."""

    workdir: Path
    base: str

    def path(self, name: str) -> Path:
        return self.workdir / name

    def exists(self, name: str) -> bool:
        p = self.path(name)
        return p.exists() and p.stat().st_size > 0


@dataclass
class PreparationProtocol:
    """Prepare and run a solvated simulation of ``base``.

    Parameters
    ----------
    workdir : path
        All GROMACS files are produced here.
    base : str
        File stem, e.g. ``"trpcage"``.
    gro, topol : str
        Solute coordinate and topology filenames (outputs of ``pdb2gmx``),
        relative to ``workdir``.
    gmx : Gromacs
    box_nm : float
        Minimum solute-to-box-edge distance.
    temperature, pressure : float
    force : bool
        Redo steps even when their outputs exist.
    """

    workdir: Path
    base: str
    gro: str
    topol: str
    gmx: Gromacs
    box_nm: float = 1.2
    temperature: float = 300.0
    pressure: float = 1.0
    dt: float = 0.002
    force: bool = False
    tc_grps: str = "Protein Non-Protein"
    compressed_x_grps: str = "Protein"
    mdp: dict[str, Path] = field(default_factory=dict)
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.state = ProtocolState(self.workdir, self.base)
        # The topology is mutated in place by solvate/genion, so work on a copy
        # and leave the pristine one from pdb2gmx alone.
        self.full_top = f"{self.base}-full.top"

    # ------------------------------------------------------------- utilities
    def _skip(self, step: str, output: str) -> bool:
        if not self.force and self.state.exists(output):
            print(f"  [skip] {step}: {output} already exists "
                  f"(pass force=True to redo)")
            self.skipped.append(step)
            return True
        return False

    def write_mdps(self, production_ns: float = 100.0,
                   equil_nvt_ps: float = 100.0,
                   equil_npt_ps: float = 100.0) -> dict[str, Path]:
        """Generate all ``.mdp`` files consistently from physical time units."""
        self.mdp = write_mdp_set(
            self.workdir, production_ns=production_ns, dt=self.dt,
            ref_t=self.temperature, ref_p=self.pressure,
            equil_nvt_ps=equil_nvt_ps, equil_npt_ps=equil_npt_ps,
            compressed_x_grps=self.compressed_x_grps, tc_grps=self.tc_grps,
        )
        print(f"  wrote {len(self.mdp)} mdp files; production = "
              f"{production_ns} ns = {ns_to_steps(production_ns, self.dt)} steps")
        return self.mdp

    # ----------------------------------------------------------------- steps
    def prepare_topology_copy(self) -> Path:
        """Copy the pdb2gmx topology to the one solvate/genion will edit."""
        dst = self.state.path(self.full_top)
        if not self.force and dst.exists():
            print(f"  [skip] topology copy: {self.full_top} already exists")
            self.skipped.append("topology copy")
            return dst
        copyfile(self.state.path(self.topol), dst)
        print(f"  {self.topol} -> {self.full_top}")
        self.done.append("topology copy")
        return dst

    def make_box(self) -> str:
        out = f"{self.base}-box.gro"
        if not self._skip("box", out):
            self.gmx.editconf(f=self.gro, o=out, c=True, d=self.box_nm,
                              bt="dodecahedron")
            self.done.append("box")
        return out

    def solvate(self) -> str:
        out = f"{self.base}-solv.gro"
        if not self._skip("solvate", out):
            self.gmx.solvate(cp=f"{self.base}-box.gro", cs="spc216.gro",
                             o=out, p=self.full_top)
            self.done.append("solvate")
        return out

    def add_ions(self) -> str:
        out = f"{self.base}-ions.gro"
        if not self._skip("ions", out):
            self.gmx.grompp(f="ions.mdp", c=f"{self.base}-solv.gro",
                            p=self.full_top, o="ions.tpr", maxwarn=2)
            self.gmx.genion(s="ions.tpr", o=out, p=self.full_top,
                            pname="NA", nname="CL", neutral=True, group="SOL")
            self.done.append("ions")
        return out

    def minimize(self, background: bool = False):
        if self._skip("minimize", "em.gro"):
            return None
        self.gmx.grompp(f="minim.mdp", c=f"{self.base}-ions.gro",
                        p=self.full_top, o="em.tpr", maxwarn=2)
        self.done.append("minimize")
        if background:
            return self.gmx.mdrun_background(deffnm="em")
        self.gmx.mdrun(deffnm="em")
        return None

    def equilibrate_nvt(self, background: bool = True):
        if self._skip("nvt", "nvt.gro"):
            return None
        self.gmx.grompp(f="nvt.mdp", c="em.gro", r="em.gro",
                        p=self.full_top, o="nvt.tpr", maxwarn=2)
        self.done.append("nvt")
        if background:
            return self.gmx.mdrun_background(deffnm="nvt")
        self.gmx.mdrun(deffnm="nvt")
        return None

    def equilibrate_npt(self, background: bool = True):
        if self._skip("npt", "npt.gro"):
            return None
        self.gmx.grompp(f="npt.mdp", c="nvt.gro", r="nvt.gro", t="nvt.cpt",
                        p=self.full_top, o="npt.tpr", maxwarn=2)
        self.done.append("npt")
        if background:
            return self.gmx.mdrun_background(deffnm="npt")
        self.gmx.mdrun(deffnm="npt")
        return None

    def equilibration_report(self) -> dict:
        """Extract temperature, pressure and density from the NPT run.

        These three traces are the check that equilibration actually worked:
        temperature should sit at the setpoint, density should plateau, and
        pressure should fluctuate around the setpoint with a large spread
        (instantaneous pressure in a small box is inherently noisy -- the
        *mean* is what must match).
        """
        out = {}
        for term, fname in (("Temperature", "temp.xvg"),
                            ("Pressure", "press.xvg"),
                            ("Density", "dens.xvg")):
            try:
                self.gmx.energy(f="npt.edr", o=fname, terms=term)
                out[term.lower()] = read_xvg(self.state.path(fname))
            except Exception as exc:
                out[term.lower()] = None
                print(f"  could not extract {term}: {exc}")
        return out

    def production(self, plumed: str = "plumed.dat", from_checkpoint: bool = False,
                   background: bool = True, **mdrun_kw) -> BackgroundJob | None:
        """Start the biased production run.

        A ``HILLS`` file is created if missing, because the generated
        ``plumed.dat`` contains ``RESTART`` and PLUMED refuses to start if the
        file it is told to resume from does not exist.
        """
        hills = self.state.path("HILLS")
        if not hills.exists():
            hills.touch()
            print("  created empty HILLS (plumed.dat uses RESTART)")

        if not self.state.exists("md.tpr") or self.force:
            self.gmx.grompp(f="md.mdp", c="npt.gro", t="npt.cpt",
                            p=self.full_top, o="md.tpr", maxwarn=2)
        else:
            print("  [skip] grompp: md.tpr already exists")

        kw = dict(deffnm="md", plumed=plumed, **mdrun_kw)
        if from_checkpoint:
            kw["cpi"] = "md.cpt"
        self.done.append("production")
        if background:
            return self.gmx.mdrun_background(**kw)
        self.gmx.mdrun(**kw)
        return None

    def extend(self, additional_ns: float) -> None:
        """Extend a finished run; then call ``production(from_checkpoint=True)``."""
        ps = additional_ns * 1000.0
        self.gmx.convert_tpr(s="md.tpr", o="md-new.tpr", extend=ps)
        (self.workdir / "md-new.tpr").replace(self.workdir / "md.tpr")
        print(f"  extended md.tpr by {additional_ns} ns; "
              f"restart with production(from_checkpoint=True)")

    def postprocess_trajectory(self, traj: str = "md.xtc",
                               out: str = "fit.xtc") -> str:
        """Undo periodic-boundary jumps, then fit out global rotation/translation.

        Both steps are required before re-featurizing: the featurizer uses
        inter-atomic distances, which a molecule split across the periodic
        boundary would corrupt.
        """
        if self._skip("postprocess", out):
            return out
        self.gmx.trjconv(f=traj, s=f"{self.base}-box.gro", pbc="nojump",
                         o="pbc.xtc", groups="Protein\nProtein\n")
        self.gmx.trjconv(f="pbc.xtc", s=f"{self.base}-box.gro", fit="rot+trans",
                         o=out, groups="Protein\nProtein\n")
        self.done.append("postprocess")
        return out

    # --------------------------------------------------------------- summary
    def status(self) -> str:
        checks = [
            ("topology (full)", self.full_top),
            ("box", f"{self.base}-box.gro"),
            ("solvated", f"{self.base}-solv.gro"),
            ("neutralized", f"{self.base}-ions.gro"),
            ("minimized", "em.gro"),
            ("NVT", "nvt.gro"),
            ("NPT", "npt.gro"),
            ("production tpr", "md.tpr"),
            ("production traj", "md.xtc"),
            ("HILLS", "HILLS"),
        ]
        lines = [f"PreparationProtocol in {self.workdir}"]
        for label, fname in checks:
            mark = "done" if self.state.exists(fname) else "--"
            lines.append(f"  [{mark:>4}] {label:<16} {fname}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = []
        for line in self.status().splitlines()[1:]:
            done = "[done]" in line
            label = line.split("]", 1)[1].strip()
            colour = "#1a7f37" if done else "#8c959f"
            mark = "&#10003;" if done else "&#183;"
            rows.append(
                f"<tr><td style='padding:1px 8px;color:{colour}'>{mark}</td>"
                f"<td style='padding:1px 8px;color:{colour}'>{label}</td></tr>"
            )
        return (f"<table style='font-size:92%;border-collapse:collapse'>"
                f"<caption style='text-align:left'><b>MD protocol</b> &mdash; "
                f"<code>{self.workdir}</code></caption>"
                + "".join(rows) + "</table>")

    def __repr__(self) -> str:
        return self.status()
