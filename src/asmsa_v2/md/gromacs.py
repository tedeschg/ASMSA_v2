"""Thin, typed wrapper over the ``gmx`` command line.

v1 used the third-party ``GromacsWrapper`` package plus a Kubernetes job
submitter. This wrapper instead builds ``gmx`` argument lists directly and
hands them to a :class:`~asmsa_v2.md.runner.Runner`, which keeps three
properties the notebooks need:

- the exact command is visible and reproducible (printed, and logged),
- commands can be built and inspected **without** GROMACS installed, which
  is what makes the pipeline testable and the notebooks readable on a laptop,
- the execution backend is swappable.

Only the subcommands the ASMSA pipeline actually uses are wrapped.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .runner import BackgroundJob, LocalRunner, RunResult, Runner, run_background


def _flags(opts: dict) -> list[str]:
    """Turn a kwargs dict into ``gmx``-style flags.

    ``True`` becomes a bare ``-flag``, ``False``/``None`` is dropped, a list
    becomes repeated values after one flag, and ``_`` in names becomes ``-``
    so Python-legal keywords map onto GROMACS' hyphenated options.
    """
    out: list[str] = []
    for key, val in opts.items():
        if val is None or val is False:
            continue
        flag = "-" + key.replace("__", "#").replace("_", "-").replace("#", "_")
        if val is True:
            out.append(flag)
        elif isinstance(val, (list, tuple)):
            out.append(flag)
            out.extend(str(v) for v in val)
        else:
            out.extend([flag, str(val)])
    return out


class Gromacs:
    """Build and run ``gmx`` commands.

    Parameters
    ----------
    workdir : path
        Directory the commands run in; all relative paths are resolved there.
    runner : Runner, optional
        Defaults to a :class:`LocalRunner` on ``workdir``.
    gmx : str
        Executable name. Use ``"gmx_mpi"`` for an MPI build.
    dry_run : bool
        Build and record commands without executing them. Lets a notebook be
        read end to end, and the pipeline be tested, on a machine with no
        GROMACS.
    """

    def __init__(
        self,
        workdir: str | Path = ".",
        runner: Runner | None = None,
        gmx: str = "gmx",
        dry_run: bool = False,
        echo: bool = True,
    ):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.gmx = gmx
        self.dry_run = dry_run
        self.runner = runner or LocalRunner(self.workdir, echo=echo)
        self.commands: list[list[str]] = []

    # --------------------------------------------------------------- plumbing
    @property
    def available(self) -> bool:
        return shutil.which(self.gmx) is not None

    def version(self) -> str:
        if not self.available:
            return f"{self.gmx}: not found on PATH"
        res = self.runner.run([self.gmx, "--version"])
        for line in res.stdout.splitlines():
            if "GROMACS version" in line:
                return line.strip()
        return res.stdout.splitlines()[0] if res.stdout else "unknown"

    def _call(self, subcommand: str, stdin: str | None = None,
              check: bool = True, timeout: float | None = None, **opts) -> RunResult:
        args = [self.gmx, subcommand] + _flags(opts)
        self.commands.append(args)
        if self.dry_run:
            print(f"[dry-run] $ {' '.join(args)}"
                  + (f"  <<< {stdin!r}" if stdin else ""))
            return RunResult(args, 0, stdout="(dry run)")
        res = self.runner.run(args, stdin=stdin, cwd=self.workdir, timeout=timeout)
        return res.check() if check else res

    def script(self) -> str:
        """Every command issued so far, as a shell script.

        Handy for reproducibility: paste it into a cluster job instead of
        re-running the notebook.
        """
        return "\n".join(["#!/bin/sh", "set -e", ""]
                         + [" ".join(c) for c in self.commands] + [""])

    # ------------------------------------------------------------ topology
    def pdb2gmx(self, f, o, p, n=None, water="tip3p", ff="amber99sb-ildn",
                ignh=True, **opts) -> RunResult:
        """Build a topology from a PDB. ``ignh=True`` re-adds hydrogens."""
        return self._call("pdb2gmx", f=f, o=o, p=p, n=n, water=water, ff=ff,
                          ignh=ignh, **opts)

    def editconf(self, f, o, c=True, d=1.0, bt="dodecahedron", **opts) -> RunResult:
        """Place the solute in a box."""
        return self._call("editconf", f=f, o=o, c=c, d=d, bt=bt, **opts)

    def solvate(self, cp, o, p, cs="spc216.gro", **opts) -> RunResult:
        return self._call("solvate", cp=cp, cs=cs, o=o, p=p, **opts)

    def grompp(self, f, c, p, o, r=None, t=None, n=None,
               maxwarn=None, **opts) -> RunResult:
        """Preprocess into a ``.tpr``."""
        return self._call("grompp", f=f, c=c, p=p, o=o, r=r, t=t, n=n,
                          maxwarn=maxwarn, **opts)

    def genion(self, s, o, p, pname="NA", nname="CL", neutral=True,
               n=None, group="SOL", **opts) -> RunResult:
        """Add counter-ions, replacing solvent molecules.

        ``group`` is fed on stdin to answer the interactive group prompt.
        """
        return self._call("genion", stdin=f"{group}\n", s=s, o=o, p=p,
                          pname=pname, nname=nname, neutral=neutral, n=n, **opts)

    def select(self, s, on, select, **opts) -> RunResult:
        return self._call("select", s=s, on=on, select=select, **opts)

    def make_ndx(self, f, o, groups: str = "q\n", **opts) -> RunResult:
        return self._call("make_ndx", stdin=groups, f=f, o=o, **opts)

    # ------------------------------------------------------------------- md
    def mdrun(self, deffnm, plumed=None, nt=None, ntomp=None, gpu_id=None,
              cpi=None, check: bool = True, timeout=None, **opts) -> RunResult:
        """Run MD in the foreground. Use :meth:`mdrun_background` for production."""
        return self._call("mdrun", deffnm=deffnm, plumed=plumed, nt=nt,
                          ntomp=ntomp, gpu_id=gpu_id, cpi=cpi,
                          check=check, timeout=timeout, **opts)

    def mdrun_background(self, deffnm, plumed=None, nt=None, ntomp=None,
                         gpu_id=None, cpi=None, **opts) -> BackgroundJob:
        """Start MD detached so the notebook kernel stays usable.

        A production metadynamics run takes hours to days; poll the returned
        job's :meth:`~asmsa_v2.md.runner.BackgroundJob.status` and run the
        analysis cells against the partial trajectory meanwhile.
        """
        args = [self.gmx, "mdrun"] + _flags(
            dict(deffnm=deffnm, plumed=plumed, nt=nt, ntomp=ntomp,
                 gpu_id=gpu_id, cpi=cpi, **opts)
        )
        self.commands.append(args)
        if self.dry_run:
            print(f"[dry-run, background] $ {' '.join(args)}")
            raise RuntimeError(
                "cannot start a background job in dry_run mode; "
                "set dry_run=False once GROMACS is available"
            )
        if not isinstance(self.runner, LocalRunner):
            raise TypeError("mdrun_background requires a LocalRunner")
        return run_background(self.runner, args, cwd=self.workdir)

    # -------------------------------------------------------------- analysis
    def trjconv(self, f, o, s=None, n=None, pbc=None, fit=None,
                groups: str | None = None, **opts) -> RunResult:
        """Convert/clean a trajectory.

        ``groups`` answers the interactive group selection; ``pbc='nojump'``
        then ``fit='rot+trans'`` is the standard pair for making a
        metadynamics trajectory analysable.
        """
        return self._call("trjconv", stdin=groups, f=f, o=o, s=s, n=n,
                          pbc=pbc, fit=fit, **opts)

    def energy(self, f, o, terms: str = "Potential", **opts) -> RunResult:
        return self._call("energy", stdin=f"{terms}\n0\n", f=f, o=o, **opts)

    def rms(self, s, f, o=None, n=None, f2=None, m=None, groups=None,
            **opts) -> RunResult:
        return self._call("rms", stdin=groups, s=s, f=f, o=o, n=n, f2=f2,
                          m=m, **opts)

    def convert_tpr(self, s, o, extend=None, until=None, **opts) -> RunResult:
        """Extend a finished run so it can continue from its checkpoint."""
        return self._call("convert_tpr", s=s, o=o, extend=extend,
                          until=until, **opts)


def read_xvg(path: str | Path):
    """Read a GROMACS ``.xvg`` into an array, skipping comment lines."""
    import numpy as np

    return np.loadtxt(str(path), comments=["#", "@"])


def check_environment(gmx: str = "gmx", plumed: str = "plumed") -> dict[str, str]:
    """Report which external tools are present.

    Called by the notebooks up front so a missing dependency is reported in
    one place instead of surfacing as a confusing failure twenty cells later.
    """
    out: dict[str, str] = {}
    g = shutil.which(gmx)
    out[gmx] = g or "NOT FOUND"
    if g:
        try:
            res = LocalRunner(echo=False).run([gmx, "--version"])
            for line in res.stdout.splitlines():
                if "GROMACS version" in line:
                    out[gmx] = line.split(":", 1)[-1].strip()
                    break
        except Exception:
            pass
    p = shutil.which(plumed)
    out[plumed] = p or "NOT FOUND"
    if p:
        try:
            res = LocalRunner(echo=False).run([plumed, "info", "--version"])
            if res.stdout.strip():
                out[plumed] = res.stdout.strip().splitlines()[0]
        except Exception:
            pass
    # Is PLUMED's pytorch module (PYTORCH_MODEL_CV) compiled in?
    if p:
        try:
            res = LocalRunner(echo=False).run([plumed, "info", "--modules"])
            mods = (res.stdout or "").lower()
            out["plumed pytorch module"] = (
                "enabled" if "pytorch" in mods else
                "NOT ENABLED - PYTORCH_MODEL_CV will be unavailable"
            )
        except Exception:
            out["plumed pytorch module"] = "unknown"
    return out
