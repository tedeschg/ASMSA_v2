"""Pluggable execution backends for external commands.

v1 ran GROMACS as Kubernetes Jobs on a specific cluster, which made the
notebooks unusable anywhere else. Execution is isolated behind this small
interface instead: :class:`LocalRunner` shells out on the current machine,
and a cluster backend can be added later without touching the notebooks or
the :class:`~asmsa_v2.md.gromacs.Gromacs` wrapper.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    """Outcome of one command."""

    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    log_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def check(self) -> "RunResult":
        """Raise with a useful excerpt if the command failed."""
        if self.ok:
            return self
        tail = "\n".join((self.stderr or self.stdout or "").splitlines()[-25:])
        raise RuntimeError(
            f"command failed (exit {self.returncode}):\n"
            f"  {' '.join(self.args)}\n"
            f"--- last lines of output ---\n{tail}"
            + (f"\n--- full log: {self.log_path} ---" if self.log_path else "")
        )

    def __repr__(self) -> str:
        return (f"RunResult(ok={self.ok}, returncode={self.returncode}, "
                f"{self.duration_s:.1f}s, cmd={' '.join(self.args[:4])}...)")


class Runner(ABC):
    """Executes external commands."""

    @abstractmethod
    def run(self, args: list[str], stdin: str | None = None,
            cwd: str | Path | None = None, env: dict | None = None,
            log_path: str | Path | None = None, timeout: float | None = None,
            ) -> RunResult:
        ...

    def which(self, program: str) -> str | None:
        return shutil.which(program)

    def available(self, program: str) -> bool:
        return self.which(program) is not None


class LocalRunner(Runner):
    """Run commands as subprocesses on this machine.

    Parameters
    ----------
    workdir : path
        Default working directory for commands.
    log_dir : path, optional
        Where to tee combined output. Defaults to ``workdir/logs``. Keeping
        the full log matters because GROMACS failures are usually explained
        hundreds of lines before the end of its output.
    echo : bool
        Print a one-line summary of each command as it is run.
    """

    def __init__(self, workdir: str | Path = ".", log_dir: str | Path | None = None,
                 echo: bool = True):
        self.workdir = Path(workdir).resolve()
        self.log_dir = Path(log_dir) if log_dir else self.workdir / "logs"
        self.echo = echo

    def run(self, args, stdin=None, cwd=None, env=None, log_path=None,
            timeout=None) -> RunResult:
        args = [str(a) for a in args]
        cwd = Path(cwd) if cwd else self.workdir
        cwd.mkdir(parents=True, exist_ok=True)
        full_env = {**os.environ, **(env or {})}

        if log_path is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stem = "-".join(args[:2]).replace("/", "_")[:60] or "cmd"
            log_path = self.log_dir / f"{stem}-{int(time.time())}.log"
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.echo:
            print(f"$ {' '.join(args)}"
                  + (f"  <<< {stdin!r}" if stdin else "")
                  + f"   [log: {log_path.name}]")

        t0 = time.time()
        try:
            proc = subprocess.run(
                args, input=stdin, cwd=str(cwd), env=full_env,
                capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{args[0]!r} not found on PATH. "
                + (f"Install GROMACS, or use the project's Docker image which "
                   f"bundles it." if args[0].startswith("gmx") else "")
            ) from None
        except subprocess.TimeoutExpired as exc:
            dt = time.time() - t0
            log_path.write_text((exc.stdout or "") + (exc.stderr or ""))
            return RunResult(args, 124, exc.stdout or "", exc.stderr or "",
                             dt, log_path)

        dt = time.time() - t0
        log_path.write_text(
            f"$ {' '.join(args)}\n"
            + (f"stdin: {stdin}\n" if stdin else "")
            + f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
        )
        return RunResult(args, proc.returncode, proc.stdout, proc.stderr, dt, log_path)


@dataclass
class BackgroundJob:
    """A long-running command started with :meth:`LocalRunner.run_background`."""

    args: list[str]
    popen: "subprocess.Popen"
    log_path: Path
    started: float = field(default_factory=time.time)

    @property
    def running(self) -> bool:
        return self.popen.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.popen.poll()

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started

    def log(self, tail: int = 25) -> str:
        """Last ``tail`` lines of the job's combined output."""
        if not self.log_path.exists():
            return "(no output yet)"
        lines = self.log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    def status(self) -> str:
        if self.running:
            return (f"RUNNING for {self.elapsed_s / 60:.1f} min  "
                    f"(log: {self.log_path})")
        return (f"{'DONE' if self.returncode == 0 else f'FAILED ({self.returncode})'} "
                f"after {self.elapsed_s / 60:.1f} min  (log: {self.log_path})")

    def wait(self, timeout: float | None = None) -> int:
        return self.popen.wait(timeout=timeout)

    def kill(self) -> None:
        if self.running:
            self.popen.terminate()
            try:
                self.popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.popen.kill()

    def __repr__(self) -> str:
        return f"BackgroundJob({self.status()})"


def run_background(runner: LocalRunner, args, cwd=None, env=None,
                   log_path=None) -> BackgroundJob:
    """Start a command detached, streaming its output to a log file.

    Production MD runs for hours; blocking the notebook kernel on it would
    make the intermediate-analysis cells unusable, which is the whole point
    of watching a metadynamics run converge.
    """
    args = [str(a) for a in args]
    cwd = Path(cwd) if cwd else runner.workdir
    cwd.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        runner.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = runner.log_dir / f"{'-'.join(args[:2]).replace('/', '_')[:60]}.log"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = open(log_path, "w")
    fh.write(f"$ {' '.join(args)}\n")
    fh.flush()
    popen = subprocess.Popen(
        args, cwd=str(cwd), env={**os.environ, **(env or {})},
        stdout=fh, stderr=subprocess.STDOUT, text=True,
    )
    # close our handle once the child exits, without blocking the caller
    threading.Thread(target=lambda: (popen.wait(), fh.close()), daemon=True).start()
    if runner.echo:
        print(f"$ {' '.join(args)}  [background, log: {log_path}]")
    return BackgroundJob(args, popen, log_path)
