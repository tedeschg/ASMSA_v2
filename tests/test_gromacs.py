import pytest

from asmsa_v2.md import gromacs, runner as md_runner
from asmsa_v2.md.runner import LocalRunner, RunResult, run_background


@pytest.mark.parametrize(
    ("manual_result", "expected"),
    [
        (RunResult([], 0, stdout="PYTORCH_MODEL manual"), "enabled"),
        (
            RunResult([], 1, stderr="LIST OF DOCUMENTED ACTIONS"),
            "NOT REGISTERED in the selected PLUMED executable",
        ),
        (
            RunResult([], 1, stderr="unknown option"),
            "unknown - could not verify PYTORCH_MODEL",
        ),
    ],
)
def test_check_environment_probes_pytorch_action(
    monkeypatch, manual_result, expected
):
    calls = []

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, args):
            calls.append(args)
            if args == ["/fake/gmx", "--version"]:
                return RunResult(args, 0, stdout="GROMACS version: 2026.3")
            if args == ["/fake/plumed", "info", "--version"]:
                return RunResult(args, 0, stdout="2.11")
            if args == [
                "/fake/plumed", "manual", "--action", "PYTORCH_MODEL"
            ]:
                return manual_result
            raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(gromacs.shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(gromacs, "LocalRunner", FakeRunner)

    result = gromacs.check_environment()

    assert result["plumed pytorch module"].startswith(expected)
    assert result["plumed executable"] == "/fake/plumed"
    assert [
        "/fake/plumed", "manual", "--action", "PYTORCH_MODEL"
    ] in calls


def test_mdrun_uses_gromacs_gpu_id_flag(tmp_path):
    runner = gromacs.Gromacs(tmp_path, dry_run=True, echo=False)

    result = runner.mdrun(deffnm="md", gpu_id="0")

    assert result.args == [
        "gmx", "mdrun", "-deffnm", "md", "-gpu_id", "0"
    ]


def test_mdrun_background_uses_gromacs_gpu_id_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gromacs, "run_background", lambda runner, args, cwd: args
    )
    runner = gromacs.Gromacs(tmp_path, dry_run=False, echo=False)

    args = runner.mdrun_background(deffnm="md", gpu_id="0")

    assert args == ["gmx", "mdrun", "-deffnm", "md", "-gpu_id", "0"]


def test_mdrun_background_can_detach_from_kernel(monkeypatch, tmp_path):
    calls = {}

    def fake_run_background(runner, args, cwd, detached=False):
        calls["detached"] = detached
        return args

    monkeypatch.setattr(gromacs, "run_background", fake_run_background)
    runner = gromacs.Gromacs(tmp_path, dry_run=False, echo=False)

    runner.mdrun_background(deffnm="md", detached=True)

    assert calls["detached"] is True


def test_run_background_starts_detached_session(monkeypatch, tmp_path):
    calls = {}

    class FakePopen:
        def __init__(self, *args, **kwargs):
            calls.update(kwargs)

        def wait(self):
            return 0

    monkeypatch.setattr(md_runner.subprocess, "Popen", FakePopen)
    runner = LocalRunner(tmp_path, echo=False)

    run_background(runner, ["gmx", "mdrun"], detached=True)

    assert calls["start_new_session"] is True