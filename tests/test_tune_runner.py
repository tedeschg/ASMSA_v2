import json

import numpy as np
import pytest
import torch

from asmsa_v2.tune import runner
from asmsa_v2.utils.project import Project


def test_run_config_loads_saved_features_and_records_status(tmp_path, monkeypatch):
    project = Project(name="test", root=tmp_path)
    project.save()
    features = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    torch.save({"features": features}, project.features_file)
    np.savez(project.split_file, train=np.array([0, 1, 2, 3]),
             val=np.array([4, 5]), test=np.array([6, 7, 8, 9]))

    config_path = tmp_path / "search.json"
    config_path.write_text(json.dumps({
        "project_dir": str(tmp_path),
        "search": {
            "space": {"ae_layers": [2, 3], "lr_ae": [0.0001, 0.001]},
            "n_trials": 4,
            "epochs": 5,
            "objective": "kl",
            "vamp_lag": 20,
            "kl_threshold": 0.25,
            "n_jobs": 1,
            "prune": True,
            "seed": 7,
        },
    }))
    status_path = tmp_path / "status.json"
    captured = {}
    expected_result = object()

    def fake_tune(train, val, **kwargs):
        assert json.loads(status_path.read_text())["state"] == "running"
        captured["train"] = train["complex"]
        captured["val"] = val["complex"]
        captured["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(runner, "tune", fake_tune)

    result = runner.run_config(config_path, status_path)

    assert result is expected_result
    torch.testing.assert_close(captured["train"], features[:4])
    torch.testing.assert_close(captured["val"], features[4:6])
    assert captured["kwargs"]["n_trials"] == 4
    assert captured["kwargs"]["space"].ae_layers == [2, 3]
    assert json.loads(status_path.read_text())["state"] == "complete"


def test_run_config_records_failure_and_reraises(tmp_path, monkeypatch):
    project = Project(name="test", root=tmp_path)
    project.save()
    torch.save({"features": torch.zeros((2, 2))}, project.features_file)
    np.savez(project.split_file, train=np.array([0]), val=np.array([1]))
    config_path = tmp_path / "search.json"
    config_path.write_text(json.dumps({
        "project_dir": str(tmp_path),
        "search": {
            "space": {},
            "n_trials": 1,
            "epochs": 1,
            "objective": "kl",
            "vamp_lag": 1,
            "kl_threshold": 0.25,
            "n_jobs": 1,
            "prune": True,
            "seed": 0,
        },
    }))
    status_path = tmp_path / "status.json"

    def fail_tune(*args, **kwargs):
        raise ValueError("invalid test search")

    monkeypatch.setattr(runner, "tune", fail_tune)

    with pytest.raises(ValueError, match="invalid test search"):
        runner.run_config(config_path, status_path)

    status = json.loads(status_path.read_text())
    assert status["state"] == "failed"
    assert status["error"] == "ValueError: invalid test search"
