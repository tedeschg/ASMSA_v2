"""Run a tuning search from files, independently of a notebook kernel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.project import Project
from .space import SearchSpace
from .tuner import tune


def run_config(config_path: str | Path, status_path: str | Path):
    """Load prepared features and run the search described by a JSON config."""
    config = json.loads(Path(config_path).read_text())
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"state": "running"}
    status_path.write_text(json.dumps(status, indent=2))

    try:
        project = Project.load(config["project_dir"])
        bundle = torch.load(project.features_file, weights_only=False)
        features = bundle["features"]
        split = np.load(project.split_file, allow_pickle=False)
        train_tensors = {"complex": features[split["train"]]}
        val_tensors = {"complex": features[split["val"]]}

        search: dict[str, Any] = config["search"]
        result = tune(
            train_tensors,
            val_tensors,
            prior_factory=project.make_prior,
            space=SearchSpace(**search["space"]),
            n_trials=search["n_trials"],
            epochs=search["epochs"],
            objective=search["objective"],
            vamp_lag=search["vamp_lag"],
            kl_threshold=search["kl_threshold"],
            n_jobs=search["n_jobs"],
            prune=search["prune"],
            seed=search["seed"],
            out_dir=project.dir("tuning"),
            extra_train_kwargs={"time_lagged": bool(project.time_lag)},
        )
    except Exception as exc:
        status_path.write_text(json.dumps({
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2))
        raise

    status_path.write_text(json.dumps({
        "state": "complete",
        "result": str(project.tuning_file),
    }, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON file describing the search")
    parser.add_argument("--status", required=True, help="Path to write search status")
    args = parser.parse_args(argv)
    result = run_config(args.config, args.status)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
