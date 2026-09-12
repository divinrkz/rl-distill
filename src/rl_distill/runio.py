"""Run directories and provenance. Every run saves the config that produced it."""

import json
from datetime import datetime
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_run_dir(base, run_name):
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir, config):
    config = dict(config)
    config["_saved_at"] = datetime.now().isoformat(timespec="seconds")
    with open(Path(run_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
