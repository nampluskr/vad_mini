# src/vad_mini/config.py

import os
import yaml


def load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths_cfg(cfg: dict) -> dict:
    """
    Resolve paths configuration.
    Priority:
      1) Environment variable VAD_PROFILE
      2) cfg['profile']
      3) single cfg['paths']
    """
    if "profiles" in cfg:
        profile = os.getenv("VAD_PROFILE", cfg.get("profile"))
        if profile is None:
            raise ValueError(
                "profiles defined but no profile selected. "
                "Set `profile:` in yaml or export VAD_PROFILE."
            )
        if profile not in cfg["profiles"]:
            raise ValueError(f"Unknown profile: {profile}")
        return cfg["profiles"][profile]

    if "paths" in cfg:
        return cfg["paths"]

    raise ValueError("Invalid paths.yaml format")


def validate_paths(paths: dict):
    for key, value in paths.items():
        if key.endswith("_dir"):
            if not os.path.exists(value):
                raise FileNotFoundError(f"{key} not found: {value}")
