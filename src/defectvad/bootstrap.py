# src/defectvad/bootstrap.py

import os
from defectvad.config import load_yaml, resolve_paths_cfg, validate_paths
from defectvad.common.backbone import set_backbone_dir


def init_from_paths_yaml(paths_yaml: str = "configs/paths.yaml") -> dict:
    """
    Initialize project paths from configs/paths.yaml.
    Must be called BEFORE any model/backbone is created.
    """
    cfg = load_yaml(paths_yaml)
    paths = resolve_paths_cfg(cfg)

    # 환경 변수로 backbone_dir 오버라이드 허용
    backbone_dir = os.getenv("BACKBONE_DIR", paths.get("backbone_dir"))
    if backbone_dir is None:
        raise ValueError("backbone_dir is not defined")

    set_backbone_dir(backbone_dir)

    # 선택: 다른 경로들도 반환
    validate_paths(paths)

    return paths
