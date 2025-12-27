# 1. `configs/paths.example.yaml` (레포에 커밋)

```yaml
# configs/paths.example.yaml
# 각 개발 환경에 맞게 profile을 선택하거나, paths.yaml을 개인용으로 복사해서 사용

profile: wsl

profiles:
  wsl:
    backbone_dir: /mnt/d/deep_learning/backbones
    dataset_dir: /mnt/d/datasets/mvtec
    output_dir: /mnt/d/experiments/defectvad/outputs
    cache_dir: /mnt/d/experiments/defectvad/cache

  linux_server:
    backbone_dir: /data/backbones
    dataset_dir: /data/datasets/mvtec
    output_dir: /data/outputs/defectvad
    cache_dir: /data/cache/defectvad

  windows:
    backbone_dir: D:/deep_learning/backbones
    dataset_dir: D:/datasets/mvtec
    output_dir: D:/experiments/defectvad/outputs
    cache_dir: D:/experiments/defectvad/cache
```

---

# 2. `configs/paths.yaml` (개인용, gitignore)

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

`.gitignore`에 추가:

```gitignore
configs/paths.yaml
```

---

# 3. 설정 로더 구현

`src/defectvad/utils/config.py`

```python
# src/defectvad/utils/config.py

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
```

---

# 4. 프로젝트 bootstrap (핵심)

`src/defectvad/bootstrap.py`

```python
# src/defectvad/bootstrap.py

import os
from defectvad.utils.config import load_yaml, resolve_paths_cfg, validate_paths
from defectvad.components.backbone import set_backbone_dir


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
```

---

# 5. backbone.py는 그대로 유지 (이미 매우 잘 구현됨)

```python
# models/components/backbone.py
BACKBONE_DIR = os.getenv('BACKBONE_DIR', '/mnt/d/deep_learning/backbones')
```

이 부분은 **절대 수정하지 않습니다**.
경로 주입은 bootstrap에서만 합니다.

---

# 6. 실험 스크립트 적용 예시

`experiments/train_cflow_mvtec.py`

```python
# experiments/train_cflow_mvtec.py

from defectvad.bootstrap import init_from_paths_yaml

# ★ 가장 중요: 모델 import 이전에 실행
paths = init_from_paths_yaml("configs/paths.yaml")

print(">> Paths loaded:")
for k, v in paths.items():
    print(f"   {k}: {v}")

# 이제 안전하게 모델 생성 가능
from defectvad.models.cflow.trainer import CflowTrainer
from defectvad.datasets.mvtec import MVTecDataset
from torch.utils.data import DataLoader

train_dataset = MVTecDataset(
    root=paths["dataset_dir"],
    split="train",
)
test_dataset = MVTecDataset(
    root=paths["dataset_dir"],
    split="test",
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

trainer = CflowTrainer(
    backbone="wide_resnet50_2",
)

trainer.fit(train_loader, max_epochs=200, valid_loader=test_loader)
```

---

# 7. 환경별 실행 방법

### 7.1 profile 변경 (권장)

```bash
export VAD_PROFILE=linux_server
python experiments/train_cflow_mvtec.py
```

---

### 7.2 backbone_dir만 즉시 override

```bash
export BACKBONE_DIR=/fast_ssd/backbones
python experiments/train_cflow_mvtec.py
```

> **profile + 환경 변수 override** 조합은 실무에서 가장 많이 쓰입니다.

---

# 8. 이 구조의 장점 (정리)

* 코드에 **절대 경로 하드코딩 없음**
* 모델 코드 수정 없이 환경 이동 가능
* WSL / Windows / 서버 / Docker 모두 대응
* backbone / dataset / output 경로 일관 관리
* anomalib / Lightning 스타일과 구조적으로 동일

---

# 9. 최종 한 줄 정리

> **`configs/paths.yaml`을 단일 진실 소스로 두고,
> 실행 entry-point에서 `bootstrap.init_from_paths_yaml()`로 경로를 주입하는 구조가
> 가장 안전하고, 확장 가능하며, 실무에서 가장 많이 쓰이는 방식이다.**

다음 단계로 원하시면 바로 이어서 도와드릴 수 있습니다.

* backbone **자동 다운로드 + checksum 검증**
* offline 서버용 backbone mirror 설계
* Docker + volume mount 구조
* anomalib 내부 path 관리와 1:1 비교
