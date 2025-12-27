# 1. `models/components/backbone.py` 위치가 적절한가?

## 1.1 현재 상태의 의미

현재 구조:

```
src/defectvad/
 └─ models/
    └─ components/
       └─ backbone.py
```

이 구조는 암묵적으로 다음을 의미합니다.

> “backbone은 모델 구성 요소(component)의 일부이며
> 모델 구현과 강하게 결합되어 있다”

### 장점

* STFPM / EfficientAD / CFlow 등 **모델 코드에서 접근하기 편함**
* anomalib / timm 내부 구조와 유사
* 소규모 프로젝트에서는 충분히 합리적

### 단점 (중요)

* `backbone.py`의 책임이 **모델 컴포넌트 범위를 초과**
* 실제 역할:

  * ❌ 모델 로직
  * ✅ **리소스 경로 관리**
  * ✅ **환경 의존 설정**
* 즉, **“도메인(model)”이 아니라 “인프라(infra)” 성격**

👉 프로젝트가 커질수록 위치가 어색해짐

---

## 1.2 Backbone은 “모델”이 아니라 “외부 리소스”

`backbone.py`가 다루는 것은:

* weight 파일 위치
* 파일 존재 여부
* safetensors / pth 구분
* 환경 변수

이는 **모델의 수학적 정의나 구조와 무관**합니다.

즉:

> backbone weight path management = **Infrastructure / Resource layer**

---

# 2. 함수들을 한 파일에 두는 것이 적절한가?

현재 `backbone.py`의 역할은 세 가지입니다.

| 함수                    | 역할                          |
| --------------------- | --------------------------- |
| `get_backbone_dir()`  | 전역 상태 조회                    |
| `set_backbone_dir()`  | 전역 상태 변경                    |
| `get_backbone_path()` | backbone name → 실제 파일 경로 해석 |

### 문제점

* **설정 관리(set/get)** 와
* **도메인 로직(backbone name → 파일 구조)** 가 섞여 있음

이는 **SRP(Single Responsibility Principle)** 위반에 가깝습니다.

---

# 3. 아키텍처적으로 가장 깔끔한 분리 기준

다음 3가지를 분리하는 것이 이상적입니다.

```
[1] 설정 소스 (어디에 저장할지)
[2] 전역 상태 관리 (현재 선택된 backbone_dir)
[3] backbone 규칙 (이름 → 파일 구조)
```

---

# 4. 추천 구조 (실무 기준, 점진적 개선)

## 4.1 최종적으로 가장 많이 쓰이는 구조 (권장)

```
src/defectvad/
 ├─ bootstrap.py
 ├─ config/
 │   ├─ paths.py            # 전역 경로 상태 관리
 │   └─ __init__.py
 ├─ resources/
 │   ├─ backbone.py         # backbone name → weight path 규칙
 │   └─ __init__.py
 ├─ models/
 │   └─ ...
```

---

## 4.2 역할 분담 (핵심)

### (A) `config/paths.py` – 전역 경로 상태 관리

```python
# src/defectvad/config/paths.py

import os

_BACKBONE_DIR = os.getenv("BACKBONE_DIR")

def set_backbone_dir(path: str):
    global _BACKBONE_DIR
    _BACKBONE_DIR = path

def get_backbone_dir() -> str:
    if _BACKBONE_DIR is None:
        raise RuntimeError("BACKBONE_DIR is not initialized")
    return _BACKBONE_DIR
```

➡️ **환경 / 설정 책임만 가짐**

---

### (B) `resources/backbone.py` – backbone 규칙만 담당

```python
# src/defectvad/resources/backbone.py

import os
from defectvad.config.paths import get_backbone_dir

BACKBONE_WEIGHT_FILES = {
    ...
}

def get_backbone_path(backbone: str) -> str:
    backbone_dir = get_backbone_dir()
    ...
    return os.path.join(backbone_dir, filename)
```

➡️ **순수 규칙 + 해석 로직**

---

### (C) `bootstrap.py` – 연결만 담당

```python
# src/defectvad/bootstrap.py

from defectvad.config.paths import set_backbone_dir

def init_from_paths_yaml(...):
    set_backbone_dir(backbone_dir)
```

---

# 5. 그럼 지금 당장 옮겨야 하나?

## 5.1 결론

> **지금 당장 옮길 필요는 없다.
> 하지만 지금 구조는 “중간 단계”로 보는 것이 정확하다.**

### 지금 구조가 허용되는 경우

* 개인 연구
* 단일 레포
* 모델 수 5~10개 수준

### 분리를 권장하는 시점

* 모델 외 리소스 증가 (dataset cache, checkpoints, tokenizer 등)
* CLI / API / 서비스화
* 팀 협업 본격화

---

# 6. 현실적인 타협안 (추천)

**파일은 그대로 두되, 역할을 명확히 하는 방법**

```
models/components/backbone.py
```

⬇️ **주석으로 의도 명확화**

```python
"""
Backbone weight resource resolver.

NOTE:
- This module manages *external resources*, not model definitions.
- It may be relocated to defectvad/resources in future refactoring.
"""
```

그리고:

* `set_backbone_dir()` 호출은 **bootstrap에서만**
* 모델 코드에서는 `get_backbone_path()`만 사용

---

# 7. 판단 요약 (의사결정용)

| 질문                               | 답                           |
| -------------------------------- | --------------------------- |
| components 밑에 backbone.py가 적절한가? | **동작은 OK, 아키텍처적으로는 아님**     |
| 함수들이 한 파일에 있어야 하는가?              | **아니오, 역할 분리 여지 큼**         |
| 지금 당장 리팩토링해야 하나?                 | **아니오 (단계적 개선 권장)**         |
| 가장 이상적인 위치는?                     | `config/` + `resources/` 분리 |

---

# 8. 한 줄 결론

> **`backbone.py`는 모델 컴포넌트라기보다 “외부 리소스 관리 코드”이므로,
> 장기적으로는 `config`(상태)와 `resources`(규칙)로 분리하는 것이 가장 깨끗한 설계다.
> 현재 위치는 과도기적 구조로는 충분히 합리적이다.**

원하시면 다음 단계로
**“현재 코드를 최소 수정으로 분리하는 실제 리팩토링 diff”**
또는
**anomalib / Lightning 소스와의 구조 비교**
까지 바로 이어서 진행할 수 있습니다.
