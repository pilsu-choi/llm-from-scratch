---
type: decision
title: 핵심 설계 결정
description: 프로젝트 시작 시점의 범위·환경·RSI 해석 결정
tags: [decision]
timestamp: 2026-10-21T00:00:00Z
---

- **RSI = Recursive Self-Improvement**, verifier 기반 STaR/ReST-EM. 소형 모델 self-judge는 배제 (소유자 선택 + 소형 모델 한계).
- **연산 자원: PGX**(GB10). 로컬 RTX 4070은 단위 테스트·디버깅용.
- **진행: 단계별 학습형** — 한 단계씩 코드+원리 페이지, 소유자 확인 후 다음 단계.
- **언어: 영어(TinyStories) → 한국어 확장.** 토크나이저를 byte-level로 만들어 한국어 확장 시 OOV가 없게 함.
- **숫자는 한 자리씩 토큰화** (`\d` chunk, 앞 공백 없음): RSI 산술 과제에서 자릿수별 일관된 표현. 토크나이저는 pretrain~RSI 동안 고정되므로 3단계 전에 확정.
- **의존성 최소화** (소유자 요청, 2026-10-21; numpy 허용은 같은 날 소유자 결정): 외부 의존성은 **`numpy` + `torch`**.
  - `numpy`: 1.5단계 tensor autograd, 토큰 샤드(`np.memmap`), 데이터 전처리. pure Python 대비 배열 연산이 C/BLAS로 빠름.
  - `torch`: 2단계부터 GB10 GPU 학습용 (numpy는 CPU 전용이라 학습이 ~50배 느림). 텐서 연산·autograd·`nn.Module`/`nn.Parameter`만 사용,
    `F.scaled_dot_product_attention`은 직접 구현과 비교 후 사용. `nn.Linear`/`nn.Embedding`/`nn.LayerNorm`/`nn.Transformer*`, `torch.optim`은 쓰지 않고 직접 구현.
  - 토크나이저·RSI verifier·합성 이미지: stdlib/numpy만 (`regex`, `PIL`, `datasets` 미사용; 다운로드는 `curl`). 테스트는 stdlib `unittest`.
  - 스칼라 autograd 대신 numpy **tensor** autograd로 역전파를 직접 구현 (matmul·broadcasting·softmax backward까지, torch와 같은 구조).
