---
type: log
title: 작업 로그
---

## 2026-10-21 (UTC)

- 프로젝트 시작. 레포 골격(`scratchlm/`, `tests/`, `scripts/`, `wiki/`) 생성, 브랜치 `dev`.
- 0단계: PGX(GB10) 확인, NGC `pytorch:26.09-py3` 컨테이너 채택, `scripts/pgx.sh` 작성.
- 1단계: byte-level BPE 토크나이저 구현 + 테스트(lossless pre-tokenize, roundtrip, specials, save/load).
- 환경 검증: torch 2.14.0a0 (nv26.09), CUDA 13.4(forward-compat on driver 580), GB10 sm_121, bf16 matmul ~68 TFLOPS.
- TinyStories train(2.1GB)/valid(22MB) 다운로드 (PGX `data/`).
- 8k tokenizer 학습: train 앞 50MB, ~5초. valid 압축률 **4.10 bytes/token**.
  한국어 문장 50 bytes → **50 tokens** (merge 없음 = 바이트 그대로) — 7단계 한국어 확장의 동기.
- 결정: 숫자 한 자리씩 토큰화(`\d`)로 변경 후 tok8k 재학습. 압축률 변화 없음(4.10), `3051` → `3|0|5|1`.
- 의존성 정책 확정: 외부 의존성 `torch` 하나(저수준만). `numpy` 선언 제거, 테스트 pytest → stdlib unittest. 로드맵에 1.5단계(stdlib autograd) 추가.
