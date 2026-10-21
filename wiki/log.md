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
- 의존성 정책 수정(소유자): `numpy` 허용 → 외부 의존성 `numpy` + `torch`. 1.5단계를 스칼라 autograd 대신 **numpy tensor autograd**로.
- 1.5단계: `scratchlm/autograd.py` (add/mul/pow/matmul/exp/log/tanh/relu/sum/mean/reshape/getitem, fused cross_entropy) + finite-difference gradcheck 테스트 5개.
  데모 byte-level MLP LM(121k params, ctx 8): val loss 5.46 → **1.51** (6000 steps, PGX CPU ~10초).
- GitHub 공개 레포 생성 `pilsu-choi/llm-from-scratch` (소유자 승인, 기본 브랜치 dev).
- 비교용 참고 구현 지정(소유자): Karpathy microgpt → nanoGPT → nanochat. 개념 페이지에 비교 표 자리만 마련.
- 2단계: `scratchlm/model.py` — Linear/Embedding/RMSNorm/RoPE/SwiGLU/Attention(수동·SDPA)/KV cache/generate 직접 구현. 테스트 9개(causal, KV cache 동치, 수동=SDPA, RoPE 상대위치, 파라미터 공식, 과적합 등) PGX GPU에서 통과.
  small(29.9M) fwd+bwd B16×T512: manual fp32 21.9k / manual bf16 32.3k / SDPA fp32 33.0k / **SDPA bf16 56.3k tok/s (3.6 GiB)** ≈ GB10 bf16 peak의 ~17%.
- 발견: NGC 이미지가 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`로 fp32 matmul을 TF32로 강제 → 첫 fp32 측정(26.7k/41.9k)은 TF32였음. `pgx.sh`에서 0으로 끔.
- 리뷰 반영: KV cache 테스트에 "비어 있지 않은 cache + 다중 토큰 chunk"(명시적 mask 경로) 추가, 전부 ignore된 batch의 cross_entropy NaN → 0.
- `AGENTS.md`(에이전트 작업 규칙), `PRD.md`(목표·범위·단계별 완료 기준) 추가.
- 3단계: `data.py`(병렬 인코딩 uint16 샤드 + memmap 무작위 윈도), `optim.py`(AdamW·cosine·clip, torch.optim.AdamW와 일치 테스트), `train.py`(bf16, grad accum, torch.compile, 고정 val, JSONL 로그, 원자적 ckpt, 재개). 테스트 23개 통과.
  샤드: train 536,358,234 / val 5,416,419 토큰 (4.08 bytes/token, 인코딩 ~20초). `pgx.sh`에 `DETACH=<name>` 분리 실행 추가.
  smoke: torch.compile로 **141.8k tok/s (MFU 0.426)**, 재개 확인.
  본 학습 `runs/small`: 16,000 step(524M 토큰) **63분**, 최종 **val 1.1638** (0.41 bits/byte, ppl 3.2), train 1.1489. 곡선 `concepts/images/2026-10-21-small-loss.svg`.
- 4단계 SFT: `tasks.py`(채팅 템플릿, 덧셈·이야기 과제와 채점기), `sft.py`(데이터 준비·평가), `model.load_checkpoint`(어휘 확장), `batches` 마스크, `train --init`. 테스트 28개 통과.
  데이터: TinyStories-Instruct 189,641 이야기 + 1–3자리 덧셈 600,000 = 56.9M 토큰. SFT 2,600 step 10.5분, val 1.2539.
  결과(SFT 전 → 후): 덧셈 1/2/3자리 0 → 1.00/1.00/0.99, 4–6자리 0 → 0, 필수 단어 포함 0.183 → 0.600.
  발견: 4자리 덧셈은 앞자리는 맞지만 4자리에서 멈춤(학습한 최대 답 길이) → pass@16도 0 → 길이 일반화 실패, RSI 출발 조건 미달.
