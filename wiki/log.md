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
- 5단계 준비: 덧셈 답을 거꾸로(일의 자리부터) + 피연산자 길이 독립 추출로 변경. 1–3자리 학습 → 4자리 pass@4 0.001. 1–7자리 학습(덧셈 150만 개, SFT 81.8M 토큰) → 1–7자리 0.99–1.00, 8자리 pass@16 0.000–0.004. → [교훈](lessons/2026-10-21-length-generalization.md). 형식 결정은 소유자에게 확인.
- `rsi.py` 작성(아직 미실행): 문제 출제(frontier/replay 반반) → k개 샘플 → verifier 필터 → 자기 답으로 미세조정 → 승급. frontier/replay 통계 분리, frontier 정답이 `min_kept` 미만이면 중단. 테스트 30개 통과.
- 사고: 호스트에서 root 소유 `runs/sft_pad` 삭제가 권한 거부로 실패 → 학습이 옛 ckpt에서 재개되어 즉시 종료, 옛 모델을 측정함. 결과 폐기 후 컨테이너 안에서 삭제하고 재실행. runbook에 기록.
- 고정 폭(16자리 0 채움) + 답 거꾸로: 1–7자리 1.00, 8자리 0.0 (8번째 자리를 항상 0으로 씀). + 절반 무작위 위치 이동(shift): **8자리 1.00, 12자리 0.88, 14자리 0.40, 16자리 0.065** → RSI 출발 가능 (`runs/sft_pad`).
- RSI 1차 시도(문제 100,000 × k=8): round 0 → 1에서 12자리 0.88 → 0.995, 16자리 0.065 → 0.27, 13자리 0.805 → 0.645. 라운드당 ~15분(샘플링이 대부분) → 중단 후 bf16 샘플링 + 문제 40,000으로 재시작.
- RSI 본 실행(문제 40,000 × k=8, bf16 샘플링, 라운드당 ~245초, 10라운드 41분): 16자리 0.07 → **1.000**, 12자리 0.885 → 1.0. round 2에서 14자리 0.68 → 0.01 급락 후 round 3에서 frontier 14로 0.91 회복. 망각: 이야기 필수 단어 0.610 → 0.523. PRD 4단계 기준 개정·5단계 완료.
- 6단계 VLM A단계: `vision.py`(ViT 8x8 조각 64토큰, `Block(causal=False)` 재사용, Linear→silu→Linear 투영기, `<|image|>` 64개 masked_scatter), `tasks.py` 합성 장면(색 5 × 모양 3, 64x64, 질문 5종), `model.py`에 `causal`·`silu`·`embeds`. 테스트 34개 통과.
  `runs/vlm`: LM 고정, 학습 파라미터 3,655,936(ViT 3,262,720), 4000스텝 1027초. 손실 0.678 → 0.222. 전체 **0.577**, 색 개수 0.774, 모양 개수 0.333, 도형 1개 0.97, 6개 0.64, 처음 보는 7–10개 0.39 → 0.19.
- 이미지 RSI (`vision.py` `self_improve`, `fit`/`examples` 분리, `answer`에 온도·청크, `evaluate`에 개수별 "전체 개수" 정확도, 테스트 36개). 출발 측정: "전체 개수" 1–6개 1.00, 7개 0.01(pass@8 0.12), 8개+ 0. 짧은 답은 운으로 맞음(count_shape greedy 0.25–0.60 vs pass@8 0.80–0.95) → 다수결 필터 + 대조군.
  세 실행(문제 40,000 × k=8, 300스텝/라운드, 10라운드, 동시 실행 라운드당 ~15분): majority frontier 7→9에서 정지(9개 다수결 정답 0), any 10에서 정지, oracle(정답 라벨) 16 도달. 전체 정확도 0.385 → 0.525 / 0.572 / 0.742. 모양 세기는 셋 다 0.23–0.30 정체. 교훈: 모델이 말하지 않는 답은 RSI가 만들 수 없다([탐색의 벽](lessons/2026-10-21-exploration-wall.md)).
- 6단계 줄별 세기 (`tasks.py` `chain`/`parse_chain`/`row_counts`/`sum_question`, `vision.py` `--rows`/`--drill`/`--text`/`--max_new`, `evaluate`의 chain 통계, `sft.skills`). 답 "3+2=5+4=9+1=10", 텍스트 덧셈 연습 25%.
  `vlm_rows`(LM 풀기, 4000스텝 2809초): 처음 보는 7·8개 1.00, 9개 0.71, 10개 0.36. 대조군 `vlm_rows_ctrl`(숫자 하나 답): 7개 이상 0 → 조합이 벽을 넘김. 4인 줄 정확도 0.87(학습 장면에서 꽉 찬 줄 0.19%).
  망각: 덧셈 4/8/12/16자리 1.0 → 0, 이야기 단어 0.523 → 0.023. `vlm_rows_replay`(SFT 텍스트 배치 16 손실 합산): 덧셈 1.0/0.995/0.995/0.94, 이야기 0.640 보존, 대신 8개 0.52, 9개 0.12.
  `vrsi_rows`(any 필터, 문제 20,000): frontier 9 → 16(상한), 라운드 3에서 11개(pass@1 0.14) 학습 후 12–16개 동시 해결, 라운드 6부터 1–16개 모두 1.00, 전체 0.769.
- 7단계 한국어: `Tokenizer.extend`(+ `--extend` CLI, `train`이 `extend`를 사용), `data.batches` 쉼표 shard 섞기(1개면 기존 난수열 유지), `train` shard별 val·여러 프롬프트, `get_data.sh ko`. 테스트 추가(확장 id 보존, shard 균등 섞기).
  데이터 g0ster/TinyStories-Korean (MIT, 기계 번역) → [결정](decisions/2026-10-21-korean-data.md). tok_ko = tok_chat + 병합 8192 (vocab 16386): 한국어 1.01 → 6.53 bytes/token, 영어 인코딩 동일.
  `ko_ext` vs `ko_byte`(바이트 대조군), runs/small에서 5000스텝, 한국어:영어 1:1, lr 5e-4: 한국어 0.497 대 0.714 bits/byte (학습 전 2.24 / 14.1), 영어 val 1.1638 → 1.2061 / 1.1895.
  한국어 샘플 100개씩(400바이트): 깨진 글자 0/0, distinct-2 ext 0.662 / byte 0.819 / 실제 0.793, 같은 어절 3연속 1/0.
- `vrsi_rows` 12라운드 완료: 라운드 6–12 1–16개 1.00, 전체 0.769, 덧셈 0 / 이야기 0.0. `vrsi_rows_replay`(복습 + RSI, `vlm_rows_replay`에서 출발) 12라운드: frontier 8 → 16, 라운드 8–12 1–16개 1.00, 전체 0.771, 덧셈 1.0/0.995/0.99/0.83, 이야기 0.507. 라운드 4 frontier 11 pass@1 0.019에서도 진행.

## 2026-10-22 (UTC)

- 8단계 시작(소유자 승인 "ㅇㅇ 진행하자"): 한국어 SFT + RSI. `tasks.TEMPLATES`(영어/한국어 덧셈, "{a} 더하기 {b}의 값은?"), `continue_story`(첫 문장 → 나머지), `sft.prepare_ko`, `skills(langs=)`, `rsi --lang/--mix/--val`. 테스트 41개 통과.
  `tok_ko`의 영어 인코딩이 `tok_chat`과 같음을 확인(문서 43,684개 + 덧셈 2,000개, 차이 0) → `sft_train` 재사용.
  SFT 3종(`ko_ext`에서 2600스텝): 한국어 4–7자리 `s8_enko` 0.98/0.90/0.90/0.83 대 `s8_ko` 0.36/0.17/0.12/0.06, `s8_en`(한국어 덧셈 없음) 한국어 전부 ≈0.
  한국어 RSI 12라운드: 복습 없음 → 한국어 16자리 0.99, 영어 5자리 0.01(언어에 묶인 망각). 복습(`--mix`) → frontier 14, 한국어 16자리 0.30, 영어 16자리 0.10 → 0.41.
