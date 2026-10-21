# PRD — llm-from-scratch

## 1. 목적

LLM과 VLM이 **어떻게 동작하고 학습되는지**를 직접 만들면서 이해한다. 마지막에는 모델이 스스로 만든 데이터로
더 나아지는 **RSI(Recursive Self-Improvement) 루프**를 작은 규모에서 재현하고 관찰한다.

- 사용자: 프로젝트 소유자 한 명 (학습자). 산출물은 코드 + 한국어 원리 문서 + 측정 결과.
- 성공의 정의: 각 단계의 원리를 **코드와 측정으로 설명할 수 있다.** 모델 품질 자체가 목표는 아니다.
- 비교 기준: 나중에 Karpathy의 microgpt → nanoGPT → nanochat과 설계·결과를 나란히 비교한다.

## 2. 제약

| 항목 | 내용 |
|---|---|
| 연산 | PGX: NVIDIA GB10 (Blackwell, aarch64), 통합메모리 ~119GB 중 ~70% 사용 가능. bf16 matmul ~68 TFLOPS 측정 |
| 의존성 | `numpy` + `torch`(저수준만) + stdlib. 사전학습 가중치·HF 라이브러리 금지 ([AGENTS.md](AGENTS.md) §3) |
| 데이터 | 영어 TinyStories (2.1GB ≈ 5억 토큰) → 이후 한국어 코퍼스 |
| 진행 | 한 단계씩. 각 단계 완료 후 소유자 확인 |

## 3. 범위

포함:
- byte-level BPE 토크나이저, numpy autograd, decoder-only GPT, 사전학습, SFT, RSI, ViT 기반 VLM, 한국어 확장
- 단계별 원리 문서(wiki/concepts)와 재현 가능한 측정

제외 (필요해지면 별도 결정):
- 대규모 학습(>~150M params), 분산 학습, 서빙/배포, 웹 UI
- RLHF·보상모델 학습 (RSI는 프로그램 verifier로만)

## 4. 단계와 완료 기준

| # | 단계 | 완료 기준 (Done when) | 상태 |
|---|---|---|---|
| 0 | 환경 | 컨테이너에서 GPU bf16 matmul 동작, `pgx.sh`로 원격 실행 | ✅ |
| 1 | Tokenizer | roundtrip 무손실(영어·한국어·이모지), 숫자 한 자리 토큰, TinyStories 압축률 측정 (4.10 bytes/token) | ✅ |
| 1.5 | numpy autograd | 모든 연산 gradcheck 통과, byte-level MLP LM이 val loss를 ln 256에서 의미 있게 낮춤 (5.46 → 1.51) | ✅ |
| 2 | GPT 모델 | 수동 attention = SDPA, KV cache = full forward, causal, RoPE 상대위치, 파라미터 공식 테스트 통과, 처리량 측정 (56.3k tok/s) | ✅ |
| 3 | 사전학습 | 직접 만든 AdamW + warmup/cosine + bf16 + grad accum으로 small(29.9M) 학습. val loss 곡선과 샘플을 기록하고, 문법적으로 맞는 TinyStories 문장을 생성. checkpoint 저장/재개 | ⏳ |
| 4 | SFT | chat 템플릿 + prompt loss masking. 지시(질문→답) 형식을 따르는 비율을 측정. 산술 과제의 iteration 0 정확도 > 0. **개정(2026-10-21, 소유자 결정)**: 덧셈은 16자리 고정 폭 + 답 거꾸로 + shift, 학습 1–7자리, frontier는 '정확도 0.9 미만인 첫 자릿수' | ✅ (1–8자리 1.00, 12자리 0.88, 16자리 0.065; 단어 18→61%) |
| 5 | RSI | 생성 → verifier 필터 → 재학습을 N회 반복. held-out 정확도가 iteration마다 오르는지, 다양성(distinct-n, pass@k)이 붕괴하는지 기록. 커리큘럼(자릿수 증가) 효과 측정 | ✅ (10라운드 41분: 16자리 0.07 → 1.00, round 2 급락·회복 관찰, 이야기 0.61 → 0.52 망각) |
| 6 | VLM | 직접 만든 ViT + projector를 LM에 연결. 합성 이미지(도형·개수·색) QA 정확도 측정. 같은 RSI 루프를 이미지 과제에 적용 | ✅ (A단계: 전체 0.577, 처음 보는 7–10개 0.39→0.19. 이미지 RSI: "전체 개수" frontier 7→9(다수결)/10(any)에서 탐색의 벽으로 정지, 정답 라벨 대조군은 16까지. 줄별 누적합 답 + RSI로 1–16개 모두 1.00, 텍스트 복습으로 망각 방지) |
| 7 | 한국어 | 토크나이저 확장(vocab surgery) 또는 재학습으로 한국어 bytes/token 개선 (현재 1.0) + continued pretraining 후 한국어 샘플 품질 비교 | ✅ (확장 1.01 → 6.5 bytes/token, 5000스텝 한국어 0.497 bits/byte 대 바이트 대조군 0.714, 영어 val 1.164 → 1.206) |

## 5. RSI 설계 (핵심 요구사항)

- **판정은 코드가 한다.** 정답을 프로그램으로 검증할 수 있는 과제만 쓴다 (여러 자리 덧셈·곱셈, 작은 논리/프로그램 퍼즐, 합성 이미지 QA).
- **시작 조건**: iteration 0 정확도가 0이면 루프가 출발하지 못한다. 해당 과제 데이터를 사전학습/SFT에 소량 섞는다.
- **관찰 지표**: held-out pass@1, pass@k, 출력 다양성, 난이도별 정확도, 학습 데이터에 남은 정답 비율.
- **위험 신호**: 다양성 붕괴(mode collapse), verifier 허점 악용(reward hacking), 쉬운 문제로의 편향. 각 iteration을 기록해서 재현한다.

## 6. 측정·기록

- 모든 학습은 `runs/<name>/` (PGX)에 설정, loss 로그(JSONL), checkpoint, 샘플을 남긴다.
- wiki에는 측정한 숫자만 쓰고, 추정치는 추정이라고 표시한다.
- 단계마다 concept 페이지에 "참고 구현 비교" 표를 두고 나중에 채운다.

## 7. 위험

| 위험 | 대응 |
|---|---|
| 작은 모델이 산술을 못 배워 RSI가 시작되지 않음 | 과제 데이터를 사전학습에 혼합, 쉬운 난이도부터 커리큘럼 |
| PGX 메모리를 vLLM과 공유 | 학습 메모리를 측정한 뒤 배치 크기 결정. vLLM 중지는 소유자에게 확인 |
| 직접 구현한 연산의 숨은 버그 | gradcheck와 동치 테스트를 먼저 작성 |
| 한국어 데이터 확보 | 7단계 시작 시 라이선스를 확인한 코퍼스로 결정 |
