---
type: decision
title: 7단계 한국어 데이터와 토크나이저 방식
description: 한국어 코퍼스로 기계 번역 TinyStories(MIT)를 쓰고, 토크나이저는 재학습 대신 기존 병합 위에 확장하기로 한 결정
tags: [decision, korean, dataset, tokenizer]
timestamp: 2026-10-21T00:00:00Z
---

- **데이터: [g0ster/TinyStories-Korean](https://huggingface.co/datasets/g0ster/TinyStories-Korean)** (MIT 라이선스). 영어 TinyStoriesV2를 EEVE 모델로 기계 번역한 것. train 3.1GB, valid 31MB.
  - 이유: 영어 단계와 **같은 이야기 분포**라서, 차이가 언어와 토크나이저에서만 온다. 30M 모델이 배울 수 있는 쉬운 문장이다. 라이선스가 명확하다. `curl`로 받을 수 있다 (`scripts/get_data.sh ko`).
  - 한계: 번역체(영어 어순, 음역 이름). 위키·뉴스 같은 자연 한국어는 이 단계의 목표(원리)를 벗어나 다루지 않는다.
  - 사전학습 가중치는 쓰지 않는다. 번역에 쓰인 모델은 데이터를 만든 도구일 뿐, 우리 모델에 들어가지 않는다.
- **토크나이저: 확장 (재학습 아님)**. 기존 8192 병합과 id를 그대로 두고 한국어 병합 8192개를 덧붙였다 (vocab 16386). 3단계 모델 가중치를 버리지 않고 이어서 학습하기 위해서다. 대조군으로 확장 없이(바이트) 같은 학습을 돌렸다. 결과는 [한국어 확장](../concepts/2026-10-21-korean.md).
- **자율 진행**: 소유자가 2026-10-21 밤에 "7단계 진행할 시간 오면 바로 진행해도 좋음"이라고 해서, 단계 전환 확인 없이 시작했다.
