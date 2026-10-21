---
type: concept
title: Byte-level BPE 토크나이저
description: scratchlm/tokenizer.py에 구현된 byte-level BPE의 원리, 학습/인코딩/디코딩 알고리즘, 한국어 비효율 문제와 실습
tags: [tokenizer, bpe, byte-level, korean, scratchlm]
timestamp: 2026-10-21T00:00:00Z
resource: scratchlm/tokenizer.py
---

# Byte-level BPE 토크나이저

전체 학습 계획은 [로드맵](../2026-10-21-roadmap.md) 참고. 이 페이지는 `scratchlm/tokenizer.py`(stdlib만 사용, 약 130줄)를 원리 중심으로 설명한다.

파이프라인: `text -> 정규식 pre-tokenize(chunk) -> UTF-8 bytes -> merge 적용 -> ids`

## 1. 왜 토크나이저가 필요한가

모델은 정수 id의 시퀀스만 본다. 문자열을 어떤 단위로 자를지가 핵심 설계 선택이다.

| 단위 | vocab 크기 | 시퀀스 길이 | 문제 |
|---|---|---|---|
| 문자(char) | 작음 (수천~십만, 유니코드 전체) | 매우 김 | attention 비용 O(T^2), 문맥 창을 낭비 |
| 단어(word) | 매우 큼 (수십만+) | 짧음 | OOV(`<unk>`), 어형 변화마다 별도 항목, embedding 표 거대 |
| 서브워드(BPE) | 중간 (8k~100k) | 중간 | 자주 나오는 조각은 한 토큰, 드문 단어는 조각으로 분해 |

트레이드오프는 두 비용의 균형이다.

- vocab이 커지면: embedding 표(`V x d`)와 출력 층(`d x V`, weight tying 안 하면 2배)이 커지고, 드문 토큰은 학습 신호가 부족하다.
- vocab이 작으면: 같은 텍스트가 더 긴 시퀀스가 되어 학습/추론 연산량이 늘고, 같은 context length에 담기는 정보가 줄어든다.

예: d=512, V=8192면 embedding만 약 4.2M 파라미터. 작은 모델(수십M)에서는 이 비중이 크므로 TinyStories 같은 좁은 도메인에서는 8k 정도가 합리적이다.

## 2. Byte-level의 의미

기본 vocab은 **0~255 바이트 256개**(`self.vocab = {i: bytes([i]) ...}`)다. 어떤 문자열이든 UTF-8로 인코딩하면 바이트열이 되므로 표현 불가능한 입력이 없다. 즉 **`<unk>`이 필요 없다** (이모지, 한글, 깨진 문자 모두 가능). merge로 생기는 새 토큰(256~)은 "바이트열을 이어붙인 것"일 뿐이다.

### 한국어 문제

UTF-8에서 ASCII 영문 1글자 = 1 byte, **한글 1글자 = 3 bytes**다.

- 영어 위주 코퍼스(TinyStories)로 학습한 BPE에는 한글 바이트 조합 merge가 거의 없다.
- 그래서 한글 1글자가 거의 그대로 **3 토큰**이 된다 (`"안녕"` -> 6 바이트 -> 6 토큰). 영어는 단어 하나가 1~2 토큰인 것과 큰 차이다.
- 결과: 같은 의미를 표현하는 데 3~5배 많은 토큰, 즉 context 낭비 + 느린 학습/추론. 또 merge가 한글 바이트의 중간(3바이트 중 일부)에서 끝나 토큰 하나를 단독 디코딩하면 깨진 문자(`�`)가 나오기도 한다.

해결은 이후 **한국어 확장 단계**에서 다룬다: 한국어 코퍼스로 tokenizer를 재학습하거나, 기존 merge를 유지한 채 vocab을 확장(추가 merge 학습 후 embedding 행 추가)한다. 이 페이지의 학습 코드가 그대로 재사용된다.

## 3. Pre-tokenization 정규식 (PAT)

```python
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")
```

GPT-2 스타일 분할. 각 alternative:

| 패턴 | 매칭 대상 |
|---|---|
| `'(?:[sdmt]\|ll\|ve\|re)` | 축약형 `'s 'd 'm 't 'll 've 're` |
| ` ?[^\W\d_]+` | (선택적 앞 공백) + 유니코드 **글자** 연속. `\w`에서 숫자와 `_`를 뺀 것 |
| ` ?\d{1,3}` | (공백) + 숫자 1~3자리 묶음 |
| ` ?(?:[^\s\w]\|_)+` | (공백) + 구두점/기호/이모지/`_` 연속 |
| `\s+(?!\S)` | 뒤에 비공백이 오지 않는 공백 (마지막 공백 하나는 다음 단어에 붙도록 남김) |
| `\s+` | 나머지 공백 |

왜 필요한가:

- **merge가 chunk 경계를 넘지 않는다.** 각 chunk를 따로 바이트열로 만들어 merge하므로 `dog.`의 `g`와 `.`이 합쳐지는 일이 없다. 의미 단위가 깨끗하게 유지되고 `the`/` the`처럼 공백 유무만 다른 변형이 분리된다.
- **효율**: 학습 시 chunk 단위로 중복을 제거할 수 있다 (4절).
- **반드시 lossless**여야 한다. `"".join(PAT.findall(t)) == t`가 아니면 일부 문자가 어떤 alternative에도 안 걸려 사라지고, decode(encode(x)) != x가 된다. `tests/test_tokenizer.py::test_pretokenize_is_lossless`가 영어/한글/숫자/이모지/탭/`\r\n`/`_`로 이를 검증한다. (마지막 `\s+`가 안전망 역할.)

주의: GPT-2 원본은 서드파티 `regex` 패키지의 `\p{L}`, `\p{N}`을 쓴다. 이 프로젝트는 stdlib `re`만 쓰려고 `[^\W\d_]`(= 유니코드 글자)로 대체했다. `²`처럼 `\w`이지만 `\d`(10진수)는 아닌 문자는 글자 클래스 `[^\W\d_]` 쪽으로 분류되는 등 사소한 차이가 있지만 lossless는 유지된다.

## 4. 학습 알고리즘

BPE 아이디어: **가장 자주 인접하는 토큰 쌍을 새 토큰 하나로 합치기를 vocab이 찰 때까지 반복**한다.

### 코드 흐름 (`Tokenizer.train`)

1. `counts = Counter(chunk ...)`: 전체 텍스트를 PAT으로 나눈 **고유 chunk -> 빈도**. 코퍼스 수십 MB도 고유 chunk는 수만 개 수준이라 훨씬 작다.
2. `words`: 각 고유 chunk를 바이트 id 리스트로. `freqs`: 그 빈도.
3. `stats[pair]`: 인접 쌍의 **빈도 가중 합계**. `where[pair]`: 그 쌍을 포함하는 word 인덱스 집합.
4. 반복 (`new = 256, 257, ...`, 단 specials 자리를 빼고 `vocab_size - len(specials)`까지):
   - `pair = max(stats, key=stats.get)`: 최빈 쌍 선택, `merges[pair] = new` 기록.
   - `where.pop(pair)`로 그 쌍이 있는 word만 순회: 옛 쌍들을 `stats`에서 빼고(0 이하면 삭제), `merge()`로 치환한 뒤 새 쌍들을 더한다.

### 작은 예

코퍼스 chunk 빈도: `low`x5, ` lower`x2, ` newest`x6 (공백은 `_`로 표기, 바이트로 보면 각 글자가 1개 id).

```
low     (5): l o w
_lower  (2): _ l o w e r
_newest (6): _ n e w e s t
```

쌍 빈도(가중): `(l,o)`=5+2=7, `(o,w)`=7, `(_,l)`=2, `(w,e)`=2+6=8, `(e,s)`=6, `(s,t)`=6, `(_,n)`=6, `(n,e)`=6, `(e,w)`=6 ...

1. 최빈 `(w,e)`=8 -> 새 토큰 `we`. `where[(w,e)]`에 든 `_lower`, `_newest` 두 word만 갱신: `_ l o we r`, `_ n e we s t` (옛 쌍 `(o,w)`, `(e,w)` 등의 카운트가 줄고 `(o,we)`, `(we,r)`, `(e,we)`, `(we,s)`가 생긴다). `low`는 건드리지 않는다.
2. 다음 최빈은 `(l,o)`=7 (동률이면 먼저 만들어진 키가 선택됨) -> `lo`, 이어서 `(lo,w)` 등.
3. 반복하면 `low`, ` low`, ` new`, `est` 같은 자주 쓰이는 조각이 순서대로 토큰이 된다.

(실제 코퍼스의 merge 순서는 7절 실습 3에서 `merges` dict로 확인한다.)

### 왜 `where` 인덱스가 빠른가

순진한 구현은 merge마다 전체 코퍼스의 모든 쌍을 다시 센다: 비용 `O(merges x 코퍼스 크기)`. 이 코드는

- 고유 chunk로 압축해 코퍼스 크기 대신 **고유 단어 수 x 평균 길이**만 다룬다.
- `where`로 **그 쌍을 가진 word만** 갱신한다. 드문 쌍일수록 갱신량이 작다. 전체 비용은 대략 "각 merge가 건드리는 word 수의 합"이다.

남은 병목: 매 iteration `max(stats, ...)`가 `O(|stats|)`이다 (heap/lazy 갱신으로 개선 가능, 여기서는 단순함을 택함). 또 `where`는 갱신 후 word에서 쌍이 사라져도 인덱스에 남는다(stale). 그래도 결과는 정확하다: 해당 word의 **현재** 쌍들을 모두 뺀 뒤, `merge()`가 아무것도 바꾸지 않으므로 같은 쌍들을 그대로 다시 더해 순변화가 0이다. 불필요한 순회만 생긴다 (실습 5에서 직접 따져 보자).

## 5. 인코딩과 디코딩

### encode

`_encode_chunk`: chunk를 바이트 id로 만든 뒤, 현재 쌍들 중 **rank(merges dict 삽입 순서 = 학습 순서 = id)가 가장 낮은, 즉 가장 먼저 학습된 쌍**을 합치고, 더 합칠 쌍이 없을 때까지 반복한다.

왜 이 순서인가: 학습은 "그 시점에 가장 빈번한 쌍"을 순서대로 합쳤다. 인코딩이 같은 순서를 따라야 학습 때 본 분할을 재현하고, 학습 때와 일관된 토큰열이 나온다. 나중에 학습된 merge는 앞선 merge의 결과 토큰에 의존하는 경우가 많으므로(`lo` 다음에 `lo`+`w`) 순서를 어기면 적용 자체가 불가능하기도 하다.

`self._cache`: 같은 chunk는 항상 같은 결과이므로 메모이제이션. 자연어는 chunk 반복이 많아 큰 속도 향상.

### decode

`b"".join(self.vocab[i] ...).decode("utf-8", errors="replace")`. id를 바이트로 되돌려 **전부 이어붙인 뒤 한 번에** UTF-8 디코딩한다. 토큰 하나씩 디코딩하면 한글이 3바이트 중간에서 잘려 있을 수 있다. 스트리밍 생성 중이거나 전체가 아닌 일부 id만 디코딩할 때처럼 UTF-8이 불완전하면 `errors="replace"`가 예외 대신 `�`로 처리한다.

## 6. Special tokens

`<|endoftext|>` 같은 특수 토큰은 일반 텍스트 규칙으로 쪼개면 안 된다.

- **encode**: `self._split`(specials를 `re.escape`로 묶은 캡처 그룹 정규식)으로 **PAT보다 먼저** 텍스트를 나누고, special과 일치하는 조각은 해당 id 하나로 바로 내보낸다. 나머지만 PAT -> merge. (캡처 그룹이라 `split`이 구분자도 결과에 남긴다.)
- **train**: CLI에서 `re.split`으로 문서를 special 기준으로 나눠 학습한다. 안 그러면 `<|endoftext|>` 문자열이 일반 문자 chunk로 학습되어 `<`, `|`, `end` 같은 merge가 낭비되거나 문서 경계를 넘어 이어지는 merge가 생긴다.
- special id는 merge 뒤에 붙는다 (`n = 256 + len(merges)`부터). 테스트가 `eot == tok.vocab_size - 1`을 확인한다. 그래서 학습 루프는 `vocab_size - len(specials)`까지만 돈다.

## 7. 실습

1. **압축률 비교**: TinyStories valid에서 vocab 1024 / 8192 / 32768로 각각 학습(`--vocab`, 파일은 `--max-mb`로 제한)한 뒤, `bytes / tokens` 비율을 표로 정리하라. 증가폭이 어디서 둔화되는가? embedding 파라미터(`V x d`)와 함께 비교해 보자.
2. **한국어 비용**: 8k tokenizer로 `"옛날 옛적에 작은 강아지가 살았어요."`를 encode해 토큰 수와 UTF-8 바이트 수를 비교하고, 같은 뜻의 영어 문장과 토큰 수 비율을 구하라. 토큰 하나씩 `decode([i])` 하면 `�`가 보이는가?
3. **첫 50개 merge 보기**: `tok.merges`를 순회하며 `tok.vocab[new]`를 출력. 공백+단어(` the`)나 `er`, `ing` 같은 조각이 먼저 나오는지 관찰하라.
4. **PAT 해부**: `PAT.findall("He didn't like_rain!! 안녕 123456")`를 출력해 chunk를 눈으로 확인하고, ` ?\d{1,3}` 줄을 지우면 `test_pretokenize_is_lossless`와 압축률이 어떻게 변하는지 보라.
5. **코드 읽기 질문**: `where` 인덱스가 stale일 때 `stats`를 잘못 빼지 않는가? (힌트: word 안에 쌍이 실제로 있는지와 `stats[p] <= 0` 삭제를 따져 보라.) 그리고 `encode`의 `min(...)` 대신 rank 정렬 후 순차 적용으로 바꾸면 결과가 같은가?

## 8. 실행 방법

로컬 테스트:

```bash
uv run --no-project --with pytest pytest -q tests
```

테스트: lossless pre-tokenize, 학습 후 roundtrip(처음 보는 한글/이모지 포함)과 압축 확인, special 토큰 id 및 save/load 일관성.

PGX에서 학습:

```bash
scripts/pgx.sh python -m scratchlm.tokenizer data/TinyStoriesV2-GPT4-train.txt data/tok8k.json --vocab 8192 --max-mb 50
```

학습이 끝나면 샘플 300자에 대한 `bytes -> tokens` 수와 앞 40개 토큰이 출력된다. 결과 JSON은 `{"merges": [[a, b, new], ...], "specials": {...}}` 형식이며 `Tokenizer.load()`로 복원한다.
