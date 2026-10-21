---
type: concept
title: numpy Tensor autograd (reverse-mode 자동미분)
description: scratchlm/autograd.py의 reverse-mode autograd 원리, 연산별 backward 유도, broadcasting, fused cross-entropy, gradcheck, byte-level MLP LM 데모와 실습
tags: [autograd, backprop, numpy, gradcheck, cross-entropy, scratchlm]
timestamp: 2026-10-21T00:00:00Z
resource: scratchlm/autograd.py
---

# numpy Tensor autograd

전체 학습 계획은 [로드맵](../2026-10-21-roadmap.md), 입력 쪽 단계는 [BPE 토크나이저](2026-10-21-bpe-tokenizer.md) 참고. 이 페이지는 `scratchlm/autograd.py`(numpy만 사용)를 원리 중심으로 설명한다.

## 1. 왜 autograd인가

학습은 스칼라 손실 `L(θ)`의 기울기 `dL/dθ`를 구해 파라미터를 조금씩 옮기는 일이다. 모델은 수십 개의 단순 연산(add, matmul, tanh ...)의 합성이므로 **계산 그래프**(노드 = 텐서, 간선 = 연산)로 볼 수 있고, 기울기는 **연쇄 법칙**으로 조립된다.

```
y = f(u), u = g(x)  =>  dL/dx = dL/dy * dy/du * du/dx
```

곱의 순서를 어느 쪽부터 계산하느냐가 두 방식을 가른다.

- **forward mode**: 입력 하나를 골라 `dx/dx = 1`에서 출발해 앞에서 뒤로 전파. 입력 하나당 한 번의 pass가 필요하다. 파라미터가 N개면 N번.
- **reverse mode**: 출력에서 `dL/dL = 1`로 출발해 뒤에서 앞으로 전파. **출력이 하나(스칼라 loss)면 한 번의 backward로 모든 파라미터의 기울기**가 나온다.

LLM은 입력(파라미터) 수억 개 대 출력(loss) 1개이므로 reverse mode가 압도적으로 유리하다. 비용은 대략 forward 1~3배다. 대가로 backward에 필요한 중간값을 메모리에 들고 있어야 한다. `backward()`가 하는 일이 정확히 이것이고, torch의 `loss.backward()`와 같은 원리다.

## 2. 설계

```python
class Tensor:
    data   # np.float64 배열
    grad   # dL/d(this), backward 후 data와 같은 shape
    _prev  # 이 텐서를 만든 부모 텐서들
    _fns   # 부모별 함수: 상류 기울기 g = dL/d(out)  ->  dL/d(부모) 기여분
```

연산은 결과 텐서를 만들 때 "부모마다 하나의 backward 함수"를 클로저로 붙인다 (`_out`). 예: `a*b`는 `(a, lambda g: g*b)`, `(b, lambda g: g*a)`를 단다. 연쇄 법칙의 한 조각 `dL/d(parent) = dL/d(out) * d(out)/d(parent)`를 그대로 코드로 옮긴 것이다.

### backward()

1. `visit`: **post-order DFS**. 부모를 모두 방문한 뒤 자신을 `order`에 추가하므로 `order`는 위상 정렬(부모가 항상 자식보다 앞)이다. `seen`(id 기준)으로 한 노드를 한 번만 넣는다.
2. 모든 노드의 `grad`를 0으로 만들고, 루트(`loss`)만 `ones`로 둔다 (`dL/dL = 1`).
3. `reversed(order)`로 순회하며 각 노드의 `grad`를 부모의 `fn`에 통과시켜 부모 `grad`에 더한다.

왜 역위상 순서인가: 노드 `t`를 처리할 시점에는 `t`를 사용하는 **모든 자식이 이미 처리되어** `t.grad`가 완성되어 있어야 한다. 역위상 순서가 이를 보장한다.

### 왜 `+=` (누적)인가

텐서 `x`가 두 연산에 쓰이면(`x*x`, 잔차 연결 `x + f(x)`) 변화가 두 경로로 `L`에 영향을 준다. 다변수 연쇄 법칙은 경로별 기여의 **합**이다.

```
L = h(u(x), v(x))   =>   dL/dx = dL/du * du/dx + dL/dv * dv/dx
```

코드의 `p.grad = p.grad + fn(t.grad)`가 그 합이다. 덮어쓰면 마지막 경로만 남아 틀린 기울기가 나온다. (in-place `+=` 대신 새 배열을 만드는 것은 `broadcast_to`가 읽기 전용 view를 돌려주기 때문에 안전하기도 하다.) 파라미터는 매 step마다 새로 그래프를 만들고 `backward()`가 grad를 0부터 다시 시작하므로 `zero_grad`가 따로 필요 없다.

## 3. 연산별 backward

`g`는 상류 기울기(출력과 같은 shape)다.

| 연산 | forward | 부모로 보내는 기여 (코드) |
|---|---|---|
| add | `a+b` | a: `g`, b: `g` (각각 `_unbroadcast`) |
| mul | `a*b` | a: `g*b`, b: `g*a` |
| pow | `x**k` (k는 상수) | `g * k * x**(k-1)` |
| exp | `e=exp(x)` | `g * e` |
| log | `log(x)` | `g / x` |
| tanh | `t=tanh(x)` | `g * (1 - t*t)` |
| relu | `max(x,0)` | `g * (x > 0)` |
| sum | `x.sum(axis)` | `g`를 x의 shape으로 broadcast |
| mean | `sum * (1/n)` | sum과 mul의 합성 |
| reshape | `x.reshape(s)` | `g.reshape(x.shape)` |
| matmul | `A@B` | A: `g @ Bᵀ`, B: `Aᵀ @ g` |
| getitem | `E[idx]` | `np.add.at(zeros, idx, g)` |

### 유도 요약

- **add/mul**: `∂(a+b)/∂a = 1`, `∂(ab)/∂a = b`. 연쇄 법칙에 따라 `g`에 곱한다. 파생 연산(단항 `-`, 이항 `-`, `/`)은 새 backward 없이 합성된다: `-x = x*(-1)`, `a-b = a+(-b)`, `a/b = a*b**-1`. 연산 몇 개만 정확하면 나머지는 공짜다.
- **pow/exp/log**: 미분 공식 `kx^(k-1)`, `e^x`, `1/x` 그대로. `exp`는 forward에서 계산한 `e`를 클로저가 재사용한다 (재계산 없음).
- **tanh**: `tanh' = 1 - tanh²`. forward 결과 `t`로 표현되므로 `x`를 다시 안 봐도 된다.
- **relu**: `x>0`이면 1, 아니면 0. `x=0`의 미분은 정의되지 않으며 코드는 0으로 둔다 (`>`).
- **sum**: `y = Σ x_i`면 `∂y/∂x_i = 1`이므로 모든 원소가 같은 `g`를 받는다 = `g`를 x의 shape으로 **복제(broadcast)**. `keepdims=False`로 축이 사라졌다면 `g`는 그 축이 없으므로 `expand_dims(g, axis)`로 길이 1 축을 먼저 되살려야 `broadcast_to`가 맞는 방향으로 펼친다. `axis=None`이면 `g`는 스칼라라 그대로 broadcast된다.
- **mean**: `sum * (1/n)`으로 구현하므로 기울기가 자동으로 `g/n`이 된다.
- **reshape**: 원소 배치만 바꾸므로 `g`를 원래 shape으로 되돌리면 끝.
- **matmul**: 다음 절.

### matmul 유도 (shape으로 따라가기)

`C = A B`, `A: (n,k)`, `B: (k,m)`, `C: (n,m)`, `G = dL/dC: (n,m)`.

```
C_ij = Σ_p A_ip B_pj
dL/dA_ip = Σ_j G_ij * ∂C_ij/∂A_ip = Σ_j G_ij B_pj = (G Bᵀ)_ip     shape (n,m)@(m,k) = (n,k) = A의 shape
dL/dB_pj = Σ_i G_ij * ∂C_ij/∂B_pj = Σ_i G_ij A_ip = (Aᵀ G)_pj     shape (k,n)@(n,m) = (k,m) = B의 shape
```

shape 검사가 기억법이다: 결과가 부모 shape이 되도록 `G`에 나머지 하나의 전치를 붙인다. 코드는 배치 차원을 위해 `swapaxes(-1,-2)`를 쓴다. 한쪽에 배치 차원이 없으면 기울기에 배치 축이 남으므로 `_unbroadcast`로 합산해 되돌린다.

### 임베딩 lookup (`__getitem__`)

`E[ids]`는 `E`의 행을 골라 복사한다. 선택된 행에는 기울기가 그대로 흘러가고 선택되지 않은 행의 기울기는 0이다. backward는 `E`와 같은 shape의 0 배열 `z`에 `g`를 해당 행 위치로 **흩뿌려 더한다**.

```python
z[idx] += g          # 틀림: idx에 같은 id가 반복되면 마지막 하나만 반영
np.add.at(z, idx, g) # 맞음: unbuffered, 반복된 id의 기여를 모두 누적
```

`z[idx] += g`는 `z[idx] = z[idx] + g`로 풀리는데, 반복된 인덱스는 같은 낡은 값에 각각 더한 뒤 덮어쓰므로 하나만 남는다. `test_embedding_lookup_accumulates`는 id 2가 세 번 나오는 경우를 검증한다. 이는 2절의 "여러 경로의 합" 원리가 임베딩 행 안에서 일어나는 것이다 (같은 토큰이 여러 위치에 쓰임).

## 4. Broadcasting과 `_unbroadcast`

numpy는 `(3,4) + (4,)`에서 `(4,)`를 3행에 복제해 더한다. forward에서 한 값이 여러 번 쓰였으므로 2절의 원리에 따라 **backward에서는 그 사본들의 기울기를 합해야 한다**.

예: `y = X + b`, `X: (3,4)`, `b: (4,)`. 같은 `b_j`가 세 행에 쓰였다.

```
y_ij = X_ij + b_j
dL/db_j = Σ_i G_ij        (G = dL/dy, shape (3,4))
```

`_unbroadcast(G, (4,))`가 이 합이다.

1. `while g.ndim > len(shape): g = g.sum(0)`: 앞쪽에 새로 붙은 축을 합산. `(3,4) -> (4,)`.
2. `for i, s in enumerate(shape): if s == 1 and g.shape[i] != 1: g = g.sum(i, keepdims=True)`: 원래 길이 1이던 축이 펼쳐졌다면 합산하되 축을 유지. 예: `b: (1,4)`, `G: (3,4)` -> `(1,4)`.

`X`의 기울기는 shape이 같으므로 `G` 그대로다. 바이어스 기울기 = 배치 전체의 기울기 합이 되고, 데모에서 `b1`, `b2`가 이 경로를 쓴다. 반대로 broadcast의 backward가 합이고 sum의 backward가 broadcast인 것은 서로 전치 관계다.

## 5. cross_entropy의 fused backward

로짓 `z ∈ R^C`, 정답 `y`에 대해 `p = softmax(z)`, `L = -log p_y`.

```
L = -z_y + log Σ_j exp(z_j)
∂L/∂z_k = -1[k=y] + exp(z_k) / Σ_j exp(z_j) = p_k - 1[k=y]
```

즉 **softmax에서 정답 위치만 1을 뺀 벡터**다. 배치 평균(`mean`)이면 행당 손실에 `1/N`이 붙으므로 `dL/dZ = (softmax(Z) - onehot) / N`. 코드는 `d = exp(logp); d[arange(n), targets] -= 1; return g * d / n`로 정확히 이 식이고 `g`는 스칼라 상류 기울기다.

왜 fused인가: `softmax`, `log`, `-`, `mean`을 따로 쓰면 `1/p` 같은 항이 나와 `p`가 0에 가까울 때 불안정하고 느리다. 합쳐서 한 번에 유도하면 결과가 단순하고 안정적이다.

**수치 안정성 (max-shift / log-sum-exp)**: `exp(z)`는 `z`가 크면 overflow. softmax는 상수 `m`을 빼도 불변이므로

```
log softmax(z)_k = (z_k - m) - log Σ_j exp(z_j - m),   m = max_j z_j
```

코드의 `z = logits - max`, `logp = z - log(Σ exp z)`가 그것이다. 가장 큰 항의 지수가 `exp(0)=1`이라 overflow가 없고 합이 최소 1이라 `log`의 인수가 0이 될 수 없다. 기울기 `exp(logp)`도 이 안정한 값에서 얻는다. 기울기 합이 0인 점(`Σ_k (p_k - onehot_k) = 0`)은 로짓에 상수를 더해도 loss가 불변이라는 성질과 일치한다.

## 6. Gradient check

backward를 손으로 짰으니 틀릴 수 있고, 틀려도 loss가 그럴듯하게 줄어드는 경우가 있다. 검증은 **중심 유한차분**과 비교한다.

```
(f(x+ε) - f(x-ε)) / (2ε) = f'(x) + O(ε²)
```

테일러 전개에서 `f(x±ε) = f ± εf' + ε²f''/2 ± ε³f'''/6`이므로 짝수 차항이 상쇄되어 오차가 `O(ε²)`다 (전진차분은 `O(ε)`). 단, `ε`을 너무 줄이면 `f(x+ε)-f(x-ε)`가 반올림 오차에 묻힌다. 그래서 **float64**가 필수다: 기계 엡실론 약 1e-16에 `ε=1e-6`이면 상대 반올림 오차가 약 1e-10, 절단 오차 약 1e-12로 둘 다 작다. float32(약 1e-7)로는 `rtol=1e-5` 비교 자체가 불가능하다. 그래서 `Tensor`는 항상 `float64`로 변환한다.

`tests/test_autograd.py`의 `gradcheck(f, *arrays, eps=1e-6)`: 모든 입력 원소를 하나씩 `±eps` 흔들어 수치 기울기를 만들고 `assert_allclose(rtol=1e-5, atol=1e-7)`로 `t.grad`와 비교한다. 이 한 함수로 다음을 모두 점검한다.

- `test_elementwise_and_broadcast`: add/mul/pow/div/tanh/exp/log/relu, `(3,4)`와 `(1,4)`, `(3,)` broadcast
- `test_matmul_reduce_reshape`: 3차원 matmul, `axis`, `keepdims`, reshape
- `test_embedding_lookup_accumulates`: 반복 id
- `test_cross_entropy`: gradcheck + 직접 계산한 참값과 비교
- `test_mlp_learns_xor`: 500 step SGD 후 loss < 0.05 (end-to-end)

한계: relu의 꺾이는 점, `|x|` 같은 비미분 점 근처는 유한차분이 틀리므로 테스트 입력을 피해서 골라야 한다. 비용은 파라미터 수에 비례하므로 작은 입력으로만 쓴다.

## 7. 데모: byte-level MLP 언어 모델

`python -m scratchlm.autograd`는 Bengio et al. (2003)식 신경 언어 모델을 학습한다. TinyStories valid의 앞 5MB를 바이트 그대로 쓴다 (토크나이저 없이 vocab V=256, [BPE 페이지](2026-10-21-bpe-tokenizer.md)의 기본 256개와 같다).

```
이전 8 byte -> E[x]: (B,8,24) -> reshape (B,192) -> @W1 + b1 -> tanh (B,256) -> @W2 + b2 -> logits (B,256) -> cross_entropy
```

| 파라미터 | shape | 개수 |
|---|---|---|
| E | 256x24 | 6,144 |
| W1 | 192x256 | 49,152 |
| b1 | 256 | 256 |
| W2 | 256x256 | 65,536 |
| b2 | 256 | 256 |
| 합계 | | **121,344** |

SGD, batch 128, lr 0.1 (step < 70%) 이후 0.01. 매 step `p.data -= lr * p.grad`가 전부다.

PGX에서 실제 측정:

| step | val loss |
|---|---|
| 0 | 5.460 |
| 6000 | 1.513 (약 10초) |

- **step 0의 5.460**: 균등분포 예측의 손실은 `ln 256 = 5.545`다. `W2`를 `* 0.1`로 작게 초기화했으므로 로짓이 거의 0에 가깝고 softmax가 거의 균등하다. 그래서 초기 loss가 `ln V` 근처에서 시작한다 (큰 초기화면 무작위 확신 때문에 훨씬 큰 loss로 시작해 처음 몇 step을 loss 낮추기에 낭비한다). 초기 loss가 `ln V`와 크게 다르면 버그의 신호다.
- **샘플** (학습 후, 프롬프트 "Once upon a time"): `Once upon a time, thel, yilaus and yox pazed, she tree juyp bean frieed...`

영어처럼 **보이는** 이유: 직전 8바이트로 다음 바이트 분포를 배우므로 철자 수준의 통계(자주 쓰는 글자 조합, 공백 위치, 쉼표 뒤 공백)는 잘 잡는다. 의미가 **없는** 이유: 문맥이 고정 8바이트라 단어 하나 길이 정도만 보고, 8바이트 밖의 정보(주어, 앞 문장)는 전혀 쓸 수 없다. attention처럼 문맥 전체를 내용 기반으로 참조하는 구조가 없고, 위치별 가중치(`W1`의 8개 블록)가 고정이다. 이 한계가 이후 단계에서 Transformer로 가는 동기다. 참고로 val 1.513 nats/byte는 바이트당 약 2.18 bit이다.

## 8. torch와의 대응, 한계

| 이 구현 | torch |
|---|---|
| `Tensor`, `_prev`, `_fns` | `torch.Tensor`, `grad_fn`의 `Node` 그래프 |
| `backward()` 위상 순회 | `loss.backward()` (autograd engine) |
| `_unbroadcast` | 모든 broadcasting 연산 내장 |
| `cross_entropy` fused | `F.cross_entropy` (log-softmax + nll fused kernel) |
| `p.data -= lr*p.grad` | `torch.optim.SGD/AdamW` |

torch가 더하는 것: GPU 실행과 fused/최적화 kernel(FlashAttention 등), bf16/fp16 mixed precision, in-place 연산의 버전 추적과 안전한 autograd, 메모리 관리(activation checkpointing, 그래프 해제), 수백 개의 검증된 연산, 분산 학습. 이 구현은 float64 CPU 전용에 연산 수가 적고, 매 step마다 중간 배열을 새로 만들어 메모리와 속도 모두 비효율적이다.

그래서 2단계부터는 torch를 쓴다. 대략 CPU numpy는 GB10 GPU 학습 대비 약 50배 느린 수준이다 (이 수치는 측정이 아닌 추정치). 중요한 점은 **원리는 동일**하다는 것이다. 이 페이지를 이해하면 `loss.backward()`가 마법이 아니게 된다.

## 9. 실습

1. **새 연산 추가**: `softmax(axis)` 또는 `layernorm`을 `Tensor` 메서드로 추가하고 `gradcheck`로 검증하라. softmax의 야코비안은 `diag(p) - ppᵀ`다. 합성(`exp`, `sum`, `/`)으로 만든 것과 직접 backward를 쓴 것의 기울기가 같은지, 코드 길이가 어떻게 다른지 비교하라.
2. **tanh -> relu**: 데모의 `tanh()`를 `relu()`로 바꿔 같은 6000 step에서 val loss와 학습 속도를 비교하라. 초기 lr이 맞는지 확인하고, 죽은 relu 유닛 비율(`h == 0`)을 세어 보라.
3. **context 확장**: `ctx = 16`으로 늘리면 `W1`이 `(384,256)`이 된다. 파라미터 수를 직접 계산한 뒤 출력과 맞춰 보고, val loss가 얼마나 개선되는지, 샘플이 달라지는지 보라.
4. **Optimizer 구현**: momentum SGD (`v = βv + g; p -= lr*v`), 이어서 Adam(1차/2차 모멘트와 bias correction)을 직접 만들어 같은 step 수의 val loss를 비교하라. lr 스케줄 없이도 되는지 보라.
5. **희귀 바이트의 임베딩 기울기**: 한 step 후 `E.grad`의 행별 노름을 출력하라. 그 batch에 안 나온 바이트의 행은 정확히 0인가? 자주 나오는 공백(32)과 희귀 문자의 노름은 얼마나 다른가? 3절의 `np.add.at`을 `z[idx] += g`로 바꾸면 gradcheck가 실패함을 확인하라.

## 10. 실행 방법

로컬 테스트 (gradcheck 포함):

```bash
uv run --no-project --with numpy python -m unittest -v
```

PGX에서 데모 학습 (기본 데이터 `data/TinyStoriesV2-GPT4-valid.txt`, 인자로 다른 경로 지정 가능):

```bash
scripts/pgx.sh python -m scratchlm.autograd
```

출력: 파라미터 수와 train 바이트 수, 1000 step마다 `train`/`val` loss와 경과 시간, 마지막으로 샘플 300바이트.
