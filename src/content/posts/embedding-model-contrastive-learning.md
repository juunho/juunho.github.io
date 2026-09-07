---
title: 임베딩 모델은 어떻게 학습되는가 — 대조학습과 MNRL
date: 2025-02-18
description: >-
  Triplet Loss의 margin은 왜 필요한가부터, 하드 네거티브 채굴과
  MNRL이 검색 성능을 끌어올리는 지점까지.
tags:
  - Embedding
  - Contrastive Learning
  - RAG
draft: false
---

RAG 파이프라인에서 검색(Retrieval)의 품질은 임베딩 모델이 거의 결정합니다.
이 글은 검색용 임베딩 모델을 대조학습(Contrastive Learning)과
MNRL(Multiple Negatives Ranking Loss)로 학습시키면서 정리한 내용입니다.

## 대조 학습

같은 클래스(positive)를 가진 텍스트가 다른 클래스(negative)를 가진 텍스트보다
임베딩 공간에서 더 가깝게 위치하도록 학습하는 방식입니다. 데이터는 주로
Triplet 형태를 씁니다.

### 텍스트 임베딩

텍스트 데이터 $x$ 를 $d$ 차원 유클리드 공간에 매핑합니다.

$$
f(x) \in \mathbb{R}^d
$$

좋은 임베딩이 되려면 Anchor–Positive 거리가 Anchor–Negative 거리보다
짧아야 합니다.

$$
\|f(x_i^a) - f(x_i^p)\|_2^2 + \alpha < \|f(x_i^a) - f(x_i^n)\|_2^2
$$

여기서 margin $\alpha$ 가 왜 필요한지가 중요합니다. $\alpha$ 가 없으면
$f(A)$, $f(P)$, $f(N)$ 이 **전부 0으로 수렴해도** 부등식이 성립합니다.
태스크가 너무 쉽게 풀려버려서 학습이 일어나지 않습니다. 그래서
threshold 관점에서 margin을 넣습니다.

### Triplet Dataset

(Anchor, Positive, Negative) 형태의 텍스트 샘플로 구성합니다.

- **Anchor** ($a$): 기준이 되는 텍스트 — 쿼리
- **Positive** ($p$): 앵커와 같은 클래스의 텍스트 — 정답
- **Negative** ($n$): 앵커와 다른 클래스의 텍스트 — 오답

### Triplet Loss

Anchor–Positive 거리를 최소화하고 Anchor–Negative 거리를 최대화하는 것이
목표입니다.

$$
L = \sum_{i=1}^{N} \left[ \|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|_2^2 + \alpha \right]_+
$$

풀어 쓰면 이렇게 됩니다.

$$
L = \max\left(\|f(A) - f(P)\|_2^2 - \|f(A) - f(N)\|_2^2 + \alpha,\; 0\right)
$$

## Negative Sampling

### Negative Sample

모델이 부정 예시(negative example)를 학습할 수 있도록 필요한 샘플입니다.
일반적으로는 랜덤하게 골라 모델이 구별하도록 학습시킵니다.

`cat` 과 관련된 임베딩을 학습한다고 하면, 부정 예시로 `banana`, `car`, `tree`
같은 것들을 무작위로 선택합니다. 이 예시들은 `cat` 과 연관성이 거의 없습니다.

### Hard Negative Sample

부정 예시 중에서도 **모델이 정답과 혼동할 가능성이 높은** 샘플입니다.
태스크의 난이도를 높여서 성능을 더 끌어올리는 것이 목적이고, 비슷한 정답이
많은 상황에서 특히 효과가 큽니다.

같은 예로, `cat` 에 대해 `dog`, `lion`, `tiger` 를 고르는 쪽입니다.
연관성이 있어서 모델이 더 헷갈립니다.

### 난이도 구분

Triplet Loss의 목표는 이렇게 정리됩니다.

$$
\|f(A)-f(P)\|_2^2 \to 0 \ ,\quad \|f(A)-f(N)\|_2^2 > \|f(A)-f(P)\|_2^2 + \alpha
$$

Anchor–Positive 거리는 0에 수렴시키고, Anchor–Negative 거리는 거기에 margin을
더한 값보다 크게 만들어야 합니다. 이 두 거리의 관계에 따라 negative의 난이도가
갈립니다.

**Easy Negative** — $\|f(x_i^a) - f(x_i^p)\|_2^2 < \|f(x_i^a) - f(x_i^n)\|_2^2$

Anchor–Positive 거리가 Anchor–Negative 거리보다 작은 경우입니다.
쉬운 샘플이고, loss가 낮아서 학습이 안 되거나 느립니다.

**Hard Negative** — $\|f(x_i^a) - f(x_i^n)\|_2^2 < \|f(x_i^a) - f(x_i^p)\|_2^2$

Anchor–Negative 거리가 더 가까운 경우입니다. 가장 어렵지만,
global minimum을 제대로 못 찾을 수 있습니다.

**Semi-hard Negative** — $\|f(x_i^a) - f(x_i^p)\|_2^2 < \|f(x_i^a) - f(x_i^n)\|_2^2 < \|f(x_i^a) - f(x_i^p)\|_2^2 + \alpha$

Anchor–Negative 거리가 Anchor–Positive보다는 크지만 그 차이가 margin보다는
작은 경우입니다. **적절한 난이도**를 가진 샘플이고, 이걸 찾기 위해
하드 마이닝을 합니다.

## 하드 마이닝 전략

FAQ처럼 '답변 – 예상 질의' 쌍이 미리 구성된 Q-A 데이터에서 semi-hard negative를
찾는 방법입니다. 핵심은 **학습 전 모델(vanilla)로 먼저 검색을 돌려서 난이도를
측정하는 것**입니다.

1. 파인튜닝 전 베이스 모델로, 보유한 '예상 질의'를 전부 쿼리로 써서 답변을 검색합니다.
2. 쿼리마다 상위 $k$ 개 결과를 유사도 순으로 뽑습니다.
3. 검색 결과 안에 실제 정답이 없으면 `Not Found` 로 기록합니다.

여기서 두 갈래가 나옵니다.

- **결과에 정답이 있는 쿼리** — 비교적 쉬운 예제입니다. 정답이 아닌 결과 중
  가장 유사도가 낮은 오답을 negative로 씁니다. 이미 상위권에 올라온 오답이라
  자연스럽게 hard negative가 됩니다.
- **`Not Found` 인 쿼리** — 베이스 모델이 이미 정답을 못 찾은 어려운 예제입니다.
  여기에 hard negative까지 붙이면 학습이 무너지므로, 다른 답변 중에서 무작위로
  골라 easy negative로 씁니다.

두 쪽을 결합해 파인튜닝 데이터셋을 만듭니다. 난이도가 섞여 있는 게 중요합니다.

### Train/Test 분할에서 조심할 것

FAQ 데이터는 하나의 답변에 여러 개의 예상 질의가 달린 1:N 구조입니다.
그냥 무작위로 나누면 **같은 답변에 달린 질의들이 train과 test에 흩어져서**
data leakage가 생깁니다.

그래서 positive(답변)를 기준 키로 삼아 관련 anchor들을 그룹으로 묶고,
**그룹 단위로** train/test를 나눕니다. 이렇게 해야 모델이 test에서
완전히 새로운 답변군을 만나게 됩니다.

## 학습 설정

- **Model Type** — Bi-encoder (Sentence Transformer)
- **Base Model** — bge-m3
- **Maximum Sequence Length** — 8192 tokens
- **Output Dimensionality** — 1024
- **Similarity Function** — Cosine Similarity
- **Loss Function** — Multiple Negatives Ranking Loss
- **Evaluator** — Triplet Evaluator

주요 하이퍼파라미터는 다음과 같습니다.

```yaml
eval_strategy: steps
per_device_train_batch_size: 100
per_device_eval_batch_size: 100
learning_rate: 1e-05
num_train_epochs: 1
warmup_ratio: 0.05
gradient_checkpointing: true
batch_sampler: no_duplicates
```

`batch_sampler: no_duplicates` 가 중요합니다. MNRL은 배치 안의 다른 샘플을
전부 negative로 쓰기 때문에, 같은 positive가 배치에 두 번 들어가면
**정답을 오답으로 학습**하게 됩니다.

## MNRL

Triplet Loss는 triplet 하나당 negative가 하나입니다. MNRL은 배치 전체를
negative pool로 씁니다. 배치 크기가 $B$ 면 각 anchor에 대해 $B-1$ 개의
in-batch negative가 공짜로 생깁니다.

```python
import torch
import torch.nn.functional as F


def mnrl(anchor, positive, scale=20.0):
    """In-batch negatives — 배치 전체가 negative pool이 된다."""
    # (B, D) x (B, D) -> (B, B) 유사도 행렬
    sim = scale * F.cosine_similarity(
        anchor.unsqueeze(1), positive.unsqueeze(0), dim=-1
    )
    # 대각 성분이 정답 쌍이므로 label 은 0..B-1
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)
```

`scale` 은 유사도를 logit으로 바꿀 때의 온도(temperature) 역수입니다.
코사인 유사도는 $[-1, 1]$ 범위라 그대로 softmax에 넣으면 분포가 너무
평평해집니다. 보통 20 정도를 씁니다.

배치가 클수록 negative가 많아져서 학습 신호가 강해집니다. MNRL을 쓸 때
배치 크기를 최대한 키우는 이유입니다.
