# TL;DR

T에이닷 FAQ RAG의 검색(Retrieval)에 사용되는 Embedding Model의 설명 자료입니다.

모델 학습은 대조학습과 MNRL(Multiple Negative Ranking Loss)를 바탕으로 합니다.

적용 사례로 바로 넘어가고 싶다면, **Our Strategy** 부터 참고하시면 됩니다.

양자화에 대한 내용은 **Model Quantization** 부터 참고하시면 됩니다.

#### 대조 학습 (Contrastive Learning)

![[Pasted image 20250217143431.png]]
###### 대조 학습
-  동일한 클래스(positive)을 가진 텍스트가 다른 클래스(negative)을 가진 텍스트보다 임베딩 공간에서 더 가깝게 위치하도록 학습하는 방식
-  Triplet 형태의 데이터셋을 주로 활용

###### 텍스트 임베딩
-  텍스트 데이터 𝑥 를 𝑑 차원 유클리드 공간에 매핑
	-  $f(x) ∈ R^d$
	-  좋은 임베딩 성능을 위해서는 Anchor와 Positive와의 거리가 Anchor와 Negative와의 거리보다 짧아야 함

-  $||f(x^a_i) − f(x^p_i)||^2_2 + α <||f(x^a_i) − f(x^n_i)||^2_2$
	-  $f(A)$, $f(P)$, $f(N)$이 전부 0으로 수렴하는 경우, Task가 쉽게 풀려 학습이 이루어지지 않음
	-  Threshold 관점에서 Margin $α$ 를 추가

###### Triplet Dataset
- (Anchor, Positive, Negative) 형태의 텍스트 샘플로 구성
-  $∀ (f(x^a_i), f(x^p_i), f(x^n_i)) ∈ T$
	- Anchor($a$): 기준이 되는 텍스트 (쿼리)
	- Positive($p$): 앵커와 같은 클래스의 텍스트 (정답)
	- Negative($n$): 앵커와 다른 클래스의 텍스트 (오답)

###### Triplet Loss Function
- Anchor-Positive 간 거리를 최소화하고, Anchor-Negative 간 거리를 최대화하는 것이 목표
$$
L =\sum_{i=1}^N \left[ \|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|2^2 + \alpha \right]+
$$
이를 풀어내면,
$$
L=max(∣∣f(A)−f(P)∣∣^2_2−∣∣f(A)−f(N)∣∣^2_2+α, 0)
$$

#### Negative Sampling 이란?
##### Negative Sample
-  학습 과정에서 모델이 부정 예시(negative example)를 학습할 수 있도록 필요한 샘플
	-  부정 예시는 정답이 아닌 데이터를 의미
- 일반적으로, Negative Sampling에서는 랜덤하게 부정 예시를 선택하여 모델이 구별하도록 학습
- cat과 관련된 임베딩을 학습한다고 할 때, 부정 예시로 banana, car, tree 등을 랜덤하게 선택
	-  선택된 부정 예시들은 cat과 연관성이 거의 없음

##### Hard Negative Sample
-  모델이 학습하는 부정 예시 중 특히 모델이 정답과 혼동할 가능성이 높은 샘플
	-  모델에게 task의 난이도를 높여 모델의 성능을 더욱 향상시키는 것을 목적
	-  비슷한 정답이 많은 상황에서 더 좋은 성능을 발휘

-  cat과 관련된 임베딩을 학습한다고 할 때, 부정 예시로 dog, lion, tiger 등을 선택
	-  선택된 부정 예시들은 cat과 연관성이 있고, 모델이 더 혼동할 가능성이 있음

##### Loss Function과 Sampling
$$
\|f(A)-f(P)\|_2^2\to0\ ,\quad\|f(A)-f(N)\|_2^2>\|f(A)-f(P)\|_2^2+\alpha
$$
-  Triplet Loss의 목표는 Anchor와 Positive Sample의 거리는 0에 수렴하도록 만드는 것
-  Anchor와 Negative Sample의 거리는 Anchor와 Postive Sample의 거리에 Margin을 더한 값보다 크게 만들어야 함

<img src="file:///Users/a11429/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image001.png" alt="Image" width="250">
-  Loss의 목표를 바탕으로, Triplet 내 Sample을 선정
	-  위 그림에서 빨간 원의 반지름인 $\|f(x_i^a) - f(x_i^p)\|_2^2$과,
	-  원점으로부터 $n$ 까지의 거리에 해당하는 $\|f(x_i^a) - f(x_i^n)\|_2^2$ 에 따라 난이도가 구분됨

##### Easy Negative
-  $\|f(x_i^a) - f(x_i^p)\|_2^2 < \|f(x_i^a) - f(x_i^n)\|_2^2$
	-  Anchor-Positive 거리가 Anchor-Negative 거리보다 작은 경우
	-  쉬운 샘플에 해당, Loss가 낮아서 학습이 안 되거나 느림

##### Hard Negative
- $\|f(x_i^a) - f(x_i^n)\|_2^2 < \|f(x_i^a) - f(x_i^p)\|_2^2$
	- Anchor-Negative 거리가 Anchor-Positive 거리보다 작은 경우
	- 가장 어려운 난이도의 샘플이며, Global Minimum을 제대로 못 찾을 수 있음

##### Semi-hard Negative
- $\|f(x_i^a) - f(x_i^p)\|_2^2 < \|f(x_i^a) - f(x_i^n)\|_2^2 < \|f(x_i^a) - f(x_i^p)\|_2^2 + \alpha$
	-  Anchor-Negative 거리가 Anchor-Positive 거리보다 크지만, 그 차이가 Margin(α)보다는 작은 경우
	-  적절한 난이도를 가진 샘플이며, 이를 찾기 위해 우리는 Hard Mining을 수행함

#### Our Strategy
##### FAQ 데이터
- '답변 - 예상 질의' 쌍이 사전에 구성되어 있는 Q-A 형태의 FAQ 데이터 셋에서 Semi-Hard Negative를 찾기 위함
	-  답변은 전체 **1,584**건이 존재하며, 각 답변 마다 매칭된 예상 질의는 최소 1개에서 최대 1천 개 이상으로 매우 다양함
	-  임베딩 모델의 목표는 자연어 형태의 질의를 input으로 받아 올바른 답변을 output 하는 것

##### Hard Mining
-  bge-m3 Vanila Model을 활용하여, 현재 보유한 약 **25만** 건의 '예상 질의'를 쿼리로 사용하여 답변 검색 수행
	-  모든 쿼리에 대해 5개의 검색 결과(Results)를 뽑으며, 결과는 유사도 순으로 정렬
	-  검색 결과 중 실제 정답(’T전화 답변’)이 존재하지 않으면 ‘Not Found’ 기록

-  결과(Results)가 존재하는 쿼리는 비교적 검색 난이도가 낮은 예제
	-  반대로, 결과(Results)가 'Not Found'인 경우 모델이 정답을 찾지 못한 난이도가 높은 예제
	-  테스트 결과 bge-m3 Vanila Model이 정답을 찾지 못한(Not Found) 쿼리는 15,307건 존재

##### Table 1. 검색 결과 예시
| 예상 질의              | Results (유사도 Top 5 검색 결과)                                                                                                                                                                                                                                                                                                                                                          | 정답 (T전화 답변)     | Negative Mining |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | --------------- |
| 자동안심 통화가 뭔지?       | Rank 1: 자동안심 T 로밍은 ...<span style="color:rgb(255, 0, 0)"> (오답)  </span><br>Rank 2: 안심통보 서비스 … <span style="color:rgb(255, 0, 0)">(오답)  </span><br>Rank 3: 자동안심 T 로밍 음성은 … <span style="color:rgb(0, 176, 80)">(정답)  </span><br>Rank 4: 휴대폰결제 안심통보는 휴대폰 … <span style="color:rgb(255, 0, 0)">(오답) </span> <br>Rank 5: 안심문자는 월 990원 … <span style="color:rgb(255, 0, 0)">(오답)</span> | 자동안심 T 로밍 음성은 … | Hard Negative   |
| 자동으로 로밍 통화 막는게 뭐지? | Not Found (난이도 上)                                                                                                                                                                                                                                                                                                                                                                  | 자동안심 T 로밍 음성은 … | Easy Negative   |

##### Table 2. Triplet 데이터 구조화
| Anchor (=예상 질의)    | Positive (=정답)  | Negative                                                                   |
| ------------------ | --------------- | -------------------------------------------------------------------------- |
| 자동안심 통화가 뭔지?       | 자동안심 T 로밍 음성은 … | <span style="color:rgb(255, 0, 0)">(Hard)</span> 자동안심 T 로밍은 ...            |
| 자동으로 로밍 통화 막는게 뭐지? | 자동안심 T 로밍 음성은 … | <span style="color:rgb(0, 176, 80)">(Easy)</span> 데이터가 필요할 때 최소 100MB 부터 … |
-  검색 결과에 대하여, 실제 정답인 'T 전화 답변'을 Key로 하고 해당 답변에 대한 (Query, Results) 쌍을 Value로 하는 Triplet Dict 생성
	-  각 ’T 전화 답변'에 대해 10개의 쿼리(Anchor)를 랜덤 샘플링
    -  일반적인 데이터와 달리, FAQ는 하나의 답변에 대해 여러 개의 예상 질의를 갖고 있기 때문 (1:N)
    -  **1,584** 건의 정답('T전화 답변')에 대한 10개의 Query('예상 질의') 총 **15,840** 건 샘플링

- 이 때, Positive는 해당 '예상 질의'에 맞는 정답('T전화 답변') 그대로 사용
	- 각 Anchor의 Negative는 정답('T전화 답변')이 아닌 값이어야 함

###### Negative Sampling 전략
-  Results 값이 존재할 경우 (= _'자동안심 통화가 뭔지?'_)
    -  정답이 아닌 결과 4개 중 가장 유사도가 낮은 오답(검색 결과의 마지막 유사도 순) 선택
		- Hard Negative Sampling 수행 (**15,840** 건)

-  Results 결과가 없어 Not Found 일 경우 (= _'자동으로 로밍 통화 막는게 뭐지?'_)
    -  다른 'T전화 답변' 중에서 무작위로 선택
	    -  이미 모델이 맞추지 못하는 어려운 난이도의 질문이기 때문에 Easy Negative Sampling 수행 (**15,307** 건)

-  난이도에 따라 샘플링한 Hard Negative(**15,840** 건)과 Easy Negative(**15,307** 건)을 결합하여 Fine-tuning 데이터셋 생성(**31,147** 건)

##### Train/Test Split
- Triplet 그룹화
	-  Positive를 기준 키로 하고, 해당 Positive와 관련된 10개의 Anchor를 그룹화
	-  Data Leakage를 방지하고, 모델이 test 세트에서 완전히 새로운 Anchor에 대한 검색 결과들을 평가할 수 있음

- Train/Test 분할
	-  Unique하게 그룹화된 Positive들을 9:1 비율로 train과 test 세트로 나눔
		-  쿼리를 기반으로 계층화 분할(stratified split) 수행
	-  전체 데이터 중 90%를 train용으로, 10%를 test용으로 할당
		-  Train : **27,630** 건
		-  Test : **3,517** 건

#### Model Training
##### 학습 전략
<img src="FAQ_RAG-임베딩_0106_2.png" alt="Image" width="450">

##### Model Description
-  Model Type : Bi-encoder (Sentence Transformer)
-  Base Model : bge-m3
-  Maximum Sequence Length: 8192 tokens
-  Output Dimensionality : 1024
-  Similarity Function : Cosine Similarity
-  Loss Function : Multiple Negative Ranking Loss
-  Evaluator : Triplet Evaluator

##### Training Hyperparameter
-  eval_strategy : steps
-  per_device_train_batch_size : 100
-  per_device_eval_batch_size: 100
-  learning_rate: 1e-05
-  num_train_epochs: 1
-  warmup_ratio: 0.05
-  gradient_checkpointing: True
-  batch_sampler: no_duplicates

#### Results
##### 05.02 FAQ 데이터 테스트 결과
-  BGE-M3 (Vanilla)
	-  검증 데이터 : 245,538건
	-  MRR@5 : 0.8364
	-  Recall@5 : 0.9377
	-  Not Found : 15,307건

-  Our Model (Fine-tuned)
	-  검증 데이터 : 245,538건
	-  MRR@5 : 0.9555
	-  Recall@5 : 0.9929
	-  Not Found : 1,749건

##### 07.18 FAQ 데이터 테스트 결과
-  BGE-M3 (Vanilla)
	-  검증 데이터 : 243,117건
	-  MRR@5 : 0.8354
	-  Recall@5 : 0.9379
	-  Not Found : 15,101건
	
-  Our Model (Fine-tuned)
	-  검증 데이터 : 243,117건
	-  MRR@5 : 0.9565
	-  Recall@5 : 0.9930
	-  Not Found : 1,694건

##### 08.28 모델 드리프트 검증 결과
-  Our Model (Fine-tuned)
	-  검증 데이터 : 2,621건
	-  MRR@5 : 0.9882
	-  Recall@5 : 1.0
	-  Not Found : 0건

#### Model Quantization
###### ONNX (Open Neural Network Exchange)
-  


MNRL 추가