## Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation

### I. Overview (Abstract)
- MTSF(다변량 시계열 예측)에 LLM을 도입하기 위해, textual prompt 튜닝을 시도하는 연구가 진행되고 있음.
- LLM을 사용하여 MTSF를 수행하는 경우, 추론 과정에서 LLM의 효율이 떨어짐.
- 이 문제를 해결하기 위해, Calibrated Language Model(CLM; 보정된 언어 모델)과 Privileged Knowledge Distillation(PKD; 특권 지식 정제)을 활용하는 효율적인 MTSF 프레임워크인 "TimeKD"를 제안함.
- TimeKD는 고퀄리티 미래 예측의 수행을 목표로 하며, 이를 위해 이기종 Teacher Model 과 효과적인 Student Model을 고안하고자 함.
- Teacher Model은 정답 프롬프트를 가진 CLM를 활용함. 또한, Subtractive Cross Attention(SCA) 매커니즘을 사용하여 생성된 표현에서 텍스트를 제거함. (시계열 예측에 필요한 것만 남김)
- Student Model은 PKD를 활용하는데, 이를 통해 Student Model은 Teacher Model의 지식을 그대로 사용(Replicate)할 수 있으며, 이를 통해 Teacher 와 Student의 출력(Output) 불균형을 최소화함.

----
### II. Introduction
- MTSF 수행을 위하여 전통적인 방법(CNN, FCN, Transformer)을 사용하는 경우, 특징 추출에 한계가 있음.
- MTSF 수행을 위하여 LLM을 사용하는 경우, 추론 효율이 떨어지고 처리 시간 증가함.
- 위의 두 가지 문제를 해결하기 위하여, 지식 증류를 활용함. Teacher Model의 표현 능력을 정제하여, 소규모 Classical Model(=Student Model)에 적용
- 이러한 접근은 깊은 특징 추출 (Deep Feature Extraction)과 빠른 추론을 가능하게 하지만, 몇 가지 문제가 있음. (기존의 문제)
```
1. High-Quality Teacher Model 훈련과 관련된 문제
- 시계열 데이터(=Historical Data)의 경우, 분포가 제한적이므로 예측 성능 하락
- Teacher 모델의 학습에 의해 Student Model의 예측에 편향(Bias)이 발생할 수 있음.
2. 효과적인 Student Model 생성과 관련된 문제
- Student 와 Teacher의 Output 차이가 줄어드는 방향으로 학습함.
- Teacher 모델의 학습 과정을 알 수 없으므로, 불완전한 지식 전달 발생. (사고과정 학습 X -> 모든 지식 적용 불가)
```
- 이러한 문제를 해결하기 위해, 이기종 데이터를 처리할 수 있는 TimeKD 프레임워크를 제안.
#### Teacher Model (for Challenge 1)
- Teacher Model의 학습에 LLM을 활용하되, 시계열 데이터를 텍스트 프롬프트 템플릿에 적용하여 사용함.
- Learning Under Privileged Information (LUPI) 기법 적용 - 효과적인 예측을 수행하기 위해, 학습에 유용하게 활용할 수 있는 Text 정보를 LLM에 함께 투입.
- 참값에 해당하는 데이터를 사용해도 문제 없음. (Teacher Model은 Training에만 활용됨 - Testing에는 활용되지 않음.)
- Calibrated Language Models (CLM) 적용 - 학습을 위하여 LLM에 투입할 Prompt 정제하기 / 이기종 데이터의 혼합을 억제하고, 동일 종류 데이터의 상관관계 증가
- Subtractive Cross-Attention (SCA) 적용 - 예측된 시계열 데이터 표현에 포함된 Text 정보를 제거함. (시계열 데이터만 보존하기)

#### Student Model (for Challenge 2)
- Privileged Knowledge Distillation (PKD) 도입 - Teacher 모델의 표현 능력을 Student 모델로 전달하기 위해, 상관관계 파악과 특징 정제를 수행함.
- 상관관계 파악: Student Model은 Teacher Model의 추론 과정을 복사
- 특정 정제(Feature Distillation): Teacher Model 와 Student Model의 Output 불균형이 최소화되는 방향으로 학습

----
### III. Related Works
#### 1. LLM 기반 시계열 데이터 예측
- TEMPO, OFA, LLM4TS, TEST 모델 (파인튜닝한 LLM을 시계열 데이터 예측에 사용)
- 시계열 데이터 예측을 위한 멀티모달 (다양한 형태의 데이터를 동시에 처리할 수 있는) LLM은 "channel-independent model"과 "channel-dependent model"로 구분 가능
- channel-independent model: 각 시계열 변수를 별도로 처리 (변수 간 관계를 무시함) -> 학습 시간이 길고, 예측 결과(퀄리티) 또한 최적이 아님. (suboptimal performance)
- channel-dependent model: 여러 변수 간의 상관관계를 파악하여, 시계열 데이터의 특징을 효과적으로 추출하고자 함.
- 현재의 LLM 기반 예측 모델은, 경량화/효율성/확장 가능성의 측면에서 문제가 있음.

#### 2. LLM 기반 지식 정제 (LLM-based Knowledge Distillation; KD)
- Knowledge Distillation 은 Black-Box 정제와 White-Box 정제로 구분 가능
- Black-Box 정제: Student Model은 Teacher Model의 예측 정보만 접근 가능 / LLM의 발전으로, 서로 다른 도메인의 정보를 활용하고, 높은 연산 부담을 완화하는 기술로 부상.
- White-Box 정제: Student Model은 Teacher Model의 내부 가중치 (추론 과정) 활용 가능
- LLM은 시계열 데이터 분석을 목적으로 설계되지 않음. -> Student Model 학습을 위해 LLM에만 의존할 수는 없음. (Downstream Task 수행을 위해 필요한 정보를 파악하기 힘듦.)

----
### IV. Problem Definition
#### Def. 1) 다변량 시계열 데이터
- 시간 순서로 나열된 데이터 X = { X_1, X_2, ..., X_|X| } (X_i: 시간이 i일 때, N개의 변수(특징)를 표현하는 N차원 벡터)
- v_i: 시점 i에서의 시계열 값 (벡터 X_i를 구성하는 개별 변수의 값) - 데이터셋에서, Training 을 목적으로 분리한 부분
- X_H: Teacher Model 학습에 사용되는 시계열 데이터 (historical time series)
- X_G: Teacher Model이 사용하는 참값 (Ground Truth)
- X_O: 예측 값 성능 평가를 위해 사용되는 관측값 (Observed Time Series) - 데이터셋에서, Test를 목적으로 분리한 부분

#### Def. 2) Prompt
- Text Template 에 시계열 데이터를 합침. (ex. From <t-H+1> to <t>, values were <h_i, ... , h_j>)
- X_H 는 텍스트 프롬프트 P_HD(Historical Data)로 변환되고, 텍스트 P_HD에서의 단어 개수를 W_HD라고 정의함.


#### 목표: 다변량 시계열 예측
- 시간이 M 만큼 흘렀을 때의 시계열 데이터 X_M 을 예측하고, 이것을 X_O와 비교 (성능 평가) (M: 미래 시점 // X_O 데이터에 해당하는 시점)
- Student Model이 X_M을 예측하며, Student Model은 PKD를 활용하여 Teacher Model로부터 학습하였음.
- 구체적으로, Teacher Model은 X_H, P_HD, X_G, P_GT를 활용하여 학습하고, Student Model은 PKD 과정을 통해 교사 추론 과정과 내부 표현을 따라하도록 훈련.
(Student가 직접 X_G, P_GT에 접근하지 않음.)

----
### V. Methodology
#### 