## Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation

### Overview (Abstract)
- MTSF(다변량 시계열 예측)에 LLM을 도입하기 위해, textual prompt 튜닝을 시도하는 연구가 진행되고 있음.
- LLM을 사용하여 MTSF를 수행하는 경우, 추론 과정에서 LLM의 효율이 떨어짐.
- 이 문제를 해결하기 위해, Calibrated Language Model(CLM; 보정된 언어 모델)과 Privileged Knowledge Distillation(PKD; 특권 지식 정제)을 활용하는 효율적인 MTSF 프레임워크인 "TimeKD"를 제안함.
- TimeKD는 고퀄리티 미래 예측의 수행을 목표로 하며, 이를 위해 이기종 Teacher Model 과 효과적인 Student Model을 고안하고자 함.
- Teacher Model은 정답 프롬프트를 가진 CLM를 활용함. 또한, Subtractive Cross Attention(SCA) 매커니즘을 사용하여 생성된 표현에서 텍스트를 제거함. (시계열 예측에 필요한 것만 남김)
- Student Model은 PKD를 활용하는데, 이를 통해 Student Model은 Teacher Model의 지식을 그대로 사용(Replicate)할 수 있으며, 이를 통해 Teacher 와 Student의 출력(Output) 불균형을 최소화함.

### Introduction
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
- Teacher Model의 학습에 LLM을 활용하되, 시계열 데이터를 텍스트 프롬프트 템플릿에 적용하여 사용함.