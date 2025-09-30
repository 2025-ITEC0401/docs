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
생략

----

### VI. Experimantal Evaluation
#### Experiment Setup
1. Dataset: ETT Series, Exchange, Weather (for Long-Term) / PEMS Series (for Short-Term)
   - ETTh1 & ETTh2: 변압기의 Oil 및 부하(Load) 정보를 1시간 단위로 기록한 데이터
   - ETTm1 & ETTm2: 변압기위 Oil 및 부하(Load) 정보를 15분 단위로 기록한 데이터
   - Weather: 독일에서 측정한 기상 지표 21가지 (ex. 온도, 습도 등) 를 10분 단위로 기록한 데이터
   - Exchange: 8개국의 일간 환율을 기록한 데이터 (1990 - 2016)
   - PEMS04 & PEMS08: 캘리포니아의 네트워크 트래픽을 5분 단위로 기록한 데이터.
  
2. Baselines
   - TimeCMA: 교차 모달리티 정렬을 활용하여, 효율적인 시계열 예측을 위한 분리된 특징(disentangled features)을 추출하는 LLM 기반 모델
   - Time-LLM: 텍스트 프로토타입으로 시계열을 재구성하는 방식으로 시계열 데이터를 예측하는 LLM 기반 모델
   - UniTime: 순수 텍스트 지침 (Pure Text Instruction)을 통합하여 교차 도메인 예측을 수행하는 LLM 기반 모델
   - OFA: 어텐션 및 피드포워드 레이어를 고정시키고, 다른 레이어를 미세 조정하여 시계열 예측을 수행하는 LLM 기반 모델
   - iTransformer: 변수 간의 장기 의존성을 나타내기 위해 역임베딩(Inverted-Embedding)을 도입하여, 시계열 예측을 위해 트랜스포머 인코더를 활용하는 모델
   - PatchTST: 패칭 매커니즘과 채널 독립적 전략을 트랜스포머 기반 모델에 도입한 모델

3. 평가 지표
   - MSE: 평균제곱오차 / 오차를 제곱한 것의 평균 (낮을수록 좋음)
   - MAE: 평균절대오차 / 오차의 절댓값의 평균 (낮을수록 좋음)

4. Implementation Details
   - batch size = 1
   - input lengtg = 96
   - forecasting horizons = 24, 36, 48, 96, 192
   - AdamW Optimizer 적용
   - Lowest Average Validaiton Loss for Testing
   - 사용한 LLM: BERT, GPT-2, LLaMA-3.2
   - LLM Layer = 12
   - Hidden Dimension of the Transformer = 64
   - The number of Transformer Layer = 2
  
#### Experiment Result
1. 장기 예측 성능 비교
   - TimeKD는 가장 성능이 높은 baseline(TimeCMA)보다 9.11% 낮은 MSE, 7.52% 낮은 MAE를 기록함.
   - 특히 ETTm2 데이터셋에서의 성능 향상 정도가 가장 높음.
   - 또한, 일반적으로 LLM 기반 모델이 Transformer 기반 모델보다 더 높은 성으을 보임. (iTransformer는 모든 데이터셋에서 가장 낮은 성능을 보임.)

2. 단기 예측 성능 비교
   - TimeKD는 가장 성능이 높은 baseline(TimeCMA)보다 높은 성능을 보임.
   - PEMS04 데이터셋에서, TimeKD는 TimeCMA 보다 10.81% 낮은 MSE를 기록하였고, PEMS08 데이터셋에서는 TimeCMA 보다 10.26% 낮은 MSE를 기록함.
   - PEMS04 데이터셋에서, TimeKD는 iTransformer 보다 15.38% 낮은 MSE를 기록하였고, PEMS08 데이터셋에서는 iTransformer 보다 11.39% 낮은 MSE를 기록함.
   - Time-LLM, OFA, UniTime, PatchTST의 경우, 각 센서의 데이터를 독립적으로 취급하였으며, 결과적으로 낮은 성능을 보임.
  
3. Model Desing에 대한 절제(Ablation Studies) 실험 지표
   - w/o PI: Privilegd Information(특권 정보) 없는 상태로, Historical Data만 사용하여 Teacher Model 학습
   - w/o CA: Calibrated Attention 없는 상태로, 원래의 Multi-Head Attention 매커니즘을 사용하여 학습
   - w/o CLM: Calibrated Language Model 없는 상태로 학습. (Teacher Model은 Textual Prompt에 대한 LLM의 효율적 처리 능력을 이용하지 못함.)
   - w/o SCA: Subtractive Cross Attention 없는 상태로, Direct Subtraction을 사용하여 학습.
   - w/o CD: Correlation Distillation(상관관계 증류) 없는 상태로, Privileged Transformer 및 Time series Transformer의 직접적인 Interaction 을 제거하여 학습
   - w/o FD: Feature Distilation 없이, Privileged Transformer 와 Time Series Transformer의 Output을 그대로 사용

4. Model Desing에 대한 절제(Ablation Studies) 실험 결과
   - 모든 데이터셋에서, CD와 PI가 없는 모델보다 있는 모델의 성능이 더 높게 측정됨. => CD 및 PI가 시계열 예측에 효과적으로 작용함.
   - SCA 있는 모델은 없는 모델에 비해, 최대 8.2% 더 낮은 MSE와 최대 6.5% 더 낮은 MAE를 보임.
   - CA 있는 모델은 없는 모델에 비해, 최소 8.9% 더 낮은 MSE와 최소 8.4% 더 낮은 MAE를 보임.
   - CLM을 제거한 모델이, 모든 비교 대상 중 가장 낮은 성능을 보임. (-> CLM이 모델 전체 성능에 가장 크게 기여함.)
  
5. Open-Source LLM에 대한 절제 실험 
   - TimeKD에 BERT, GPT-2, LLaMA-3.2를 적용하여 비교 실험 수행함.
   - 성능은 LLaMA-3.2가 가장 우수하며, 그 다음으로 GPT-2의 성능이 우수함. BERT의 성능이 가장 낮은 것으로 측정됨
   - 모델이 클수록 더 복잡한 언어 패턴을 잘 포착하여, 지식 증류(Distillation)에 더 유리하기 때문인 것으로 보임.
   - 하지만, LLaMA-3.2의 성능이 가장 높았으나, GPT-2와 비교하였을 때, 정확도 차이는 미미한 반면, 메모리 사용량과 추론 시간(Computing Cost)은 훨씬 높았음.
   - Computational Cost를 고려할 때, TimeKD 모델의 기본 LLM으로 GPT-2를 사용하는 것이 가장 유리하다고 판단함.
  
6. 리소스 효율성
   - 메모리 사용량 및 추론 속도: TimeKD는 모든 모델 중에서 가장 적은 메모리를 사용하고, 추론 속도 또한 가장 빨랐음.
   - 교사 모델의 last Token Etractor 설계와 학생 모델의 하이퍼파라미터 단순화 덕분인 것으로 보임.
   - 학습 파라미터 수 및 학습 시간: LLM 기반 모델 중, TimeKD의 학습 가능한 파라미터 수가 가장 적고, 학습 시간도 가장 짧음.

7. 확장성 (Scalability)
   - 훈련 데이터의 양을 20%에서 100%까지 늘리자, 모든 데이터셋에서 오차(MSE, MAE)가 꾸준히 감소함.
   - 이는 TimeKD가 추가적인 데이터를 효과적으로 활용하여 예측 정확도를 높인다는 것을 의미함. (다양한 데이터 규모에 대한 적응력과 확장력이 뛰어남)
   - TimeKD는 데이터가 제한된 상황에서도 확장성이 뛰어나, 데이터 가용성이 다양한 실제 환경에 적용하기에 적합.

8. 퓨샷 예측 (Few-Shot Forecasting)
   - TimeKD는 제한된 데이터 환경에서, 다른 모든 경쟁 모델보다 더 안정적이고 우수한 성능을 보임.
   - TimeCMA보다 최대 3.18% 낮은 MSE, 최대 9.79% 낮은 MAE를 기록함. 이는 데이터가 부족한 상황에서 "지식 증류"가 중요함을 의미함.
   - 전반적으로, LLM 기반 모델들이 Transformer 기반 모델보다 더 나은 성능을 보임. (LLM 기반 모델은 LLM이 사전에 훈련한 지식을 기반으로 하기 때문임.)
  
9. 제로샷 예측 (Zero-Shot)
   - TimeKD는 지속적으로 더 나은 성능을 보임. TimeCMA 보다 최대 9.15% 낮은 MSE, 최대 11.4% 더 낮은 MAE를 기록함.
   - TimeKD는 특권적 지식 증류(Privileged knowledge distillation)를 통해 여러 데이터셋으로 학습한 지식을 효과적으로 Student 에게 전달함.
   - 다양한 데이터셋에서 공통적인 시계열 패턴을 추출하여, 특정 데이터셋을 직접 훈련하지 않고도 일반화된 예측을 잘 수행함.

10. 어텐션 맵 시각화 (Attention Maps Visualization)
    - 모델의 서로 다른 두 가지 트랜스포머가 변수 간의 관계를 어떻게 파악하는지 시각화함.
    - LLM 기반 어텐션 (Privileged Attention): LLM의 사전 훈련 지식과 프롬프트 정보를 활용하여, 변수 사이의 보편적이고 전역적인(Global) 의존성 포착
    - 시계열 트랜스포머 어텐션: 특정 변수에 초점을 맞춘 지역적(Local) 의존성 포착
    - TimeKD는 전역적 관계를 시계열 트랜스포머(Local)에 증류(Distillaion)하는 방식으로 작동함. (지역적 정보 + 전역적 정보를 모두 활용)

11. 특징 시각화 (Feature Visualization)
    - 두 트랜스포머가 추출한 특징들이, 변수 사이에 어떤 상호작용 패턴을 보이는지 시각화.
    - LLM 기반 특징: LLM의 Global한 문맥 지식을 활용하여, 변수 사이의 관계가 더 포괄적이고 균형 잡힘을 확인할 수 있음.
    - 시계열 트랜스포머 특징: 지역적(Local)으로 집중된 패턴을 보여주며, 이는 모델이 지역적인 시계열 의존성에 중점을 두는 것을 의미함.
    - TimeKD는 이 두 가지의 특징을 결합하여 활용함.

12. 실제값과 예측값 비교 (Ground Truth vs. Prediction)
    - ETTh1 데이터셋의 HUFL, MUFL, LUFL, OT 변수의 실제값과 예측값을 그래프로 표현.
    - 예측 값인 파란색 곡선이, 실제 값인 주황색 곡선을 매우 유사하게 따라감을 확인할 수 있음.
    - TimeKD가 데이터의 주기적인 경향과 시간에 따른 변동을 잘 학습하고 보존하며, 모델이 여러 시계열 변수에 거쳐 정확한 예측을 생성할 수 있음을 의미함.

-----

### VII. Conclusion
- TimeKD: 다변량 시계열 예측(MTSF) 프레임워크.
- 보정된 언어 모델과 특권적 지식 증류 기술을 통합한 새로운 모델.
  
#### 주요 구성 요소
1. 교사 모델: 보정된 언어 모델 (CLM) + 차감 교차 어텐션 (SCA)
- Calibrated Language Model: LLM의 사전 훈련 지식과 "특권적 텍스트 프롬프트"를 활용하여 미래에 대한 강력한 표현 추출
- Subtractive Cross-Attention: 추출된 표현을 시계열 데이터에 적합하도록 정제
1. 학생 모델: 교사 모델로부터 지식을 전달받아 가볍고 효율적으로 작동하는 모델

### 학습 방법: 특권적 지식 증류
- TimeKD는 상관관계 증류와 특징 증류를 포함한 혁신적인 "특권적 지식 증류" 방식을 제안함.
- 특히, 다변량 시계열 데이터 예측을 위해 오픈소스 LLM에 특권적 지식 증류를 적용한 최초의 사례.
- 실제 데이터셋을 통한 광범위한 실험으로, TimeKD의 효과와 효율성을 입증함.