DTW(동적 시간 워핑, Dynamic Time Warping)은 길이가 다르거나 시계열 데이터의 속도가 일정하지 않을 때, 두 시계열 데이터 간의 유사성을 측정하는 알고리즘

Q. "범용(Universal)" 프레임워크가 되려면 해결해야 할 가장 큰 기술적 난제는 무엇이며, 어떻게 접근해야 하는가?

A: 가장 큰 난제는 **도메인 이동(Domain Shift)**과 다양한 주기(Frequency) 처리입니다. (예: 주식 데이터 vs 심전도 데이터)

접근법:

입력 정규화(Normalization): RevIN(Reversible Instance Normalization) 같은 기법을 필수적으로 사용하여 데이터의 통계적 분포 차이를 제거해야 합니다.

주파수 도메인 활용: 시간 도메인뿐만 아니라 Fourier Transform 등을 통해 주파수 도메인 특징을 함께 학습하면 주기성이 다른 데이터에 더 강건해질 수 있습니다.

가변 패치: 고정된 패치 길이 대신 다양한 크기의 패치를 동시에 학습(Multi-scale patching)하여 다양한 패턴을 포착해야 합니다.



Q. 하나의 임베딩으로 예측(Forecasting), 분류(Classification), 이상 탐지(Anomaly Detection)를 모두 수행하려면 아키텍처를 어떻게 구성해야 하는가? 

A: [Pre-training] -> [Universal Embedding] -> [Lightweight Heads] 구조가 필요합니다.백본(Backbone): PatchTST나 LLM 기반 인코더를 사용하여 데이터의 문맥을 압축한 고차원 벡터(z)를 생성합니다. 이때 인코더는 특정 태스크에 종속되지 않도록 마스킹(Masking)이나 대조 학습(Contrastive Learning)으로 훈련합니다.헤드(Heads): 생성된 임베딩 $z$ 위에 태스크별로 아주 얇은 레이어(Linear Layer 등)만 붙입니다.예측: $z \rightarrow$ Linear $\rightarrow$ 미래 시점 값분류: $z \rightarrow$ Pooling $\rightarrow$ Linear $\rightarrow$ 클래스 확률이상 탐지: $z \rightarrow$ Reconstruction $\rightarrow$ 입력과의 차이 계산


Q. 학부생/석사생 수준의 자원(GPU)으로 aLLM4TS 같은 거대 모델 기반 프레임워크를 효율적으로 실험하려면?

A: PEFT(Parameter-Efficient Fine-Tuning) 기법과 **지식 증류(Knowledge Distillation)**를 적극 활용해야 합니다.

LLM 전체를 학습시키는 것은 불가능하므로, **LoRA(Low-Rank Adaptation)**나 Adapter 방식을 사용하여 훈련 파라미터 수를 0.1% 수준으로 줄여야 합니다.

또는, 거대 모델(Nexus Oracle)은 추론만 하여 '정답지(Soft label)'를 만들고, 실제 서비스용 모델(Prism Core)은 PatchTST 같은 가벼운 모델로 만들어 거대 모델의 지식을 배우게 하는(Distillation) 전략이 현실적입니다.

Q. PyTorch 모델 클래스(nn.Module) 안에서 '공유 백본(Shared Backbone)'과 '개별 헤드(Separate Heads)'를 어떻게 구성해야 데이터 흐름이 꼬이지 않을까?

A: __init__에서는 인코더를 하나만 정의하고, 헤드를 별도로 정의합니다. 핵심은 forward 함수에서의 **분기(Branching)**입니다.

구조: self.encoder (PatchTST Backbone)는 하나만 존재합니다. self.head_forecast (Linear)와 self.head_class (Pooling + Linear)를 따로 만듭니다.

흐름: forward(x) 함수에서 먼저 z = self.encoder(x)로 잠재 표현(Latent Representation)을 얻습니다. 그 후 이 z를 두 헤드에 각각 통과시킵니다.

Python

# 예시 코드
def forward(self, x):
    # 1. 공통 인코딩
    z = self.encoder(x) # Shape: [Batch, Patch_Num, D_model]
    # 2. 예측 태스크 (Flatten -> Linear)
    pred_out = self.head_forecast(z.flatten(1))
    # 3. 분류 태스크 (Pooling -> Linear)
    cls_feat = z.mean(dim=1) # Global Average Pooling
    cls_out = self.head_class(cls_feat)
    return pred_out, cls_out
```

Q. 예측(Forecasting)과 분류(Classification)는 요구하는 임베딩의 성격이 다른데, 인코더가 혼란스러워하지 않을까? (Negative Transfer 문제)

A: 맞습니다. 예측은 **'지역적이고 세밀한 변화(Local details)'**가 중요하고, 분류는 **'전체적인 형상과 특징(Global shape)'**이 중요합니다. 이를 해결하기 위해 헤드 디자인을 차별화해야 합니다.

분류 헤드: 전체 시퀀스의 정보를 압축하는 Global Average Pooling이나 Max Pooling을 사용하여 인코더가 전체 맥락을 보도록 유도합니다.

예측 헤드: 모든 패치의 정보를 살려야 하므로 Pooling 없이 Flatten 하거나, 마지막 패치(Last Patch)에 가중치를 더 주는 방식을 사용합니다.


Q. 두 태스크의 손실 함수(Loss Function) 단위가 다른데, 어떻게 하나로 합쳐서 역전파(Backpropagation)를 시켜야 할까?

A: 가중 합(Weighted Sum) 손실 함수를 설계해야 합니다. 예측은 주로 MSE(평균 제곱 오차), 분류는 Cross-Entropy를 사용합니다.$$Loss_{total} = \lambda_1 \cdot MSE(y_{pred}, y_{true}) + \lambda_2 \cdot CE(c_{pred}, c_{true})$$여기서 $\lambda$(람다)는 하이퍼파라미터입니다. 초기에는 예측 손실값이 분류 손실값보다 훨씬 클 수 있으므로, 두 손실값의 스케일(Scale)을 비슷하게 맞춰주는 튜닝이 필요합니다. (예: 예측 Loss가 100이고 분류 Loss가 1이면, $\lambda_1=0.01$로 설정)


Q. 하나의 배치(Batch)로 학습할 때, 라벨이 없는 데이터는 어떻게 처리하나? (예: 예측용 데이터는 있는데 분류 라벨이 없는 경우)

A: 현실적인 문제입니다. 이때는 **'마스킹된 손실 계산(Masked Loss Calculation)'**을 사용합니다.

분류 라벨이 없는 데이터가 들어오면, 해당 배치의 분류 Loss는 0으로 처리하여 그래디언트(Gradient)가 업데이트되지 않도록 막아야 합니다.

코드 레벨에서는 loss_cls = criterion(pred, target) * mask 형태로 구현하여, 라벨이 있는 데이터만 학습에 반영되게 합니다.