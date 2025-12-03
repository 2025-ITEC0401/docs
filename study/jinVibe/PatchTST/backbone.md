Q1. 트랜스포머(Transformer)는 원래 시계열 데이터에서 과적합(Overfitting)과 연산 비용 문제가 심각했는데, PatchTST는 이를 어떻게 해결했는가

?A: 핵심은 **'패칭(Patching)'**과 **'채널 독립(Channel Independence)'**입니다.패칭: 시점 하나하나(Point-wise)를 토큰으로 쓰지 않고, 일정 구간(예: 16 time-steps)을 묶어 하나의 패치로 만듭니다. 이렇게 하면 입력 시퀀스의 길이를 $N$배 줄여 트랜스포머의 고질적인 $O(L^2)$ 연산 비용을 획기적으로 낮추고, 지역적인 문맥(Local Semantic)을 보존합니다.채널 독립: 다변량 데이터의 모든 채널을 섞어서 학습하지 않고, 각 채널을 독립적인 단변량 시계열처럼 취급하여 하나의 모델(Shared Weights)로 학습시킵니다. 이는 채널 간의 불필요한 노이즈(Correlations)가 학습을 방해하는 것을 막아 일반화 성능을 높입니다.

Q2. 범용 프레임워크를 만들 때 PatchTST의 'Representation Learning(표현 학습)' 능력은 왜 중요한가?

A: PatchTST는 마스킹(Masking) 기법을 통한 자기 지도 학습(Self-Supervised Learning)에 매우 강력합니다. 데이터의 일부를 가리고 복원하는 사전 학습(Pre-training)을 통해, 라벨이 없는 데이터에서도 시계열의 고유한 패턴을 학습할 수 있습니다. 이렇게 학습된 **임베딩(Embedding)**은 예측뿐만 아니라 분류, 이상 탐지 등 다른 다운스트림 태스크로 전이(Transfer)했을 때도 우수한 성능을 보장하므로, 범용 프레임워크의 '공통 인코더'로서 적합합니다.