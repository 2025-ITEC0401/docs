# Normalization 기법

## Prompts (Powered by Claude)

### Question)
```
딥러닝에서 Batch Normalization과 Layer Normalization의 차이점을 설명해주세요. 각각 어떤 차원에서 정규화를 수행하며, 어떤 장단점이 있나요?
```

### Answer)
Batch Normalization(BN)과 Layer Normalization(LN)은 정규화를 수행하는 차원이 다릅니다.

**Batch Normalization**
- 배치 차원을 따라 정규화를 수행합니다
- 같은 feature의 배치 내 모든 샘플들에 대해 평균과 분산을 계산
- 입력 형태가 (N, C, H, W)일 때, N(배치) 차원에 걸쳐 정규화
- 장점: 학습 안정화, 높은 학습률 사용 가능, Internal Covariate Shift 감소
- 단점: 배치 크기에 의존적, 작은 배치에서 성능 저하, 추론 시 이동 평균 필요

**Layer Normalization**
- Feature 차원을 따라 정규화를 수행합니다
- 각 샘플의 모든 feature들에 대해 평균과 분산을 계산
- 입력 형태가 (N, C, H, W)일 때, C(채널) 차원에 걸쳐 정규화
- 장점: 배치 크기에 독립적, RNN/Transformer에 적합, 추론 시 추가 계산 불필요
- 단점: CNN에서는 BN보다 성능이 낮을 수 있음

핵심 차이는 BN은 "배치 내 다른 샘플들"을 비교하여 정규화하고, LN은 "한 샘플 내 다른 특성들"을 비교하여 정규화한다는 점입니다.

---

### Question)
```
시계열 데이터나 Transformer 모델에서는 왜 Batch Normalization보다 Layer Normalization을 주로 사용하나요? 구체적인 이유를 설명해주세요.
```

### Answer)
시계열 데이터와 Transformer에서 Layer Normalization을 선호하는 이유는 다음과 같습니다:

**1. 시퀀스 길이의 가변성**
- 시계열 데이터는 샘플마다 길이가 다를 수 있습니다
- BN은 배치 내 샘플들이 같은 위치에서 통계를 계산하는데, 길이가 다르면 문제가 발생
- LN은 각 샘플을 독립적으로 처리하므로 이 문제가 없습니다

**2. 배치 크기 의존성**
- RNN/Transformer 학습 시 메모리 제약으로 작은 배치를 사용하는 경우가 많습니다
- BN은 작은 배치에서 통계량 추정이 불안정해집니다
- LN은 배치 크기와 무관하게 동작합니다

**3. 시간적 의존성 문제**
- 시계열의 경우, 시간 step마다 분포가 다를 수 있습니다
- BN을 시간 차원에 적용하면 미래 정보가 과거에 영향을 줄 수 있습니다
- LN은 각 시간 step을 독립적으로 정규화하여 이 문제를 방지합니다

**4. Attention 메커니즘과의 호환성**
- Transformer의 self-attention은 입력 순서에 민감합니다
- LN은 각 토큰의 embedding 차원에 대해 정규화하여 attention 계산을 안정화
- BN은 배치 내 다른 문장의 같은 위치 토큰을 비교하는데, 이는 의미론적으로 부적절합니다

**5. 추론 시의 일관성**
- BN은 학습 시와 추론 시 동작이 다릅니다 (이동 평균 사용)
- LN은 학습과 추론이 동일하게 동작하여 더 일관된 결과를 제공합니다

이러한 이유로 BERT, GPT, T5 등 대부분의 Transformer 기반 모델은 Layer Normalization을 표준으로 채택하고 있습니다.

---

### Question)
```
Instance Normalization과 Group Normalization에 대해 설명하고, 각각 어떤 상황에서 유용한지 알려주세요.
```

### Answer)
Instance Normalization과 Group Normalization은 BN과 LN의 한계를 보완하기 위해 제안된 기법들입니다.

**Instance Normalization (IN)**
- 각 샘플의 각 채널마다 독립적으로 정규화를 수행합니다
- 입력 형태 (N, C, H, W)에서 H×W 차원에 대해 정규화
- 즉, 배치와 채널 모두 독립적으로 처리됩니다

사용 사례:
- **Style Transfer**: 이미지의 스타일 정보가 feature 통계량에 담겨있어, IN이 스타일을 제거하고 콘텐츠를 보존
- **GAN**: 각 이미지를 독립적으로 처리하여 생성 품질 향상
- 배치 간 독립성이 중요한 경우

**Group Normalization (GN)**
- 채널을 여러 그룹으로 나누고, 각 그룹 내에서 정규화를 수행합니다
- BN과 LN의 중간 형태로 볼 수 있습니다
- 예: 32개 채널을 8개 그룹으로 나누면, 각 그룹(4채널)마다 정규화

사용 사례:
- **작은 배치 크기**: BN의 대안으로, 배치 크기에 독립적
- **Object Detection/Segmentation**: Mask R-CNN 등에서 효과적
- **비디오 처리**: 배치 크기가 작은 경우가 많아 GN이 유리
- **분산 학습**: 각 GPU의 배치가 작을 때 BN보다 안정적

**정규화 기법 비교**
- Batch Norm: 배치 × 공간 차원
- Layer Norm: 채널 × 공간 차원
- Instance Norm: 공간 차원만 (채널별로)
- Group Norm: 그룹 × 공간 차원

**선택 가이드**
- 큰 배치 + CNN: Batch Normalization
- 시계열/Transformer/작은 배치: Layer Normalization
- Style Transfer/GAN: Instance Normalization
- Object Detection/작은 배치 CNN: Group Normalization

Group Normalization은 특히 컴퓨터 비전 분야에서 배치 크기 제약이 있을 때 BN의 강력한 대안으로 자리잡고 있습니다.
