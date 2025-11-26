# Contrastive Learning for Time Series Data

## Prompts (Powered by Claude)

### Question)
```
시계열 데이터에서 Contrastive Learning이란 무엇이며,
왜 시계열 분석에 효과적인가요?
전통적인 supervised learning과의 차이점도 설명해주세요.
```

### Answer)
Contrastive Learning은 self-supervised learning의 한 방법으로, 레이블 없이 데이터 자체의 구조를 학습하는 기법입니다. 시계열 데이터에서는 같은 시계열의 다른 부분(또는 augmented view)을 positive pair로, 다른 시계열을 negative pair로 설정하여 representation을 학습합니다.

**시계열에서의 효과성:**
- **레이블 부족 문제 해결**: 시계열 데이터는 레이블링 비용이 높은데, contrastive learning은 레이블 없이 학습 가능
- **시간적 패턴 학습**: 시계열의 temporal dynamics와 invariant features를 동시에 학습
- **전이 학습 가능**: 학습된 representation은 다양한 downstream task에 활용 가능

**Supervised Learning과의 차이:**
- Supervised: 레이블이 필요하며, 특정 task에 특화된 학습
- Contrastive: 레이블 불필요, 범용적인 representation 학습 후 fine-tuning 가능

---

### Question)
```
Contrastive Learning에서 Positive Pair와 Negative Pair를
어떻게 구성하나요? 시계열 데이터에서의 구체적인 예시와 함께
데이터 augmentation 기법도 설명해주세요.
```

### Answer)
Contrastive Learning의 핵심은 유사한 샘플(positive pair)은 가깝게, 다른 샘플(negative pair)은 멀리 배치하는 것입니다.

**Positive Pair 구성 방법:**
1. **Same instance augmentation**: 같은 시계열에 서로 다른 augmentation 적용
2. **Temporal subseries**: 같은 시계열의 overlapping된 구간들
3. **Contextual positives**: 시간적으로 인접한 구간들

**Negative Pair 구성 방법:**
- 다른 시계열 샘플들 (배치 내의 다른 인스턴스)
- 시간적으로 먼 구간들 (경우에 따라)

**시계열 Data Augmentation 기법:**
- **Jittering**: 노이즈 추가 (x_aug = x + ε, ε ~ N(0, σ²))
- **Scaling**: 진폭 변경 (x_aug = α · x)
- **Time warping**: 시간축 왜곡으로 속도 변화 시뮬레이션
- **Cropping**: 시계열 일부 구간 추출
- **Permutation**: 시간 순서 일부 섞기
- **Masking**: 특정 구간을 0 또는 평균값으로 대체

**예시:**
```
원본 시계열: [1.0, 1.2, 1.5, 1.3, 1.1, 0.9, 1.0]
Jittering:   [1.1, 1.1, 1.6, 1.4, 1.0, 0.8, 1.1]  → Positive
Scaling:     [1.5, 1.8, 2.25, 1.95, 1.65, 1.35, 1.5] → Positive
다른 샘플:    [0.2, 0.3, 0.1, 0.4, 0.2, 0.3, 0.1]  → Negative
```

---

### Question)
```
InfoNCE Loss란 무엇이며, 어떻게 동작하나요?
수식과 함께 설명하고, 시계열 모델 TS2Vec과 TNC에서
어떻게 활용되는지 구체적으로 설명해주세요.
```

### Answer)
InfoNCE (Noise Contrastive Estimation) Loss는 contrastive learning에서 가장 널리 사용되는 손실 함수입니다.

**InfoNCE Loss 수식:**

```
L_InfoNCE = -log( exp(sim(z_i, z_i^+) / τ) / (exp(sim(z_i, z_i^+) / τ) + Σ_j exp(sim(z_i, z_j^-) / τ)) )
```

여기서:
- `z_i`: anchor sample의 representation
- `z_i^+`: positive sample의 representation
- `z_j^-`: negative samples의 representation
- `sim(·,·)`: 유사도 함수 (주로 cosine similarity)
- `τ`: temperature parameter (0.07~0.5, 작을수록 hard negative에 집중)

**동작 원리:**
1. Positive pair의 유사도는 최대화 (분자)
2. Negative pair의 유사도는 최소화 (분모의 합)
3. Temperature τ로 softmax의 sharpness 조절

**TS2Vec (Time Series to Vector):**
- **계층적 contrastive learning** 적용
- Timestamp-level과 instance-level에서 모두 contrastive loss 계산
- **Temporal augmentation**: 같은 시계열의 다른 시간 구간을 positive로 활용
- **Contextual consistency**: 시간적으로 인접한 부분들이 유사한 representation을 갖도록 학습

```python
# TS2Vec의 dual contrastive loss
L_total = L_temporal + L_instance
L_temporal = InfoNCE(z_t, z_t', negatives_in_batch)
L_instance = InfoNCE(z_i, z_i', negatives_in_batch)
```

**TNC (Temporal Neighborhood Coding):**
- **시간적 neighborhood 개념** 활용
- Close in time → positive, Far in time → negative
- **Temporal window**: δ 범위 내의 샘플은 positive, 범위 밖은 negative
- **Discriminative task**: 두 시계열 구간이 temporal neighbor인지 판별

```python
# TNC의 접근
if |t_i - t_j| < δ:
    # Positive pair
    L += -log(σ(f(x_i)^T f(x_j)))
else:
    # Negative pair
    L += -log(1 - σ(f(x_i)^T f(x_j)))
```

**주요 차이점:**
- **TS2Vec**: 다양한 augmentation 활용, 범용적 representation 학습
- **TNC**: 시간적 근접성에 기반, 시계열의 temporal structure에 집중

**대표적인 활용 사례:**
- Anomaly detection: 정상 패턴 학습 후 이상치 탐지
- Classification: Pre-training 후 fine-tuning으로 적은 레이블로 학습
- Forecasting: 과거 패턴 representation을 활용한 미래 예측

---

## References
- TS2Vec: "TS2Vec: Towards Universal Representation of Time Series" (AAAI 2022)
- TNC: "Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding" (ICLR 2021)
- SimCLR: "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)
