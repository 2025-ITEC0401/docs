# Loss Functions Study

## Prompts (Powered by Claude)

### Question)
```
회귀 문제에서 사용되는 MSE, MAE, Huber Loss의 차이점과 각각 어떤 상황에서 사용하는 것이 적합한지 설명해주세요.
```

### Answer)
회귀 문제에서 사용되는 세 가지 주요 손실 함수는 다음과 같은 특징을 가지고 있습니다:

**MSE (Mean Squared Error)**
- 예측값과 실제값의 차이를 제곱하여 평균을 계산합니다
- 수식: L = (1/n) Σ(y_pred - y_true)²
- 큰 오차에 대해 더 큰 페널티를 부여합니다 (제곱 효과)
- 미분 가능하고 수학적으로 다루기 쉽습니다
- 이상치(outlier)에 민감하여 크게 영향을 받습니다
- 적합한 상황: 이상치가 적고, 큰 오차를 강하게 억제하고 싶을 때

**MAE (Mean Absolute Error)**
- 예측값과 실제값의 절대 차이를 평균합니다
- 수식: L = (1/n) Σ|y_pred - y_true|
- 모든 오차에 동일한 가중치를 부여합니다
- 이상치에 대해 MSE보다 강건(robust)합니다
- 0 근처에서 미분 불가능한 점이 있어 최적화가 까다로울 수 있습니다
- 적합한 상황: 이상치가 많거나, 모든 오차를 동등하게 취급하고 싶을 때

**Huber Loss**
- MSE와 MAE의 장점을 결합한 손실 함수입니다
- 작은 오차에는 MSE처럼, 큰 오차에는 MAE처럼 동작합니다
- 수식: L = { 0.5 × (y_pred - y_true)² if |y_pred - y_true| ≤ δ, δ × |y_pred - y_true| - 0.5 × δ² otherwise }
- δ(delta) 파라미터로 MSE와 MAE 사이의 전환점을 조절합니다
- 미분 가능하면서도 이상치에 강건합니다
- 적합한 상황: 이상치가 있지만 작은 오차는 정밀하게 최적화하고 싶을 때

---

## Prompts (Powered by Claude)

### Question)
```
딥러닝 분류 문제에서 Cross Entropy Loss가 왜 널리 사용되는지, 그리고 Binary Cross Entropy와 Categorical Cross Entropy의 차이점을 설명해주세요.
```

### Answer)
Cross Entropy Loss는 분류 문제에서 가장 널리 사용되는 손실 함수입니다.

**Cross Entropy를 사용하는 이유**

1. **확률 분포 간의 차이를 측정**: 모델이 예측한 확률 분포와 실제 레이블의 확률 분포 사이의 차이를 효과적으로 측정합니다

2. **로그 함수의 특성**: 잘못된 예측에 대해 큰 페널티를 부여하며, 확신 있는 올바른 예측에는 작은 손실을 부여합니다

3. **Softmax와의 조합**: Softmax 활성화 함수와 함께 사용할 때 수치적으로 안정적이고 gradient 계산이 간단합니다

4. **정보 이론적 근거**: 두 확률 분포 간의 차이를 정보 이론 관점에서 정량화합니다

**Binary Cross Entropy (BCE)**
- 이진 분류 문제에서 사용됩니다 (클래스가 2개)
- 수식: L = -[y × log(p) + (1-y) × log(1-p)]
- 출력층에 Sigmoid 활성화 함수를 사용합니다
- 각 샘플이 하나의 클래스에만 속하는 경우에 적합합니다
- 예시: 스팸 메일 분류, 질병 진단 (양성/음성)

**Categorical Cross Entropy (CCE)**
- 다중 클래스 분류 문제에서 사용됩니다 (클래스가 3개 이상)
- 수식: L = -Σ(y_i × log(p_i)) for all classes i
- 출력층에 Softmax 활성화 함수를 사용합니다
- 레이블이 원-핫 인코딩 형태로 제공됩니다
- 각 샘플이 정확히 하나의 클래스에만 속합니다
- 예시: 손글씨 숫자 인식 (0-9), 이미지 분류 (고양이/개/새)

**핵심 차이점**
- BCE는 출력 노드가 1개 (확률 p), CCE는 클래스 수만큼의 출력 노드
- BCE는 독립적인 이진 결정, CCE는 상호 배타적인 다중 클래스 결정
- Multi-label 분류(한 샘플이 여러 클래스에 속할 수 있음)에서는 BCE를 클래스마다 적용합니다

---

## Prompts (Powered by Claude)

### Question)
```
Contrastive Learning에서 사용되는 Triplet Loss와 InfoNCE Loss의 작동 원리와 차이점을 설명해주세요. 각각 어떤 응용 분야에 적합한가요?
```

### Answer)
Contrastive Learning은 유사한 샘플은 가깝게, 다른 샘플은 멀게 임베딩하는 학습 방법입니다.

**Triplet Loss**

작동 원리:
- 세 개의 샘플을 사용합니다: Anchor(기준), Positive(같은 클래스), Negative(다른 클래스)
- 수식: L = max(0, d(a, p) - d(a, n) + margin)
  - d(a, p): anchor와 positive 사이의 거리
  - d(a, n): anchor와 negative 사이의 거리
  - margin: 최소 분리 거리
- 목표: positive는 anchor에 가깝게, negative는 margin 이상 멀게 배치

특징:
- 직관적이고 이해하기 쉬운 구조
- Triplet 샘플링이 중요 (hard negative mining 등)
- 샘플링 전략에 따라 성능이 크게 달라짐
- 배치 크기에 상대적으로 덜 민감함

**InfoNCE Loss (Noise Contrastive Estimation)**

작동 원리:
- 하나의 positive 샘플과 여러 negative 샘플을 동시에 고려합니다
- 수식: L = -log[exp(sim(a, p)/τ) / (exp(sim(a, p)/τ) + Σexp(sim(a, n_i)/τ))]
  - sim: 유사도 함수 (주로 cosine similarity)
  - τ (tau): temperature 파라미터
  - n_i: 여러 negative 샘플들
- 목표: positive를 모든 negative들 중에서 구분하는 분류 문제로 변환

특징:
- 배치 내 모든 negative를 활용하여 효율적
- Temperature 파라미터로 학습 난이도 조절 가능
- 대규모 배치 크기에서 더 좋은 성능 (더 많은 negative)
- SimCLR, MoCo 등 최신 self-supervised learning에서 널리 사용

**주요 차이점**

1. **비교 대상 수**: Triplet은 1:1 비교, InfoNCE는 1:N 비교
2. **정보 활용**: InfoNCE가 배치 내 정보를 더 효율적으로 활용
3. **계산 효율성**: InfoNCE가 GPU 병렬화에 더 유리
4. **안정성**: InfoNCE가 일반적으로 더 안정적인 학습

**응용 분야**

Triplet Loss 적합:
- 얼굴 인식 (Face Recognition): FaceNet 등
- 행동 재식별 (Person Re-identification)
- 이미지 검색 (Image Retrieval)
- 메트릭 학습이 필요한 소규모 데이터셋

InfoNCE Loss 적합:
- Self-supervised Learning: SimCLR, MoCo
- 대규모 비전 모델 사전 학습
- Contrastive Language-Image Pre-training (CLIP)
- 대량의 unlabeled 데이터 활용이 가능한 경우
- 멀티모달 학습 (이미지-텍스트 매칭 등)

현대 딥러닝에서는 InfoNCE와 그 변형들이 더 널리 사용되는 추세이며, 특히 자기 지도 학습(self-supervised learning) 분야에서 큰 성공을 거두고 있습니다.
