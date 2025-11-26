# Autoencoder 학습 자료

## Prompts (Powered by Claude)

### Question)
```
Autoencoder의 기본 구조에 대해 설명해주세요. 특히 Encoder, Decoder, Bottleneck의 역할과 동작 원리를 중심으로 설명해주세요.
```

### Answer)
Autoencoder는 입력 데이터를 압축했다가 다시 복원하는 비지도 학습 신경망입니다. 세 가지 핵심 구성요소로 이루어져 있습니다.

**1. Encoder (인코더)**
- 입력 데이터를 저차원의 잠재 표현(latent representation)으로 압축하는 역할
- 여러 층의 신경망을 통해 점진적으로 차원을 축소
- 입력 데이터의 중요한 특징을 추출하고 불필요한 정보는 제거
- 수식: z = f(x), 여기서 x는 입력, z는 잠재 벡터

**2. Bottleneck (병목층)**
- Encoder의 출력이자 가장 압축된 표현
- 데이터의 가장 본질적인 특징만을 담고 있는 저차원 공간
- 이 층의 차원 크기가 압축률을 결정
- 차원 축소와 특징 추출의 핵심 역할

**3. Decoder (디코더)**
- Bottleneck의 잠재 표현을 원본 데이터와 유사한 형태로 복원
- Encoder와 대칭적인 구조로 점진적으로 차원을 확장
- 수식: x' = g(z), 여기서 x'는 복원된 출력

**동작 원리:**
- 목적 함수: 입력 x와 출력 x' 사이의 재구성 오차(reconstruction error)를 최소화
- Loss = ||x - x'||²  (MSE 기준)
- 학습 과정에서 중요한 특징을 보존하면서 노이즈는 제거하는 방향으로 가중치가 업데이트됨

---

### Question)
```
일반 Autoencoder와 VAE(Variational Autoencoder)의 차이점은 무엇인가요? 각각의 장단점과 사용 사례를 비교해주세요.
```

### Answer)
일반 Autoencoder와 VAE는 구조적으로 유사하지만 중요한 차이점들이 있습니다.

**일반 Autoencoder**
- Bottleneck: 결정론적(deterministic) 잠재 벡터 생성
- 입력 → 고정된 하나의 잠재 벡터 → 출력
- 목적: 재구성 오차 최소화

**VAE (Variational Autoencoder)**
- Bottleneck: 확률 분포(평균 μ, 분산 σ²)를 학습
- 입력 → 분포의 파라미터 → 샘플링 → 출력
- 목적: 재구성 오차 + KL Divergence 최소화
- Reparameterization trick 사용: z = μ + σ * ε (ε ~ N(0,1))

**주요 차이점:**

1. **잠재 공간의 특성**
   - Autoencoder: 불연속적, 구조화되지 않은 잠재 공간
   - VAE: 연속적, 정규화된 잠재 공간 (보통 정규분포)

2. **생성 능력**
   - Autoencoder: 생성 모델로 사용하기 어려움 (잠재 공간이 불규칙)
   - VAE: 잠재 공간에서 샘플링하여 새로운 데이터 생성 가능

3. **손실 함수**
   - Autoencoder: L = ||x - x'||²
   - VAE: L = ||x - x'||² + KL(q(z|x) || p(z))

**장단점 비교:**

| 측면 | Autoencoder | VAE |
|------|-------------|-----|
| 재구성 품질 | 높음 | 상대적으로 낮음 (흐릿함) |
| 생성 능력 | 약함 | 강함 |
| 학습 안정성 | 높음 | KL vanishing 등 이슈 |
| 이상 탐지 | 적합 | 적합하나 복잡 |

**사용 사례:**
- Autoencoder: 차원 축소, 노이즈 제거, 이상 탐지, 특징 추출
- VAE: 이미지 생성, 데이터 증강, 반정형 학습, 약물 설계

---

### Question)
```
시계열 데이터의 이상 탐지(Anomaly Detection)에 Autoencoder를 어떻게 활용할 수 있나요? 구체적인 방법과 임계값 설정 전략을 설명해주세요.
```

### Answer)
Autoencoder는 시계열 이상 탐지에서 강력한 도구로 활용됩니다. 정상 데이터의 패턴을 학습하고, 그 패턴에서 벗어난 데이터를 이상으로 판단하는 원리입니다.

**기본 원리:**
1. 정상 데이터로만 Autoencoder를 학습
2. 학습된 모델은 정상 패턴을 잘 재구성
3. 이상 데이터는 재구성이 잘 되지 않아 큰 오차 발생
4. 재구성 오차를 기반으로 이상 여부 판단

**시계열 이상 탐지 구현 방법:**

**1. 데이터 전처리**
```python
# 윈도우 슬라이딩 기법
window_size = 100
sequences = []
for i in range(len(data) - window_size):
    sequences.append(data[i:i+window_size])
```

**2. 모델 구조 설계**
- LSTM/GRU Autoencoder: 시간적 의존성 포착
- Conv1D Autoencoder: 지역적 패턴 학습
- Transformer Autoencoder: 장기 의존성 학습

예시 구조:
```
Encoder: [100, 64, 32, 16] (시계열 길이 100 → 잠재 벡터 16)
Decoder: [16, 32, 64, 100] (잠재 벡터 16 → 복원 100)
```

**3. 이상 점수(Anomaly Score) 계산**

- **방법 1: MSE (Mean Squared Error)**
  ```
  score = (1/n) * Σ(x_i - x'_i)²
  ```

- **방법 2: MAE (Mean Absolute Error)**
  ```
  score = (1/n) * Σ|x_i - x'_i|
  ```

- **방법 3: MAPE (Mean Absolute Percentage Error)**
  ```
  score = (1/n) * Σ|((x_i - x'_i) / x_i)| * 100
  ```

**4. 임계값(Threshold) 설정 전략**

**전략 1: 통계적 방법**
- 정상 데이터의 재구성 오차 분포 분석
- threshold = μ + k*σ (μ: 평균, σ: 표준편차, k: 2~3)
- 장점: 이론적 근거, 단점: 분포 가정 필요

**전략 2: 백분위수(Percentile) 방법**
- threshold = 95th or 99th percentile of reconstruction errors
- 상위 5% 또는 1%를 이상으로 판단
- 장점: 분포 가정 불필요, 단점: 이상 비율 사전 설정

**전략 3: Precision-Recall 곡선**
- 검증 데이터(레이블 있음)로 최적 임계값 탐색
- F1-score가 최대가 되는 지점 선택
- 장점: 성능 최적화, 단점: 레이블 데이터 필요

**전략 4: 동적 임계값**
- 시간 윈도우별로 임계값 조정
- threshold_t = μ_t + k*σ_t (시간 t에서의 통계)
- 장점: 시계열 변화 대응, 단점: 계산 복잡도 증가

**실전 팁:**
1. 여러 임계값을 동시에 사용 (경고/위험 수준 구분)
2. 연속된 여러 시점의 이상을 함께 고려 (False Positive 감소)
3. 도메인 지식과 결합하여 임계값 조정
4. 정기적인 모델 재학습으로 개념 변화(concept drift) 대응

**평가 지표:**
- Precision, Recall, F1-score
- ROC-AUC, PR-AUC
- 탐지 지연 시간(Detection Delay)
