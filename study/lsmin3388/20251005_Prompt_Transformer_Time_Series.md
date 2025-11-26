# Transformer in Time Series

## Prompts (Powered by Claude)

### Question)
```
Transformer의 기본 구조에 대해 설명해주세요. 특히 인코더-디코더 구조가 시계열 예측에서 어떻게 활용되나요?
```

### Answer)
Transformer는 2017년 "Attention is All You Need" 논문에서 제안된 구조로, 크게 인코더(Encoder)와 디코더(Decoder)로 구성됩니다.

**기본 구조:**
- **인코더**: 입력 시퀀스를 받아 Self-Attention과 Feed-Forward Network를 통해 컨텍스트 정보를 학습합니다. 여러 인코더 레이어를 쌓아 깊은 표현을 학습할 수 있습니다.
- **디코더**: 인코더의 출력과 이전 출력을 입력으로 받아 다음 시퀀스를 생성합니다. Masked Self-Attention을 사용하여 미래 정보 누출을 방지합니다.

**시계열 예측에서의 활용:**
- **인코더**: 과거 시계열 데이터(lookback window)를 입력으로 받아 패턴과 추세를 학습합니다.
- **디코더**: 인코더가 학습한 정보를 바탕으로 미래 시점의 값을 순차적으로 예측합니다.
- **Self-Attention 메커니즘**: 시계열의 장기 의존성(long-range dependency)을 효과적으로 포착할 수 있습니다. RNN과 달리 병렬 처리가 가능하여 학습 속도가 빠릅니다.

시계열 예측에서는 입력 시퀀스의 길이만큼 인코더로 과거를 인코딩하고, 예측하고자 하는 미래 시점만큼 디코더가 출력을 생성하는 방식으로 활용됩니다.

---

### Question)
```
시계열 데이터에 Transformer를 적용할 때의 장점과 한계는 무엇인가요?
```

### Answer)
**장점:**

1. **장기 의존성 포착**: Self-Attention 메커니즘을 통해 멀리 떨어진 시점 간의 관계를 직접적으로 학습할 수 있습니다. RNN/LSTM의 gradient vanishing 문제를 회피합니다.

2. **병렬 처리**: RNN과 달리 순차적 계산이 필요 없어 GPU를 활용한 병렬 처리가 가능하며, 학습 속도가 빠릅니다.

3. **다변량 시계열 처리**: 여러 변수 간의 상호작용을 Attention을 통해 자연스럽게 모델링할 수 있습니다.

4. **해석 가능성**: Attention weight를 시각화하여 모델이 어느 시점에 주목하는지 분석할 수 있습니다.

**한계:**

1. **계산 복잡도**: Self-Attention의 시간 복잡도가 O(L²)로, 긴 시퀀스에서 메모리와 계산 비용이 급격히 증가합니다(L은 시퀀스 길이).

2. **데이터 요구량**: Transformer는 많은 파라미터를 가지고 있어 충분한 학습을 위해 대량의 데이터가 필요합니다.

3. **Position Encoding의 한계**: 시계열의 복잡한 temporal pattern을 단순한 positional encoding으로 표현하기 어려울 수 있습니다.

4. **Overfitting 위험**: 작은 데이터셋에서는 과적합되기 쉽습니다.

이러한 한계를 극복하기 위해 Informer, Autoformer 등의 개선된 모델들이 제안되었습니다.

---

### Question)
```
시계열 예측을 위한 대표적인 Transformer 기반 모델들(Informer, Autoformer 등)에 대해 설명해주세요.
```

### Answer)
시계열 예측을 위해 기존 Transformer의 한계를 개선한 여러 모델들이 제안되었습니다.

**1. Informer (2021, AAAI Best Paper)**
- **핵심 아이디어**: ProbSparse Self-Attention을 통해 계산 복잡도를 O(L log L)로 감소
- **주요 특징**:
  - 중요한 Query만 선택하여 Attention 계산 (상위 u개의 Query만 사용)
  - Self-Attention Distilling을 통해 레이어를 거치며 시퀀스 길이 축소
  - Generative decoder 대신 한 번에 예측하는 방식 사용
- **장점**: 장기 시계열 예측(LSTF: Long Sequence Time-Series Forecasting)에서 효율적

**2. Autoformer (2021, NeurIPS)**
- **핵심 아이디어**: Auto-Correlation 메커니즘 도입
- **주요 특징**:
  - 시계열의 주기성(periodicity)을 명시적으로 모델링
  - Series Decomposition을 각 레이어에 통합 (Trend + Seasonal 분리)
  - Auto-Correlation으로 시간 지연 집계를 학습
- **장점**: 복잡한 주기 패턴을 가진 시계열에서 우수한 성능

**3. FEDformer (2022, ICML)**
- **핵심 아이디어**: 주파수 도메인(Frequency Domain)에서 Attention 수행
- **주요 특징**:
  - Fourier Transform을 사용하여 시계열을 주파수 영역으로 변환
  - 주파수 공간에서 중요한 성분만 선택하여 계산
- **장점**: 계산 효율성과 성능 모두 개선

**4. PatchTST (2023, ICLR)**
- **핵심 아이디어**: 시계열을 패치(patch) 단위로 분할
- **주요 특징**:
  - 이미지의 Vision Transformer(ViT)처럼 시계열을 패치로 나누어 처리
  - Channel Independence: 각 변수를 독립적으로 처리
- **장점**: 최신 모델 중 하나로 다양한 벤치마크에서 SOTA 달성

이러한 모델들은 각각 Transformer의 계산 복잡도, 장기 의존성, 주기성 포착 등 특정 한계를 해결하는 데 초점을 맞추고 있습니다.
