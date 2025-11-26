# Self-Supervised Learning

## Prompts (Powered by Claude)

### Question 1)
```
Self-Supervised Learning과 Supervised Learning의 차이점을 설명해주세요.
특히 레이블링의 필요성과 학습 방식의 차이를 중심으로 알려주세요.
```

### Answer 1)
Self-Supervised Learning과 Supervised Learning의 주요 차이점은 레이블 데이터의 필요성과 학습 방식에 있습니다.

**Supervised Learning (지도 학습)**
- 명시적인 레이블이 필요합니다 (예: 이미지 분류를 위한 클래스 레이블)
- 사람이 직접 데이터에 레이블을 부여해야 하므로 비용과 시간이 많이 소요됩니다
- 레이블된 데이터로 직접 목표 태스크를 학습합니다
- 레이블 데이터의 양과 질에 따라 성능이 크게 좌우됩니다

**Self-Supervised Learning (자기 지도 학습)**
- 별도의 사람이 부여한 레이블 없이 데이터 자체에서 supervision signal을 생성합니다
- 데이터의 일부를 가리고(mask) 예측하거나, 데이터의 구조적 특성을 활용합니다
- Pretext Task를 통해 일반적인 표현(representation)을 학습합니다
- 대량의 unlabeled 데이터를 활용할 수 있어 확장성이 뛰어납니다

예를 들어, 이미지의 일부를 가리고 복원하거나, 문장의 일부 단어를 마스킹하고 예측하는 방식으로 데이터 자체가 supervision을 제공합니다.

---

### Question 2)
```
Self-Supervised Learning에서 사용되는 Pretext Task의 개념과 대표적인 예시를 설명해주세요.
특히 Masked Prediction과 Temporal Order 방식을 중심으로 알려주세요.
```

### Answer 2)
Pretext Task는 Self-Supervised Learning에서 레이블 없이 모델이 데이터의 유용한 표현을 학습하도록 설계된 보조 과제입니다. 이를 통해 모델은 데이터의 내재적 구조와 패턴을 이해하게 됩니다.

**1. Masked Prediction (마스크 예측)**

데이터의 일부를 의도적으로 가리고, 가려진 부분을 예측하도록 학습합니다.

- **NLP 분야**: BERT의 Masked Language Modeling (MLM)
  - 문장의 15% 토큰을 [MASK]로 치환
  - 예: "The cat [MASK] on the mat" → "sat" 예측
  - 양방향 컨텍스트를 활용하여 언어의 의미와 문맥 이해

- **Computer Vision 분야**: Masked Autoencoders (MAE)
  - 이미지 패치의 75%를 무작위로 마스킹
  - 마스킹된 패치를 복원하면서 이미지의 구조적 특징 학습

- **Time Series 분야**: PatchTST
  - 시계열 데이터의 특정 시간 구간을 마스킹
  - 마스킹된 구간을 예측하여 시간적 패턴과 의존성 학습

**2. Temporal Order (시간적 순서)**

시간적/공간적 순서 정보를 활용하여 데이터의 구조를 학습합니다.

- **Contrastive Predictive Coding (CPC)**
  - 현재 시점의 표현으로 미래 시점의 표현을 예측
  - 올바른 미래 표현과 무작위 표현을 구분하도록 학습

- **Video Understanding**
  - 프레임의 순서를 섞은 후 올바른 순서를 복원
  - 비디오 클립의 시간적 순서 예측

- **Audio Processing**
  - 오디오 신호의 시간적 순서 관계 학습
  - 이전 프레임으로부터 다음 프레임 예측

**핵심 원리**

Pretext Task의 목표는 해당 태스크 자체를 완벽히 수행하는 것이 아니라, 그 과정에서 downstream task에 유용한 일반적인 특징(feature)을 학습하는 것입니다. 마스킹이나 순서 예측을 통해 모델은 데이터의 본질적인 패턴, 의존성, 구조를 이해하게 됩니다.

---

### Question 3)
```
Self-Supervised Learning으로 사전학습한 모델을 실제 태스크에 어떻게 활용하나요?
Fine-tuning과 Linear Probing의 차이점과 각각의 사용 시나리오를 설명해주세요.
```

### Answer 3)
Self-Supervised Learning으로 사전학습된 모델을 실제 downstream task에 적용하는 두 가지 주요 방법이 있습니다.

**1. Fine-tuning (미세 조정)**

사전학습된 모델의 전체 또는 일부 파라미터를 downstream task에 맞게 추가로 학습시키는 방법입니다.

**특징:**
- 사전학습된 가중치를 초기값으로 사용하여 전체 네트워크를 업데이트
- Backbone network의 파라미터도 조정됨
- 작은 학습률(learning rate)을 사용하여 사전학습된 지식을 보존하면서 미세 조정

**사용 시나리오:**
- 충분한 레이블 데이터가 있을 때 (수천~수만 개 이상)
- Downstream task가 사전학습 데이터와 다소 다를 때
- 최고 성능이 필요한 경우
- 예: BERT를 감성 분석, 개체명 인식 등에 fine-tuning

**2. Linear Probing (선형 평가)**

사전학습된 모델을 feature extractor로 고정하고, 마지막에 새로운 분류 레이어(linear layer)만 학습시키는 방법입니다.

**특징:**
- 사전학습된 모델의 파라미터는 frozen (고정)
- 새로운 linear classifier만 학습
- 학습 파라미터 수가 매우 적음
- 빠른 학습 속도

**사용 시나리오:**
- 레이블 데이터가 매우 적을 때 (수백 개 이하)
- 사전학습 데이터와 downstream task가 유사할 때
- 빠른 프로토타이핑이 필요할 때
- 모델의 표현 품질을 평가할 때
- 계산 리소스가 제한적일 때

**비교 예시:**

| 측면 | Fine-tuning | Linear Probing |
|------|-------------|----------------|
| 학습 파라미터 | 전체 또는 대부분 | 분류기만 |
| 필요 데이터량 | 중간~많음 | 적음 |
| 학습 시간 | 길다 | 짧다 |
| 성능 | 일반적으로 높음 | 제한적 |
| 과적합 위험 | 있음 | 낮음 |

**실전 전략:**

1. **단계적 접근**: Linear Probing으로 빠르게 베이스라인 설정 → Fine-tuning으로 성능 개선
2. **Partial Fine-tuning**: 마지막 몇 개 레이어만 fine-tuning하여 중간 지점 선택
3. **데이터 양에 따른 선택**:
   - 레이블 데이터 < 1,000개: Linear Probing
   - 1,000~10,000개: Partial Fine-tuning
   - 10,000개 이상: Full Fine-tuning

Time Series 예측에서는 PatchTST 같은 모델을 사전학습한 후, 특정 도메인의 시계열 데이터로 fine-tuning하거나, frozen encoder 위에 forecasting head만 학습시켜 예측 성능을 향상시킬 수 있습니다.
