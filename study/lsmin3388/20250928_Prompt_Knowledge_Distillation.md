# Knowledge Distillation Study

## Prompts (Powered by Claude)

### Question 1)
```
Knowledge Distillation의 기본 개념과 Teacher-Student 모델 구조에 대해 설명해주세요.
왜 이 방법이 효과적인지, 그리고 일반적인 학습 방법과 어떤 차이가 있는지 알려주세요.
```

### Answer 1)
Knowledge Distillation(지식 증류)은 크고 복잡한 모델(Teacher 모델)의 지식을 작고 효율적인 모델(Student 모델)로 전달하는 기법입니다.

**기본 구조:**
- **Teacher Model**: 높은 성능을 가진 대규모 모델 또는 앙상블 모델
- **Student Model**: 경량화된 소규모 모델로, Teacher의 지식을 학습

**작동 원리:**
1. Teacher 모델을 먼저 학습시켜 높은 성능을 달성
2. Student 모델은 두 가지를 동시에 학습:
   - Ground truth label (정답 레이블)
   - Teacher의 출력 분포 (soft predictions)
3. Student는 Teacher의 예측 확률 분포를 모방하면서 학습

**효과적인 이유:**
- Teacher의 출력 분포는 클래스 간 유사도 정보를 담고 있음
- 예: 고양이 이미지를 분류할 때, Teacher가 "고양이 0.8, 개 0.15, 호랑이 0.05"로 예측했다면, 이는 "개나 호랑이가 고양이와 유사하다"는 추가 정보를 제공
- Hard label(정답만)보다 더 풍부한 정보를 전달하여 Student가 더 잘 일반화

**일반 학습과의 차이:**
- 일반 학습: 정답 레이블만 사용 (one-hot encoding)
- Knowledge Distillation: Teacher의 soft predictions를 추가로 활용하여 더 많은 정보 습득

---

### Question 2)
```
Soft Label과 Hard Label의 차이점은 무엇인가요?
Temperature 파라미터는 어떤 역할을 하며, Knowledge Distillation에서 왜 중요한가요?
```

### Answer 2)
**Hard Label vs Soft Label:**

**Hard Label:**
- One-hot encoding 형태의 정답 레이블
- 예: [0, 0, 1, 0, 0] - 3번째 클래스가 정답
- 정답 클래스만 1, 나머지는 0
- 클래스 간 관계 정보가 없음

**Soft Label:**
- 확률 분포 형태의 레이블
- 예: [0.05, 0.10, 0.70, 0.10, 0.05]
- 모든 클래스에 대한 확률 정보 포함
- 클래스 간 유사도와 불확실성 정보 담김

**Temperature (T) 파라미터:**

Softmax 함수에 temperature를 적용:
```
P(i) = exp(z_i / T) / Σ exp(z_j / T)
```

- **T = 1**: 일반 softmax (원래 확률 분포)
- **T > 1**: 확률 분포가 부드러워짐 (softer)
  - 예: [0.8, 0.15, 0.05] → [0.5, 0.3, 0.2] (T=3일 때)
  - 낮은 확률 클래스의 정보가 더 명확해짐
- **T < 1**: 확률 분포가 날카로워짐 (sharper)

**Knowledge Distillation에서의 중요성:**
1. **정보 전달 극대화**: 높은 T 값으로 Teacher의 미세한 차이까지 전달
2. **학습 단계**:
   - Teacher와 Student 모두 같은 T로 softmax 적용
   - 학습 후에는 T=1로 복원하여 inference
3. **Dark Knowledge**: 낮은 확률의 클래스들도 의미 있는 정보를 담고 있으며, T를 높여야 이를 효과적으로 전달 가능

**최종 Loss Function:**
```
L = α * L_soft(Student, Teacher, T) + (1-α) * L_hard(Student, Ground Truth)
```
- L_soft: 높은 T를 사용한 KL divergence
- L_hard: 정답 레이블과의 cross-entropy
- α: 두 loss의 균형을 맞추는 하이퍼파라미터

---

### Question 3)
```
시계열 예측에서 Knowledge Distillation을 어떻게 적용할 수 있나요?
TimeKD와 같은 방법론에서의 특징과, Privileged Information의 개념에 대해 설명해주세요.
```

### Answer 3)
**시계열에서의 Knowledge Distillation 적용:**

시계열 예측은 분류 문제와 달리 연속적인 값을 예측하므로, KD 적용 방식이 다릅니다.

**1. 출력 레벨 증류 (Output-level Distillation):**
- Teacher와 Student의 예측값 차이를 최소화
- MSE Loss: `L = ||y_student - y_teacher||²`
- 시계열의 전체적인 패턴 학습

**2. 특징 레벨 증류 (Feature-level Distillation):**
- 중간 레이어의 representation을 전달
- 시계열의 계층적 특징(단기/장기 패턴) 학습
- Attention map이나 hidden state를 증류

**3. 관계 레벨 증류 (Relation-level Distillation):**
- 시점 간 관계나 변수 간 상관관계 전달
- 다변량 시계열에서 특히 유용

**TimeKD 등의 방법론 특징:**
- **Temporal Pattern Distillation**: 시간적 의존성 패턴을 명시적으로 전달
- **Multi-horizon Distillation**: 다양한 예측 구간에 대해 증류
- **Autoregressive Distillation**: 순차적 예측 과정에서 발생하는 에러 누적 완화

**Privileged Information (특권 정보) 개념:**

Privileged Information은 "학습 시에만 접근 가능하고, 추론 시에는 사용할 수 없는 정보"를 의미합니다.

**시계열에서의 예시:**
1. **미래 정보**:
   - 학습 시: 예측 시점 이후의 실제 데이터를 Teacher가 볼 수 있음
   - 추론 시: 당연히 미래 데이터는 없음
   - Teacher는 미래를 보고 학습, Student는 과거만으로 예측하도록 학습

2. **고해상도 데이터**:
   - 학습 시: 1분 단위 상세 데이터
   - 추론 시: 1시간 단위 데이터만 가능
   - Teacher는 상세 패턴을 배우고 Student에게 전달

3. **보조 변수**:
   - 학습 시: 비용이 높은 센서 데이터 활용
   - 추론 시: 저비용 센서 데이터만 사용

**LUPI (Learning Using Privileged Information) 프레임워크:**
```
Teacher: f(x, x*) → y
Student: g(x) → y

x: 일반 입력 (항상 사용 가능)
x*: 특권 정보 (학습 시에만 사용 가능)
```

**시계열 예측에서의 장점:**
- Teacher가 더 많은 맥락을 활용하여 더 정확한 지식 생성
- Student는 제한된 정보로도 Teacher의 풍부한 지식을 상속
- 실용적: 추론 시 계산 비용이나 데이터 수집 비용 절감

**적용 예시:**
```
문제: 1시간 후 전력 수요 예측
- Teacher: 과거 24시간 + 미래 1시간(특권) 데이터로 학습
- Student: 과거 24시간만으로 Teacher의 예측 능력 모방
- 결과: Student가 미래 정보 없이도 더 정확한 예측 가능
```

이러한 접근 방식은 실시간 시계열 예측에서 특히 유용하며, 제한된 자원으로도 높은 성능을 달성할 수 있게 합니다.
