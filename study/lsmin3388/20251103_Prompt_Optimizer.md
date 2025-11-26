# Optimizer 학습 자료

## Prompts (Powered by Claude)

### Question)
```
딥러닝에서 자주 사용되는 SGD, Adam, AdamW 옵티마이저의 차이점과 각각의 장단점을 비교해주세요. 어떤 상황에서 어떤 옵티마이저를 선택하는 것이 좋을까요?
```

### Answer)
**SGD (Stochastic Gradient Descent)**
- 가장 기본적인 옵티마이저로, 학습률을 고정하여 그래디언트 방향으로 파라미터를 업데이트합니다
- 장점: 단순하고 메모리 효율적이며, 적절한 학습률과 모멘텀을 사용하면 좋은 일반화 성능을 보입니다
- 단점: 학습률 튜닝이 까다롭고, 수렴 속도가 느릴 수 있습니다
- 사용 권장: CV 분야(특히 ResNet, Vision Transformer 등)에서 긴 학습 시간을 투자할 수 있을 때

**Adam (Adaptive Moment Estimation)**
- 각 파라미터마다 적응적 학습률을 사용하며, 1차 모멘텀(gradient)과 2차 모멘텀(squared gradient)을 활용합니다
- 장점: 학습률 튜닝이 상대적으로 쉽고, 빠른 수렴 속도를 보입니다
- 단점: 일부 경우 일반화 성능이 SGD보다 떨어질 수 있고, 메모리 사용량이 더 많습니다
- 사용 권장: NLP, 강화학습, 빠른 프로토타이핑이 필요할 때

**AdamW (Adam with Weight Decay)**
- Adam에 올바른 방식의 Weight Decay를 적용한 버전입니다
- Adam은 L2 정규화를 그래디언트에 추가하는 방식이지만, AdamW는 파라미터에 직접 decay를 적용합니다
- 장점: Adam의 빠른 수렴 속도와 더 나은 일반화 성능을 동시에 제공합니다
- 사용 권장: Transformer 모델(BERT, GPT 등), 최신 딥러닝 모델 학습 시 기본 선택

---

### Question)
```
Learning Rate Scheduler의 역할과 주요 스케줄러(Cosine Annealing, Step Decay 등)의 특징을 설명해주세요. 각 스케줄러는 언제 사용하는 것이 효과적인가요?
```

### Answer)
**Learning Rate Scheduler의 역할**
- 학습 과정에서 학습률을 동적으로 조정하여 최적화 성능을 향상시킵니다
- 초기에는 큰 학습률로 빠르게 학습하고, 후반부에는 작은 학습률로 세밀한 최적화를 수행합니다
- Local minima 탈출과 안정적인 수렴에 도움을 줍니다

**주요 스케줄러 비교**

1. **Step Decay (StepLR)**
   - 특정 에포크마다 학습률을 일정 비율로 감소시킵니다
   - 예: 30 에포크마다 학습률을 0.1배로 감소
   - 장점: 구현이 간단하고 예측 가능
   - 단점: 하이퍼파라미터(step size, gamma) 튜닝 필요
   - 사용 권장: 전통적인 CNN 모델, 학습 스케줄이 명확한 경우

2. **Cosine Annealing**
   - 코사인 함수 형태로 학습률을 부드럽게 감소시킵니다
   - 수식: η_t = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2
   - 장점: 부드러운 감소로 안정적 학습, 주기적 재시작(Warm Restart) 가능
   - 단점: 최대 에포크 수를 미리 정해야 함
   - 사용 권장: Transformer 모델, 최신 딥러닝 아키텍처

3. **Exponential Decay**
   - 매 에포크마다 지수적으로 학습률 감소
   - 장점: 연속적이고 부드러운 감소
   - 단점: decay rate 튜닝 필요
   - 사용 권장: 강화학습, 긴 학습이 필요한 경우

4. **ReduceLROnPlateau**
   - Validation loss가 개선되지 않을 때 학습률 감소
   - 장점: 학습 상황에 적응적으로 반응
   - 단점: Validation 세트 필요, 불규칙한 스케줄
   - 사용 권장: 학습 동향을 예측하기 어려운 경우

---

### Question)
```
Weight Decay는 무엇이고, 딥러닝 학습에서 어떤 역할을 하나요? L2 정규화와는 어떻게 다른가요?
```

### Answer)
**Weight Decay의 정의와 역할**

Weight Decay는 모델의 가중치가 너무 커지는 것을 방지하기 위해 최적화 과정에서 파라미터에 페널티를 주는 기법입니다.

**주요 역할:**
1. **과적합 방지**: 모델의 복잡도를 제한하여 일반화 성능 향상
2. **가중치 크기 제어**: 큰 가중치로 인한 수치적 불안정성 방지
3. **더 부드러운 결정 경계**: 입력 변화에 덜 민감한 모델 학습

**Weight Decay vs L2 정규화**

많은 경우 혼용되지만, 실제로는 다른 개념입니다:

- **L2 정규화**: 손실 함수에 ||w||²를 추가
  - Loss = Original Loss + (λ/2) × ||w||²
  - 그래디언트: ∇L = ∇Original Loss + λw

- **Weight Decay**: 옵티마이저 업데이트 시 직접 가중치 감소
  - w_t+1 = w_t - η × ∇L - η × λ × w_t

**차이가 중요한 이유 (Adam, AdamW 예시):**

- **Adam + L2 정규화**: adaptive learning rate 때문에 L2 정규화 효과가 희석됨
- **AdamW**: 파라미터에 직접 decay를 적용하여 정규화 효과가 제대로 작동

**실무 권장사항:**
- SGD 사용 시: L2 정규화와 Weight Decay는 거의 동일한 효과
- Adam 사용 시: AdamW를 사용하거나 명시적으로 Weight Decay 구현 필요
- 일반적인 값: 0.01 ~ 0.0001 (모델과 데이터셋에 따라 조정)
- Batch Normalization과 함께 사용하면 상호보완적 효과

PyTorch 예시:
```python
# AdamW 사용 (권장)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Adam + L2 (비권장, 효과 약함)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
```
