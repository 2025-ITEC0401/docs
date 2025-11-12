# Dropout 학습 자료

## Prompts

### Question)
```
딥러닝에서 Dropout이란 무엇이고, 어떻게 과적합을 방지하는지 설명해주세요. 학습 시와 추론 시의 동작 차이도 알려주세요.
```

### Answer)
**Dropout의 정의**

Dropout은 학습 과정에서 무작위로 일부 뉴런의 출력을 0으로 만드는 정규화 기법입니다. 각 학습 iteration마다 서로 다른 뉴런들이 비활성화되어, 모델이 특정 뉴런에 과도하게 의존하는 것을 방지합니다.

**과적합 방지 메커니즘:**
1. **앙상블 효과**: 서로 다른 부분 네트워크를 학습하는 효과
2. **Co-adaptation 방지**: 뉴런들이 서로 독립적으로 유용한 특징 학습
3. **모델 불확실성 도입**: 더 robust한 특징 추출

**학습 시 vs 추론 시 동작:**

- **학습 시**: 확률 p로 뉴런을 무작위 비활성화, 활성 뉴런의 출력을 1/(1-p)로 스케일링
- **추론 시**: 모든 뉴런 활성화, Dropout 적용하지 않음

```python
import torch.nn as nn

# PyTorch에서 Dropout 사용
model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # 50% 확률로 비활성화
    nn.Linear(128, 10)
)

# 학습 모드
model.train()  # Dropout 활성화

# 추론 모드
model.eval()  # Dropout 비활성화
```

---

### Question)
```
Dropout 비율(p)은 어떻게 설정하는 것이 좋나요? 레이어별로 다르게 적용해야 하나요?
```

### Answer)
**일반적인 Dropout 비율 가이드라인:**

1. **은닉층(Hidden Layers)**: 0.2 ~ 0.5
   - 레이어가 깊어질수록 낮은 비율 권장
   - 첫 번째 은닉층: 0.2, 깊은 층: 0.3~0.5

2. **입력층**: 0.1 ~ 0.2
   - 너무 높으면 정보 손실 우려

3. **출력층 직전**: 일반적으로 사용하지 않음

**레이어별 차별화 전략:**

```python
class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.drop1 = nn.Dropout(0.2)  # 초기 레이어: 낮은 비율

        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.3)  # 중간 레이어

        self.fc3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(0.5)  # 후반 레이어: 높은 비율

        self.fc4 = nn.Linear(128, 10)  # 출력층: Dropout 없음
```

**주의사항:**
- Batch Normalization과 함께 사용 시 주의 필요 (순서: Conv → BN → ReLU → Dropout)
- CNN에서는 Spatial Dropout (채널 단위) 사용 권장
- 작은 데이터셋일수록 높은 Dropout 비율 고려

---

### Question)
```
Dropout의 변형인 DropConnect, Spatial Dropout, DropBlock은 각각 무엇인가요?
```

### Answer)
**Dropout 변형 기법들:**

1. **DropConnect**
   - 뉴런 출력 대신 가중치(weight)를 무작위로 0으로 설정
   - 더 세밀한 정규화 효과
   - 계산 비용이 더 높음

2. **Spatial Dropout (Dropout2D)**
   - CNN에서 채널 전체를 드롭
   - 인접 픽셀들의 상관관계 고려
   - 공간적 특징의 co-adaptation 방지

```python
# Spatial Dropout 예시
self.spatial_drop = nn.Dropout2d(p=0.25)
# 전체 feature map 채널을 드롭
```

3. **DropBlock**
   - 연속된 영역을 블록 단위로 드롭
   - 인접 activation들의 의미적 연관성 고려
   - 객체 탐지, 세그멘테이션에 효과적

4. **Alpha Dropout**
   - SELU 활성화 함수와 함께 사용
   - 자기 정규화 특성 유지

**사용 권장:**
- FC Layer: 기본 Dropout
- CNN: Spatial Dropout 또는 DropBlock
- RNN/LSTM: Variational Dropout (시간축 일관성 유지)
