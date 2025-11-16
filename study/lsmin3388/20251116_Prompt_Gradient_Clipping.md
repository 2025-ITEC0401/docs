# Gradient Clipping 학습 자료

## Prompts

### Question)
```
Gradient Clipping이란 무엇이고, 왜 필요한가요? Gradient Exploding 문제와 어떤 관련이 있나요?
```

### Answer)
**Gradient Clipping 정의**

역전파 과정에서 계산된 그래디언트의 크기가 특정 임계값을 초과할 경우, 해당 값으로 제한(clip)하는 기법입니다.

**Gradient Exploding 문제:**
- 깊은 네트워크에서 역전파 시 그래디언트가 기하급수적으로 커지는 현상
- 특히 RNN/LSTM에서 긴 시퀀스 처리 시 발생
- 파라미터가 NaN이 되거나 학습 발산

**Gradient Clipping의 역할:**
1. 학습 안정성 확보
2. 급격한 파라미터 업데이트 방지
3. Loss spike 후 복구 가능
4. 더 큰 학습률 사용 가능

**문제 발생 상황:**
```
일반 학습: gradient = 0.1, 0.2, 0.15, ...
Exploding: gradient = 0.1, 0.5, 10, 1000, NaN
```

```python
import torch.nn.utils as utils

# 학습 루프에서 Gradient Clipping 적용
optimizer.zero_grad()
loss.backward()

# Gradient Clipping 적용
utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

---

### Question)
```
Gradient Clipping의 두 가지 방식(Norm Clipping vs Value Clipping)의 차이점은 무엇인가요?
```

### Answer)
**1. Gradient Norm Clipping (L2 Norm)**

전체 그래디언트의 L2 norm이 임계값을 초과하면 비례적으로 스케일링합니다.

```python
# g = g * (max_norm / ||g||) if ||g|| > max_norm

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**특징:**
- 그래디언트 방향 유지
- 전체적인 업데이트 크기 제한
- 가장 널리 사용되는 방식

**2. Gradient Value Clipping**

각 그래디언트 요소를 개별적으로 특정 범위로 제한합니다.

```python
# g = clip(g, -max_value, max_value)

torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**특징:**
- 그래디언트 방향이 변할 수 있음
- 개별 요소 단위로 적용
- 상대적으로 덜 사용됨

**비교:**

| 특성 | Norm Clipping | Value Clipping |
|------|--------------|----------------|
| 방향 보존 | O | X |
| 적용 단위 | 전체 | 개별 요소 |
| 일반적 사용 | 더 많음 | 적음 |
| 권장 상황 | 대부분 | 특수 경우 |

**권장 사항:**
- 기본적으로 Norm Clipping 사용
- max_norm = 1.0 ~ 5.0 범위에서 시작
- Transformer: 1.0, RNN: 5.0 정도 권장

---

### Question)
```
Gradient Clipping의 임계값(max_norm)은 어떻게 설정하나요? 너무 작거나 크면 어떤 문제가 생기나요?
```

### Answer)
**임계값 설정 가이드라인:**

**일반적인 권장값:**
- Transformer 모델: 1.0
- RNN/LSTM: 1.0 ~ 5.0
- CNN: 보통 불필요, 필요시 10.0 이상
- Fine-tuning: 1.0

**임계값이 너무 작을 때:**
- 학습 속도 저하
- 그래디언트 정보 손실
- 수렴에 도달하지 못할 수 있음

**임계값이 너무 클 때:**
- Clipping 효과 미미
- 여전히 불안정한 학습 가능
- Exploding gradient 방지 실패

**최적값 찾기:**

```python
# 학습 중 gradient norm 모니터링
def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# 학습 루프에서 로깅
for batch in train_loader:
    loss.backward()
    grad_norm = get_grad_norm(model)
    print(f"Gradient Norm: {grad_norm:.4f}")

    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

**실험적 접근:**
1. 먼저 clipping 없이 gradient norm 분포 관찰
2. 95 percentile 값을 초기 임계값으로 설정
3. 학습 안정성에 따라 조정

**주의사항:**
- 너무 빈번한 clipping은 문제의 증상이지 해결책이 아님
- 근본적인 원인(학습률, 아키텍처 등) 점검 필요
- Gradient accumulation 사용 시 accumulation 후 clipping
