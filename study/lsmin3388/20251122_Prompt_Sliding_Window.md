# Sliding Window 기법 학습 자료

## Prompts

### Question)
```
시계열 데이터에서 Sliding Window 기법이란 무엇이고, 왜 사용하나요? 주요 파라미터는 무엇인가요?
```

### Answer)
**Sliding Window 정의**

연속적인 시계열 데이터를 고정 크기의 윈도우로 분할하여 샘플을 생성하는 기법입니다. 딥러닝 모델의 입력으로 사용할 수 있는 형태로 변환합니다.

**사용 이유:**
1. 가변 길이 시계열을 고정 길이로 변환
2. 데이터 증강 효과 (많은 샘플 생성)
3. 시간적 패턴 포착
4. 배치 학습 가능하게 함

**주요 파라미터:**

```python
def create_sliding_windows(data, window_size, stride=1, horizon=1):
    """
    Args:
        data: 원본 시계열 데이터
        window_size: 입력 윈도우 크기 (look-back)
        stride: 윈도우 이동 간격
        horizon: 예측 기간 (예측 태스크용)
    """
    X, y = [], []
    for i in range(0, len(data) - window_size - horizon + 1, stride):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + horizon])
    return np.array(X), np.array(y)

# 예시: 100일 데이터, 7일 윈도우, 1일 예측
X, y = create_sliding_windows(data, window_size=7, stride=1, horizon=1)
```

**파라미터 설명:**
- **window_size**: 과거 몇 시점을 볼 것인가
- **stride**: 윈도우 간 겹침 정도 (stride=1이면 최대 겹침)
- **horizon**: 미래 몇 시점을 예측할 것인가

---

### Question)
```
Window size와 Stride를 어떻게 설정해야 하나요? 데이터의 특성에 따라 어떻게 달라지나요?
```

### Answer)
**Window Size 설정 가이드:**

**고려 요소:**
1. 데이터의 주기성 (seasonality)
2. 패턴의 길이
3. 모델 복잡도와 메모리
4. 도메인 지식

**일반적인 권장:**

| 데이터 주기 | 권장 Window Size |
|------------|-----------------|
| 시간별 데이터 | 24~168 (1일~1주) |
| 일별 데이터 | 7~30 (1주~1달) |
| 월별 데이터 | 12~24 (1~2년) |

```python
# 주기성 기반 설정
if seasonality == 'daily':
    window_size = 24  # 하루 데이터
elif seasonality == 'weekly':
    window_size = 7 * 24  # 일주일 데이터
```

**Stride 설정 가이드:**

```python
# stride = 1: 최대 샘플 수, 높은 중복
# stride = window_size: 중복 없음, 적은 샘플
# stride = window_size // 2: 50% 중복 (권장 시작점)

# 샘플 수 계산
n_samples = (len(data) - window_size) // stride + 1
```

**데이터 특성별 전략:**

| 특성 | Window Size | Stride |
|------|------------|--------|
| 데이터 양 많음 | 크게 | 크게 |
| 데이터 양 적음 | 작게 | 1 |
| 빠른 변화 | 작게 | 작게 |
| 느린 변화 | 크게 | 크게 |
| 강한 주기성 | 주기의 배수 | 주기 단위 |

---

### Question)
```
예측 태스크에서 Multi-step Prediction을 위한 윈도우 구성은 어떻게 하나요?
```

### Answer)
**Multi-step Prediction 방식:**

**1. Direct Multi-step**
각 미래 시점에 대해 독립적으로 예측합니다.

```python
def create_direct_multistep(data, window_size, horizon):
    X, Y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        Y.append(data[i + window_size:i + window_size + horizon])
    return np.array(X), np.array(Y)

# 모델 출력: (batch, horizon)
model = nn.Linear(window_size, horizon)
```

**2. Recursive Multi-step**
한 시점씩 예측하고 결과를 다음 입력으로 사용합니다.

```python
def recursive_predict(model, initial_window, horizon):
    predictions = []
    current_window = initial_window.copy()

    for _ in range(horizon):
        # 한 시점 예측
        pred = model.predict(current_window.reshape(1, -1))
        predictions.append(pred[0])

        # 윈도우 업데이트
        current_window = np.roll(current_window, -1)
        current_window[-1] = pred[0]

    return np.array(predictions)
```

**3. Sequence-to-Sequence (Encoder-Decoder)**

```python
class Seq2SeqPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, horizon):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.horizon = horizon

    def forward(self, x):
        # Encoder
        _, (h, c) = self.encoder(x)

        # Decoder
        outputs = []
        dec_input = x[:, -1:, :]
        for _ in range(self.horizon):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.fc(out)
            outputs.append(pred)
            dec_input = pred

        return torch.cat(outputs, dim=1)
```

**방식 비교:**

| 방식 | 장점 | 단점 |
|------|------|------|
| Direct | 단순, 오차 누적 없음 | 시점 간 의존성 무시 |
| Recursive | 시점 간 의존성 | 오차 누적 |
| Seq2Seq | 유연, 가변 길이 | 복잡, 학습 어려움 |

**실무 권장:**
- 짧은 horizon (1~5): Direct 또는 Recursive
- 긴 horizon: Seq2Seq 또는 Transformer
- 불확실성 추정 필요: 확률적 예측 모델
