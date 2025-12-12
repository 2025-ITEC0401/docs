# Early Stopping 학습 자료

## Prompts

### Question)
```
Early Stopping이란 무엇이고, 과적합 방지에 어떻게 도움이 되나요? 어떤 기준으로 학습을 중단하나요?
```

### Answer)
**Early Stopping 정의**

검증 성능이 더 이상 개선되지 않을 때 학습을 조기에 중단하는 정규화 기법입니다.

**과적합 방지 메커니즘:**
1. 과적합 시작 시점 포착: Train loss는 계속 감소하지만 Validation loss가 증가하기 시작
2. 최적 시점에서 학습 중단
3. 불필요한 학습 반복 방지

**학습 중단 기준:**
- Validation loss가 N 에포크 동안 개선되지 않음
- Validation accuracy가 개선되지 않음
- 사용자 정의 메트릭 기준

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 사용 예시
early_stopping = EarlyStopping(patience=10)
for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

### Question)
```
Early Stopping의 patience와 min_delta 파라미터는 어떻게 설정하나요? 적절한 값의 기준이 있나요?
```

### Answer)
**Patience 설정:**

검증 성능이 개선되지 않아도 기다리는 에포크 수입니다.

**권장 값:**
- 소규모 데이터셋: 5~10 에포크
- 대규모 데이터셋: 10~20 에포크
- 복잡한 모델: 15~30 에포크

**고려사항:**
- 너무 작으면: 일시적인 성능 저하에도 조기 종료
- 너무 크면: 과적합 진행 후에야 종료

**min_delta 설정:**

개선으로 간주할 최소 변화량입니다.

**권장 값:**
- Loss 기준: 0.0001 ~ 0.001
- Accuracy 기준: 0.001 ~ 0.01

```python
# Loss 기반 Early Stopping
early_stopping = EarlyStopping(
    patience=15,      # 15 에포크 기다림
    min_delta=0.001   # 0.1% 이상 개선되어야 함
)

# Accuracy 기반 Early Stopping
early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.005,  # 0.5% 이상 개선
    mode='max'        # 높을수록 좋음
)
```

**설정 팁:**
1. 처음엔 patience를 크게 설정하고 학습 곡선 관찰
2. 과적합 시작 시점 파악 후 patience 조정
3. 노이즈가 많은 데이터는 patience 높게 설정

---

### Question)
```
Early Stopping과 함께 Model Checkpoint를 사용해야 하는 이유는 무엇인가요? 어떻게 구현하나요?
```

### Answer)
**함께 사용해야 하는 이유:**

Early Stopping만 사용하면 마지막 에포크의 모델이 저장되는데, 이는 최적의 모델이 아닐 수 있습니다.

**시나리오:**
```
Epoch 10: val_loss = 0.25 (최저)
Epoch 11: val_loss = 0.26
Epoch 12: val_loss = 0.27
...
Epoch 20: val_loss = 0.30 (Early Stop)
```

- Early Stopping 시점의 모델 (Epoch 20)은 최적이 아님
- Epoch 10의 모델을 저장해야 함

**구현:**

```python
class EarlyStoppingWithCheckpoint:
    def __init__(self, patience=7, path='best_model.pt'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # 최고 성능 모델 저장
            torch.save(model.state_dict(), self.path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 사용 예시
checkpoint = EarlyStoppingWithCheckpoint(patience=10)

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    checkpoint(val_loss, model)

    if checkpoint.early_stop:
        # 최고 성능 모델 로드
        model.load_state_dict(torch.load('best_model.pt'))
        break
```

**PyTorch Lightning 사용 시:**

```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min'
    )
]

trainer = Trainer(callbacks=callbacks)
```

**추가 권장사항:**
- 여러 메트릭 기준으로 다중 checkpoint 저장
- 마지막 N개 checkpoint 유지 (save_top_k)
- 모델과 옵티마이저 상태 함께 저장
