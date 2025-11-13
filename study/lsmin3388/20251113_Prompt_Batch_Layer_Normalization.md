# Batch Normalization vs Layer Normalization 학습 자료

## Prompts

### Question)
```
Batch Normalization과 Layer Normalization의 차이점은 무엇인가요? 각각 어떤 상황에서 사용하는 것이 좋을까요?
```

### Answer)
**Batch Normalization (BN)**
- 미니배치 내에서 각 특성(feature/channel)별로 정규화
- 정규화 축: 배치 차원
- 학습 시: 미니배치 통계 사용
- 추론 시: 전체 데이터셋의 이동 평균 통계 사용

**Layer Normalization (LN)**
- 각 샘플 내에서 모든 특성을 함께 정규화
- 정규화 축: 특성 차원
- 학습/추론 시 동일한 방식으로 동작
- 배치 크기에 독립적

**주요 차이점:**

| 특성 | Batch Norm | Layer Norm |
|------|-----------|------------|
| 정규화 축 | 배치 | 레이어/특성 |
| 배치 크기 의존성 | 있음 | 없음 |
| 추론 시 동작 | 다름 | 동일 |
| 순차 데이터 | 비적합 | 적합 |

**사용 권장:**
- **Batch Norm**: CNN, 이미지 분류, 큰 배치 크기
- **Layer Norm**: Transformer, RNN, NLP, 작은 배치 크기

```python
import torch.nn as nn

# Batch Normalization
bn = nn.BatchNorm2d(num_features=64)  # CNN용

# Layer Normalization
ln = nn.LayerNorm(normalized_shape=768)  # Transformer용
```

---

### Question)
```
Instance Normalization과 Group Normalization은 무엇인가요? Batch Norm, Layer Norm과 비교해주세요.
```

### Answer)
**Instance Normalization (IN)**
- 각 샘플의 각 채널별로 독립적으로 정규화
- 스타일 전이(Style Transfer)에서 주로 사용
- 스타일 정보가 평균/분산에 인코딩되어 있다는 가정

**Group Normalization (GN)**
- 채널을 여러 그룹으로 나누어 그룹 내에서 정규화
- 배치 크기에 독립적
- 객체 탐지, 세그멘테이션에서 효과적

**비교표:**

| 방법 | 정규화 범위 | 배치 독립 | 주 사용처 |
|------|-----------|----------|----------|
| BN | (N, H, W) | X | CNN 분류 |
| LN | (C, H, W) | O | Transformer |
| IN | (H, W) | O | Style Transfer |
| GN | (C/G, H, W) | O | 객체 탐지 |

```python
# Group Normalization (32개 그룹)
gn = nn.GroupNorm(num_groups=32, num_channels=256)

# Instance Normalization
instance_norm = nn.InstanceNorm2d(num_features=64)
```

**선택 가이드라인:**
1. 배치 크기가 충분히 크면: Batch Norm
2. 시계열/NLP: Layer Norm
3. 작은 배치 + CNN: Group Norm
4. 스타일 변환: Instance Norm

---

### Question)
```
Transformer 모델에서 Layer Normalization의 위치(Pre-LN vs Post-LN)가 왜 중요한가요?
```

### Answer)
**Post-LN (원본 Transformer)**
```
x → Attention → Add(x) → LayerNorm → FFN → Add → LayerNorm
```
- 원래 논문에서 제안된 구조
- 잔차 연결 후 정규화
- 학습 초기 불안정할 수 있음
- Learning rate warmup 필수

**Pre-LN (개선된 방식)**
```
x → LayerNorm → Attention → Add(x) → LayerNorm → FFN → Add
```
- 잔차 연결 전에 정규화
- 더 안정적인 학습
- Warmup 없이도 학습 가능
- 깊은 모델에서 gradient flow 개선

**성능 비교:**
- Pre-LN: 학습 안정성 ↑, 수렴 속도 ↑
- Post-LN: 최종 성능이 약간 더 높을 수 있음 (충분한 튜닝 시)

```python
# Pre-LN Transformer Block
class PreLNTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

**실무 권장:**
- 새로운 프로젝트: Pre-LN 사용
- 기존 구현 재현: 원본과 동일하게 유지
- GPT 계열: Pre-LN 사용
- BERT 원본: Post-LN
