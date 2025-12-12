# Multi-Head Attention 학습 자료

## Prompts

### Question)
```
Multi-Head Attention이 Single-Head Attention보다 좋은 이유는 무엇인가요? 여러 Head를 사용하면 어떤 이점이 있나요?
```

### Answer)
**Multi-Head Attention의 핵심 아이디어:**

하나의 attention으로 모든 관계를 포착하는 대신, 여러 개의 attention을 병렬로 수행하여 다양한 유형의 관계를 학습합니다.

**Single-Head의 한계:**
```python
# 하나의 attention만으로는
# - 구문적 관계 (주어-동사)
# - 의미적 관계 (동의어)
# - 위치적 관계 (인접 단어)
# 를 동시에 포착하기 어려움
```

**Multi-Head의 이점:**

1. **다양한 표현 부분 공간**: 각 head가 서로 다른 관계 패턴 학습
2. **앙상블 효과**: 여러 관점의 정보 통합
3. **안정적인 학습**: 특정 head 실패해도 다른 head로 보완
4. **표현력 증가**: 같은 파라미터로 더 풍부한 표현

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # 각 head별로 Q, K, V 생성
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # (batch, heads, seq_len, d_k)로 변환
        Q, K, V = [t.transpose(1, 2) for t in (Q, K, V)]

        # 각 head에서 attention 수행
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ V

        # head 결과 합치기
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)

        return self.W_o(context)
```

---

### Question)
```
Head 수는 어떻게 정하나요? Head 수가 너무 많거나 적으면 어떤 문제가 생기나요?
```

### Answer)
**Head 수 설정 가이드라인:**

**일반적인 규칙:**
```
num_heads는 d_model의 약수여야 함
d_k = d_model / num_heads
```

**대표적인 모델들의 설정:**

| 모델 | d_model | num_heads | d_k |
|------|---------|-----------|-----|
| BERT-base | 768 | 12 | 64 |
| BERT-large | 1024 | 16 | 64 |
| GPT-2 | 768 | 12 | 64 |
| GPT-3 | 12288 | 96 | 128 |

**Head 수가 너무 적을 때:**
- 다양한 관계 패턴 포착 어려움
- 표현력 제한
- 단일 유형의 attention만 학습

**Head 수가 너무 많을 때:**
- d_k가 너무 작아짐 (표현력 감소)
- 각 head의 capacity 부족
- 중복된 패턴 학습

```python
# 적절한 d_k 유지 중요
# d_k >= 32 권장, 일반적으로 64

# 좋은 예
d_model = 512, num_heads = 8  # d_k = 64

# 나쁜 예
d_model = 512, num_heads = 64  # d_k = 8 (너무 작음)
```

**실험적 가이드:**
- 소규모 모델: 4-8 heads
- 중규모 모델: 8-12 heads
- 대규모 모델: 12-16+ heads
- d_k는 32-128 범위 유지

---

### Question)
```
각 Attention Head가 실제로 어떤 패턴을 학습하는지 시각화하고 분석하는 방법은 무엇인가요?
```

### Answer)
**Attention 가중치 시각화:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, head_idx=0):
    """
    attention_weights: (num_heads, seq_len, seq_len)
    """
    # 특정 head의 attention 추출
    attn = attention_weights[head_idx].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                cmap='viridis', annot=True, fmt='.2f')
    plt.title(f'Attention Head {head_idx}')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.show()
```

**Head별 역할 분석:**

```python
def analyze_heads(model, input_ids):
    # Hook으로 attention 가중치 수집
    attention_maps = []

    def hook(module, input, output):
        attention_maps.append(output[1])  # attention weights

    for layer in model.transformer.layers:
        layer.self_attn.register_forward_hook(hook)

    # Forward pass
    _ = model(input_ids)

    return attention_maps
```

**발견되는 패턴 유형:**

1. **위치 기반 Head**: 인접 토큰에 집중
```
   [0.1, 0.7, 0.1, 0.1, ...]  # 이전 토큰 주목
```

2. **구문 Head**: 문법적 관계 포착
```
   주어 → 동사, 수식어 → 피수식어
```

3. **BOS/EOS Head**: 특수 토큰에 집중
```
   모든 위치가 [CLS]나 [SEP]에 높은 attention
```

4. **복사 Head**: 같은 토큰 반복 포착

**HuggingFace BertViz 사용:**
```python
from bertviz import head_view, model_view

# 전체 head 시각화
head_view(attention, tokens)

# 레이어별 패턴 분석
model_view(attention, tokens)
```

**분석 시 고려사항:**
- 낮은 레이어: 지역적, 구문적 패턴
- 높은 레이어: 전역적, 의미적 패턴
- 특정 태스크에 중요한 head 식별 가능
- Head Pruning 연구에 활용
