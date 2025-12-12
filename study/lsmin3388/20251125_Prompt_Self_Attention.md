# Self-Attention 학습 자료

## Prompts

### Question)
```
Self-Attention 메커니즘의 동작 원리를 Query, Key, Value 관점에서 설명해주세요.
```

### Answer)
**Self-Attention 기본 원리:**

입력 시퀀스의 각 위치가 다른 모든 위치와의 관계를 계산하여 문맥을 파악합니다.

**Q, K, V의 역할:**

1. **Query (Q)**: "어떤 정보를 찾고 싶은가"
2. **Key (K)**: "나는 어떤 정보를 가지고 있는가"
3. **Value (V)**: "실제로 전달할 정보"

**계산 과정:**

```python
def self_attention(X, W_q, W_k, W_v):
    """
    X: 입력 시퀀스 (seq_len, d_model)
    """
    # 1. Q, K, V 생성
    Q = X @ W_q  # (seq_len, d_k)
    K = X @ W_k  # (seq_len, d_k)
    V = X @ W_v  # (seq_len, d_v)

    # 2. Attention Score 계산
    scores = Q @ K.T / math.sqrt(d_k)  # (seq_len, seq_len)

    # 3. Softmax로 가중치 변환
    attention_weights = F.softmax(scores, dim=-1)

    # 4. Value 가중합
    output = attention_weights @ V  # (seq_len, d_v)

    return output, attention_weights
```

**직관적 이해:**
```
문장: "The cat sat on the mat"

"cat"의 Q가 다른 모든 단어의 K와 비교:
- "The"와 얼마나 관련? → attention score
- "sat"와 얼마나 관련? → attention score
- "mat"와 얼마나 관련? → attention score

높은 score를 가진 단어들의 V를 많이 반영
```

---

### Question)
```
Scaled Dot-Product Attention에서 왜 sqrt(d_k)로 나누나요? 스케일링의 중요성은 무엇인가요?
```

### Answer)
**스케일링이 필요한 이유:**

d_k(key 차원)가 커지면 Q·K의 내적 값도 커집니다.

**수학적 분석:**
```
Q, K의 각 요소가 평균 0, 분산 1이라 가정
Q·K = Σ(q_i * k_i)

E[Q·K] = 0
Var[Q·K] = d_k  # 차원에 비례하여 분산 증가!
```

**분산이 커지면 생기는 문제:**

```python
# d_k = 64일 때 Q·K 값 분포
scores = Q @ K.T  # 값의 분산이 64

# Softmax 적용 시 극단적인 확률 분포
# 큰 값 → 1에 가까움
# 작은 값 → 0에 가까움
probs = softmax(scores)  # [0.99, 0.005, 0.005, ...]
```

**Gradient 문제:**
- Softmax의 gradient는 확률이 0 또는 1에 가까우면 매우 작아짐
- 학습이 거의 일어나지 않음 (Gradient Vanishing)

**스케일링 적용:**

```python
def scaled_dot_product_attention(Q, K, V, d_k):
    # sqrt(d_k)로 나누어 분산을 1로 유지
    scores = (Q @ K.T) / math.sqrt(d_k)

    # 이제 softmax가 부드러운 분포 생성
    attention_weights = F.softmax(scores, dim=-1)

    return attention_weights @ V
```

**시각화:**
```
스케일링 전: [100, 50, 30, 20] → softmax → [0.99, 0.01, 0.00, 0.00]
스케일링 후: [12.5, 6.25, 3.75, 2.5] → softmax → [0.85, 0.10, 0.03, 0.02]
```

---

### Question)
```
Self-Attention의 시간 복잡도는 왜 O(n²)인가요? 긴 시퀀스에서 효율적인 대안은 무엇인가요?
```

### Answer)
**O(n²) 복잡도의 원인:**

```python
# Attention Score 행렬
scores = Q @ K.T  # (n, d) @ (d, n) = (n, n)

# 모든 위치 쌍에 대해 attention 계산
# n개의 query × n개의 key = n² 연산
```

**메모리 사용:**
- Attention 행렬: O(n²)
- n=1000 → 1M 요소
- n=10000 → 100M 요소 (수 GB)

**효율적인 대안들:**

**1. Sparse Attention (Longformer, BigBird)**
```python
# 전체가 아닌 일부만 attention
# - Local: 주변 k개만
# - Global: 특정 토큰만 전체와
# 복잡도: O(n × k)
```

**2. Linear Attention**
```python
# softmax 대신 커널 함수 사용
# Attention = φ(Q) @ (φ(K)^T @ V)
# 계산 순서 변경으로 O(n) 달성

def linear_attention(Q, K, V):
    Q = F.elu(Q) + 1
    K = F.elu(K) + 1

    # (n, d) @ ((d, n) @ (n, d)) vs ((n, d) @ (d, n)) @ (n, d)
    KV = K.T @ V  # (d, d) - 작음!
    output = Q @ KV
    return output
```

**3. Flash Attention**
```python
# 알고리즘 최적화 (메모리 접근 패턴)
# 복잡도는 O(n²)이지만 실제 속도 2-4배 향상
# HuggingFace Transformers에서 지원

from transformers import AutoModel
model = AutoModel.from_pretrained(
    "bert-base",
    attn_implementation="flash_attention_2"
)
```

**방법별 비교:**

| 방법 | 시간 복잡도 | 메모리 | 성능 |
|------|-----------|--------|------|
| Vanilla | O(n²) | O(n²) | 최고 |
| Sparse | O(n×k) | O(n×k) | 좋음 |
| Linear | O(n) | O(n) | 보통 |
| Flash | O(n²) | O(n) | 최고 |

**시계열에서의 선택:**
- 짧은 시퀀스 (< 512): Vanilla
- 중간 (512-2048): Flash Attention
- 긴 시퀀스 (> 2048): Sparse 또는 Linear
