# Positional Encoding 학습 자료

## Prompts

### Question)
```
Transformer에서 Positional Encoding이 왜 필요한가요? Self-Attention만으로는 왜 위치 정보를 파악할 수 없나요?
```

### Answer)
**Positional Encoding이 필요한 이유:**

Self-Attention 메커니즘은 모든 위치의 토큰을 동등하게 처리합니다. 즉, 순서 정보가 내재되어 있지 않습니다.

**Self-Attention의 특성:**
```python
# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
# Q, K, V는 순서와 무관하게 계산됨
# "I love you"와 "You love I"가 동일하게 처리될 수 있음
```

**위치 정보 없이 발생하는 문제:**
1. **순서 무시**: 문장의 의미가 단어 순서에 의존하는데 이를 파악 못함
2. **시계열 패턴**: 시간적 선후 관계를 학습하지 못함
3. **문맥 오해**: "The cat sat on the mat"에서 무엇이 어디에 있는지 모름

**RNN과의 차이:**
```
RNN: 순차적 처리로 위치 정보 자연스럽게 내재
     h_t = f(h_{t-1}, x_t)

Transformer: 병렬 처리로 위치 정보 별도 추가 필요
             Attention(X + PE)
```

**Positional Encoding의 역할:**
- 각 위치에 고유한 벡터 추가
- 모델이 상대적/절대적 위치 학습 가능
- 순서 정보를 입력에 주입

---

### Question)
```
Sinusoidal Positional Encoding의 원리와 수식을 설명해주세요. 왜 사인/코사인 함수를 사용하나요?
```

### Answer)
**Sinusoidal Positional Encoding 수식:**

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- pos: 시퀀스 내 위치 (0, 1, 2, ...)
- i: 차원 인덱스
- d_model: 임베딩 차원

**사인/코사인 사용 이유:**

1. **상대적 위치 표현**: PE(pos+k)를 PE(pos)의 선형 변환으로 표현 가능
2. **값 범위 제한**: [-1, 1] 범위로 안정적
3. **유일성**: 각 위치마다 고유한 패턴
4. **일반화**: 학습 시보다 긴 시퀀스에도 적용 가능

```python
import torch
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]
```

**주파수 변화:**
- 낮은 차원: 긴 주기 (전체적 위치)
- 높은 차원: 짧은 주기 (세밀한 위치)

---

### Question)
```
Learnable Positional Encoding과 Sinusoidal의 차이점은 무엇이고, 각각 언제 사용하나요?
```

### Answer)
**Learnable Positional Encoding**

위치 임베딩을 학습 가능한 파라미터로 설정합니다.

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

**비교:**

| 특성 | Sinusoidal | Learnable |
|------|-----------|-----------|
| 파라미터 수 | 0 | max_len × d_model |
| 길이 일반화 | 가능 | 어려움 |
| 학습 데이터 필요 | 불필요 | 필요 |
| 사용 모델 | 원본 Transformer, T5 | BERT, GPT |

**Sinusoidal 권장 상황:**
- 가변 길이 시퀀스
- 학습 시보다 긴 시퀀스 처리 필요
- 파라미터 수 최소화
- 시계열 데이터 (긴 시퀀스)

**Learnable 권장 상황:**
- 고정 길이 입력
- 충분한 학습 데이터
- 도메인 특화 위치 패턴
- NLP 사전학습 모델 (BERT, GPT)

**Relative Positional Encoding (최신 트렌드):**

```python
# 상대적 위치만 인코딩
# Transformer-XL, XLNet 등에서 사용
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # -max_len ~ max_len 범위의 상대 위치
        self.pe = nn.Embedding(2 * max_len + 1, d_model)

    def forward(self, q_pos, k_pos):
        relative_pos = q_pos - k_pos + self.max_len
        return self.pe(relative_pos)
```

**RoPE (Rotary Position Embedding):**
- LLaMA, PaLM 등 최신 LLM에서 사용
- 회전 행렬로 상대 위치 인코딩
- 긴 시퀀스 일반화 우수
