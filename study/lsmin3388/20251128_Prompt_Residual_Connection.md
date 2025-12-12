# Residual Connection 학습 자료

## Prompts

### Question)
```
Residual Connection(잔차 연결)이란 무엇이고, 딥러닝에서 왜 중요한가요?
```

### Answer)
**Residual Connection 정의:**

입력을 출력에 직접 더하는 연결 방식입니다. "Skip Connection"이라고도 합니다.

```python
# 일반 레이어
output = F(x)

# Residual Connection
output = x + F(x)
```

**중요성:**

1. **Gradient Flow 개선:**
   - 역전파 시 gradient가 직접 전달되는 경로 제공
   - Vanishing Gradient 문제 완화

2. **깊은 네트워크 학습 가능:**
   - ResNet이 100+ 레이어 학습 가능하게 함
   - Transformer도 이 덕분에 깊은 구조 가능

3. **항등 함수 학습 용이:**
   - 최악의 경우에도 F(x) = 0을 학습하면 identity
   - 네트워크가 "해를 끼치지 않음"

**수학적 분석:**

```python
# Gradient 흐름
# y = x + F(x)
# ∂L/∂x = ∂L/∂y × (1 + ∂F/∂x)

# 1이 항상 더해지므로 gradient가 사라지지 않음
```

**Transformer에서의 사용:**

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Attention with residual
        attn_output = self.attention(x)
        x = x + attn_output  # Residual connection

        # FFN with residual
        ffn_output = self.ffn(x)
        x = x + ffn_output  # Residual connection

        return x
```

---

### Question)
```
Pre-Norm과 Post-Norm의 차이점은 무엇이고, 각각 어떤 장단점이 있나요?
```

### Answer)
**Post-Norm (원본 Transformer)**

Residual 연결 후 정규화:

```python
# x → Sublayer → Add → LayerNorm
def post_norm_block(self, x):
    attn_out = self.attention(x)
    x = self.norm1(x + attn_out)  # Add then Norm

    ffn_out = self.ffn(x)
    x = self.norm2(x + ffn_out)
    return x
```

**Pre-Norm (개선된 방식)**

정규화 후 Sublayer:

```python
# x → LayerNorm → Sublayer → Add
def pre_norm_block(self, x):
    attn_out = self.attention(self.norm1(x))
    x = x + attn_out  # Norm then Add

    ffn_out = self.ffn(self.norm2(x))
    x = x + ffn_out
    return x
```

**비교:**

| 특성 | Post-Norm | Pre-Norm |
|------|----------|----------|
| 학습 안정성 | 낮음 | 높음 |
| Warmup 필요성 | 필수 | 선택적 |
| 최종 성능 | 약간 높음 | 좋음 |
| 깊은 모델 | 어려움 | 쉬움 |
| 사용 모델 | BERT 원본 | GPT, LLaMA |

**Gradient 흐름 분석:**

```python
# Post-Norm
∂L/∂x = ∂L/∂norm × ∂norm/∂(x+F) × (1 + ∂F/∂x)
# Norm의 gradient가 곱해져 불안정

# Pre-Norm
∂L/∂x = ∂L/∂y + ∂L/∂F × ∂F/∂norm × ∂norm/∂x
# 직접적인 gradient 경로 유지
```

**권장:**
- 새 프로젝트: Pre-Norm
- 안정적 학습: Pre-Norm
- 기존 모델 재현: 원본 따라가기

---

### Question)
```
Residual Connection에서 스케일링(Residual Scaling)은 언제 필요하고 어떻게 적용하나요?
```

### Answer)
**Residual Scaling이 필요한 상황:**

1. **매우 깊은 네트워크**: 100+ 레이어
2. **학습 불안정**: Loss spike, gradient 폭발
3. **대규모 모델**: 파라미터 수 증가에 따른 불안정

**스케일링 방법들:**

**1. 고정 스케일링**
```python
# 레이어 수에 따른 스케일 조정
scale = 1 / math.sqrt(num_layers)
output = x + scale * F(x)
```

**2. 학습 가능한 스케일링 (ReZero)**
```python
class ReZeroBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.alpha * self.sublayer(x)

# 초기에 alpha=0으로 시작하여 점진적 학습
```

**3. DeepNet 스케일링**
```python
# Microsoft의 DeepNet 방식
# 매우 깊은 Transformer (1000+ 레이어) 학습 가능

class DeepNetBlock(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha  # Residual 스케일
        self.beta = beta    # 초기화 스케일

    def forward(self, x):
        # Attention
        attn = self.attention(self.norm1(x))
        x = x + self.alpha * attn

        # FFN
        ffn = self.ffn(self.norm2(x))
        x = x + self.alpha * ffn
        return x
```

**스케일링 값 설정:**

```python
# DeepNet 권장 값
# N: 레이어 수

# Encoder
alpha = (2 * N) ** 0.25
beta = (8 * N) ** -0.25

# Decoder
alpha = (2 * N) ** 0.25
beta = (8 * N) ** -0.25
```

**실무 가이드:**

| 레이어 수 | 스케일링 | 방법 |
|----------|---------|------|
| < 12 | 불필요 | 기본 |
| 12-48 | 선택적 | ReZero |
| 48+ | 권장 | DeepNet |

```python
# 일반적인 적용
class StableTransformer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.scale = 1 / math.sqrt(2 * num_layers)
        # ...

    def forward(self, x):
        for layer in self.layers:
            x = x + self.scale * layer(self.norm(x))
        return x
```
