# Feed Forward Network in Transformer 학습 자료

## Prompts

### Question)
```
Transformer의 Feed Forward Network(FFN)은 어떤 역할을 하나요? Attention만으로는 왜 충분하지 않나요?
```

### Answer)
**Feed Forward Network의 역할:**

Transformer 블록에서 Self-Attention 다음에 오는 2층 MLP입니다. 각 위치에 독립적으로 적용됩니다.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 확장 → 비선형 → 축소
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

**Attention만으로 부족한 이유:**

1. **선형 변환의 한계**: Attention은 가중합(linear combination)
   - 비선형 패턴 포착 불가
   - 복잡한 특징 변환 어려움

2. **위치별 처리 부재**: Attention은 위치 간 관계만 모델링
   - 각 위치의 표현을 개별적으로 변환하는 과정 필요

3. **용량 증가**: 파라미터의 대부분이 FFN에 있음
   - 모델의 "기억"을 저장하는 역할

**FFN의 구체적 역할:**

```python
# Attention 출력: 문맥 정보 통합
context = attention(x)

# FFN: 비선형 특징 변환
# - 복잡한 패턴 학습
# - 위치별 표현 정제
# - 지식 저장 (key-value memory로 해석 가능)
output = ffn(context)
```

---

### Question)
```
FFN의 은닉층 크기(d_ff)는 왜 보통 d_model의 4배인가요? 다른 비율을 사용하면 어떻게 되나요?
```

### Answer)
**d_ff = 4 × d_model 설정 이유:**

1. **경험적 최적값**: 원본 Transformer 논문에서 실험적으로 발견
2. **충분한 표현력**: 병목 구조로 정보 압축 후 복원
3. **계산 비용 균형**: Attention과 FFN의 계산량 균형

**구조 분석:**
```
입력: d_model = 512
확장: d_ff = 2048 (4배)
축소: d_model = 512

파라미터 수: 512 × 2048 + 2048 × 512 ≈ 2M
```

**다른 비율의 영향:**

| 비율 | 장점 | 단점 |
|------|------|------|
| 1x | 파라미터 적음 | 표현력 부족 |
| 2x | 균형 | 약간 부족 |
| 4x | 검증된 성능 | 표준 |
| 8x | 표현력 증가 | 파라미터 과다 |

**최신 연구 동향:**

```python
# GLU 변형 사용 시 더 작은 비율 가능
# SwiGLU (LLaMA, PaLM)
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # d_ff를 2/3로 줄여도 유사 성능
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# LLaMA: d_ff = 2.7 × d_model (SwiGLU 사용)
```

**실무 권장:**
- 기본: 4배 사용
- 효율성 중시: 2-3배 + GLU 변형
- 메모리 제약: 2배 또는 Mixture of Experts

---

### Question)
```
FFN의 활성화 함수로 ReLU, GELU, SiLU 중 어떤 것을 선택해야 하나요?
```

### Answer)
**주요 활성화 함수 비교:**

**1. ReLU (원본 Transformer)**
```python
# f(x) = max(0, x)
output = F.relu(self.linear1(x))
```
- 장점: 단순, 빠름
- 단점: Dead neuron 문제, 음수 영역 무시

**2. GELU (BERT, GPT-2)**
```python
# f(x) = x × Φ(x) ≈ x × sigmoid(1.702x)
output = F.gelu(self.linear1(x))
```
- 장점: 부드러운 활성화, 음수 일부 허용
- NLP에서 널리 사용

**3. SiLU/Swish (GPT-3, 최신 모델)**
```python
# f(x) = x × sigmoid(x)
output = F.silu(self.linear1(x))
```
- 장점: 자기 게이팅, 안정적 학습
- 최신 LLM에서 선호

**4. SwiGLU (LLaMA, PaLM)**
```python
# GLU 변형 + SiLU
class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = self.linear1(x), self.linear2(x)
        return self.linear3(F.silu(x1) * x2)
```
- 장점: SOTA 성능
- 단점: 파라미터 50% 증가

**성능 비교:**

| 활성화 함수 | 계산량 | NLP 성능 | 시계열 |
|------------|--------|---------|--------|
| ReLU | 최소 | 기본 | 좋음 |
| GELU | 중간 | 좋음 | 좋음 |
| SiLU | 중간 | 매우 좋음 | 좋음 |
| SwiGLU | 높음 | 최고 | 좋음 |

**권장 선택:**
- 빠른 실험: ReLU
- 일반적 선택: GELU
- SOTA 추구: SwiGLU
- 시계열 모델: GELU 또는 ReLU

```python
# PyTorch 설정 예시
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, activation='gelu'):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
```
