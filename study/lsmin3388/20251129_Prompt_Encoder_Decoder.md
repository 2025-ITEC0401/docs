# Encoder-Decoder 아키텍처 학습 자료

## Prompts

### Question)
```
Encoder-Only, Decoder-Only, Encoder-Decoder 아키텍처의 차이점과 각각의 대표 모델은 무엇인가요?
```

### Answer)
**1. Encoder-Only 아키텍처**

입력을 양방향으로 처리하여 표현을 학습합니다.

```python
# 양방향 Self-Attention (모든 위치 참조 가능)
# [CLS] token으로 전체 문장 표현
class EncoderOnly(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Bidirectional attention
        return x
```

**대표 모델**: BERT, RoBERTa, ALBERT
**적합한 태스크**: 분류, NER, 문장 임베딩

**2. Decoder-Only 아키텍처**

Causal attention으로 왼쪽에서 오른쪽으로만 처리합니다.

```python
# 인과적 마스킹 (미래 토큰 참조 불가)
class DecoderOnly(nn.Module):
    def forward(self, x):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        for layer in self.layers:
            x = layer(x, mask=mask)  # Causal attention
        return x
```

**대표 모델**: GPT, LLaMA, Claude
**적합한 태스크**: 텍스트 생성, 언어 모델링

**3. Encoder-Decoder 아키텍처**

Encoder가 입력을 인코딩하고 Decoder가 출력을 생성합니다.

```python
class EncoderDecoder(nn.Module):
    def forward(self, src, tgt):
        # Encoder: 입력 인코딩
        memory = self.encoder(src)

        # Decoder: Cross-attention으로 encoder 출력 참조
        output = self.decoder(tgt, memory)
        return output
```

**대표 모델**: T5, BART, 원본 Transformer
**적합한 태스크**: 번역, 요약, 질의응답

**비교표:**

| 구조 | Attention | 대표 모델 | 주 용도 |
|------|-----------|----------|--------|
| Encoder | 양방향 | BERT | 이해 |
| Decoder | 단방향 | GPT | 생성 |
| Enc-Dec | 혼합 | T5 | 변환 |

---

### Question)
```
Cross-Attention은 무엇이고, Self-Attention과 어떻게 다른가요?
```

### Answer)
**Self-Attention vs Cross-Attention:**

**Self-Attention:**
- Q, K, V가 모두 같은 시퀀스에서 생성
- 입력 시퀀스 내부의 관계 모델링

```python
def self_attention(x):
    Q = self.W_q(x)
    K = self.W_k(x)
    V = self.W_v(x)
    return attention(Q, K, V)
```

**Cross-Attention:**
- Q는 한 시퀀스, K와 V는 다른 시퀀스에서 생성
- 두 시퀀스 간의 관계 모델링

```python
def cross_attention(decoder_state, encoder_output):
    Q = self.W_q(decoder_state)    # Decoder에서 Query
    K = self.W_k(encoder_output)   # Encoder에서 Key
    V = self.W_v(encoder_output)   # Encoder에서 Value
    return attention(Q, K, V)
```

**역할:**
```
번역 예시: "I love you" → "나는 너를 사랑해"

Decoder가 "나는"을 생성할 때:
- Q: "나는"의 표현
- K, V: "I", "love", "you"의 표현
- Attention: "I"에 높은 가중치 → "I"의 정보 활용
```

**Encoder-Decoder에서의 구조:**
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        # Decoder Self-Attention (causal)
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model)

    def forward(self, x, encoder_output, tgt_mask):
        # 1. Masked Self-Attention
        x = x + self.self_attn(x, x, x, attn_mask=tgt_mask)[0]

        # 2. Cross-Attention
        x = x + self.cross_attn(x, encoder_output, encoder_output)[0]

        # 3. FFN
        x = x + self.ffn(x)
        return x
```

---

### Question)
```
시계열 예측에서 Encoder-Decoder 구조는 어떻게 활용되나요?
```

### Answer)
**시계열에서 Encoder-Decoder 활용:**

**구조:**
```
Encoder: 과거 시퀀스 인코딩 (look-back window)
Decoder: 미래 시퀀스 생성 (prediction horizon)
```

```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, input_dim, output_dim):
        super().__init__()
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(output_dim, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_decoder_layers
        )
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # src: (batch, look_back, input_dim)
        # tgt: (batch, horizon, output_dim)

        # Encoder
        src_emb = self.encoder_embedding(src)
        memory = self.encoder(src_emb)

        # Decoder
        tgt_emb = self.decoder_embedding(tgt)
        tgt_mask = self.generate_causal_mask(tgt.size(1))
        output = self.decoder(tgt_emb, memory, tgt_mask)

        return self.output_layer(output)
```

**Teacher Forcing vs Autoregressive:**

```python
# 학습: Teacher Forcing
# 실제 미래 값을 decoder 입력으로 사용
pred = model(past_values, future_values[:, :-1])
loss = criterion(pred, future_values[:, 1:])

# 추론: Autoregressive
# 예측값을 다음 입력으로 사용
def predict(model, past_values, horizon):
    predictions = []
    decoder_input = start_token

    for _ in range(horizon):
        pred = model(past_values, decoder_input)
        next_pred = pred[:, -1:]
        predictions.append(next_pred)
        decoder_input = torch.cat([decoder_input, next_pred], dim=1)

    return torch.cat(predictions, dim=1)
```

**대표 시계열 모델:**
- **Informer**: Sparse attention으로 긴 시퀀스 처리
- **Autoformer**: Auto-correlation 기반 attention
- **FEDformer**: 주파수 도메인 attention

**장점:**
- 가변 길이 예측 가능
- 과거-미래 관계 명시적 모델링
- Multi-step prediction에 효과적
