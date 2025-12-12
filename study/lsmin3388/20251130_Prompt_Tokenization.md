# Tokenization 학습 자료

## Prompts

### Question)
```
NLP에서 Tokenization이란 무엇이고, 왜 중요한가요? 주요 토크나이저의 종류와 특징을 설명해주세요.
```

### Answer)
**Tokenization 정의:**

텍스트를 모델이 처리할 수 있는 작은 단위(토큰)로 분할하는 과정입니다.

**중요성:**
1. 어휘 크기 관리: 무한한 단어를 유한한 어휘로 매핑
2. OOV(Out-of-Vocabulary) 처리: 학습되지 않은 단어 처리
3. 언어 독립성: 다양한 언어에 적용 가능
4. 의미 단위 분할: 서브워드 수준의 의미 포착

**주요 토크나이저 종류:**

**1. Word-level Tokenization**
```python
text = "I love machine learning"
tokens = text.split()
# ['I', 'love', 'machine', 'learning']
```
- 단점: 큰 어휘 크기, OOV 문제

**2. Character-level Tokenization**
```python
tokens = list(text)
# ['I', ' ', 'l', 'o', 'v', 'e', ...]
```
- 장점: OOV 없음
- 단점: 긴 시퀀스, 의미 손실

**3. Subword Tokenization (현재 표준)**
```python
# "unhappiness" → ["un", "happiness"] 또는 ["un", "happy", "ness"]
```
- BPE, WordPiece, Unigram 등
- OOV 최소화 + 적절한 어휘 크기

**비교:**

| 방식 | 어휘 크기 | OOV | 시퀀스 길이 |
|------|----------|-----|-----------|
| Word | 매우 큼 | 많음 | 짧음 |
| Char | 작음 | 없음 | 매우 김 |
| Subword | 적절 | 적음 | 중간 |

---

### Question)
```
BPE(Byte Pair Encoding)와 WordPiece의 차이점은 무엇인가요? 각각 어떤 모델에서 사용되나요?
```

### Answer)
**BPE (Byte Pair Encoding)**

빈도 기반으로 문자 쌍을 반복 병합합니다.

```python
# 알고리즘:
# 1. 모든 문자를 개별 토큰으로 시작
# 2. 가장 빈번한 연속 쌍을 병합
# 3. 원하는 어휘 크기까지 반복

vocab = ['l', 'o', 'w', 'e', 'r', ...]
# 'l' + 'o' → 'lo' (빈도 기반 병합)
# 'lo' + 'w' → 'low'
```

**사용 모델:** GPT-2, GPT-3, RoBERTa

**WordPiece**

우도(likelihood) 기반으로 병합합니다.

```python
# 알고리즘:
# 병합 후 코퍼스 우도 증가가 최대인 쌍 선택

# 병합 결정 기준:
# score = freq(ab) / (freq(a) * freq(b))
# → 개별적으로 드물지만 함께 자주 등장하는 쌍 우선
```

**사용 모델:** BERT, DistilBERT, ELECTRA

**주요 차이점:**

| 특성 | BPE | WordPiece |
|------|-----|-----------|
| 병합 기준 | 빈도 | 우도 |
| 접두사 | 없음 | '##' 사용 |
| 희귀 토큰 | 그대로 | 분리 경향 |

```python
# WordPiece 예시 (BERT)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("unhappiness")
# ['un', '##happiness'] 또는 ['un', '##hap', '##pi', '##ness']

# BPE 예시 (GPT-2)
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer.tokenize("unhappiness")
# ['un', 'happiness'] 또는 ['Ġun', 'happ', 'iness']
```

---

### Question)
```
시계열 데이터에서 Tokenization 개념은 어떻게 적용되나요?
```

### Answer)
**시계열에서의 Tokenization:**

연속적인 시계열 값을 이산적인 토큰으로 변환합니다.

**1. Quantization 기반**
```python
def quantize_time_series(values, n_bins=1024):
    """연속 값을 이산 토큰으로 변환"""
    min_val, max_val = values.min(), values.max()
    bins = np.linspace(min_val, max_val, n_bins)
    tokens = np.digitize(values, bins)
    return tokens

# 예: 0.5 → 토큰 512, 0.75 → 토큰 768
```

**2. VQ-VAE 기반 (학습된 코드북)**
```python
class VQTokenizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim):
        super().__init__()
        self.codebook = nn.Embedding(n_embeddings, embedding_dim)

    def forward(self, z):
        # z를 가장 가까운 코드북 벡터로 매핑
        distances = torch.cdist(z, self.codebook.weight)
        tokens = distances.argmin(dim=-1)
        return tokens
```

**3. Patch 기반 (PatchTST 스타일)**
```python
def patchify(time_series, patch_size, stride):
    """시계열을 패치 단위로 분할"""
    patches = []
    for i in range(0, len(time_series) - patch_size + 1, stride):
        patch = time_series[i:i + patch_size]
        patches.append(patch)
    return np.array(patches)

# 각 패치가 하나의 "토큰"으로 취급됨
```

**시계열 토크나이저 장점:**

| 방법 | 장점 | 단점 |
|------|------|------|
| Quantization | 단순, 해석 가능 | 정보 손실 |
| VQ-VAE | 학습된 표현 | 복잡 |
| Patch | 효율적, 컨텍스트 | 경계 처리 |

**TimeGPT, Lag-Llama 등 최신 모델:**
```python
# LLM 스타일 시계열 모델
class TimeSeriesTokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size

    def encode(self, values):
        # 정규화
        normalized = (values - values.mean()) / values.std()
        # 양자화
        tokens = ((normalized + 3) / 6 * self.vocab_size).astype(int)
        return np.clip(tokens, 0, self.vocab_size - 1)

    def decode(self, tokens):
        # 역변환
        values = (tokens / self.vocab_size * 6) - 3
        return values
```
