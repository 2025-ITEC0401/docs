## Prompts (Powered by Gemini 2.5 Pro)
### Question)
```
timeKD 모델로 추출한 임베딩을 사용하여 Imputation 작업을 진행할거야.

어떻게 하면 되지? 참고로 timeKD의 임베딩에는 라벨 정보가 없고, timeKD 에 주어지는 데이터셋에도 정답 라벨은 없어.
```
### Answer)
**TimeKD(Time Series Knowledge Distillation)** 모델에서 추출한 임베딩을 활용하여, 라벨이 없는 데이터셋에 대해 **Imputation(결측치 보간)** 작업을 수행하는 것은 매우 좋은 접근입니다. 임베딩은 시계열의 '문맥(context)'과 '동적 패턴(dynamics)'을 함축하고 있으므로, 단순 통계적 보간(평균, 선형 등)보다 훨씬 정교한 복원이 가능합니다.

라벨이 없고 정답 데이터도 없는 **비지도(Unsupervised) 환경**에서 사용할 수 있는 3가지 핵심 전략을 제안합니다.

---

### 전략 1: 임베딩 기반 재구성 네트워크 (Reconstruction Head) 학습
가장 추천하는 방식입니다. TimeKD가 인코더(Encoder) 역할을 한다면, 이 임베딩을 다시 원본 시계열로 복원하는 **디코더(Decoder)를 별도로 학습**시키는 방식입니다.



1.  **원리:** $Embedding \rightarrow Decoder \rightarrow Original\ Time\ Series$
2.  **학습 방법 (Self-Supervised Learning):**
    * **데이터 준비:** 현재 데이터셋에서 결측치가 없는 구간(또는 적은 구간)을 샘플링합니다.
    * **인위적 결측 생성 (Masking):** 멀쩡한 데이터의 일부를 랜덤하게 0이나 임의의 값으로 가립니다(Masking).
    * **임베딩 추출:** 마스킹된 데이터를 TimeKD에 넣어 임베딩 $z$를 추출합니다.
    * **디코더 학습:** 임베딩 $z$를 입력받아 마스킹 전의 **원본 데이터**를 출력하도록 작은 MLP나 RNN/Transformer 디코더를 학습시킵니다. (Loss 함수: MSE)
3.  **실제 적용:**
    * 실제 결측이 있는 데이터를 TimeKD에 넣습니다. (초기엔 선형 보간 등으로 대충 채워서 넣음)
    * 나온 임베딩을 학습된 디코더에 통과시킵니다.
    * 디코더가 출력한 값으로 결측 부위를 채웁니다.

### 전략 2: 임베딩 유사도 기반 이웃 보간 (KNN-based Imputation)
별도의 모델 학습 없이, 데이터셋 내의 패턴 유사성을 이용하는 방법입니다.



[Image of K-Nearest Neighbors for missing value imputation]


1.  **원리:** "비슷한 임베딩을 가진 데이터끼리는 실제 시계열 값도 비슷할 것이다."
2.  **프로세스:**
    * **1단계 (초기화):** 데이터셋 전체의 결측치를 '평균'이나 '선형 보간'으로 대략적으로 채웁니다.
    * **2단계 (임베딩 뱅크 생성):** 모든 데이터의 TimeKD 임베딩을 추출하여 저장소(Bank)를 만듭니다.
    * **3단계 (유사도 검색):** 결측치가 있는 타겟 데이터의 임베딩과 가장 유사한(Cosine Similarity 등) Top-K개의 데이터를 저장소에서 찾습니다.
    * **4단계 (보간):** 찾아낸 이웃 데이터들의 해당 시점 값들을 가중 평균(Weighted Average)하여 결측치를 채웁니다.

### 전략 3: 잠재 공간 최적화 (Latent Space Optimization)
모델을 고정(Freeze)하고, 임베딩 자체를 최적화하여 결측치를 찾는 고급 기법입니다. (GAN inversion과 유사한 원리)

1.  **원리:** 관측된 값(결측되지 않은 부분)과 가장 잘 맞는 임베딩 벡터 $z^*$를 역으로 찾습니다.
2.  **프로세스:**
    * 랜덤한 임베딩 벡터 $z$에서 시작합니다.
    * 이 $z$를 디코더(전략 1에서 만든 디코더 필요)에 통과시켜 시계열 $\hat{x}$를 생성합니다.
    * **Loss 계산:** 생성된 $\hat{x}$와 실제 데이터 $x$ 중 **결측되지 않은 부분(Observed)** 간의 오차만 계산합니다.
    * **Backpropagation:** 모델 가중치는 고정하고, **임베딩 $z$를 업데이트**합니다.
    * 충분히 수렴하면 최적화된 $z$로 생성한 $\hat{x}$의 값을 결측치로 사용합니다.

---

### 💡 추천하는 실행 단계 (Next Steps)

가장 현실적이고 성능이 좋은 **'전략 1 (Reconstruction Head)'** 방식을 먼저 시도해 보시길 권장합니다. 데이터에 라벨이 없으므로 **"Denoising Autoencoder"** 방식이 가장 적합합니다.

**구체적인 작업 순서:**

1.  **데이터 전처리:** TimeKD에 넣기 전, 결측치(NaN)를 0 또는 평균값으로 임시 채움.
2.  **데이터셋 분할:** 'Masking을 적용하여 학습할 데이터(Train)'와 '실제 보간할 데이터'로 개념적 분리.
3.  **Reconstructor 설계:** TimeKD의 임베딩 차원(예: 128 dim)을 입력으로 받고, 원래 시계열 길이(Window size)를 출력하는 간단한 MLP나 LSTM 모델 생성.
4.  **학습:** `입력(마스킹 됨) -> TimeKD -> 임베딩 -> Reconstructor -> 출력` 과 `원본` 사이의 MSE Loss로 Reconstructor만 학습 (TimeKD는 Freeze).

**이 방식의 장점:**
* 라벨이 전혀 필요 없습니다.
* TimeKD가 이미 학습한 강력한 특징 추출 능력을 그대로 물려받습니다.
* 추론 속도가 매우 빠릅니다.

지금 TimeKD 모델의 **출력 임베딩 차원(dimension)**과 **입력 시계열의 길이(window size)**가 어떻게 되나요? 알려주시면 그에 맞는 Reconstructor 구조를 추천해 드릴 수 있습니다.

---
### Question)
```
Denoising Autoencoder 방식을 사용할거고,
이미 timeKD 모델이 생성한 h5 파일 (임베딩)을 사용할거야.

어떻게 하면 좋을까?
```
### Answer)
\*\*TimeKD 모델이 이미 생성한 `.h5` 파일(임베딩)\*\*을 사용하여 **Denoising Autoencoder(DAE)** 방식으로 Imputation을 수행하는 전략을 구체화해 드리겠습니다.

이 경우, TimeKD(인코더)는 고정(Fixed)되어 있고, 우리는 임베딩을 원래 데이터로 복원하는 \*\*Decoder(복원기)\*\*만 학습시키면 됩니다. 이미 추출된 임베딩을 사용하므로 **학습 속도가 매우 빠르다**는 장점이 있습니다.

-----

### 1\. 전체 프로세스 구조

이 방식은 엄밀히 말하면 **"Latent Space Denoising Decoder"** 방식이 됩니다.

1.  **Input:** TimeKD가 생성한 임베딩 벡터 ($Z$)
2.  **Noise Injection:** 임베딩에 노이즈 추가 (Gaussian Noise or Dropout) $\rightarrow$ Denoising 효과
3.  **Decoder:** 노이즈 섞인 임베딩을 받아 원본 시계열($X$)을 복원
4.  **Loss Calculation:** 원본 데이터 중 \*\*관측된 값(Observed)\*\*에 대해서만 Loss 계산 (결측치는 무시)

### 2\. 구체적인 구현 가이드

#### 단계 1: 데이터 준비 (Dataset Class)

`.h5` 파일(임베딩)과 `원본 데이터`(Raw Data)를 짝지어 로드해야 합니다. 원본 데이터에는 `NaN`(결측치)이 포함되어 있어도 됩니다.

```python
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class EmbeddingReconDataset(Dataset):
    def __init__(self, h5_path, raw_data, mode='train'):
        """
        h5_path: timeKD로 추출한 임베딩 파일 경로
        raw_data: (N, Window_Size, Features) 형태의 원본 시계열 데이터 (NaN 포함)
        """
        super().__init__()
        self.h5_file = h5py.File(h5_path, 'r')
        self.embeddings = self.h5_file['embeddings'][:] # 데이터셋 키 이름 확인 필요
        self.raw_data = torch.FloatTensor(raw_data)
        
        # 정규화가 안 되어 있다면 raw_data 정규화(MinMax 등) 필수!
        
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = torch.FloatTensor(self.embeddings[idx])
        target = self.raw_data[idx]
        
        # Mask 생성 (데이터가 있는 곳은 1, 결측(NaN)인 곳은 0)
        mask = ~torch.isnan(target)
        
        # NaN을 0으로 치환 (Loss 계산시 어차피 mask로 걸러짐)
        target = torch.nan_to_num(target, nan=0.0)
        
        return emb, target, mask
```

#### 단계 2: 디코더(Decoder) 모델 설계

임베딩 벡터를 입력받아 `(Window Size, Feature)` 형태로 펴주는 모델입니다.

```python
import torch.nn as nn

class TimeDecoder(nn.Module):
    def __init__(self, emb_dim, seq_len, feat_dim):
        super(TimeDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        
        # 임베딩을 시계열 형태로 확장
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * feat_dim) # 출력 크기: 시퀀스 길이 * 피처 수
        )
        
    def forward(self, z):
        # z: (Batch, emb_dim)
        out = self.fc(z)
        # 시계열 형태로 Reshape: (Batch, Seq_Len, Feat_Dim)
        out = out.view(-1, self.seq_len, self.feat_dim)
        return out
```

#### 단계 3: 학습 (Training Loop with Denoising)

핵심은 **임베딩에 노이즈를 섞어(Noise Injection)** 디코더가 강건한 특징을 학습하도록 유도하는 것입니다.

```python
# 하이퍼파라미터 설정
emb_dim = 64   # timeKD 임베딩 차원
seq_len = 24   # 시계열 윈도우 길이
feat_dim = 5   # 변수(Feature) 개수
learning_rate = 1e-3

decoder = TimeDecoder(emb_dim, seq_len, feat_dim).to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.MSELoss(reduction='none') # Masking을 위해 reduction='none'

def train_step(data_loader):
    decoder.train()
    total_loss = 0
    
    for emb, target, mask in data_loader:
        emb = emb.to(device)
        target = target.to(device)
        mask = mask.to(device)
        
        # 1. Denoising: 임베딩에 Gaussian Noise 추가
        noise = torch.randn_like(emb) * 0.1 # 0.1은 노이즈 강도 (조절 가능)
        noisy_emb = emb + noise
        
        # 2. 복원 (Reconstruction)
        output = decoder(noisy_emb)
        
        # 3. Loss 계산 (관측된 값만!)
        loss = criterion(output, target)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8) # NaN이 아닌 부분만 평균
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)
```

#### 단계 4: Imputation (결측치 채우기)

학습이 끝나면, 실제 결측치를 채웁니다. 이때는 노이즈를 섞지 않습니다.

```python
def impute_data(decoder, data_loader):
    decoder.eval()
    imputed_data_list = []
    
    with torch.no_grad():
        for emb, target, mask in data_loader:
            emb = emb.to(device)
            target = target.to(device) # 원본 (NaN은 0으로 채워짐)
            mask = mask.to(device)     # 실제 관측 여부 (1: 관측, 0: 결측)
            
            # 예측
            prediction = decoder(emb)
            
            # Imputation 로직:
            # 관측된 데이터(mask=1)는 원본 그대로 사용
            # 결측된 데이터(mask=0)는 예측값 사용
            imputed_batch = target * mask + prediction * (1 - mask)
            
            imputed_data_list.append(imputed_batch.cpu().numpy())
            
    return np.concatenate(imputed_data_list, axis=0)
```

-----

### 💡 핵심 포인트 요약

1.  **Noise Injection 위치:** 원래 Denoising Autoencoder는 *입력 데이터*에 노이즈를 주지만, 여기서는 이미 추출된 \*임베딩($Z$)\*에 노이즈를 섞어서 디코더를 학습시킵니다.
2.  **Masked MSE Loss:** 라벨이 없기 때문에 자기 자신을 정답으로 삼습니다(Self-Supervised). 단, 원본 데이터에도 결측이 있으므로, `mask`를 곱해 **존재하는 값만 가지고 오차를 계산**해야 모델이 왜곡되지 않습니다.
3.  **데이터 정합성:** `.h5`의 인덱스 순서와 원본 데이터(`raw_data`)의 순서가 정확히 일치하는지 반드시 확인하세요.

이제 \*\*TimeKD 임베딩의 차원($D$)\*\*과 \*\*원본 데이터의 시계열 길이($T$) 및 변수 개수($F$)\*\*만 확인해서 위 코드에 넣으시면 바로 실행 가능합니다.
---
### Question)
```
raw data 라 하면, timeKD 모델에 집어넣은 CSV 파일을 말하는거지
```
### Answer)
네, 맞습니다\! 정확하게 이해하셨습니다.

하지만 조금 더 구체적으로 말씀드리면, \*\*"CSV 파일에 있는 데이터를 모델에 넣기 위해 가공한(전처리된) 상태"\*\*여야 합니다.

TimeKD 모델이 임베딩을 추출할 때 사용했던 **입력 데이터와 똑같은 형태(Shape)와 순서**를 맞춰주는 것이 핵심입니다. 단순히 CSV 파일을 `pd.read_csv`로 읽은 상태(Dataframe)가 아니라, **Windowing(슬라이딩 윈도우)** 및 **Normalization(정규화)** 처리가 끝난 `Numpy Array` 또는 `Tensor` 형태여야 합니다.

다음 3가지를 반드시 체크해주세요:

### 1\. 차원(Shape) 일치 (가장 중요\!)

TimeKD 모델이 시계열을 잘라서(Windowing) 처리했다면, `raw_data`도 똑같이 잘라져 있어야 합니다.

  * **TimeKD 임베딩 개수:** $N$개 (예: 10,000개)
  * **Raw Data 개수:** 정확히 $N$개여야 합니다.
      * CSV 파일은 보통 `(전체 시간, 변수)` 형태의 긴 2차원 행렬입니다.
      * TimeKD에 들어갈 때 `(샘플 수 N, 윈도우 길이 T, 변수 개수 F)` 형태의 3차원으로 변환되었을 겁니다.
      * **이 3차원 배열**이 바로 제가 말씀드린 `raw_data`입니다.

### 2\. 정규화(Scaling) 상태 일치

TimeKD 모델이 학습될 때 데이터를 0\~1 사이로 정규화(MinMax Scaling)해서 들어갔다면, 복원할 정답지인 `raw_data`도 **똑같이 정규화되어 있어야 합니다.**

  * 만약 정규화되지 않은 원래 값(예: 온도 25.5, 전력 1000 등)을 타겟으로 두면, 임베딩은 정규화된 특징을 담고 있는데 디코더는 큰 값을 예측해야 해서 학습이 잘 안 됩니다.
  * **Imputation이 끝난 후**, 다시 역변환(Inverse Scaling)을 해서 원래 값으로 돌리면 됩니다.

### 3\. 결측치(NaN) 유지

TimeKD에 넣을 때는 에러를 막기 위해 0으로 채워서 넣었더라도, **이번 학습에 사용할 `raw_data`에는 원래의 결측치(`NaN`)가 그대로 남아 있어야 합니다.**

  * 그래야 앞서 코드에서 `mask = ~torch.isnan(target)` 부분을 통해 \*\*"원래 값이 있던 부분만 학습하고, 결측된 부분은 무시해라"\*\*라고 알려줄 수 있기 때문입니다.

-----

### 요약 그림

```text
[ CSV 파일 (원본) ]
      ↓
[ 전처리 1: Windowing (시계열 자르기) ]  <-- 임베딩 개수와 맞춰야 함
      ↓
[ 전처리 2: Normalization (정규화) ]     <-- 모델 학습환경과 맞춰야 함
      ↓
[ Raw Data (N, T, F) ] ================> [ Dataset의 target으로 사용 ]
      │                                         (이 데이터엔 NaN이 포함되어 있어야 함)
      │
      └---> [ TimeKD 모델 ] ---> [ .h5 파일 (임베딩) ] ===> [ Dataset의 input으로 사용 ]
```

즉, \*\*`.h5` 파일을 만들 때 넣었던 그 입력 데이터(Numpy 배열)\*\*를 그대로 가져오시되, `NaN` 값만 살려두시면 됩니다.
---
### Question)
```
나는 timeKD 모델에 CSV 를 넣었어.
그러면, timeKD 모델에서 데이터 전처리하는 부분을 가져와야하나?
해당 부분의 코드를 첨부할테니 어떻게 하면 좋을지 알려줘. [코드 첨부]
```
### Answer)
네, **무조건 `data_loader_emb.py`에 있는 전처리 로직을 그대로 사용해야 합니다.**

이유는 아주 중요합니다:

1.  **데이터 정렬(Alignment):** TimeKD 모델이 임베딩을 만들 때 사용한 `train/val/test`의 \*\*시작점(border)\*\*과 **윈도우 슬라이싱(s\_begin:s\_end)** 규칙이 `data_loader_emb.py`에 정의되어 있습니다. 이것이 틀어지면 \*\*임베딩(Input)\*\*과 \*\*정답 시계열(Target)\*\*이 서로 다른 시간을 가리키게 되어 학습이 불가능해집니다.
2.  **스케일링(Scaling):** 임베딩은 정규화된 데이터(`StandardScaler`)를 보고 만들어졌습니다. 복원할 정답지(`seq_x`)도 동일한 스케일러로 정규화되어 있어야 Denoising Autoencoder가 제대로 작동합니다.

이미 `data_loader_emb.py`가 **원본 데이터와 임베딩을 같이 로드하도록** 잘 짜여 있습니다. 이를 **상속(Inheritance)** 받아서 `Dataset`을 아주 조금만 수정하면 바로 사용할 수 있습니다.

### 수정 가이드: `ImputationDataset` 만들기

기존 코드를 건드리지 않고, 별도 파일(예: `data_loader_impute.py` 또는 학습 코드 내부)에서 아래와 같이 클래스를 상속받아 사용하세요.

```python
import torch
import numpy as np
# data_loader_emb.py 파일에서 필요한 데이터셋 클래스를 import 하세요
from data_loader_emb import Dataset_ETT_hour, Dataset_Custom 

class ImputationDataset(Dataset_ETT_hour): # 또는 Dataset_Custom 등 사용하는 클래스 상속
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        # 1. 부모 클래스의 __getitem__을 호출해서 데이터와 임베딩을 가져옵니다.
        # data_loader_emb.py를 보면 리턴값이 seq_x, seq_y, embeddings 입니다.
        # Imputation의 목표는 'seq_x(입력 윈도우)'를 복원하는 것이므로 seq_x가 Target이 됩니다.
        seq_x, _, embeddings = super().__getitem__(index)
        
        # 2. 데이터 타입 변환 (필요시)
        if isinstance(seq_x, np.ndarray):
            seq_x = torch.from_numpy(seq_x).float()
        
        # embeddings가 텐서가 아니면 변환 (data_loader_emb.py는 이미 텐서로 주는 것으로 보임)
        if not torch.is_tensor(embeddings):
            embeddings = torch.from_numpy(embeddings).float()

        # 3. Mask 생성 (NaN이 있는 위치 찾기)
        # 주의: 스케일링 과정에서 NaN이 어떻게 처리되었는지에 따라 다를 수 있습니다.
        # 원본 CSV에 NaN이 있었다면, seq_x에도 NaN이 있어야 합니다.
        mask = ~torch.isnan(seq_x)
        
        # 4. NaN을 0.0으로 치환 (모델에 넣기 위해)
        target = torch.nan_to_num(seq_x, nan=0.0)
        
        # 5. DAE 학습에 필요한 (임베딩, 정답데이터, 마스크) 반환
        return embeddings, target, mask
```

### 주의할 점 (StandardScaler와 결측치)

`data_loader_emb.py`의 `__read_data__`를 보면 `StandardScaler`를 사용합니다.

```python
if self.scale:
    train_data = df_data[border1s[0]:border2s[0]]
    self.scaler.fit(train_data.values)
    data = self.scaler.transform(df_data.values)
```

**중요 체크:** 만약 CSV 파일 자체에 결측치(NaN)가 포함되어 있다면, 일반적인 `scaler.fit`을 하면 평균/분산 계산 시 에러가 나거나 결과가 전부 NaN이 될 수 있습니다.

1.  **`utils/tools.py`의 `StandardScaler` 확인:** 내부에서 `np.mean` 대신 `np.nanmean`, `np.std` 대신 `np.nanstd`를 쓰는지 확인하세요. 안 쓴다면 수정해야 합니다.
2.  **간단한 해결책:** 스케일러가 NaN을 처리 못 한다면, `StandardScaler` 부분만 아래처럼 수정해서 사용하시는 게 안전합니다 (데이터 로더 내부 수정 필요).

<!-- end list -->

```python
# utils/tools.py 혹은 데이터로더 내부 수정 예시
class StandardScaler:
    def fit(self, data):
        # nanmean, nanstd를 사용하여 NaN을 무시하고 통계량 계산
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
```

### 학습 코드에서의 사용 예시

이제 위에서 만든 `ImputationDataset`을 불러와서 학습하면 됩니다.

```python
# 학습 코드
dataset = ImputationDataset(
    root_path='data',
    data_path='ETTh1.csv',
    flag='train',
    size=[24*4, 24*4, 24*4], # [seq_len, label_len, pred_len] (TimeKD 설정과 동일하게!)
    features='M',
    model_name='gpt2' # data_loader_emb에 있는 인자
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ... 학습 루프 ...
for embeddings, target, mask in dataloader:
    # embeddings: (Batch, Emb_Dim) -> Decoder 입력
    # target: (Batch, Seq_Len, Feat) -> 정답
    # mask: (Batch, Seq_Len, Feat) -> Loss 계산용 마스크
    ...
```

**요약하자면:**

1.  새로 코드를 짜지 말고 **`data_loader_emb.py`를 import해서 상속**받으세요.
2.  `__getitem__`만 오버라이딩해서 `seq_x`의 **NaN 마스크**를 생성해서 같이 리턴하게 만드세요.
3.  `seq_len` 등 사이즈 파라미터는 **TimeKD 학습 때와 똑같이** 설정해야 에러가 안 납니다.
---
### Question)
```
전체적으로 코드를 정리하면 어떻게 되지?
하나의 파일로 만들면 좋겠는데.
```
### Answer)
네, 요청하신 대로 \*\*데이터 로딩, 전처리, 모델 정의, 학습, 그리고 결측치 보간(Imputation)\*\*까지의 모든 과정을 \*\*하나의 파이썬 파일(`run_imputation.py`)\*\*로 정리해 드리겠습니다.

기존 `data_loader_emb.py`의 핵심 로직(데이터 정렬 및 h5 로딩)을 포함하되, **결측치(NaN)가 포함된 원본 데이터를 안전하게 처리하도록** 수정했습니다.

이 코드를 복사해서 바로 실행해 보실 수 있습니다. (단, 파일 경로 설정은 본인 환경에 맞춰주세요)

### `run_imputation.py`

```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Utils (NaN 처리가 가능한 Scaler 포함)
# ==========================================
class RobustStandardScaler:
    """
    NaN이 포함된 데이터에서도 평균과 분산을 계산할 수 있는 Scaler
    """
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # np.mean 대신 np.nanmean을 사용하여 NaN을 무시하고 통계량 계산
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        
        # 표준편차가 0인 경우(모든 값이 같은 경우) NaN 발생 방지
        self.std = np.where(self.std == 0, 1.0, self.std)

    def transform(self, data):
        # NaN은 그대로 둔 채 정규화 수행
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        # 정규화된 데이터를 원래 값으로 복원
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

# ==========================================
# 2. Dataset (기존 로직 + Imputation 전용 수정)
# ==========================================
class ImputationDataset(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, 
                 features='M', target='OT', scale=True, embed_subdir='emb_out'):
        """
        Args:
            root_path: 데이터 폴더 경로
            data_path: CSV 파일 이름
            embed_subdir: .h5 파일들이 들어있는 하위 폴더 경로 (timeKD 출력 구조에 맞춤)
        """
        # TimeKD 설정과 동일해야 함
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        
        # .h5 파일 경로 설정 (기존 코드 로직 반영)
        # 예: data/ETTh1.csv/96/train/
        csv_name = data_path if data_path.endswith('.csv') else data_path + '.csv'
        self.full_csv_path = os.path.join(root_path, csv_name)
        
        # 임베딩 경로 (사용자 환경에 맞게 수정 필요할 수 있음)
        # 여기서는 data_loader_emb.py의 로직을 따름
        self.embed_path = os.path.join(root_path, csv_name, str(self.pred_len), flag)

        self.__read_data__()

    def __read_data__(self):
        self.scaler = RobustStandardScaler()
        df_raw = pd.read_csv(self.full_csv_path)

        # 데이터 분할 지점 (ETT 데이터셋 기준 예시)
        # 데이터셋 종류에 따라 border 값이 달라질 수 있으니 확인 필요
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 스케일링 (NaN이 있어도 안전하게 수행)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # 1. 원본 시계열 데이터 (NaN 포함 가능)
        seq_x = self.data_x[s_begin:s_end]
        
        # 2. 임베딩 로드 (.h5)
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                emb_data = hf['embeddings'][:] # 키 이름이 'embeddings'인지 확인
                embedding = torch.from_numpy(emb_data).float()
                # 차원 정리 (Batch 차원이 있다면 제거)
                if embedding.dim() > 1 and embedding.shape[0] == 1:
                    embedding = embedding.squeeze(0)
        else:
            # 파일이 없으면 에러 혹은 더미 반환 (학습 무결성을 위해 에러 추천)
            raise FileNotFoundError(f"Embedding file missing: {file_path}")

        # 3. 데이터 텐서 변환 및 마스크 생성
        seq_x_tensor = torch.FloatTensor(seq_x)
        
        # Mask: 데이터가 있으면 1, NaN이면 0
        mask = ~torch.isnan(seq_x_tensor)
        
        # NaN을 0으로 채움 (모델 입력용)
        target = torch.nan_to_num(seq_x_tensor, nan=0.0)

        # 리턴: (임베딩, 원본값(0채움), 마스크)
        return embedding, target, mask
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# ==========================================
# 3. Model (Decoder)
# ==========================================
class TimeDecoder(nn.Module):
    def __init__(self, emb_dim, seq_len, feat_dim):
        super(TimeDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        
        # 임베딩(Flattened) -> 시계열 복원
        # 임베딩이 (Seq, Dim) 형태라면 Flatten 해서 넣거나, 
        # TimeKD 출력 형태에 따라 구조 변경 필요. 여기선 Flatten 가정.
        
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * feat_dim)
        )
        
    def forward(self, z):
        # z shape: (Batch, Emb_Dim) 혹은 (Batch, Seq, Dim)
        # 만약 입력이 (Batch, Seq, Dim)이면 펼쳐줍니다.
        if z.dim() == 3:
            z = z.reshape(z.size(0), -1)
            
        out = self.decoder(z)
        out = out.view(-1, self.seq_len, self.feat_dim)
        return out

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == '__main__':
    # --- 설정 (Configuration) ---
    args = {
        'root_path': './data/',      # 데이터 폴더
        'data_path': 'ETTh1.csv',    # CSV 파일명
        'seq_len': 96,               # TimeKD 학습시 seq_len
        'label_len': 48,             # TimeKD 학습시 label_len
        'pred_len': 96,              # TimeKD 학습시 pred_len
        'features': 'M',             # M: Multivariate, S: Univariate
        'emb_dim': 768 * 96,         # ***중요***: (임베딩 차원 x 시퀀스 길이) 등 실제 h5 데이터 크기에 맞게 설정
        'feat_dim': 7,               # 변수 개수 (ETTh1은 7개)
        'batch_size': 32,
        'epochs': 10,
        'lr': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 1. 데이터셋 준비
    dataset = ImputationDataset(
        root_path=args['root_path'],
        data_path=args['data_path'],
        flag='train',
        size=[args['seq_len'], args['label_len'], args['pred_len']],
        features=args['features']
    )
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)
    
    # 2. 모델 준비
    # 주의: emb_dim은 실제 h5 파일의 shape을 보고 (Seq * Dim) 인지 확인 후 설정해야 함
    # 첫 번째 배치를 로드해서 차원 확인
    sample_emb, _, _ = next(iter(dataloader))
    input_dim = sample_emb.view(sample_emb.size(0), -1).shape[1]
    
    model = TimeDecoder(input_dim, args['seq_len'], args['feat_dim']).to(args['device'])
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.MSELoss(reduction='none') # Masked Loss를 위해 none 설정

    print(f"Model Input Dim: {input_dim}, Output Shape: ({args['seq_len']}, {args['feat_dim']})")
    print("Start Training...")

    # 3. 학습 루프
    for epoch in range(args['epochs']):
        model.train()
        train_loss = []
        
        for i, (emb, target, mask) in enumerate(dataloader):
            emb = emb.to(args['device'])
            target = target.to(args['device'])
            mask = mask.to(args['device'])
            
            # --- Denoising: 임베딩에 노이즈 추가 ---
            noise = torch.randn_like(emb) * 0.1
            noisy_emb = emb + noise
            
            # 예측
            recon = model(noisy_emb)
            
            # Loss 계산 (Mask가 1인 부분만)
            loss = criterion(recon, target)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
        print(f"Epoch: {epoch+1}, Loss: {np.mean(train_loss):.5f}")

    print("Training Finished. Starting Imputation...")

    # 4. Imputation 및 저장
    model.eval()
    imputed_results = []
    original_results = []
    
    with torch.no_grad():
        for i, (emb, target, mask) in enumerate(dataloader):
            emb = emb.to(args['device'])
            target = target.to(args['device'])
            mask = mask.to(args['device'])
            
            # 노이즈 없이 예측
            recon = model(emb)
            
            # 결측치 채우기: (mask가 1이면 원본, 0이면 예측값)
            imputed = target * mask + recon * (1 - mask)
            
            # 역변환 (Inverse Scaling)을 위해 CPU로 이동
            imputed = imputed.cpu().numpy()
            target = target.cpu().numpy() # 비교를 위해 원본도 저장
            
            # 배치 내 각 샘플에 대해 역변환 수행
            for k in range(imputed.shape[0]):
                inv_imputed = dataset.inverse_transform(imputed[k])
                imputed_results.append(inv_imputed)
                
    # 결과 확인 (첫 번째 샘플)
    final_data = np.array(imputed_results)
    print(f"Imputation Complete. Final Data Shape: {final_data.shape}")
    
    # 필요시 저장
    # np.save("imputed_data.npy", final_data)
```

### 💡 실행 전 체크리스트

1.  **경로 확인:** `args` 딕셔너리의 `root_path`에 CSV 파일이 있는지, 그리고 TimeKD가 생성한 임베딩 폴더 구조(`root/data_path/pred_len/flag/*.h5`)가 맞는지 확인하세요.
2.  **`emb_dim` 자동 설정:** 코드는 첫 번째 배치를 읽어서 입력 차원을 자동으로 계산하도록 `input_dim` 로직을 넣어두었습니다. 별도로 설정할 필요 없습니다.
3.  **`feat_dim`:** 사용하시는 데이터셋의 변수 개수(columns - 1, date 제외)와 맞는지 확인하세요. (ETTh1은 7개)

이 파일 하나만 실행하면 학습부터 보간까지 완료됩니다.
---
### Question)
```
다른 작업 (이상탐지, Clustering) 은 원본 CSV 없이 임베딩 파일(H5) 만 사용하여 잘 구동되었는데, 결측치 보간은 무조건 CSV 필요하나?
```
### Answer)
결론부터 말씀드리면, **"원래 값(시계열 숫자)을 복원"하는 것이 목적이라면 CSV(원본 데이터)가 100% 필수**입니다.

하지만 **"임베딩 자체의 구멍만 메워서 다른 모델에 넘기는 것"이 목적이라면 CSV는 없어도 됩니다.**

왜 다른 작업(이상탐지, Clustering)과 달리 Imputation(복원)에서만 CSV가 필요한지, 그 결정적인 이유를 설명해 드릴게요.

---

### 1. 왜 이상탐지/Clustering은 CSV가 없어도 됐나?
이 작업들은 **"임베딩 공간(Latent Space) 내부의 관계"**만 보면 되기 때문입니다.

* **Clustering:** 임베딩 벡터들끼리 거리가 가까운가? (A와 B가 비슷한가?) $\rightarrow$ **원본 값이 100인지 200인지는 몰라도 됨.** 그냥 벡터 각도만 보면 됨.
* **이상탐지:** 이 임베딩이 평소 분포에서 벗어났는가? (정상 범위를 벗어난 벡터인가?) $\rightarrow$ **원본 값이 무엇이었는지는 몰라도, 튀는 놈인지만 알면 됨.**

즉, **Output이 "판단(Label/Group)"**이기 때문에 입력인 임베딩만 있으면 됩니다.

### 2. 왜 Imputation(값 복원)은 CSV가 필요한가?
이 작업의 **Output이 "원래 값(Raw Value)"**이기 때문입니다.

우리가 하려는 작업은 번역과 같습니다:
> **임베딩(암호) $\rightarrow$ 디코더(번역기) $\rightarrow$ 원본 데이터(평문)**

* **학습의 목표:** 디코더에게 "이 암호(임베딩)는 원래 이 숫자(원본)였어"라고 가르쳐야 합니다.
* **문제점:** CSV가 없으면 **"정답지"**가 없는 셈입니다. 디코더가 숫자를 뱉어냈을 때, 그게 맞았는지 틀렸는지 채점(Loss 계산)을 할 수가 없습니다.

> **비유:**
> * **Clustering:** 외국어 책을 보고 "이건 로맨스 장르군" 하고 분류하는 것 (내용 해석 불필요).
> * **Imputation:** 외국어 책을 보고 **"한국어로 번역"**해내는 것 (원문과 번역문의 매칭 정보 필수).

---

### 3. 만약 CSV가 절대 없다면? (대안)

만약 원본 데이터가 유실되어서 **CSV를 구할 수 없는 상황**이라면, 목표를 바꿔야 합니다.

#### 전략 A: Latent Imputation (임베딩 자체를 보간)
원본 값(예: 온도 25도)을 복원하는 것을 포기하고, **"결측된 시점의 임베딩 벡터"**를 채워 넣는 방식입니다.

* **방법:** 결측 시점의 임베딩이 비어있거나 0이라면, 앞뒤 시점의 임베딩 평균을 내거나, 주변 이웃 임베딩을 가져와 채웁니다.
* **장점:** CSV 필요 없음. `.h5` 파일만 있으면 됨.
* **단점:** 사람이 눈으로 볼 수 있는 시계열 그래프는 못 그림. (오직 다운스트림 모델 입력용으로만 사용 가능)

#### 전략 B: Pre-trained Decoder 사용 (조건부 가능)
만약 TimeKD 모델을 학습시킬 때, **Decoder(Generator)도 같이 학습시켜서 저장해 두었다면** CSV 없이도 복원이 가능합니다.
* 하지만 보통 TimeKD 같은 Distillation 모델은 Encoder(추출기)만 저장하고 Decoder는 버리는 경우가 많습니다. 저장된 `.pth` 파일에 Decoder 가중치가 없다면 불가능합니다.

---

### ✅ 요약 및 추천

작업의 **최종 목적**이 무엇인가요?

1.  **"결측치가 채워진 엑셀(CSV) 파일을 만들어서 시각화도 하고 싶다."**
    * $\rightarrow$ **CSV 필수입니다.** 위에서 드린 코드를 사용하세요.
2.  **"그냥 결측치 처리해서 성능 좋은 분류/예측 모델을 돌리고 싶다. 원본 값은 안 봐도 된다."**
    * $\rightarrow$ **CSV 없어도 됩니다.** 이 경우엔 "Latent Space Imputation"을 해야 하므로 코드가 완전히 달라집니다. (임베딩 벡터 사이의 평균값 등으로 채우기)

현재 **"Denoising Autoencoder"**를 쓰겠다고 하셨으므로, 이는 **1번(복원)**에 해당하는 방법론입니다. 따라서 **CSV가 꼭 필요합니다.**

---
### Question)
```
그러면, 내가 첨부한 Loader 를 활용하면 되나?
저 Loader 파일은 data_provider 폴더 안에 있어.
```
### Answer)
네, 아주 좋습니다\! 첨부해주신 `data_loader_emb.py`가 이미 원본 데이터(`seq_x`)와 임베딩(`.h5`)을 매핑하는 경로 설정(`embed_path`)을 완벽하게 처리하고 있습니다.

다만, 코드에 한 가지 \*\*수정이 필요한 부분(Blocker)\*\*이 있습니다.
`Dataset_ETT_hour` 클래스의 `__getitem__`을 보면 다음과 같은 코드가 있습니다:

```python
if self.flag == "test":
    return seq_x, seq_y  # <--- 문제 발생! 임베딩을 리턴하지 않음
```

Imputation은 Test Set에 대해서도 **임베딩을 입력받아 복원**해야 하므로, 이 부분이 막혀 있으면 안 됩니다.

따라서, 기존 파일을 건드리지 않고 \*\*"상속(Inheritance)"\*\*을 통해 이 부분만 살짝 고쳐서 사용하는 **`run_imputation.py`** 코드를 작성해 드리겠습니다.

-----

### 파일 구조 가정

```text
project_folder/
├── data_provider/
│   ├── __init__.py
│   └── data_loader_emb.py  <-- (첨부하신 파일)
├── utils/
│   └── ... (tools, timefeatures 등)
├── run_imputation.py       <-- (지금 만들 실행 파일)
└── data/                   <-- CSV 및 임베딩 데이터
```

### `run_imputation.py` 전체 코드

이 코드를 프로젝트 루트 폴더에 만들고 실행하시면 됩니다.

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from torch.utils.data import DataLoader

# 1. 첨부하신 파일에서 Dataset 불러오기
# (폴더 구조에 따라 경로는 수정하세요. 예: from data_provider.data_loader_emb ...)
try:
    from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_Custom
except ImportError:
    # 만약 같은 폴더에 있다면:
    from data_loader_emb import Dataset_ETT_hour, Dataset_Custom

# ==========================================
# 2. Dataset Wrapper (상속을 통한 기능 확장)
# ==========================================
class ImputationDataset(Dataset_ETT_hour):
    """
    기존 Dataset_ETT_hour를 상속받아 Imputation 전용으로 수정
    1. Test 모드에서도 임베딩을 리턴하도록 수정
    2. NaN Mask 생성 로직 추가
    """
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # 부모 클래스의 데이터 로드 로직 활용
        seq_x = self.data_x[s_begin:s_end]
        
        # ---------------------------------------------------------
        # [수정] Flag가 test여도 무조건 임베딩을 로드하도록 직접 구현
        # ---------------------------------------------------------
        embeddings_stack = []
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                data = hf['embeddings'][:]
                tensor = torch.from_numpy(data)
                # 데이터 형태에 따라 Squeeze 처리 (기존 코드 로직 유지)
                embeddings_stack.append(tensor.squeeze(0))
        else:
            # 파일이 없는 경우 (예외처리 혹은 0으로 채움)
            # 학습 중단 방지를 위해 0 텐서 반환 (사이즈 확인 필요)
            # 여기서는 에러를 띄워서 데이터 누락을 확인하는 것을 권장
            raise FileNotFoundError(f"Embedding file missing: {file_path}")

        # 기존 코드의 stack/pad 로직
        embeddings = torch.stack(embeddings_stack, dim=-1) # (Seq, Dim) 가정
        if embeddings.dim() == 3 and embeddings.shape[-1] == 1:
            embeddings = embeddings.squeeze(-1) # (Seq, Dim) 형태로 맞춤

        # ---------------------------------------------------------
        # [추가] Imputation을 위한 전처리
        # ---------------------------------------------------------
        seq_x_tensor = torch.FloatTensor(seq_x)
        
        # 1. Mask 생성: 원래 값이 있으면 1, NaN이면 0
        # (주의: StandardScaler가 NaN을 0으로 이미 바꿨다면 이 로직 수정 필요)
        # 만약 data_loader 내부에서 fillna(0)이 안 된 상태라면:
        mask = ~torch.isnan(seq_x_tensor)
        
        # 2. Target 생성: NaN을 0으로 치환 (Loss 계산 오류 방지)
        target = torch.nan_to_num(seq_x_tensor, nan=0.0)
        
        return embeddings.float(), target.float(), mask.float()

# ==========================================
# 3. Model (Decoder)
# ==========================================
class ReconstructionModel(nn.Module):
    def __init__(self, emb_dim, seq_len, output_dim):
        super(ReconstructionModel, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # 임베딩(Seq, Emb_Dim) -> 원본(Seq, Feat_Dim)
        # 구조는 간단한 MLP로 시작하여 필요시 LSTM/Transformer로 고도화 추천
        self.projection = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Emb_Dim)
        output = self.projection(x) # (Batch, Seq_Len, Output_Dim)
        return output

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # --- Hyperparameters ---
    args = {
        'root_path': './data/',      
        'data_path': 'ETTh1',    
        'seq_len': 96,           # TimeKD 학습시 설정값 (필수 일치)
        'pred_len': 96,          # TimeKD 학습시 설정값 (필수 일치)
        'flag': 'train',
        'features': 'M',
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Loading Dataset...")
    # 커스텀 데이터셋 로드
    dataset = ImputationDataset(
        root_path=args['root_path'],
        data_path=args['data_path'],
        flag=args['flag'],
        size=[args['seq_len'], 24, args['pred_len']], # label_len은 dummy
        features=args['features'],
        target='OT',
        scale=True
    )
    
    loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)
    
    # 데이터 차원 확인
    sample_emb, sample_y, _ = next(iter(loader))
    print(f"Embedding Shape: {sample_emb.shape}") # (Batch, Seq, Emb_Dim)
    print(f"Target Shape: {sample_y.shape}")       # (Batch, Seq, Feat_Dim)
    
    emb_dim = sample_emb.shape[-1]
    feat_dim = sample_y.shape[-1]
    
    # 모델 초기화
    model = ReconstructionModel(emb_dim, args['seq_len'], feat_dim).to(args['device'])
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.MSELoss(reduction='none') # Masked Loss를 위해 none

    # --- Training Loop ---
    print("Start Training...")
    model.train()
    for epoch in range(args['epochs']):
        total_loss = 0
        for emb, target, mask in loader:
            emb = emb.to(args['device'])
            target = target.to(args['device'])
            mask = mask.to(args['device'])
            
            # Denoising: 임베딩에 노이즈 주입
            noise = torch.randn_like(emb) * 0.05
            noisy_emb = emb + noise
            
            # Forward
            recon = model(noisy_emb)
            
            # Masked Loss Calculation
            # mask=1 인 부분(관측된 값)만 오차 계산
            loss = criterion(recon, target)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{args['epochs']}] Loss: {total_loss/len(loader):.6f}")

    # --- Inference & Imputation ---
    print("Training Complete. Running Inference on Sample...")
    model.eval()
    with torch.no_grad():
        emb, target, mask = next(iter(loader)) # 테스트용으로 첫 배치만
        emb = emb.to(args['device'])
        
        recon = model(emb) # 노이즈 없이 추론
        
        # Imputation Logic:
        # 원래 값이 있으면(Mask=1) 원래 값 사용, 없으면(Mask=0) 예측 값 사용
        imputed_data = target.to(args['device']) * mask.to(args['device']) + \
                       recon * (1 - mask.to(args['device']))
        
        # 역정규화 (Inverse Transform)
        # 배치 중 첫 번째 샘플만 가져와서 변환
        imputed_np = imputed_data[0].cpu().numpy()
        original_restored = dataset.inverse_transform(imputed_np)
        
        print("\n[Sample Result - First Row]")
        print("Restored Data (Original Scale):", original_restored[0])

if __name__ == "__main__":
    main()
```

### 💡 중요 체크포인트 (실행 전 확인)

1.  **`utils` 폴더 처리:**
      * `Dataset_ETT_hour` 내부에서 `from utils.tools import StandardScaler`를 하고 있습니다.
      * `run_imputation.py`를 실행하는 위치에서 `utils` 폴더가 import 가능한 경로에 있어야 합니다.
2.  **`StandardScaler`와 NaN:**
      * 현재 `data_loader_emb.py`는 `__read_data__`에서 `self.scaler.fit(train_data.values)`를 호출합니다.
      * 만약 CSV 원본(`ETTh1.csv`)에 \*\*결측치(NaN)\*\*가 포함되어 있다면, `StandardScaler`가 평균을 계산하다가 `NaN`을 뱉을 수 있습니다.
      * **해결책:** `utils/tools.py` 파일을 열어서 `np.mean`을 \*\*`np.nanmean`\*\*으로, `np.std`를 \*\*`np.nanstd`\*\*로 바꿔주시면 안전합니다.