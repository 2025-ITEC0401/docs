## Prompts (Powered by Gemini 2.5 Pro)
### Question)
```
weather.csv 데이터셋을 사용하여, timeKD 모델에 대한 임베딩을 추출하려고 해.
파라미터가 대략 21개쯤 되는데, output length 를 24로 하면, 약 23GB 정도의 VRAM을 사용하는 것으로 파악돼.
output length = (24, 36, 48, 96, 192) 에 대한 임베딩을 뽑아내야 하는데,
length >= 32 일 때는 VRAM Exceed 발생해.

RTX3090 24GB 그래픽카드 두 장을 사용할 수 있는 상태에서, 어떻게 length>=32 에 대한 임베딩을 뽑아낼 수 있을까?
```
### Answer)
안녕하세요! TimeKD 모델과 weather.csv 데이터셋으로 임베딩을 추출하는 과정에서 VRAM 문제에 직면하셨군요. output_length가 커질수록 VRAM 사용량이 급증하는 것은 자연스러운 현상입니다. RTX 3090 24GB 두 장을 활용하여 이 문제를 해결할 수 있는 효과적인 방법들을 단계별로 알려드릴게요.

가장 핵심적인 해결책은 **모델 병렬화(Model Parallelism)**와 혼합 정밀도(Mixed-Precision) 연산을 함께 사용하는 것입니다.

## 1단계: 혼합 정밀도(Mixed-Precision) 적용하기
가장 먼저 시도해 볼 수 있는 가장 간단하면서도 효과적인 방법입니다. 모델의 가중치와 중간 계산 결과(activation)를 32비트 부동소수점(FP32) 대신 16비트(FP16)로 처리하여 VRAM 사용량을 거의 절반으로 줄일 수 있습니다.

PyTorch에서는 torch.cuda.amp (Automatic Mixed Precision)를 통해 매우 쉽게 구현할 수 있습니다. 임베딩 추출(추론) 코드에 torch.autocast 컨텍스트 매니저를 추가하기만 하면 됩니다.

적용 예시 (PyTorch 코드):

Python

import torch

# 모델과 데이터를 GPU로 이동
model.to('cuda:0')
data = data.to('cuda:0')

# 자동 혼합 정밀도 컨텍스트 내에서 모델 실행
with torch.autocast(device_type='cuda', dtype=torch.float16):
    # 이 블록 안의 연산들은 자동으로 FP16을 사용하여 메모리를 절약합니다.
    embeddings = model(data)

이 방법만으로도 output_length=36 또는 48 정도는 단일 GPU에서 실행될 가능성이 있습니다. 먼저 이 방법을 적용해서 VRAM 사용량 변화를 확인해 보세요.

## 2단계: 모델 병렬화(Model Parallelism)로 두 GPU 활용하기
혼합 정밀도를 적용해도 VRAM이 부족하다면, 이제 두 개의 GPU를 모두 사용해야 합니다. 해결책은 하나의 모델을 두 GPU에 나누어 올리는 모델 병렬화입니다.

예를 들어, 모델의 앞부분 레이어들은 첫 번째 GPU(cuda:0)에, 뒷부분 레이어들은 두 번째 GPU(cuda:1)에 할당하는 방식입니다.

### 방법 A: 직접 모델 레이어 분할 (Manual Model Parallelism)
가장 기본적인 방법으로, 모델의 구조를 직접 수정하여 각 부분을 다른 GPU 장치에 할당합니다.

개념적인 구현 예시:
모델이 features와 regressor 두 부분으로 나뉘어 있다고 가정해 보겠습니다.

Python

import torch
import torch.nn as nn

class YourTimeKDModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 앞부분 레이어들은 'cuda:0'에 할당
        self.features = nn.Sequential(
            # ... timeKD 모델의 앞부분 레이어들 ...
        ).to('cuda:0')

        # 뒷부분 레이어들은 'cuda:1'에 할당
        self.regressor = nn.Sequential(
            # ... timeKD 모델의 뒷부분 레이어들 ...
        ).to('cuda:1')

    def forward(self, x):
        # 1. 입력 데이터를 첫 번째 GPU로 보냄
        x = x.to('cuda:0')
        
        # 2. 첫 번째 GPU에서 연산 수행
        x = self.features(x)
        
        # 3. 중간 결과를 두 번째 GPU로 보냄
        x = x.to('cuda:1')
        
        # 4. 두 번째 GPU에서 나머지 연산 수행
        x = self.regressor(x)
        
        return x

# 모델 생성
model = YourTimeKDModel()

# 데이터로 임베딩 추출
# (데이터 로딩 및 전처리는 기존과 동일)
input_data = torch.randn(batch_size, seq_len, features) # 예시 입력 데이터
embeddings = model(input_data)
핵심은 forward 함수 내에서 데이터(중간 결과)를 .to('cuda:1')을 통해 다음 GPU로 넘겨주는 것입니다. 모델의 어느 지점에서 분할할지는 VRAM 사용량을 보며 결정해야 합니다. 대략 절반 정도의 파라미터를 갖는 지점을 기준으로 나누는 것이 좋습니다.

### 방법 B: Hugging Face Accelerate 라이브러리 사용 (강력 추천)
모델 코드를 직접 수정하는 것이 복잡하게 느껴진다면, Accelerate 라이브러리를 사용하는 것이 훨씬 간편하고 강력한 해결책입니다. Accelerate는 단 몇 줄의 코드 추가만으로 복잡한 분산 처리, 모델 병렬화, CPU 오프로딩 등을 자동으로 처리해 줍니다.

device_map="auto" 옵션을 사용하면 라이브러리가 알아서 최적의 방식으로 모델 레이어를 여러 GPU와 CPU RAM에 분산시켜 줍니다.

설치:

Bash

pip install accelerate
적용 예시:

Python

from accelerate import Accelerator, dispatch_model
from accelerate.utils import get_balanced_memory

# 모델 로드 (GPU에 올리지 않은 상태)
model = YourTimeKDModel() 

# Accelerate가 최적의 분할 지점을 찾아 여러 장치에 모델을 분산 할당
# max_memory 옵션으로 각 GPU의 최대 VRAM을 지정해줄 수 있습니다.
max_memory = get_balanced_memory(
    model,
    max_memory_per_gpu="23GB", # 사용 가능한 메모리보다 약간 작게 설정
    no_split_module_classes=["..."] # 분할하면 안 되는 레이어 클래스 지정 (필요시)
)

device_map = {"": "cpu"} # 시작은 CPU로
device_map = dispatch_model(model, device_map=device_map, max_memory=max_memory)


# 이제 모델을 사용하면 Accelerate가 알아서 데이터를 올바른 장치로 보내 처리합니다.
input_data = torch.randn(batch_size, seq_len, features)
# input_data는 CPU에 있어도 됩니다. Accelerate가 첫 레이어가 있는 GPU로 보내줍니다.
embeddings = model(input_data)
Accelerate를 사용하면 복잡한 장치 할당 및 데이터 이동 코드를 직접 작성할 필요가 없어 매우 편리합니다.

## 최종 권장 워크플로우
배치 사이즈 확인: 가장 먼저, 임베딩 추출 시 배치 사이즈를 1로 설정하여 단일 데이터 처리 시의 VRAM 사용량을 최소화합니다.

혼합 정밀도(1단계) 적용: 코드를 간단히 수정하여 torch.autocast를 적용하고, output_length=36, 48에 대해 테스트를 진행합니다.

Accelerate 라이브러리(2단계) 도입: 혼합 정밀도만으로 부족하다면, Accelerate를 사용하여 device_map="auto" 기능으로 모델을 두 GPU에 자동으로 분산시킵니다. 이 방법은 직접 모델을 수정하는 것보다 훨씬 안정적이고 효율적입니다.

(조합) Accelerate를 사용하면서 동시에 torch.autocast 컨텍스트를 함께 사용하면 VRAM을 최대한으로 절약할 수 있어 output_length=96, 192와 같은 매우 큰 경우에도 안정적으로 임베딩을 추출할 수 있습니다.

이 단계들을 순서대로 적용해 보시면 VRAM 초과 문제없이 원하시는 모든 output_length에 대한 임베딩을 성공적으로 추출하실 수 있을 겁니다. 💪

-----
