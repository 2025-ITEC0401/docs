## Prompts (Powered by Gemini 2.5 Pro)
### Question)
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB (GPU 0; 23.69 GiB total capacity; 21.93 GiB already allocated; 111.88 MiB free; 23.27 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF



이 에러는 어떻게 해결하지?

현재 RTX3090 GPU 2장이 장착되어 있고, 물리적인 VRAM 크기를 변경할 수는 없어.
```
### Answer)
이 오류는 GPU 메모리(VRAM)가 부족해서 발생하는 문제입니다. 오류 메시지를 보면, 23.69 GiB 용량의 GPU에서 이미 21.93 GiB가 사용 중인 상태에서 추가로 1.50 GiB를 할당하려다 실패한 상황입니다.

물리적인 VRAM을 늘릴 수는 없으므로, 코드와 실행 환경을 수정하여 메모리 사용량을 줄여야 합니다. RTX 3090 2장을 가지고 계시니 활용할 수 있는 방법이 많습니다.

## 해결 방법 💡
가장 효과적인 방법부터 순서대로 시도해 보세요.

1. 배치 사이즈(Batch Size) 줄이기
가장 먼저 시도해야 할 가장 간단하고 효과적인 방법입니다. 한 번에 처리하는 데이터의 양을 줄여 메모리 사용량을 직접적으로 낮춥니다.

방법: 스크립트를 실행할 때 --batch_size (또는 비슷한 이름의) 인자 값을 현재보다 절반 정도로 줄여서 실행해 보세요.

단점: 배치 사이즈를 너무 줄이면 학습이 불안정해지거나 느려져, 총 학습 시간이 늘어날 수 있습니다.

2. 그래디언트 축적 (Gradient Accumulation)
배치 사이즈를 줄이면서도 큰 배치 사이즈와 유사한 효과를 내는 기법입니다. 여러 작은 배치(mini-batch)의 그래디언트를 메모리에 계속 더한 뒤, 일정 횟수가 되면 한꺼번에 모델 파라미터를 업데이트합니다.

예시: 배치 사이즈를 32로 쓰고 싶지만 메모리가 부족해 8밖에 못 쓴다면, 배치 사이즈 8로 4번 반복한 뒤 파라미터를 업데이트하면 됩니다.

방법: 학습 코드의 optimizer.step() 호출 부분을 수정해야 합니다.

Python

# 일반적인 학습 코드
for i, (inputs, labels) in enumerate(dataloader):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 그래디언트 축적 적용 코드 (accumulation_steps = 4)
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # Loss 정규화
    loss.backward()
    
    # 4번 마다 모델 업데이트
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
3. 혼합 정밀도(Mixed-Precision) 학습 사용
32비트 부동소수점(float32) 연산을 16비트(float16)와 함께 사용하는 기법입니다. 메모리 사용량을 거의 절반으로 줄이고 학습 속도도 높일 수 있습니다. PyTorch에서는 torch.cuda.amp (Automatic Mixed Precision)를 통해 쉽게 사용할 수 있습니다.

방법: 학습 코드에 GradScaler와 autocast를 추가합니다.

Python

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for i, (inputs, labels) in enumerate(dataloader):
    optimizer.zero_grad()
    
    with autocast():  # 이 블록 안의 연산은 float16으로 자동 변환
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
    scaler.scale(loss).backward() # 스케일링된 로스에 대해 그래디언트 계산
    scaler.step(optimizer)        # 옵티마이저 스텝
    scaler.update()               # 스케일러 업데이트
4. 여러 GPU 활용하기 (Data Parallelism)
GPU 2장을 보유하고 있으니, 모델을 두 GPU에 복사하고 데이터를 나누어 동시에 처리하는 데이터 병렬 처리를 사용할 수 있습니다. DistributedDataParallel이 더 효율적이지만, DataParallel이 가장 간단하게 적용할 수 있습니다.

방법: 모델을 nn.DataParallel로 감싸주면 됩니다.

Python

import torch.nn as nn

# ... 모델 정의 후 ...
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)
주의: DataParallel은 데이터를 0번 GPU로 모아서 처리하는 과정 때문에 0번 GPU에 메모리 불균형이 발생할 수 있습니다. 더 효율적인 병렬 처리를 원한다면 DistributedDataParallel(DDP)를 알아보시는 것을 추천합니다.

5. 불필요한 메모리 캐시 비우기
학습 루프에서 더 이상 필요 없는 변수나 텐서가 메모리를 차지하고 있을 수 있습니다. torch.cuda.empty_cache()를 호출하여 PyTorch가 사용하지 않는 캐시 메모리를 정리할 수 있습니다.

방법: 학습 루프의 끝에 추가합니다.

Python

# ... 학습 루프 ...
torch.cuda.empty_cache()
주의: 이 방법은 근본적인 해결책이 아니며, 약간의 성능 저하를 유발할 수 있습니다. 다른 방법들이 효과 없을 때 시도해 보세요.

----
### Question)
```
1~5의 과정을 적용했을 때,
모델의 성능에 영향을 끼칠 수 있나?
```
### Answer)
네, 일부 과정은 모델의 최종 성능에 영향을 미칠 수 있습니다.

각 방법이 성능에 미치는 영향은 다음과 같이 요약할 수 있습니다.

방법	성능 영향	영향의 종류
1. 배치 사이즈 줄이기	있음 (가장 큼)	최종 정확도, 수렴 속도에 직접적인 영향
2. 그래디언트 축적	거의 없음	이론적으로는 배치 사이즈 줄이기의 단점을 보완
3. 혼합 정밀도	거의 없음	드물게 약간의 성능 변화 가능성 존재
4. 데이터 병렬 처리	거의 없음	이론적으로 동일. 구현에 따라 미세한 차이 발생 가능
5. 캐시 비우기	없음	학습 과정이 아닌 메모리 관리에만 관여

Sheets로 내보내기
## 1. 배치 사이즈 줄이기 (영향 큼 큼)
배치 사이즈를 줄이는 것은 모델의 최종 성능에 직접적인 영향을 줄 수 있습니다.

성능 저하 가능성: 배치 사이즈가 너무 작아지면, 각 배치마다 계산되는 그래디언트의 노이즈가 심해집니다. 이로 인해 모델이 최적의 방향을 찾지 못하고 학습이 불안정해지거나, 지역 최적점(local minimum)에 빠져 최종 성능이 낮아질 수 있습니다.

일반화 성능: 반대로, 적절히 작은 배치 사이즈는 오히려 노이즈를 통한 규제(regularization) 효과를 주어 모델의 일반화 성능(새로운 데이터에 대한 성능)을 높이기도 합니다.

결론적으로, 배치 사이즈는 성능에 직접적인 영향을 미치는 중요한 하이퍼파라미터이므로 신중하게 조절해야 합니다.

## 2. 그래디언트 축적 (영향 거의 없음)
이론적으로 그래디언트 축적은 최종 성능에 영향을 주지 않아야 합니다.

이 기법은 단순히 작은 배치들의 그래디언트를 여러 번 더해서 큰 배치의 그래디언트를 흉내 내는 것입니다. 예를 들어, 배치 사이즈 8로 4번 축적하는 것은 배치 사이즈 32로 한번에 학습하는 것과 수학적으로 거의 동일한 효과를 냅니다. 따라서 배치 사이즈를 줄여서 발생하는 성능 저하를 방지하는 좋은 대안이 됩니다.

## 3. 혼합 정밀도 (영향 거의 없음)
대부분의 경우 성능에 영향을 미치지 않지만, 미세한 변화가 있을 수 있습니다.

16비트 부동소수점(float16)은 32비트(float32)보다 표현할 수 있는 숫자의 정밀도가 낮습니다. 이로 인해 매우 작은 그래디언트 값이 0으로 처리되는 등 미세한 정보 손실이 발생할 수 있습니다. 하지만 torch.cuda.amp의 GradScaler와 같은 기능이 이러한 문제를 효과적으로 방지해주기 때문에, 최종 성능에 미치는 영향은 거의 없거나 무시할 수 있는 수준입니다.

## 4. 데이터 병렬 처리 (영향 거의 없음)
이론적으로 성능에 영향을 주지 않습니다.

DataParallel이나 DistributedDataParallel은 전체 데이터를 여러 GPU에 나누어 동시에 처리한 뒤, 그 결과를 합치는 방식입니다. 학습 알고리즘 자체를 바꾸는 것이 아니므로, 단일 GPU에서 더 큰 배치 사이즈로 학습하는 것과 동일한 결과를 내도록 설계되었습니다.

## 5. 캐시 비우기 (영향 없음)
성능에 전혀 영향을 주지 않습니다.

torch.cuda.empty_cache()는 학습 알고리즘이나 모델 파라미터에는 전혀 관여하지 않습니다. 단지 PyTorch가 재사용을 위해 잡아두었던 여분의 메모리를 운영체제에 반환하는 역할만 하므로, 모델의 최종 성능과는 무관합니다.

----
### Question)
```
DistributedDataParallel 를 적용하는 방법을 알려줘.
```

### Answer)
네, DistributedDataParallel (DDP)를 적용하는 방법을 단계별로 알려드릴게요. DDP는 여러 GPU 또는 여러 장비(노드)에서 모델을 효율적으로 훈련시키는 PyTorch의 공식적인 병렬 처리 방식입니다. DataParallel(DP)보다 성능이 좋고 권장되는 방법입니다.

DDP의 핵심 아이디어는 각 GPU마다 별도의 프로세스를 생성하여 모델의 복사본을 독립적으로 실행하고, 각 프로세스는 데이터의 일부(subset)만 처리한 뒤 계산된 그래디언트(gradient)만 서로 동기화하는 것입니다.

## DDP 적용 3단계 요약
DDP를 적용하려면 기존의 일반적인 PyTorch 학습 코드를 크게 3부분 수정해야 합니다.

프로세스 그룹 초기화: 여러 GPU 프로세스가 서로 통신할 수 있도록 환경을 설정합니다.

데이터 분산 샘플링: 전체 데이터셋을 각 GPU 프로세스가 중복 없이 나눠 갖도록 설정합니다.

모델 래핑(Wrapping): 생성된 모델을 DDP 클래스로 감싸줍니다.

## 1단계: 프로세스 그룹 초기화 및 환경 설정
가장 먼저 각 GPU에서 실행될 프로세스들을 설정하고, 이들이 서로 통신할 수 있는 '그룹'을 만들어야 합니다. 이 과정은 보통 학습 스크립트의 초반에 이루어집니다.

os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']: 여러 장비(노드)가 통신할 때 기준이 되는 '대장' 컴퓨터의 주소와 포트. 단일 장비 내에서만 DDP를 사용해도 설정해주는 것이 좋습니다.

dist.init_process_group(): 프로세스 그룹을 초기화합니다.

backend: 통신 방식을 지정합니다. NVIDIA GPU를 사용하면 보통 nccl이 가장 빠릅니다.

rank: 현재 프로세스의 고유 번호 (0, 1, 2, ...). 0번 프로세스가 보통 메인 역할을 합니다.

world_size: 전체 프로세스의 개수 (GPU 개수와 동일).

torch.cuda.set_device(): 현재 프로세스가 어떤 GPU를 사용할지 명시적으로 지정합니다. 이는 매우 중요합니다.

Python

# main.py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    """프로세스 그룹 초기화 및 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 프로세스 그룹 초기화 (backend는 NCCL 사용)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 현재 프로세스가 사용할 GPU를 지정
    torch.cuda.set_device(rank)

def cleanup():
    """프로세스 그룹 정리"""
    dist.destroy_process_group()

# ... (이후 학습 코드에서 이 setup 함수를 호출)
## 2단계: 데이터 분산 샘플링 (DistributedSampler)
각 GPU 프로세스가 데이터셋의 서로 다른 부분을 가져가도록 만들어야 합니다. 그렇지 않으면 모든 GPU가 똑같은 데이터를 학습하여 비효율적입니다. PyTorch의 DistributedSampler가 이 역할을 자동으로 해줍니다.

기존의 DataLoader를 만들 때 sampler 인자에 DistributedSampler를 추가하기만 하면 됩니다. shuffle 옵션은 DistributedSampler가 대신 처리하므로 False로 설정해야 합니다.

Python

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ... 데이터셋(train_dataset) 로드 후 ...

# 1. DistributedSampler 생성
train_sampler = DistributedSampler(train_dataset, shuffle=True)

# 2. DataLoader에 sampler를 전달
#    - shuffle은 반드시 False여야 합니다 (sampler가 대신 처리).
#    - batch_size는 이제 각 GPU당 배치 사이즈가 됩니다.
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,  # 중요!
    sampler=train_sampler
)
## 3단계: 모델 래핑 및 학습
모델을 생성한 뒤, DDP로 감싸주면 됩니다. DDP로 감싸진 모델은 학습 시 backward()가 호출될 때 자동으로 모든 프로세스의 그래디언트를 평균 내어 동기화합니다.

model.to(rank): 모델을 해당 프로세스에 할당된 GPU로 보냅니다.

DDP(model, device_ids=[rank]): 모델을 DDP로 감쌉니다. device_ids를 명시해주는 것이 중요합니다.

Python

# ... setup(rank, world_size) 호출 후 ...

# 1. 모델 생성 후 해당 GPU로 이동
model = MyModel().to(rank)

# 2. 모델을 DDP로 감싸기
ddp_model = DDP(model, device_ids=[rank])

optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(num_epochs):
    # 매 에폭마다 sampler에 현재 에폭 정보를 알려줘야 데이터 셔플링이 제대로 됨
    train_loader.sampler.set_epoch(epoch)

    for data, labels in train_loader:
        # 데이터도 해당 GPU로 이동
        data = data.to(rank)
        labels = labels.to(rank)

        optimizer.zero_grad()
        outputs = ddp_model(data) # DDP 모델 사용
        loss = criterion(outputs, labels)
        loss.backward() # 여기서 그래디언트 동기화가 자동으로 일어남
        optimizer.step()

# 학습 종료 후 프로세스 그룹 정리
cleanup()
## 4단계: 스크립트 실행 (torchrun)
DDP 코드는 일반적인 python main.py 방식으로 실행하지 않습니다. PyTorch가 제공하는 torchrun (또는 구버전의 torch.distributed.launch)을 사용해 여러 프로세스를 자동으로 실행시켜야 합니다.

torchrun은 world_size와 rank 같은 환경 변수를 각 프로세스에 자동으로 설정해주므로 코드가 훨씬 깔끔해집니다.

--nproc_per_node: 장비(노드) 1개당 실행할 프로세스 수 (사용할 GPU 개수).

터미널에 다음과 같이 실행합니다. (GPU 4개를 사용한다고 가정)

Bash

torchrun --nproc_per_node=4 main.py
이렇게 실행하면 torchrun이 main.py를 4번 실행하면서, 각 실행마다 rank를 0, 1, 2, 3으로, world_size를 4로 자동으로 설정해줍니다. 그러면 코드의 setup(rank, world_size) 함수가 이 값들을 받아 프로세스 그룹을 초기화하게 됩니다.

----

### Question)
```
Setting ds_accelerator to cuda (auto detect)
Traceback (most recent call last):
  File "/hdd/intern/aLLM4TS/run_LLM4TS.py", line 146, in <module>
    args.local_rank = int(os.environ["LOCAL_RANK"])
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/os.py", line 680, in __getitem__
    raise KeyError(key) from None
KeyError: 'LOCAL_RANK'
```
### Answer)
이 오류는 분산 훈련(distributed training)을 위해 작성된 코드를 일반적인 파이썬 명령어로 실행했기 때문에 발생합니다.

KeyError: 'LOCAL_RANK'는 LOCAL_RANK라는 환경 변수가 존재하지 않는다는 의미입니다. 이 환경 변수는 여러 GPU를 사용하는 분산 훈련을 시작할 때, 각 프로세스(GPU)에 자신의 고유 번호(0, 1, 2...)를 알려주는 역할을 합니다.

일반적으로 python run_LLM4TS.py와 같이 실행하면 이 변수가 설정되지 않아 오류가 발생합니다.

## 해결 방법: torchrun 사용하기 🚀
이 문제를 해결하려면 PyTorch가 제공하는 분산 훈련 실행 도구인 **torchrun**을 사용해야 합니다. torchrun이 스크립트를 실행하면서 각 프로세스에 LOCAL_RANK 환경 변수를 자동으로 설정해 줍니다.

터미널에서 다음과 같이 명령어를 입력하여 스크립트를 실행하세요.

Bash

# 기본 형식
torchrun --nproc_per_node=<사용할 GPU 개수> <스크립트 경로> --<기존에 사용하던 인자들>

# 예시: RTX 3090 GPU 2장을 모두 사용하는 경우
torchrun --nproc_per_node=2 /hdd/intern/aLLM4TS/run_LLM4TS.py --batch_size 16 --epochs 10
명령어 설명
torchrun: PyTorch의 분산 훈련 실행기입니다.

--nproc_per_node=2: 현재 장비(node)에서 2개의 프로세스(GPU)를 사용하겠다는 의미입니다. 보유하신 RTX 3090 2장을 모두 활용하려면 2로 설정합니다. 만약 1개만 사용하고 싶다면 1로 설정하면 됩니다.

/hdd/intern/aLLM4TS/run_LLM4TS.py: 실행할 파이썬 스크립트입니다.

--<기존에 사용하던 인자들>: 배치 사이즈, 에폭 수 등 원래 스크립트에 전달하던 인자들을 그대로 뒤에 붙여주면 됩니다.

이 명령어로 실행하면, torchrun이 2개의 프로세스를 생성하고 첫 번째 프로세스에는 LOCAL_RANK=0을, 두 번째 프로세스에는 LOCAL_RANK=1을 자동으로 할당하여 코드가 정상적으로 실행되도록 합니다.

-----
### Question)
```
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
Setting ds_accelerator to cuda (auto detect)
Setting ds_accelerator to cuda (auto detect)
Use GPU: cuda:0
Traceback (most recent call last):
  File "/hdd/intern/aLLM4TS/run_LLM4TS.py", line 180, in <module>
    exp = Exp(args)
  File "/hdd/intern/aLLM4TS/exp/exp_LLM4TS.py", line 28, in __init__
    super(Exp_Main, self).__init__(args)
  File "/hdd/intern/aLLM4TS/exp/exp_basic.py", line 12, in __init__
    torch.cuda.set_device(self.device)
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/cuda/__init__.py", line 350, in set_device
    torch._C._cuda_setDevice(device)
RuntimeError: CUDA error: invalid device ordinal
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Use GPU: cuda:0
*********************cuda:0***************************
gpt2 = GPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-5): 6 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 1841029 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 1 (pid: 1841030) of binary: /hdd/conda_envs/envs/aLLM4TS/bin/python
Traceback (most recent call last):
  File "/hdd/conda_envs/envs/aLLM4TS/bin/torchrun", line 7, in <module>
    sys.exit(main())
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
    return f(*args, **kwargs)
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/run.py", line 794, in main
    run(args)
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
run_LLM4TS.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-10-12_17:32:19
  host      : explorer-DL
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 1841030)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```

### Answer)
이 오류는 DDP(DistributedDataParallel)의 초기화 순서가 잘못되었기 때문에 발생합니다.

RuntimeError: CUDA error: invalid device ordinal 오류는 torch.cuda.set_device()가 호출될 때, DDP 환경이 아직 완전히 설정되지 않아 올바른 GPU 번호(device ordinal)를 받지 못했음을 의미합니다.

로그를 보면, local_rank=1인 두 번째 프로세스가 실패했습니다. 이는 DDP의 "프로세스 간 통신 그룹"이 만들어지기 전에 코드가 각 프로세스에 GPU를 할당하려고 시도했기 때문입니다.

## 문제의 핵심 원인 🧐
DDP는 반드시 다음 순서로 설정되어야 합니다.

dist.init_process_group() 호출: 모든 GPU 프로세스들이 서로를 인식하고 통신할 수 있는 '그룹'을 먼저 만듭니다. 이 단계가 완료되어야 각 프로세스는 자신의 고유 번호(rank)를 알게 됩니다.

torch.cuda.set_device() 호출: 그룹이 생성된 후, 각 프로세스는 자신의 local_rank에 해당하는 GPU를 점유합니다.

현재 코드는 이 순서가 반대로 되어, dist.init_process_group()이 호출되기도 전에 exp_basic.py의 __init__에서 torch.cuda.set_device()를 먼저 실행하면서 오류가 발생한 것입니다.

## 해결 방법: 초기화 로직 분리 및 순서 조정 💡
가장 깔끔하고 확실한 해결책은 DDP 관련 초기화 코드를 메인 스크립트(run_LLM4TS.py)의 시작 부분으로 옮겨, 실험(Exp) 객체가 생성되기 전에 모든 DDP 환경 설정이 완료되도록 하는 것입니다.

1단계: exp_basic.py에서 DDP 초기화 코드 제거
먼저, Exp_Basic 클래스의 __init__ 메서드에서 DDP 초기화와 관련된 코드를 제거하거나 주석 처리합니다.

파일: /hdd/intern/aLLM4TS/exp/exp_basic.py

Python

# 수정 전 ❌
# class Exp_Basic:
#     def __init__(self, args):
#         # ...
#         dist.init_process_group(backend='nccl') # 이런 코드가 여기에 있다면 문제
#         self.device = torch.device('cuda:{}'.format(args.local_rank))
#         torch.cuda.set_device(self.device) # 오류 발생 지점
#         # ...

# 수정 후 ✅
class Exp_Basic:
    def __init__(self, args):
        self.args = args
        # DDP 초기화 코드는 여기서 모두 제거하고, device 설정만 남겨둡니다.
        self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        # torch.cuda.set_device() 호출도 메인 스크립트로 옮깁니다.
        # ...
2단계: run_LLM4TS.py에서 DDP 환경 선제적 설정
메인 스크립트에서 Exp 객체를 생성하기 전에 DDP 프로세스 그룹을 초기화하고 기본 장치를 설정합니다.

파일: /hdd/intern/aLLM4TS/run_LLM4TS.py

Python

import torch
import torch.distributed as dist
import os

def main():
    # ... (기존의 ArgumentParser 코드) ...
    args = parser.parse_args()

    # ==================== DDP 설정 추가 시작 ==================== ✅
    # torchrun이 설정해준 환경 변수를 사용하여 local_rank 설정
    args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # 1. DDP 프로세스 그룹 초기화 (가장 먼저 수행!)
    dist.init_process_group(backend="nccl", init_method="env://")

    # 2. 현재 프로세스가 사용할 기본 GPU 장치 설정
    torch.cuda.set_device(args.local_rank)
    # ===================== DDP 설정 추가 끝 =====================

    print(f'Use GPU: cuda:{args.local_rank}') # 확인용 print

    # ... (Fix seed 등 다른 설정) ...

    # 모든 DDP 설정이 끝난 후 Exp 객체를 생성합니다.
    exp = Exp(args) # line 180

    # ... (나머지 코드) ...

if __name__ == '__main__':
    main()
참고: init_method="env://"는 torchrun이 설정한 MASTER_ADDR, MASTER_PORT 등의 환경 변수를 자동으로 읽어와 DDP 그룹을 설정하는 가장 편리한 방법입니다.

## 요약
문제는 "준비가 되기 전에 행동"한 것입니다. 비행기 좌석에 앉으려면 먼저 탑승 수속을 하고 좌석 번호를 받아야 하는 것처럼, DDP에서도 **프로세스 그룹을 먼저 초기화(dist.init_process_group)**하여 각자 자신의 번호(local_rank)를 할당받은 뒤에, 그 번호에 맞는 **GPU 좌석에 착석(torch.cuda.set_device)**해야 합니다.

위와 같이 run_LLM4TS.py의 시작 부분에서 DDP 설정을 먼저 완료하면 문제가 해결됩니다.

---

### Question)
```
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
```
### Answer)
이것은 오류가 아니라, 시스템 성능을 위한 유용한 경고 메시지입니다. 간단히 말해, torchrun이 시스템 과부하를 막기 위해 각 훈련 프로세스가 사용할 CPU 스레드 개수를 안전한 기본값인 1개로 자동 설정했다는 뜻입니다.

## 무슨 의미인가요? 🤔
분산 훈련을 시작하면 GPU 개수만큼 별도의 프로세스가 실행됩니다. PyTorch가 내부적으로 사용하는 데이터 처리 라이브러리(MKL, OpenBLAS 등)는 연산 속도를 높이기 위해 여러 개의 CPU 코어를 동시에 사용하는 경향이 있습니다.

만약 GPU 프로세스 4개를 실행하는데, 각각의 프로세스가 시스템의 CPU 코어 32개를 모두 사용하려고 든다면 어떻게 될까요? 4개의 프로세스가 32개의 코어를 차지하기 위해 서로 경쟁하면서(4 x 32 = 128 스레드 경쟁), CPU 자원 병목 현상이 발생해 오히려 전체 시스템이 극심하게 느려지는 '스레드 폭풍' 현상이 발생할 수 있습니다.

이러한 문제를 방지하기 위해 torchrun은 각 프로세스가 사용할 수 있는 CPU 스레드 개수를 1개(OMP_NUM_THREADS=1)로 제한하는 안전장치를 기본으로 걸어두는 것입니다.

## 이 설정을 바꿔야 할까요?
데이터 로딩 병목 현상이 의심될 때 이 값을 조절하면 성능을 높일 수 있습니다.

nvidia-smi 명령어로 GPU 사용률(GPU-Util)을 확인했을 때, GPU가 열심히 일하지 않고 계속 낮은 상태에 머무른다면, 이는 CPU가 데이터를 충분히 빨리 처리해서 GPU에 전달해주지 못하고 있다는 신호입니다. 즉, GPU가 CPU를 기다리느라 놀고 있는 것입니다. 이럴 때 각 프로세스에 더 많은 CPU 스레드를 할당해주면 데이터 전처리 및 로딩 속도가 빨라져 GPU를 더 효율적으로 활용할 수 있습니다.

## 어떻게 조절하나요? ✅
목표는 균형점을 찾는 것입니다. 전체 CPU 코어 수를 GPU 프로세스 수로 나누는 것이 좋은 출발점입니다.

1. 전체 CPU 코어 수 확인
터미널에서 다음 명령어를 실행합니다.

Bash

nproc
예를 들어 이 명령의 결과가 32라고 가정해 봅시다.

2. 프로세스당 스레드 수 계산
GPU 2개를 사용(--nproc_per_node=2)한다면, 계산은 다음과 같습니다.
전체 코어 32개 / 2개 프로세스 = 프로세스당 16 스레드

3. 환경 변수 설정 후 실행
torchrun을 실행하기 전에 OMP_NUM_THREADS 환경 변수를 설정합니다.

Bash

# 변수를 설정하고 스크립트를 실행
export OMP_NUM_THREADS=16

# 평소처럼 훈련 명령 실행
torchrun --nproc_per_node=2 run_LLM_TS.py ...
혹은 더 간단하게 한 줄로 실행할 수도 있습니다.

Bash

OMP_NUM_THREADS=16 torchrun --nproc_per_node=2 run_LLM_TS.py ...
8, 12, 16 등 여러 값으로 테스트해보면서 가장 좋은 훈련 속도를 내는 최적의 값을 찾을 수 있습니다. 만약 병목 현상이 GPU의 연산 자체에 있다면(GPU-Util이 항상 90% 이상), 이 값을 바꿔도 성능 향상은 거의 없습니다.

----
### Question)
```
왜 이 문제가 발생하지?
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
Setting ds_accelerator to cuda (auto detect)
Setting ds_accelerator to cuda (auto detect)
Traceback (most recent call last):
  File "/hdd/intern/aLLM4TS/run_LLM4TS.py", line 148, in <module>
    torch.cuda.set_device(args.local_rank)
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/cuda/__init__.py", line 350, in set_device
    torch._C._cuda_setDevice(device)
RuntimeError: CUDA error: invalid device ordinal
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Use GPU: cuda:0
*********************cuda:0***************************
gpt2 = GPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-5): 6 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 1841907 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 1 (pid: 1841908) of binary: /hdd/conda_envs/envs/aLLM4TS/bin/python
Traceback (most recent call last):
  File "/hdd/conda_envs/envs/aLLM4TS/bin/torchrun", line 7, in <module>
    sys.exit(main())
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
    return f(*args, **kwargs)
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/run.py", line 794, in main
    run(args)
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/hdd/conda_envs/envs/aLLM4TS/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
run_LLM4TS.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-10-12_17:36:51
  host      : explorer-DL
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 1841908)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================ 
```
### Answer)
이 오류는 두 번째 GPU 프로세스(local_rank=1)가 자신에게 할당된 GPU(cuda:1)를 찾지 못해서 발생합니다.

오류 메시지 RuntimeError: CUDA error: invalid device ordinal은 "존재하지 않는 번호의 GPU를 사용하려고 했습니다"라는 의미입니다.

쉽게 비유하자면, local_rank=1 프로세스는 **"나는 1번 좌석표를 받았으니 1번 GPU 좌석에 앉겠다"**고 했는데, 정작 그 프로세스가 탑승한 비행기에는 0번 좌석밖에 없는 상황인 것입니다.

## 문제의 핵심 원인 🧐
이 문제의 가장 유력한 원인은 CUDA_VISIBLE_DEVICES 환경 변수가 잘못 설정되었기 때문입니다.

CUDA_VISIBLE_DEVICES는 특정 프로세스가 어떤 GPU를 "볼 수 있는지"를 제한하는 역할을 합니다. torchrun으로 2개의 GPU를 사용해 스크립트를 실행하면, 내부적으로 다음과 같이 동작합니다.

프로세스 0 (local_rank=0): CUDA_VISIBLE_DEVICES=0 환경에서 실행됩니다. 이 프로세스에게 GPU는 0번 하나만 보입니다.

프로세스 1 (local_rank=1): CUDA_VISIBLE_DEVICES=1 환경에서 실행됩니다. 이 프로세스에게 GPU는 1번 하나만 보입니다. 하지만 이 프로세스 내부에서 그 보이는 GPU는 다시 0번으로 인덱싱됩니다.

그런데 코드의 다른 부분이나 실행 스크립트(all_s16.sh)에서 CUDA_VISIBLE_DEVICES를 0으로 강제로 고정해버리면, 두 프로세스 모두에게 0번 GPU 하나만 보이게 됩니다.

이 상황에서 local_rank=1인 두 번째 프로세스는 torch.cuda.set_device(1)을 실행하게 되고, 자신에게는 0번 GPU 하나밖에 보이지 않으므로 "1번 GPU는 존재하지 않습니다 (invalid device ordinal)"라는 오류를 내뱉는 것입니다.

## 해결 방법 💡
torchrun이 GPU 할당을 자동으로 처리하도록, 수동으로 설정된 CUDA_VISIBLE_DEVICES를 제거해야 합니다.

1. 셸 스크립트 파일 확인 (가장 먼저)
코드를 실행하는 데 사용하는 all_s16.sh 같은 셸 스크립트 파일 안에 CUDA_VISIBLE_DEVICES를 설정하는 부분이 있는지 확인하고 제거하세요.

터미널에서 스크립트 내용을 확인합니다.

Bash

cat all_s16.sh
아래와 같은 줄이 있다면 주석 처리하거나 삭제하세요.

Bash

# 수정 전 ❌
export CUDA_VISIBLE_DEVICES=0,1 # 또는 export CUDA_VISIBLE_DEVICES=0 등

torchrun --nproc_per_node=2 run_LLM4TS.py ...

# 수정 후 ✅
# 위 export 라인을 완전히 제거하거나 #으로 주석 처리

torchrun --nproc_per_node=2 run_LLM4TS.py ...
2. 파이썬 코드 확인
파이썬 스크립트(run_LLM4TS.py 등) 내부에서 os.environ을 이용해 이 변수를 설정하는 부분이 있는지 확인하고 제거하세요.

Python

# 수정 전 ❌
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 수정 후 ✅
# 위 두 줄을 완전히 제거하거나 주석 처리
3. 터미널 환경 변수 확인
터미널에서 직접 export CUDA_VISIBLE_DEVICES=... 명령을 실행한 적이 있다면, 새 터미널을 열어서 아무것도 설정되지 않은 깨끗한 상태에서 다시 실행해 보세요.

## 요약
문제의 핵심은 torchrun의 자동 GPU 할당 기능과 수동 CUDA_VISIBLE_DEVICES 설정이 충돌한 것입니다. torchrun을 사용할 때는 GPU를 직접 제어하려는 코드를 모두 제거하고 torchrun에게 모든 것을 맡기는 것이 가장 좋습니다. 스크립트 파일에서 CUDA_VISIBLE_DEVICES 관련 설정을 제거하면 문제가 해결될 것입니다.

----
### Question)
```
내 말은, 모델 학습에 30GB 의 VRAM이 필요한 상황일 때, RTX3090 한 장으로는 VRAM 초과되지만, 두 장으로는 15GB 씩 나눠서 작업되냐는 말이야.
```

### Answer)
아니요, 아쉽게도 그렇게 동작하지 않습니다. 그것이 바로 분산 학습에서 가장 흔히 혼동하는 지점입니다.

결론부터 말하면, 30GB VRAM이 필요한 모델은 RTX 3090 (24GB) 두 장으로도 일반적인 분산 학습 방식(DistributedDataParallel)으로는 실행할 수 없습니다.

## 왜 안 되나요? 모델 복제 vs. 모델 분할
사용자께서 생각하신 방식은 **'모델 분할'**에 가깝습니다. 하지만 DistributedDataParallel(DDP) 같은 표준적인 데이터 병렬 방식은 '모델 복제' 방식을 사용합니다.

데이터 병렬 처리 (DDP - 우리가 사용 중인 방식)

동작 방식: 30GB짜리 모델 전체를 복사해서 0번 GPU에도 올리고, 1번 GPU에도 올립니다.

결과: 각 GPU는 30GB의 VRAM을 확보해야 합니다. 하지만 RTX 3090은 24GB이므로, 모델을 VRAM에 올리는 첫 단계부터 '메모리 부족(Out of Memory)' 오류가 발생합니다.

목표: 훈련 속도를 높이는 것.

비유: 똑같은 설계도(모델)를 가진 공장(GPU) 두 개를 지어서, 원자재(데이터)를 절반씩 나눠 동시에 생산하는 것과 같습니다. 공장 하나에 설계도 전체가 들어갈 수 있어야 합니다.

모델 병렬 처리 (Model Parallelism - 사용자께서 생각하신 방식)

동작 방식: 30GB짜리 모델을 쪼개서 0번 GPU에 15GB, 1번 GPU에 15GB를 나눠서 올립니다.

결과: 24GB VRAM을 가진 GPU 두 장으로 30GB 모델을 실행할 수 있습니다.

목표: 단일 GPU의 VRAM 용량을 초과하는 거대한 모델을 훈련시키는 것.

비유: 하나의 거대한 컨베이어 벨트(모델)를 두 명의 작업자(GPU)가 나눠서 담당하는 것과 같습니다. 작업자 A가 앞부분 공정을 처리하고, 그 결과물을 작업자 B에게 넘겨 뒷부분 공정을 처리합니다.

## 요약
구분	데이터 병렬 처리 (Data Parallelism)	모델 병렬 처리 (Model Parallelism)
무엇을 나누나요?	데이터	모델
각 GPU의 상태	모델 전체를 복제해서 가짐	모델의 일부분을 나눠서 가짐
주된 목적	학습 속도 향상	단일 GPU 메모리를 초과하는 모델 학습
구현 난이도	상대적으로 쉬움 (DDP 사용)	훨씬 복잡함 (수동 코드 또는 전문 라이브러리 필요)

Sheets로 내보내기
따라서 지금 사용하고 계신 torchrun과 DistributedDataParallel 설정은 데이터 병렬 처리 방식이므로, 30GB 모델을 24GB VRAM GPU에서 실행할 수 없습니다. 이 문제를 해결하려면 모델 병렬 처리라는 더 복잡한 기술을 도입해야 합니다.  