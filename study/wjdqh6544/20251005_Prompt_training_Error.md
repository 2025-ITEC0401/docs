## Prompts (Powered by Gemini 2.5 Pro)
### Question)
```
timeKD 논문 코드를 받아서 실행할거야.

File "train.py", line 377, in <module>

    main()

  File "train.py", line 181, in main

    engine = trainer(

  File "train.py", line 101, in __init__

    print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))

  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1729, in __getattr__

    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

AttributeError: 'Dual' object has no attribute 'count_trainable_params'


이런 에러가 뜨는데, 뭐지?
```
### Answer)
해당 에러는 Dual이라는 모델 객체에 count_trainable_params라는 함수(메서드)가 없기 때문에 발생합니다.

쉽게 말해, 훈련 코드(train.py)는 모델의 학습 가능한 파라미터 개수를 세기 위해 model.count_trainable_params()를 호출했는데, 정작 Dual이라는 모델 클래스 안에는 그런 기능을 하는 함수가 정의되어 있지 않은 상황입니다. 이 함수는 PyTorch의 기본 기능이 아니라 연구/개발자가 따로 만들어줘야 하는 커스텀 함수일 가능성이 높습니다.

원인
AttributeError: 'Dual' object has no attribute 'count_trainable_params' 메시지는 다음을 의미합니다.

'Dual' object: 현재 사용 중인 모델 객체의 클래스 이름이 Dual입니다.

has no attribute 'count_trainable_params': 이 Dual 객체 내부에 count_trainable_params라는 이름의 속성이나 메서드가 존재하지 않습니다.

훈련 코드 train.py의 101번째 줄에서 self.model.count_trainable_params()를 호출하고 있지만, self.model(즉, Dual 모델)에는 해당 함수가 없습니다.

해결 방법
가장 간단한 해결책은 학습 가능한 파라미터 수를 직접 계산하는 코드로 해당 부분을 대체하는 것입니다.

1. 코드 수정
train.py 파일의 101번째 줄을 찾으세요.

기존 코드 (101번째 줄):

Python

print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
수정 코드:
이 줄을 아래 코드로 바꿔주세요. PyTorch에서 학습 가능한 파라미터 수를 세는 표준적인 방법입니다.

Python

# 기존 라인을 주석 처리하고 아래 코드를 추가하거나, 그냥 덮어쓰세요.
# print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
print(f"The number of trainable parameters: {num_params}")
2. 코드 설명
self.model.parameters(): 모델의 모든 파라미터(가중치, 편향 등)를 가져옵니다.

if p.requires_grad: 파라미터 중에서 학습이 가능한 것만 필터링합니다. (freeze된 레이어는 제외)

p.numel(): 각 파라미터 텐서의 총 요소 수를 계산합니다. (예: (64, 3, 3, 3) 모양의 텐서 -> 64 * 3 * 3 * 3 = 1728)

sum(...): 모든 학습 가능한 파라미터의 요소 수를 합산하여 총개수를 구합니다.

이 방법으로 코드를 수정하면 count_trainable_params 함수 없이도 파라미터 수를 정상적으로 출력하고 다음 단계로 넘어갈 수 있습니다. 👍

