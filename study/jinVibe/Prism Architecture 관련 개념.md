DTW(동적 시간 워핑, Dynamic Time Warping)은 길이가 다르거나 시계열 데이터의 속도가 일정하지 않을 때, 두 시계열 데이터 간의 유사성을 측정하는 알고리즘

Q. "범용(Universal)" 프레임워크가 되려면 해결해야 할 가장 큰 기술적 난제는 무엇이며, 어떻게 접근해야 하는가?

A: 가장 큰 난제는 **도메인 이동(Domain Shift)**과 다양한 주기(Frequency) 처리입니다. (예: 주식 데이터 vs 심전도 데이터)

접근법:

입력 정규화(Normalization): RevIN(Reversible Instance Normalization) 같은 기법을 필수적으로 사용하여 데이터의 통계적 분포 차이를 제거해야 합니다.

주파수 도메인 활용: 시간 도메인뿐만 아니라 Fourier Transform 등을 통해 주파수 도메인 특징을 함께 학습하면 주기성이 다른 데이터에 더 강건해질 수 있습니다.

가변 패치: 고정된 패치 길이 대신 다양한 크기의 패치를 동시에 학습(Multi-scale patching)하여 다양한 패턴을 포착해야 합니다.