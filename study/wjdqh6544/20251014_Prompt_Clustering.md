## Prompts (Powered by Gemini 2.5 Pro)
### Question)
```
Forecasting 목적으로 추출한 임베딩을 Clustering 작업에 사용할거야.

임베딩 벡터는 768차원 정도인 상황에서, 작업 과정을 알려줘.

1. 차원 축소 -> 클러스터링 알고리즘 적용 -> 시각화

2. 클러스터링 알고리즘 적용 -> 차원 축소 -> 시각화
```
### Answer)
네, 768차원의 고차원 임베딩 벡터를 클러스터링하는 두 가지 접근 방식에 대해 설명해 드리겠습니다. 각 방법은 장단점이 있으며, 데이터의 특성과 분석 목표에 따라 선택이 달라질 수 있습니다.

1. 차원 축소 → 클러스터링 → 시각화
이 방법은 가장 일반적이고 권장되는 접근 방식입니다. 고차원의 데이터를 저차원으로 압축하여 노이즈를 줄이고, 계산 효율성을 높인 후 클러스터링을 수행합니다.

작업 과정
차원 축소 (Dimensionality Reduction): 768차원의 임베딩 벡터를 PCA, t-SNE, UMAP과 같은 알고리즘을 사용하여 2차원 또는 3차원의 저차원 벡터로 변환합니다.

PCA (Principal Component Analysis): 데이터의 분산을 최대한 보존하는 축을 찾아 차원을 축소합니다. 계산 속도가 빠르고 데이터의 전체적인 구조를 파악하는 데 유용합니다.

t-SNE (t-Distributed Stochastic Neighbor Embedding): 고차원에서 가까웠던 데이터 포인트들이 저차원에서도 가깝게 유지되도록 하는 데 중점을 둡니다. 시각화에 매우 효과적이지만, 계산 비용이 높고 대규모 데이터셋에는 부적합할 수 있습니다.

UMAP (Uniform Manifold Approximation and Projection): t-SNE와 유사하게 데이터의 지역적 구조를 보존하면서도, 전역적인 구조(global structure)까지 더 잘 유지하고 계산 속도도 빠르다는 장점이 있습니다.

클러스터링 알고리즘 적용 (Clustering): 차원 축소된 데이터에 K-Means, DBSCAN, 계층적 클러스터링 등의 알고리즘을 적용하여 군집을 형성합니다.

K-Means: 군집의 개수(K)를 미리 지정해야 하며, 계산이 빠르고 간단합니다. 군집이 원형의 형태를 띨 때 좋은 성능을 보입니다.

DBSCAN: 밀도 기반 클러스터링으로, 복잡한 형태의 군집을 잘 찾아내고 노이즈를 구분할 수 있습니다. 군집의 개수를 미리 정할 필요가 없습니다.

시각화 (Visualization): 2D 또는 3D로 차원 축소된 결과를 산점도(scatter plot)로 시각화하고, 각 데이터 포인트를 클러스터링 결과에 따라 다른 색상으로 표시하여 군집의 분포와 형태를 직관적으로 확인합니다.

2. 클러스터링 → 차원 축소 → 시각화
이 방법은 원본 고차원 공간에서 직접 클러스터링을 수행한 후, 그 결과를 확인하기 위해 시각화 목적으로만 차원 축소를 사용합니다.

작업 과정
클러스터링 알고리즘 적용 (Clustering): 768차원의 원본 임베딩 벡터에 직접 클러스터링 알고리즘을 적용합니다. 이때 '코사인 유사도'와 같이 고차원 텍스트 데이터에 적합한 거리 측정 방식을 사용하는 것이 중요할 수 있습니다.

고차원 데이터에서는 유클리드 거리(Euclidean distance)가 잘 동작하지 않는 경우가 많아(차원의 저주), 거리 기반 알고리즘(예: K-Means)의 성능이 저하될 수 있습니다.

차원 축소 (Dimensionality Reduction): 클러스터링이 완료된 후, 각 데이터 포인트에 할당된 군집 레이블(cluster label) 정보를 유지한 채로 시각화를 위해 차원 축소를 진행합니다.

시각화 (Visualization): 차원 축소된 데이터 포인트를 시각화하되, 1번 단계에서 결정된 군집별로 색상을 다르게 표시합니다. 이를 통해 고차원 공간에서 형성된 군집이 저차원 공간에서 어떻게 표현되는지 확인할 수 있습니다.

두 방법의 비교 및 추천
특징	1. 차원 축소 → 클러스터링 (권장)	2. 클러스터링 → 차원 축소
장점	차원의 저주(curse of dimensionality) 회피, 계산 효율성 증대, 노이즈 감소	원본 데이터의 정보를 손실 없이 클러스터링에 모두 활용 가능
단점	차원 축소 과정에서 정보 손실이 발생하여 클러스터링 성능에 영향을 줄 수 있음	차원의 저주로 인해 거리 기반 클러스터링 알고리즘의 성능이 저하될 수 있음
주요 사용 목적	효율적이고 안정적인 클러스터링 및 시각화	원본 공간에서의 군집 구조를 최대한 확인하고 싶을 때
추천 알고리즘 조합	UMAP/PCA + K-Means/DBSCAN	K-Means (코사인 유사도 기반) → t-SNE/UMAP (시각화용)

Sheets로 내보내기
결론: 어떤 방법을 선택해야 할까?
대부분의 경우 **첫 번째 방법(차원 축소 → 클러스터링)**이 더 나은 결과를 제공합니다. 768차원과 같은 고차원 공간에서는 '차원의 저주'로 인해 데이터 포인트 간의 거리 계산이 의미를 잃기 쉽고, 이로 인해 클러스터링 알고리즘의 성능이 크게 저하될 수 있습니다.

따라서 UMAP이나 PCA를 이용해 데이터의 중요한 구조는 유지하면서 차원을 먼저 줄인 후, 저차원 공간에서 클러스터링을 수행하는 것이 노이즈의 영향을 줄이고 계산적으로도 효율적이며 더 안정적인 군집을 찾는 데 도움이 됩니다.

만약 차원 축소로 인한 정보 손실이 우려된다면, 두 번째 방법을 시도해볼 수는 있지만, 클러스터링 알고리즘과 거리 측정 방식을 신중하게 선택해야 합니다.

----
### Question)
```
그러면, 차원 축소를 먼저 진행하고, 클러스터링 알고리즘을 적용해 볼게.

차원 축소는 어떻게 하지? t-SNE 를 사용해야 할 것 같아.
```
### Answer)
네, 좋은 선택입니다! 차원 축소를 먼저 진행하면 '차원의 저주'를 피하고, 시각적으로 데이터를 탐색하며 클러스터링의 방향을 잡는 데 큰 도움이 됩니다.

t-SNE를 사용한 차원 축소 과정에 대해 주요 파라미터와 실제 코드 예시를 중심으로 자세히 알려드릴게요.

t-SNE (t-Distributed Stochastic Neighbor Embedding) 란?
t-SNE는 고차원 공간에서 서로 가까운 데이터 포인트들이 저차원(주로 2D 또는 3D)에서도 가깝게 유지되도록 하는 시각화 전문 차원 축소 기법입니다. 데이터의 복잡한 구조와 군집을 시각적으로 파악하는 데 매우 효과적입니다.

t-SNE 사용법: 핵심 파라미터 튜닝하기
t-SNE는 몇 가지 중요한 파라미터를 어떻게 설정하느냐에 따라 결과가 크게 달라지므로, 각 파라미터의 의미를 이해하는 것이 중요합니다.

n_components: 축소할 차원의 수를 의미합니다. 시각화가 목적이므로 보통 2 (2D) 또는 3 (3D)으로 설정합니다.

추천: n_components=2

perplexity: t-SNE에서 가장 중요한 파라미터입니다. 각 데이터 포인트가 주변의 몇 개의 이웃을 고려할 것인지를 결정하며, 데이터의 지역적/전역적 구조를 어떻게 포착할지 조절합니다.

낮은 값 (e.g., 5): 데이터의 매우 지역적인(local) 구조에 집중합니다.

높은 값 (e.g., 50): 데이터의 좀 더 전역적인(global) 구조를 고려합니다.

추천: 보통 5에서 50 사이의 값을 사용하며, 데이터셋의 크기에 따라 조절합니다. 30을 시작점으로 여러 값을 시도해보는 것이 좋습니다.

learning_rate: 최적화 과정에서 각 데이터 포인트의 위치를 얼마나 크게 조정할지를 결정하는 학습률입니다.

추천: scikit-learn 최신 버전에서는 **'auto'**가 기본값이며, 자동으로 최적의 값을 찾아주므로 그대로 사용하는 것을 권장합니다. (이전 버전에서는 10~1000 사이, 보통 200이 기본값)

n_iter: 최적화를 위한 반복 횟수입니다. 포인트들이 안정적인 위치에 수렴할 때까지 충분한 횟수를 지정해야 합니다.

추천: 최소 1000 이상을 권장합니다. (n_iter < 250이면 안 됩니다)

init: 초기 포인트 배치를 어떻게 할지 결정합니다. PCA 결과를 사용하면 더 안정적이고 빠르게 수렴할 수 있습니다.

추천: 'pca' 를 사용하면 초기 노이즈를 줄이고 재현성 있는 결과를 얻는 데 도움이 됩니다.

💡 매우 중요한 팁: PCA 먼저 적용하기
768차원과 같은 고차원 데이터에 t-SNE를 직접 적용하면 노이즈가 많고 계산 시간이 매우 오래 걸릴 수 있습니다. 일반적인 모범 사례는 PCA를 사용해 먼저 50~100차원 정도로 1차 축소를 한 뒤, 그 결과에 t-SNE를 적용하는 것입니다.

장점:

노이즈 감소: PCA가 데이터의 주요 분산 방향(중요한 정보)만 남겨 노이즈를 줄여줍니다.

계산 시간 단축: t-SNE의 계산 복잡도를 크게 줄여줍니다.

Python 코드 예시 (scikit-learn 사용)
아래는 768차원 임베딩 벡터(embedding_vectors)가 있다고 가정하고 t-SNE를 적용하는 전체 과정입니다.

Python

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 샘플 데이터 생성 (실제 768차원 임베딩 벡터로 대체)
# 예시: 500개의 데이터 포인트, 각 768차원
embedding_vectors = np.random.rand(500, 768)

# 2. (강력 추천) PCA를 이용한 1차 차원 축소
print("Running PCA...")
# 노이즈 감소 및 속도 향상을 위해 50차원으로 축소
pca = PCA(n_components=50, random_state=42)
pca_result = pca.fit_transform(embedding_vectors)
print("PCA complete. Shape after PCA:", pca_result.shape)


# 3. t-SNE 모델 생성 및 학습
print("Running t-SNE...")
tsne = TSNE(
    n_components=2,      # 2D로 축소
    perplexity=30,       # 일반적인 추천 값
    learning_rate='auto',# 자동으로 학습률 조절
    n_iter=1000,         # 충분한 반복 횟수
    init='pca',          # PCA 기반 초기화
    random_state=42      # 결과를 재현하기 위한 시드
)

# PCA 결과에 t-SNE 적용
tsne_result = tsne.fit_transform(pca_result)
print("t-SNE complete. Shape after t-SNE:", tsne_result.shape)


# 4. 결과 시각화
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    palette="viridis", # 나중에 클러스터링 결과(label)를 여기에 넣으면 색상으로 구분됨
    legend="full"
)

plt.title('t-SNE visualization of 768D embeddings')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.grid(True)
plt.show()

이제 이 코드를 실행하면 얻게 되는 2차원 산점도(tsne_result)를 바탕으로 다음 단계인 클러스터링 알고리즘(예: K-Means, DBSCAN)을 적용하면 됩니다. 시각적으로 군집이 잘 형성되었다면, 클러스터링도 성공적으로 수행될 가능성이 높습니다.

----
### Question)
```
차원 축소 혹은 클러스터링 알고리즘에서, GPU를 사용하는 부분이 있나?
```
### Answer)
네, 당연히 있습니다. 특히 768차원과 같은 고차원 데이터를 다룰 때 GPU를 사용하면 계산 시간을 수십 배에서 수백 배까지 단축시킬 수 있어 매우 유용합니다.

핵심은 NVIDIA의 RAPIDS 생태계, 그중에서도 cuML (CUDA Machine Learning Library) 라이브러리를 사용하는 것입니다. cuML은 scikit-learn과 API가 거의 동일하여 기존 코드에서 몇 줄만 수정하면 바로 GPU 가속을 활용할 수 있습니다.

1. GPU 가속이 가능한 알고리즘
거의 모든 주요 차원 축소 및 클러스터링 알고리즘이 cuML을 통해 GPU 가속을 지원합니다.

💻 차원 축소 (Dimensionality Reduction)
t-SNE: CPU로 실행하면 매우 느린 대표적인 알고리즘입니다. cuML의 t-SNE는 엄청난 속도 향상을 보여주므로, 대규모 데이터셋에서는 거의 필수적입니다.

UMAP: t-SNE보다 빠르지만 역시 계산량이 많습니다. cuML을 사용하면 UMAP 또한 극적으로 빨라집니다.

PCA: cuML의 PCA는 대규모 행렬 연산을 GPU로 처리하여 빠르게 수행합니다.

🚀 클러스터링 (Clustering)
DBSCAN: 밀도 기반 클러스터링으로, 데이터가 많아지면 CPU에서 상당한 시간이 걸립니다. cuML의 DBSCAN은 GPU의 병렬 처리 능력을 활용해 매우 빠르게 동작합니다.

K-Means: 각 데이터 포인트와 중심점 간의 거리를 반복적으로 계산하는 과정이 GPU 병렬 처리에 매우 적합하여 속도 향상 효과가 큽니다.

계층적 클러스터링 (Hierarchical Clustering): 이 역시 cuML에서 지원하여 대규모 데이터에서도 빠르게 군집을 형성할 수 있습니다.

2. 코드 적용 방법: scikit-learn vs cuML
API가 얼마나 유사한지 코드로 직접 비교해 보면 쉽게 이해할 수 있습니다.

t-SNE 예시
기존 scikit-learn 코드 (CPU 사용)

Python

from sklearn.manifold import TSNE

# CPU를 사용하여 t-SNE 실행
tsne_cpu = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result_cpu = tsne_cpu.fit_transform(embedding_vectors)
cuML 코드 (GPU 사용)

Python

# 라이브러리 이름만 cuml로 변경
from cuml.manifold import TSNE 

# GPU를 사용하여 t-SNE 실행
# API와 파라미터가 거의 동일함
tsne_gpu = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result_gpu = tsne_gpu.fit_transform(embedding_vectors)
DBSCAN 예시
기존 scikit-learn 코드 (CPU 사용)

Python

from sklearn.cluster import DBSCAN

# CPU를 사용하여 DBSCAN 실행
dbscan_cpu = DBSCAN(eps=0.5, min_samples=5)
clusters_cpu = dbscan_cpu.fit_predict(tsne_result) 
cuML 코드 (GPU 사용)

Python

# 라이브러리 이름만 cuml로 변경
from cuml.cluster import DBSCAN

# GPU를 사용하여 DBSCAN 실행
dbscan_gpu = DBSCAN(eps=0.5, min_samples=5)
clusters_gpu = dbscan_gpu.fit_predict(tsne_result)
보시다시피 import 구문만 바꾸면 기존의 워크플로우를 그대로 유지하면서 GPU의 강력한 성능을 활용할 수 있습니다.

3. 사용을 위한 준비 사항 ✅
GPU 가속을 사용하려면 몇 가지 사전 준비가 필요합니다.

NVIDIA GPU: 당연히 CUDA 코어를 탑재한 NVIDIA 그래픽 카드가 필요합니다.

CUDA Toolkit: NVIDIA 드라이버와 CUDA Toolkit이 설치되어 있어야 합니다.

RAPIDS 설치: cuML은 RAPIDS의 일부입니다. pip보다는 conda를 사용하여 설치하는 것이 종속성 문제를 해결하기에 훨씬 수월합니다.

Conda를 사용한 RAPIDS 설치 명령어 예시:

Bash

conda create -n rapids-23.10 -c rapidsai -c conda-forge -c nvidia \
    rapids=23.10 python=3.9 cudatoolkit=11.8
(버전은 RAPIDS 공식 문서를 참조하여 시스템에 맞게 조정해야 합니다.)

결론
768차원의 임베딩 벡터와 같이 크고 복잡한 데이터를 다룰 때, GPU 가속은 선택이 아닌 필수에 가깝습니다. 특히 반복적인 실험과 파라미터 튜닝이 필요한 분석 과정에서 cuML을 사용하면 작업 시간을 획기적으로 줄여 분석의 효율성을 극대화할 수 있습니다.

---
### Question)
```
그러면, GPU를 활용하여 차원 축소 및 클러스터링 작업을 수행하는 코드를 알려줘.
```
### Answer)
네, GPU 가속 라이브러리인 cuML을 사용하여 차원 축소와 클러스터링을 수행하는 전체 과정을 코드로 보여드리겠습니다.

이 코드는 **PCA (1차 축소) → t-SNE (2차 시각화용 축소) → DBSCAN (클러스터링)**의 권장 파이프라인을 따릅니다. scikit-learn 대신 cuml을 사용하며, 데이터는 GPU 메모리 위에서 효율적으로 처리됩니다.

사전 준비
먼저, RAPIDS 환경이 구축되어 있어야 합니다. conda를 사용하여 RAPIDS를 설치하는 것이 가장 안정적입니다.

GPU 활용 차원 축소 및 클러스터링 전체 코드
Python

import numpy as np
import cupy as cp  # GPU에서 NumPy처럼 사용되는 라이브러리
import cudf      # GPU에서 Pandas처럼 사용되는 라이브러리
from cuml.manifold import TSNE
from cuml.decomposition import PCA
from cuml.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 샘플 데이터 생성 ---
# 실제 768차원 임베딩 벡터로 이 부분을 대체하세요.
# 예시: 5000개의 데이터 포인트, 각 768차원
print("Generating sample data on CPU...")
cpu_embedding_vectors = np.random.rand(5000, 768).astype(np.float32)

# NumPy 배열(CPU)을 CuPy 배열(GPU)로 전송
print("Moving data from CPU to GPU...")
gpu_embedding_vectors = cp.asarray(cpu_embedding_vectors)
print(f"Data is now on GPU. Shape: {gpu_embedding_vectors.shape}")

# --- 2. 1차 차원 축소: PCA (GPU 사용) ---
# 768차원 -> 50차원
# 고차원에서 t-SNE를 바로 적용하기 전, 노이즈를 줄이고 속도를 높이기 위함
print("\nRunning PCA on GPU...")
pca_gpu = PCA(n_components=50)
pca_result_gpu = pca_gpu.fit_transform(gpu_embedding_vectors)
print(f"PCA complete. Shape after PCA: {pca_result_gpu.shape}")

# --- 3. 2차 차원 축소: t-SNE (GPU 사용) ---
# 50차원 -> 2차원 (시각화를 위해)
print("\nRunning t-SNE on GPU...")
tsne_gpu = TSNE(
    n_components=2,
    perplexity=30,
    # cuML의 t-SNE는 learning_rate_method='barnes-hut'가 기본이며 효율적임
    method='barnes-hut', 
    random_state=42
)
tsne_result_gpu = tsne_gpu.fit_transform(pca_result_gpu)
print(f"t-SNE complete. Shape after t-SNE: {tsne_result_gpu.shape}")

# --- 4. 클러스터링: DBSCAN (GPU 사용) ---
# 2D t-SNE 결과에 클러스터링 적용
print("\nRunning DBSCAN on GPU...")
# eps와 min_samples는 데이터 분포를 보고 조정해야 하는 하이퍼파라미터
dbscan_gpu = DBSCAN(eps=1.0, min_samples=5) 
# DBSCAN은 2D 시각화 결과에 적용하여 직관적인 군집을 찾거나,
# PCA 결과(50차원)에 직접 적용하여 더 풍부한 정보로 군집을 찾을 수도 있음
clusters_gpu = dbscan_gpu.fit_predict(tsne_result_gpu)
print("DBSCAN complete.")

# --- 5. 결과 시각화 ---
# 시각화를 위해 GPU 데이터(CuPy)를 CPU(NumPy)로 다시 가져옴
print("\nMoving results back to CPU for plotting...")
tsne_result_cpu = cp.asnumpy(tsne_result_gpu)
clusters_cpu = cp.asnumpy(clusters_gpu)

# 고유한 클러스터 개수 확인 (노이즈는 -1로 표시됨)
n_clusters = len(set(clusters_cpu)) - (1 if -1 in clusters_cpu else 0)
print(f"Found {n_clusters} clusters.")

# cuDF DataFrame을 사용하여 데이터 결합
df_plot = cudf.DataFrame()
df_plot['tsne-2d-one'] = tsne_result_gpu[:, 0]
df_plot['tsne-2d-two'] = tsne_result_gpu[:, 1]
df_plot['cluster_label'] = clusters_gpu

# Pandas로 변환하여 Seaborn에서 시각화
df_plot_pd = df_plot.to_pandas()

plt.figure(figsize=(14, 10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="cluster_label",
    palette=sns.color_palette("hsv", len(set(clusters_cpu))),
    data=df_plot_pd,
    legend="full",
    alpha=0.7
)

plt.title('GPU Accelerated t-SNE & DBSCAN Clustering')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.show()

코드 주요 포인트 짚어보기 💡
데이터 이동: 분석 시작 시점에 cp.asarray()를 통해 데이터를 CPU에서 GPU로 한 번만 옮기고, 모든 계산(PCA, t-SNE, DBSCAN)은 GPU 내에서 처리됩니다. 시각화를 위해 마지막에 cp.asnumpy()로 결과를 다시 CPU로 가져옵니다.

라이브러리 임포트: sklearn 대신 cuml을, numpy 대신 cupy를 사용합니다. API가 거의 같아 코드 전환이 매우 쉽습니다.

하이퍼파라미터: DBSCAN의 **eps**와 min_samples 값은 데이터의 밀도에 따라 달라지므로, 시각화된 결과를 보고 적절히 조정해야 합니다. 초기 t-SNE 산점도에서 점들이 얼마나 떨어져 있는지 관찰하여 eps의 초기값을 정할 수 있습니다.

클러스터링 대상: 위 코드에서는 시각화된 2D t-SNE 결과에 DBSCAN을 적용했습니다. 이는 사람이 결과를 직관적으로 이해하기 좋은 방식입니다. 만약 더 많은 원본 데이터의 정보를 활용하고 싶다면, pca_result_gpu (50차원)에 직접 DBSCAN을 적용하는 것도 좋은 전략이 될 수 있습니다.

----
### Question)
```
사용할 GPU를 선택할 수는 없나?

그리고, 시각화를 위한 수치 데이터를 따로 저장해야 해.
```
### Answer)
네, 물론입니다. 특정 GPU를 지정해서 사용하는 것과 시각화 데이터를 파일로 저장하는 것 모두 간단하게 처리할 수 있습니다.

1. 사용할 GPU 선택하기 (CUDA_VISIBLE_DEVICES)
여러 개의 GPU가 장착된 시스템에서 특정 GPU만 사용하도록 지정할 수 있습니다. 가장 일반적인 방법은 파이썬 스크립트 초기에 os 모듈을 사용하여 CUDA_VISIBLE_DEVICES 환경 변수를 설정하는 것입니다.

이 코드는 cupy나 cuml 같은 GPU 라이브러리를 import 하기 전에 반드시 실행되어야 합니다.

os.environ["CUDA_VISIBLE_DEVICES"] = "0": 첫 번째 GPU를 사용합니다.

os.environ["CUDA_VISIBLE_DEVICES"] = "1": 두 번째 GPU를 사용합니다.

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2": 첫 번째와 세 번째 GPU를 함께 사용합니다.

코드 적용 예시
Python

import os

# 📢 중요: cuml, cupy, torch 등 CUDA 관련 라이브러리 임포트 전에 실행!
# 시스템의 2번 GPU (세 번째 GPU)를 사용하도록 지정합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

import numpy as np
import cupy as cp
from cuml.manifold import TSNE
# ... (이하 코드는 동일)
2. 시각화 데이터 파일로 저장하기
시각화에 사용된 2차원 좌표와 각 점의 클러스터 레이블 정보를 파일로 저장하는 것은 매우 간단합니다. 이전 코드에서 생성한 Pandas DataFrame (df_plot_pd)을 CSV 파일로 저장하는 것이 가장 일반적이고 편리한 방법입니다.

코드 적용 예시
아래 코드는 이전 코드의 시각화 단계에 추가하면 됩니다.

Python

# ... (이전 DBSCAN 및 시각화 준비 코드) ...

# Pandas DataFrame으로 변환하여 시각화 준비
df_plot_pd = df_plot.to_pandas()

# --- ✨ 데이터 저장 단계 추가 ---
# 시각화에 사용된 t-SNE 좌표와 클러스터링 결과를 CSV 파일로 저장합니다.
output_filename = 'visualization_data.csv'
df_plot_pd.to_csv(output_filename, index=False) # index=False는 불필요한 행 번호를 저장하지 않음

print(f"\n시각화 데이터가 '{output_filename}' 파일로 저장되었습니다.")


# --- 시각화 단계 (기존과 동일) ---
plt.figure(figsize=(14, 10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="cluster_label",
    palette=sns.color_palette("hsv", len(set(clusters_cpu))),
    data=df_plot_pd,
    legend="full",
    alpha=0.7
)
plt.title('GPU Accelerated t-SNE & DBSCAN Clustering')
plt.show()
이렇게 저장된 visualization_data.csv 파일은 다음과 같은 형태로 구성됩니다. 나중에 이 파일만 불러와서 다시 시각화하거나 다른 분석에 활용할 수 있습니다.

tsne-2d-one	tsne-2d-two	cluster_label
15.432	-20.198	0
-5.876	33.451	1
16.001	-21.543	0
...	...	...
