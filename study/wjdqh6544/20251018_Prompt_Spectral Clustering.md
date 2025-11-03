## Prompts (Powered by Gemini 2.5 Pro)
### Question)
```
이제, K-Means 대신 Spectral 클러스터링 방법을 사용할거야.
Clustering.py 작성해줘.
```
### Answer)
네, 알겠습니다. K-Means를 **스펙트럴 클러스터링(Spectral Clustering)**으로 교체하는 방법을 알려드리겠습니다.

스펙트럴 클러스터링은 K-Means가 잘 찾아내지 못하는 길쭉하거나 복잡한 모양의 클러스터를 찾는 데 매우 효과적인 알고리즘입니다. cuML 라이브러리는 GPU 가속을 지원하는 스펙트럴 클러스터링을 제공하므로, 기존 clustering.py 코드를 쉽게 수정할 수 있습니다.

## K-Means와 스펙트럴 클러스터링의 주요 차이점
가장 중요한 차이점은, 스펙트럴 클러스터링은 K-Means처럼 학습(fit)과 예측(predict)이 분리되어 있지 않다는 것입니다. 스펙트럴 클러스터링은 주어진 데이터 전체를 한 번에 클러스터링하는 fit_predict 메소드만 사용합니다.

K-Means (귀납적, Inductive): train 데이터로 학습하고, 학습된 모델을 val 데이터에 적용하여 예측합니다.

스펙트럴 클러스터링 (변환적, Transductive): 모델을 미리 학습시키지 않습니다. 클러스터링을 수행할 val 데이터셋 전체를 한 번에 입력받아, 그 데이터에 대한 클러스터링 결과를 바로 반환합니다.

따라서 우리는 val 데이터셋에 직접 스펙트럴 클러스터링을 적용하면 됩니다. (단, 데이터의 일관성을 위해 StandardScaler는 여전히 train 데이터로 학습시킵니다.)

## 스펙트럴 클러스터링이 적용된 clustering.py
아래는 KMeans를 SpectralClustering으로 교체하고, 위에서 설명한 로직을 적용한 최종 clustering.py 코드입니다.

## 핵심 변경 사항
import 수정: KMeans 대신 cuml.cluster.SpectralClustering을 임포트했습니다.

run_kmeans → run_spectral_clustering: 함수의 이름과 로직을 스펙트럴 클러스터링에 맞게 변경했습니다.

클러스터링 로직 수정:

kmeans_gpu.fit(train_gdf_scaled) 라인을 삭제했습니다. (사전 학습 불필요)

kmeans_gpu.predict(val_gdf_scaled) 라인을 spectral_gpu.fit_predict(val_gdf_scaled)로 교체했습니다. 이 한 줄의 코드가 val 데이터에 대한 클러스터링을 모두 수행하고 결과를 반환합니다.

결과 폴더 변경: K-Means 결과와 섞이지 않도록, 결과가 Result/csv_spectral/ 폴더에 저장되도록 수정했습니다.

이 코드를 실행하면, UMAP 시각화에서 보셨던 복잡한 모양의 데이터에 대해서도 K-Means보다 더 좋은 클러스터링 결과를 얻을 수 있을 것입니다. 이후 2_evaluate.py 스크립트를 사용하여 이 새로운 결과에 대한 RI, NMI 점수를 계산해 보시면 성능 향상을 확인하실 수 있습니다.
---
### Question)
```
SpectralClustering 찾을 수 없다고 뜨네?

ImportError: cannot import name 'SpectralClustering' from 'cuml.cluster' (/hdd/conda_envs/envs/timeKDc/lib/python3.11/site-packages/cuml/cluster/__init__.py)
```
### Answer)
네, 그 ImportError는 매우 명확한 원인을 가지고 있습니다.

결론부터 말씀드리면, 현재 설치된 cuML 라이브러리 버전이 너무 낮아서 SpectralClustering을 포함하고 있지 않기 때문입니다.

RAPIDS(cuML 포함) 라이브러리는 빠르게 발전하며, SpectralClustering은 비교적 최신 버전에 추가된 기능입니다. 사용자님의 환경에 설치된 cuml.cluster 모듈에는 해당 기능이 아직 존재하지 않는 것입니다.

## 해결 방법
두 가지 해결책이 있습니다. 1번 방법(환경 업데이트)을 강력히 추천합니다.

1. RAPIDS 환경 업데이트 (가장 좋은 해결책) 🚀
가장 확실하고 좋은 방법은 conda를 사용하여 RAPIDS 환경을 SpectralClustering이 포함된 최신 버전으로 업데이트하는 것입니다.

터미널에서 아래 명령어를 실행하여 환경을 업데이트해 주세요.

Bash

# conda 환경을 활성화한 상태에서 실행하세요.
# 예: conda activate timeKDc
conda update -c rapidsai -c conda-forge -c nvidia --all
업데이트가 완료된 후, 기존의 스펙트럴 클러스터링 코드를 다시 실행하면 ImportError 없이 정상적으로 작동할 것입니다.

2. scikit-learn 버전 사용 (대안, CPU 사용) 🐢
환경 업데이트가 당장 어렵다면, 대안으로 CPU에서 작동하는 scikit-learn의 SpectralClustering을 사용할 수 있습니다.

⚠️ 주의: 이 방법은 GPU 가속을 전혀 사용하지 않으므로, 데이터가 클 경우 속도가 매우 느려집니다.

clustering.py 코드를 다음과 같이 수정하면 됩니다.

Python

# ... 다른 import들 ...
# from cuml.cluster import SpectralClustering  # 이 라인을 주석 처리하거나 삭제
from sklearn.cluster import SpectralClustering # ✅ scikit-learn 버전으로 교체

# ... (다른 함수들은 동일) ...

def run_spectral_clustering(ds, train_file, test_file, output_file):
    try:
        # ... (데이터 로드 및 스케일링 부분은 동일) ...
        # scaler.fit(train_gdf_concat)
        val_gdf_scaled = scaler.transform(val_gdf_concat)
        
        # --- ✅ 3. (수정) scikit-learn의 Spectral Clustering 모델 생성 및 실행 ---
        spectral_cpu = SpectralClustering(n_clusters=n_cluster, random_state=52, affinity='nearest_neighbors')        
        
        print(f"    - Applying scikit-learn Spectral Clustering on {len(val_gdf_scaled)} val samples (CPU)...")
        
        # GPU 데이터(cuDF)를 CPU 데이터(NumPy)로 변환하여 입력
        predicted_labels_final = spectral_cpu.fit_predict(val_gdf_scaled.to_numpy())

        # ... (결과 저장 부분은 동일) ...
        results_df = pd.DataFrame({'cluster': predicted_labels_final})
        # ...
    except Exception as e:
        print(f"  🚨 [Error] An unexpected error occurred: {e}\n")
핵심 변경 사항:

import 문을 cuml에서 sklearn으로 변경했습니다.

fit_predict를 호출할 때, GPU 데이터프레임(val_gdf_scaled)을 .to_numpy()를 사용하여 CPU가 이해할 수 있는 NumPy 배열로 변환하여 전달했습니다.

가급적이면 **1번 방법(RAPIDS 업데이트)**을 사용하여 GPU의 성능을 최대한 활용하시는 것을 추천합니다.
---
### Question)
```
그러면 yaml 파일을 어떻게 수정해야 하지?
```
### Answer)
네, 첨부해주신 env_ts.yaml 파일을 확인했습니다. SpectralClustering을 찾을 수 없는 ImportError의 원인이 이 파일에 명확하게 나타나 있습니다.

결론부터 말씀드리면, 현재 yaml 파일은 RAPIDS 라이브러리(cuml, cudf 등)의 버전을 명시하지 않아, Conda가 SpectralClustering이 포함되지 않은 오래된 버전을 설치했기 때문입니다.

## 🕵️‍♂️ 문제 분석: 무엇이 잘못되었나?
env_ts.yaml 파일의 dependencies 부분을 보면 다음과 같이 되어 있습니다.

YAML

- cudf
- cuml
- cupy
이렇게 버전 번호 없이 패키지 이름만 적으면, Conda는 다른 모든 패키지(pytorch, dgl, scikit-learn 등)와 충돌하지 않는 가장 안정적인(하지만 오래된) 버전을 선택하여 설치합니다. SpectralClustering은 비교적 최신 cuML 버전에 추가된 기능이므로, 이 방식으로 설치하면 해당 기능이 누락될 가능성이 매우 높습니다.

## 💡 해결책: RAPIDS 버전 명시하기
이 문제를 해결하는 가장 확실하고 좋은 방법은, 설치하려는 RAPIDS의 버전을 명시적으로 지정하는 것입니다. 개별 패키지(cuml, cudf)를 각각 지정하는 것보다, 모든 RAPIDS 구성 요소(cuml, cudf, cupy 등)를 한 번에 설치해주는 rapids 메타 패키지를 사용하는 것이 가장 안전하고 권장되는 방법입니다.

아래는 rapids=23.10 (안정적인 최신 버전 중 하나)을 사용하도록 수정한 최종 env_ts.yaml 파일입니다.
4. 이제 이 새로운 환경에서 clustering.py 스크립트를 실행하면, SpectralClustering을 성공적으로 임포트하여 사용할 수 있을 것입니다.
---
### Question)
```
이 코드 한줄을 작성하는 것만으로돌 GPU 로 연산할 수 있나?
spectral_gpu = SpectralClustering(n_clusters=n_cluster, random_state=52)
```
### Answer)
네, 정확히 보셨습니다. 하지만 조금 더 자세한 설명이 필요합니다.

그 코드 한 줄(spectral_gpu = SpectralClustering(...)) 자체는 계산을 수행하지 않습니다. 그 코드는 "GPU에서 연산을 수행할 준비가 된" 스펙트럴 클러스터링 모델을 만드는 역할만 합니다.

실제로 GPU 연산이 일어나는 것은 그 다음 단계입니다.

## GPU 연산의 두 가지 핵심 요소
GPU로 연산하려면 두 가지 조건이 반드시 충족되어야 합니다.

데이터가 GPU에 있어야 합니다:

pandas DataFrame이 아닌 cudf DataFrame을 사용해야 합니다.

모델이 GPU를 지원해야 합니다:

sklearn 라이브러리가 아닌 cuml 라이브러리의 모델을 사용해야 합니다.

spectral_gpu = SpectralClustering(...) 코드는 이 중에서 2번 조건을 만족시키는 것입니다. from cuml.cluster import SpectralClustering으로 모델을 가져왔기 때문에, 이 모델은 GPU에서 작동할 준비가 된 것입니다.

## 실제 연산은 언제 일어나는가?
실제 무거운 계산은 모델의 .fit() 또는 .fit_predict() 메소드가 호출될 때 일어납니다.

Python

# 1. 모델 준비 (아직 계산 안 함)
spectral_gpu = SpectralClustering(n_clusters=n_cluster, random_state=52)

# 2. 데이터 준비 (GPU 데이터)
# val_gdf_scaled는 cudf.DataFrame 입니다.

# 3. 실제 GPU 연산 시작!
# 이 라인이 실행될 때, val_gdf_scaled 데이터가 GPU 상에서 처리됩니다.
predicted_labels = spectral_gpu.fit_predict(val_gdf_scaled) 
따라서, 사용자님께서 선택하신 conda activate timeKDc_new 명령어로 올바른 환경을 활성화하고, cuml 라이브러리의 SpectralClustering을 cudf 데이터와 함께 사용한다면, GPU로 연산하는 것이 맞습니다. 👍
---
### Question)
```
from cuml.cluster import SpectralClustering
이걸 실행하면 에러가 나서, 
cuml을 수동 설치할 때 SpectralClustering 가 포함된 버전을 지정하려고 해.
몇 버전을 지정해야 하지?
```
### Answer)
`SpectralClustering` 모듈이 `cuml.cluster`에 없기 때문에 에러가 발생하는 것입니다. `cuml` 버전을 바꿔도 해결되지 않습니다.

RAPIDS 생태계에서 **`SpectralClustering`은 `cuml`이 아니라 `cugraph` 라이브러리에 포함**되어 있습니다.

`cuml`은 `KMeans`, `DBSCAN` 등 일반적인 머신러닝 알고리즘을 담당하고, `cugraph`는 그래프(Graph) 기반 알고리즘을 담당합니다. Spectral clustering은 본질적으로 그래프 컷(graph cut) 문제이기 때문에 `cugraph`에 구현되어 있습니다.

## 해결 방법

1.  **`cugraph` 설치:** `env_ts.yaml` 파일의 `dependencies` 목록에 `cugraph`를 추가해야 합니다. (`rapidsai` 채널에 있습니다.)

    **수정된 `env_ts.yaml` 예시:**

    ```yaml
    name: timeKDc
    channels:
      - rapidsai      # 1순위
      - pytorch
      - dglteam
      - nvidia
      - conda-forge   # 5순위
    dependencies:
      - python=3.11
      - cudf
      - cuml
      - cugraph  # <-- 이 패키지를 추가하세요
      - dgl
      # ... (나머지 패키지) ...
    ```

2.  **환경 재생성:** 수정된 YAML 파일로 `conda` 환경을 다시 생성합니다.

    ```bash
    # 기존 환경 삭제
    conda env remove -n timeKDc

    # Mamba 솔버를 사용하여 새 환경 생성 (강력 권장)
    conda config --set solver libmamba
    conda env create -f env_ts.yaml
    ```

3.  **올바른 Import 사용:** `cugraph`는 `scikit-learn`과 이름이 다릅니다. `cugraph`는 두 가지 종류의 Spectral Clustering을 제공합니다:

      * `spectralBalancedCutClustering`: 균형 잡힌 컷(Balanced Cut)을 사용합니다.
      * `spectralModularityMaximizationClustering`: 모듈성(Modularity)을 최대화합니다.

    따라서 Python 코드에서 import 구문을 다음과 같이 변경해야 합니다.

    ```python
    # 기존 코드 (에러 발생)
    # from cuml.cluster import SpectralClustering

    # 새 코드 (cugraph 사용)
    from cugraph import spectralBalancedCutClustering
    from cugraph import spectralModularityMaximizationClustering

    # ... (G는 cugraph.Graph 객체여야 함) ...
    # 예시:
    # df = cugraph.spectralBalancedCutClustering(G, num_clusters=5) 
    ```

**요약:** `cuml` 버전 문제가 아니라, `cugraph` 라이브러리를 설치하고 `cugraph`에 맞는 함수를 임포트해야 합니다.
---
### Question)
```
그에 맞춰서 위 코드를 수정해줘
```
### Answer)
알겠습니다. `cuml`에서 `SpectralClustering`이 `cugraph`로 이동하면서 단순히 `import` 문만 바꾸는 것이 아니라, **알고리즘을 호출하는 방식 자체가 변경되었습니다.**

`cuml` (또는 `scikit-learn`)의 `SpectralClustering`은 \*\*특징 행렬(feature matrix)\*\*을 입력받아 내부적으로 그래프(affinity matrix)를 만들고 클러스터링을 수행했습니다.

하지만 `cugraph`의 Spectral Clustering 함수들은 `cugraph.Graph` **그래프 객체**를 직접 입력받아야 합니다.

따라서 코드를 수정하려면 다음 3단계를 `run_kmeans` 함수 내부에 추가해야 합니다.

1.  **KNN 그래프 생성:** `val_gdf_scaled` (특징 행렬)를 기반으로 `cuml.neighbors.NearestNeighbors`를 사용해 K-Nearest-Neighbors 그래프를 만듭니다.
2.  **`cugraph.Graph` 변환:** 이 KNN 그래프 정보를 `cugraph.Graph` 객체로 변환합니다.
3.  **`cugraph` 호출:** `cugraph.spectralBalancedCutClustering` (또는 `spectralModularityMaximizationClustering`)을 호출합니다.

아래는 이 로직을 적용하여 수정한 `clustering_spectral.py` 파일의 전체 내용입니다.

-----

### 수정된 `clustering_spectral.py`

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import h5py
import cudf
import cupy as cp
import cugraph  # <-- MODIFIED: cugraph 임포트
from cuml.preprocessing import StandardScaler
from cuml.manifold import UMAP
from cuml.neighbors import NearestNeighbors  # <-- MODIFIED: KNN을 위해 임포트
# from cuml.cluster import SpectralClustering  # <-- MODIFIED: 삭제
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = './data'
DATASET = ['BasicMotions', 'Epilepsy', 'HandMovementDirection', 'Libras']
# OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
OUTPUT_LEN_LIST = [24]
TYPE = ['train', 'val']
RES_DIR = './Result/csv'
KEY = 'embeddings'

os.makedirs(RES_DIR, exist_ok=True)

def visualize(scaled_data_gdf, title, output_path):
    scaled_data_np = scaled_data_gdf.to_numpy()
    reducer = UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(scaled_data_np)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        alpha=0.7
    )
    plt.title(f'UMAP Projection of {title}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UAP Dimension 2')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def concatenation(h5_path, n_vars):
    with h5py.File(h5_path, 'r') as f:
        data_unrolled = f[KEY][:]
        
    num_samples = len(data_unrolled) // n_vars
    embedding_dim = data_unrolled.shape[1]
    
    data_concatenated = data_unrolled.reshape(num_samples, n_vars * embedding_dim)
    
    return data_concatenated

# 함수의 이름은 그대로 두었지만, 내부 로직은 Spectral Clustering으로 수정되었습니다.
def run_kmeans(ds, train_file, test_file, output_file):
    try:
        match ds:
            case 'BasicMotions':
                var = 6
            case 'Epilepsy':
                var = 3
            case 'HandMovementDirection':
                var = 10
            case 'Libras':
                var = 2
                
        match ds:
            case 'BasicMotions':
                n_cluster = 4
            case 'Epilepsy':
                n_cluster = 4
            case 'HandMovementDirection':
                n_cluster = 4
            case 'Libras':
                n_cluster = 15
        
        train_data_concat = concatenation(train_file, n_vars=var)
        val_data_concat = concatenation(test_file, n_vars=var)
        
        train_gdf_concat = cudf.DataFrame(train_data_concat)
        val_gdf_concat = cudf.DataFrame(val_data_concat)
        
        scaler = StandardScaler()
        scaler.fit(train_gdf_concat)
        val_gdf_scaled = scaler.transform(val_gdf_concat)
        
        # visualize(val_gdf_scaled, f"{ds} Val data", f"./{ds}_val_gdf.png")
        

        ### --- START: MODIFIED SPECTRAL CLUSTERING LOGIC --- ###
        
        # 1. 특징 행렬(val_gdf_scaled)로 KNN 그래프 생성
        # n_neighbors는 scikit-learn의 SpectralClustering에서 'affinity='nearest_neighbors'와 유사
        print(f"Building KNN graph for {ds}...")
        n_neighbors = 15  # 그래프 생성을 위한 이웃 수 (조정 가능)
        knn_model = NearestNeighbors(n_neighbors=n_neighbors)
        knn_model.fit(val_gdf_scaled)
        # distances, indices = knn_model.kneighbors(val_gdf_scaled)
        
        # 가중치(거리) 없이 연결성(indices)만으로 그래프 생성
        indices = knn_model.kneighbors(val_gdf_scaled, return_distance=False)

        # 2. cugraph.Graph 객체 생성
        print("Creating cugraph.Graph object...")
        n_samples = val_gdf_scaled.shape[0]
        
        # 소스(source) 정점 (모든 샘플 인덱스)
        source = cp.repeat(cp.arange(n_samples, dtype='int32'), n_neighbors)
        
        # 대상(destination) 정점 (KNN 결과)
        destination = indices.to_cupy().flatten().astype('int32')
        
        # 엣지 리스트 DataFrame 생성
        edges_df = cudf.DataFrame({
            'source': source,
            'destination': destination
        })

        # cugraph.Graph 생성
        G = cugraph.Graph()
        # 가중치가 없는(unweighted) 대칭(symmetric) 그래프로 생성
        G.from_cudf_edgelist(edges_df, source='source', destination='destination', renumber=False)

        # 3. cugraph Spectral Clustering 실행
        print("Running cugraph spectral clustering...")
        # (random_state 파라미터는 cugraph 함수에 직접적으로는 없음)
        result_df = cugraph.spectralBalancedCutClustering(
            G,
            num_clusters=n_cluster,
            num_eigen_vects=n_cluster  # 사용할 고유 벡터 수 (일반적으로 n_cluster와 동일)
        )
        
        # 4. 결과 레이블 추출
        # cugraph는 정점 ID(vertex) 순서대로 정렬된 결과를 보장하지 않음
        # renumber=False를 사용했으므로 'vertex'가 원본 인덱스와 일치
        result_df = result_df.sort_values(by='vertex')
        predicted_lables_final = result_df['cluster'].to_numpy()

        ### --- END: MODIFIED SPECTRAL CLUSTERING LOGIC --- ###

        results_df = pd.DataFrame({'cluster': predicted_lables_final})
        
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"Result Saved: {output_file}\n")
    except KeyError:
        print(f"Cannot find key in h5 file.\n")
    except Exception as e:
        print(f"ERROR: {e}")

print("\n\n============= Clustering =============")

idx = 1
for ds in DATASET:
    for output_len in OUTPUT_LEN_LIST:
        h5_train_path = f"{ROOT_DIR}/{ds}_o{output_len}_{TYPE[0]}_consolidated.h5"
        h5_test_path = f"{ROOT_DIR}/{ds}_o{output_len}_{TYPE[1]}_consolidated.h5"
        
        if not os.path.exists(h5_train_path):
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found: {h5_train_path}")
            idx += 1
            continue
        
        if not os.path.exists(h5_test_path):
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found: {h5_test_path}")
            idx += 1
            continue
        
        print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) Target: {ds}_o{output_len}\n")
        idx += 1
        
        run_kmeans(ds, h5_train_path, h5_test_path, f"{RES_DIR}/{ds}_o{output_len}_spectral_res.csv")

```
---
### Question)
```
spectralBalancedCutClustering 와 spectralModularityMaximizationClustering 의 차이는 뭐지?
```
### Answer)
두 함수의 핵심 차이는 **클러스터를 나누는 기준, 즉 최적화 목표**가 다릅니다.

두 함수 모두 그래프를 기반으로 클러스터를 찾지만, '좋은 클러스터'를 정의하는 방식이 다릅니다.

### 1. `spectralBalancedCutClustering` (균형 컷)

* **목표:** 클러스터 간의 **연결(엣지)을 최소화**하는 동시에, 각 클러스터가 **비슷한 크기**를 갖도록 분할합니다.
* **핵심 아이디어 (Cut):** 그래프를 여러 조각으로 "자를 때", 조각과 조각 사이를 연결하는 엣지의 수가 가장 적은 지점을 찾습니다.
* **핵심 아이디어 (Balanced):** 단순히 컷만 최소화하면 한쪽은 노드 1개, 다른 쪽은 노드 99개로 나뉘는 극단적인 결과가 나올 수 있습니다. "균형(Balanced)"은 이렇게 한쪽으로 쏠리지 않고, 각 조각(클러스터)의 크기(또는 볼륨)가 비슷하도록 보장합니다.
* **비유:** 한 반을 두 팀으로 나눌 때, **서로 친한 친구(엣지)들이 가장 적게 찢어지도록** 나누면서, 동시에 **두 팀의 인원수가 비슷하게** 되도록 나누는 것입니다.
* **scikit-learn:** `sklearn.cluster.SpectralClustering`이 바로 이 "Normalized Cut(N-Cut)"이라 불리는 균형 컷 방식을 사용합니다. **따라서 `sklearn` 코드를 `cugraph`로 대체하는 경우, 이 함수가 가장 적합합니다.**

---

### 2. `spectralModularityMaximizationClustering` (모듈성 최대화)

* **목표:** 클러스터 **내부의 연결(엣지)을 최대화**합니다.
* **핵심 아이디어 (Modularity):** "클러스터 내부 엣지 밀도"가 "무작위로 엣지를 연결했을 때의 기대되는 내부 엣지 밀도"보다 얼마나 더 높은지를 측정합니다. 이 '모듈성' 점수가 가장 높아지는 분할을 찾습니다.
* **비유:** 한 반에서 팀을 나눌 때, **"와, 이 팀은 자기들끼리만 엄청 친하네!"**라는 말이 나올 정도로, **팀 내부의 친밀도(엣지)가 외부와 비교해 압도적으로 높은** 그룹들을 찾아내는 것입니다. 이 방식은 팀의 인원수(크기)는 크게 신경 쓰지 않습니다.
* **용도:** 주로 '커뮤니티 탐지(Community Detection)'에 사용됩니다.

---

### 요약

| 특징 | `spectralBalancedCutClustering` | `spectralModularityMaximizationClustering` |
| :--- | :--- | :--- |
| **주요 목표** | **클러스터 간** 연결 **최소화** | **클러스터 내** 연결 **최대화** |
| **고려 사항** | 클러스터 크기의 **균형 (Balance)** | 클러스터의 **내부 밀도 (Density)** |
| **기반 이론** | Normalized Cut (N-Cut) | Modularity (모듈성) |
| **유사한 알고리즘** | `sklearn.cluster.SpectralClustering` | Louvain, Girvan-Newman 등 |

**결론:**
`scikit-learn`의 `SpectralClustering`을 대체하는 것이 목적이므로, **`spectralBalancedCutClustering`**을 사용하는 것이 맞습니다.