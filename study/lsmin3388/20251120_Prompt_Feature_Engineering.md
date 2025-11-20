# Feature Engineering 학습 자료

## Prompts

### Question)
```
시계열 데이터에서 Feature Engineering이란 무엇이고, 왜 중요한가요? 대표적인 특성 추출 방법에는 어떤 것들이 있나요?
```

### Answer)
**Feature Engineering 정의**

원시 데이터로부터 모델 학습에 유용한 특성(feature)을 생성하거나 변환하는 과정입니다. 시계열 데이터에서는 시간적 패턴을 포착하는 특성을 만드는 것이 핵심입니다.

**중요성:**
1. 모델 성능 향상의 핵심 요소
2. 도메인 지식 반영 가능
3. 해석 가능한 특성 생성
4. 학습 시간 단축

**대표적인 특성 추출 방법:**

**1. 통계적 특성**
```python
def extract_statistical_features(series):
    return {
        'mean': np.mean(series),
        'std': np.std(series),
        'min': np.min(series),
        'max': np.max(series),
        'median': np.median(series),
        'skewness': scipy.stats.skew(series),
        'kurtosis': scipy.stats.kurtosis(series),
        'q25': np.percentile(series, 25),
        'q75': np.percentile(series, 75)
    }
```

**2. 시간 기반 특성**
```python
def extract_time_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df
```

**3. 이동 통계 (Rolling Statistics)**
```python
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_7'] = df['value'].rolling(window=7).std()
df['rolling_max_7'] = df['value'].rolling(window=7).max()
```

---

### Question)
```
Lag Features와 Rolling Features의 차이점은 무엇이고, 각각 어떻게 생성하나요?
```

### Answer)
**Lag Features (지연 특성)**

과거 시점의 값을 현재 시점의 특성으로 사용합니다.

```python
# 이전 시점 값들을 특성으로
df['lag_1'] = df['value'].shift(1)   # 1시점 전
df['lag_7'] = df['value'].shift(7)   # 7시점 전
df['lag_30'] = df['value'].shift(30) # 30시점 전

# 차분 (변화량)
df['diff_1'] = df['value'].diff(1)
df['diff_7'] = df['value'].diff(7)
```

**특징:**
- 자기회귀(AR) 패턴 포착
- 과거 값과 현재 값의 직접적 관계 모델링
- 계절성 포착 (예: 7일 전 = 같은 요일)

**Rolling Features (이동 특성)**

일정 윈도우 내의 집계 통계를 특성으로 사용합니다.

```python
# 이동 평균
df['rolling_mean_7'] = df['value'].rolling(7).mean()
df['rolling_mean_30'] = df['value'].rolling(30).mean()

# 이동 표준편차
df['rolling_std_7'] = df['value'].rolling(7).std()

# 이동 최댓값/최솟값
df['rolling_max_7'] = df['value'].rolling(7).max()
df['rolling_min_7'] = df['value'].rolling(7).min()

# Exponential Moving Average
df['ema_7'] = df['value'].ewm(span=7).mean()
```

**비교:**

| 특성 | Lag Features | Rolling Features |
|------|-------------|------------------|
| 형태 | 단일 과거 값 | 과거 구간 통계 |
| 정보 | 특정 시점 | 추세/변동성 |
| 노이즈 | 민감 | 평활화됨 |
| 사용 | 자기회귀 | 추세 파악 |

**결합 사용 예시:**
```python
# 현재 값과 rolling mean의 비율
df['ratio_to_rolling_mean'] = df['value'] / df['rolling_mean_7']

# lag와 rolling의 조합
df['lag1_diff_from_mean'] = df['lag_1'] - df['rolling_mean_7']
```

---

### Question)
```
주파수 도메인 특성(FFT, Wavelet 등)은 시계열에서 어떻게 활용되나요?
```

### Answer)
**FFT (Fast Fourier Transform)**

시계열을 주파수 성분으로 분해하여 주기적 패턴을 파악합니다.

```python
from scipy.fft import fft, fftfreq

def extract_fft_features(series, n_components=10):
    fft_values = fft(series)
    fft_magnitude = np.abs(fft_values)

    # 상위 n개 주파수 성분
    top_indices = np.argsort(fft_magnitude)[-n_components:]

    features = {
        'dominant_freq': fftfreq(len(series))[top_indices[-1]],
        'spectral_energy': np.sum(fft_magnitude ** 2),
        'spectral_entropy': scipy.stats.entropy(fft_magnitude)
    }

    for i, idx in enumerate(top_indices):
        features[f'fft_magnitude_{i}'] = fft_magnitude[idx]

    return features
```

**활용:**
- 주기성 탐지 (일간, 주간, 연간 패턴)
- 노이즈 제거
- 주파수 기반 분류

**Wavelet Transform**

시간-주파수 동시 분석이 가능합니다.

```python
import pywt

def extract_wavelet_features(series, wavelet='db4', level=4):
    coeffs = pywt.wavedec(series, wavelet, level=level)

    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_mean_{i}'] = np.mean(coeff)
        features[f'wavelet_std_{i}'] = np.std(coeff)
        features[f'wavelet_energy_{i}'] = np.sum(coeff ** 2)

    return features
```

**FFT vs Wavelet:**

| 특성 | FFT | Wavelet |
|------|-----|---------|
| 분석 도메인 | 주파수 | 시간-주파수 |
| 시간 해상도 | 없음 | 있음 |
| 비정상 신호 | 부적합 | 적합 |
| 계산 복잡도 | O(n log n) | O(n) |

**시계열 분류에서 활용:**
```python
def create_frequency_features(df):
    fft_features = extract_fft_features(df['value'].values)
    wavelet_features = extract_wavelet_features(df['value'].values)

    return {**fft_features, **wavelet_features}
```

**주의사항:**
- 정상성(Stationarity) 가정 고려
- 윈도우 크기 선택 중요
- 도메인 지식 활용하여 의미 있는 주파수 선택
