# PWI Analysis

반도체 공정 데이터(Metro + EDS)를 결합하여 **PWI(Process Window Index)** 를 산출하고, 최적 공정 윈도우(Window High / Low)를 결정하는 Python 분석 패키지입니다.

---

## 분석 컨셉

```
[Metro 데이터]  +  [EDS 데이터]
       │                │
       └────── 병합 ────┘
                 │
           이상치 제거
                 │
        등분산 그룹화 (N 그룹)
                 │
       정규성 판단 (Skewness)
          ┌──────┴──────┐
        정규           비정규
       ANOVA       Kruskal-Wallis
          └──────┬──────┘
             p ≥ 0.05?
              │       │
            종료    사후 검정
                  (Tamhane T2 / Dunn's)
                      │
                Good Group 식별
                      │
              좌/우 2차 회귀 분석
                      │
             Window High / Low 결정
                      │
                 PWI Index 산출
```

### 핵심 개념

| 용어 | 설명 |
|------|------|
| **Metro 데이터** | 공정 측정값 (`item_value`). 분석 축(X축) 역할 |
| **EDS 데이터** | 수율 지표 (`bin_value`). 분석 반응(Y축) 역할 |
| **Good Group** | 사후 검정에서 기준 그룹(최소 bin_value)과 유의미한 차이가 없는 그룹 |
| **Window** | Good Group 중심 기준 좌/우 2차 회귀의 `y_target` 교점 |
| **PWI Index** | 전체 웨이퍼 중 Window 범위 안에 들어오는 비율 (%) |

---

## 프로젝트 구조

```
pwi_analysis/           ← 분석 패키지
├── config.py           ← AnalysisConfig: 모든 파라미터 중앙 관리
├── preprocess.py       ← 데이터 병합 + Metro/EDS 이상치 제거
├── grouping.py         ← qcut 기반 등분산 그룹화 + 유효 그룹 필터
├── hypothesis.py       ← ANOVA / Kruskal-Wallis 분기 검정
├── posthoc.py          ← Tamhane T2 / Dunn's 사후 검정, Good Group 식별
├── windowing.py        ← 좌/우 2차 회귀 → Window 경계, PWI 계산
├── pipeline.py         ← 전체 파이프라인 오케스트레이터
└── parallel.py         ← m_key2 × bin_id 병렬 실행

tests/                  ← 단계별 단위 테스트 (36개)
├── conftest.py
├── test_preprocess.py
├── test_grouping.py
├── test_hypothesis.py
├── test_posthoc.py
├── test_windowing.py
└── test_pipeline.py

main.py                 ← 실행 데모 (단일 / 병렬 모드)
requirements.txt
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 빠른 시작

### 데모 실행 (합성 데이터)

```bash
# 단일 분석 (m_key2 1개 × bin_id 1개)
python main.py

# 병렬 분석 (m_key2 3개 × bin_id 2개 = 6 tasks)
python main.py --parallel

# 옵션 조합
python main.py --parallel --n 8 --samples 1000 --keys 5 --bins 3
```

**실행 옵션**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--parallel` | False | 병렬 다중 키 분석 활성화 |
| `--n` | 10 | 그룹 수 |
| `--samples` | 800 | 키당 샘플 수 |
| `--keys` | 3 | m_key2 종류 수 |
| `--bins` | 2 | bin_id 종류 수 |

---

### 코드에서 직접 사용

#### 1. 단일 분석

```python
import pandas as pd
from pwi_analysis import AnalysisConfig, pwi_analysis

metro = pd.read_csv("metro.csv")   # root_lot_id, wafer_id, item_value, m_key2
eds   = pd.read_csv("eds.csv")     # root_lot_id, wafer_id, bin_value,  bin_id

cfg = AnalysisConfig(
    n_groups=10,        # 등분산 그룹 수
    conf_level=0.95,    # 신뢰 수준 (사후 검정 alpha = 1 - conf_level)
    min_group_count=15, # 그룹 유지 최소 샘플 수
)

result, message = pwi_analysis(metro, eds, cfg)

if result:
    print(f"Window : [{result.window_low:.3f}, {result.window_high:.3f}]")
    print(f"PWI    : {result.pwi_index:.1f}%")
    print(f"p-value: {result.p_value:.4f}")
    print(f"Good groups: {result.good_groups}")
else:
    print(f"분석 종료: {message}")
```

#### 2. 병렬 분석 (다중 m_key2 × bin_id)

```python
from pwi_analysis import AnalysisConfig, run_parallel_pwi

metro_all = pd.read_csv("metro_all.csv")
eds_all   = pd.read_csv("eds_all.csv")

cfg = AnalysisConfig(n_groups=10)

# n_jobs=-1 → 사용 가능한 모든 CPU 코어 사용
results = run_parallel_pwi(metro_all, eds_all, cfg=cfg, n_jobs=-1)

for r in results:
    if r["result"]:
        res = r["result"]
        print(f"{r['m_key2']} × {r['bin_id']}: PWI={res.pwi_index}%  window=[{res.window_low:.2f}, {res.window_high:.2f}]")
    else:
        print(f"{r['m_key2']} × {r['bin_id']}: SKIP ({r['message']})")
```

---

## 데이터 스펙

### Metro 데이터

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `root_lot_id` | int/str | Lot 식별자 (EDS와 조인 키) |
| `wafer_id` | int/str | Wafer 식별자 (EDS와 조인 키) |
| `item_value` | float | 공정 측정값 (분석 X축) |
| `m_key2` | str | 측정 항목 키 (병렬 처리 분할 단위) |

### EDS 데이터

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `root_lot_id` | int/str | Lot 식별자 (Metro와 조인 키) |
| `wafer_id` | int/str | Wafer 식별자 (Metro와 조인 키) |
| `bin_value` | float | 수율 지표 (분석 Y축, 낮을수록 양호) |
| `bin_id` | str | Bin 종류 (병렬 처리 분할 단위) |

---

## 파라미터 설정 (`AnalysisConfig`)

```python
from pwi_analysis import AnalysisConfig

cfg = AnalysisConfig(
    n_groups=10,                # 등분산 그룹 수 (기본: 10)
    conf_level=0.95,            # 신뢰 수준 (기본: 0.95)
    min_group_count=15,         # 그룹 유지 최소 샘플 수 (기본: 15)
    min_valid_groups=3,         # 통계 검정에 필요한 최소 그룹 수 (기본: 3)
    skew_threshold=1.3,         # 정규/비정규 판단 왜도 임계값 (기본: ±1.3)
    metro_outlier_q_low=0.005,  # Metro 하위 이상치 컷 분위수 (기본: 0.5%)
    metro_outlier_q_high=0.995, # Metro 상위 이상치 컷 분위수 (기본: 99.5%)
    eds_outlier_q_high=0.9995,  # EDS 상위 이상치 컷 분위수 (기본: 99.95%)
    y_target_sigma_factor=0.25, # y_target = good_mean + factor × good_std (기본: 0.25)
    imag_tolerance=1e-10,       # 허근 판별 허수부 임계값 (기본: 1e-10)
)
```

---

## 테스트

```bash
# 전체 테스트 실행
python -m pytest tests/ -v

# 특정 모듈만
python -m pytest tests/test_pipeline.py -v
```

---

## 분석이 조기 종료되는 경우

| 메시지 | 원인 | 조치 |
|--------|------|------|
| `No matching data after merge` | Metro/EDS 키 불일치 | `root_lot_id`, `wafer_id` 컬럼 확인 |
| `Insufficient valid groups` | 샘플 수 부족 또는 `min_group_count` 과도 | `min_group_count` 낮추거나 데이터 추가 |
| `No significant difference` | 그룹 간 `bin_value` 차이 없음 | 정상 — 공정 윈도우 내 데이터로 판단 |
| `Complex roots` | 회귀 곡선이 `y_target`과 미교차 | `y_target_sigma_factor` 조정 |
