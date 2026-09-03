# Quantylab ETF Swing Trainer

TIGER ETF를 대상으로 강화학습 모델을 학습하고 백테스트·랭킹 추론·모의투자를 수행하는 프로젝트입니다.

현재 구현은 두 가지 전략을 제공합니다.

- `quantylab.trainer.etf_single_swing`: ETF별 점수를 산출해 상위 종목을 선택하는 단일 ETF PPO 전략
- `quantylab.trainer.etf_portfolio_swing`: 여러 ETF와 현금의 목표 비중을 동시에 산출하는 포트폴리오 PPO 전략

대상 종목은 [`src/target_etfs.py`](src/target_etfs.py)의 `TARGET_ETFS`에 정의된 TIGER ETF 64종입니다.

## 프로젝트 구조

```text
quantylab-trainer/
├── bin/                         # 데이터·학습·백테스트·예측·모의투자 실행 스크립트
├── src/
│   ├── etf_single_swing/        # 단일 ETF 선택형 PPO
│   │   ├── train.py             # 학습 CLI
│   │   ├── backtest.py          # 백테스트 CLI
│   │   ├── prediction.py        # 최신 랭킹 추론
│   │   ├── swing_trading.py     # Kiwoom REST 모의투자
│   │   ├── environment.py       # 단일 ETF 환경
│   │   ├── network.py           # 정책·가치 네트워크
│   │   └── trainer.py           # PPO 학습 루프
│   ├── etf_portfolio_swing/     # 포트폴리오 비중형 PPO
│   │   ├── train.py
│   │   ├── backtest.py
│   │   ├── environment.py
│   │   ├── network.py
│   │   └── trainer.py
│   ├── feature.py               # 학습 데이터 및 피처 생성
│   ├── target_etfs.py           # 투자 대상 ETF 목록
│   └── monitor.py               # 학습 모니터링
├── data/                        # /data/quantylab-trainer/data 심볼릭 링크
├── models/                      # /data/quantylab-trainer/models 심볼릭 링크
├── scalers/                     # 데이터셋별 스케일러
├── output/                      # 현재 학습 결과와 중간 산출물
├── logs/                        # 백테스트·튜닝 로그
├── scripts/                     # 튜닝 스크립트
└── tests/                       # 주문 모니터링 등 테스트
```

## 설치와 실행 환경

Python 3.10 이상과 PyTorch가 필요합니다. 패키지 이름은 `quantylab-trainer`이고 소스 패키지는 `quantylab.trainer`로 노출됩니다.

```bash
cd /home/quantylab/quantylab-trainer
python -m pip install -e .
```

`quantylab-trainer`는 독립 패키지입니다. DB 모델, feature-vector 처리, Quantylab REST 및 Kiwoom REST 클라이언트를 저장소 내부에 포함하므로 다른 Quantylab 패키지를 설치하지 않습니다.

GPU 사용 여부는 다음 명령으로 확인합니다.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 주요 환경변수

민감정보는 셸이나 서비스 환경에서 주입하고 저장소에 기록하지 않습니다.

| 변수 | 용도 |
|---|---|
| `QUANTYLAB_API_KEY` | 사전 계산된 feature-vector API 인증 |
| `QUANTYLAB_API_URL` | Quantylab API 주소 |
| Kiwoom REST 인증 변수 | `quantylab.trainer.kiwoom_rest.KiwoomRestClient`의 모의/실계좌 인증 |

`feature.py`는 API 키가 있으면 API를 우선 사용하고, 없으면 기존 DB 기반 경로를 사용합니다.

## 데이터셋 생성

전체 대상 ETF의 통합 학습 데이터를 feature-vector API에서 생성합니다.

```bash
cd /home/quantylab/quantylab-trainer
export QUANTYLAB_API_KEY='...'
./bin/build_training_data.sh 20150101 '' etf_$(date +%Y%m%d) 500
```

인자는 순서대로 시작일, 종료일, 데이터셋 이름, ETF별 최소 캔들 수입니다. `FEATURE_VERSION`의 기본값은 `1`입니다.

직접 실행할 때는 다음 모듈을 사용합니다.

```bash
python -m quantylab.trainer.feature \
  --unified \
  --source api \
  --feature-version 1 \
  --start-date 20150101 \
  --min-candles 500 \
  --name etf_20260904
```

생성되는 주요 파일은 다음과 같습니다.

```text
data/<dataset>/
├── environment.csv
├── training_scaled.csv
└── etf_codes.csv

scalers/<dataset>/
└── scaler.pkl
```

## 단일 ETF PPO

### 학습

단일 ETF 전략의 정식 모듈 경로는 `quantylab.trainer.etf_single_swing.train`입니다.

```bash
python -m quantylab.trainer.etf_single_swing.train \
  --base-path /home/quantylab/quantylab-trainer \
  --dataset etf_20260410 \
  --trading-method swing \
  --episodes 500 \
  --lr-policy 0.0001 \
  --lr-value 0.0003 \
  --gamma 0.995 \
  --hold-threshold 0.20 \
  --d-model 128 \
  --n-blocks 3 \
  --d-state 16 \
  --device auto \
  --output-dir output/train \
  --log-dir output/train \
  --clean-run
```

현재 기본 구조는 Mamba 정책·가치 네트워크입니다.

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--episodes` | 500 | 학습 에피소드 수 |
| `--hold-threshold` | 0.2 | 학습 환경의 행동 임계값 |
| `--d-model` | 128 | Mamba 표현 차원 |
| `--n-blocks` | 3 | Mamba 블록 수 |
| `--d-state` | 16 | SSM 상태 차원 |
| `--lr-policy` | 0.0001 | 정책 네트워크 학습률 |
| `--lr-value` | 0.0003 | 가치 네트워크 학습률 |
| `--gamma` | 0.995 | 할인율 |
| `--device` | `auto` | `auto`, `cpu`, `cuda` 중 선택 |

학습 결과는 먼저 `output/train/`에 생성됩니다. 버전 모델로 보존할 때는 체크포인트와 설정을 함께 `models/<model-name>/`에 둡니다.

```text
models/<model-name>/
├── policy_best.pt
├── value_best.pt
├── policy_final.pt
├── value_final.pt
└── train_config.json
```

### 백테스트

```bash
python -m quantylab.trainer.etf_single_swing.backtest \
  --base-path /home/quantylab/quantylab-trainer \
  --dataset etf_20260410 \
  --model etf-single-swing-v1 \
  --selector-mode auto \
  --hold-threshold 0.10 \
  --min-hold-days 3 \
  --max-holdings 5 \
  --max-buy-per-day 5 \
  --drawdown-reduce-pct 0.10 \
  --drawdown-pause-pct 0.25 \
  --device auto
```

주요 거래 비용 기본값은 편도 수수료 `0.00015`, ETF 거래세 `0`, 슬리피지 `0.0003`입니다. `--sequential`을 사용하면 연도별 순차 백테스트를 수행합니다.

### 최신 랭킹 추론

```bash
python -m quantylab.trainer.etf_single_swing.prediction \
  --base-path /home/quantylab/quantylab-trainer \
  --model etf-single-swing-v1 \
  --dataset etf_20260410 \
  --top-n 10 \
  --lookback 1 \
  --device cpu
```

## 포트폴리오 PPO

포트폴리오 전략은 각 ETF의 개별 순위 대신 ETF 전체와 현금의 목표 비중을 동시에 출력합니다.

### 학습

```bash
python -m quantylab.trainer.etf_portfolio_swing.train \
  --dataset data/etf_20260410 \
  --model-name etf-portfolio-swing-v2 \
  --lookback 20 \
  --d-model 64 \
  --n-heads 4 \
  --episodes 300 \
  --device cuda
```

`--model-name`을 지정하면 완료된 모델과 `train_config.json`이 `models/<model-name>/`에 저장됩니다. 추가 학습은 `--base-model <name> --update`를 사용합니다.

### 백테스트

```bash
python -m quantylab.trainer.etf_portfolio_swing.backtest \
  --dataset data/etf_20260410 \
  --model-name etf-portfolio-swing-v1 \
  --oos-start-date 20250101 \
  --realistic \
  --min-trading-price 10000 \
  --rebalance-band 0.01 \
  --trading-fee 0.00015 \
  --device cpu
```

`--model-name`을 사용하면 모델의 `train_config.json`에서 ETF 코드, lookback, 네트워크 차원을 불러오고 결과를 해당 모델 디렉터리의 `backtest_result.json`에 저장합니다.

## 모의투자

모의투자 구현은 [`src/etf_single_swing/swing_trading.py`](src/etf_single_swing/swing_trading.py)입니다. 래퍼 스크립트가 패키지 경로와 `--base-path`를 설정하므로 운영 시에는 래퍼 사용을 권장합니다.

```bash
cd /home/quantylab/quantylab-trainer

# 계좌 상태
./bin/swing_trade.sh status

# 시그널 생성
./bin/swing_trade.sh signal --model-dir models/etf-single-swing-v1

# 주문 없이 리밸런싱 검증
./bin/swing_trade.sh rebalance --model-dir models/etf-single-swing-v1 --dry-run

# 모의계좌 리밸런싱
./bin/swing_trade.sh rebalance --model-dir models/etf-single-swing-v1
```

### 액션별 실제 동작

| 액션 | 단일 ETF 모델 | Portfolio 모델 |
|---|---|---|
| `signal` | ETF별 점수·순위 생성 | ETF·현금 목표 비중 생성 |
| `buy` | 빈 슬롯에 상위 종목 매수 | 지원하지 않음 |
| `sell` | 새 시그널 없이 관리 대상 보유분 전량 매도 | 지원하지 않음 |
| `rebalance` | 하위 종목 매도 후 상위 종목 매수 | 목표 비중과 현재 비중의 차이를 주문 |
| `status` | 계좌와 저장된 당일 시그널 표시 | 포트폴리오 목표 수량 표시 |

> `sell` 단독 실행은 전량 매도 동작입니다. 일상적인 교체 매매는 반드시 `rebalance --dry-run`으로 먼저 검증하십시오.

### 주문 모니터링

기본 모의계좌 주문은 최초 현재가 ±0.5% 범위의 지정가로 시작합니다. 60초마다 미체결을 확인하고 15시 20분까지 체결 우선 방향으로 재호가합니다.

| 옵션 | 기본값 |
|---|---:|
| `--price-band-pct` | 0.005 |
| `--monitor-interval` | 60초 |
| `--monitor-end` | 15:20 |
| `--max-order-minutes` | 30분 |
| `--max-requotes` | 20회 |

지정가 모니터링은 `real=False`, `dry_run=False`, 모니터링 활성화 상태에서만 사용됩니다. `--no-monitor-orders` 또는 `--real`은 시장가·즉시 실행 경로를 사용하므로 주의해야 합니다.

로그와 시그널은 현재 구현상 다음 위치에 저장됩니다.

```text
src/logs/swing_trading/YYYYMMDD.log
src/logs/swing_trading/signals_YYYYMMDD.csv
```

## 모델과 데이터 현황

저장소의 `models/`와 `data/`는 대용량 데이터 볼륨을 가리키는 심볼릭 링크입니다. 2026-09-04 기준 확인되는 모델은 다음과 같습니다.

| 모델 | 유형 | 주요 파일 |
|---|---|---|
| `etf-single-swing-v1` | 단일 ETF PPO | `policy_best.pt`, `value_best.pt`, `train_config.json` |
| `etf-portfolio-swing-v1` | 포트폴리오 PPO | `policy_best.pt`, `value_best.pt`, `train_config.json`, 백테스트 결과 |

모델 이름만으로 구조를 추정하지 말고 항상 `train_config.json`과 체크포인트를 함께 확인합니다.

## 모델 배포

배포 전에는 반드시 dry-run으로 전송 대상을 확인합니다.

```bash
./bin/deploy_models.sh --dry-run
./bin/deploy_models.sh etf-single-swing-v1 --dry-run
```

실제 배포 스크립트는 로컬 `models/`, `scalers/`를 `quantylab.com:/home/quantylab/quantylab/` 아래로 동기화합니다. 전체 모델 배포에는 `--delete`가 포함되므로 대상과 백업 상태를 확인한 뒤 실행합니다.

## 테스트

```bash
cd /home/quantylab/quantylab-trainer
pytest -q
```

주문 모니터링 변경 시에는 최소한 다음 테스트를 별도로 실행합니다.

```bash
pytest -q tests/test_order_monitor.py
```

## 운영 원칙

1. 데이터셋, 스케일러, 모델의 피처 버전을 함께 관리합니다.
2. 백테스트 결과에는 수수료·슬리피지·보유 제한과 OOS 기간을 명시합니다.
3. 모의투자는 `status → signal → rebalance --dry-run → rebalance` 순서로 진행합니다.
4. 실계좌 옵션인 `--real`은 별도 승인과 검증 없이 사용하지 않습니다.
5. 모델 디렉터리를 배포할 때 체크포인트와 `train_config.json`을 함께 배포합니다.
