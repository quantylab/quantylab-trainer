#!/bin/bash
# TIGER ETF Swing Trading RL(PPO) 학습 스크립트
cd "$(dirname "$0")/.."

trap 'echo -e "\n중단 신호 수신 — 학습 종료 중..."; kill -- -$$ 2>/dev/null; exit 1' SIGINT SIGTERM

LATEST_DATA=$(ls -d data/etf_* 2>/dev/null | sort | tail -1 | xargs -r basename)
DATA_SET="${1:-${LATEST_DATA:-etf_20260323}}"
EPISODES="${2:-500}"
LR_POLICY="${3:-0.0001}"
LR_VALUE="${4:-0.0003}"
CHUNK_YEARS="${5:-1}"

# 데이터 확인
DATA_DIR="data/$DATA_SET"
TRAINING_FILE="$DATA_DIR/training_scaled.csv"
ETF_CODES_FILE="$DATA_DIR/etf_codes.csv"

if [ ! -f "$TRAINING_FILE" ]; then
    echo "학습 데이터가 없습니다: $TRAINING_FILE"
    echo "먼저 데이터를 빌드하세요: ./bin/build_training_data.sh"
    exit 1
fi

# Validation 시작점 계산 (마지막 ETF의 시작점)
VAL_START=$(python3 -c "
import pandas as pd
etf_codes = pd.read_csv('$ETF_CODES_FILE')['etf_code']
boundaries = [0]
for i in range(1, len(etf_codes)):
    if etf_codes.iloc[i] != etf_codes.iloc[i-1]:
        boundaries.append(i)
print(boundaries[-1])
")

DATA_LENGTH=$(wc -l < "$TRAINING_FILE")
DATA_LENGTH=$((DATA_LENGTH - 1))

echo "=========================================="
echo "  TIGER ETF Swing Trading RL(PPO) 학습"
echo "=========================================="
echo "  데이터셋    : $DATA_SET"
echo "  전체 데이터  : $DATA_LENGTH 행"
echo "  에피소드    : $EPISODES"
echo "  Policy LR   : $LR_POLICY"
echo "  Value LR    : $LR_VALUE"
echo "  청크 연수   : $CHUNK_YEARS"
echo "  Validation  : start=$VAL_START"
echo "=========================================="

# 최신 base-model 자동 탐지 (etf-swing-v* 중 가장 높은 버전)
BASE_MODEL=$(ls -d models/etf-swing-v* 2>/dev/null | sort -t'v' -k2 -n | tail -1 | xargs -r basename)
if [ -z "$BASE_MODEL" ]; then
    echo "base 모델 없음 → from scratch 학습"
    OUTPUT_MODEL="etf-swing-v1"
    BASE_ARGS=""
else
    # base-model 버전에서 +1 → output 모델 버전
    BASE_VER=$(echo "$BASE_MODEL" | grep -oP 'v\K[0-9]+')
    NEXT_VER=$((BASE_VER + 1))
    OUTPUT_MODEL="etf-swing-v${NEXT_VER}"
    BASE_ARGS="--base-model $BASE_MODEL --update"
    echo "base 모델: $BASE_MODEL → 출력: $OUTPUT_MODEL"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="output/train"

mkdir -p output/train

CMD="python -m quantylab.trainer.etf_single_swing.train \
    --dataset $DATA_SET \
    --trading-method swing \
    --episodes $EPISODES \
    --lr-policy $LR_POLICY \
    --lr-value $LR_VALUE \
    --chunk-years $CHUNK_YEARS \
    --val-start $VAL_START \
    --log-dir $LOG_DIR \
    --output-dir output/train \
    --gamma 0.995 \
    --hold-threshold 0.20 \
    --drawdown-penalty-scale 25.0 \
    --drawdown-penalty-threshold 0.12 \
    --rolling-sharpe-window 20 \
    --rolling-sharpe-scale 2.0 \
    --loss-aversion 1.2 \
    --policy-dropout 0.15 \
    --value-dropout 0.15 \
    --policy-weight-decay 0.0001 \
    --value-weight-decay 0.0003 \
    --validation-interval 5 \
    --early-stop-patience 35 \
    --early-stop-min-delta 0.10 \
    --early-stop-warmup-episodes 120 \
    $BASE_ARGS \
    --clean-run"

echo ""
echo "$CMD"
$CMD 2>&1 | tee output/train/train_run.log

# output → 새 버전 모델로 복사
echo ""
echo "모델 저장: models/$OUTPUT_MODEL"
cp -r output/train models/$OUTPUT_MODEL

# train_run.log가 symlink인 경우 실제 파일로 교체 (공유 파일이 다음 학습에 덮어쓰이는 것을 방지)
if [ -L "models/$OUTPUT_MODEL/train_run.log" ]; then
    real=$(readlink -f "models/$OUTPUT_MODEL/train_run.log")
    cp "$real" "models/$OUTPUT_MODEL/train_run.log.tmp"
    mv "models/$OUTPUT_MODEL/train_run.log.tmp" "models/$OUTPUT_MODEL/train_run.log"
fi

echo ""
echo "=========================================="
echo "  학습 완료 → $OUTPUT_MODEL"
echo "=========================================="
