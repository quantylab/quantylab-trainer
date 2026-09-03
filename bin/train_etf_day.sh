#!/bin/bash
# TIGER ETF Day Trading 지도학습 스크립트
# 전체 데이터를 한 번에 학습 (셔플 가능 — 매일 독립 IID)
cd "$(dirname "$0")/.."

trap 'echo -e "\n중단 신호 수신 — 학습 종료 중..."; kill -- -$$ 2>/dev/null; exit 1' SIGINT SIGTERM

LATEST_DATA=$(ls -d data/etf_* 2>/dev/null | sort | tail -1 | xargs -r basename)
DATA_SET="${1:-${LATEST_DATA:-etf_20260323}}"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-256}"
LR="${4:-0.001}"

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
echo "  TIGER ETF Day Trading 지도학습"
echo "=========================================="
echo "  데이터셋  : $DATA_SET"
echo "  전체 데이터: $DATA_LENGTH 행"
echo "  에폭      : $EPOCHS"
echo "  배치 크기 : $BATCH_SIZE"
echo "  학습률    : $LR"
echo "  Validation: start=$VAL_START"
echo "=========================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/train_${TIMESTAMP}"

CMD="python -m quantylab.trainer.etf_single_swing.train \
    --dataset $DATA_SET \
    --trading-method day \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --val-start $VAL_START \
    --log-dir $LOG_DIR \
    --output-dir output \
    --clean-run"

echo ""
echo "$CMD"
$CMD

echo ""
echo "=========================================="
echo "  학습 완료"
echo "=========================================="
