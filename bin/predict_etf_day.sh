#!/bin/bash
# ETF Day Trading 예측/랭킹 스크립트
cd "$(dirname "$0")/.."

MODEL="${1:-etf-day-v1}"
DATA_SET="${2:-etf_20260317}"
TOP_N="${3:-10}"
LOOKBACK="${4:-1}"

# 모델 확인
MODEL_DIR="models/$MODEL"
if [ ! -d "$MODEL_DIR" ]; then
    echo "모델이 없습니다: $MODEL_DIR"
    echo "먼저 학습하세요: ./bin/train_etf_day.sh"
    exit 1
fi

# 데이터 확인
DATA_DIR="data/$DATA_SET"
if [ ! -f "$DATA_DIR/training_scaled.csv" ]; then
    echo "데이터가 없습니다: $DATA_DIR/training_scaled.csv"
    exit 1
fi

echo "=========================================="
echo "  ETF Day Trading 예측/랭킹"
echo "=========================================="
echo "  모델      : $MODEL"
echo "  데이터셋  : $DATA_SET"
echo "  Top-N     : $TOP_N"
echo "  Lookback  : $LOOKBACK"
echo "=========================================="

python -m quantylab.trainer.etf_single_swing.prediction \
    --model "$MODEL" \
    --dataset "$DATA_SET" \
    --top-n "$TOP_N" \
    --lookback "$LOOKBACK"
