#!/bin/bash
# ETF Swing Trading 예측/랭킹 스크립트
cd "$(dirname "$0")/.."

MODEL="${1:-etf-swing-v1}"
START_DATE="${2:-$(date -d '7 days ago' +%Y%m%d)}"
END_DATE="${3:-}"
SCALER="${4:-etf_20260323}"
TOP_N="${5:-10}"
LOOKBACK="${6:-1}"

# 모델 확인
MODEL_DIR="models/$MODEL"
if [ ! -d "$MODEL_DIR" ]; then
    echo "모델이 없습니다: $MODEL_DIR"
    echo "먼저 학습하세요: ./bin/train_etf_swing.sh"
    exit 1
fi

echo "=========================================="
echo "  ETF Swing Trading 예측/랭킹"
echo "=========================================="
echo "  모델      : $MODEL"
echo "  스케일러  : $SCALER"
echo "  Top-N     : $TOP_N"
echo "  Lookback  : $LOOKBACK"

if [ -n "$START_DATE" ]; then
    echo "  기간      : $START_DATE ~ ${END_DATE:-최신}"
    echo "=========================================="
    python -m quantylab.trainer.etf_single_swing.prediction \
        --model "$MODEL" \
        --start-date "$START_DATE" \
        ${END_DATE:+--end-date "$END_DATE"} \
        --scaler "$SCALER" \
        --top-n "$TOP_N" \
        --lookback "$LOOKBACK"
else
    echo "  데이터셋  : $SCALER"
    echo "=========================================="
    python -m quantylab.trainer.etf_single_swing.prediction \
        --model "$MODEL" \
        --dataset "$SCALER" \
        --top-n "$TOP_N" \
        --lookback "$LOOKBACK"
fi
