#!/bin/bash
# 전체 TIGER ETF 통합 학습 데이터 빌드 (일반화 모델용)
cd "$(dirname "$0")/.."

START_DATE="${1:-20150101}"
END_DATE="${2:-}"
NAME="${3:-}"
MIN_CANDLES="${4:-500}"
FEATURE_VERSION="${FEATURE_VERSION:-1}"

if [ -z "$NAME" ]; then
    NAME="etf_$(date +%Y%m%d)"
fi

echo "=========================================="
echo "  TIGER ETF 통합 학습 데이터 빌드"
echo "=========================================="
echo "  시작일        : $START_DATE"
echo "  종료일        : ${END_DATE:-최신}"
echo "  데이터셋 이름 : $NAME"
echo "  최소 캔들 수  : $MIN_CANDLES"
echo "  피처 버전     : $FEATURE_VERSION"
echo "  데이터 소스   : feature-vector API"
echo "=========================================="

CMD="python -m quantylab.trainer.feature --unified --source api --feature-version $FEATURE_VERSION --start-date $START_DATE --min-candles $MIN_CANDLES --name $NAME"
if [ -n "$END_DATE" ]; then
    CMD="$CMD --end-date $END_DATE"
fi

echo "$CMD"
$CMD
