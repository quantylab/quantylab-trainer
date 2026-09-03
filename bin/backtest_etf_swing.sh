#!/bin/bash
# ETF Swing Trading 백테스트 스크립트
# 사용법:
#   ./bin/backtest_etf_swing.sh                                    # 최신 모델로 순차 백테스트
#   ./bin/backtest_etf_swing.sh --model etf-swing-v1               # 특정 모델 지정
#   ./bin/backtest_etf_swing.sh --no-sequential                    # 단일 백테스트
#   ./bin/backtest_etf_swing.sh --start-step 0 --max-steps 10000  # 특정 구간만

cd "$(dirname "$0")/.."

LATEST_DATA=$(ls -d data/etf_* 2>/dev/null | sort | tail -1 | xargs -r basename)
# 첫 인자가 --로 시작하지 않으면 데이터셋으로 사용
if [[ -n "$1" && "$1" != --* ]]; then
  DATA_SET="$1"
  shift
else
  DATA_SET="${LATEST_DATA:-etf_20260323}"
fi
POLICY=""
VALUE=""

# 최신 swing 모델 자동 탐지
LATEST_MODEL=$(ls -d models/etf-swing-v* 2>/dev/null | sort -t'v' -k2 -n | tail -1 | xargs -r basename)

# 옵션 파싱
EXTRA_ARGS=()
NO_SEQUENTIAL=0
HAS_MODEL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sequential)
      NO_SEQUENTIAL=1
      shift
      ;;
    --policy)
      POLICY="$2"
      shift 2
      ;;
    --value)
      VALUE="$2"
      shift 2
      ;;
    --model)
      LATEST_MODEL="$2"
      HAS_MODEL=1
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$LATEST_MODEL" ]; then
  echo "Swing 모델을 찾을 수 없습니다. --model 옵션으로 지정하세요."
  exit 1
fi

MODEL_ARGS=(--model "$LATEST_MODEL")
[ -n "$POLICY" ] && MODEL_ARGS+=(--policy "$POLICY")
[ -n "$VALUE"  ] && MODEL_ARGS+=(--value  "$VALUE")

echo "모델: $LATEST_MODEL"

if [ "$NO_SEQUENTIAL" -eq 1 ]; then
  echo "=== ETF Swing Trading 단일 백테스트 ==="
  python -m quantylab.trainer.etf_single_swing.backtest --dataset "$DATA_SET" "${MODEL_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
  echo "=== ETF Swing Trading 전체 연속 백테스트 ==="
  python -m quantylab.trainer.etf_single_swing.backtest --dataset "$DATA_SET" "${MODEL_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
