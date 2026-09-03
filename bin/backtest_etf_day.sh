#!/bin/bash
# ETF Day Trading 백테스트 스크립트
# 사용법:
#   ./bin/backtest_etf_day.sh                                    # 전체 데이터 순차 백테스트
#   ./bin/backtest_etf_day.sh --no-sequential                    # 단일 백테스트
#   ./bin/backtest_etf_day.sh --start-step 0 --max-steps 10000  # 특정 구간만
#   ./bin/backtest_etf_day.sh --model etf-day-v1

cd "$(dirname "$0")/.."

LATEST_DATA=$(ls -d data/etf_* 2>/dev/null | sort | tail -1 | xargs -r basename)
DATA_SET="${1:-${LATEST_DATA:-etf_20260323}}"
POLICY=""
VALUE=""

# 옵션 파싱
EXTRA_ARGS=()
NO_SEQUENTIAL=0
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
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MODEL_ARGS=()
[ -n "$POLICY" ] && MODEL_ARGS+=(--policy "$POLICY")
[ -n "$VALUE"  ] && MODEL_ARGS+=(--value  "$VALUE")

# 기본 모델: output/ 디렉토리
if [ -z "$POLICY" ] && [ -z "$VALUE" ]; then
  for arg in "${EXTRA_ARGS[@]}"; do
    if [ "$arg" = "--model" ]; then
      HAS_MODEL=1
      break
    fi
  done
  if [ -z "$HAS_MODEL" ]; then
    MODEL_ARGS+=(--policy output/policy_best.pt --value output/value_best.pt)
  fi
fi

if [ "$NO_SEQUENTIAL" -eq 1 ]; then
  echo "=== ETF Day Trading 단일 백테스트 ==="
  python -m quantylab.trainer.etf_single_swing.backtest --dataset "$DATA_SET" --trading-method day "${MODEL_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
  echo "=== ETF Day Trading 순차 백테스트 ==="
  python -m quantylab.trainer.etf_single_swing.backtest --dataset "$DATA_SET" --trading-method day --sequential --chunk-size 21600 --hold-threshold 0.5 "${MODEL_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
