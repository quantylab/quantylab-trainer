#!/bin/bash
# ETF Swing Trading 모의투자 스크립트
# 
# 사용법:
#   ./bin/swing_trade.sh signal          # 시그널만 확인
#   ./bin/swing_trade.sh rebalance       # 매도 + 매수 (리밸런싱)
#   ./bin/swing_trade.sh buy             # 매수만
#   ./bin/swing_trade.sh sell            # 매도만
#   ./bin/swing_trade.sh status          # 계좌 상태 조회
#
#   ./bin/swing_trade.sh rebalance --dry-run   # 시뮬레이션 (주문 안 함)
#   ./bin/swing_trade.sh rebalance --real      # 실계좌 (주의!)
#   ./bin/swing_trade.sh rebalance --price-band-pct 0.003 --monitor-end 15:15
#     # 모의계좌: 최초 현재가 ±0.3% 범위에서 장중 재호가
#
cd "$(dirname "$0")/.."

ACTION="${1:-signal}"
shift  # 첫 번째 인자(ACTION) 제거, 나머지는 pass-through

LATEST_DATA=$(ls -d data/etf_* 2>/dev/null | sort | tail -1 | xargs -r basename)
DATASET="${LATEST_DATA:-etf_20260323}"

echo "=========================================="
echo "  ETF Swing Trading 모의투자"
echo "=========================================="
echo "  액션     : $ACTION"
echo "  데이터셋 : $DATASET"
echo "  추가옵션 : $@"
echo "=========================================="
echo ""

# The source package is exposed as quantylab.trainer.etf_single_swing by pyproject.toml.
# Keep this wrapper aligned with that package name so the scheduled/manual
# paper-trading commands do not fail before the strategy starts.
# Remove the legacy standalone project path for this subprocess while retaining
# the Quantylab source path needed by the Kiwoom client.
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python -c '
import runpy
import sys

sys.path = [path for path in sys.path if "qlt-rl-etf-single-swing/src" not in path]
runpy.run_module("quantylab.trainer.etf_single_swing.swing_trading", run_name="__main__")
' \
    --action "$ACTION" \
    --base-path "$PWD" \
    --dataset "$DATASET" \
    "$@"
