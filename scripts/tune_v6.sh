#!/bin/bash
# etf-swing-v1 파라미터 튜닝 스크립트
cd "$(dirname "$0")/.."

LOG_DIR="output/tuning/v6_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "튜닝 결과 저장: $LOG_DIR"
echo ""

# hold-threshold / min-hold-days / max-holdings 조합
configs=(
  "ht05_mh1_h3  --hold-threshold 0.05 --min-hold-days 1 --max-holdings 3 --max-buy-per-day 3"
  "ht05_mh3_h3  --hold-threshold 0.05 --min-hold-days 3 --max-holdings 3 --max-buy-per-day 3"
  "ht05_mh5_h3  --hold-threshold 0.05 --min-hold-days 5 --max-holdings 3 --max-buy-per-day 3"
  "ht10_mh1_h3  --hold-threshold 0.10 --min-hold-days 1 --max-holdings 3 --max-buy-per-day 3"
  "ht10_mh3_h3  --hold-threshold 0.10 --min-hold-days 3 --max-holdings 3 --max-buy-per-day 3"
  "ht10_mh5_h3  --hold-threshold 0.10 --min-hold-days 5 --max-holdings 3 --max-buy-per-day 3"
  "ht10_mh3_h5  --hold-threshold 0.10 --min-hold-days 3 --max-holdings 5 --max-buy-per-day 3"
  "ht10_mh5_h5  --hold-threshold 0.10 --min-hold-days 5 --max-holdings 5 --max-buy-per-day 3"
  "ht15_mh3_h5  --hold-threshold 0.15 --min-hold-days 3 --max-holdings 5 --max-buy-per-day 3"
  "ht15_mh5_h5  --hold-threshold 0.15 --min-hold-days 5 --max-holdings 5 --max-buy-per-day 3"
  "ht20_mh3_h5  --hold-threshold 0.20 --min-hold-days 3 --max-holdings 5 --max-buy-per-day 3"
  "ht20_mh5_h5  --hold-threshold 0.20 --min-hold-days 5 --max-holdings 5 --max-buy-per-day 3"
  # 낙폭 방어 없이
  "ht10_mh3_h5_nodp --hold-threshold 0.10 --min-hold-days 3 --max-holdings 5 --max-buy-per-day 3 --drawdown-reduce-pct 0 --drawdown-pause-pct 0"
  # 낙폭 방어 강화
  "ht10_mh3_h5_dp   --hold-threshold 0.10 --min-hold-days 3 --max-holdings 5 --max-buy-per-day 3 --drawdown-reduce-pct 0.08 --drawdown-pause-pct 0.20"
)

TOTAL=${#configs[@]}
echo "총 ${TOTAL}개 설정 백테스트 시작"
echo "========================================"

for entry in "${configs[@]}"; do
  NAME=$(echo "$entry" | awk '{print $1}')
  ARGS=$(echo "$entry" | cut -d' ' -f2-)

  OUT_DIR="$LOG_DIR/$NAME"
  mkdir -p "$OUT_DIR"
  echo "[$NAME] $ARGS"

  python -m quantylab.trainer.etf_single_swing.backtest \
    --dataset etf_20260410 \
    --model etf-swing-v1 \
    --output-dir "$OUT_DIR" \
    --no-visualize \
    $ARGS \
    > "$LOG_DIR/${NAME}.log" 2>&1

  # 결과 요약 출력
  RESULT="$OUT_DIR/backtest_result.json"
  if [ -f "$RESULT" ]; then
    python3 -c "
import json
d = json.load(open('$RESULT'))
cagr  = d.get('cagr', 0) * 100
sharpe = d.get('sharpe_ratio', 0)
mdd   = d.get('max_drawdown', 0) * 100
calmar = d.get('calmar_ratio', 0)
wr    = d.get('win_rate', 0) * 100
print(f'  → CAGR={cagr:.2f}%  Sharpe={sharpe:.3f}  MDD={mdd:.2f}%  Calmar={calmar:.3f}  WR={wr:.1f}%')
"
  else
    echo "  → 결과 없음 (로그: $LOG_DIR/${NAME}.log)"
  fi
done

echo ""
echo "========================================"
echo "집계 결과 (CAGR 기준 정렬)"
echo "========================================"
python3 -c "
import json, glob, os
rows = []
for f in sorted(glob.glob('$LOG_DIR/*/backtest_result.json')):
    name = os.path.basename(os.path.dirname(f))
    d = json.load(open(f))
    rows.append((name, d.get('cagr',0)*100, d.get('sharpe_ratio',0), d.get('max_drawdown',0)*100, d.get('calmar_ratio',0)))
rows.sort(key=lambda x: -x[1])
print(f'{'설정':<30} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8}')
print('-'*70)
for name, cagr, sharpe, mdd, calmar in rows:
    print(f'{name:<30} {cagr:>7.2f}% {sharpe:>8.3f} {mdd:>7.2f}% {calmar:>8.3f}')
"

echo ""
echo "튜닝 완료: $LOG_DIR"
