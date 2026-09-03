#!/bin/bash
# etf-swing-v1 2차 튜닝 — 베스트 기준(ht20_mh5_h5) 리스크 레버 탐색
cd "$(dirname "$0")/.."

LOG_DIR="output/tuning/v6_r2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "2차 튜닝 결과 저장: $LOG_DIR"

BASE="--hold-threshold 0.20 --min-hold-days 5 --max-holdings 5 --max-buy-per-day 3"

configs=(
  # baseline
  "base             $BASE"
  # trailing stop 단독
  "ts05             $BASE --trailing-stop-pct 0.05"
  "ts10             $BASE --trailing-stop-pct 0.10"
  "ts15             $BASE --trailing-stop-pct 0.15"
  "ts20             $BASE --trailing-stop-pct 0.20"
  # stop-loss 단독
  "sl08             $BASE --stop-loss-pct 0.08"
  "sl12             $BASE --stop-loss-pct 0.12"
  "sl15             $BASE --stop-loss-pct 0.15"
  # trailing-stop + stop-loss 조합
  "ts10_sl12        $BASE --trailing-stop-pct 0.10 --stop-loss-pct 0.12"
  "ts15_sl15        $BASE --trailing-stop-pct 0.15 --stop-loss-pct 0.15"
  # max-exposure 제한
  "exp80            $BASE --max-exposure 0.80"
  "exp85_ts10       $BASE --max-exposure 0.85 --trailing-stop-pct 0.10"
  # sell-threshold 명시적 조정
  "st10             $BASE --sell-threshold 0.10"
  "st05             $BASE --sell-threshold 0.05"
  # 종합 최적화 시도
  "combo_a          $BASE --trailing-stop-pct 0.12 --stop-loss-pct 0.12 --max-exposure 0.90"
  "combo_b          $BASE --trailing-stop-pct 0.10 --stop-loss-pct 0.10 --max-exposure 0.85 --sell-threshold 0.08"
)

TOTAL=${#configs[@]}
echo "총 ${TOTAL}개 설정"
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

  RESULT="$OUT_DIR/backtest_result.json"
  if [ -f "$RESULT" ]; then
    python3 -c "
import json
d = json.load(open('$RESULT'))
cagr   = d.get('cagr', 0)
sharpe = d.get('sharpe_ratio', 0)
mdd    = d.get('max_drawdown', 0)
calmar = d.get('calmar_ratio', 0)
wr     = d.get('win_rate', 0)
plr    = d.get('profit_loss_ratio', 0)
print(f'  → CAGR={cagr:.2f}%  Sharpe={sharpe:.3f}  MDD={mdd:.2f}%  Calmar={calmar:.3f}  WR={wr:.1f}%  PLR={plr:.3f}')
"
  else
    echo "  → 실패 (로그 확인: $LOG_DIR/${NAME}.log)"
  fi
done

echo ""
echo "========================================"
echo "집계 (CAGR 순)"
echo "========================================"
python3 -c "
import json, glob, os
rows = []
for f in sorted(glob.glob('$LOG_DIR/*/backtest_result.json')):
    name = os.path.basename(os.path.dirname(f))
    d = json.load(open(f))
    rows.append((name, d.get('cagr',0), d.get('sharpe_ratio',0), d.get('max_drawdown',0), d.get('calmar_ratio',0), d.get('win_rate',0), d.get('profit_loss_ratio',0)))
rows.sort(key=lambda x: -x[1])
print(f'{\"설정\":<20} {\"CAGR\":>8} {\"Sharpe\":>8} {\"MDD\":>8} {\"Calmar\":>8} {\"WR\":>6} {\"PLR\":>6}')
print('-'*75)
for r in rows:
    print(f'{r[0]:<20} {r[1]:>7.2f}% {r[2]:>8.3f} {r[3]:>7.2f}% {r[4]:>8.3f} {r[5]:>5.1f}% {r[6]:>6.3f}')
"
echo "완료: $LOG_DIR"
