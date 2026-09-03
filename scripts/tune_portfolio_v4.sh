#!/bin/bash
# Portfolio 파라미터 튜닝 스크립트 — etf-swing-v4
# 6개 config × 150 ep 순차 실행 후 결과 비교
cd "$(dirname "$0")/.."

EPISODES=150
DATASET="data/etf_20260410"
BASE_MODEL="etf-swing-v3"
LOG_DIR="logs/tune_portfolio_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "  Portfolio 파라미터 튜닝 시작"
echo "  episodes=$EPISODES, log=$LOG_DIR"
echo "=================================================="

# ── 튜닝 Config 정의 ──
# 형식: "이름|lr_policy|lr_value|reward_scale|fee_penalty_scale|drawdown_scale|sharpe_scale|terminal_scale|entropy_start"
configs=(
  "A_baseline|0.0002|0.0005|10.0|5.0|15.0|2.0|30.0|0.10"
  "B_lower_lr|0.0001|0.0003|10.0|5.0|15.0|2.0|30.0|0.10"
  "C_high_sharpe|0.0002|0.0005|10.0|5.0|15.0|4.0|30.0|0.10"
  "D_high_terminal|0.0002|0.0005|10.0|5.0|15.0|2.0|50.0|0.10"
  "E_high_drawdown|0.0002|0.0005|10.0|5.0|25.0|2.0|30.0|0.10"
  "F_balanced|0.00015|0.0004|12.0|4.0|20.0|3.0|40.0|0.12"
  "G_high_reward|0.0002|0.0005|15.0|3.0|15.0|3.0|40.0|0.10"
  "H_aggressive|0.0003|0.0007|10.0|5.0|10.0|3.0|50.0|0.15"
)

RESULTS=()

for cfg in "${configs[@]}"; do
  IFS='|' read -r NAME LR_P LR_V RS FP DD SS TS ES <<< "$cfg"
  OUT_DIR="$LOG_DIR/$NAME"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "--------------------------------------------------"
  echo "[$NAME] lr=$LR_P | sharpe=$SS | dd=$DD | terminal=$TS | entropy=$ES"
  echo "--------------------------------------------------"

  python -m quantylab.trainer.etf_portfolio_swing.train \
    --dataset "$DATASET" \
    --episodes "$EPISODES" \
    --output-dir "$OUT_DIR" \
    --log-dir "$OUT_DIR" \
    --lr-policy "$LR_P" \
    --lr-value "$LR_V" \
    --reward-scale "$RS" \
    --fee-penalty-scale "$FP" \
    --drawdown-penalty-scale "$DD" \
    --drawdown-penalty-threshold 0.10 \
    --rolling-sharpe-scale "$SS" \
    --reward-terminal-scale "$TS" \
    --entropy-coef-start "$ES" \
    --entropy-coef-end 0.03 \
    --entropy-decay-episodes 300 \
    --base-model "$BASE_MODEL" \
    --update \
    --update-lr-scale 1.0 \
    --device cuda \
    --d-model 64 \
    --n-heads 4 \
    > "$OUT_DIR/run.log" 2>&1

  # 결과 집계
  RESULT=$(python3 -c "
import json, glob, sys
log_file = '$OUT_DIR/train_log.jsonl'
try:
    lines = open(log_file).readlines()
    records = [json.loads(l) for l in lines if l.strip()]
    if not records:
        print('$NAME|NO_DATA|0|0|0|0')
        sys.exit()
    best = max(records, key=lambda x: x.get('cagr', -999))
    last50 = records[-50:] if len(records) >= 50 else records
    avg_cagr = sum(r.get('cagr',0) for r in last50) / len(last50) * 100
    avg_mdd  = min(r.get('mdd', 0) for r in last50) * 100
    avg_sharpe = sum(r.get('sharpe',0) for r in last50) / len(last50)
    best_cagr = best.get('cagr', 0) * 100
    print(f'$NAME|{best_cagr:.2f}|{avg_cagr:.2f}|{avg_mdd:.2f}|{avg_sharpe:.3f}|{len(records)}')
except Exception as e:
    print(f'$NAME|ERR|0|0|0|0')
" 2>/dev/null)
  RESULTS+=("$RESULT")
  echo "결과: $RESULT"
done

# ── 최종 결과 비교 ──
echo ""
echo "=================================================="
echo "  튜닝 결과 요약 (last-50ep 기준)"
echo "  이름                | best_CAGR | avg_CAGR | avg_MDD  | avg_Sharpe | eps"
echo "--------------------------------------------------"
BEST_CFG=""
BEST_AVG=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r N BC AC MD SP EP <<< "$r"
  printf "  %-20s | %8s%% | %7s%% | %8s%% | %10s | %s\n" "$N" "$BC" "$AC" "$MD" "$SP" "$EP"
  # avg_CAGR 기준 최고 config 선택
  if python3 -c "exit(0 if float('${AC:-0}') > float('${BEST_AVG:-0}') else 1)" 2>/dev/null; then
    BEST_CFG="$N"
    BEST_AVG="$AC"
  fi
done
echo "=================================================="
echo "  최고 Config: $BEST_CFG (avg_CAGR=$BEST_AVG%)"
echo "=================================================="

# 최고 config의 파라미터 출력
for cfg in "${configs[@]}"; do
  IFS='|' read -r NAME LR_P LR_V RS FP DD SS TS ES <<< "$cfg"
  if [ "$NAME" = "$BEST_CFG" ]; then
    echo ""
    echo "  → v4 추천 파라미터:"
    echo "     lr_policy=$LR_P  lr_value=$LR_V"
    echo "     reward_scale=$RS  fee_penalty=$FP"
    echo "     drawdown_scale=$DD  sharpe_scale=$SS"
    echo "     terminal_scale=$TS  entropy_start=$ES"
  fi
done

echo ""
echo "로그 저장: $LOG_DIR"
