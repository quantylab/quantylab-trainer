#!/bin/bash
# ETF Swing 파라미터 튜닝 스크립트
# 08-parameter-tuning.md 기반 3라운드 59개 설정 실행
#
# 사용법:
#   ./bin/tune_etf_swing.sh                        # 전체 3라운드 실행
#   ./bin/tune_etf_swing.sh --round 1              # Round 1만 실행
#   ./bin/tune_etf_swing.sh --round 2              # Round 2만 실행
#   ./bin/tune_etf_swing.sh --round 3              # Round 3만 실행
#   ./bin/tune_etf_swing.sh etf_20260410 --round 1 # 데이터셋 지정

cd "$(dirname "$0")/.."

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
DATASET=""
ROUND="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --round)
            ROUND="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --*)
            echo "알 수 없는 옵션: $1"; exit 1
            ;;
        *)
            DATASET="$1"
            shift
            ;;
    esac
done

# 최신 데이터셋 자동 탐지
if [ -z "$DATASET" ]; then
    DATASET=$(ls -d data/etf_* 2>/dev/null | sort | tail -1 | xargs -r basename)
    DATASET="${DATASET:-etf_20260410}"
fi

# 최신 모델 자동 탐지
if [ -z "$MODEL" ]; then
    MODEL=$(ls -d models/etf-swing-v* 2>/dev/null | sort -t'v' -k2 -n | tail -1 | xargs -r basename)
    MODEL="${MODEL:-etf-swing-v5}"
fi

BASE_CMD="python -m quantylab.trainer.etf_single_swing.backtest --model $MODEL --dataset $DATASET --sequential --no-visualize"

# 기준선 파라미터: 개별 효과 측정을 위해 모든 추가 옵션을 비활성화
BASELINE="--hold-threshold 0.05 --min-hold-days 0 --stop-loss-pct 0 \
    --trailing-stop-pct 0 --vol-target 0 \
    --drawdown-reduce-pct 0 --drawdown-pause-pct 0"

# ── 실행 카운터 ────────────────────────────────────────────────────────────────
TOTAL=0
COUNT=0
ROUND_DIR=""
SCRIPT_START=$SECONDS

progress_bar() {
    local filled=$(( $1 * 30 / $2 ))
    local bar=""
    for ((i=0; i<filled; i++));      do bar+="█"; done
    for ((i=filled; i<30; i++));     do bar+="░"; done
    printf "%s" "$bar"
}

fmt_time() {
    local s=$1
    if   (( s < 60 ));   then printf "%ds"         "$s"
    elif (( s < 3600 )); then printf "%dm%02ds"     $((s/60))   $((s%60))
    else                      printf "%dh%02dm%02ds" $((s/3600)) $(( (s%3600)/60 )) $((s%60))
    fi
}

run() {
    local label="$1"
    local dir_name="$2"
    shift 2
    COUNT=$((COUNT + 1))

    local pct=$(( COUNT * 100 / TOTAL ))
    local bar; bar=$(progress_bar "$COUNT" "$TOTAL")
    local run_start=$SECONDS

    echo ""
    echo "══════════════════════════════════════════════════════════"
    printf "  [%2d / %d]  %d%%  %s\n" "$COUNT" "$TOTAL" "$pct" "$bar"
    printf "  %s\n" "$label"
    echo "══════════════════════════════════════════════════════════"

    $BASE_CMD "$@" --output-dir "${ROUND_DIR}/${dir_name}"

    local elapsed=$(( SECONDS - run_start ))
    local total_elapsed=$(( SECONDS - SCRIPT_START ))
    local eta=""
    if (( COUNT > 0 && COUNT < TOTAL )); then
        local avg=$(( total_elapsed / COUNT ))
        local remaining=$(( avg * (TOTAL - COUNT) ))
        eta="  ETA $(fmt_time $remaining)"
    fi
    printf "  ── 완료 %-8s  누적 %s%s\n" \
        "$(fmt_time $elapsed)" "$(fmt_time $total_elapsed)" "$eta"
}

# ── Round 1: 개별 파라미터 탐색 (26개) ─────────────────────────────────────────
run_round1() {
    ROUND_DIR="output/tuning/round1"
    echo ""
    echo "██████████████████████████████████████████████████████"
    echo "  ROUND 1: 개별 파라미터 탐색 (26개 설정)"
    echo "██████████████████████████████████████████████████████"

    # ── 기준선 (1) ──────────────────────────────────────────────────────────────
    run "baseline" "baseline" \
        $BASELINE

    # ── 손절 (3) ────────────────────────────────────────────────────────────────
    run "손절 sl10" "sl10" \
        $BASELINE --stop-loss-pct 0.10

    run "손절 sl15" "sl15" \
        $BASELINE --stop-loss-pct 0.15

    run "손절 sl20" "sl20" \
        $BASELINE --stop-loss-pct 0.20

    # ── 최소 보유 기간 (4) ───────────────────────────────────────────────────────
    run "최소보유 mh2" "mh2" \
        $BASELINE --min-hold-days 2

    run "최소보유 mh3" "mh3" \
        $BASELINE --min-hold-days 3

    run "최소보유 mh4" "mh4" \
        $BASELINE --min-hold-days 4

    run "최소보유 mh5" "mh5" \
        $BASELINE --min-hold-days 5

    # ── 보유 임계값 (3) ──────────────────────────────────────────────────────────
    run "보유임계값 ht03" "ht03" \
        $BASELINE --hold-threshold 0.03

    run "보유임계값 ht08" "ht08" \
        $BASELINE --hold-threshold 0.08

    run "보유임계값 ht10" "ht10" \
        $BASELINE --hold-threshold 0.10

    # ── 변동성 타겟 (3) ──────────────────────────────────────────────────────────
    run "변동성타겟 vt005" "vt005" \
        $BASELINE --vol-target 0.005

    run "변동성타겟 vt01" "vt01" \
        $BASELINE --vol-target 0.01

    run "변동성타겟 vt015" "vt015" \
        $BASELINE --vol-target 0.015

    # ── 낙폭 방어 (3) ────────────────────────────────────────────────────────────
    run "낙폭방어 dd08/20" "dd08_20" \
        $BASELINE --drawdown-reduce-pct 0.08 --drawdown-pause-pct 0.20

    run "낙폭방어 dd10/25" "dd10_25" \
        $BASELINE --drawdown-reduce-pct 0.10 --drawdown-pause-pct 0.25

    run "낙폭방어 dd12/30" "dd12_30" \
        $BASELINE --drawdown-reduce-pct 0.12 --drawdown-pause-pct 0.30

    # ── 최대 보유 종목 (3) ───────────────────────────────────────────────────────
    run "최대보유 h3" "h3" \
        $BASELINE --max-holdings 3

    run "최대보유 h5" "h5" \
        $BASELINE --max-holdings 5

    run "최대보유 h7" "h7" \
        $BASELINE --max-holdings 7

    # ── 트레일링 스톱 (3) ────────────────────────────────────────────────────────
    run "트레일링스톱 ts10" "ts10" \
        $BASELINE --trailing-stop-pct 0.10

    run "트레일링스톱 ts15" "ts15" \
        $BASELINE --trailing-stop-pct 0.15

    run "트레일링스톱 ts20" "ts20" \
        $BASELINE --trailing-stop-pct 0.20

    # ── 최대 노출도 (3) ──────────────────────────────────────────────────────────
    run "최대노출도 me75" "me75" \
        $BASELINE --max-exposure 0.75

    run "최대노출도 me85" "me85" \
        $BASELINE --max-exposure 0.85

    run "최대노출도 me95" "me95" \
        $BASELINE --max-exposure 0.95
}

# ── Round 2: 상위 파라미터 조합 (17개) ─────────────────────────────────────────
# mh3 베이스 + 상위 파라미터 조합 (mh3이 모든 조합에 공통)
run_round2() {
    ROUND_DIR="output/tuning/round2"
    echo ""
    echo "██████████████████████████████████████████████████████"
    echo "  ROUND 2: 상위 파라미터 조합 (17개 설정)"
    echo "  ※ 약어: mh3=min-hold-days 3, ht10=hold-threshold 0.10"
    echo "  ※       h5=max-holdings 5, sl20=stop-loss 20%"
    echo "  ※       vt01=vol-target 0.01, dd=drawdown-defense 10%/25%"
    echo "██████████████████████████████████████████████████████"

    # ── 단독 (1) ────────────────────────────────────────────────────────────────
    run "mh3" "mh3" \
        $BASELINE --min-hold-days 3

    # ── 2-조합 (5) ──────────────────────────────────────────────────────────────
    run "mh3 + sl20" "mh3_sl20" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.20

    run "mh3 + vt01" "mh3_vt01" \
        $BASELINE --min-hold-days 3 --vol-target 0.01

    run "mh3 + ht10" "mh3_ht10" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10

    run "mh3 + h5" "mh3_h5" \
        $BASELINE --min-hold-days 3 --max-holdings 5

    run "mh3 + dd" "mh3_dd" \
        $BASELINE --min-hold-days 3 --drawdown-reduce-pct 0.10 --drawdown-pause-pct 0.25

    # ── 3-조합 (9) ──────────────────────────────────────────────────────────────
    run "mh3 + ht10 + h5  ★ RANK 1" "mh3_ht10_h5" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --max-holdings 5

    run "mh3 + sl20 + vt01  ★ RANK 2" "mh3_sl20_vt01" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.20 --vol-target 0.01

    run "mh3 + dd + sl20" "mh3_dd_sl20" \
        $BASELINE --min-hold-days 3 --drawdown-reduce-pct 0.10 --drawdown-pause-pct 0.25 --stop-loss-pct 0.20

    run "mh3 + dd + vt01" "mh3_dd_vt01" \
        $BASELINE --min-hold-days 3 --drawdown-reduce-pct 0.10 --drawdown-pause-pct 0.25 --vol-target 0.01

    run "mh3 + ht10 + sl20" "mh3_ht10_sl20" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --stop-loss-pct 0.20

    run "mh3 + ht10 + vt01" "mh3_ht10_vt01" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --vol-target 0.01

    run "mh3 + sl20 + h5" "mh3_sl20_h5" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.20 --max-holdings 5

    run "mh3 + vt01 + h5" "mh3_vt01_h5" \
        $BASELINE --min-hold-days 3 --vol-target 0.01 --max-holdings 5

    run "mh3 + dd + h5" "mh3_dd_h5" \
        $BASELINE --min-hold-days 3 --drawdown-reduce-pct 0.10 --drawdown-pause-pct 0.25 --max-holdings 5

    # ── 4-조합 (2) ──────────────────────────────────────────────────────────────
    run "mh3 + ht10 + h5 + sl20" "mh3_ht10_h5_sl20" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --max-holdings 5 --stop-loss-pct 0.20

    run "mh3 + ht10 + h5 + vt01" "mh3_ht10_h5_vt01" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --max-holdings 5 --vol-target 0.01
}

# ── Round 3: 미세 조정 (16개) ───────────────────────────────────────────────────
run_round3() {
    ROUND_DIR="output/tuning/round3"
    echo ""
    echo "██████████████████████████████████████████████████████"
    echo "  ROUND 3: 미세 조정 (16개 설정)"
    echo "  Winner A: mh3 + ht10 + h5  (CAGR 최대화)"
    echo "  Winner B: mh3 + sl20 + vt01  (Sharpe 최대화)"
    echo "██████████████████████████████████████████████████████"

    # ── Winner A 변형 (9) ────────────────────────────────────────────────────────
    run "A-base: mh3 + ht10 + h5  ★ BEST CAGR" "a_base" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --max-holdings 5

    run "A-h4: max-holdings 4" "a_h4" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --max-holdings 4

    run "A-h6: max-holdings 6" "a_h6" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --max-holdings 6

    run "A-ht08: hold-threshold 0.08" "a_ht08" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.08 --max-holdings 5

    run "A-ht12: hold-threshold 0.12" "a_ht12" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.12 --max-holdings 5

    run "A+sl20: +stop-loss 20%" "a_sl20" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --max-holdings 5 --stop-loss-pct 0.20

    run "A+dd: +drawdown-defense  ★ BEST MDD" "a_dd" \
        $BASELINE --min-hold-days 3 --hold-threshold 0.10 --max-holdings 5 \
            --drawdown-reduce-pct 0.10 --drawdown-pause-pct 0.25

    run "A-mh2: min-hold-days 2" "a_mh2" \
        $BASELINE --min-hold-days 2 --hold-threshold 0.10 --max-holdings 5

    run "A-mh4: min-hold-days 4" "a_mh4" \
        $BASELINE --min-hold-days 4 --hold-threshold 0.10 --max-holdings 5

    # ── Winner B 변형 (7) ────────────────────────────────────────────────────────
    run "B-base: mh3 + sl20 + vt01  ★ BEST Sharpe" "b_base" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.20 --vol-target 0.01

    run "B-sl18: stop-loss 18%" "b_sl18" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.18 --vol-target 0.01

    run "B-sl22: stop-loss 22%" "b_sl22" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.22 --vol-target 0.01

    run "B-sl25: stop-loss 25%" "b_sl25" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.25 --vol-target 0.01

    run "B-vt012: vol-target 0.012" "b_vt012" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.20 --vol-target 0.012

    run "B+ht10: +hold-threshold 0.10" "b_ht10" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.20 --vol-target 0.01 --hold-threshold 0.10

    run "B+h5: +max-holdings 5" "b_h5" \
        $BASELINE --min-hold-days 3 --stop-loss-pct 0.20 --vol-target 0.01 --max-holdings 5
}

# ── 실행 ───────────────────────────────────────────────────────────────────────
echo ""
echo "모델: $MODEL"
echo "데이터셋: $DATASET"
echo "라운드: $ROUND"

case "$ROUND" in
    1)   TOTAL=26; run_round1 ;;
    2)   TOTAL=17; run_round2 ;;
    3)   TOTAL=16; run_round3 ;;
    all) TOTAL=59; run_round1; run_round2; run_round3 ;;
    *)
        echo "오류: --round 옵션은 1, 2, 3, all 중 하나여야 합니다."
        exit 1
        ;;
esac

echo ""
echo "══════════════════════════════════════════════════════"
echo "  튜닝 완료: 총 $TOTAL개 설정 실행"
echo "══════════════════════════════════════════════════════"
