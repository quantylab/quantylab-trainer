#!/bin/bash
# 학습된 모델과 스케일러를 quantylab.com (192.168.0.48) 에 배포
#
# 사용법:
#   ./bin/deploy_models.sh                      # 최신 모델 + 스케일러 배포
#   ./bin/deploy_models.sh etf-swing-v5         # 특정 모델만 배포
#   ./bin/deploy_models.sh --scaler-only        # 스케일러만 배포
#   ./bin/deploy_models.sh --dry-run            # 시뮬레이션 (전송 안 함)

cd "$(dirname "$0")/.."

SSH_KEY="/home/quantylab/.ssh/id_rsa_quantylab"
REMOTE_USER="quantylab"
REMOTE_HOST="quantylab.com"
REMOTE_BASE="/home/quantylab/quantylab"
LOCAL_MODELS="models"
LOCAL_SCALERS="scalers"

# 옵션 파싱
MODEL_NAME=""
SCALER_ONLY=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scaler-only) SCALER_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) MODEL_NAME="$1"; shift ;;
  esac
done

RSYNC_OPTS="-avz --progress -e ssh -i $SSH_KEY"
[ "$DRY_RUN" -eq 1 ] && RSYNC_OPTS="$RSYNC_OPTS --dry-run"

echo "=========================================="
echo "  모델 배포: quantylab.com"
echo "=========================================="

# 모델 배포
if [ "$SCALER_ONLY" -eq 0 ]; then
  if [ -n "$MODEL_NAME" ]; then
    # 특정 모델만
    if [ ! -d "$LOCAL_MODELS/$MODEL_NAME" ]; then
      echo "모델을 찾을 수 없습니다: $LOCAL_MODELS/$MODEL_NAME"
      exit 1
    fi
    echo "[1/2] 모델 배포: $MODEL_NAME"
    rsync $RSYNC_OPTS \
      "$LOCAL_MODELS/$MODEL_NAME/" \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/models/$MODEL_NAME/"
  else
    # 전체 모델
    echo "[1/2] 전체 모델 배포"
    rsync $RSYNC_OPTS --delete \
      "$LOCAL_MODELS/" \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/models/"
  fi
  echo ""
else
  echo "[1/2] 모델 배포 건너뜀 (--scaler-only)"
fi

# 스케일러 배포
echo "[2/2] 스케일러 배포"
rsync $RSYNC_OPTS --delete \
  "$LOCAL_SCALERS/" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/scalers/"

echo ""
echo "=========================================="
echo "  배포 완료"
[ "$DRY_RUN" -eq 1 ] && echo "  (dry-run 모드 — 실제 전송 안 함)"
echo "=========================================="
