"""
모의투자 모듈: ETF Swing Trading (Kiwoom REST API)

스윙 트레이딩 전략:
  - 매일 모든 타겟 ETF에 대해 모델 추론
  - 예측 수익률 상위 N개 종목 보유 (최대 max_select개)
  - 이미 보유 중인 종목은 유지, 시그널 하위 종목만 매도
  - 빈 슬롯에 신규 매수 후보 진입

사용법:
    # 신호 확인만 (주문 실행 안 함)
    python src/swing_trading.py --dry-run

    # 리밸런싱 (매도 후 매수)
    python src/swing_trading.py --action rebalance

    # 모의투자 실행 (매수만)
    python src/swing_trading.py --action buy

    # 모의투자 실행 (매도만)
    python src/swing_trading.py --action sell

    # 계좌 상태 조회
    python src/swing_trading.py --action status
"""
import os
import sys
import argparse
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sqlalchemy import text

# quantylab-trainer base 경로
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .network import MambaRegressionNetwork, MambaPolicyNetwork
from ..etf_portfolio_swing.network import PortfolioPolicyNetwork
from ..target_etfs import TARGET_ETFS
from ..feature import (
    load_all_data,
    build_single_etf_features,
    clean_and_clip_outliers,
    select_features,
    get_selected_features,
    remove_zero_variance,
    load_tiger_etf_list,
    DataQualityReport,
)

from ..kiwoom_rest import KiwoomRestClient
from ..db import db


# ── 로깅 ──
LOG_DIR = os.path.join(BASE_PATH, 'logs', 'swing_trading')
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger('swing_trading')
logger.setLevel(logging.INFO)

# 파일 핸들러 (일별 로그)
log_file = os.path.join(LOG_DIR, f'{datetime.now().strftime("%Y%m%d")}.log')
fh = logging.FileHandler(log_file, encoding='utf-8')
fh.setLevel(logging.INFO)

# 콘솔 핸들러
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


# ═══════════════════════════════════════════════════════
# 당일 시세 조회 (Kiwoom REST)
# ═══════════════════════════════════════════════════════

def _fetch_live_candles(client: KiwoomRestClient, etf_codes: list, data: dict):
    """Kiwoom REST API로 당일 시세 조회 → data dict에 주입

    DB에는 전일까지의 캔들만 있으므로, 장중/장후에 당일 시세를
    가져와서 각 DataFrame 끝에 추가한다.
    """
    today = datetime.now().strftime("%Y%m%d")

    # ── 1) ETF 당일 캔들 ──
    injected = 0
    for code in etf_codes:
        key = f'etf_{code}'
        if key not in data or data[key] is None or data[key].empty:
            continue

        existing_dates = set(data[key]['date'].astype(str))
        if today in existing_dates:
            continue  # 이미 있음

        try:
            candle_df = client.get_stock_day_candles(code, start_date=today)
            if candle_df is None or candle_df.empty:
                continue

            # 당일 행만 추출
            row = candle_df[candle_df['일자'] == today]
            if row.empty:
                continue

            row = row.iloc[0]
            prev_close = float(data[key].iloc[-1]['close']) if len(data[key]) > 0 else 0
            cur_price = abs(float(row['현재가']))
            diff = float(row['전일대비'])
            diff_ratio = (diff / (cur_price - diff) * 100) if (cur_price - diff) != 0 else 0.0

            new_row = pd.DataFrame([{
                'code': code,
                'date': today,
                'open': abs(float(row['시가'])),
                'high': abs(float(row['고가'])),
                'low': abs(float(row['저가'])),
                'close': cur_price,
                'diff': diff,
                'diff_ratio': diff_ratio,
                'diff_sign': str(row.get('전일대비기호', '')),
                'volume': int(abs(float(row['거래량']))),
                'price': int(abs(float(row['거래대금']))),
            }])

            data[key] = pd.concat([data[key], new_row], ignore_index=True)
            injected += 1
            logger.info(f"  당일시세 주입: {code} | {today} | "
                        f"시가={new_row.iloc[0]['open']:,.0f} "
                        f"현재가={cur_price:,.0f} "
                        f"거래량={new_row.iloc[0]['volume']:,}")
        except Exception as e:
            logger.warning(f"  {code} 당일시세 조회 실패: {e}")

    # ── 2) KOSPI/KOSDAQ 당일 시장 캔들 ──
    for mkt in ['kospi', 'kosdaq']:
        if mkt not in data or data[mkt] is None or data[mkt].empty:
            continue

        existing_dates = set(data[mkt]['date'].astype(str))
        if today in existing_dates:
            continue

        try:
            mkt_df = client.get_market_day_candles(mkt, start_date=today)
            if mkt_df is None or mkt_df.empty:
                continue

            row = mkt_df[mkt_df['일자'] == today]
            if row.empty:
                continue

            row = row.iloc[0]
            cur_price = float(row['현재가'])
            prev_close = float(data[mkt].iloc[-1]['close']) if len(data[mkt]) > 0 else 0
            diff = cur_price - prev_close if prev_close > 0 else 0
            diff_ratio = (diff / prev_close * 100) if prev_close > 0 else 0.0

            new_row = pd.DataFrame([{
                'code': mkt,
                'date': today,
                'open': float(row['시가']),
                'high': float(row['고가']),
                'low': float(row['저가']),
                'close': cur_price,
                'diff': diff,
                'diff_ratio': diff_ratio,
                'diff_sign': '',
                'volume': float(row['거래량']),
                'amount': float(row['거래대금']),
            }])

            data[mkt] = pd.concat([data[mkt], new_row], ignore_index=True)
            logger.info(f"  당일시세 주입: {mkt.upper()} | {today} | "
                        f"현재가={cur_price:,.2f} "
                        f"등락률={diff_ratio:+.2f}%")
        except Exception as e:
            logger.warning(f"  {mkt} 당일 시장시세 조회 실패: {e}")

    logger.info(f"당일시세 주입 완료: ETF {injected}/{len(etf_codes)}개"
                f" + KOSPI/KOSDAQ 시장지수")


# ═══════════════════════════════════════════════════════
# 피처 빌드
# ═══════════════════════════════════════════════════════

def build_today_features(
    etf_codes: list,
    scaler_path: str = None,
    start_date: str = "20150101",
    include_meta: bool = True,
    kiwoom_client: KiwoomRestClient = None,
    api_key: str = None,
    feature_version: str = "1",
) -> dict:
    """피처 벡터 조회.

    QUANTYLAB_API_KEY 환경변수(또는 api_key 인자)가 설정된 경우
    quantylab-api REST API에서 사전 계산된 피처 벡터를 가져옵니다.
    미설정 시 DB에서 직접 빌드합니다 (하위 호환).

    Returns:
        {etf_code: {'features': np.ndarray, 'date': str}}
    """
    from ..feature import load_feature_vectors_from_api

    _api_key = api_key or os.environ.get("QUANTYLAB_API_KEY", "")

    if _api_key:
        logger.info(f"피처 벡터 API 조회: {len(etf_codes)}개 ETF")
        api_result = load_feature_vectors_from_api(
            codes=etf_codes,
            version=feature_version,
            n=1,
            api_key=_api_key,
        )
        results = {}
        for code, info in api_result.items():
            results[code] = {
                'features': info['features'],
                'date': info['date'],
            }
            logger.info(f"  {code} ({TARGET_ETFS.get(code, '')}): date={info['date']}, "
                        f"features={len(info['features'])}d")
        logger.info(f"피처 벡터 API 완료: {len(results)}/{len(etf_codes)}개 성공")
        return results

    # ── fallback: DB 직접 빌드 ──────────────────────────────
    logger.info(f"피처 빌드 시작 (DB 직접): {len(etf_codes)}개 ETF")
    qc = DataQualityReport()

    # 스케일러 로드
    if not scaler_path or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"스케일러 파일 없음: {scaler_path}. "
            "QUANTYLAB_API_URL/QUANTYLAB_API_KEY를 설정하거나 scaler_path를 지정하세요."
        )
    scaler = joblib.load(scaler_path)

    # 공통 데이터 로드
    data = load_all_data(etf_codes, start_date, qc=qc)

    # 당일 시세 주입 (Kiwoom REST)
    if kiwoom_client:
        logger.info("당일 시세 조회 (Kiwoom REST API)")
        _fetch_live_candles(kiwoom_client, etf_codes, data)
    else:
        logger.warning("Kiwoom 클라이언트 없음 — 당일 시세 미반영 (DB 데이터만 사용)")

    # ETF 메타 정보
    etf_list = load_tiger_etf_list(min_candles=100)
    etf_info = {e['code']: e for e in etf_list}

    feature_cols = get_selected_features(include_meta=include_meta)
    results = {}

    for code in etf_codes:
        try:
            info = etf_info.get(code, {'name': TARGET_ETFS.get(code, ''), 'extra': {}})
            feature_df = build_single_etf_features(
                code, data,
                etf_name=info['name'],
                etf_extra=info.get('extra', {}),
                include_meta=include_meta,
                verbose=False,
                qc=qc,
            )

            # 정제
            feature_df = clean_and_clip_outliers(feature_df, warm_up_period=60, verbose=False)
            if len(feature_df) < 10:
                logger.warning(f"{code}: 데이터 부족 ({len(feature_df)}행)")
                continue

            # 피처 선택
            selected = select_features(feature_df, feature_cols)
            selected = remove_zero_variance(selected, verbose=False)

            # 스케일링 (학습 시 사용한 스케일러 적용)
            meta_cols = [c for c in selected.columns if c.startswith('meta_sector_')]
            non_meta_cols = [c for c in selected.columns if c not in meta_cols]

            scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else non_meta_cols
            aligned = pd.DataFrame(0.0, index=selected.index, columns=scaler_cols)
            for c in scaler_cols:
                if c in selected.columns:
                    aligned[c] = selected[c].values

            scaled_values = scaler.transform(aligned)
            scaled_values = np.clip(scaled_values, -5.0, 5.0)
            scaled_df = pd.DataFrame(scaled_values, columns=scaler_cols, index=selected.index)

            for c in meta_cols:
                if c in selected.columns:
                    scaled_df[c] = selected[c].values

            last_row = scaled_df.iloc[-1].values.astype(np.float32)
            last_row = np.nan_to_num(last_row, nan=0.0, posinf=0.0, neginf=0.0)

            env_cols = ['date', 'etf_open', 'etf_close']
            env_row = feature_df[[c for c in env_cols if c in feature_df.columns]].iloc[-1]
            latest_date = env_row.get('date', '')

            results[code] = {
                'features': last_row,
                'date': str(latest_date),
                'open': float(env_row.get('etf_open', 0)),
                'close': float(env_row.get('etf_close', 0)),
            }
            logger.info(f"  {code} ({TARGET_ETFS.get(code, '')}): date={latest_date}, "
                        f"features={len(last_row)}d")

        except Exception as e:
            logger.error(f"  {code} 피처 빌드 실패: {e}")
            continue

    logger.info(f"피처 빌드 완료: {len(results)}/{len(etf_codes)}개 성공")
    return results


# ═══════════════════════════════════════════════════════
# 모델 추론
# ═══════════════════════════════════════════════════════

def load_model(model_path: str, device: str = 'cpu'):
    """MambaRegressionNetwork 또는 MambaPolicyNetwork 로드"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    input_dim = checkpoint.get('input_dim', None)
    state_dict = checkpoint.get('state_dict', checkpoint)

    if input_dim is None:
        # state_dict에서 추론
        for key in ['input_proj.0.weight']:
            if key in state_dict:
                input_dim = state_dict[key].shape[1]
                break
    if input_dim is None:
        raise ValueError("모델 입력 차원을 감지할 수 없습니다")

    # 모델 타입 자동 감지: alpha_head 키가 있으면 PolicyNetwork
    is_policy = any(k.startswith('alpha_head') for k in state_dict)
    if is_policy:
        model = MambaPolicyNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
        logger.info("모델 타입: MambaPolicyNetwork (Beta distribution)")
    else:
        model = MambaRegressionNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
        logger.info("모델 타입: MambaRegressionNetwork")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"모델 로드: {model_path} (input_dim={input_dim})")
    return model, input_dim


def run_inference(model, features_dict: dict, model_input_dim: int, device: str = 'cpu') -> list:
    """전체 ETF에 대해 모델 추론 수행

    Returns:
        [(code, predicted_return, date), ...] sorted by predicted_return desc
    """
    predictions = []

    with torch.no_grad():
        for code, info in features_dict.items():
            features = info['features']

            # 입력 차원 맞춤
            if len(features) < model_input_dim:
                padded = np.zeros(model_input_dim, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            elif len(features) > model_input_dim:
                features = features[:model_input_dim]

            state = torch.FloatTensor(features).unsqueeze(0).to(device)
            output = model(state)
            if isinstance(output, tuple):
                # MambaPolicyNetwork: (alpha, beta) → Beta분포 평균 = alpha/(alpha+beta)
                alpha, beta = output
                pred = (alpha / (alpha + beta)).item()
            else:
                pred = output.item()

            predictions.append({
                'code': code,
                'name': TARGET_ETFS.get(code, ''),
                'predicted_return': pred,
                'date': info['date'],
                'open': info['open'],
                'close': info['close'],
            })

    predictions.sort(key=lambda x: x['predicted_return'], reverse=True)
    return predictions


# ═══════════════════════════════════════════════════════
# 포트폴리오 모델 추론
# ═══════════════════════════════════════════════════════

def _load_train_config(model_path: str) -> dict:
    """모델 인접 train_config.json 로드."""
    config_path = os.path.join(os.path.dirname(model_path), 'train_config.json')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


def detect_model_type(model_path: str) -> str:
    """단일 ETF 모델인지 portfolio 모델인지 감지."""
    cfg = _load_train_config(model_path)
    if cfg.get('model_type') == 'portfolio':
        return 'portfolio'

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'policy' in checkpoint and 'value' in checkpoint:
        return 'portfolio'
    return 'single'


def _fit_feature_dim(values: Any, n_features: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == n_features:
        return arr
    fitted = np.zeros(n_features, dtype=np.float32)
    fitted[:min(arr.size, n_features)] = arr[:n_features]
    return fitted


def _load_portfolio_config(model_path: str, default_lookback: int = 20,
                           default_d_model: int = 64, default_n_heads: int = 4) -> dict:
    cfg = _load_train_config(model_path)
    codes = [str(c).zfill(6) for c in cfg.get('etf_codes', TARGET_ETFS.keys())]
    return {
        'model_name': cfg.get('model_name', os.path.basename(os.path.dirname(model_path))),
        'etf_codes': codes,
        'n_assets': int(cfg.get('n_assets', len(codes))),
        'n_features': int(cfg.get('n_features', 115)),
        'lookback': int(cfg.get('lookback', default_lookback)),
        'd_model': int(cfg.get('d_model', default_d_model)),
        'n_heads': int(cfg.get('n_heads', default_n_heads)),
    }


def _load_portfolio_policy_model(model_path: str, cfg: dict, device: str = 'cpu'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'policy' in checkpoint:
        state_dict = checkpoint['policy']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        state_dict = checkpoint

    model = PortfolioPolicyNetwork(
        n_assets=cfg['n_assets'],
        n_features=cfg['n_features'],
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("모델 타입: PortfolioPolicyNetwork")
    logger.info(f"모델 로드: {model_path} (assets={cfg['n_assets']}, features={cfg['n_features']}, "
                f"lookback={cfg['lookback']})")
    return model


def _load_portfolio_feature_sequences_from_db(
    codes: list[str],
    lookback: int,
    n_features: int,
    feature_version: str,
) -> tuple[np.ndarray, str]:
    sequences = []
    latest_dates = []
    missing = []

    with db.get_session() as session:
        for code in codes:
            rows = session.execute(
                text(
                    """
                    SELECT x, y
                    FROM feature_vector
                    WHERE code = :code AND version = :version
                    ORDER BY x DESC
                    LIMIT :lookback
                    """
                ),
                {'code': code, 'version': feature_version, 'lookback': lookback},
            ).mappings().all()

            if len(rows) < lookback:
                missing.append(f"{code}({len(rows)}/{lookback})")
                continue

            ordered = list(reversed(rows))
            sequences.append(np.stack([
                _fit_feature_dim(row['y'], n_features) for row in ordered
            ], axis=0))
            latest_dates.append(str(ordered[-1]['x']))

    if missing:
        raise RuntimeError(f"피처 벡터 부족: {', '.join(missing[:10])}")
    if not sequences:
        raise RuntimeError("피처 벡터가 없습니다.")

    latest_date = min(latest_dates)
    if len(set(latest_dates)) > 1:
        logger.warning(f"ETF별 최신 피처 날짜가 다릅니다. 공통 기준일={latest_date}")
    return np.stack(sequences, axis=0), latest_date


def _load_portfolio_feature_sequences_from_api(
    codes: list[str],
    lookback: int,
    n_features: int,
    feature_version: str,
    api_key: str,
) -> tuple[np.ndarray, str]:
    from ..feature import load_feature_vectors_from_api

    result = load_feature_vectors_from_api(
        codes=codes,
        version=feature_version,
        n=lookback,
        api_key=api_key,
    )

    sequences = []
    latest_dates = []
    missing = []
    for code in codes:
        info = result.get(code) or {}
        records = info.get('all_records', [])
        if len(records) < lookback:
            missing.append(f"{code}({len(records)}/{lookback})")
            continue
        sequences.append(np.stack([
            _fit_feature_dim(r['features'], n_features) for r in records[-lookback:]
        ], axis=0))
        latest_dates.append(str(records[-1]['date']))

    if missing:
        raise RuntimeError(f"API 피처 벡터 부족: {', '.join(missing[:10])}")
    if not sequences:
        raise RuntimeError("API 피처 벡터가 없습니다.")

    latest_date = min(latest_dates)
    if len(set(latest_dates)) > 1:
        logger.warning(f"ETF별 최신 API 피처 날짜가 다릅니다. 공통 기준일={latest_date}")
    return np.stack(sequences, axis=0), latest_date


def _build_prev_weights(codes: list[str], holdings: dict, cash: float) -> np.ndarray:
    values = np.array([
        holdings.get(code, {}).get('qty', 0) * holdings.get(code, {}).get('cur_price', 0)
        for code in codes
    ], dtype=np.float32)
    total = float(values.sum() + cash)
    if total <= 0:
        return np.ones(len(codes) + 1, dtype=np.float32) / (len(codes) + 1)
    return np.concatenate([values / total, np.array([cash / total], dtype=np.float32)])


def generate_portfolio_signal(
    model_path: str,
    device: str = 'cpu',
    feature_version: str = "1",
    api_key: str = None,
    prev_weights: np.ndarray = None,
) -> dict:
    cfg = _load_portfolio_config(model_path)
    codes = cfg['etf_codes']

    _api_key = api_key or os.environ.get("QUANTYLAB_API_KEY", "")
    if _api_key:
        features, feature_date = _load_portfolio_feature_sequences_from_api(
            codes, cfg['lookback'], cfg['n_features'], feature_version, _api_key
        )
        feature_source = 'api'
    else:
        features, feature_date = _load_portfolio_feature_sequences_from_db(
            codes, cfg['lookback'], cfg['n_features'], feature_version
        )
        feature_source = 'db'

    model = _load_portfolio_policy_model(model_path, cfg, device)
    if prev_weights is None:
        prev_weights = np.ones(cfg['n_assets'] + 1, dtype=np.float32) / (cfg['n_assets'] + 1)

    with torch.no_grad():
        feature_t = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        prev_t = torch.as_tensor(prev_weights, dtype=torch.float32, device=device).unsqueeze(0)
        weights = model(feature_t, prev_t).squeeze(0).cpu().numpy()

    weights = np.clip(weights.astype(np.float64), 0.0, None)
    if weights.sum() <= 0:
        raise RuntimeError("모델 비중 합계가 0입니다.")
    weights = weights / weights.sum()

    asset_weights = weights[:len(codes)]
    cash_weight = float(weights[len(codes)])
    ranking = [
        {
            'code': code,
            'name': TARGET_ETFS.get(code, ''),
            'predicted_return': float(weight),
            'target_weight': float(weight),
            'date': feature_date,
        }
        for code, weight in zip(codes, asset_weights)
    ]
    ranking.sort(key=lambda x: x['target_weight'], reverse=True)

    return {
        'feature_source': feature_source,
        'feature_date': feature_date,
        'codes': codes,
        'weights': weights,
        'asset_weights': {code: float(weight) for code, weight in zip(codes, asset_weights)},
        'cash_weight': cash_weight,
        'ranking': ranking,
        'config': cfg,
    }


# ═══════════════════════════════════════════════════════
# 주문 실행
# ═══════════════════════════════════════════════════════

def get_available_cash(client: KiwoomRestClient) -> float:
    """주문 가능 현금 조회"""
    try:
        deposit = client.get_deposit(qry_tp="3")  # 추정 예수금
        # 추정 예수금 사용
        cash = float(deposit.get('entr', 0))
        logger.info(f"예수금: {cash:,.0f}원")
        return cash
    except Exception as e:
        logger.error(f"예수금 조회 실패: {e}")
        # fallback: 계좌평가현황
        balance = client.get_balance()
        summary = balance.get('summary', {})
        cash = float(summary.get('dps_evlt', 0))
        logger.info(f"예수금 (fallback): {cash:,.0f}원")
        return cash


def get_current_holdings(client: KiwoomRestClient) -> dict:
    """현재 보유 종목 조회

    Returns:
        {code: {'qty': int, 'avg_price': float, 'cur_price': float}}
    """
    holdings = {}
    try:
        df = client.get_holding_stocks()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                code = str(row['종목코드']).zfill(6)
                qty = int(row['보유수량'])
                if qty > 0:
                    holdings[code] = {
                        'qty': qty,
                        'avg_price': float(row['매입가']),
                        'cur_price': float(row['현재가']),
                        'name': row.get('종목명', ''),
                    }
    except Exception as e:
        logger.error(f"보유 종목 조회 실패: {e}")
    return holdings


def get_current_price(client: KiwoomRestClient, code: str) -> float:
    """현재가 조회"""
    try:
        info = client.get_stock_info(code)
        return abs(float(info.get('cur_prc', 0)))
    except Exception as e:
        logger.error(f"{code} 현재가 조회 실패: {e}")
        return 0


def _etf_tick_price(price: float, side: str) -> int:
    """ETF 호가단위(5원)에 맞춰 가격을 보수적으로 정렬한다."""
    if price <= 0:
        return 0
    # 부동소수점으로 10050이 10049.999...가 되는 경계 오차를 제거한다.
    ticks = price / 5
    epsilon = 1e-9
    return int((np.floor(ticks + epsilon) if side == 'buy' else np.ceil(ticks - epsilon)) * 5)


def _limit_price(reference_price: float, side: str, band_pct: float, progress: float) -> int:
    """허용 밴드 안에서 시간 경과에 따라 체결 우선 방향으로 이동한다."""
    progress = min(1.0, max(0.0, progress))
    band_pct = max(0.0, band_pct)
    low = reference_price * (1 - band_pct)
    high = reference_price * (1 + band_pct)
    raw = low + (high - low) * progress if side == 'buy' else high - (high - low) * progress
    return _etf_tick_price(raw, side)


def _submit_limit_order(client, side: str, code: str, qty: int, price: int) -> dict:
    fn = client.buy_order if side == 'buy' else client.sell_order
    return fn(code, qty, price=price, trde_tp="0")


def monitor_limit_orders(
    client: KiwoomRestClient,
    orders: list,
    band_pct: float = 0.005,
    poll_seconds: int = 60,
    end_time: str = "15:20",
    max_order_age_seconds: int = 1800,
    max_requotes: int = 20,
    signal_check_fn=None,
    now_fn=datetime.now,
    sleep_fn=time.sleep,
) -> list:
    """미체결 지정가 주문을 한 묶음으로 감시하고 밴드 안에서 재호가한다.

    ``orders``는 execute_* 함수가 반환한 결과 중 주문번호가 있는 항목이다.
    종료 시각에는 남은 주문을 취소하여 익일 주문으로 남기지 않는다.
    """
    pending = {
        str(o['ord_no']): dict(o, _result=o, remaining_qty=int(o['qty']), missing_count=0)
        for o in orders if o.get('success') and o.get('ord_no') not in ('', 'DRY_RUN')
    }
    if not pending:
        return orders

    start = now_fn()
    for order in pending.values():
        order['created_at'] = start
        order['requotes'] = 0
    hour, minute = map(int, end_time.split(':'))
    deadline = start.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if deadline <= start:
        logger.warning("주문 모니터링 종료 시각이 지났습니다. 신규 주문만 확인 후 잔량을 취소합니다.")
    duration = max(1.0, (deadline - start).total_seconds())
    logger.info(f"장중 주문 모니터링 시작: {len(pending)}건, 종료={end_time}, 주기={poll_seconds}초, "
                f"주문유효={max_order_age_seconds // 60}분, 최대재호가={max_requotes}회")

    interrupted = False
    try:
        while pending:
            now = now_fn()
            try:
                open_df = client.get_open_orders()
            except Exception as exc:
                logger.error(f"미체결 조회 실패: {exc}")
                if now >= deadline:
                    break
                sleep_fn(max(1, poll_seconds))
                continue

            open_by_no = {}
            if open_df is not None and not open_df.empty:
                for _, row in open_df.iterrows():
                    open_by_no[str(row['주문번호']).strip()] = row

            for ord_no, order in list(pending.items()):
                row = open_by_no.get(ord_no)
                if row is None:
                    # 주문 직후 조회 반영 지연을 체결로 오인하지 않도록 2회 연속 부재를 확인한다.
                    order['missing_count'] += 1
                    if order['missing_count'] >= 2:
                        logger.info(f"체결 완료: {order['code']} {order['side']} {order['qty']}주")
                        order['_result'].update(filled=True, remaining_qty=0,
                                                final_ord_no=ord_no,
                                                final_order_price=order['order_price'])
                        pending.pop(ord_no)
                    continue
                order['missing_count'] = 0

                remaining = int(abs(float(row.get('미체결수량', order['remaining_qty']))))
                order['remaining_qty'] = remaining
                order['_result']['remaining_qty'] = remaining
                if remaining <= 0:
                    order['_result'].update(filled=True, final_ord_no=ord_no,
                                            final_order_price=order['order_price'])
                    pending.pop(ord_no)
                    continue

                # 모델/전략이 제공한 신호가 더 이상 유효하지 않으면 추격하지 않는다.
                if signal_check_fn is not None:
                    try:
                        signal_valid = bool(signal_check_fn(order, now))
                    except Exception as exc:
                        logger.warning(f"신호 확인 실패 — 주문 유지: {order['code']} {exc}")
                        signal_valid = True
                    if not signal_valid:
                        cancel = client.cancel_order(ord_no, order['code'], cncl_qty=str(remaining))
                        if str(cancel.get('return_code', '0')) == '0':
                            order['_result'].update(cancelled_qty=remaining,
                                                    remaining_qty=remaining, filled=False,
                                                    cancel_reason='signal_invalid', final_ord_no=ord_no,
                                                    final_order_price=order['order_price'])
                            pending.pop(ord_no)
                            logger.info(f"신호 약화로 주문 취소: {order['code']} {order['side']}")
                        continue

                if now >= deadline:
                    continue

                age = (now - order['created_at']).total_seconds()
                if max_order_age_seconds > 0 and age >= max_order_age_seconds:
                    cancel = client.cancel_order(ord_no, order['code'], cncl_qty=str(remaining))
                    if str(cancel.get('return_code', '0')) == '0':
                        order['_result'].update(cancelled_qty=remaining, remaining_qty=remaining,
                                                filled=False, cancel_reason='order_timeout',
                                                final_ord_no=ord_no, final_order_price=order['order_price'])
                        pending.pop(ord_no)
                        logger.info(f"주문 유효시간 초과로 취소: {order['code']} {remaining}주")
                    continue

                if max_requotes > 0 and order['requotes'] >= max_requotes:
                    cancel = client.cancel_order(ord_no, order['code'], cncl_qty=str(remaining))
                    if str(cancel.get('return_code', '0')) == '0':
                        order['_result'].update(cancelled_qty=remaining, remaining_qty=remaining,
                                                filled=False, cancel_reason='max_requotes',
                                                final_ord_no=ord_no, final_order_price=order['order_price'])
                        pending.pop(ord_no)
                        logger.info(f"최대 재호가 횟수 초과로 취소: {order['code']} {remaining}주")
                    continue

                progress = (now - start).total_seconds() / duration
                new_price = _limit_price(order['reference_price'], order['side'], band_pct, progress)
                if new_price == order.get('order_price'):
                    continue

                cancel = client.cancel_order(ord_no, order['code'], cncl_qty=str(remaining))
                if str(cancel.get('return_code', '0')) != '0':
                    logger.warning(f"취소 실패: {order['code']} 주문={ord_no} {cancel.get('return_msg', '')}")
                    continue
                replacement = _submit_limit_order(client, order['side'], order['code'], remaining, new_price)
                new_no = str(replacement.get('ord_no', '')).strip()
                if not new_no or str(replacement.get('return_code', '0')) != '0':
                    logger.error(f"재주문 실패: {order['code']} {replacement.get('return_msg', '')}")
                    order['_result'].update(success=False, remaining_qty=remaining,
                                            reason='취소 후 재주문 실패')
                    pending.pop(ord_no)
                    continue
                pending.pop(ord_no)
                order.update(ord_no=new_no, qty=remaining, order_price=new_price)
                order['requotes'] += 1
                order['_result'].update(final_ord_no=new_no, final_order_price=new_price,
                                        remaining_qty=remaining)
                pending[new_no] = order
                logger.info(f"재호가: {order['code']} {order['side']} {remaining}주 @ {new_price:,}원")

            if pending and now_fn() < deadline:
                sleep_fn(max(1, poll_seconds))
            else:
                break
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("모니터링 중단 요청 — 미체결 잔량을 취소합니다.")

    for ord_no, order in list(pending.items()):
        try:
            client.cancel_order(ord_no, order['code'], cncl_qty=str(order['remaining_qty']))
            logger.info(f"종료 시각 잔량 취소: {order['code']} {order['remaining_qty']}주")
            order['_result'].update(cancelled_qty=order['remaining_qty'],
                                    remaining_qty=order['remaining_qty'], filled=False,
                                    final_ord_no=ord_no,
                                    final_order_price=order['order_price'])
        except Exception as exc:
            logger.error(f"잔량 취소 실패: {order['code']} 주문={ord_no}: {exc}")
    if interrupted:
        logger.info("미체결 정리 후 모니터링을 종료했습니다.")
    return orders


def execute_buy(
    client: KiwoomRestClient,
    code: str,
    cash: float,
    invest_ratio: float,
    trading_fee: float = 0.00015,
    max_invest: float = 0,
    dry_run: bool = True,
    limit_band_pct: float = 0.0,
) -> dict:
    """매수 주문 실행

    Args:
        client: Kiwoom REST client
        code: 종목코드
        cash: 가용 현금
        invest_ratio: 투자 비율 (0~1)
        trading_fee: 거래 수수료율
        max_invest: 종목당 최대 투자 금액 (0이면 제한 없음)
        dry_run: True이면 주문 실행하지 않음

    Returns:
        {'success': bool, 'code': str, 'qty': int, 'price': float, 'amount': float, 'ord_no': str}
    """
    name = TARGET_ETFS.get(code, '')

    # 현재가 조회
    price = get_current_price(client, code)
    if price <= 0:
        logger.error(f"매수 실패: {code} ({name}) 현재가 조회 불가")
        return {'success': False, 'code': code, 'reason': '현재가 조회 실패'}

    # 투자 금액 계산
    invest_amount = cash * invest_ratio
    if max_invest > 0:
        invest_amount = min(invest_amount, max_invest)
        logger.info(f"종목당 최대 투자 금액 제한: {max_invest:,.0f}원")
    fee_amount = invest_amount * trading_fee / (1 + trading_fee)
    net_invest = invest_amount - fee_amount
    qty = int(net_invest / price)

    if qty <= 0:
        logger.warning(f"매수 수량 0: {code} ({name}), 현재가={price:,.0f}, 투자금={invest_amount:,.0f}")
        return {'success': False, 'code': code, 'reason': '매수 수량 0'}

    actual_amount = qty * price
    logger.info(f"매수 {'(DRY RUN) ' if dry_run else ''}| {code} ({name}) | "
                f"{qty}주 × {price:,.0f}원 = {actual_amount:,.0f}원 "
                f"(투자비율: {invest_ratio:.1%})")

    if dry_run:
        return {'success': True, 'code': code, 'qty': qty, 'price': price,
                'amount': actual_amount, 'ord_no': 'DRY_RUN', 'dry_run': True}

    # 모니터링 모드는 지정가, 그 외에는 시장가
    try:
        order_price = _limit_price(price, 'buy', limit_band_pct, 0.0) if limit_band_pct > 0 else 0
        result = client.buy_order(code, qty, price=order_price,
                                  trde_tp="0" if limit_band_pct > 0 else "3")
        ord_no = result.get('ord_no', '')
        return_code = result.get('return_code', '')
        return_msg = result.get('return_msg', '')

        if return_code and str(return_code) != '0':
            logger.error(f"매수 주문 실패: {code} ({name}), code={return_code}, msg={return_msg}")
            return {'success': False, 'code': code, 'reason': return_msg}

        logger.info(f"매수 주문 성공: {code} ({name}), 주문번호={ord_no}")
        return {'success': True, 'code': code, 'qty': qty, 'price': price,
                'amount': actual_amount, 'ord_no': ord_no, 'side': 'buy',
                'reference_price': price, 'order_price': order_price}

    except Exception as e:
        logger.error(f"매수 주문 예외: {code} ({name}), {e}")
        return {'success': False, 'code': code, 'reason': str(e)}


def execute_buy_qty(
    client: KiwoomRestClient,
    code: str,
    qty: int,
    price: float = None,
    dry_run: bool = True,
    limit_band_pct: float = 0.0,
) -> dict:
    """지정 수량만큼 매수 주문."""
    name = TARGET_ETFS.get(code, '')
    if qty <= 0:
        return {'success': False, 'code': code, 'reason': '매수 수량 0'}

    if price is None or price <= 0:
        price = get_current_price(client, code)
    if price <= 0:
        return {'success': False, 'code': code, 'reason': '현재가 조회 실패'}

    amount = qty * price
    logger.info(f"매수 {'(DRY RUN) ' if dry_run else ''}| {code} ({name}) | "
                f"{qty}주 × {price:,.0f}원 = {amount:,.0f}원")

    if dry_run:
        return {'success': True, 'code': code, 'qty': qty, 'price': price,
                'amount': amount, 'ord_no': 'DRY_RUN', 'dry_run': True}

    try:
        order_price = _limit_price(price, 'buy', limit_band_pct, 0.0) if limit_band_pct > 0 else 0
        result = client.buy_order(code, qty, price=order_price,
                                  trde_tp="0" if limit_band_pct > 0 else "3")
        ord_no = result.get('ord_no', '')
        return_code = result.get('return_code', '')
        return_msg = result.get('return_msg', '')
        if return_code and str(return_code) != '0':
            logger.error(f"매수 주문 실패: {code} ({name}), code={return_code}, msg={return_msg}")
            return {'success': False, 'code': code, 'reason': return_msg}
        return {'success': True, 'code': code, 'qty': qty, 'price': price,
                'amount': amount, 'ord_no': ord_no, 'side': 'buy',
                'reference_price': price, 'order_price': order_price}
    except Exception as e:
        logger.error(f"매수 주문 예외: {code} ({name}), {e}")
        return {'success': False, 'code': code, 'reason': str(e)}


def execute_sell_qty(
    client: KiwoomRestClient,
    code: str,
    qty: int,
    price: float = None,
    dry_run: bool = True,
    limit_band_pct: float = 0.0,
) -> dict:
    """지정 수량만큼 매도 주문."""
    name = TARGET_ETFS.get(code, '')
    if qty <= 0:
        return {'success': False, 'code': code, 'reason': '매도 수량 0'}

    if price is None or price <= 0:
        price = get_current_price(client, code)
    if price <= 0:
        return {'success': False, 'code': code, 'reason': '현재가 조회 실패'}

    amount = qty * price
    logger.info(f"매도 {'(DRY RUN) ' if dry_run else ''}| {code} ({name}) | "
                f"{qty}주 × {price:,.0f}원 = {amount:,.0f}원")

    if dry_run:
        return {'success': True, 'code': code, 'qty': qty, 'price': price,
                'amount': amount, 'ord_no': 'DRY_RUN', 'dry_run': True}

    try:
        order_price = _limit_price(price, 'sell', limit_band_pct, 0.0) if limit_band_pct > 0 else 0
        result = client.sell_order(code, qty, price=order_price,
                                   trde_tp="0" if limit_band_pct > 0 else "3")
        ord_no = result.get('ord_no', '')
        return_code = result.get('return_code', '')
        return_msg = result.get('return_msg', '')
        if return_code and str(return_code) != '0':
            logger.error(f"매도 주문 실패: {code} ({name}), code={return_code}, msg={return_msg}")
            return {'success': False, 'code': code, 'reason': return_msg}
        return {'success': True, 'code': code, 'qty': qty, 'price': price,
                'amount': amount, 'ord_no': ord_no, 'side': 'sell',
                'reference_price': price, 'order_price': order_price}
    except Exception as e:
        logger.error(f"매도 주문 예외: {code} ({name}), {e}")
        return {'success': False, 'code': code, 'reason': str(e)}


def execute_sell_all(
    client: KiwoomRestClient,
    target_codes: list = None,
    dry_run: bool = True,
    limit_band_pct: float = 0.0,
) -> list:
    """보유 종목 전량 매도

    Args:
        client: Kiwoom REST client
        target_codes: 매도 대상 종목 코드 (None이면 전체)
        dry_run: True이면 주문 실행하지 않음

    Returns:
        [{'success': bool, 'code': str, 'qty': int, ...}]
    """
    holdings = get_current_holdings(client)
    if not holdings:
        logger.info("보유 종목 없음 — 매도 불필요")
        return []

    results = []
    for code, info in holdings.items():
        if target_codes and code not in target_codes:
            continue

        name = info.get('name', TARGET_ETFS.get(code, ''))
        qty = info['qty']

        if qty <= 0:
            continue

        logger.info(f"매도 {'(DRY RUN) ' if dry_run else ''}| {code} ({name}) | "
                     f"{qty}주 (현재가: {info['cur_price']:,.0f}원)")

        if dry_run:
            results.append({'success': True, 'code': code, 'qty': qty,
                            'price': info['cur_price'],
                            'amount': qty * info['cur_price'],
                            'ord_no': 'DRY_RUN', 'dry_run': True})
            continue

        try:
            order_price = (_limit_price(info['cur_price'], 'sell', limit_band_pct, 0.0)
                           if limit_band_pct > 0 else 0)
            result = client.sell_order(code, qty, price=order_price,
                                       trde_tp="0" if limit_band_pct > 0 else "3")
            ord_no = result.get('ord_no', '')
            return_code = result.get('return_code', '')
            return_msg = result.get('return_msg', '')

            if return_code and str(return_code) != '0':
                logger.error(f"매도 주문 실패: {code} ({name}), code={return_code}, msg={return_msg}")
                results.append({'success': False, 'code': code, 'reason': return_msg})
                continue

            logger.info(f"매도 주문 성공: {code} ({name}), 주문번호={ord_no}")
            results.append({'success': True, 'code': code, 'qty': qty,
                            'price': info['cur_price'],
                            'amount': qty * info['cur_price'],
                            'ord_no': ord_no, 'side': 'sell',
                            'reference_price': info['cur_price'],
                            'order_price': order_price})

        except Exception as e:
            logger.error(f"매도 주문 예외: {code} ({name}), {e}")
            results.append({'success': False, 'code': code, 'reason': str(e)})

    return results


# ═══════════════════════════════════════════════════════
# 시그널 저장/로드
# ═══════════════════════════════════════════════════════

def save_signals(predictions: list, output_path: str):
    """시그널 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    logger.info(f"시그널 저장: {output_path}")


def load_signals(signal_path: str) -> list:
    """시그널 로드"""
    if not os.path.exists(signal_path):
        raise FileNotFoundError(f"시그널 파일 없음: {signal_path}")
    df = pd.read_csv(signal_path)
    return df.to_dict('records')


# ═══════════════════════════════════════════════════════
# 메인 트레이딩 로직
# ═══════════════════════════════════════════════════════

def generate_signals(
    model_path: str,
    scaler_path: str = None,
    include_meta: bool = True,
    device: str = 'cpu',
    kiwoom_client: KiwoomRestClient = None,
    api_key: str = None,
    feature_version: str = "1",
) -> list:
    """모델 추론으로 매매 시그널 생성

    Returns:
        [(code, name, predicted_return, date), ...] sorted by predicted_return desc
    """
    # 모델 로드
    model, input_dim = load_model(model_path, device)

    # 타겟 ETF 코드
    etf_codes = list(TARGET_ETFS.keys())

    # 피처 벡터 조회 (API 우선, fallback: DB 직접 빌드)
    features_dict = build_today_features(
        etf_codes, scaler_path,
        include_meta=include_meta,
        kiwoom_client=kiwoom_client,
        api_key=api_key,
        feature_version=feature_version,
    )

    if not features_dict:
        logger.error("피처 빌드 결과 없음")
        return []

    # 추론
    predictions = run_inference(model, features_dict, input_dim, device)
    return predictions


def run_swing_trading(
    action: str,
    model_path: str,
    scaler_path: str = None,
    hold_threshold: float = 0.001,
    invest_ratio: float = 0.95,
    trading_fee: float = 0.00015,
    include_meta: bool = True,
    device: str = 'cpu',
    dry_run: bool = True,
    real: bool = False,
    max_select: int = 5,
    max_invest: float = 30_000_000,
    api_key: str = None,
    feature_version: str = "1",
    min_order_amount: float = 10_000,
    monitor_orders: bool = True,
    price_band_pct: float = 0.005,
    monitor_interval: int = 60,
    monitor_end: str = "15:20",
    max_order_age_minutes: int = 30,
    max_requotes: int = 20,
):
    """스윙 트레이딩 실행

    Args:
        action: 'signal' | 'buy' | 'sell' | 'rebalance' | 'status'
        model_path: 학습된 모델 경로
        scaler_path: 스케일러 경로
        hold_threshold: 매수 기준 (예측 수익률 > threshold)
        invest_ratio: 현금 대비 투자 비율
        trading_fee: 거래 수수료율
        include_meta: 메타 피처 포함 여부
        device: 'cpu' or 'cuda'
        dry_run: True이면 주문 실행 안 함
        real: True이면 실계좌 (False: 모의)
        max_select: 최대 동시 보유 종목 수
        max_invest: 종목당 최대 투자 금액
    """
    today = datetime.now().strftime('%Y%m%d')
    signal_path = os.path.join(LOG_DIR, f'signals_{today}.csv')

    logger.info("=" * 60)
    logger.info("  ETF Swing Trading 모의투자")
    logger.info("=" * 60)
    logger.info(f"  날짜          : {today}")
    logger.info(f"  모드          : {'실계좌' if real else '모의투자'}")
    logger.info(f"  액션          : {action}")
    logger.info(f"  매수 기준     : predicted_return > {hold_threshold}")
    logger.info(f"  투자 비율     : {invest_ratio:.0%}")
    logger.info(f"  최대 보유     : {max_select}종목")
    logger.info(f"  DRY RUN       : {dry_run}")
    monitored_band = price_band_pct if monitor_orders and not real and not dry_run else 0.0
    logger.info(f"  주문 방식     : {'지정가 장중 모니터링' if monitored_band > 0 else '시장가/즉시 실행'}")
    if monitored_band > 0:
        logger.info(f"  가격 범위     : 최초 현재가 ±{monitored_band:.2%}, 종료 {monitor_end}")
    logger.info(f"  종목당 최대   : {max_invest:,.0f}원" if max_invest > 0 else "  종목당 최대   : 제한 없음")
    logger.info("=" * 60)

    model_type = detect_model_type(model_path)
    logger.info(f"  모델 타입     : {model_type}")

    # Kiwoom REST 클라이언트 (시그널 생성 시에도 당일 시세 조회용으로 필요)
    client = KiwoomRestClient(real=real)
    logger.info(f"Kiwoom REST 연결: {'실계좌' if real else '모의투자'}")

    if model_type == 'portfolio':
        if action in ('buy', 'sell'):
            raise ValueError("portfolio 모델은 action=signal|rebalance|status만 지원합니다.")

        cfg = _load_portfolio_config(model_path)
        codes = cfg['etf_codes']
        managed_codes = set(codes)
        cash = get_available_cash(client)
        holdings = get_current_holdings(client)
        holdings = {code: info for code, info in holdings.items() if code in managed_codes}
        prev_weights = _build_prev_weights(codes, holdings, cash)

        logger.info("")
        logger.info("[1] 포트폴리오 비중 생성")
        signal = generate_portfolio_signal(
            model_path=model_path,
            device=device,
            feature_version=feature_version,
            api_key=api_key,
            prev_weights=prev_weights,
        )

        logger.info(f"  피처 소스     : {signal['feature_source']}")
        logger.info(f"  피처 날짜     : {signal['feature_date']}")
        logger.info(f"  현금 비중     : {signal['cash_weight']:.2%}")
        logger.info("")
        logger.info(f"{'='*90}")
        logger.info(f"{'순위':>4} {'코드':>8} {'종목명':<20} {'목표비중':>10} {'보유':>4}  {'결정사유'}")
        logger.info(f"{'='*90}")
        for i, item in enumerate(signal['ranking'], start=1):
            is_held = item['code'] in holdings
            held = '●' if is_held else ' '
            reason = '리밸런싱 대상' if item['target_weight'] * invest_ratio > 0 else '현금 유지'
            logger.info(f"{i:>4} {item['code']:>8} {item['name']:<20} "
                        f"{item['target_weight']:>9.2%} {held:>4}  {reason}")

        save_signals(signal['ranking'], signal_path)

        if action == 'signal':
            logger.info("")
            logger.info("완료")
            return

        prices = {code: get_current_price(client, code) for code in codes}
        managed_value = sum(holdings.get(code, {}).get('qty', 0) * prices.get(code, 0) for code in codes)
        capital = managed_value + cash * invest_ratio
        logger.info("")
        logger.info("[2] 포트폴리오 리밸런싱" if action == 'rebalance' else "[2] 포트폴리오 상태")
        logger.info(f"운용 기준금액: {capital:,.0f}원 (보유평가 {managed_value:,.0f}원 + 예수금×{invest_ratio:.0%})")

        target_qty = {}
        for code in codes:
            weight = signal['asset_weights'].get(code, 0.0)
            price = prices.get(code, 0)
            target_amount = capital * weight
            target_qty[code] = int(target_amount / price) if price > 0 else 0

        sell_results = []
        buy_results = []
        cash_after_sells = cash

        if action == 'rebalance':
            for code in codes:
                current_qty = holdings.get(code, {}).get('qty', 0)
                diff_qty = target_qty[code] - current_qty
                price = prices.get(code, 0)
                amount = abs(diff_qty) * price
                if diff_qty >= 0 or price <= 0 or amount < min_order_amount:
                    continue
                sell_qty = abs(diff_qty)
                logger.info(f"  매도 대상: {code} ({TARGET_ETFS.get(code, '')}) "
                            f"| 현재 {current_qty}주 → 목표 {target_qty[code]}주")
                result = execute_sell_qty(client, code, sell_qty, price=price, dry_run=dry_run,
                                          limit_band_pct=monitored_band)
                sell_results.append(result)
                if result.get('success') and monitored_band <= 0:
                    cash_after_sells += result.get('amount', 0) * (1 - trading_fee)

            for code in codes:
                current_qty = holdings.get(code, {}).get('qty', 0)
                diff_qty = target_qty[code] - current_qty
                price = prices.get(code, 0)
                amount = diff_qty * price
                if diff_qty <= 0 or price <= 0 or amount < min_order_amount:
                    continue
                affordable_qty = int(cash_after_sells / (price * (1 + trading_fee)))
                buy_qty = min(diff_qty, affordable_qty)
                if buy_qty <= 0:
                    logger.info(f"  매수 스킵: {code} 예수금 부족")
                    continue

                result = execute_buy_qty(client, code, buy_qty, price=price, dry_run=dry_run,
                                         limit_band_pct=monitored_band)
                buy_results.append(result)
                if result.get('success'):
                    cash_after_sells -= result.get('amount', 0) * (1 + trading_fee)

            logger.info(f"리밸런싱 완료: 매도 {len([r for r in sell_results if r.get('success')])}건, "
                        f"매수 {len([r for r in buy_results if r.get('success')])}건")
            if monitored_band > 0:
                monitor_limit_orders(client, sell_results + buy_results, monitored_band,
                                     monitor_interval, monitor_end,
                                     max_order_age_minutes * 60, max_requotes)

        logger.info("")
        logger.info(f"{'='*60}")
        logger.info("  포트폴리오 목표 수량")
        logger.info(f"{'='*60}")
        logger.info(f"  {'코드':>8} {'종목명':<20} {'목표비중':>10} {'목표수량':>10} {'현재수량':>10}")
        logger.info(f"  {'-'*66}")
        for item in signal['ranking'][:15]:
            code = item['code']
            logger.info(f"  {code:>8} {item['name']:<20} {item['target_weight']:>9.2%} "
                        f"{target_qty.get(code, 0):>9,} {holdings.get(code, {}).get('qty', 0):>9,}")

        logger.info("")
        logger.info("완료")
        return

    # 매매 내역 수집용
    sell_results = []
    buy_results = []

    # ── 현재 보유 종목 조회 ──
    holdings = {}
    if client:
        holdings = get_current_holdings(client)
        held_codes = set(holdings.keys())
        logger.info(f"현재 보유: {len(held_codes)}종목 — {', '.join(held_codes) if held_codes else '없음'}")
    else:
        held_codes = set()

    # ── 시그널 생성 ──
    predictions = []
    if action in ('signal', 'buy', 'rebalance'):
        logger.info("")
        logger.info("[1] 매매 시그널 생성")
        predictions = generate_signals(
            model_path, scaler_path, include_meta, device,
            kiwoom_client=client,
            api_key=api_key,
            feature_version=feature_version,
        )

        if not predictions:
            logger.error("시그널 생성 실패 — 종료")
            return

        # 결과 출력 (결정 사유 포함)
        logger.info("")
        logger.info(f"{'='*100}")
        logger.info(f"{'순위':>4} {'코드':>8} {'종목명':<20} {'예측수익률':>10} {'날짜':>10} {'보유':>4}  {'결정사유'}")
        logger.info(f"{'='*100}")

        # 상위 max_select 종목 코드 (매수 유지 대상)
        top_codes = set(p['code'] for p in predictions[:max_select]
                        if p['predicted_return'] > hold_threshold)
        selected_buy_codes = set()
        open_slots = max_select - len(held_codes)
        for p in predictions:
            if p['predicted_return'] > hold_threshold and p['code'] not in held_codes:
                if len(selected_buy_codes) < open_slots:
                    selected_buy_codes.add(p['code'])

        for i, p in enumerate(predictions):
            signal = '★' if p['predicted_return'] > hold_threshold else ' '
            is_held = p['code'] in held_codes
            held = '●' if is_held else ' '
            pred = p['predicted_return']

            # 결정 사유
            if pred <= hold_threshold:
                reason = f'관망 — 예측수익률({pred:+.4f}) ≤ 기준({hold_threshold})'
            elif is_held and p['code'] in top_codes:
                reason = '보유유지 — 상위 시그널 유지 중'
            elif is_held and p['code'] not in top_codes:
                reason = f'매도예정 — 상위 {max_select}위 밖 (순위 {i+1}위)'
            elif p['code'] in selected_buy_codes:
                reason = '매수예정 — 빈 슬롯 진입'
            elif open_slots <= 0 or len(selected_buy_codes) >= open_slots:
                reason = f'관망 — 슬롯 부족 ({max_select}개 초과)'
            else:
                reason = '관망'

            logger.info(f"{i+1:>3}{signal} {p['code']:>8} {p['name']:<20} "
                        f"{pred:>+10.4f} {p['date']:>10} {held:>4}  {reason}")

        # 시그널 저장
        save_signals(predictions, signal_path)

    # ── 매도: 시그널 하위 보유종목 정리 ──
    if action in ('sell', 'rebalance'):
        logger.info("")
        logger.info("[2] 매도 — 시그널 하위 보유종목 정리")

        if not holdings:
            logger.info("보유 종목 없음 — 매도 스킵")
        else:
            # 시그널 기반 매도 대상 결정
            if predictions:
                pred_map = {p['code']: p['predicted_return'] for p in predictions}
                # 상위 max_select 종목 코드
                top_codes = set(p['code'] for p in predictions[:max_select]
                                if p['predicted_return'] > hold_threshold)

                # 매도 대상: 보유 중이나 상위 N에 없는 종목
                sell_codes = [c for c in holdings if c not in top_codes]

                if sell_codes:
                    for code in sell_codes:
                        pred_val = pred_map.get(code, 0)
                        # 매도 사유 구체화
                        rank = next((i+1 for i, p in enumerate(predictions) if p['code'] == code), '?')
                        if pred_val <= hold_threshold:
                            sell_reason = f'예측수익률({pred_val:+.4f}) ≤ 기준({hold_threshold})'
                        else:
                            sell_reason = f'순위 {rank}위 — 상위 {max_select}위 밖'
                        logger.info(f"  매도 대상: {code} ({holdings[code].get('name', '')}) "
                                    f"— {sell_reason}")
                    results = execute_sell_all(client, target_codes=sell_codes, dry_run=dry_run,
                                               limit_band_pct=monitored_band)
                    sell_results.extend(results)
                    sold_count = len([r for r in results if r['success']])
                    total_amount = sum(r.get('amount', 0) for r in results if r['success'])
                    logger.info(f"매도 완료: {sold_count}건, 총 {total_amount:,.0f}원")
                    # 매도된 종목 제거 (매수 슬롯 확보)
                    for code in sell_codes:
                        held_codes.discard(code)
                else:
                    logger.info("모든 보유종목이 상위 시그널 — 매도 불필요")
            else:
                # 시그널 없이 sell만 실행: 전량 매도
                logger.info("시그널 없음 — 전량 매도")
                target_codes = list(TARGET_ETFS.keys())
                results = execute_sell_all(client, target_codes=target_codes, dry_run=dry_run,
                                           limit_band_pct=monitored_band)
                sell_results.extend(results)
                if results:
                    total_amount = sum(r.get('amount', 0) for r in results if r['success'])
                    logger.info(f"매도 완료: {len([r for r in results if r['success']])}건, "
                                f"총 {total_amount:,.0f}원")

    # ── 매수: 빈 슬롯에 신규 종목 진입 ──
    if action in ('buy', 'rebalance'):
        logger.info("")
        logger.info("[3] 매수 — 빈 슬롯에 상위 시그널 종목 진입")

        if not predictions:
            logger.info("시그널 없음 — 매수 스킵")
        else:
            # 매수 후보: threshold 초과 & 미보유
            buy_candidates = [p for p in predictions
                              if p['predicted_return'] > hold_threshold
                              and p['code'] not in held_codes]

            # 빈 슬롯 수
            open_slots = max_select - len(held_codes)
            logger.info(f"현재 보유 {len(held_codes)}종목, 빈 슬롯 {open_slots}개, "
                        f"매수 후보 {len(buy_candidates)}개")

            if open_slots <= 0:
                logger.info(f"보유 슬롯 가득 참 ({max_select}/{max_select}) — 매수 스킵")
                # 관망 사유 표시: threshold 초과했지만 매수 못하는 종목
                watch_list = [p for p in predictions
                              if p['predicted_return'] > hold_threshold
                              and p['code'] not in held_codes]
                if watch_list:
                    logger.info(f"매수 신호 관망 종목 ({len(watch_list)}개):")
                    for p in watch_list[:5]:
                        rank = next((i+1 for i, pp in enumerate(predictions) if pp['code'] == p['code']), '?')
                        logger.info(f"  {p['code']} ({p['name']}): "
                                    f"예측={p['predicted_return']:+.4f}, 순위 {rank}위 "
                                    f"— 슬롯 부족으로 관망")
            elif not buy_candidates:
                logger.info("매수 후보 없음 — 스킵")
                # 관망 사유 표시: threshold 초과 종목이 이미 보유 중인 경우
                above_threshold = [p for p in predictions if p['predicted_return'] > hold_threshold]
                held_above = [p for p in above_threshold if p['code'] in held_codes]
                if held_above:
                    logger.info("매수 신호 상위 종목 중 이미 보유 중:")
                    for p in held_above:
                        rank = next((i+1 for i, pp in enumerate(predictions) if pp['code'] == p['code']), '?')
                        logger.info(f"  {p['code']} ({p['name']}): "
                                    f"예측={p['predicted_return']:+.4f}, 순위 {rank}위 "
                                    f"— 이미 보유 중이므로 유지")
            else:
                selected = buy_candidates[:open_slots]
                # 매수 사유 표시
                for s in selected:
                    rank = next((i+1 for i, p in enumerate(predictions) if p['code'] == s['code']), '?')
                    logger.info(f"  매수 대상: {s['code']} ({s['name']}) "
                                f"— 예측수익률={s['predicted_return']:+.4f}, 순위 {rank}위")
                # 관망 종목 (매수 신호 있지만 슬롯 부족으로 탈락)
                skipped = buy_candidates[open_slots:]
                if skipped:
                    logger.info(f"슬롯 부족으로 관망 ({len(skipped)}개):")
                    for s in skipped[:5]:
                        rank = next((i+1 for i, p in enumerate(predictions) if p['code'] == s['code']), '?')
                        logger.info(f"  {s['code']} ({s['name']}): "
                                    f"예측={s['predicted_return']:+.4f}, 순위 {rank}위 "
                                    f"— 슬롯 부족으로 관망")

                cash = get_available_cash(client)
                per_etf_cash = cash / len(selected)

                for s in selected:
                    result = execute_buy(
                        client, s['code'], per_etf_cash,
                        invest_ratio=invest_ratio,
                        trading_fee=trading_fee,
                        max_invest=max_invest,
                        dry_run=dry_run,
                        limit_band_pct=monitored_band,
                    )
                    buy_results.append(result)
                    if result.get('success'):
                        held_codes.add(s['code'])

    if monitored_band > 0 and action in ('buy', 'sell', 'rebalance'):
        monitor_limit_orders(client, sell_results + buy_results, monitored_band,
                             monitor_interval, monitor_end,
                             max_order_age_minutes * 60, max_requotes)

    # ── 상태 조회 ──
    if action == 'status':
        logger.info("")
        logger.info("계좌 상태 조회")

        # 오늘 시그널 있으면 표시
        if os.path.exists(signal_path):
            signals = load_signals(signal_path)
            buy_signals = [s for s in signals if s['predicted_return'] > hold_threshold]
            if buy_signals:
                logger.info("")
                logger.info(f"오늘 매수 신호 상위 {min(5, len(buy_signals))}개:")
                for s in buy_signals[:5]:
                    held = '(보유중)' if s['code'] in held_codes else ''
                    logger.info(f"  {s['code']} ({s['name']}): "
                                f"predicted={s['predicted_return']:+.4f} {held}")

    # ── 매매내역 & 보유내역 요약 ──
    if action in ('buy', 'sell', 'rebalance'):
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info("  매매내역 요약")
        logger.info(f"{'='*60}")

        if sell_results:
            logger.info("")
            logger.info(f"  [매도] {len([r for r in sell_results if r['success']])}건 성공 / "
                        f"{len([r for r in sell_results if not r['success']])}건 실패")
            for r in sell_results:
                name = TARGET_ETFS.get(r.get('code', ''), '')
                if r.get('success'):
                    dr = ' (DRY RUN)' if r.get('dry_run') else ''
                    logger.info(f"    매도{dr}: {r['code']} ({name}) | "
                                f"{r['qty']}주 × {r['price']:,.0f}원 = {r['amount']:,.0f}원")
                else:
                    logger.info(f"    매도 실패: {r.get('code', '?')} ({name}) | 사유: {r.get('reason', '알 수 없음')}")
        else:
            logger.info("")
            logger.info("  [매도] 없음")

        if buy_results:
            logger.info("")
            logger.info(f"  [매수] {len([r for r in buy_results if r['success']])}건 성공 / "
                        f"{len([r for r in buy_results if not r['success']])}건 실패")
            for r in buy_results:
                name = TARGET_ETFS.get(r.get('code', ''), '')
                if r.get('success'):
                    dr = ' (DRY RUN)' if r.get('dry_run') else ''
                    logger.info(f"    매수{dr}: {r['code']} ({name}) | "
                                f"{r['qty']}주 × {r['price']:,.0f}원 = {r['amount']:,.0f}원")
                else:
                    logger.info(f"    매수 실패: {r.get('code', '?')} ({name}) | 사유: {r.get('reason', '알 수 없음')}")
        else:
            logger.info("")
            logger.info("  [매수] 없음")

    # ── 최종 보유내역 ──
    if action in ('buy', 'sell', 'rebalance', 'status'):
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info("  최종 보유내역")
        logger.info(f"{'='*60}")

        final_holdings = get_current_holdings(client) if client else {}
        cash = get_available_cash(client) if client else 0

        if final_holdings:
            total_value = 0
            total_invested = 0
            logger.info("")
            logger.info(f"  {'코드':>8} {'종목명':<20} {'수량':>6} {'현재가':>12} "
                        f"{'평가금액':>14} {'평균단가':>12} {'수익률':>8}")
            logger.info(f"  {'-'*82}")
            for code, info in final_holdings.items():
                name = info.get('name', TARGET_ETFS.get(code, ''))
                value = info['qty'] * info['cur_price']
                total_value += value
                total_invested += info['qty'] * info['avg_price']
                pnl = (info['cur_price'] - info['avg_price']) / info['avg_price'] * 100 if info['avg_price'] > 0 else 0
                logger.info(f"  {code:>8} {name:<20} {info['qty']:>5}주 "
                            f"{info['cur_price']:>11,.0f}원 {value:>13,.0f}원 "
                            f"{info['avg_price']:>11,.0f}원 {pnl:>+7.2f}%")

            total_pnl = total_value - total_invested
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            logger.info(f"  {'-'*82}")
            logger.info(f"  보유 평가액  : {total_value:>14,.0f}원 (손익: {total_pnl:>+,.0f}원, {total_pnl_pct:>+.2f}%)")
            logger.info(f"  예수금       : {cash:>14,.0f}원")
            logger.info(f"  총 자산      : {total_value + cash:>14,.0f}원")
            logger.info(f"  슬롯         : {len(final_holdings)}/{max_select} 사용 중")
        else:
            logger.info("")
            logger.info("  보유 종목 없음")
            logger.info(f"  예수금       : {cash:>14,.0f}원")
            logger.info(f"  슬롯         : 0/{max_select} 사용 중")

    logger.info("")
    logger.info("완료")


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ETF Swing Trading 모의투자')

    # 액션
    parser.add_argument('--action', type=str, default='signal',
                        choices=['signal', 'buy', 'sell', 'rebalance', 'status'],
                        help='실행할 동작 (rebalance=매도+매수)')

    # 모델 경로
    parser.add_argument('--model', type=str, default=None,
                        help='모델 파일 경로 (model_best.pt)')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='모델 디렉토리 (output/ 등)')

    # 데이터
    parser.add_argument('--dataset', type=str, default=None,
                        help='데이터셋 이름 (스케일러 결정, e.g. etf_20260323)')

    # 거래 파라미터
    parser.add_argument('--hold-threshold', type=float, default=0.001,
                        help='매수 기준 예측 수익률')
    parser.add_argument('--invest-ratio', type=float, default=0.95,
                        help='가용 현금 대비 투자 비율')
    parser.add_argument('--trading-fee', type=float, default=0.00015,
                        help='거래 수수료율')
    parser.add_argument('--max-select', type=int, default=5,
                        help='최대 동시 보유 종목 수')
    parser.add_argument('--max-invest', type=float, default=30_000_000,
                        help='종목당 최대 투자 금액 (기본: 3천만원, 0이면 제한 없음)')
    parser.add_argument('--min-order-amount', type=float, default=10_000,
                        help='포트폴리오 리밸런싱 시 최소 주문 금액')
    parser.add_argument('--price-band-pct', type=float, default=0.005,
                        help='모의투자 지정가 허용 범위 (최초 현재가 대비, 기본: 0.005=±0.5%%)')
    parser.add_argument('--monitor-interval', type=int, default=60,
                        help='미체결 확인 및 재호가 주기(초)')
    parser.add_argument('--monitor-end', type=str, default='15:20',
                        help='장중 주문 모니터링 종료 시각(HH:MM)')
    parser.add_argument('--max-order-minutes', type=int, default=30,
                        help='개별 지정가 주문 최대 유효 시간(분, 0이면 장 마감까지)')
    parser.add_argument('--max-requotes', type=int, default=20,
                        help='주문별 최대 재호가 횟수(0이면 제한 없음)')
    parser.add_argument('--no-monitor-orders', action='store_true', default=False,
                        help='모의투자 장중 지정가 모니터링 비활성화(기존 시장가 주문)')

    # 실행 모드
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='주문 실행 안 함 (시뮬레이션)')
    parser.add_argument('--real', action='store_true', default=False,
                        help='실계좌 사용 (기본: 모의)')

    # 기타
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--no-meta', action='store_true', default=False,
                        help='메타 피처 미포함')
    parser.add_argument('--base-path', type=str, default=BASE_PATH)
    parser.add_argument('--api-key', type=str, default=None,
                        help='내부 API 키 (기본: QUANTYLAB_API_KEY 환경변수)')
    parser.add_argument('--feature-version', type=str, default="1",
                        help='피처 벡터 버전 (기본: 1)')

    args = parser.parse_args()
    if args.price_band_pct < 0 or args.price_band_pct > 0.1:
        parser.error('--price-band-pct는 0 이상 0.1 이하이어야 합니다.')
    if args.monitor_interval < 1:
        parser.error('--monitor-interval은 1초 이상이어야 합니다.')
    if args.max_order_minutes < 0:
        parser.error('--max-order-minutes는 0 이상이어야 합니다.')
    if args.max_requotes < 0:
        parser.error('--max-requotes는 0 이상이어야 합니다.')
    try:
        monitor_hour, monitor_minute = map(int, args.monitor_end.split(':'))
        if not (0 <= monitor_hour <= 23 and 0 <= monitor_minute <= 59):
            raise ValueError
    except ValueError:
        parser.error('--monitor-end는 HH:MM 형식이어야 합니다.')

    # 모델 경로 결정
    if args.model:
        model_path = args.model if os.path.isabs(args.model) \
            else os.path.join(args.base_path, args.model)
    elif args.model_dir:
        model_dir = args.model_dir if os.path.isabs(args.model_dir) \
            else os.path.join(args.base_path, args.model_dir)
        model_path = os.path.join(model_dir, 'policy_best.pt')
    else:
        model_path = os.path.join(args.base_path, 'output', 'policy_best.pt')

    model_type = detect_model_type(model_path)

    # API 모드 여부 확인
    _api_key = args.api_key or os.environ.get("QUANTYLAB_API_KEY", "")

    if model_type == 'portfolio':
        scaler_path = None
        logger.info(f"모델: {model_path}")
        logger.info(f"모델 타입: {model_type}")
        logger.info(f"피처 소스: {'quantylab-api' if _api_key else 'feature_vector DB'}, version={args.feature_version}")
    elif _api_key:
        # API 모드: 스케일러 불필요
        scaler_path = None
        logger.info(f"모델: {model_path}")
        logger.info(f"피처 소스: quantylab-api, version={args.feature_version}")
    else:
        # DB 직접 빌드 모드: 스케일러 필요
        dataset = args.dataset
        if dataset is None:
            data_dir = os.path.join(args.base_path, 'data')
            if os.path.exists(data_dir):
                datasets = sorted([d for d in os.listdir(data_dir)
                                   if d.startswith('etf_') and os.path.isdir(os.path.join(data_dir, d))])
                if datasets:
                    dataset = datasets[-1]

        if dataset is None:
            logger.error("데이터셋을 찾을 수 없습니다. --dataset 옵션으로 지정하거나 API 환경변수를 설정하세요.")
            sys.exit(1)

        scaler_path = os.path.join(args.base_path, 'scalers', dataset, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            logger.error(f"스케일러 파일 없음: {scaler_path}")
            sys.exit(1)

        logger.info(f"모델: {model_path}")
        logger.info(f"데이터셋: {dataset}")
        logger.info(f"스케일러: {scaler_path}")

    run_swing_trading(
        action=args.action,
        model_path=model_path,
        scaler_path=scaler_path,
        hold_threshold=args.hold_threshold,
        invest_ratio=args.invest_ratio,
        trading_fee=args.trading_fee,
        include_meta=not args.no_meta,
        device=args.device,
        dry_run=args.dry_run,
        real=args.real,
        max_select=args.max_select,
        max_invest=args.max_invest,
        api_key=_api_key or None,
        feature_version=args.feature_version,
        min_order_amount=args.min_order_amount,
        monitor_orders=not args.no_monitor_orders,
        price_band_pct=args.price_band_pct,
        monitor_interval=args.monitor_interval,
        monitor_end=args.monitor_end,
        max_order_age_minutes=args.max_order_minutes,
        max_requotes=args.max_requotes,
    )


if __name__ == '__main__':
    main()
