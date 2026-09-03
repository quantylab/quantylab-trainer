"""
ETF 예측/랭킹 모듈

학습된 모델을 이용하여 ETF별 투자 점수를 산출하고
랭킹 기반 투자 대상 선정 및 투자 판단을 수행한다.

모델 타입:
  - regression (Day Trading 지도학습): 일중 수익률 직접 예측
  - rl (Swing Trading 강화학습): Beta 분포 position_score + 가치 추정
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

from .network import (
    MambaPolicyNetwork, MambaValueNetwork,
    MambaRegressionNetwork,
    GRNPolicyNetwork, GRNValueNetwork,
    ContinuousPolicyNetwork, ValueNetwork,
)
from ..target_etfs import TARGET_ETFS
from ..feature import (
    load_all_data, load_tiger_etf_list,
    build_single_etf_features,
    clean_and_clip_outliers,
    select_features, get_selected_features,
    remove_zero_variance,
    DataQualityReport,
)

# 포트폴리오 feature 기본값 (거래 이력 없는 초기 상태)
DEFAULT_PORTFOLIO_FEATURES = np.array([
    0.0,  # cumulative_return_scaled
    0.0,  # drawdown_scaled
    0.0,  # win_rate_scaled (50% → (0.5-0.5)*4 = 0)
    0.0,  # streak_scaled
    1.0,  # vol_scaled (기본 변동성)
], dtype=np.float32)

PORTFOLIO_FEATURE_NUM = 5


def _detect_network_type(state_dict_keys):
    """state_dict 키로 실제 네트워크 타입 감지 (메타데이터 불일치 대비)"""
    keys = set(state_dict_keys)
    if any('ssm_gate' in k for k in keys):
        return 'mamba'
    if any('grn' in k.lower() or 'gating' in k.lower() for k in keys):
        return 'grn'
    if any('res_blocks' in k for k in keys):
        return 'standard'
    return 'mamba'


def load_model(model_dir: str, device: str = 'cpu'):
    """학습된 모델 로드 (regression 또는 RL 자동 감지)

    Returns:
        dict with keys:
          - 'type': 'regression' or 'rl'
          - 'model': regression network (if regression)
          - 'policy_net', 'value_net': RL networks (if rl)
          - 'input_dim': input dimension
    """
    # Regression 모델 우선 탐지
    regression_path = os.path.join(model_dir, 'model_best.pt')
    if os.path.exists(regression_path):
        ckpt = torch.load(regression_path, map_location=device, weights_only=False)
        input_dim = ckpt.get('input_dim', 121)
        model = MambaRegressionNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return {
            'type': 'regression',
            'model': model,
            'input_dim': input_dim,
        }

    # RL 모델 (기존)
    policy_path = os.path.join(model_dir, 'policy_best.pt')
    value_path = os.path.join(model_dir, 'value_best.pt')

    policy_ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    value_ckpt = torch.load(value_path, map_location=device, weights_only=False)

    input_dim = policy_ckpt.get('input_dim', 126)
    network_type = _detect_network_type(policy_ckpt['state_dict'].keys())

    if network_type == 'mamba':
        policy_net = MambaPolicyNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
        value_net = MambaValueNetwork(input_dim, d_model=64, d_state=16, n_blocks=2)
    elif network_type == 'grn':
        policy_net = GRNPolicyNetwork(input_dim, hidden_dim=128, num_blocks=2)
        value_net = GRNValueNetwork(input_dim, hidden_dim=128, num_blocks=2)
    else:
        policy_net = ContinuousPolicyNetwork(input_dim, hidden_dim=256, num_blocks=3)
        value_net = ValueNetwork(input_dim, hidden_dim=256, num_blocks=3)

    policy_net.load_state_dict(policy_ckpt['state_dict'])
    value_net.load_state_dict(value_ckpt['state_dict'])
    policy_net.eval()
    value_net.eval()

    return {
        'type': 'rl',
        'policy_net': policy_net,
        'value_net': value_net,
        'input_dim': input_dim,
    }


def load_dataset(data_dir: str):
    """데이터셋 로드 (environment.csv, training_scaled.csv, etf_codes.csv)"""
    env_data = pd.read_csv(os.path.join(data_dir, 'environment.csv'),
                           dtype={'etf_code': str})
    training_data = pd.read_csv(os.path.join(data_dir, 'training_scaled.csv')).values
    etf_codes = pd.read_csv(os.path.join(data_dir, 'etf_codes.csv'),
                            dtype={'etf_code': str})['etf_code'].values
    return env_data, training_data, etf_codes


def score_etfs(model_info: dict, training_data: np.ndarray,
               env_data: pd.DataFrame, etf_codes: np.ndarray,
               lookback: int = 1, device: str = 'cpu'):
    """
    ETF별 점수 산출 (regression / rl 자동 분기)

    Returns:
        DataFrame with columns: etf_code, date, close, predicted_return,
        position_score, composite_score, ...
    """
    if model_info['type'] == 'regression':
        return _score_etfs_regression(
            model_info['model'], training_data, env_data, etf_codes,
            lookback, device,
        )
    else:
        return _score_etfs_rl(
            model_info['policy_net'], model_info['value_net'],
            training_data, env_data, etf_codes, lookback, device,
            model_input_dim=model_info.get('input_dim'),
        )


def _score_etfs_regression(model, training_data, env_data, etf_codes,
                           lookback, device):
    """Regression 모델로 ETF 점수 산출 (일중 수익률 예측)"""
    unique_codes = np.unique(etf_codes)
    results = []

    for code in unique_codes:
        mask = etf_codes == code
        indices = np.where(mask)[0]

        if len(indices) < lookback:
            continue

        target_indices = indices[-lookback:]
        features = training_data[target_indices]
        features_tensor = torch.FloatTensor(features).to(device)

        with torch.no_grad():
            predicted_returns = model(features_tensor).cpu().numpy()

        last_idx = indices[-1]
        last_env = env_data.iloc[last_idx]
        mean_pred = float(predicted_returns.mean())

        results.append({
            'etf_code': str(code).zfill(6),
            'etf_name': TARGET_ETFS.get(str(code).zfill(6), ''),
            'date': int(last_env['date']),
            'close': float(last_env['close']),
            'predicted_return': mean_pred,
            'position_score': max(0, mean_pred * 100),  # 예측 수익률 → 0~1 스케일
            'confidence': abs(mean_pred) * 1000,
            'value_score': mean_pred * 100,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    # composite_score = predicted_return 기반 랭킹
    df['composite_score'] = df['predicted_return']
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = 'rank'

    return df


def _score_etfs_rl(policy_net, value_net, training_data, env_data,
                   etf_codes, lookback, device, model_input_dim=None):
    """RL 모델로 ETF 점수 산출 (Beta 분포 기반)"""
    unique_codes = np.unique(etf_codes)
    results = []

    for code in unique_codes:
        mask = etf_codes == code
        indices = np.where(mask)[0]

        if len(indices) < lookback:
            continue

        target_indices = indices[-lookback:]
        states = []
        for idx in target_indices:
            features = training_data[idx]
            state = np.concatenate([features, DEFAULT_PORTFOLIO_FEATURES])
            # 모델 입력 차원에 맞춰 패딩 또는 트리밍
            if model_input_dim is not None:
                if len(state) < model_input_dim:
                    state = np.concatenate([state, np.zeros(model_input_dim - len(state))])
                elif len(state) > model_input_dim:
                    state = state[:model_input_dim]
            states.append(state)

        states_tensor = torch.FloatTensor(np.array(states)).to(device)

        with torch.no_grad():
            alpha, beta = policy_net(states_tensor)
            values = value_net(states_tensor)

        mean_position = (alpha / (alpha + beta)).cpu().numpy()
        concentration = (alpha + beta).cpu().numpy()
        value_scores = values.squeeze(-1).cpu().numpy()

        last_idx = indices[-1]
        last_env = env_data.iloc[last_idx]

        results.append({
            'etf_code': str(code).zfill(6),
            'etf_name': TARGET_ETFS.get(str(code).zfill(6), ''),
            'date': int(last_env['date']),
            'close': float(last_env['close']),
            'position_score': float(mean_position.mean()),
            'confidence': float(concentration.mean()),
            'value_score': float(value_scores.mean()),
        })

    df = pd.DataFrame(results)

    if len(df) == 0:
        return df

    # composite_score: position 60% + normalized value 40%
    v_min, v_max = df['value_score'].min(), df['value_score'].max()
    v_range = v_max - v_min
    if v_range > 1e-8:
        norm_value = (df['value_score'] - v_min) / v_range
    else:
        norm_value = 0.5

    df['composite_score'] = df['position_score'] * 0.6 + norm_value * 0.4
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    df.index.name = 'rank'

    return df


def build_live_data(start_date: str, end_date: str, scaler_path: str,
                    min_candles: int = 60, include_meta: bool = True):
    """
    DB에서 직접 데이터를 로드하여 피처를 생성하고 기존 스케일러로 변환

    Args:
        start_date: 시작일 (예: '20260101')
        end_date: 종료일 (예: '20260327')
        scaler_path: 학습 시 저장된 scaler.pkl 경로
        min_candles: ETF별 최소 캔들 수
        include_meta: 메타 피처 포함 여부

    Returns:
        (env_data, training_data, etf_codes) — load_dataset()과 동일한 형식
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 없음: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # 기술적 지표 warm-up을 위해 시작일보다 충분히 앞선 데이터 로드
    # 120일 이동평균 등에 필요한 최소 기간
    from datetime import datetime, timedelta
    dt = datetime.strptime(start_date, '%Y%m%d')
    warmup_start = (dt - timedelta(days=365)).strftime('%Y%m%d')

    # ETF 목록
    etf_list = load_tiger_etf_list(min_candles=min_candles)
    etf_codes_list = [e['code'] for e in etf_list]
    etf_info = {e['code']: e for e in etf_list}

    print(f"\n[데이터 빌드] {start_date} ~ {end_date}")
    print(f"  warm-up 시작: {warmup_start}")
    print(f"  ETF 후보: {len(etf_codes_list)}개")

    qc = DataQualityReport()
    data = load_all_data(etf_codes_list, warmup_start, end_date, qc=qc)

    all_env = []
    all_features = []
    all_etf_codes = []
    feature_cols = None

    for i, etf_code in enumerate(etf_codes_list):
        info = etf_info[etf_code]
        try:
            feature_df = build_single_etf_features(
                etf_code, data,
                etf_name=info['name'], etf_extra=info['extra'],
                include_meta=include_meta, verbose=False, qc=qc,
            )
        except Exception as e:
            print(f"  [SKIP] {etf_code} ({info['name']}): {e}")
            continue

        # 환경 데이터
        env_cols = ['date', 'etf_open', 'etf_high', 'etf_low', 'etf_close', 'etf_volume']
        available_env = [c for c in env_cols if c in feature_df.columns]
        env_df = feature_df[available_env].copy()
        env_df.columns = [c.replace('etf_', '') for c in env_df.columns]

        # 정제 (warm-up 제거)
        feature_df = clean_and_clip_outliers(feature_df, warm_up_period=60, verbose=False)
        env_df = env_df.iloc[-len(feature_df):].reset_index(drop=True)
        feature_df = feature_df.reset_index(drop=True)

        # start_date 이후만 필터링
        start_int = int(start_date)
        date_mask = env_df['date'].astype(int) >= start_int
        if end_date:
            date_mask &= env_df['date'].astype(int) <= int(end_date)

        env_df = env_df[date_mask].reset_index(drop=True)
        feature_df = feature_df[date_mask].reset_index(drop=True)

        if len(env_df) < 1:
            continue

        # 피처 선택
        selected = select_features(feature_df,
                                   get_selected_features(include_meta=include_meta))

        if feature_cols is None:
            feature_cols = list(selected.columns)
        else:
            for c in feature_cols:
                if c not in selected.columns:
                    selected[c] = 0.0
            selected = selected[feature_cols]

        env_df['etf_code'] = etf_code
        all_env.append(env_df)
        all_features.append(selected)
        all_etf_codes.extend([etf_code] * len(env_df))

    if not all_features:
        raise ValueError("유효한 ETF 데이터가 없습니다.")

    unified_env = pd.concat(all_env, ignore_index=True)
    unified_features = pd.concat(all_features, ignore_index=True)
    unified_features = remove_zero_variance(unified_features, verbose=False)

    # 기존 스케일러로 변환 (fit 하지 않음!)
    meta_cols = [c for c in unified_features.columns if c.startswith('meta_sector_')]
    non_meta_cols = [c for c in unified_features.columns if c not in meta_cols]

    # 스케일러 피처와 현재 피처 매칭
    scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else non_meta_cols
    # 스케일러에 있는 피처만 사용, 없으면 0으로 채움
    for c in scaler_features:
        if c not in unified_features.columns:
            unified_features[c] = 0.0
    non_meta_for_scale = [c for c in scaler_features if c not in meta_cols]

    scaled_values = scaler.transform(unified_features[non_meta_for_scale])
    scaled_values = np.clip(scaled_values, -5.0, 5.0)
    scaled_df = pd.DataFrame(scaled_values, columns=non_meta_for_scale)
    for c in meta_cols:
        if c in unified_features.columns:
            scaled_df[c] = unified_features[c].values

    etf_codes_arr = np.array(all_etf_codes)
    training_data = scaled_df.values

    n_etfs = len(set(all_etf_codes))
    print(f"  ETF: {n_etfs}개, 데이터: {len(unified_env):,}행, 피처: {scaled_df.shape[1]}개")

    return unified_env, training_data, etf_codes_arr


def select_investments(scores_df: pd.DataFrame,
                       top_n: int = 10,
                       min_position_score: float = 0.5,
                       min_composite_score: float = 0.4):
    """
    투자 대상 ETF 선정

    Args:
        top_n: 최대 투자 종목 수
        min_position_score: 최소 position_score 기준
        min_composite_score: 최소 composite_score 기준

    Returns:
        (selected_df, summary_dict)
    """
    selected = scores_df[
        (scores_df['position_score'] >= min_position_score) &
        (scores_df['composite_score'] >= min_composite_score)
    ].head(top_n).copy()

    # 투자 비중 (position_score 비례 배분)
    if len(selected) > 0:
        total_score = selected['position_score'].sum()
        selected['weight'] = selected['position_score'] / total_score
    else:
        selected['weight'] = []

    summary = {
        'total_etfs': len(scores_df),
        'selected_etfs': len(selected),
        'avg_position_score': float(scores_df['position_score'].mean()),
        'avg_value_score': float(scores_df['value_score'].mean()),
        'market_signal': 'bullish' if scores_df['position_score'].mean() > 0.55
                         else 'bearish' if scores_df['position_score'].mean() < 0.45
                         else 'neutral',
    }

    return selected, summary


def main():
    parser = argparse.ArgumentParser(description='ETF 예측/랭킹')
    parser.add_argument('--base-path', type=str,
                        default='/home/quantylab/quantylab-trainer')
    parser.add_argument('--model', type=str, default='etf-day-v1',
                        help='모델명 (models/ 하위 디렉토리)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='데이터셋명 (data/ 하위 디렉토리). --start-date 미지정 시 사용')
    parser.add_argument('--start-date', type=str, default=None,
                        help='시작일 (예: 20260101). DB에서 직접 데이터 생성')
    parser.add_argument('--end-date', type=str, default=None,
                        help='종료일 (예: 20260327). 미지정 시 최신')
    parser.add_argument('--scaler', type=str, default=None,
                        help='스케일러 데이터셋명 (scalers/ 하위). 미지정 시 --dataset과 동일')
    parser.add_argument('--lookback', type=int, default=1,
                        help='최근 N일 평균 (1=마지막 날만)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='최대 투자 종목 수')
    parser.add_argument('--min-position', type=float, default=0.5,
                        help='최소 position_score')
    parser.add_argument('--min-composite', type=float, default=0.4,
                        help='최소 composite_score')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default=None,
                        help='결과 저장 경로 (CSV)')
    args = parser.parse_args()

    model_dir = os.path.join(args.base_path, 'models', args.model)

    # 데이터 모드 결정: --start-date 면 DB 직접 빌드, 아니면 기존 dataset 로드
    use_live = args.start_date is not None
    if not use_live and args.dataset is None:
        parser.error('--dataset 또는 --start-date 중 하나는 필수')

    print("=" * 70)
    print("  ETF 예측/랭킹")
    print("=" * 70)
    print(f"  모델: {model_dir}")

    # 모델 로드
    model_info = load_model(model_dir, args.device)
    print(f"  모델 타입: {model_info['type']}")
    print(f"  입력 차원: {model_info['input_dim']}")

    if use_live:
        # DB에서 직접 데이터 생성
        scaler_name = args.scaler or args.dataset
        if scaler_name is None:
            # 모델 디렉토리에서 학습에 사용된 데이터셋 추론
            # scalers/ 에서 최신 etf_ 디렉토리 사용
            scaler_dir_base = os.path.join(args.base_path, 'scalers')
            candidates = sorted([d for d in os.listdir(scaler_dir_base)
                                 if d.startswith('etf_')], reverse=True)
            if not candidates:
                parser.error('스케일러를 찾을 수 없습니다. --scaler 지정 필요')
            scaler_name = candidates[0]
            print(f"  스케일러 자동 선택: {scaler_name}")
        scaler_path = os.path.join(args.base_path, 'scalers', scaler_name, 'scaler.pkl')
        print(f"  기간: {args.start_date} ~ {args.end_date or '최신'}")
        print(f"  스케일러: {scaler_path}")
        env_data, training_data, etf_codes = build_live_data(
            args.start_date, args.end_date, scaler_path)
    else:
        data_dir = os.path.join(args.base_path, 'data', args.dataset)
        print(f"  데이터: {data_dir}")
        env_data, training_data, etf_codes = load_dataset(data_dir)

    unique_etfs = np.unique(etf_codes)
    print(f"  Lookback: {args.lookback}일")
    print(f"  ETF 수: {len(unique_etfs)}")
    print(f"  데이터 행: {len(env_data):,}")

    # 점수 산출
    scores = score_etfs(
        model_info, training_data,
        env_data, etf_codes, args.lookback, args.device,
    )

    # 투자 대상 선정
    selected, summary = select_investments(
        scores, args.top_n, args.min_position, args.min_composite,
    )

    # 전체 랭킹 출력
    is_regression = model_info['type'] == 'regression'
    print(f"\n{'='*70}")
    print(f"  전체 랭킹 (상위 20)")
    print(f"{'='*70}")

    top20 = scores.head(20)
    for _, row in top20.iterrows():
        date_str = str(int(row['date']))
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        if is_regression:
            print(f"  {row.name:>3d}. {row['etf_code']}  "
                  f"{row['etf_name']:<20s}  "
                  f"{date_fmt}  "
                  f"Pred={row['predicted_return']*100:+.3f}%  "
                  f"Close={row['close']:,.0f}")
        else:
            print(f"  {row.name:>3d}. {row['etf_code']}  "
                  f"{row['etf_name']:<20s}  "
                  f"{date_fmt}  "
                  f"Pos={row['position_score']:.3f}  "
                  f"Val={row['value_score']:+.3f}  "
                  f"Comp={row['composite_score']:.3f}  "
                  f"Conf={row['confidence']:.1f}  "
                  f"Close={row['close']:,.0f}")

    # 투자 대상 출력
    print(f"\n{'='*70}")
    print(f"  투자 대상 ({summary['selected_etfs']}/{summary['total_etfs']})")
    print(f"  시장 시그널: {summary['market_signal']}")
    print(f"{'='*70}")
    if len(selected) > 0:
        for _, row in selected.iterrows():
            print(f"  {row['etf_code']}  {row['etf_name']:<20s}  "
                  f"비중={row['weight']:.1%}  "
                  f"Pos={row['position_score']:.3f}  "
                  f"Val={row['value_score']:+.3f}  "
                  f"Comp={row['composite_score']:.3f}")
    else:
        print("  투자 대상 없음 (기준 미달)")

    # CSV 저장
    if args.output:
        out_path = args.output if os.path.isabs(args.output) \
            else os.path.join(args.base_path, args.output)
        scores.to_csv(out_path)
        print(f"\n결과 저장: {out_path}")


if __name__ == '__main__':
    main()
