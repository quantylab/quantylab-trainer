"""
포트폴리오 백테스트 CLI 엔트리포인트

Usage:
  python -m quantylab.trainer.etf_portfolio_swing.backtest --dataset <path> --model <path> [options]
"""
import argparse
import os
import json
import math
import numpy as np
import pandas as pd
import torch

from .environment import PortfolioTradingEnvironment
from .network import PortfolioPolicyNetwork, PortfolioValueNetwork
from .agent import PortfolioAgent
from ..target_etfs import TARGET_ETFS


def compute_win_plr(trade_log: list) -> dict:
    """승률 및 손익비 계산"""
    returns = [t['portfolio_return'] for t in trade_log]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    win_rate = len(wins) / len(returns) if returns else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    plr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    return {'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss, 'plr': plr}


def run_backtest(env: PortfolioTradingEnvironment, agent: PortfolioAgent, deterministic: bool = True) -> dict:
    """에피소드 단위 백테스트 실행

    deterministic=True: softmax(logits) 직접 사용 (Dirichlet 샘플링 없음)
    """
    obs = env.reset()
    done = False
    total_reward = 0.0

    agent.policy_net.eval()
    with torch.no_grad():
        while not done:
            features = obs['features']
            prev_weights = obs['weights']
            if deterministic:
                feat_t = torch.as_tensor(features, dtype=torch.float32, device=agent.device).unsqueeze(0)
                pw_t   = torch.as_tensor(prev_weights, dtype=torch.float32, device=agent.device).unsqueeze(0)
                weights = agent.policy_net(feat_t, pw_t).squeeze(0).cpu().numpy()
            else:
                weights, _ = agent.get_action(features, prev_weights)
            obs, reward, done, info = env.step(weights)
            total_reward += reward

    trade_log = env.get_trade_log()
    metrics = env.get_metrics()
    win_metrics = compute_win_plr(trade_log)
    metrics.update(win_metrics)
    metrics['total_reward'] = float(total_reward)
    return metrics, trade_log


def _metrics_from_pv(values: list, initial_balance: float, daily_returns: list) -> dict:
    """포트폴리오 가치(pv) 시계열과 일별 수익률로 성과 지표를 계산."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {}
    total_return = float(values[-1] / initial_balance - 1.0)
    n_years = len(values) / 252.0
    if n_years > 0 and values[-1] > 0:
        cagr = float((values[-1] / initial_balance) ** (1.0 / n_years) - 1.0)
    else:
        cagr = -1.0
    peak = initial_balance
    mdd = 0.0
    for v in values:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / (peak + 1e-9))
    dr = np.asarray(daily_returns, dtype=np.float64)
    if len(dr) > 1 and dr.std() > 1e-12:
        sharpe = float(dr.mean() / dr.std() * math.sqrt(252))
    else:
        sharpe = 0.0
    wins = dr[dr > 0]
    losses = dr[dr < 0]
    win_rate = float(len(wins) / len(dr)) if len(dr) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    plr = float(abs(avg_win / avg_loss)) if avg_loss != 0 else float('inf')
    return {
        'total_return': total_return, 'cagr': cagr, 'sharpe': sharpe, 'mdd': float(mdd),
        'num_steps': len(values), 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'plr': plr,
    }


def run_backtest_realistic(
    env: PortfolioTradingEnvironment,
    agent: PortfolioAgent,
    min_trading_price: float = 10_000.0,
    rebalance_band: float = 0.0,
    trading_fee: float = 0.00015,
    deterministic: bool = True,
) -> tuple:
    """정수 주(株) + 최소 거래금액 제약을 반영한 현실적 체결 백테스트.

    모델이 출력한 연속 목표 비중을 실제 정수 주 주문으로 양자화한다.
      - 매도를 먼저 실행해 현금을 확보한 뒤, 큰 주문부터 매수
      - 종목별 거래 금액이 min_trading_price 미만이거나 비중 변화가
        rebalance_band 미만이면 거래를 건너뜀(skip)
      - 1주도 못 사는 잔액은 현금으로 보유
    체결·평가는 모두 종가 기준(close-to-close)으로 학습 환경과 정합한다.
    """
    obs = env.reset()
    n = env.n_assets
    holdings = np.zeros(n, dtype=np.int64)
    cash = float(env.initial_balance)
    prev_pv = float(env.initial_balance)
    pv_series: list = []
    daily_returns: list = []
    n_buys = n_sells = n_skipped = 0
    prices = np.zeros(n, dtype=np.float64)

    agent.policy_net.eval()
    done = False
    with torch.no_grad():
        while not done:
            features = obs['features']
            idx = env._step_idx - 1  # 의사결정 시점의 최신 종가 인덱스 (체결 기준)
            prices = np.array(
                [env._close_prices[c][idx] for c in env._asset_codes],
                dtype=np.float64,
            )
            stock_val = holdings * prices
            pv = cash + float(stock_val.sum())
            if pv <= 0:
                break

            # 모델에 현재 '실제' 비중(현금 포함)을 입력 → 실전과 정합
            cur_w = np.concatenate([stock_val / pv, [cash / pv]]).astype(np.float32)
            if deterministic:
                feat_t = torch.as_tensor(features, dtype=torch.float32, device=agent.device).unsqueeze(0)
                pw_t = torch.as_tensor(cur_w, dtype=torch.float32, device=agent.device).unsqueeze(0)
                w = agent.policy_net(feat_t, pw_t).squeeze(0).cpu().numpy()
            else:
                w, _ = agent.get_action(features, cur_w)
            w = np.clip(np.asarray(w, dtype=np.float64), 0, None)
            w = w / w.sum() if w.sum() > 1e-12 else np.ones_like(w) / len(w)
            target_val = pv * w[:-1]

            # ── 매도 먼저 (현금 확보) ──
            for i in range(n):
                if prices[i] <= 0 or holdings[i] <= 0:
                    continue
                delta = holdings[i] * prices[i] - target_val[i]
                if delta <= 0:
                    continue
                if delta < min_trading_price or (delta / pv) < rebalance_band:
                    n_skipped += 1
                    continue
                sell_sh = min(int(holdings[i]), int(delta // prices[i]))
                if sell_sh <= 0:
                    continue
                proceeds = sell_sh * prices[i]
                cash += proceeds - proceeds * trading_fee
                holdings[i] -= sell_sh
                n_sells += 1

            # ── 매수 (현금 한도 내, 큰 주문부터) ──
            buy_needs = []
            for i in range(n):
                if prices[i] <= 0:
                    continue
                delta = target_val[i] - holdings[i] * prices[i]
                if delta <= 0:
                    continue
                if delta < min_trading_price or (delta / pv) < rebalance_band:
                    n_skipped += 1
                    continue
                buy_needs.append((delta, i))
            buy_needs.sort(reverse=True)
            for delta, i in buy_needs:
                affordable = cash / (1.0 + trading_fee)
                buy_sh = int(min(delta, affordable) // prices[i])
                while buy_sh > 0 and buy_sh * prices[i] * (1.0 + trading_fee) > cash:
                    buy_sh -= 1
                if buy_sh <= 0:
                    continue
                cost = buy_sh * prices[i]
                cash -= cost + cost * trading_fee
                holdings[i] += buy_sh
                n_buys += 1

            pv = cash + float((holdings * prices).sum())
            pv_series.append(pv)
            daily_returns.append(pv / prev_pv - 1.0)
            prev_pv = pv

            obs, _, done, _ = env.step(w.astype(np.float32))  # 시계 진행용

    metrics = _metrics_from_pv(pv_series, env.initial_balance, daily_returns)
    final_invested = float((holdings * prices).sum())
    final_pv = cash + final_invested
    metrics.update({
        'final_cash': float(cash),
        'final_invested': final_invested,
        'cash_ratio': float(cash / final_pv) if final_pv > 0 else 0.0,
        'n_buys': int(n_buys),
        'n_sells': int(n_sells),
        'n_skipped_small': int(n_skipped),
        'min_trading_price': float(min_trading_price),
        'rebalance_band': float(rebalance_band),
    })
    return metrics, pv_series



def load_dataset(dataset_dir: str, etf_codes: list, feature_cols: list = None) -> tuple:
    """통합 CSV에서 ETF별 OHLCV + features 분리 로드 (train.py 와 동일).

    feature_cols: 지정 시 해당 컬럼만 추출 (in-sample 피처 정합성 유지용)
    """
    env_path = os.path.join(dataset_dir, 'environment.csv')
    feat_path = os.path.join(dataset_dir, 'training_scaled.csv')
    if not os.path.exists(env_path):
        raise RuntimeError(f"environment.csv not found in {dataset_dir}")
    if not os.path.exists(feat_path):
        raise RuntimeError(f"training_scaled.csv not found in {dataset_dir}")

    env_df = pd.read_csv(env_path, dtype={'etf_code': str})
    env_df['etf_code'] = env_df['etf_code'].str.zfill(6)
    feat_df = pd.read_csv(feat_path)

    # 피처 컬럼 필터링 (모델 학습 시 피처와 정합성 맞추기)
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in feat_df.columns]
        if missing:
            raise RuntimeError(f"Dataset missing feature columns: {missing}")
        feat_df = feat_df[feature_cols]

    target_set = set(str(c).zfill(6) for c in etf_codes)
    available = set(env_df['etf_code'].unique())
    use_codes = sorted(target_set & available)
    if not use_codes:
        raise RuntimeError(f"None of requested codes found in dataset.")

    asset_data = {}
    asset_features = {}
    for code in use_codes:
        mask = env_df['etf_code'] == code
        df_code = env_df[mask].copy()
        feat_code = feat_df.loc[mask].values.astype(np.float32)
        if len(df_code) < 60:
            continue
        asset_data[code] = df_code
        asset_features[code] = feat_code

    print(f"Loaded {len(asset_data)} assets")
    return asset_data, asset_features


def main():
    parser = argparse.ArgumentParser(description='Portfolio RL Backtest')
    parser.add_argument('--dataset', required=True, help='Dataset directory path')
    parser.add_argument('--model', default=None, help='Model checkpoint path (.pt)')
    parser.add_argument('--model-name', default=None,
                        help='모델명 (e.g. etf-swing-v2). models/{name}/policy_best.pt 자동 사용')
    parser.add_argument('--codes', nargs='+', default=None, help='ETF codes')
    parser.add_argument('--lookback', type=int, default=20)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default=None, help='Output JSON path for metrics')
    parser.add_argument('--initial-balance', type=float, default=10_000_000.0)
    parser.add_argument('--oos-start-date', default=None,
                        help='OOS 평가 시작일 (YYYYMMDD). 이 날짜 이전은 lookback용으로만 사용')
    parser.add_argument('--realistic', action='store_true',
                        help='정수 주(株) + 최소 거래금액 제약을 반영한 현실적 체결 시뮬레이션')
    parser.add_argument('--min-trading-price', type=float, default=10_000.0,
                        help='현실적 체결 시 종목당 최소 거래금액 (미만이면 스킵)')
    parser.add_argument('--rebalance-band', type=float, default=0.0,
                        help='현실적 체결 시 리밸런싱 최소 비중 변화 (이하면 스킵)')
    parser.add_argument('--trading-fee', type=float, default=0.00015,
                        help='현실적 체결 시 적용할 실거래 수수료율 (편도)')
    args = parser.parse_args()

    # --model-name: config 자동 로드 (model_path는 --model이 없을 때만 덮어씀)
    model_path = args.model
    if args.model_name:
        model_dir = os.path.join('models', args.model_name)
        if model_path is None:
            model_path = os.path.join(model_dir, 'policy_best.pt')
        config_path = os.path.join(model_dir, 'train_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            args.d_model = cfg.get('d_model', args.d_model)
            args.n_heads = cfg.get('n_heads', args.n_heads)
            args.lookback = cfg.get('lookback', args.lookback)
            saved_codes = cfg.get('etf_codes')
            if saved_codes and args.codes is None:
                args.codes = saved_codes
            print(f"Loaded config: d_model={args.d_model}, n_heads={args.n_heads}, lookback={args.lookback}")
        if args.output is None:
            args.output = os.path.join(model_dir, 'backtest_result.json')

    if model_path is None:
        parser.error("--model 또는 --model-name 을 지정하세요")
    if not os.path.exists(model_path):
        parser.error(f"Model not found: {model_path}")

    etf_codes = args.codes if args.codes else list(TARGET_ETFS.keys())

    # in-sample 피처 컬럼 로드 (모델명 지정 시 해당 학습 데이터 기준)
    feature_cols = None
    if args.model_name:
        cfg_path = os.path.join('models', args.model_name, 'train_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                _cfg = json.load(f)
            train_dataset = _cfg.get('dataset', '')
            if train_dataset and os.path.exists(os.path.join(train_dataset, 'training_scaled.csv')):
                is_df = pd.read_csv(os.path.join(train_dataset, 'training_scaled.csv'), nrows=1)
                feature_cols = list(is_df.columns)
                print(f"In-sample 피처 {len(feature_cols)}개 기준으로 정합 적용")

    asset_data, asset_features = load_dataset(args.dataset, etf_codes, feature_cols=feature_cols)

    oos_start = None
    if args.oos_start_date:
        oos_start = f"{args.oos_start_date[:4]}-{args.oos_start_date[4:6]}-{args.oos_start_date[6:]}"
        print(f"OOS 평가 시작일: {oos_start}")

    env = PortfolioTradingEnvironment(
        asset_data=asset_data,
        asset_features=asset_features,
        lookback=args.lookback,
        initial_balance=args.initial_balance,
        oos_start_date=oos_start,
    )

    policy_net = PortfolioPolicyNetwork(
        n_assets=env.n_assets,
        n_features=env.n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
    )
    value_net = PortfolioValueNetwork(
        n_assets=env.n_assets,
        n_features=env.n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
    )

    agent = PortfolioAgent(
        policy_network=policy_net,
        value_network=value_net,
        device=args.device,
    )
    agent.load(model_path)
    print(f"Loaded model: {model_path}")

    print(f"Running backtest: {env.n_assets} assets, {env.num_steps} steps")
    if args.realistic:
        print(f"  [현실적 체결] min_trading_price={args.min_trading_price:,.0f}원, "
              f"rebalance_band={args.rebalance_band:.2%}, fee={args.trading_fee:.5f}")
        metrics, _ = run_backtest_realistic(
            env, agent,
            min_trading_price=args.min_trading_price,
            rebalance_band=args.rebalance_band,
            trading_fee=args.trading_fee,
        )
    else:
        metrics, trade_log = run_backtest(env, agent)

    print("\n=== Backtest Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            if k in ('cagr', 'mdd', 'win_rate', 'cash_ratio'):
                print(f"  {k:20s}: {v*100:.2f}%")
            else:
                print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        safe = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
        with open(args.output, 'w') as f:
            json.dump(safe, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == '__main__':
    main()
