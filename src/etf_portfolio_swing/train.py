"""
포트폴리오 학습 CLI 엔트리포인트

Usage:
  python -m quantylab.trainer.etf_portfolio_swing.train --dataset <path> [options]
"""
import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import torch

from .environment import PortfolioTradingEnvironment
from .network import PortfolioPolicyNetwork, PortfolioValueNetwork
from .agent import PortfolioAgent
from .trainer import PortfolioPPOTrainer
from ..target_etfs import TARGET_ETFS


def load_dataset(dataset_dir: str, etf_codes: list) -> tuple:
    """통합 CSV에서 ETF별 OHLCV + features 분리 로드.

    데이터셋 구조:
      environment.csv   — date, open, high, low, close, volume, etf_code
      training_scaled.csv — 스케일된 피처 (environment.csv 와 행 순서 동일)
    """
    env_path = os.path.join(dataset_dir, 'environment.csv')
    feat_path = os.path.join(dataset_dir, 'training_scaled.csv')
    if not os.path.exists(env_path):
        raise RuntimeError(f"environment.csv not found in {dataset_dir}")
    if not os.path.exists(feat_path):
        raise RuntimeError(f"training_scaled.csv not found in {dataset_dir}")

    print(f"Loading environment.csv ...")
    env_df = pd.read_csv(env_path, dtype={'etf_code': str})
    env_df['etf_code'] = env_df['etf_code'].str.zfill(6)

    print(f"Loading training_scaled.csv ...")
    feat_df = pd.read_csv(feat_path)

    # 요청된 코드만 필터
    target_set = set(str(c).zfill(6) for c in etf_codes)
    available = set(env_df['etf_code'].unique())
    use_codes = sorted(target_set & available)
    if not use_codes:
        raise RuntimeError(f"None of requested codes found in dataset. Available: {sorted(available)[:10]}")

    asset_data = {}
    asset_features = {}
    for code in use_codes:
        mask = env_df['etf_code'] == code
        df_code = env_df[mask].copy()
        feat_code = feat_df.loc[mask].values.astype(np.float32)
        if len(df_code) < 60:   # 데이터가 너무 짧은 ETF 제외
            continue
        asset_data[code] = df_code
        asset_features[code] = feat_code

    print(f"Loaded {len(asset_data)} assets: {list(asset_data.keys())[:5]}{'...' if len(asset_data)>5 else ''}")
    return asset_data, asset_features


def main():
    parser = argparse.ArgumentParser(description='Portfolio RL Training')
    parser.add_argument('--dataset', required=True, help='Dataset directory path')
    parser.add_argument('--model-name', default=None,
                        help='모델명 (e.g. etf-swing-v2). 지정 시 models/{name}/에 모든 결과 저장')
    parser.add_argument('--codes', nargs='+', default=None, help='ETF codes (default: all TARGET_ETFS)')
    parser.add_argument('--lookback', type=int, default=20)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--lr-policy', type=float, default=0.0002)
    parser.add_argument('--lr-value', type=float, default=0.0005)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--update-interval', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--initial-balance', type=float, default=10_000_000.0)
    # 환경 파라미터
    parser.add_argument('--reward-scale', type=float, default=10.0,
                        help='기본 로그수익률 보상 배율 (기본: 10.0)')
    parser.add_argument('--fee-penalty-scale', type=float, default=5.0,
                        help='거래비용 패널티 배율 (기본: 5.0)')
    parser.add_argument('--drawdown-penalty-threshold', type=float, default=0.10)
    parser.add_argument('--drawdown-penalty-scale', type=float, default=15.0)
    parser.add_argument('--rolling-sharpe-scale', type=float, default=2.0)
    parser.add_argument('--reward-terminal-scale', type=float, default=30.0)
    # 엔트로피 스케줄
    parser.add_argument('--entropy-coef-start', type=float, default=0.10)
    parser.add_argument('--entropy-coef-end', type=float, default=0.05)
    parser.add_argument('--entropy-decay-episodes', type=int, default=400)
    # 전이학습
    parser.add_argument('--base-model', type=str, default=None,
                        help='전이학습 base 모델 디렉토리 (models/ 하위 이름 또는 절대경로)')
    parser.add_argument('--update', action='store_true',
                        help='base-model 가중치를 로드하여 추가 학습')
    parser.add_argument('--update-lr-scale', type=float, default=0.5,
                        help='추가 학습 시 lr 배율 (기본: 0.5)')
    args = parser.parse_args()

    # 학습 중 작업 경로는 항상 output/train/ — 완료 후 models/{name}/으로 복사
    output_dir = 'output/train'
    log_dir = 'output/train'
    model_dir = os.path.join('models', args.model_name) if args.model_name else None

    os.makedirs(output_dir, exist_ok=True)

    etf_codes = args.codes if args.codes else list(TARGET_ETFS.keys())

    asset_data, asset_features = load_dataset(args.dataset, etf_codes)
    actual_codes = list(asset_data.keys())

    env = PortfolioTradingEnvironment(
        asset_data=asset_data,
        asset_features=asset_features,
        lookback=args.lookback,
        initial_balance=args.initial_balance,
        reward_scale=args.reward_scale,
        fee_penalty_scale=args.fee_penalty_scale,
        drawdown_penalty_threshold=args.drawdown_penalty_threshold,
        drawdown_penalty_scale=args.drawdown_penalty_scale,
        rolling_sharpe_scale=args.rolling_sharpe_scale,
        reward_terminal_scale=args.reward_terminal_scale,
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

    lr_policy = args.lr_policy
    lr_value = args.lr_value

    agent = PortfolioAgent(
        policy_network=policy_net,
        value_network=value_net,
        lr_policy=lr_policy,
        lr_value=lr_value,
        device=args.device,
    )

    # 전이학습: base 모델 로드
    if args.update and args.base_model:
        base_dir = args.base_model if os.path.isabs(args.base_model) \
            else os.path.join('models', args.base_model)
        base_path = os.path.join(base_dir, 'policy_best.pt')
        if os.path.exists(base_path):
            try:
                agent.load(base_path)
                lr_policy = args.lr_policy * args.update_lr_scale
                lr_value = args.lr_value * args.update_lr_scale
                for pg in agent.policy_optimizer.param_groups:
                    pg['lr'] = lr_policy
                for pg in agent.value_optimizer.param_groups:
                    pg['lr'] = lr_value
                print(f"  base 모델 로드: {base_path}")
                print(f"  lr 조정: policy={lr_policy}, value={lr_value} (×{args.update_lr_scale})")
            except Exception as e:
                print(f"  base 모델 로드 실패: {e} → 처음부터 학습")
        else:
            print(f"  base 모델 없음: {base_path} → 처음부터 학습")

    # train_config.json 저장
    config = {
        'model_type': 'portfolio',
        'model_name': args.model_name,
        'dataset': args.dataset,
        'etf_codes': actual_codes,
        'n_assets': env.n_assets,
        'n_features': env.n_features,
        'episodes': args.episodes,
        'lookback': args.lookback,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'lr_policy': args.lr_policy,
        'lr_value': args.lr_value,
        'update_interval': args.update_interval,
        'initial_balance': args.initial_balance,
        'drawdown_penalty_threshold': args.drawdown_penalty_threshold,
        'drawdown_penalty_scale': args.drawdown_penalty_scale,
        'reward_scale': args.reward_scale,
        'fee_penalty_scale': args.fee_penalty_scale,
        'rolling_sharpe_scale': args.rolling_sharpe_scale,
        'reward_terminal_scale': args.reward_terminal_scale,
        'entropy_coef_start': args.entropy_coef_start,
        'entropy_coef_end': args.entropy_coef_end,
        'entropy_decay_episodes': args.entropy_decay_episodes,
        'base_model': args.base_model,
        'network': 'EIIE4QLT (GRU×2 + LayerNorm + CrossAttn + Per-Asset Head)',
    }
    with open(os.path.join(output_dir, 'train_config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    trainer = PortfolioPPOTrainer(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        log_dir=log_dir,
        output_dir=output_dir,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        entropy_decay_episodes=args.entropy_decay_episodes,
    )

    print(f"Training portfolio RL: {env.n_assets} assets, {env.n_features} features, {env.num_steps} steps/episode")
    if args.model_name:
        print(f"저장 경로: models/{args.model_name}/")
    trainer.train()
    print("Training complete.")

    # 학습 완료 후 output/train/ → models/{name}/ 복사
    if model_dir:
        import shutil
        os.makedirs(model_dir, exist_ok=True)
        for fname in os.listdir(output_dir):
            shutil.copy2(os.path.join(output_dir, fname), os.path.join(model_dir, fname))
        print(f"모델 복사 완료: output/train/ → {model_dir}/")


if __name__ == '__main__':
    main()
