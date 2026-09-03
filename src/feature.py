"""
Feature Engineering — Trainer helpers

Feature building 로직은 quantylab.trainer.feature_vector 에 있습니다.
이 모듈은 학습 데이터 빌드 파이프라인(training_data)만 포함합니다.
"""

import json
import os
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd

# Re-export the trainer-local implementation so the project is self-contained.
from .feature_vector import *  # noqa: F401, F403
from .quantylab_rest import QuantylabRESTClient


def _resolve_data_source(source: str = "auto", api_key: Optional[str] = None) -> str:
    """데이터 소스 결정.

    - auto: QUANTYLAB_API_KEY가 있으면 api, 없으면 legacy
    - api: feature-vector REST API 강제
    - legacy: 기존 로컬 피처 빌드 경로 강제
    """
    if source not in {"auto", "api", "legacy"}:
        raise ValueError(f"지원하지 않는 source: {source}")
    if source == "auto":
        return "api" if (api_key or os.environ.get("QUANTYLAB_API_KEY", "")) else "legacy"
    return source


def _normalize_feature_names(feature_names: list, dim: int) -> list[str]:
    names = list(feature_names or [])
    if len(names) < dim:
        names.extend([f"feature_{i:03d}" for i in range(len(names), dim)])
    elif len(names) > dim:
        names = names[:dim]
    return names


def _fit_feature_vector(values: list, dim: int) -> list[float]:
    vals = list(values or [])
    if len(vals) < dim:
        vals.extend([0.0] * (dim - len(vals)))
    elif len(vals) > dim:
        vals = vals[:dim]
    return vals


def _fetch_feature_vector_frame(
    code: str,
    start_date: str,
    end_date: str,
    version: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """feature-vector REST API에서 ETF별 시계열 피처를 조회한다."""
    key = api_key or os.environ.get("QUANTYLAB_API_KEY", "")
    if not key:
        raise ValueError("feature-vector API 사용 시 QUANTYLAB_API_KEY 또는 --api-key가 필요합니다.")

    client = QuantylabRESTClient(token=key)
    records = client.get_feature_vectors(
        code=code,
        version=version,
        start_date=start_date,
        end_date=end_date,
    )
    if not records:
        return pd.DataFrame()

    records = list(reversed(records))  # API는 최신순 -> 학습용은 과거순
    sample = records[0]
    dim = len(sample.get("y") or [])
    feature_names = _normalize_feature_names((sample.get("meta") or {}).get("features", []), dim)

    rows = []
    for rec in records:
        values = _fit_feature_vector(rec.get("y"), dim)
        row = {"date": str(rec["x"])}
        row.update({feature_names[i]: values[i] for i in range(dim)})
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True)


def _build_env_frame_for_dates(code: str, dates: pd.Series,
                               start_date: str, end_date: str) -> pd.DataFrame:
    """주어진 날짜 집합에 맞는 environment.csv용 OHLCV를 구성한다."""
    env_df = load_etf_candle(code, start_date, end_date)[
        ['date', 'open', 'high', 'low', 'close', 'volume']
    ].copy()
    env_df['date'] = env_df['date'].astype(str)
    return dates.to_frame(name='date').merge(env_df, on='date', how='inner')


def _write_dataset_metadata(output_dir: str, meta: dict) -> None:
    meta_path = os.path.join(output_dir, 'dataset_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _build_training_data_from_api(
    etf_code: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    feature_version: str,
    api_key: Optional[str] = None,
) -> str:
    """단일 ETF 데이터셋을 feature-vector API 기반으로 생성한다."""
    feature_df = _fetch_feature_vector_frame(
        code=etf_code,
        start_date=start_date,
        end_date=end_date,
        version=feature_version,
        api_key=api_key,
    )
    if feature_df.empty:
        raise ValueError(f"{etf_code}: feature-vector API 데이터가 없습니다.")

    env_df = _build_env_frame_for_dates(
        code=etf_code,
        dates=feature_df['date'],
        start_date=start_date,
        end_date=end_date,
    )
    merged = feature_df.merge(env_df[['date']], on='date', how='inner')
    env_df = env_df.merge(merged[['date']], on='date', how='inner')

    if merged.empty or env_df.empty:
        raise ValueError(f"{etf_code}: feature-vector와 OHLCV 공통 날짜가 없습니다.")

    os.makedirs(output_dir, exist_ok=True)
    env_path = os.path.join(output_dir, 'environment.csv')
    full_path = os.path.join(output_dir, 'training_full.csv')
    selected_path = os.path.join(output_dir, 'training_selected.csv')
    scaled_path = os.path.join(output_dir, 'training_scaled.csv')
    etf_codes_path = os.path.join(output_dir, 'etf_codes.csv')

    env_df.to_csv(env_path, index=False)
    merged.drop(columns=['date']).to_csv(full_path, index=False)
    merged.drop(columns=['date']).to_csv(selected_path, index=False)
    merged.drop(columns=['date']).to_csv(scaled_path, index=False)
    pd.DataFrame({'etf_code': [etf_code] * len(merged)}).to_csv(etf_codes_path, index=False)

    _write_dataset_metadata(output_dir, {
        'source': 'feature-vector-api',
        'feature_version': feature_version,
        'etf_code': etf_code,
        'start_date': start_date,
        'end_date': end_date,
        'rows': int(len(merged)),
        'feature_dim': int(merged.shape[1] - 1),
    })

    print(f"\n[API] 단일 ETF 데이터셋 생성 완료: {etf_code}")
    print(f"  환경 데이터  : {env_path} ({len(env_df):,}행)")
    print(f"  학습 데이터  : {scaled_path} ({merged.shape[1] - 1}컬럼)")
    print(f"  ETF 코드     : {etf_codes_path}")
    return output_dir


def _build_unified_training_data_from_api(
    start_date: str,
    end_date: str,
    output_dir: str,
    min_candles: int,
    feature_version: str,
    api_key: Optional[str] = None,
) -> str:
    """전체 TIGER ETF 통합 데이터셋을 feature-vector API 기반으로 생성한다."""
    etf_list = load_tiger_etf_list(min_candles=min_candles)
    etf_codes = [e['code'] for e in etf_list]
    print(f"\n[API] 통합 학습 데이터 빌드: {len(etf_codes)}개 ETF")

    all_env = []
    all_features = []
    feature_cols = None

    for i, code in enumerate(etf_codes, start=1):
        print(f"\n[{i}/{len(etf_codes)}] {code}")
        feature_df = _fetch_feature_vector_frame(
            code=code,
            start_date=start_date,
            end_date=end_date,
            version=feature_version,
            api_key=api_key,
        )
        if feature_df.empty:
            print(f"  [SKIP] {code}: feature-vector 없음")
            continue

        env_df = _build_env_frame_for_dates(
            code=code,
            dates=feature_df['date'],
            start_date=start_date,
            end_date=end_date,
        )
        merged = feature_df.merge(env_df[['date']], on='date', how='inner')
        env_df = env_df.merge(merged[['date']], on='date', how='inner')

        if merged.empty or env_df.empty:
            print(f"  [SKIP] {code}: OHLCV 매칭 실패")
            continue

        current_cols = [c for c in merged.columns if c != 'date']
        if feature_cols is None:
            feature_cols = current_cols
        else:
            for c in feature_cols:
                if c not in merged.columns:
                    merged[c] = 0.0
            extra_cols = [c for c in current_cols if c not in feature_cols]
            if extra_cols:
                print(f"  [WARN] {code}: 추가 피처 {len(extra_cols)}개 무시")
            merged = merged[['date'] + feature_cols]

        env_df['etf_code'] = code
        all_env.append(env_df)
        all_features.append(merged.assign(_etf_code=code))
        print(f"  -> {len(merged):,}행 추가")

    if not all_features:
        raise ValueError("feature-vector API에서 생성 가능한 ETF 데이터가 없습니다.")

    os.makedirs(output_dir, exist_ok=True)
    unified_env = pd.concat(all_env, ignore_index=True)
    unified_features = pd.concat(all_features, ignore_index=True)
    etf_code_col = unified_features['_etf_code'].copy()
    unified_features = unified_features.drop(columns=['_etf_code'])

    env_path = os.path.join(output_dir, 'environment.csv')
    selected_path = os.path.join(output_dir, 'training_selected.csv')
    scaled_path = os.path.join(output_dir, 'training_scaled.csv')
    full_path = os.path.join(output_dir, 'training_full.csv')
    etf_codes_path = os.path.join(output_dir, 'etf_codes.csv')

    unified_env.to_csv(env_path, index=False)
    unified_features.drop(columns=['date']).to_csv(full_path, index=False)
    unified_features.drop(columns=['date']).to_csv(selected_path, index=False)
    unified_features.drop(columns=['date']).to_csv(scaled_path, index=False)
    etf_code_col.to_csv(etf_codes_path, index=False, header=['etf_code'])

    _write_dataset_metadata(output_dir, {
        'source': 'feature-vector-api',
        'feature_version': feature_version,
        'start_date': start_date,
        'end_date': end_date,
        'rows': int(len(unified_features)),
        'etf_count': int(etf_code_col.nunique()),
        'feature_dim': int(unified_features.shape[1] - 1),
    })

    print(f"\n{'='*50}")
    print("API 통합 학습 데이터 빌드 완료")
    print(f"  ETF 수       : {etf_code_col.nunique()}개")
    print(f"  총 데이터    : {len(unified_features):,}행")
    print(f"  피처 수      : {unified_features.shape[1] - 1}개")
    print(f"  환경 데이터  : {env_path}")
    print(f"  학습 데이터  : {scaled_path}")
    print(f"  ETF 코드     : {etf_codes_path}")
    print(f"{'='*50}")
    return output_dir


def build_training_data(etf_code: str, start_date: str = "20150101",
                        end_date: str = None, output_dir: str = None,
                        etf_name: str = '', etf_extra: dict = None,
                        include_meta: bool = False,
                        verbose: bool = True,
                        source: str = "auto",
                        feature_version: str = "1",
                        api_key: Optional[str] = None) -> str:
    """
    ETF 학습 데이터 빌드 파이프라인

    Returns:
        출력 디렉토리 경로
    """
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', f'{etf_code}_{timestamp}'
        )

    resolved_source = _resolve_data_source(source, api_key=api_key)
    if resolved_source == 'api':
        return _build_training_data_from_api(
            etf_code=etf_code,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            feature_version=feature_version,
            api_key=api_key,
        )

    os.makedirs(output_dir, exist_ok=True)

    # 품질 리포트
    qc = DataQualityReport()

    # 데이터 로드
    data = load_all_data([etf_code], start_date, end_date, qc=qc)

    # 피처 빌드
    print(f"\n[2/2] 피처 엔지니어링: {etf_code}")
    feature_df = build_single_etf_features(
        etf_code, data, etf_name=etf_name, etf_extra=etf_extra,
        include_meta=include_meta, verbose=verbose, qc=qc)

    # 환경 데이터 (OHLCV)
    env_cols = ['date', 'etf_open', 'etf_high', 'etf_low', 'etf_close', 'etf_volume']
    available_env = [c for c in env_cols if c in feature_df.columns]
    env_df = feature_df[available_env].copy()
    env_df.columns = [c.replace('etf_', '') for c in env_df.columns]

    # 정제
    feature_df = clean_and_clip_outliers(feature_df, warm_up_period=60, verbose=verbose)
    env_df = env_df.iloc[-len(feature_df):].reset_index(drop=True)

    # 피처 선택
    selected = select_features(feature_df,
                               get_selected_features(include_meta=include_meta))
    selected = remove_zero_variance(selected, verbose=verbose)

    # 스케일링
    scaler_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'scalers', os.path.basename(output_dir)
    )
    scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
    scaled_df, scaler = scale_features(selected, scaler_path)

    # 저장
    env_path = os.path.join(output_dir, 'environment.csv')
    full_path = os.path.join(output_dir, 'training_full.csv')
    selected_path = os.path.join(output_dir, 'training_selected.csv')
    scaled_path = os.path.join(output_dir, 'training_scaled.csv')

    env_df.to_csv(env_path, index=False)
    feature_df.to_csv(full_path, index=False)
    selected.to_csv(selected_path, index=False)
    scaled_df.to_csv(scaled_path, index=False)

    print(f"\n  환경 데이터  : {env_path} ({len(env_df):,}행)")
    print(f"  전체 피처    : {full_path} ({feature_df.shape[1]}컬럼)")
    print(f"  선택 피처    : {selected_path} ({selected.shape[1]}컬럼)")
    print(f"  스케일링 피처: {scaled_path} ({scaled_df.shape[1]}컬럼)")
    print(f"  스케일러     : {scaler_path}")
    print(f"  피처 목록    : {list(scaled_df.columns)}")

    # 최종 스케일링 데이터 품질 검사
    qc.check_dataframe(scaled_df, 'final_scaled')
    qc.print_report(f'{etf_code} 데이터 품질 리포트')

    return output_dir


def build_unified_training_data(
    start_date: str = "20150101",
    end_date: str = None,
    output_dir: str = None,
    min_candles: int = 500,
    verbose: bool = True,
    source: str = "auto",
    feature_version: str = "1",
    api_key: Optional[str] = None,
) -> str:
    """
    전체 TIGER ETF 통합 학습 데이터 빌드 (일반화 모델용)

    모든 ETF의 피처를 동일한 스케일러로 정규화 후 연결 (concatenate).
    ETF별 구분은 메타 피처(섹터 등)로 조건화.
    environment.csv에 etf_code 컬럼 추가로 경계 표시.

    Returns:
        출력 디렉토리 경로
    """
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', f'unified_{timestamp}'
        )

    resolved_source = _resolve_data_source(source, api_key=api_key)
    if resolved_source == 'api':
        return _build_unified_training_data_from_api(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            min_candles=min_candles,
            feature_version=feature_version,
            api_key=api_key,
        )

    os.makedirs(output_dir, exist_ok=True)

    # 1. ETF 목록
    etf_list = load_tiger_etf_list(min_candles=min_candles)
    etf_codes = [e['code'] for e in etf_list]
    etf_info = {e['code']: e for e in etf_list}
    print(f"\n통합 학습 데이터 빌드: {len(etf_codes)}개 ETF")

    # 품질 리포트
    qc = DataQualityReport()

    # 2. 공통 데이터 1회 로드 (시장/매크로/섹터)
    data = load_all_data(etf_codes, start_date, end_date, qc=qc)

    # 3. ETF별 피처 빌드
    all_env = []
    all_features = []
    feature_cols = None

    for i, etf_code in enumerate(etf_codes):
        info = etf_info[etf_code]
        print(f"\n[{i+1}/{len(etf_codes)}] {etf_code} ({info['name']})")

        try:
            feature_df = build_single_etf_features(
                etf_code, data,
                etf_name=info['name'], etf_extra=info['extra'],
                include_meta=True, verbose=verbose, qc=qc,
            )
        except Exception as e:
            print(f"  [SKIP] {etf_code} 피처 빌드 실패: {e}")
            continue

        # 환경 데이터
        env_cols = ['date', 'etf_open', 'etf_high', 'etf_low', 'etf_close', 'etf_volume']
        available_env = [c for c in env_cols if c in feature_df.columns]
        env_df = feature_df[available_env].copy()
        env_df.columns = [c.replace('etf_', '') for c in env_df.columns]

        # 정제
        feature_df = clean_and_clip_outliers(feature_df, warm_up_period=60, verbose=False)
        env_df = env_df.iloc[-len(feature_df):].reset_index(drop=True)

        if len(feature_df) < 120:  # 최소 6개월
            print(f"  [SKIP] {etf_code} 데이터 부족 ({len(feature_df)}행)")
            continue

        # 피처 선택
        selected = select_features(feature_df,
                                   get_selected_features(include_meta=True))

        # etf_code 추가 (환경 데이터에 ETF 경계 표시)
        env_df['etf_code'] = etf_code
        selected['_etf_code'] = etf_code  # 스케일링 후 제거

        # 피처 컬럼 정합성 맞추기
        if feature_cols is None:
            feature_cols = [c for c in selected.columns if c != '_etf_code']
        else:
            for c in feature_cols:
                if c not in selected.columns:
                    selected[c] = 0.0
            selected = selected[['_etf_code'] + feature_cols]

        all_env.append(env_df)
        all_features.append(selected)
        print(f"  → {len(feature_df)}행 추가")

    if not all_features:
        raise ValueError("유효한 ETF 데이터가 없습니다.")

    # 4. 통합
    unified_env = pd.concat(all_env, ignore_index=True)
    unified_features = pd.concat(all_features, ignore_index=True)

    # etf_code 컬럼 분리
    etf_code_col = unified_features['_etf_code'].copy()
    unified_features = unified_features.drop(columns=['_etf_code'])
    unified_features = remove_zero_variance(unified_features, verbose=verbose)

    # 5. 통합 스케일링
    scaler_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'scalers', os.path.basename(output_dir)
    )
    scaler_path = os.path.join(scaler_dir, 'scaler.pkl')

    # 메타 피처는 이미 0/1이므로 스케일 안 함
    meta_cols = [c for c in unified_features.columns if c.startswith('meta_sector_')]
    non_meta_cols = [c for c in unified_features.columns if c not in meta_cols]

    scaler = StandardScaler()
    scaled_non_meta = scaler.fit_transform(unified_features[non_meta_cols])
    scaled_non_meta = np.clip(scaled_non_meta, -5.0, 5.0)

    scaled_df = pd.DataFrame(scaled_non_meta, columns=non_meta_cols)
    for c in meta_cols:
        scaled_df[c] = unified_features[c].values

    # 원래 피처 순서 유지
    final_cols = [c for c in unified_features.columns if c in scaled_df.columns]
    scaled_df = scaled_df[final_cols]

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # 6. 저장
    env_path = os.path.join(output_dir, 'environment.csv')
    selected_path = os.path.join(output_dir, 'training_selected.csv')
    scaled_path = os.path.join(output_dir, 'training_scaled.csv')
    etf_codes_path = os.path.join(output_dir, 'etf_codes.csv')

    unified_env.to_csv(env_path, index=False)
    unified_features.to_csv(selected_path, index=False)
    scaled_df.to_csv(scaled_path, index=False)
    etf_code_col.to_csv(etf_codes_path, index=False, header=['etf_code'])

    total_rows = len(scaled_df)
    n_etfs = etf_code_col.nunique()
    print(f"\n{'='*50}")
    print(f"통합 학습 데이터 빌드 완료")
    print(f"  ETF 수       : {n_etfs}개")
    print(f"  총 데이터    : {total_rows:,}행")
    print(f"  피처 수      : {scaled_df.shape[1]}개")
    print(f"  환경 데이터  : {env_path}")
    print(f"  학습 데이터  : {scaled_path}")
    print(f"  ETF 코드     : {etf_codes_path}")
    print(f"  스케일러     : {scaler_path}")
    print(f"{'='*50}")

    # 최종 스케일링 데이터 품질 검사
    qc.check_dataframe(scaled_df, 'final_scaled')
    qc.print_report('통합 데이터 품질 리포트')

    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETF 학습 데이터 빌드')
    parser.add_argument('--etf-code', type=str, default=None,
                        help='ETF 코드 (예: 069500=TIGER200). 미지정 시 --unified 필수')
    parser.add_argument('--start-date', type=str, default='20150101')
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--name', type=str, default=None,
                        help='데이터셋 이름 (output-dir 자동 결정)')
    parser.add_argument('--unified', action='store_true',
                        help='전체 TIGER ETF 통합 학습 데이터 빌드 (일반화 모델용)')
    parser.add_argument('--min-candles', type=int, default=500,
                        help='통합 모드 최소 캔들 수 (기본 500)')
    parser.add_argument('--source', type=str, default='auto',
                        choices=['auto', 'api', 'legacy'],
                        help='데이터셋 생성 소스 (auto=API 키가 있으면 API, 없으면 기존 방식)')
    parser.add_argument('--feature-version', type=str, default='1',
                        help='feature-vector 버전 (API 소스에서만 사용)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Quantylab REST API 토큰 (미지정 시 QUANTYLAB_API_KEY 사용)')

    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None and args.name:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', args.name
        )

    if args.unified:
        build_unified_training_data(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            min_candles=args.min_candles,
            source=args.source,
            feature_version=args.feature_version,
            api_key=args.api_key,
        )
    elif args.etf_code:
        build_training_data(
            etf_code=args.etf_code,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            include_meta=True,
            source=args.source,
            feature_version=args.feature_version,
            api_key=args.api_key,
        )
    else:
        parser.error('--etf-code 또는 --unified 중 하나를 지정하세요.')
