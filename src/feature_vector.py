"""
Feature Engineering Module for ETF Swing Trading

피처 빌드, DB 저장, REST API 로더, Prefect task를 포함합니다.

데이터 소스 (models.py 기반):
  1. DayEtfCandle: ETF 일봉 캔들 (OHLCV)
  2. StockMarketDayCandle: KOSPI/KOSDAQ 시장 일봉
  3. ForeignStockMarketDayCandle: 해외 시장 (S&P500, NASDAQ, 다우 등)
  4. Fx: 환율 (USD/KRW 등)
  5. Vix / KospiVix: 변동성 지수
  6. Sox: 필라델피아 반도체 지수
  7. UsBond10YearYield / KrBond3YearYield: 채권 수익률
  8. Gsci: 원자재 지수 (S&P GSCI)
  9. Dx: 달러 인덱스
  10. CnnFearGreed: 공포/탐욕 지수
  11. StockIndexPerPbr: 지수 PER/PBR
  12. CitiSurprise: 시티 서프라이즈 지수
  13. StockCredit: 신용 거래 (융자 잔고, 신규, 상환 등)
  14. FutureDayCandle: KOSPI200 선물 일봉 (베이시스, 거래량비율)
  15. ProgramVolume: 프로그램 매매 (순매수, 순매수 증감)
  16. InvestorBuySell: 투자자별 매매 동향 (외국인/기관/개인)
  17. StockMarketShortVolume: 시장 공매도 거래량
  18. ShortBalance: 종목별 공매도 잔고
  19. Bdi: Baltic Dry Index (해운 운임)
  20. Scfi: Shanghai Containerized Freight Index (컨테이너 운임)
  21. NaverFinCommodity: WTI 유가, 금, 은, 구리
  22. InvestingCommodity: 철광석
  23. MsciIndex: MSCI EM/WORLD 지수
"""

import os
import argparse
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from prefect import task
    from prefect.tasks import task_input_hash
except Exception:  # Prefect is optional for local/standalone trainer usage.
    def task(**_kwargs):
        return lambda function: function

    def task_input_hash(*_args, **_kwargs):
        return None

from .db import psql
from .models import FeatureVector


DEFAULT_N = 20
MIN_FEATURE_ROWS = 120
OUTLIER_WARMUP_PERIOD = 60
TRADING_TO_CALENDAR_FACTOR = 1.6
LOAD_BUFFER_MARGIN_DAYS = 30

# 타겟 TIGER ETF 고정 리스트 {코드: 이름}
TARGET_ETFS = {
    # ── 시장 지수 ──
    "102110": "TIGER 200",
    "277630": "TIGER 코스피",
    "277640": "TIGER 코스피대형주",
    "277650": "TIGER 코스피중형주",
    "232080": "TIGER 코스닥150",
    "292160": "TIGER KRX300",
    "228820": "TIGER KTOP30",
    "292150": "TIGER 코리아TOP10",
    "252000": "TIGER 200동일가중",
    "310960": "TIGER 200TR",
    "496080": "TIGER 코리아밸류업",
    # ── 반도체/IT ──
    "091230": "TIGER 반도체",
    "139260": "TIGER 200 IT",
    "396500": "TIGER 반도체TOP10",
    "471760": "TIGER AI반도체핵심공정",
    "157490": "TIGER 소프트웨어",
    "261060": "TIGER 코스닥150IT",
    "315270": "TIGER 200커뮤니케이션서비스",
    "471780": "TIGER 코리아테크액티브",
    "365040": "TIGER AI코리아그로스액티브",
    # ── 2차전지/모빌리티 ──
    "305540": "TIGER 2차전지테마",
    "364980": "TIGER 2차전지TOP10",
    "387280": "TIGER 퓨처모빌리티액티브",
    # ── 금융 ──
    "091220": "TIGER 은행",
    "139270": "TIGER 200 금융",
    "157500": "TIGER 증권",
    "466940": "TIGER 은행고배당플러스TOP10",
    # ── 헬스케어/바이오 ──
    "143860": "TIGER 헬스케어",
    "227540": "TIGER 200 헬스케어",
    "261070": "TIGER 코스닥150바이오테크",
    "364970": "TIGER 바이오TOP10",
    "307510": "TIGER 의료기기",
    # ── 산업재/건설/중공업 ──
    "139220": "TIGER 200 건설",
    "139230": "TIGER 200 중공업",
    "227550": "TIGER 200 산업재",
    "139240": "TIGER 200 철강소재",
    "494670": "TIGER 조선TOP10",
    # ── 에너지/소재 ──
    "139250": "TIGER 200 에너지화학",
    # ── 소비재/서비스 ──
    "139290": "TIGER 200 경기소비재",
    "227560": "TIGER 200 생활소비재",
    "139280": "TIGER 경기방어",
    "228790": "TIGER 화장품",
    "228800": "TIGER 여행레저",
    "228810": "TIGER 미디어컨텐츠",
    # ── 게임/인터넷 ──
    "300610": "TIGER K게임",
    "364990": "TIGER 게임TOP10",
    "365000": "TIGER 인터넷TOP10",
    "364960": "TIGER BBIG",
    # ── 방산/우주 ──
    "463250": "TIGER K방산&우주",
    # ── 배당/가치/스타일 ──
    "210780": "TIGER 코스피고배당",
    "211560": "TIGER 배당성장",
    "445910": "TIGER MKF배당귀족",
    "227570": "TIGER 우량가치",
    "147970": "TIGER 모멘텀",
    "174350": "TIGER 로우볼",
    "261140": "TIGER 우선주",
    # ── 그룹주 ──
    "138520": "TIGER 삼성그룹",
    "138530": "TIGER LG그룹플러스",
    "138540": "TIGER 현대차그룹플러스",
    "307520": "TIGER 지주회사",
    # ── ESG/테마 ──
    "376410": "TIGER 탄소효율그린뉴딜",
    "404540": "TIGER KRX기후변화솔루션",
    "417630": "TIGER KEDI혁신기업ESG30",
    # ── 리츠/부동산 ──
    "329200": "TIGER 리츠부동산인프라",
}


# ============================================================
# 0. 데이터 품질 리포트
# ============================================================

class DataQualityReport:
    """데이터 품질 이슈 수집 및 리포트"""

    def __init__(self):
        self.missing_sources: List[str] = []     # DB에 데이터 없음
        self.empty_sources: List[str] = []        # 로드 실패 또는 0행
        self.load_errors: List[Tuple[str, str]] = []  # (소스명, 에러메시지)
        self.column_issues: Dict[str, dict] = {}  # {col: {nan, inf, zero_var, ...}}
        self.merge_issues: List[str] = []          # 머지 후 NaN 발생 컬럼

    def check_loaded_data(self, data: dict, required_sources: List[str] = None):
        """로드된 데이터 소스별 품질 검사"""
        if required_sources is None:
            required_sources = []
        for src in required_sources:
            if src not in data:
                self.missing_sources.append(src)
            elif isinstance(data[src], pd.DataFrame) and len(data[src]) == 0:
                self.empty_sources.append(src)

        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and len(df) == 0:
                if key not in self.empty_sources:
                    self.empty_sources.append(key)

    def check_dataframe(self, df: pd.DataFrame, stage: str = ''):
        """DataFrame의 NaN, Inf, 제로 분산 컬럼 검사"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            issues = {}
            nan_count = int(df[col].isna().sum())
            inf_count = int(np.isinf(df[col]).sum()) if df[col].dtype != object else 0
            if nan_count > 0:
                nan_pct = nan_count / len(df) * 100
                issues['nan'] = f'{nan_count} ({nan_pct:.1f}%)'
            if inf_count > 0:
                issues['inf'] = inf_count
            if df[col].std() == 0 and len(df) > 1:
                issues['zero_variance'] = True
            if issues:
                key = f'{stage}/{col}' if stage else col
                self.column_issues[key] = issues

    def print_report(self, title: str = '데이터 품질 리포트'):
        """품질 리포트 출력"""
        has_issues = (self.missing_sources or self.empty_sources or
                      self.load_errors or self.column_issues)
        if not has_issues:
            print(f"\n  [QC] {title}: 이슈 없음 ✓")
            return

        print(f"\n{'='*60}")
        print(f"  [QC] {title}")
        print(f"{'='*60}")

        if self.missing_sources:
            print(f"\n  ■ DB 데이터 없음 ({len(self.missing_sources)}개):")
            for src in self.missing_sources:
                print(f"    - {src}")

        if self.empty_sources:
            print(f"\n  ■ 빈 데이터 ({len(self.empty_sources)}개):")
            for src in self.empty_sources:
                print(f"    - {src}")

        if self.load_errors:
            print(f"\n  ■ 로드 실패 ({len(self.load_errors)}개):")
            for src, err in self.load_errors:
                print(f"    - {src}: {err}")

        if self.column_issues:
            nan_cols = {k: v for k, v in self.column_issues.items() if 'nan' in v}
            inf_cols = {k: v for k, v in self.column_issues.items() if 'inf' in v}
            zv_cols = {k: v for k, v in self.column_issues.items() if 'zero_variance' in v}

            if nan_cols:
                print(f"\n  ■ NaN 컬럼 ({len(nan_cols)}개):")
                for col, info in sorted(nan_cols.items(),
                                        key=lambda x: -int(x[1]['nan'].split()[0])):
                    print(f"    - {col}: {info['nan']}")

            if inf_cols:
                print(f"\n  ■ Inf 컬럼 ({len(inf_cols)}개):")
                for col, info in inf_cols.items():
                    print(f"    - {col}: {info['inf']}건")

            if zv_cols:
                print(f"\n  ■ 제로 분산 컬럼 ({len(zv_cols)}개):")
                for col in list(zv_cols.keys())[:20]:
                    print(f"    - {col}")
                if len(zv_cols) > 20:
                    print(f"    ... 외 {len(zv_cols) - 20}개")

        print(f"{'='*60}")


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_etf_candle(code: str, start_date: str = "20150101",
                    end_date: str = None, limit: int = None) -> pd.DataFrame:
    """ETF 일봉 캔들 로드 (DayEtfCandle)"""
    from .db import psql
    from .models import DayEtfCandle
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(DayEtfCandle).where(
            DayEtfCandle.code == code,
            DayEtfCandle.date >= start_date,
        )
        if end_date:
            query = query.where(DayEtfCandle.date <= end_date)
        query = query.order_by(DayEtfCandle.date.asc())
        if limit:
            query = query.limit(limit)
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    print(f"  ETF {code}: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_market_candle(code: str = "kospi", start_date: str = "20150101",
                       end_date: str = None) -> pd.DataFrame:
    """시장 일봉 캔들 로드 (StockMarketDayCandle)"""
    from .db import data_access
    df = data_access.get_stock_market_day_candles(code, start_date, end_date)
    print(f"  시장 {code}: {len(df)}행 로드")
    return df


def load_foreign_market(code: str, start_date: str = "20150101",
                        end_date: str = None) -> pd.DataFrame:
    """해외 시장 일봉 로드 (ForeignStockMarketDayCandle)"""
    from .db import data_access
    df = data_access.get_foreign_stock_market_day_candles(code, start_date, end_date)
    print(f"  해외시장 {code}: {len(df)}행 로드")
    return df


def load_fx(code: str = "FX_USDKRW", start_date: str = "20150101",
            end_date: str = None) -> pd.DataFrame:
    """환율 로드"""
    from .db import data_access
    df = data_access.get_fx(code, start_date, end_date)
    print(f"  환율 {code}: {len(df)}행 로드")
    return df


def load_vix(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """VIX 로드"""
    from .db import data_access
    df = data_access.get_vix(start_date, end_date)
    print(f"  VIX: {len(df)}행 로드")
    return df


def load_kospi_vix(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """KOSPI VIX 로드"""
    from .db import data_access
    df = data_access.get_kospi_vix(start_date, end_date)
    print(f"  KOSPI VIX: {len(df)}행 로드")
    return df


def load_us_bond_10y(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """미국 10년 국채 수익률 로드"""
    from .db import data_access
    df = data_access.get_us_bond_10_year_yield(start_date, end_date or "29991231")
    print(f"  US Bond 10Y: {len(df)}행 로드")
    return df


def load_kr_bond_3y(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """한국 3년 국채 수익률 로드"""
    from .db import data_access
    df = data_access.get_kr_bond_3_year_yield(start_date, end_date or "29991231")
    print(f"  KR Bond 3Y: {len(df)}행 로드")
    return df


def load_sox(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """SOX 반도체 지수 로드"""
    from .db import data_access
    df = data_access.get_sox(start_date, end_date)
    print(f"  SOX: {len(df)}행 로드")
    return df


def load_gsci(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """GSCI 원자재 지수 로드"""
    from .db import data_access
    df = data_access.get_gsci(start_date, end_date)
    print(f"  GSCI: {len(df)}행 로드")
    return df


def load_dx(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """달러 인덱스 로드"""
    from .db import data_access
    df = data_access.get_dx(start_date, end_date)
    print(f"  DX: {len(df)}행 로드")
    return df


def load_cnn_fear_greed(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """CNN Fear & Greed 지수 로드"""
    from .db import data_access
    df = data_access.get_cnn_fear_greed_index(start_date, end_date)
    print(f"  CNN Fear&Greed: {len(df)}행 로드")
    return df


def load_stock_index_per_pbr(name: str = "코스피", start_date: str = "20150101",
                             end_date: str = None) -> pd.DataFrame:
    """지수 PER/PBR 로드"""
    from .db import data_access
    df = data_access.get_stock_index_per_pbr(name, start_date, end_date)
    if df is None:
        df = pd.DataFrame()
    print(f"  지수 PER/PBR ({name}): {len(df)}행 로드")
    return df


def load_citi_surprise(group_name: str = "South Korea",
                       start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """시티 서프라이즈 지수 로드"""
    from .db import data_access
    df = data_access.get_citi_surprise(group_name, start_date, end_date)
    print(f"  Citi Surprise ({group_name}): {len(df)}행 로드")
    return df


def load_futures_candle(code: str = "kospi200f", start_date: str = "20150101",
                        end_date: str = None) -> pd.DataFrame:
    """선물 일봉 캔들 로드 (FutureDayCandle)"""
    from .db import psql
    from .models import FutureDayCandle
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(FutureDayCandle).where(
            FutureDayCandle.code == code,
            FutureDayCandle.date >= start_date,
        )
        if end_date:
            query = query.where(FutureDayCandle.date <= end_date)
        query = query.order_by(FutureDayCandle.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    print(f"  선물 {code}: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_program_volume(code: str, start_date: str = "20150101",
                        end_date: str = None) -> pd.DataFrame:
    """프로그램 거래량 로드 (ProgramVolume) — date가 Integer"""
    from .db import psql
    from .models import ProgramVolume
    from sqlalchemy import select

    start_int = int(start_date)
    end_int = int(end_date) if end_date else None

    with psql.get_session() as session:
        query = select(ProgramVolume).where(
            ProgramVolume.code == code,
            ProgramVolume.date >= start_int,
        )
        if end_int:
            query = query.where(ProgramVolume.date <= end_int)
        query = query.order_by(ProgramVolume.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    # date를 문자열로 변환 (머지 호환)
    if len(df) > 0:
        df['date'] = df['date'].astype(str)

    print(f"  프로그램거래 {code}: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_investor_buy_sell(code: str, start_date: str = "20150101",
                           end_date: str = None) -> pd.DataFrame:
    """투자자 매매 동향 로드 (InvestorBuySell)"""
    from .db import psql
    from .models import InvestorBuySell
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(InvestorBuySell).where(
            InvestorBuySell.code == code,
            InvestorBuySell.date >= start_date,
        )
        if end_date:
            query = query.where(InvestorBuySell.date <= end_date)
        query = query.order_by(InvestorBuySell.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    print(f"  투자자매매 {code}: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_short_volume(code: str = "kospi", start_date: str = "20150101",
                      end_date: str = None) -> pd.DataFrame:
    """시장 공매도 거래량 로드 (StockMarketShortVolume, JSONB content)"""
    from .db import psql
    from .models import StockMarketShortVolume
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(StockMarketShortVolume).where(
            StockMarketShortVolume.code == code,
            StockMarketShortVolume.date >= start_date,
        )
        if end_date:
            query = query.where(StockMarketShortVolume.date <= end_date)
        query = query.order_by(StockMarketShortVolume.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    # JSONB content를 컬럼으로 펼침
    if len(df) > 0 and 'content' in df.columns:
        content_df = pd.json_normalize(df['content'])
        for col in content_df.columns:
            # 공백 제거 (키에 공백 있을 수 있음)
            clean_col = col.strip()
            df[f'short_vol_{clean_col}'] = pd.to_numeric(content_df[col], errors='coerce').fillna(0)
        df.drop(columns=['content'], inplace=True)

    print(f"  시장공매도거래량 {code}: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_short_balance(code: str, start_date: str = "20150101",
                       end_date: str = None) -> pd.DataFrame:
    """공매도 잔고 로드 (ShortBalance)"""
    from .db import psql
    from .models import ShortBalance
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(ShortBalance).where(
            ShortBalance.code == code,
            ShortBalance.date >= start_date,
        )
        if end_date:
            query = query.where(ShortBalance.date <= end_date)
        query = query.order_by(ShortBalance.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    print(f"  공매도잔고 {code}: {len(df)}행 로드")
    if len(df) == 0:
        return df
    return df.sort_values('date').reset_index(drop=True)


def load_bdi(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """BDI (Baltic Dry Index) 로드"""
    from .db import psql
    from .models import Bdi
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(Bdi).where(Bdi.date >= start_date)
        if end_date:
            query = query.where(Bdi.date <= end_date)
        query = query.order_by(Bdi.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    print(f"  BDI: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_commodity(type_name: str, start_date: str = "20150101",
                   end_date: str = None,
                   source: str = "naver") -> pd.DataFrame:
    """원자재 로드 (NaverFinCommodity 또는 InvestingCommodity)"""
    from .db import psql
    from sqlalchemy import select

    if source == "investing":
        from .models import InvestingCommodity as Model
    else:
        from .models import NaverFinCommodity as Model

    with psql.get_session() as session:
        query = select(Model).where(
            Model.type_name == type_name,
            Model.date >= start_date,
        )
        if end_date:
            query = query.where(Model.date <= end_date)
        query = query.order_by(Model.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    print(f"  원자재 {type_name}: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True) if len(df) > 0 else df


def load_scfi(start_date: str = "20150101", end_date: str = None) -> pd.DataFrame:
    """SCFI (Shanghai Containerized Freight Index) 로드 — 주간 데이터, JSONB content"""
    from .db import psql
    from .models import Scfi
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(Scfi).where(Scfi.date >= start_date)
        if end_date:
            query = query.where(Scfi.date <= end_date)
        query = query.order_by(Scfi.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    if len(df) > 0 and 'content' in df.columns:
        df['value'] = df['content'].apply(
            lambda x: x.get('COMPREHENSIVE INDEX', 0) if isinstance(x, dict) else 0
        )
        df.drop(columns=['content'], inplace=True)

    print(f"  SCFI: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_msci_index(name: str, start_date: str = "20150101",
                    end_date: str = None) -> pd.DataFrame:
    """MSCI 지수 로드"""
    from .db import psql
    from .models import MsciIndex
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(MsciIndex).where(
            MsciIndex.name == name,
            MsciIndex.date >= start_date,
        )
        if end_date:
            query = query.where(MsciIndex.date <= end_date)
        query = query.order_by(MsciIndex.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    print(f"  MSCI {name}: {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_stock_credit(code: str, start_date: str = "20150101",
                      end_date: str = None, qry_tp: str = "1") -> pd.DataFrame:
    """신용 거래 로드 (StockCredit). 기본값 qry_tp='1' (융자)"""
    from .db import psql
    from .models import StockCredit
    from sqlalchemy import select

    with psql.get_session() as session:
        query = select(StockCredit).where(
            StockCredit.code == code,
            StockCredit.qry_tp == qry_tp,
            StockCredit.date >= start_date,
        )
        if end_date:
            query = query.where(StockCredit.date <= end_date)
        query = query.order_by(StockCredit.date.asc())
        result = session.execute(query).scalars().all()
        df = psql.get_df_from_result(result)

    print(f"  신용거래 {code} (qry_tp={qry_tp}): {len(df)}행 로드")
    return df.sort_values('date').reset_index(drop=True)


def load_tiger_etf_list(min_candles: int = 250) -> List[dict]:
    """타겟 TIGER ETF 목록 로드 (고정 리스트 기반)

    Args:
        min_candles: 최소 캔들 수 (데이터 부족 ETF 제외)

    Returns:
        [{code, name, extra, candles}, ...]
    """
    from .models import EtfCode, DayEtfCandle
    from sqlalchemy import select, func

    with psql.get_session() as session:
        etfs = []
        for code, name in TARGET_ETFS.items():
            # DB에서 메타 정보 조회
            etf_obj = session.execute(
                select(EtfCode).where(EtfCode.code == code)
            ).scalar_one_or_none()

            extra = etf_obj.extra if etf_obj and etf_obj.extra else {}

            # 캔들 수 확인
            count = session.execute(
                select(func.count()).select_from(DayEtfCandle)
                .where(DayEtfCandle.code == code)
            ).scalar()
            if count < min_candles:
                print(f"  [SKIP] {code} ({name}): {count}캔들 < {min_candles}")
                continue
            etfs.append({
                'code': code, 'name': name,
                'extra': extra, 'candles': count,
            })

    etfs.sort(key=lambda x: x['candles'], reverse=True)
    print(f"타겟 TIGER ETF: {len(etfs)}개 (고정 리스트 {len(TARGET_ETFS)}개 중)")
    return etfs


def classify_etf_sector(name: str) -> str:
    """ETF 이름으로 섹터 분류"""
    sector_map = [
        ('반도체', 'semiconductor'), ('IT', 'it'), ('소프트웨어', 'software'),
        ('AI', 'ai'), ('게임', 'game'), ('인터넷', 'internet'),
        ('미디어', 'media'), ('코리아테크', 'tech'),
        ('2차전지', 'battery'), ('에너지화학', 'energy'),
        ('전력', 'power'), ('원자력', 'nuclear'),
        ('헬스케어', 'healthcare'), ('바이오', 'bio'), ('의료', 'medical'),
        ('은행', 'bank'), ('금융', 'finance'), ('증권', 'securities'),
        ('건설', 'construction'), ('중공업', 'heavy_industry'),
        ('조선', 'shipbuilding'), ('철강', 'steel'),
        ('산업재', 'industrial'), ('기계', 'machinery'),
        ('화장품', 'cosmetics'), ('여행레저', 'travel'),
        ('생활소비재', 'consumer_staples'), ('경기소비재', 'consumer_disc'),
        ('경기방어', 'defensive'),
        ('자동차', 'auto'), ('현대차', 'auto'), ('모빌리티', 'mobility'),
        ('방산', 'defense'), ('우주', 'space'),
        ('리츠', 'reits'), ('부동산', 'reits'),
        ('배당', 'dividend'), ('고배당', 'dividend'), ('밸류', 'value'),
        ('모멘텀', 'momentum'), ('로우볼', 'lowvol'), ('그로스', 'growth'),
        ('ESG', 'esg'), ('탄소', 'esg'), ('기후', 'esg'),
        ('200', 'index'), ('코스피', 'index'), ('코스닥', 'index'),
        ('KRX', 'index'), ('KTOP', 'index'), ('TOP10', 'index'),
        ('삼성', 'conglomerate'), ('LG', 'conglomerate'),
        ('지주회사', 'conglomerate'), ('그룹', 'conglomerate'),
        ('로봇', 'robot'), ('휴머노이드', 'robot'),
    ]
    for keyword, sector in sector_map:
        if keyword in name:
            return sector
    return 'other'


# 섹터 원-핫 인코딩용 전체 섹터 목록
ALL_SECTORS = [
    'semiconductor', 'it', 'software', 'ai', 'game', 'internet', 'media', 'tech',
    'battery', 'energy', 'power', 'nuclear',
    'healthcare', 'bio', 'medical',
    'bank', 'finance', 'securities',
    'construction', 'heavy_industry', 'shipbuilding', 'steel', 'industrial', 'machinery',
    'cosmetics', 'travel', 'consumer_staples', 'consumer_disc', 'defensive',
    'auto', 'mobility', 'defense', 'space',
    'reits', 'dividend', 'value', 'momentum', 'lowvol', 'growth', 'esg',
    'index', 'conglomerate', 'robot', 'other',
]


def create_etf_meta_features(df: pd.DataFrame, etf_name: str,
                             etf_extra: dict = None) -> pd.DataFrame:
    """ETF 메타 피처 생성 (일반화 모델 조건 입력)

    ETF의 정적 특성을 모든 행에 동일하게 추가:
    - 섹터 원-핫 인코딩
    - 롤링 변동성 수준 (ETF 고유 특성)
    - 상장 경과 일수 (정규화)
    """
    result = df.copy()

    # 섹터 원-핫
    sector = classify_etf_sector(etf_name)
    for s in ALL_SECTORS:
        result[f'meta_sector_{s}'] = 1.0 if s == sector else 0.0

    # 롤링 변동성 프로파일 (ETF마다 다른 변동성 수준)
    if 'etf_returns' in result.columns:
        result['meta_avg_volatility'] = result['etf_returns'].rolling(
            window=60, min_periods=5).std().fillna(0).clip(0, 0.1)
    else:
        result['meta_avg_volatility'] = 0.0

    # 상장 경과 일수 (정규화: 10년 = 1.0)
    if etf_extra and 'listeddate' in etf_extra:
        listed_date = pd.to_datetime(str(etf_extra['listeddate']), format='%Y%m%d')
        dates = pd.to_datetime(result['date'], format='%Y%m%d')
        result['meta_listing_age'] = ((dates - listed_date).dt.days / 3650.0).clip(0, 3)
    else:
        result['meta_listing_age'] = 0.0

    # 총 상장주식수 (유동성 프록시, 로그 정규화)
    if etf_extra and '총상장주식수' in etf_extra:
        total_shares = max(etf_extra['총상장주식수'], 1)
        result['meta_log_shares'] = np.log10(total_shares) / 8.0  # 1억주 = 1.0
    else:
        result['meta_log_shares'] = 0.0

    return result


def get_meta_feature_names() -> List[str]:
    """메타 피처 이름 리스트"""
    features = [f'meta_sector_{s}' for s in ALL_SECTORS]
    features += ['meta_avg_volatility', 'meta_listing_age', 'meta_log_shares']
    return features


def load_all_data(etf_codes: List[str], start_date: str = "20150101",
                  end_date: str = None,
                  qc: DataQualityReport = None) -> dict:
    """모든 데이터 소스 로드

    Args:
        etf_codes: TIGER ETF 코드 리스트
        start_date: 시작일
        end_date: 종료일

    Returns:
        딕셔너리 {소스명: DataFrame}
    """
    print(f"[1/2] 데이터 로드: {start_date} ~ {end_date or 'latest'}")
    data = {}

    # ETF 캔들 + 신용 거래 + 프로그램매매 + 투자자매매 + 공매도잔고
    for code in etf_codes:
        data[f'etf_{code}'] = load_etf_candle(code, start_date, end_date)
        for loader_name, loader_fn in [
            (f'credit_{code}', lambda c=code: load_stock_credit(c, start_date, end_date)),
            (f'program_{code}', lambda c=code: load_program_volume(c, start_date, end_date)),
            (f'investor_{code}', lambda c=code: load_investor_buy_sell(c, start_date, end_date)),
            (f'short_bal_{code}', lambda c=code: load_short_balance(c, start_date, end_date)),
        ]:
            try:
                data[loader_name] = loader_fn()
            except Exception as e:
                print(f"  [WARN] {loader_name} 로드 실패: {e}")
                if qc is not None:
                    qc.load_errors.append((loader_name, str(e)))

    # 시장 캔들
    data['kospi'] = load_market_candle('kospi', start_date, end_date)
    data['kosdaq'] = load_market_candle('kosdaq', start_date, end_date)

    # 해외 시장 (전일 야간 수익률 → 당일 갭 선행지표)
    foreign_code_map = {
        'snp500': 'SPI@SPX',
        'nasdaq': 'NAS@IXIC',
        'dow': 'DJI@DJI',
    }
    for name, db_code in foreign_code_map.items():
        try:
            data[name] = load_foreign_market(db_code, start_date, end_date)
        except Exception as e:
            print(f"  [WARN] {name} 로드 실패: {e}")
            if qc is not None:
                qc.load_errors.append((name, str(e)))

    # 환율, 금리, 변동성, 센티먼트, 선물, 공매도, BDI, MSCI
    macro_loaders = [
        ('fx_usdkrw', lambda: load_fx('FX_USDKRW', start_date, end_date)),
        ('vix', lambda: load_vix(start_date, end_date)),
        ('kospi_vix', lambda: load_kospi_vix(start_date, end_date)),
        ('us_bond_10y', lambda: load_us_bond_10y(start_date, end_date)),
        ('kr_bond_3y', lambda: load_kr_bond_3y(start_date, end_date)),
        ('sox', lambda: load_sox(start_date, end_date)),
        ('gsci', lambda: load_gsci(start_date, end_date)),
        ('dx', lambda: load_dx(start_date, end_date)),
        ('cnn_fear_greed', lambda: load_cnn_fear_greed(start_date, end_date)),
        ('kospi_per_pbr', lambda: load_stock_index_per_pbr('코스피', start_date, end_date)),
        ('citi_surprise_kr', lambda: load_citi_surprise('Asia Pacific', start_date, end_date)),
        ('futures', lambda: load_futures_candle('kospi200f', start_date, end_date)),
        ('short_volume', lambda: load_short_volume('kospi', start_date, end_date)),
        ('bdi', lambda: load_bdi(start_date, end_date)),
        ('scfi', lambda: load_scfi(start_date, end_date)),
        ('wti', lambda: load_commodity('oil_wti', start_date, end_date)),
        ('gold', lambda: load_commodity('world_gold', start_date, end_date)),
        ('silver', lambda: load_commodity('silver', start_date, end_date)),
        ('iron', lambda: load_commodity('iron', start_date, end_date, source='investing')),
        ('copper', lambda: load_commodity('copper', start_date, end_date)),
        ('msci_em', lambda: load_msci_index('EM', start_date, end_date)),
        ('msci_world', lambda: load_msci_index('WORLD', start_date, end_date)),
    ]
    for name, loader in macro_loaders:
        try:
            data[name] = loader()
        except Exception as e:
            print(f"  [WARN] {name} 로드 실패: {e}")
            if qc is not None:
                qc.load_errors.append((name, str(e)))

    # 데이터 품질 검사
    if qc is not None:
        expected_sources = (
            [f'etf_{c}' for c in etf_codes]
            + ['kospi', 'kosdaq', 'snp500', 'nasdaq', 'dow']
            + [name for name, _ in macro_loaders]
        )
        qc.check_loaded_data(data, expected_sources)

    return data


# ============================================================
# 2. 피처 생성 함수
# ============================================================

def _safe_merge(base_df: pd.DataFrame, other_df: pd.DataFrame,
                prefix: str, cols: List[str]) -> pd.DataFrame:
    """date 기준 안전한 머지 (forward fill)"""
    if other_df is None or len(other_df) == 0:
        for c in cols:
            base_df[f'{prefix}_{c}'] = 0.0
        return base_df

    other = other_df[['date'] + [c for c in cols if c in other_df.columns]].copy()
    other.columns = ['date'] + [f'{prefix}_{c}' for c in cols if c in other_df.columns]

    # 누락 컬럼 보완
    for c in cols:
        col_name = f'{prefix}_{c}'
        if col_name not in other.columns:
            other[col_name] = 0.0

    base_df = pd.merge(base_df, other, on='date', how='left')
    # forward fill (주말/휴일 갭)
    for c in cols:
        col_name = f'{prefix}_{c}'
        if col_name in base_df.columns:
            base_df[col_name] = base_df[col_name].ffill().bfill().fillna(0)
    return base_df


def create_etf_features(df: pd.DataFrame, prefix: str = 'etf') -> pd.DataFrame:
    """ETF 자체 기술적 피처"""
    result = df.copy()

    close = result[f'{prefix}_close']
    open_ = result[f'{prefix}_open']
    high = result[f'{prefix}_high']
    low = result[f'{prefix}_low']
    volume = result[f'{prefix}_volume']

    # 수익률
    result[f'{prefix}_returns'] = close.pct_change().clip(-0.15, 0.15).fillna(0)
    result[f'{prefix}_log_returns'] = np.log(close / close.shift(1).clip(lower=1)).clip(-0.15, 0.15).fillna(0)

    # 일중 수익률(시가→종가): 타겟에 가장 직접적인 피처
    result[f'{prefix}_intraday_return'] = ((close - open_) / open_.clip(lower=1)).clip(-0.15, 0.15).fillna(0)

    # 갭 (전일종가→당일시가)
    result[f'{prefix}_gap'] = ((open_ - close.shift(1)) / close.shift(1).clip(lower=1)).clip(-0.1, 0.1).fillna(0)

    # 이동평균
    for w in [5, 10, 20, 60, 120]:
        ma = close.rolling(window=w, min_periods=1).mean()
        result[f'{prefix}_ma_{w}'] = ma
        result[f'{prefix}_price_to_ma_{w}'] = (close / ma.clip(lower=1)).clip(0.8, 1.2)

    # 모멘텀
    for w in [5, 10, 20, 60]:
        mom = close - close.shift(w)
        result[f'{prefix}_momentum_pct_{w}'] = (mom / close.shift(w).clip(lower=1)).clip(-0.3, 0.3)

    # RSI
    for w in [14, 30]:
        delta = close.diff().fillna(0)
        gain = delta.where(delta > 0, 0).rolling(window=w, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=w, min_periods=1).mean()
        rs = gain / loss.clip(lower=1e-10)
        result[f'{prefix}_rsi_{w}'] = (100 - (100 / (1 + rs))).clip(0, 100)

    # MACD
    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    result[f'{prefix}_macd'] = macd
    result[f'{prefix}_macd_signal'] = macd_signal
    result[f'{prefix}_macd_hist'] = macd - macd_signal

    # 볼린저밴드
    for w in [20]:
        rm = close.rolling(window=w, min_periods=1).mean()
        rs = close.rolling(window=w, min_periods=1).std().clip(lower=0.01)
        bb_upper = rm + rs * 2
        bb_lower = rm - rs * 2
        bb_range = (bb_upper - bb_lower).clip(lower=0.01)
        result[f'{prefix}_bb_width_{w}'] = (bb_range / rm.clip(lower=1)).clip(0, 0.3)
        result[f'{prefix}_bb_position_{w}'] = ((close - bb_lower) / bb_range).clip(0, 1)

    # 변동성
    for w in [5, 10, 20]:
        result[f'{prefix}_volatility_{w}'] = result[f'{prefix}_returns'].rolling(
            window=w, min_periods=1).std().clip(0, 0.15)

    # ATR (normalized)
    high_low = (high - low).clip(lower=0)
    high_close = np.abs(high - close.shift()).fillna(0)
    low_close = np.abs(low - close.shift()).fillna(0)
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    for w in [14]:
        atr = true_range.rolling(window=w, min_periods=1).mean()
        result[f'{prefix}_atr_{w}'] = (atr / close.clip(lower=1)).clip(0, 0.15)

    # ADX
    result[f'{prefix}_adx_14'] = _calculate_adx(high, low, close, 14) / 100.0

    # Stochastic
    for w in [14]:
        ll = low.rolling(window=w, min_periods=1).min()
        hh = high.rolling(window=w, min_periods=1).max()
        hl = (hh - ll).clip(lower=0.01)
        result[f'{prefix}_stoch_k_{w}'] = (100 * (close - ll) / hl).clip(0, 100)
        result[f'{prefix}_stoch_d_{w}'] = result[f'{prefix}_stoch_k_{w}'].rolling(
            window=3, min_periods=1).mean()

    # 거래량
    result[f'{prefix}_volume_change'] = volume.pct_change().fillna(0).clip(-0.9, 10)
    for w in [5, 20]:
        vm = volume.rolling(window=w, min_periods=1).mean().clip(lower=1)
        result[f'{prefix}_volume_ratio_{w}'] = (volume / vm).clip(0, 10)

    # OBV
    obv = [0.0]
    prices = close.values
    volumes = volume.clip(lower=0).values
    for i in range(1, len(result)):
        if prices[i] > prices[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    result[f'{prefix}_obv_raw'] = obv
    om = result[f'{prefix}_obv_raw'].rolling(window=60, min_periods=1).mean()
    os_ = result[f'{prefix}_obv_raw'].rolling(window=60, min_periods=1).std().clip(lower=1)
    result[f'{prefix}_obv'] = ((result[f'{prefix}_obv_raw'] - om) / os_).clip(-3, 3)

    # 가격 위치
    for w in [20, 60]:
        rmax = close.rolling(window=w, min_periods=1).max()
        rmin = close.rolling(window=w, min_periods=1).min()
        rr = (rmax - rmin).clip(lower=0.01)
        result[f'{prefix}_price_position_{w}'] = ((close - rmin) / rr).clip(0, 1)

    # Z-Score
    for w in [20, 60]:
        rm = close.rolling(window=w, min_periods=1).mean()
        rstd = close.rolling(window=w, min_periods=1).std().clip(lower=1e-6)
        result[f'{prefix}_z_score_{w}'] = ((close - rm) / rstd).clip(-4, 4)

    # 일중 범위
    h_l_range = (high - low).clip(lower=0)
    result[f'{prefix}_intraday_range_pct'] = (h_l_range / close.clip(lower=1)).clip(0, 0.15)

    # 연속 상승/하락일
    up = (result[f'{prefix}_returns'] > 0).astype(int)
    result[f'{prefix}_consecutive_up'] = up.groupby((up != up.shift()).cumsum()).cumsum()
    down = (result[f'{prefix}_returns'] < 0).astype(int)
    result[f'{prefix}_consecutive_down'] = down.groupby((down != down.shift()).cumsum()).cumsum()

    # ── Look-ahead bias 방지: 기술적 지표 1일 shift ──
    # Day trading (시초가 매수 → 종가 매도) 기준:
    #   매수 결정 시점(장 시작)에 알 수 있는 정보 = 전일 OHLCV + 당일 시가
    #   당일 종가/고가/저가/거래량 기반 지표는 사용 불가 → 1일 shift
    base_cols = {
        'date', f'{prefix}_open', f'{prefix}_high', f'{prefix}_low',
        f'{prefix}_close', f'{prefix}_volume', f'{prefix}_diff_ratio',
        f'{prefix}_gap',  # gap은 당일 시가 기반이므로 shift 불필요
    }
    shift_cols = [c for c in result.columns
                  if c.startswith(f'{prefix}_') and c not in base_cols]
    for col in shift_cols:
        result[col] = result[col].shift(1).fillna(0)

    # ── 당일 시가 기반 피처 (shift 불필요) ──
    # 시가 vs 전일 이동평균 (MA는 이미 1일 shift됨 = 전일 MA)
    for w in [5, 20, 60, 120]:
        ma_col = f'{prefix}_ma_{w}'
        if ma_col in result.columns:
            result[f'{prefix}_open_to_ma_{w}'] = (
                open_ / result[ma_col].clip(lower=1)
            ).clip(0.8, 1.2).fillna(1.0)

    # 시가 vs 전일 종가 (= gap과 동일하지만 비율 형태)
    result[f'{prefix}_open_return'] = (
        (open_ - close.shift(1)) / close.shift(1).clip(lower=1)
    ).clip(-0.1, 0.1).fillna(0)

    return result


def _calculate_adx(high, low, close, window=14):
    """ADX 계산"""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    atr = tr.rolling(window, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window, min_periods=1).mean() / atr.clip(lower=1e-6))
    minus_di = 100 * (minus_dm.rolling(window, min_periods=1).mean() / atr.clip(lower=1e-6))

    dx = (np.abs(plus_di - minus_di) / np.abs(plus_di + minus_di).clip(lower=1e-6)) * 100
    adx = dx.rolling(window, min_periods=1).mean()
    return adx.fillna(0).clip(0, 100)


def create_market_features(base_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """시장 피처: KOSPI/KOSDAQ 캔들 + 해외 시장 야간 수익률"""
    result = base_df.copy()

    # KOSPI
    if 'kospi' in data and len(data['kospi']) > 0:
        kospi = data['kospi'][['date', 'close', 'diff_ratio', 'volume', 'amount']].copy()
        kospi.columns = ['date', 'mkt_kospi_close', 'mkt_kospi_diff_ratio',
                         'mkt_kospi_volume', 'mkt_kospi_amount']
        result = pd.merge(result, kospi, on='date', how='left')
        for c in ['mkt_kospi_close', 'mkt_kospi_diff_ratio', 'mkt_kospi_volume', 'mkt_kospi_amount']:
            result[c] = result[c].ffill().bfill().fillna(0)

        # KOSPI 기술적 지표
        k_close = result['mkt_kospi_close']
        result['mkt_kospi_returns'] = k_close.pct_change().clip(-0.1, 0.1).fillna(0)
        for w in [5, 20, 60]:
            ma = k_close.rolling(window=w, min_periods=1).mean()
            result[f'mkt_kospi_to_ma_{w}'] = (k_close / ma.clip(lower=1)).clip(0.8, 1.2)
        result['mkt_kospi_rsi_14'] = _calc_rsi(k_close, 14)
        result['mkt_kospi_volatility_20'] = result['mkt_kospi_returns'].rolling(
            window=20, min_periods=1).std().clip(0, 0.1)

    # KOSDAQ
    if 'kosdaq' in data and len(data['kosdaq']) > 0:
        kosdaq = data['kosdaq'][['date', 'close', 'diff_ratio']].copy()
        kosdaq.columns = ['date', 'mkt_kosdaq_close', 'mkt_kosdaq_diff_ratio']
        result = pd.merge(result, kosdaq, on='date', how='left')
        for c in ['mkt_kosdaq_close', 'mkt_kosdaq_diff_ratio']:
            result[c] = result[c].ffill().bfill().fillna(0)
        result['mkt_kosdaq_returns'] = result['mkt_kosdaq_close'].pct_change().clip(-0.1, 0.1).fillna(0)

    # 해외 시장 (전일 수익률 → 당일 갭 선행지표)
    for code in ['snp500', 'nasdaq', 'dow']:
        if code in data and len(data[code]) > 0:
            fdf = data[code][['date', 'close', 'diff_ratio']].copy()
            fdf.columns = ['date', f'foreign_{code}_close', f'foreign_{code}_diff_ratio']
            result = pd.merge(result, fdf, on='date', how='left')
            for c in [f'foreign_{code}_close', f'foreign_{code}_diff_ratio']:
                result[c] = result[c].ffill().bfill().fillna(0)
            result[f'foreign_{code}_returns'] = result[f'foreign_{code}_close'].pct_change().clip(-0.1, 0.1).fillna(0)

    # ── Look-ahead bias 방지: 시장 지표 1일 shift ──
    # KOSPI/KOSDAQ/해외시장 종가 기반 지표는 장 시작 시 미확정
    mkt_shift_cols = [
        'mkt_kospi_returns', 'mkt_kospi_diff_ratio',
        'mkt_kospi_volume', 'mkt_kospi_amount',
        'mkt_kospi_rsi_14', 'mkt_kospi_volatility_20',
        'mkt_kosdaq_returns', 'mkt_kosdaq_diff_ratio',
    ]
    for w in [5, 20, 60]:
        mkt_shift_cols.append(f'mkt_kospi_to_ma_{w}')
    for code in ['snp500', 'nasdaq', 'dow']:
        mkt_shift_cols += [f'foreign_{code}_diff_ratio', f'foreign_{code}_returns']
    for col in mkt_shift_cols:
        if col in result.columns:
            result[col] = result[col].shift(1).fillna(0)

    return result


def create_macro_features(base_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """매크로 피처: 환율, 금리, 변동성, 센티먼트"""
    result = base_df.copy()

    # 환율 (USD/KRW)
    if 'fx_usdkrw' in data and len(data['fx_usdkrw']) > 0:
        fx = data['fx_usdkrw'][['date', 'base', 'diff_ratio']].copy()
        fx.columns = ['date', 'fx_usdkrw', 'fx_usdkrw_diff_ratio']
        result = pd.merge(result, fx, on='date', how='left')
        for c in ['fx_usdkrw', 'fx_usdkrw_diff_ratio']:
            result[c] = result[c].ffill().bfill().fillna(0)
        result['fx_usdkrw_returns'] = result['fx_usdkrw'].pct_change().clip(-0.05, 0.05).fillna(0)
        for w in [5, 20]:
            ma = result['fx_usdkrw'].rolling(window=w, min_periods=1).mean()
            result[f'fx_usdkrw_to_ma_{w}'] = (result['fx_usdkrw'] / ma.clip(lower=1)).clip(0.9, 1.1)

    # VIX
    if 'vix' in data and len(data['vix']) > 0:
        result = _safe_merge(result, data['vix'], 'vix', ['close', 'diff_ratio'])
        result['vix_level'] = result['vix_close'].clip(0, 80)
        result['vix_returns'] = result['vix_close'].pct_change().clip(-0.5, 0.5).fillna(0)

    # KOSPI VIX
    if 'kospi_vix' in data and len(data['kospi_vix']) > 0:
        result = _safe_merge(result, data['kospi_vix'], 'kvix', ['close', 'diff_ratio'])
        result['kvix_level'] = result['kvix_close'].clip(0, 80)
        result['kvix_returns'] = result['kvix_close'].pct_change().clip(-0.5, 0.5).fillna(0)

    # 미국 10년 국채
    if 'us_bond_10y' in data and len(data['us_bond_10y']) > 0:
        result = _safe_merge(result, data['us_bond_10y'], 'us10y', ['close', 'diff'])
        result['us10y_level'] = result['us10y_close']
        result['us10y_change'] = result['us10y_close'].diff().fillna(0).clip(-0.2, 0.2)

    # 한국 3년 국채
    if 'kr_bond_3y' in data and len(data['kr_bond_3y']) > 0:
        result = _safe_merge(result, data['kr_bond_3y'], 'kr3y', ['close', 'diff'])
        result['kr3y_level'] = result['kr3y_close']
        result['kr3y_change'] = result['kr3y_close'].diff().fillna(0).clip(-0.2, 0.2)

    # 금리차 (미국 10Y - 한국 3Y)
    if 'us10y_level' in result.columns and 'kr3y_level' in result.columns:
        result['yield_spread'] = result['us10y_level'] - result['kr3y_level']

    # CNN Fear & Greed
    if 'cnn_fear_greed' in data and len(data['cnn_fear_greed']) > 0:
        result = _safe_merge(result, data['cnn_fear_greed'], 'fear_greed', ['value'])
        result['fear_greed_level'] = result['fear_greed_value'].clip(0, 100)

    # KOSPI PER/PBR
    if 'kospi_per_pbr' in data and len(data['kospi_per_pbr']) > 0:
        ppdf = data['kospi_per_pbr']
        if 'per' in ppdf.columns and 'pbr' in ppdf.columns:
            pp = ppdf[['date', 'per', 'pbr']].copy()
            pp.columns = ['date', 'kospi_per', 'kospi_pbr']
            result = pd.merge(result, pp, on='date', how='left')
            result['kospi_per'] = result['kospi_per'].ffill().bfill().fillna(0)
            result['kospi_pbr'] = result['kospi_pbr'].ffill().bfill().fillna(0)

    # Citi Surprise
    if 'citi_surprise_kr' in data and len(data['citi_surprise_kr']) > 0:
        result = _safe_merge(result, data['citi_surprise_kr'], 'citi_kr', ['value'])

    # ── Look-ahead bias 방지: 매크로 지표 1일 shift ──
    macro_shift_cols = [
        'fx_usdkrw_returns', 'fx_usdkrw_diff_ratio',
        'vix_level', 'vix_returns', 'vix_diff_ratio',
        'kvix_level', 'kvix_returns', 'kvix_diff_ratio',
        'us10y_level', 'us10y_change', 'us10y_diff',
        'kr3y_level', 'kr3y_change', 'kr3y_diff',
        'yield_spread',
        'fear_greed_level', 'fear_greed_value',
        'kospi_per', 'kospi_pbr',
        'citi_kr_value',
    ]
    for w in [5, 20]:
        macro_shift_cols.append(f'fx_usdkrw_to_ma_{w}')
    for col in macro_shift_cols:
        if col in result.columns:
            result[col] = result[col].shift(1).fillna(0)

    return result


def create_sector_features(base_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """섹터/원자재 피처: SOX, GSCI, DX"""
    result = base_df.copy()

    # SOX (반도체)
    if 'sox' in data and len(data['sox']) > 0:
        result = _safe_merge(result, data['sox'], 'sox', ['close', 'diff_ratio'])
        result['sox_returns'] = result['sox_close'].pct_change().clip(-0.1, 0.1).fillna(0)
        for w in [5, 20]:
            ma = result['sox_close'].rolling(window=w, min_periods=1).mean()
            result[f'sox_to_ma_{w}'] = (result['sox_close'] / ma.clip(lower=1)).clip(0.8, 1.2)

    # GSCI (원자재)
    if 'gsci' in data and len(data['gsci']) > 0:
        result = _safe_merge(result, data['gsci'], 'gsci', ['close', 'diff_ratio'])
        result['gsci_returns'] = result['gsci_close'].pct_change().clip(-0.1, 0.1).fillna(0)

    # DX (달러 인덱스)
    if 'dx' in data and len(data['dx']) > 0:
        result = _safe_merge(result, data['dx'], 'dx', ['close', 'diff_ratio'])
        result['dx_returns'] = result['dx_close'].pct_change().clip(-0.05, 0.05).fillna(0)

    # ── Look-ahead bias 방지: 섹터/원자재 지표 1일 shift ──
    sector_shift_cols = [
        'sox_returns', 'sox_diff_ratio',
        'gsci_returns', 'gsci_diff_ratio',
        'dx_returns', 'dx_diff_ratio',
    ]
    for w in [5, 20]:
        sector_shift_cols.append(f'sox_to_ma_{w}')
    for col in sector_shift_cols:
        if col in result.columns:
            result[col] = result[col].shift(1).fillna(0)

    return result


def create_futures_features(base_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """선물 피처: 선물-현물 베이시스, 수익률, 거래량비율"""
    result = base_df.copy()

    if 'futures' not in data or len(data['futures']) == 0:
        for col in ['futures_returns', 'futures_basis', 'futures_volume_ratio']:
            result[col] = 0.0
        return result

    fdf = data['futures'][['date', 'close', 'volume']].copy()
    fdf.columns = ['date', 'futures_close', 'futures_volume']
    result = pd.merge(result, fdf, on='date', how='left')
    result['futures_close'] = result['futures_close'].ffill().bfill().fillna(0)
    result['futures_volume'] = result['futures_volume'].ffill().bfill().fillna(0)

    # 선물 수익률
    result['futures_returns'] = result['futures_close'].pct_change().clip(-0.1, 0.1).fillna(0)

    # 베이시스: 선물 / KOSPI (현물 대용)
    if 'mkt_kospi_close' in result.columns:
        kospi = result['mkt_kospi_close'].clip(lower=1)
        result['futures_basis'] = (result['futures_close'] / kospi - 1).clip(-0.05, 0.05).fillna(0)
    else:
        result['futures_basis'] = 0.0

    # 선물/현물 거래량 비율
    if 'mkt_kospi_volume' in result.columns:
        mkt_vol = result['mkt_kospi_volume'].clip(lower=1)
        result['futures_volume_ratio'] = (result['futures_volume'] / mkt_vol).clip(0, 10).fillna(0)
    else:
        result['futures_volume_ratio'] = 0.0

    result.drop(columns=['futures_close', 'futures_volume'], inplace=True, errors='ignore')

    # Look-ahead bias 방지: 1일 shift
    for col in ['futures_returns', 'futures_basis', 'futures_volume_ratio']:
        result[col] = result[col].shift(1).fillna(0)

    return result


def create_program_features(base_df: pd.DataFrame, data: dict,
                            etf_code: str) -> pd.DataFrame:
    """프로그램 매매 피처: 순매수, 순매수 증감"""
    result = base_df.copy()
    key = f'program_{etf_code}'

    if key not in data or len(data[key]) == 0:
        for col in ['program_net_buy', 'program_net_buy_change']:
            result[col] = 0.0
        return result

    pdf = data[key][['date', 'net_buy_amount', 'net_buy_change_amount', 'volume']].copy()
    vol = pdf['volume'].clip(lower=1)
    pdf['program_net_buy'] = (pdf['net_buy_amount'] / vol).fillna(0)
    pdf['program_net_buy_change'] = (pdf['net_buy_change_amount'] / vol).fillna(0)
    pdf = pdf[['date', 'program_net_buy', 'program_net_buy_change']]

    result = pd.merge(result, pdf, on='date', how='left')
    for col in ['program_net_buy', 'program_net_buy_change']:
        result[col] = result[col].ffill().bfill().fillna(0)
        result[col] = result[col].shift(1).fillna(0)

    return result


def create_investor_features(base_df: pd.DataFrame, data: dict,
                             etf_code: str) -> pd.DataFrame:
    """투자자 수급 피처: 외국인/기관 순매매, 외국인-개인 차이"""
    result = base_df.copy()
    key = f'investor_{etf_code}'

    if key not in data or len(data[key]) == 0:
        for col in ['inv_foreign_net', 'inv_inst_net', 'inv_ind_foreign_diff']:
            result[col] = 0.0
        return result

    idf = data[key][['date', 'frgn', 'inst', 'ind', 'acc_volume']].copy()
    vol = idf['acc_volume'].clip(lower=1)
    idf['inv_foreign_net'] = (idf['frgn'].fillna(0) / vol).clip(-1, 1)
    idf['inv_inst_net'] = (idf['inst'].fillna(0) / vol).clip(-1, 1)
    idf['inv_ind_foreign_diff'] = ((idf['ind'].fillna(0) - idf['frgn'].fillna(0)) / vol).clip(-1, 1)
    idf = idf[['date', 'inv_foreign_net', 'inv_inst_net', 'inv_ind_foreign_diff']]

    result = pd.merge(result, idf, on='date', how='left')
    for col in ['inv_foreign_net', 'inv_inst_net', 'inv_ind_foreign_diff']:
        result[col] = result[col].ffill().bfill().fillna(0)
        result[col] = result[col].shift(1).fillna(0)

    return result


def create_short_features(base_df: pd.DataFrame, data: dict,
                          etf_code: str) -> pd.DataFrame:
    """공매도 피처: 시장 공매도 거래량 비율, 종목 공매도 잔고"""
    result = base_df.copy()

    # 시장 공매도 거래량 비율 (KOSPI 기준)
    if 'short_volume' in data and len(data['short_volume']) > 0:
        svdf = data['short_volume'].copy()
        if 'short_vol_전체' in svdf.columns:
            total = svdf['short_vol_전체'].clip(lower=1)
            # 외국인 공매도 비율
            frgn_col = [c for c in svdf.columns if '외국인' in c]
            if frgn_col:
                svdf['mkt_short_volume_ratio'] = (svdf[frgn_col[0]] / total).clip(0, 1)
            else:
                svdf['mkt_short_volume_ratio'] = 0.0
            svdf['mkt_short_change'] = svdf['mkt_short_volume_ratio'].pct_change().clip(-1, 1).fillna(0)
            svdf = svdf[['date', 'mkt_short_volume_ratio', 'mkt_short_change']]
            result = pd.merge(result, svdf, on='date', how='left')
            for col in ['mkt_short_volume_ratio', 'mkt_short_change']:
                result[col] = result[col].ffill().bfill().fillna(0)
                result[col] = result[col].shift(1).fillna(0)
        else:
            result['mkt_short_volume_ratio'] = 0.0
            result['mkt_short_change'] = 0.0
    else:
        result['mkt_short_volume_ratio'] = 0.0
        result['mkt_short_change'] = 0.0

    # 종목 공매도 잔고
    key = f'short_bal_{etf_code}'
    if key in data and len(data[key]) > 0:
        sbdf = data[key][['date', 'bal_rto']].copy()
        sbdf.columns = ['date', 'short_balance_ratio']
        sbdf['short_bal_change'] = sbdf['short_balance_ratio'].pct_change().clip(-1, 1).fillna(0)
        result = pd.merge(result, sbdf, on='date', how='left')
        for col in ['short_balance_ratio', 'short_bal_change']:
            result[col] = result[col].ffill().bfill().fillna(0)
            result[col] = result[col].shift(1).fillna(0)
    else:
        result['short_balance_ratio'] = 0.0
        result['short_bal_change'] = 0.0

    return result


def create_global_features(base_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """글로벌 피처: BDI, MSCI EM/WORLD"""
    result = base_df.copy()

    # BDI
    if 'bdi' in data and len(data['bdi']) > 0:
        bdf = data['bdi'][['date', 'value']].copy()
        bdf.columns = ['date', 'bdi_value']
        result = pd.merge(result, bdf, on='date', how='left')
        result['bdi_value'] = result['bdi_value'].ffill().bfill().fillna(0)
        result['bdi_returns'] = result['bdi_value'].pct_change().clip(-0.2, 0.2).fillna(0)
        bdi_ma20 = result['bdi_value'].rolling(window=20, min_periods=1).mean().clip(lower=1)
        result['bdi_to_ma_20'] = (result['bdi_value'] / bdi_ma20).clip(0.5, 2.0).fillna(1.0)
        result.drop(columns=['bdi_value'], inplace=True)
    else:
        result['bdi_returns'] = 0.0
        result['bdi_to_ma_20'] = 0.0

    # SCFI (주간 데이터 → ffill로 일간 보간)
    if 'scfi' in data and len(data['scfi']) > 0:
        sdf = data['scfi'][['date', 'value']].copy()
        sdf.columns = ['date', 'scfi_value']
        result = pd.merge(result, sdf, on='date', how='left')
        result['scfi_value'] = result['scfi_value'].ffill().bfill().fillna(0)
        result['scfi_returns'] = result['scfi_value'].pct_change().clip(-0.3, 0.3).fillna(0)
        scfi_ma20 = result['scfi_value'].rolling(window=20, min_periods=1).mean().clip(lower=1)
        result['scfi_to_ma_20'] = (result['scfi_value'] / scfi_ma20).clip(0.5, 2.0).fillna(1.0)
        result.drop(columns=['scfi_value'], inplace=True)
    else:
        result['scfi_returns'] = 0.0
        result['scfi_to_ma_20'] = 0.0

    # WTI
    if 'wti' in data and len(data['wti']) > 0:
        wdf = data['wti'][['date', 'value']].copy()
        wdf.columns = ['date', 'wti_value']
        result = pd.merge(result, wdf, on='date', how='left')
        result['wti_value'] = result['wti_value'].ffill().bfill().fillna(0)
        result['wti_returns'] = result['wti_value'].pct_change().clip(-0.15, 0.15).fillna(0)
        wti_ma20 = result['wti_value'].rolling(window=20, min_periods=1).mean().clip(lower=0.01)
        result['wti_to_ma_20'] = (result['wti_value'] / wti_ma20).clip(0.5, 2.0).fillna(1.0)
        result.drop(columns=['wti_value'], inplace=True)
    else:
        result['wti_returns'] = 0.0
        result['wti_to_ma_20'] = 0.0

    # 주요 원자재: 금, 은, 철광석, 구리
    commodity_map = [
        ('gold', 0.1),      # 금
        ('silver', 0.15),    # 은
        ('iron', 0.15),      # 철광석
        ('copper', 0.1),     # 구리
    ]
    for cname, clip_val in commodity_map:
        if cname in data and len(data[cname]) > 0:
            cdf = data[cname][['date', 'value']].copy()
            cdf.columns = ['date', f'{cname}_value']
            result = pd.merge(result, cdf, on='date', how='left')
            result[f'{cname}_value'] = result[f'{cname}_value'].ffill().bfill().fillna(0)
            result[f'{cname}_returns'] = result[f'{cname}_value'].pct_change().clip(-clip_val, clip_val).fillna(0)
            cma = result[f'{cname}_value'].rolling(window=20, min_periods=1).mean().clip(lower=0.01)
            result[f'{cname}_to_ma_20'] = (result[f'{cname}_value'] / cma).clip(0.5, 2.0).fillna(1.0)
            result.drop(columns=[f'{cname}_value'], inplace=True)
        else:
            result[f'{cname}_returns'] = 0.0
            result[f'{cname}_to_ma_20'] = 0.0

    # MSCI EM
    if 'msci_em' in data and len(data['msci_em']) > 0:
        em = data['msci_em'][['date', 'value']].copy()
        em.columns = ['date', 'msci_em_value']
        result = pd.merge(result, em, on='date', how='left')
        result['msci_em_value'] = result['msci_em_value'].ffill().bfill().fillna(0)
        result['msci_em_returns'] = result['msci_em_value'].pct_change().clip(-0.1, 0.1).fillna(0)
        result.drop(columns=['msci_em_value'], inplace=True)
    else:
        result['msci_em_returns'] = 0.0

    # MSCI WORLD
    if 'msci_world' in data and len(data['msci_world']) > 0:
        wdf = data['msci_world'][['date', 'value']].copy()
        wdf.columns = ['date', 'msci_world_value']
        result = pd.merge(result, wdf, on='date', how='left')
        result['msci_world_value'] = result['msci_world_value'].ffill().bfill().fillna(0)
        result['msci_world_returns'] = result['msci_world_value'].pct_change().clip(-0.1, 0.1).fillna(0)
        result.drop(columns=['msci_world_value'], inplace=True)
    else:
        result['msci_world_returns'] = 0.0

    # Look-ahead bias 방지: 1일 shift
    for col in ['bdi_returns', 'bdi_to_ma_20', 'scfi_returns', 'scfi_to_ma_20',
                'wti_returns', 'wti_to_ma_20',
                'gold_returns', 'gold_to_ma_20',
                'silver_returns', 'silver_to_ma_20',
                'iron_returns', 'iron_to_ma_20',
                'copper_returns', 'copper_to_ma_20',
                'msci_em_returns', 'msci_world_returns']:
        if col in result.columns:
            result[col] = result[col].shift(1).fillna(0)

    return result


def create_stock_credit_features(base_df: pd.DataFrame,
                                  credit_df: pd.DataFrame) -> pd.DataFrame:
    """신용 거래 피처 (융자 기준)

    파생 피처:
    - credit_balance_ratio   : 잔고율 (공시값)
    - credit_supply_ratio    : 공여율 (공시값)
    - credit_net_new         : 순신규 = new - repayment
    - credit_new_ratio       : 신규비율 = new / balance
    - credit_repayment_ratio : 상환비율 = repayment / balance
    - credit_net_ratio       : 잔고 변동률 = balance_diff / prev_balance
    - credit_to_volume       : 신용집중도 = balance / volume
    - credit_cost_ratio      : 단가비율 = (amount / balance) / close  (1 이상이면 물려있음)
    - credit_balance_ma_5/20 : 잔고 이동평균 대비 비율
    - credit_balance_mom_5/20: 잔고 모멘텀 (5/20일 변화율)
    - credit_ratio_ma_20     : 잔고율 20일 이동평균 대비
    """
    result = base_df.copy()

    if credit_df is None or len(credit_df) == 0:
        for col in [
            'credit_balance_ratio', 'credit_supply_ratio',
            'credit_net_new', 'credit_new_ratio', 'credit_repayment_ratio',
            'credit_net_ratio', 'credit_to_volume', 'credit_cost_ratio',
            'credit_balance_ma_5', 'credit_balance_ma_20',
            'credit_balance_mom_5', 'credit_balance_mom_20',
            'credit_ratio_ma_20',
        ]:
            result[col] = 0.0
        return result

    cdf = credit_df[['date', 'close', 'new', 'repayment', 'balance',
                      'amount', 'balance_diff', 'volume',
                      'supply_ratio', 'balance_ratio']].copy()

    # 파생 컬럼 계산
    bal = cdf['balance'].clip(lower=1)
    prev_bal = bal.shift(1).clip(lower=1)

    cdf['credit_balance_ratio']   = cdf['balance_ratio'].fillna(0)
    cdf['credit_supply_ratio']    = cdf['supply_ratio'].fillna(0)
    cdf['credit_net_new']         = (cdf['new'].fillna(0) - cdf['repayment'].fillna(0)) / bal
    cdf['credit_new_ratio']       = cdf['new'].fillna(0) / bal
    cdf['credit_repayment_ratio'] = cdf['repayment'].fillna(0) / bal
    cdf['credit_net_ratio']       = cdf['balance_diff'].fillna(0) / prev_bal
    cdf['credit_to_volume']       = bal / cdf['volume'].clip(lower=1)
    # 단가비율: 평균취득단가 / 현재가 (1 초과 = 평균매수가 > 현재가, 손실 중)  
    cost_per_share = cdf['amount'].fillna(0) / bal
    cdf['credit_cost_ratio']      = cost_per_share / cdf['close'].clip(lower=1)

    # 잔고 이동평균 대비
    for w in [5, 20]:
        ma = bal.rolling(window=w, min_periods=1).mean().clip(lower=1)
        cdf[f'credit_balance_ma_{w}'] = (bal / ma).clip(0.5, 2.0)

    # 잔고 모멘텀
    for w in [5, 20]:
        shifted = bal.shift(w).clip(lower=1)
        cdf[f'credit_balance_mom_{w}'] = ((bal - shifted) / shifted).clip(-1, 1)

    # 잔고율 이동평균 대비
    br = cdf['credit_balance_ratio']
    br_ma20 = br.rolling(window=20, min_periods=1).mean()
    cdf['credit_ratio_ma_20'] = (br - br_ma20).clip(-5, 5)

    merge_cols = [
        'date',
        'credit_balance_ratio', 'credit_supply_ratio',
        'credit_net_new', 'credit_new_ratio', 'credit_repayment_ratio',
        'credit_net_ratio', 'credit_to_volume', 'credit_cost_ratio',
        'credit_balance_ma_5', 'credit_balance_ma_20',
        'credit_balance_mom_5', 'credit_balance_mom_20',
        'credit_ratio_ma_20',
    ]
    cdf = cdf[merge_cols]

    result = pd.merge(result, cdf, on='date', how='left')
    credit_feature_cols = [c for c in merge_cols if c != 'date']
    for col in credit_feature_cols:
        result[col] = result[col].ffill().bfill().fillna(0)

    # ── Look-ahead bias 방지: 신용거래 지표 1일 shift ──
    # 신용거래 데이터는 당일 장 마감 후 확정 → 익일부터 사용 가능
    for col in credit_feature_cols:
        if col in result.columns:
            result[col] = result[col].shift(1).fillna(0)

    return result


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 관련 피처"""
    result = df.copy()

    # date 파싱
    dates = pd.to_datetime(result['date'], format='%Y%m%d')
    result['day_of_week'] = dates.dt.dayofweek  # 0=Mon, 4=Fri
    result['month'] = dates.dt.month
    result['day_of_month'] = dates.dt.day

    # 순환 인코딩
    result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 5)
    result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 5)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

    # 월초/월말 효과
    result['is_month_start'] = (result['day_of_month'] <= 5).astype(float)
    result['is_month_end'] = (result['day_of_month'] >= 25).astype(float)

    # 요일별 더미 (월요일 효과, 금요일 효과)
    result['is_monday'] = (result['day_of_week'] == 0).astype(float)
    result['is_friday'] = (result['day_of_week'] == 4).astype(float)

    return result


def _calc_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = prices.diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.clip(lower=1e-10)
    return (100 - (100 / (1 + rs))).clip(0, 100)


# ============================================================
# 3. 데이터 정제
# ============================================================

def clip_outliers_iqr(series: pd.Series, multiplier: float = 5) -> pd.Series:
    """IQR 기반 이상치 클리핑"""
    if series.std() == 0:
        return series
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        return series
    return series.clip(Q1 - multiplier * IQR, Q3 + multiplier * IQR)


def clean_and_clip_outliers(df: pd.DataFrame, iqr_multiplier: float = 5,
                            warm_up_period: int = 60, verbose: bool = False) -> pd.DataFrame:
    """데이터 정제: Inf 제거 → IQR 클리핑 → warm-up 제거 → NaN 처리"""
    result = df.copy()
    result.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        result[col] = clip_outliers_iqr(result[col], multiplier=iqr_multiplier)

    if len(result) > warm_up_period:
        result = result.iloc[warm_up_period:].copy()

    result = result.ffill().bfill().fillna(0)

    if verbose:
        print(f"  데이터 정제 완료: {len(result)} 행 (warm-up {warm_up_period}행 제거)")
    return result


# ============================================================
# 4. 피처 선택 및 스케일링
# ============================================================

def get_selected_features(prefix: str = 'etf', include_meta: bool = False) -> List[str]:
    """학습에 사용할 핵심 피처 리스트"""
    features = [
        # -- ETF 자체 (전일 데이터 기반, 1일 shift 적용됨) --
        f'{prefix}_returns', f'{prefix}_log_returns',
        f'{prefix}_gap',  # 당일 시가 기반 (shift 불필요)
        f'{prefix}_open_return',  # 당일 시가 vs 전일 종가
        f'{prefix}_open_to_ma_5', f'{prefix}_open_to_ma_20',  # 당일 시가 vs 전일 MA
        f'{prefix}_open_to_ma_60', f'{prefix}_open_to_ma_120',
        f'{prefix}_price_to_ma_5', f'{prefix}_price_to_ma_20',
        f'{prefix}_price_to_ma_60', f'{prefix}_price_to_ma_120',
        f'{prefix}_momentum_pct_5', f'{prefix}_momentum_pct_10',
        f'{prefix}_momentum_pct_20', f'{prefix}_momentum_pct_60',
        f'{prefix}_rsi_14',
        f'{prefix}_macd_hist',
        f'{prefix}_bb_width_20', f'{prefix}_bb_position_20',
        f'{prefix}_volatility_5', f'{prefix}_volatility_20',
        f'{prefix}_atr_14', f'{prefix}_adx_14',
        f'{prefix}_stoch_k_14', f'{prefix}_stoch_d_14',
        f'{prefix}_volume_change', f'{prefix}_volume_ratio_5', f'{prefix}_volume_ratio_20',
        f'{prefix}_obv',
        f'{prefix}_price_position_20', f'{prefix}_price_position_60',
        f'{prefix}_z_score_20', f'{prefix}_z_score_60',
        f'{prefix}_intraday_range_pct',
        f'{prefix}_consecutive_up', f'{prefix}_consecutive_down',

        # -- 시장 --
        'mkt_kospi_returns', 'mkt_kospi_diff_ratio',
        'mkt_kospi_to_ma_5', 'mkt_kospi_to_ma_20', 'mkt_kospi_to_ma_60',
        'mkt_kospi_rsi_14', 'mkt_kospi_volatility_20',
        'mkt_kosdaq_returns',

        # -- 해외 시장 (야간 수익률 = 당일 갭 선행지표) --
        'foreign_snp500_returns', 'foreign_nasdaq_returns', 'foreign_dow_returns',

        # -- 환율/금리 --
        'fx_usdkrw_returns', 'fx_usdkrw_to_ma_5', 'fx_usdkrw_to_ma_20',
        'us10y_level', 'us10y_change',
        'kr3y_level', 'kr3y_change',
        'yield_spread',

        # -- 변동성/심리 --
        'vix_level', 'vix_returns',
        'kvix_level', 'kvix_returns',
        'fear_greed_level',

        # -- 섹터/원자재 --
        'sox_returns', 'sox_to_ma_5', 'sox_to_ma_20',
        'gsci_returns',
        'dx_returns',

        # -- 밸류에이션 --
        'kospi_per', 'kospi_pbr',
        'citi_kr_value',

        # -- 신용 거래 (융자) --
        'credit_balance_ratio', 'credit_supply_ratio',
        'credit_net_new', 'credit_new_ratio', 'credit_repayment_ratio',
        'credit_net_ratio', 'credit_to_volume', 'credit_cost_ratio',
        'credit_balance_ma_5', 'credit_balance_ma_20',
        'credit_balance_mom_5', 'credit_balance_mom_20',
        'credit_ratio_ma_20',

        # -- 선물 --
        'futures_returns', 'futures_basis', 'futures_volume_ratio',

        # -- 프로그램 매매 --
        'program_net_buy', 'program_net_buy_change',

        # -- 투자자 수급 --
        'inv_foreign_net', 'inv_inst_net', 'inv_ind_foreign_diff',

        # -- 공매도 --
        'mkt_short_volume_ratio', 'mkt_short_change',
        'short_balance_ratio', 'short_bal_change',

        # -- 글로벌 (BDI, MSCI) --
        'bdi_returns', 'bdi_to_ma_20',
        'scfi_returns', 'scfi_to_ma_20',
        'wti_returns', 'wti_to_ma_20',
        'gold_returns', 'gold_to_ma_20',
        'silver_returns', 'silver_to_ma_20',
        'iron_returns', 'iron_to_ma_20',
        'copper_returns', 'copper_to_ma_20',
        'msci_em_returns', 'msci_world_returns',

        # -- 시간 --
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'is_month_start', 'is_month_end', 'is_monday', 'is_friday',
    ]

    # 메타 피처 (일반화 모델용)
    if include_meta:
        features += get_meta_feature_names()

    return features


def select_features(df: pd.DataFrame, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
    """선택된 피처만 추출"""
    if feature_list is None:
        feature_list = get_selected_features()
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  [INFO] 피처 누락 {len(missing)}개: {missing[:10]}...")
        # 누락 피처 0으로 채움
        for f in missing:
            df[f] = 0.0
        available = feature_list
    return df[available].copy()


def remove_zero_variance(df: pd.DataFrame, threshold: float = 1e-10,
                         verbose: bool = False) -> pd.DataFrame:
    """분산이 0에 가까운 피처 제거"""
    to_drop = [c for c in df.columns if df[c].std() < threshold]
    if to_drop and verbose:
        print(f"  분산 ~ 0 피처 {len(to_drop)}개 제거: {to_drop}")
    return df.drop(columns=to_drop)


def scale_features(df: pd.DataFrame, scaler_path: str,
                   clip_range: float = 5.0) -> Tuple[pd.DataFrame, StandardScaler]:
    """StandardScaler 적용 + 클리핑 + 스케일러 저장"""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    scaled = np.clip(scaled, -clip_range, clip_range)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return pd.DataFrame(scaled, columns=df.columns), scaler


# ============================================================
# 5. 통합 파이프라인
# ============================================================

def build_single_etf_features(etf_code: str, data: dict,
                              etf_name: str = '', etf_extra: dict = None,
                              include_meta: bool = False,
                              verbose: bool = False,
                              qc: DataQualityReport = None) -> pd.DataFrame:
    """단일 ETF에 대한 전체 피처 빌드"""
    etf_df = data[f'etf_{etf_code}'].copy()
    if len(etf_df) == 0:
        raise ValueError(f"ETF {etf_code} 데이터가 비어있습니다.")

    # ETF 캔들 컬럼 표준화
    prefix = 'etf'
    etf_df = etf_df.rename(columns={
        'open': f'{prefix}_open', 'high': f'{prefix}_high',
        'low': f'{prefix}_low', 'close': f'{prefix}_close',
        'volume': f'{prefix}_volume', 'diff_ratio': f'{prefix}_diff_ratio',
    })

    # 1. ETF 자체 피처
    result = create_etf_features(etf_df, prefix)
    if qc is not None:
        qc.check_dataframe(result, f'{etf_code}/etf_features')

    # 2. 시장 피처
    result = create_market_features(result, data)
    if qc is not None:
        qc.check_dataframe(result, f'{etf_code}/market_features')

    # 3. 매크로 피처
    result = create_macro_features(result, data)
    if qc is not None:
        qc.check_dataframe(result, f'{etf_code}/macro_features')

    # 4. 섹터/원자재 피처
    result = create_sector_features(result, data)

    # 5. 신용 거래 피처 (융자)
    credit_df = data.get(f'credit_{etf_code}')
    result = create_stock_credit_features(result, credit_df)

    # 6. 선물 피처
    result = create_futures_features(result, data)

    # 7. 프로그램 매매 피처
    result = create_program_features(result, data, etf_code)

    # 8. 투자자 수급 피처
    result = create_investor_features(result, data, etf_code)

    # 9. 공매도 피처
    result = create_short_features(result, data, etf_code)

    # 10. 글로벌 피처 (BDI, MSCI)
    result = create_global_features(result, data)

    # 11. 시간 피처
    result = create_time_features(result)

    # 12. 메타 피처 (일반화 모델용)
    if include_meta and etf_name:
        result = create_etf_meta_features(result, etf_name, etf_extra)

    return result



# ============================================================
# DB 저장
# ============================================================

def save_feature_vectors_to_db(
    n: int = 20,
    start_date: str = None,
    end_date: str = None,
    save_start_date: str = None,
    version: str = "1",
    etf_code: str = None,
    min_candles: int = 500,
    truncate: bool = False,
) -> int:
    """피처 벡터를 빌드하여 quantylab DB(FeatureVector 테이블)에 저장합니다.

    quantylab 패키지가 설치된 main env에서 호출해야 합니다.

    Args:
        n: 저장할 최근 일수 (기본 20). save_start_date 미지정 시 사용.
        start_date: 데이터 로드 시작일 (미지정 시 n + 180일 전)
        end_date: 데이터 로드 종료일 (기본: 오늘)
        save_start_date: 저장 시작일 (YYYYMMDD). 지정 시 n 무시. 전체 이력 생성 시 사용.
        version: 피처 벡터 버전 (기본 "1")
        etf_code: 특정 ETF 코드 (None이면 전체 TIGER ETF)
        min_candles: 최소 캔들 수
        truncate: True이면 저장 전 FeatureVector 테이블 전체 삭제

    Returns:
        저장된 총 행 수
    """
    from sqlalchemy import text

    required_trading_days = max(MIN_FEATURE_ROWS + OUTLIER_WARMUP_PERIOD, n + OUTLIER_WARMUP_PERIOD)
    load_buffer_days = int(required_trading_days * TRADING_TO_CALENDAR_FACTOR) + LOAD_BUFFER_MARGIN_DAYS

    today = datetime.now()
    if not end_date:
        end_date = today.strftime('%Y%m%d')

    if save_start_date:
        # 명시적 저장 시작일: start_date도 미지정이면 warm-up 자동 계산
        if not start_date:
            save_dt = datetime.strptime(save_start_date, '%Y%m%d')
            start_date = (save_dt - timedelta(days=load_buffer_days)).strftime('%Y%m%d')
    else:
        save_start_date = (today - timedelta(days=n)).strftime('%Y%m%d')
        if not start_date:
            load_start = today - timedelta(days=load_buffer_days)
            start_date = load_start.strftime('%Y%m%d')

    if truncate:
        print("FeatureVector 테이블 초기화 중...")
        with psql.get_session() as session:
            session.execute(text("TRUNCATE TABLE feature_vector"))
            session.commit()
        print("  완료")

    print(f"데이터 로드: {start_date} ~ {end_date}, 저장: {save_start_date} ~")

    if etf_code:
        etf_codes = [etf_code]
        etf_info = {etf_code: {'name': '', 'extra': {}}}
    else:
        etf_list = load_tiger_etf_list(min_candles=min_candles)
        etf_codes = [e['code'] for e in etf_list]
        etf_info = {e['code']: e for e in etf_list}

    print(f"피처 벡터 빌드: {len(etf_codes)}개 ETF, version={version}")

    data = load_all_data(etf_codes, start_date, end_date)

    total = 0
    for i, code in enumerate(etf_codes):
        info = etf_info.get(code, {})
        etf_name = info.get('name', '')
        etf_extra = info.get('extra', {})
        print(f"[{i+1}/{len(etf_codes)}] {code} ({etf_name})")

        qc = DataQualityReport()
        try:
            feature_df = build_single_etf_features(
                code, data,
                etf_name=etf_name, etf_extra=etf_extra,
                include_meta=True, verbose=False, qc=qc,
            )
        except Exception as e:
            print(f"  [SKIP] {code} 피처 빌드 실패: {e}")
            continue

        feature_df = clean_and_clip_outliers(
            feature_df,
            warm_up_period=OUTLIER_WARMUP_PERIOD,
            verbose=False,
        )
        if len(feature_df) < MIN_FEATURE_ROWS:
            print(f"  [SKIP] {code} 데이터 부족 ({len(feature_df)}행)")
            continue

        selected = select_features(feature_df, get_selected_features(include_meta=True))
        selected = remove_zero_variance(selected, verbose=False)

        meta_cols = [c for c in selected.columns if c.startswith('meta_sector_')]
        non_meta_cols = [c for c in selected.columns if c not in meta_cols]

        scaler = StandardScaler()
        scaled_non_meta = scaler.fit_transform(selected[non_meta_cols])
        scaled_non_meta = np.clip(scaled_non_meta, -5.0, 5.0)

        scaled_df = pd.DataFrame(scaled_non_meta, columns=non_meta_cols, index=selected.index)
        for c in meta_cols:
            scaled_df[c] = selected[c].values
        final_cols = [c for c in selected.columns if c in scaled_df.columns]
        scaled_df = scaled_df[final_cols]

        feature_names = list(scaled_df.columns)
        dates = feature_df['date'].iloc[-len(scaled_df):].reset_index(drop=True)

        save_mask = dates >= save_start_date
        dates = dates[save_mask].reset_index(drop=True)
        scaled_df = scaled_df[save_mask.values].reset_index(drop=True)

        count = 0
        with psql.get_session() as session:
            for j in range(len(scaled_df)):
                date_str = str(dates.iloc[j])
                y = scaled_df.iloc[j].values.tolist()
                psql.upsert(session, FeatureVector(
                    name=etf_name,
                    code=code,
                    version=version,
                    x=date_str,
                    y=y,
                    meta={"unit": "day", "features": feature_names},
                ))
                count += 1
            session.commit()

        print(f"  {code}: {count}행 저장 (피처 {len(feature_names)}개)")
        total += count

    print(f"\n완료: 총 {total}행 저장")
    return total


# ============================================================
# REST API 기반 피처 벡터 로더
# ============================================================


# ============================================================
# REST API 기반 피처 벡터 로더
# ============================================================

def load_feature_vectors_from_api(
    codes: List[str],
    version: str = "1",
    n: int = 5,
    api_key: str = None,
    date_until: str = None,
) -> Dict[str, dict]:
    """quantylab-api REST API에서 피처 벡터를 조회합니다.

    Args:
        codes: ETF 종목 코드 목록 (예: ['069500', '102110'])
        version: 피처 벡터 버전 (기본 "1")
        n: 반환할 최근 일수 (기본 5)
        api_key: Bearer 토큰. 미지정 시 환경변수 QUANTYLAB_API_KEY 사용.
        date_until: 조회 기준 날짜 상한 (YYYYMMDD, 미지정 시 최신)

    Returns:
        {
            code: {
                'features': np.ndarray (shape: [dim]),   # 최신일 피처 벡터
                'date': str,                              # 최신일 날짜
                'feature_names': list[str],               # 피처 이름 목록
                'all_records': list[dict],                # n일치 전체 레코드 (오래된 순)
            },
            ...
        }
        조회 실패 또는 데이터 없는 종목은 결과에서 제외됩니다.
    """
    from .quantylab_rest import QuantylabRESTClient

    key = api_key or os.environ.get("QUANTYLAB_API_KEY", "")
    if not key:
        raise ValueError("api_key 또는 환경변수 QUANTYLAB_API_KEY를 설정하세요.")

    end_date = date_until  # None → client 내부에서 오늘로 처리

    client = QuantylabRESTClient(token=key)
    results: Dict[str, dict] = {}

    for code in codes:
        try:
            records = client.get_feature_vectors(
                code=code, version=version, n=n, end_date=end_date,
            )
        except Exception as e:
            print(f"  [WARN] {code} 피처 벡터 API 호출 실패: {e}")
            continue

        if not records:
            continue

        # API는 x desc 정렬 → 최신이 records[0]; n건으로 제한
        all_records = []
        for rec in records[:n]:
            meta = rec.get("meta") or {}
            feature_names = meta.get("features", [])
            all_records.append({
                "date": rec["x"],
                "features": rec.get("y", []),
                "feature_names": feature_names,
            })

        latest = all_records[0]
        features = np.array(latest["features"], dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        results[code] = {
            "features": features,
            "date": latest["date"],
            "feature_names": latest["feature_names"],
            "all_records": list(reversed(all_records)),  # 오래된 순
        }

    return results


# ============================================================
# Prefect Task
# ============================================================


@task(
    name="feature-vector-task",
    log_prints=True,
    cache_expiration=timedelta(hours=12),
    persist_result=True,
    cache_key_fn=task_input_hash,
)
def run(n: int = DEFAULT_N, start_date: Optional[str] = None, end_date: Optional[str] = None,
        save_start_date: Optional[str] = None,
        version: str = "1", etf_code: Optional[str] = None, min_candles: int = 500):
    """
    피처 벡터 빌드 및 DB 저장

    Args:
        n: 저장할 최근 일수 (기본: 20). save_start_date 미지정 시 사용.
        start_date: 시작일 (미지정 시 n + 180일 전)
        end_date: 종료일 (기본: 오늘)
        save_start_date: 저장 시작일 (YYYYMMDD). 지정 시 n 무시.
        version: 피처 벡터 버전
        etf_code: 특정 ETF 코드 (None이면 전체 TIGER ETF)
        min_candles: 최소 캔들 수
    """
    save_feature_vectors_to_db(
        n=n,
        start_date=start_date,
        end_date=end_date,
        save_start_date=save_start_date,
        version=version,
        etf_code=etf_code,
        min_candles=min_candles,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="피처 벡터 빌드 및 DB 저장")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="저장할 최근 일수")
    parser.add_argument("--start-date", type=str, default=None, help="데이터 로드 시작일 (YYYYMMDD)")
    parser.add_argument("--end-date", type=str, default=None, help="데이터 로드 종료일 (YYYYMMDD)")
    parser.add_argument("--save-start-date", type=str, default=None,
                        help="저장 시작일 (YYYYMMDD). 지정 시 --n 무시. 전체 이력 생성 시 사용.")
    parser.add_argument("--version", type=str, default="1")
    parser.add_argument("--etf-code", type=str, default=None)
    parser.add_argument("--min-candles", type=int, default=500)
    parser.add_argument("--truncate", action="store_true", default=False,
                        help="저장 전 FeatureVector 테이블 전체 삭제")
    args = parser.parse_args()

    save_feature_vectors_to_db(
        n=args.n,
        start_date=args.start_date,
        end_date=args.end_date,
        save_start_date=args.save_start_date,
        version=args.version,
        etf_code=args.etf_code,
        min_candles=args.min_candles,
        truncate=args.truncate,
    )
