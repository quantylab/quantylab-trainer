import time

import pandas as pd
import requests

from datetime import datetime, timedelta


DEFAULT_REQ_INTERVAL = 1


def req(func):
    def wrapper(self, *args, **kwargs):
        itv = time.time() - self.last_request_time
        if itv < DEFAULT_REQ_INTERVAL:
            time.sleep(DEFAULT_REQ_INTERVAL - itv)
        self.last_request_time = time.time()
        return func(self, *args, **kwargs)
    return wrapper


class QuantylabRESTClient:
    def __init__(self, token):
        assert token is not None
        self.token = token
        self.last_request_time = 0
        self.headers = {
            "Authorization": f"Bearer {token}",
        }

    BASE_URL = "https://api.quantylab.com"

    def _fetch_all_pages(self, url):
        """Paginate using 'results' key (legacy API format)."""
        data = []
        while True:
            res = requests.get(url, headers=self.headers).json()
            data.extend(res["results"])
            if not res.get("next"):
                break
            url = res.get("next")
            time.sleep(DEFAULT_REQ_INTERVAL)
        return data

    def _fetch_all_data(self, url):
        """Paginate using 'data' key (quantylab-api format)."""
        records = []
        while True:
            res = requests.get(url, headers=self.headers).json()
            records.extend(res["data"])
            next_path = res.get("next")
            if not next_path:
                break
            url = next_path if next_path.startswith("http") else f"{self.BASE_URL}{next_path}"
            time.sleep(DEFAULT_REQ_INTERVAL)
        return records

    @req
    def get_stock_market_candles(self, code, start_date=None, end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
        url = f"https://api.quantylab.com/stock-market-candles?code={code}&start_date={start_date}&end_date={end_date}"
        data = self._fetch_all_pages(url)
        df = pd.DataFrame(data)
        return df

    @req
    def get_stock_fa(self, code, start_date=None, end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
        url = f"https://api.quantylab.com/stock-fa/?code={code}&start_date={start_date}&end_date={end_date}"
        data = self._fetch_all_pages(url)
        df = pd.DataFrame(data)
        return df

    @req
    def get_investor_buy_sell(self, code, start_date=None, end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
        url = f"https://api.quantylab.com/investor-buy-sell/?code={code}&start_date={start_date}&end_date={end_date}"
        data = self._fetch_all_pages(url)
        for item in data:
            item.update(item.pop("content"))
        df = pd.DataFrame(data)
        return df

    @req
    def get_investor_top_net_buy_stocks(self, year, investor_type):
        assert investor_type in ["ind", "inst", "foreign"]
        url = f"https://api.quantylab.com/investor-top-net-buy-stocks/?year={year}&investor_type={investor_type}"
        data = self._fetch_all_pages(url)
        df = pd.DataFrame(data)
        return df

    @req
    def get_investor_year_avg_profits(self, year):
        url = f"https://api.quantylab.com/yearly-investor-avg-profits/?year={year}"
        data = self._fetch_all_pages(url)
        df = pd.DataFrame(data)
        return df

    @req
    def get_feature_vectors(self, code: str, version: str = "1",
                            start_date: str = None, end_date: str = None,
                            n: int = None) -> list:
        """ETF 피처 벡터를 조회합니다.

        Args:
            code: ETF 종목 코드 (예: '069500')
            version: 피처 벡터 버전 (기본 "1")
            start_date: 조회 시작일 (YYYYMMDD). 미지정 시 n 또는 20일 기준 자동 계산.
            end_date: 조회 종료일 (YYYYMMDD). 미지정 시 오늘.
            n: 최근 n일치 조회. start_date 미지정 시 기간 계산에 사용.

        Returns:
            list of dict: [{x, y, name, code, version, meta}, ...]
                          API 정렬(x desc, 최신 먼저) 그대로 반환.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            days = (n or 20) + 7
            start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days)).strftime("%Y%m%d")
        url = (f"{self.BASE_URL}/feature-vectors"
               f"?code={code}&version={version}"
               f"&start_date={start_date}&end_date={end_date}")
        return self._fetch_all_data(url)
