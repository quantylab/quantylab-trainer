import datetime
import json
import inspect
import os

from ratelimit import limits, sleep_and_retry
import requests
import pandas as pd

import time


# 안전하게 0.3초당 1건으로 설정
@sleep_and_retry
@limits(calls=1, period=0.3)
def request(url, headers={}, data="", verbose=False, retries=3):
    headers_wo_auth = {k: v for k, v in headers.items() if k.lower() != "authorization"}
    if verbose:
        print(f"Requesting {url} with headers={headers_wo_auth} and data={data}")
    for i in range(retries):
        res = requests.post(url, headers=headers, data=data)
        if res.status_code == 429:
            wait = 2 ** (i + 1)
            print(f"[kiwoom_rest] 429 Rate Limit, {wait}s 대기 후 재시도 ({i+1}/{retries})...")
            time.sleep(wait)
            continue
        if res.status_code < 500:
            return res
        if i < retries - 1:
            print(f"[kiwoom_rest] {res.status_code} retrying ({i+1}/{retries})...")
            time.sleep(2 ** (i + 1))
    return res


# 0:코스피,10:코스닥,3:ELW,8:ETF,30:K-OTC,50:코넥스,5:신주인수권,4:뮤추얼펀드,6:리츠,9:하이일드
MARKET_CODE_MAP = {
    "kospi": "0",
    "kosdaq": "10",
    "etf": "8",
}

# 업종코드: 001:종합(KOSPI), 101:종합(KOSDAQ), 201:KOSPI200, 302:KOSTAR, 701:KRX100
INDEX_CODE_MAP = {
    "kospi": "001",
    "kospi_large": "002",
    "kospi_mid": "003",
    "kospi_small": "004",
    "kosdaq": "101",
    "kospi200": "201",
    "kostar": "302",
    "krx100": "701",
}


class KiwoomRestClient:
    def __init__(self, real=False):
        self.real = real
        if self.real:
            self.api = "https://api.kiwoom.com"
            self.ws_api = ""
            self.appkey = os.environ.get("KIWOOM_REAL_APPKEY", "")
            self.appsecret = os.environ.get("KIWOOM_REAL_APPSECRET", "")
            self.acntno = os.environ.get("KIWOOM_REAL_ACCOUNT", "")
        else:
            self.api = "https://mockapi.kiwoom.com"
            self.ws_api = ""
            self.appkey = os.environ.get("KIWOOM_MOCK_APPKEY", "")
            self.appsecret = os.environ.get("KIWOOM_MOCK_APPSECRET", "")
            self.acntno = os.environ.get("KIWOOM_MOCK_ACCOUNT", "")

        self.access_token = ""
        self.token_expires_dt = None
        self.get_access_token()
        assert self.access_token != ""

        self.verbose = False

    def set_verbose(self, verbose):
        self.verbose = verbose

    def get_access_token(self):
        if self.access_token and self.token_expires_dt and datetime.datetime.now() < self.token_expires_dt:
            return self.access_token
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        res = request(f"{self.api}/oauth2/token", headers=headers, data=json.dumps({
            "grant_type": "client_credentials",
            "appkey": self.appkey,
            "secretkey": self.appsecret,
        }))
        if res.status_code != 200:
            raise RuntimeError(
                f"Kiwoom token request failed: HTTP {res.status_code} {res.text}"
            )
        res.encoding = "utf-8"
        try:
            data = json.loads(res.text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Kiwoom token response was not valid JSON: {res.text}") from exc
        token = data.get("token")
        if not token:
            return_code = data.get("return_code")
            return_msg = data.get("return_msg") or data
            raise RuntimeError(
                f"Kiwoom token request failed: {return_msg} (return_code={return_code})"
            )
        self.access_token = token
        if "expires_dt" in data:
            self.token_expires_dt = datetime.datetime.strptime(data["expires_dt"], "%Y%m%d%H%M%S")
        return self.access_token
    

    def _get_list_data(self, url, api_id, data, list_key, fields, 
                       start_date=None, date_key="일자"):
        cont_yn = "N"
        next_key = ""
        df = pd.DataFrame()
        while True:
            self.get_access_token()  # 만료 시 자동 갱신
            headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "authorization": f"Bearer {self.access_token}",  # 접근토큰
                "cont-yn": cont_yn,  # 연속조회여부
                "next-key": next_key,  # 연속조회키
                "api-id": api_id,  # TR명
            }
            res = request(f"{self.api}{url}", headers=headers, data=json.dumps(data), verbose=self.verbose)
            assert res.status_code == 200, f"{res.status_code} {res.text}"
            result = res.json()
            assert list_key in result, f"{list_key} not in {result.keys()}, result={result}"
            output = []
            for item in result[list_key]:
                output.append({fields.get(k, {"name": k})["name"]: v for k, v in item.items()})
            df = pd.concat([df, pd.DataFrame(output)], ignore_index=True)
            # 시작일자가 지정되어 있으면 체크
            if start_date and date_key in df and df.iloc[-1][date_key] <= start_date:
                if self.verbose:
                    print(f"Reached start_date: {start_date} <= {df.iloc[-1][date_key]}")
                break
            # 중복이 있으면 종료
            df_new = df.drop_duplicates()
            if len(df_new) < len(df):
                df = df_new
                if self.verbose:
                    print("Duplicates found, stopping retrieval.")
                break
            # 연속 조회 설정
            cont_yn = res.headers.get("cont-yn", "N")  # 연속조회여부
            next_key = res.headers.get("next-key", "")  # 연속조회키
            if cont_yn == "N":
                break
        if df.empty:
            return None

        expected_cols = [v["name"] for v in fields.values()]
        numeric_cols = [v["name"] for v in fields.values() if v["numeric"]]

        # Some Kiwoom list responses omit sparsely populated fields entirely.
        # Reindex first so callers can rely on a stable schema.
        df = df.reindex(columns=expected_cols)

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in expected_cols:
            if col not in numeric_cols:
                df[col] = df[col].fillna("")
        return df[expected_cols]

    def _get_single_data(self, url, api_id, data):
        self.get_access_token()
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self.access_token}",
            "api-id": api_id,
        }
        res = request(f"{self.api}{url}", headers=headers, data=json.dumps(data), verbose=self.verbose)
        assert res.status_code == 200, f"{res.status_code} {res.text}"
        return res.json()

    def get_stock_info_list(self, market_code="kospi"):
        assert market_code in MARKET_CODE_MAP
        mrkt_tp = MARKET_CODE_MAP[market_code]
        assert mrkt_tp != ""

        FIELDS = {
            "code": {"name": "종목코드", "numeric": False},  # 단축코드
            "name": {"name": "종목명", "numeric": False},
            "listCount": {"name": "상장주식수", "numeric": False},
            "auditInfo": {"name": "감리구분", "numeric": False},
            "regDay": {"name": "상장일", "numeric": False},
            "lastPrice": {"name": "전일종가", "numeric": False},
            "state": {"name": "종목상태", "numeric": False},
            "marketCode": {"name": "시장구분코드", "numeric": False},
            "marketName": {"name": "시장명", "numeric": False},
            "upName": {"name": "업종명", "numeric": False},
            "upSizeName": {"name": "회사크기분류", "numeric": False},
            "companyClassName": {"name": "회사분류", "numeric": False},  # 코스닥만 존재함
            "orderWarning": {"name": "투자유의종목여부", "numeric": False},  # 0: 해당없음, 2: 정리매매, 3: 단기과열, 4: 투자위험, 5: 투자경과, 1: ETF투자주의요망(ETF인 경우만 전달
            "nxtEnable": {"name": "NXT가능여부", "numeric": False},  # Y: 가능
        }

        return self._get_list_data(
            url="/api/dostk/stkinfo",
            api_id="ka10099",
            data={"mrkt_tp": mrkt_tp},
            list_key="list",
            fields=FIELDS,
        )

    def get_stock_day_candles(self, code, start_date, end_date=None):
        # 주식일봉차트조회요청 ka10081
        FIELDS = {
            "dt": {"name": "일자", "numeric": False},
            "cur_prc": {"name": "현재가", "numeric": True},
            "open_pric": {"name": "시가", "numeric": True},
            "high_pric": {"name": "고가", "numeric": True},
            "low_pric": {"name": "저가", "numeric": True},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "pred_pre_sig": {"name": "전일대비기호", "numeric": False},
            "trde_qty": {"name": "거래량", "numeric": True},
            "trde_prica": {"name": "거래대금", "numeric": True},
            "trde_tern_rt": {"name": "거래회전율", "numeric": True},
        }

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        df = self._get_list_data(
            url="/api/dostk/chart",
            api_id="ka10081",
            data={
                "stk_cd": code,
                "base_dt": end_date,
                "upd_stkpc_tp": "1",  # 수정주가구분 (0: 무수정주가, 1: 수정주가)
            },
            list_key="stk_dt_pole_chart_qry",
            fields=FIELDS,
            start_date=start_date,
            date_key="일자",
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]].sort_values("일자").reset_index(drop=True)

    def get_investor_buy_sell(self, code, start_date, end_date=None):
        # 종목별투자자기관별요청 ka10059
        FIELDS = {
            "dt": {"name": "일자", "numeric": False},
            "cur_prc": {"name": "현재가", "numeric": True},
            "pre_sig": {"name": "대비기호", "numeric": False},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "flu_rt": {"name": "등락율", "numeric": True},
            "acc_trde_qty": {"name": "누적거래량", "numeric": True},
            "acc_trde_prica": {"name": "누적거래대금", "numeric": True},
            "ind_invsr": {"name": "개인투자자", "numeric": True},
            "frgnr_invsr": {"name": "외국인투자자", "numeric": True},
            "orgn": {"name": "기관계", "numeric": True},
            "fnnc_invt": {"name": "금융투자", "numeric": True},
            "insrnc": {"name": "보험", "numeric": True},
            "invtrt": {"name": "투신", "numeric": True},
            "etc_fnnc": {"name": "기타금융", "numeric": True},
            "bank": {"name": "은행", "numeric": True},
            "penfnd_etc": {"name": "연기금등", "numeric": True},
            "samo_fund": {"name": "사모펀드", "numeric": True},
            "natn": {"name": "국가", "numeric": True},
            "etc_corp": {"name": "기타법인", "numeric": True},
            "natfor": {"name": "내외국인", "numeric": True},
        }

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")
        
        df = self._get_list_data(
            url="/api/dostk/stkinfo",
            api_id="ka10059",
            data={
                "dt": end_date,
                "stk_cd": code,
                "amt_qty_tp": "2",  # 금액수량구분	String	Y	1	1:금액, 2:수량
                "trde_tp": "0",  # 매매구분	String	Y	1	0:순매수, 1:매수, 2:매도
                "unit_tp": "1",  # 단위구분	String	Y	4	1000:천주, 1:단주
            },
            list_key="stk_invsr_orgn",
            fields=FIELDS,
            start_date=start_date,
            date_key="일자",
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]].sort_values("일자").reset_index(drop=True)
    
    def get_stock_credit(self, code, start_date, end_date=None, qry_tp="1"):
        # 신용매매동향요청 ka10013
        # qry_tp: 1:융자, 2:대주
        FIELDS = {
            "dt": {"name": "일자", "numeric": False},
            "cur_prc": {"name": "현재가", "numeric": True},
            "pred_pre_sig": {"name": "전일대비기호", "numeric": False},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "trde_qty": {"name": "거래량", "numeric": True},
            "new": {"name": "신규", "numeric": True},
            "rpya": {"name": "상환", "numeric": True},
            "remn": {"name": "잔고", "numeric": True},
            "amt": {"name": "금액", "numeric": True},
            "pre": {"name": "대비", "numeric": True},
            "shr_rt": {"name": "공여율", "numeric": True},
            "remn_rt": {"name": "잔고율", "numeric": True},
        }

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        df = self._get_list_data(
            url="/api/dostk/stkinfo",
            api_id="ka10013",
            data={
                "stk_cd": code,
                "dt": end_date,
                "qry_tp": qry_tp,
            },
            list_key="crd_trde_trend",
            fields=FIELDS,
            start_date=start_date,
            date_key="일자",
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]].sort_values("일자").reset_index(drop=True)

    def get_stock_index_list(self, market_code="kospi"):
        # 업종코드 리스트 ka10101
        # 업종에 속한 종목 리스트
        assert market_code in MARKET_CODE_MAP
        mrkt_tp = MARKET_CODE_MAP[market_code]
        FIELDS = {
            "marketCode": {"name": "시장구분코드", "numeric": False},
            "code": {"name": "코드", "numeric": False},
            "name": {"name": "업종명", "numeric": False},
            "group": {"name": "그룹", "numeric": False},
        }
        return self._get_list_data(
            url="/api/dostk/stkinfo",
            api_id="ka10101",
            data={
                "mrkt_tp": mrkt_tp,
            },
            list_key="list",
            fields=FIELDS,
        )

    def get_stock_info(self, code):
        # 주식기본정보요청 ka10001
        # systrader: get_stockstatus, get_stockfeatures, get_marketcap
        return self._get_single_data(
            url="/api/dostk/stkinfo",
            api_id="ka10001",
            data={"stk_cd": code},
        )

    def get_stock_infos(self, codes, chunk_size=20):
        # 관심종목정보요청 ka10095
        if not codes:
            return []

        results = []
        for start in range(0, len(codes), chunk_size):
            chunk = [str(code).zfill(6) for code in codes[start:start + chunk_size]]
            data = self._get_single_data(
                url="/api/dostk/stkinfo",
                api_id="ka10095",
                data={"stk_cd": "|".join(chunk)},
            )
            results.extend(data.get("atn_stk_infr") or [])
        return results

    def get_stock_min_candles(self, code, start_date, end_date=None, tic_scope="1"):
        # 주식분봉차트조회요청 ka10080
        # tic_scope: 1:1분, 3:3분, 5:5분, 10:10분, 15:15분, 30:30분, 45:45분, 60:60분
        # systrader: get_stockcandles_min, get_bid_volume_minutely
        FIELDS = {
            "cntr_tm": {"name": "체결시간", "numeric": False},
            "cur_prc": {"name": "현재가", "numeric": True},
            "open_pric": {"name": "시가", "numeric": True},
            "high_pric": {"name": "고가", "numeric": True},
            "low_pric": {"name": "저가", "numeric": True},
            "trde_qty": {"name": "거래량", "numeric": True},
            "acc_trde_qty": {"name": "누적거래량", "numeric": True},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "pred_pre_sig": {"name": "전일대비기호", "numeric": False},
        }

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        data = {
            "stk_cd": code,
            "tic_scope": tic_scope,
            "upd_stkpc_tp": "1",
        }
        if end_date:
            data["base_dt"] = end_date

        df = self._get_list_data(
            url="/api/dostk/chart",
            api_id="ka10080",
            data=data,
            list_key="stk_min_pole_chart_qry",
            fields=FIELDS,
            start_date=start_date,
            date_key="체결시간",
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]].sort_values("체결시간").reset_index(drop=True)

    def get_market_day_candles(self, index_code="kospi", start_date=None, end_date=None):
        # 업종일봉조회요청 ka20006
        # systrader: get_marketcandles
        # 지수 값은 소수점 제거 후 100배 값으로 반환
        assert index_code in INDEX_CODE_MAP
        inds_cd = INDEX_CODE_MAP[index_code]

        FIELDS = {
            "dt": {"name": "일자", "numeric": False},
            "cur_prc": {"name": "현재가", "numeric": True},
            "open_pric": {"name": "시가", "numeric": True},
            "high_pric": {"name": "고가", "numeric": True},
            "low_pric": {"name": "저가", "numeric": True},
            "trde_qty": {"name": "거래량", "numeric": True},
            "trde_prica": {"name": "거래대금", "numeric": True},
        }

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        df = self._get_list_data(
            url="/api/dostk/chart",
            api_id="ka20006",
            data={
                "inds_cd": inds_cd,
                "base_dt": end_date,
            },
            list_key="inds_dt_pole_qry",
            fields=FIELDS,
            start_date=start_date,
            date_key="일자",
        )
        if df is None or df.empty:
            return df
        # 지수값 100으로 나누기
        for col in ["현재가", "시가", "고가", "저가"]:
            df[col] = df[col] / 100.0
        return df.assign(업종코드=inds_cd)[["업종코드"] + [v["name"] for v in FIELDS.values()]].sort_values("일자").reset_index(drop=True)

    def get_short_selling(self, code, start_date, end_date=None):
        # 공매도추이요청 ka10014
        # systrader: get_shortstockselling
        FIELDS = {
            "dt": {"name": "일자", "numeric": False},
            "close_pric": {"name": "종가", "numeric": True},
            "pred_pre_sig": {"name": "전일대비기호", "numeric": False},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "flu_rt": {"name": "등락율", "numeric": True},
            "trde_qty": {"name": "거래량", "numeric": True},
            "shrts_qty": {"name": "공매도량", "numeric": True},
            "ovr_shrts_qty": {"name": "누적공매도량", "numeric": True},
            "trde_wght": {"name": "매매비중", "numeric": True},
            "shrts_trde_prica": {"name": "공매도거래대금", "numeric": True},
            "shrts_avg_pric": {"name": "공매도평균가", "numeric": True},
        }

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        df = self._get_list_data(
            url="/api/dostk/shsa",
            api_id="ka10014",
            data={
                "stk_cd": code,
                "tm_tp": "1",
                "strt_dt": start_date,
                "end_dt": end_date,
            },
            list_key="shrts_trnsn",
            fields=FIELDS,
            start_date=start_date,
            date_key="일자",
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]].sort_values("일자").reset_index(drop=True)

    def get_holding_stocks(self, qry_tp="1", stex_tp="KRX"):
        # 계좌평가잔고내역요청 kt00018
        # systrader: get_holdingstocks
        # qry_tp: 1:합산, 2:개별
        FIELDS = {
            "stk_cd": {"name": "종목코드", "numeric": False},
            "stk_nm": {"name": "종목명", "numeric": False},
            "rmnd_qty": {"name": "보유수량", "numeric": True},
            "trde_able_qty": {"name": "매매가능수량", "numeric": True},
            "pur_pric": {"name": "매입가", "numeric": True},
            "pur_amt": {"name": "매입금액", "numeric": True},
            "cur_prc": {"name": "현재가", "numeric": True},
            "evlt_amt": {"name": "평가금액", "numeric": True},
            "evltv_prft": {"name": "평가손익", "numeric": True},
            "prft_rt": {"name": "수익률", "numeric": True},
            "poss_rt": {"name": "보유비중", "numeric": True},
        }

        df = self._get_list_data(
            url="/api/dostk/acnt",
            api_id="kt00018",
            data={
                "qry_tp": qry_tp,
                "dmst_stex_tp": stex_tp,
            },
            list_key="acnt_evlt_remn_indv_tot",
            fields=FIELDS,
        )
        return df

    def get_stock_bid(self, code):
        # 주식호가요청 ka10004
        # systrader: get_stockbid
        return self._get_single_data(
            url="/api/dostk/mrkcond",
            api_id="ka10004",
            data={"stk_cd": code},
        )

    def get_program_trading(self, code, start_date=None, end_date=None):
        # 종목일별프로그램매매추이요청 ka90013
        # systrader: get_program_volume
        FIELDS = {
            "dt": {"name": "일자", "numeric": False},
            "cur_prc": {"name": "현재가", "numeric": True},
            "pre_sig": {"name": "대비기호", "numeric": False},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "flu_rt": {"name": "등락율", "numeric": True},
            "trde_qty": {"name": "거래량", "numeric": True},
            "prm_sell_qty": {"name": "프로그램매도수량", "numeric": True},
            "prm_buy_qty": {"name": "프로그램매수수량", "numeric": True},
            "prm_netprps_qty": {"name": "프로그램순매수수량", "numeric": True},
            "prm_netprps_qty_irds": {"name": "프로그램순매수수량증감", "numeric": True},
            "prm_sell_amt": {"name": "프로그램매도금액", "numeric": True},
            "prm_buy_amt": {"name": "프로그램매수금액", "numeric": True},
            "prm_netprps_amt": {"name": "프로그램순매수금액", "numeric": True},
            "prm_netprps_amt_irds": {"name": "프로그램순매수금액증감", "numeric": True},
        }

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        df = self._get_list_data(
            url="/api/dostk/mrkcond",
            api_id="ka90013",
            data={
                "stk_cd": code,
                "amt_qty_tp": "",
                "date": end_date,
            },
            list_key="stk_daly_prm_trde_trnsn",
            fields=FIELDS,
            start_date=start_date,
            date_key="일자",
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]].sort_values("일자").reset_index(drop=True)

    def get_overtime(self, code):
        # 시간외단일가요청 ka10087
        # systrader: get_overtime_uni_daily
        return self._get_single_data(
            url="/api/dostk/mrkcond",
            api_id="ka10087",
            data={"stk_cd": code},
        )

    def get_trade_intensity(self, code):
        # 체결강도추이시간별요청 ka10046
        # systrader: get_trade_intensity
        FIELDS = {
            "cntr_tm": {"name": "체결시간", "numeric": False},
            "cur_prc": {"name": "현재가", "numeric": True},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "pred_pre_sig": {"name": "전일대비기호", "numeric": False},
            "flu_rt": {"name": "등락율", "numeric": True},
            "trde_qty": {"name": "거래량", "numeric": True},
            "acc_trde_prica": {"name": "누적거래대금", "numeric": True},
            "acc_trde_qty": {"name": "누적거래량", "numeric": True},
            "cntr_str": {"name": "체결강도", "numeric": True},
            "cntr_str_5min": {"name": "체결강도5분", "numeric": True},
            "cntr_str_20min": {"name": "체결강도20분", "numeric": True},
            "cntr_str_60min": {"name": "체결강도60분", "numeric": True},
        }

        df = self._get_list_data(
            url="/api/dostk/mrkcond",
            api_id="ka10046",
            data={"stk_cd": code},
            list_key="cntr_str_tm",
            fields=FIELDS,
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]].sort_values("체결시간").reset_index(drop=True)

    def get_trade_intensity_daily(self, code):
        # 체결강도추이일별요청 ka10047
        FIELDS = {
            "dt": {"name": "일자", "numeric": False},
            "cur_prc": {"name": "현재가", "numeric": True},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "pred_pre_sig": {"name": "전일대비기호", "numeric": False},
            "flu_rt": {"name": "등락율", "numeric": True},
            "trde_qty": {"name": "거래량", "numeric": True},
            "acc_trde_prica": {"name": "누적거래대금", "numeric": True},
            "acc_trde_qty": {"name": "누적거래량", "numeric": True},
            "cntr_str": {"name": "체결강도", "numeric": True},
            "cntr_str_5min": {"name": "체결강도5일", "numeric": True},
            "cntr_str_20min": {"name": "체결강도20일", "numeric": True},
            "cntr_str_60min": {"name": "체결강도60일", "numeric": True},
        }

        df = self._get_list_data(
            url="/api/dostk/mrkcond",
            api_id="ka10047",
            data={"stk_cd": code},
            list_key="cntr_str_daly",
            fields=FIELDS,
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]].sort_values("일자").reset_index(drop=True)

    def get_balance(self, qry_tp="1", stex_tp="KRX"):
        # 계좌평가현황요청 kt00004
        # systrader: get_balance
        # qry_tp: 0:전체, 1:상장폐지종목제외
        result = self._get_single_data(
            url="/api/dostk/acnt",
            api_id="kt00004",
            data={
                "qry_tp": qry_tp,
                "dmst_stex_tp": stex_tp,
            },
        )
        # 종목 리스트를 DataFrame으로 변환
        summary = {k: v for k, v in result.items() if k != "stk_acnt_evlt_prst" and k not in ("return_code", "return_msg")}
        stocks_df = None
        if "stk_acnt_evlt_prst" in result and result["stk_acnt_evlt_prst"]:
            stocks_df = pd.DataFrame(result["stk_acnt_evlt_prst"])
            for col in ["rmnd_qty", "avg_prc", "cur_prc", "evlt_amt", "pl_amt", "pl_rt", "pur_amt", "setl_remn"]:
                if col in stocks_df.columns:
                    stocks_df[col] = pd.to_numeric(stocks_df[col], errors="coerce")
        return {"summary": summary, "stocks": stocks_df}

    def get_volume_profile(self, code, start_date, end_date=None, dt="20", stex_tp="3"):
        # 거래원매물대분석요청 ka10043
        # systrader: get_volume_profile
        # dt: 5:5일, 10:10일, 20:20일, 40:40일, 60:60일, 120:120일
        FIELDS = {
            "dt": {"name": "일자", "numeric": False},
            "close_pric": {"name": "종가", "numeric": True},
            "pre_sig": {"name": "대비기호", "numeric": False},
            "pred_pre": {"name": "전일대비", "numeric": True},
            "sel_qty": {"name": "매도량", "numeric": True},
            "buy_qty": {"name": "매수량", "numeric": True},
            "netprps_qty": {"name": "순매수수량", "numeric": True},
            "trde_qty_sum": {"name": "거래량합", "numeric": True},
            "trde_wght": {"name": "거래비중", "numeric": True},
        }

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        df = self._get_list_data(
            url="/api/dostk/stkinfo",
            api_id="ka10043",
            data={
                "stk_cd": code,
                "strt_dt": start_date,
                "end_dt": end_date,
                "qry_dt_tp": "1",
                "pot_tp": "0",
                "dt": dt,
                "sort_base": "1",
                "mmcm_cd": "",
                "stex_tp": stex_tp,
            },
            list_key="trde_ori_prps_anly",
            fields=FIELDS,
        )
        if df is None or df.empty:
            return df
        return df.assign(종목코드=code)[["종목코드"] + [v["name"] for v in FIELDS.values()]]

    # ═══════════════════════════════════════════════════════
    # 주문 API
    # ═══════════════════════════════════════════════════════

    def buy_order(self, code, qty, price=0, trde_tp="3", stex_tp="KRX"):
        """주식 매수 주문 (kt10000)
        
        Args:
            code: 종목코드
            qty: 주문수량
            price: 주문단가 (시장가 주문 시 0)
            trde_tp: 거래유형 (0:보통, 3:시장가, 5:조건부지정가, 6:최유리, 7:최우선)
            stex_tp: 거래소 (KRX)
        
        Returns:
            dict: {ord_no, return_code, return_msg}
        """
        return self._get_single_data(
            url="/api/dostk/ordr",
            api_id="kt10000",
            data={
                "dmst_stex_tp": stex_tp,
                "stk_cd": code,
                "ord_qty": str(qty),
                "ord_uv": str(price),
                "trde_tp": trde_tp,
                "cond_uv": "0",
            },
        )

    def sell_order(self, code, qty, price=0, trde_tp="3", stex_tp="KRX"):
        """주식 매도 주문 (kt10001)
        
        Args:
            code: 종목코드
            qty: 주문수량
            price: 주문단가 (시장가 주문 시 0)
            trde_tp: 거래유형 (0:보통, 3:시장가, 5:조건부지정가, 6:최유리, 7:최우선)
            stex_tp: 거래소 (KRX)
        
        Returns:
            dict: {ord_no, dmst_stex_tp, return_code, return_msg}
        """
        return self._get_single_data(
            url="/api/dostk/ordr",
            api_id="kt10001",
            data={
                "dmst_stex_tp": stex_tp,
                "stk_cd": code,
                "ord_qty": str(qty),
                "ord_uv": str(price),
                "trde_tp": trde_tp,
                "cond_uv": "0",
            },
        )

    def cancel_order(self, orig_ord_no, code, cncl_qty="0", stex_tp="KRX"):
        """주식 취소 주문 (kt10003)
        
        Args:
            orig_ord_no: 원주문번호
            code: 종목코드
            cncl_qty: 취소수량 ("0"=잔여 전량)
            stex_tp: 거래소 (KRX)
        
        Returns:
            dict: {ord_no, return_code, return_msg}
        """
        return self._get_single_data(
            url="/api/dostk/ordr",
            api_id="kt10003",
            data={
                "dmst_stex_tp": stex_tp,
                "orig_ord_no": str(orig_ord_no),
                "stk_cd": code,
                "cncl_qty": str(cncl_qty),
            },
        )

    def get_open_orders(self, stk_cd="", stex_tp="KRX"):
        """미체결 주문 조회 (ka10075)
        
        Args:
            stk_cd: 종목코드 (""=전체)
            stex_tp: 거래소 (KRX)
        
        Returns:
            DataFrame: 미체결 주문 목록
        """
        FIELDS = {
            "acnt_no": {"name": "계좌번호", "numeric": False},
            "ord_no": {"name": "주문번호", "numeric": False},
            "stk_cd": {"name": "종목코드", "numeric": False},
            "stk_nm": {"name": "종목명", "numeric": False},
            "ord_qty": {"name": "주문수량", "numeric": True},
            "ord_pric": {"name": "주문가격", "numeric": True},
            "oso_qty": {"name": "미체결수량", "numeric": True},
            "cntr_pric": {"name": "체결가격", "numeric": True},
            "cntr_qty": {"name": "체결수량", "numeric": True},
            "ord_tp": {"name": "주문구분", "numeric": False},
            "trde_tp": {"name": "거래유형", "numeric": False},
            "ord_tm": {"name": "주문시간", "numeric": False},
        }
        return self._get_list_data(
            url="/api/dostk/acnt",
            api_id="ka10075",
            data={
                "all_stk_tp": "0",
                "trde_tp": "0",
                "stk_cd": stk_cd,
                "stex_tp": stex_tp,
            },
            list_key="oso",
            fields=FIELDS,
        )

    def get_order_fill_status(self, order_date="", stk_cd="", stex_tp="KRX"):
        """계좌별 주문/체결 현황 조회 (kt00009).

        ``get_open_orders``는 아직 미체결인 주문만 반환하므로, 주문 직후
        조회에서 빠진 뒤 체결된 주문을 확인하려면 이 당일 주문/체결 현황
        조회를 사용해야 한다.
        """
        FIELDS = {
            "ord_no": {"name": "주문번호", "numeric": False},
            "stk_cd": {"name": "종목코드", "numeric": False},
            "trde_tp": {"name": "매매구분", "numeric": False},
            "io_tp_nm": {"name": "주문유형", "numeric": False},
            "ord_qty": {"name": "주문수량", "numeric": True},
            "ord_uv": {"name": "주문단가", "numeric": True},
            "cnfm_qty": {"name": "확인수량", "numeric": True},
            "cntr_no": {"name": "체결번호", "numeric": False},
            "acpt_tp": {"name": "접수구분", "numeric": False},
            "orig_ord_no": {"name": "원주문번호", "numeric": False},
            "stk_nm": {"name": "종목명", "numeric": False},
            "cntr_qty": {"name": "체결수량", "numeric": True},
            "cntr_uv": {"name": "체결단가", "numeric": True},
            "mdfy_cncl_tp": {"name": "정정취소구분", "numeric": False},
            "cntr_tm": {"name": "체결시간", "numeric": False},
            "dmst_stex_tp": {"name": "거래소구분", "numeric": False},
        }
        return self._get_list_data(
            url="/api/dostk/acnt",
            api_id="kt00009",
            data={
                "stk_bond_tp": "1",
                "mrkt_tp": "0",
                "sell_tp": "0",
                "qry_tp": "0",
                "dmst_stex_tp": stex_tp,
                "ord_dt": order_date,
                "stk_cd": stk_cd,
                "fr_ord_no": "",
            },
            list_key="acnt_ord_cntr_prst_array",
            fields=FIELDS,
        )

    def get_deposit(self, qry_tp="3"):
        """예수금 상세 현황 조회 (kt00001)
        
        Args:
            qry_tp: 조회구분 (2:일반, 3:추정)
        
        Returns:
            dict: 예수금 정보
        """
        return self._get_single_data(
            url="/api/dostk/acnt",
            api_id="kt00001",
            data={"qry_tp": qry_tp},
        )

    def get_orderable_amount(self, code, qty=0, price=0, trde_tp="2", exp_buy_unp=None, io_amt=""):
        """주문/인출가능금액 조회 (kt00010)

        Args:
            code: 종목코드
            qty: 매매수량
            price: 매수가격
            trde_tp: 매매구분 (1:매도, 2:매수)
            exp_buy_unp: 예상매수단가
            io_amt: 입출금액

        Returns:
            dict: 주문가능금액 응답 원문
        """
        payload = {
            "io_amt": str(io_amt or ""),
            "stk_cd": code,
            "trde_tp": str(trde_tp),
            "trde_qty": str(qty or ""),
            "uv": str(price or ""),
            "exp_buy_unp": str(exp_buy_unp if exp_buy_unp is not None else price or ""),
        }
        return self._get_single_data(
            url="/api/dostk/acnt",
            api_id="kt00010",
            data=payload,
        )
