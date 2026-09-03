"""
타겟 ETF 리스트 모니터링

고정 리스트 대비 변경 감지:
  1. 비활성화된 ETF (상장폐지/비활성) → 알람
  2. 추가 후보 신규 ETF (조건 충족) → 알람
  3. 데이터 부족 ETF → 경고

사용법:
  python src/monitor_etfs.py              # 체크 + Slack 알람
  python src/monitor_etfs.py --dry-run    # 체크만 (알람 미발송)
"""
import argparse
import os
import sys
from datetime import datetime

from .target_etfs import TARGET_ETFS, EXCLUDE_KEYWORDS, MIN_CANDLES_FOR_NEW


def check_etf_changes(dry_run: bool = False) -> dict:
    """타겟 ETF 리스트 변경 감지"""
    from .db import psql
    from .models import EtfCode, DayEtfCandle
    from sqlalchemy import select, func

    today = datetime.now().strftime('%Y-%m-%d')
    alerts = []
    report = {
        'inactive': [],      # 비활성화된 타겟 ETF
        'new_candidates': [],  # 추가 가능한 신규 ETF
        'low_data': [],      # 데이터 부족 타겟 ETF
    }

    with psql.get_session() as session:
        # 전체 활성 ETF 조회
        all_etfs = session.execute(
            select(EtfCode)
        ).scalars().all()
        active_codes = {e.code: e for e in all_etfs if e.is_active}
        all_codes = {e.code: e for e in all_etfs}

        # ── 1. 비활성화 체크: 타겟 ETF가 DB에서 비활성이 된 경우 ──
        for code, name in TARGET_ETFS.items():
            if code not in all_codes:
                report['inactive'].append((code, name, 'DB에 없음'))
                alerts.append(f"⚠️ 타겟 ETF 미발견: {code} ({name}) - DB에 없음")
            elif not all_codes[code].is_active:
                report['inactive'].append((code, name, '비활성'))
                alerts.append(f"⚠️ 타겟 ETF 비활성: {code} ({name}) - 상장폐지/비활성")

        # ── 2. 신규 후보 체크: 활성 TIGER ETF 중 타겟에 없는 것 ──
        for code, etf in active_codes.items():
            if code in TARGET_ETFS:
                continue
            if 'TIGER' not in etf.name:
                continue
            # 제외 키워드 필터
            if any(kw in etf.name for kw in EXCLUDE_KEYWORDS):
                continue

            # 캔들 수 확인
            candle_count = session.execute(
                select(func.count()).select_from(DayEtfCandle)
                .where(DayEtfCandle.code == code)
            ).scalar()

            if candle_count >= MIN_CANDLES_FOR_NEW:
                listed = etf.extra.get('listeddate', 'N/A') if etf.extra else 'N/A'
                report['new_candidates'].append(
                    (code, etf.name, candle_count, listed))
                alerts.append(
                    f"🆕 신규 ETF 후보: {code} ({etf.name}) "
                    f"- {candle_count}일 데이터, 상장일={listed}")

        # ── 3. 데이터 부족 체크: 타겟 ETF 중 캔들이 적은 것 ──
        for code, name in TARGET_ETFS.items():
            if code not in active_codes:
                continue
            candle_count = session.execute(
                select(func.count()).select_from(DayEtfCandle)
                .where(DayEtfCandle.code == code)
            ).scalar()
            if candle_count < MIN_CANDLES_FOR_NEW:
                report['low_data'].append((code, name, candle_count))
                alerts.append(
                    f"📉 데이터 부족: {code} ({name}) - {candle_count}일 "
                    f"(최소 {MIN_CANDLES_FOR_NEW}일)")

    # ── 결과 출력 ──
    print(f"\n{'='*50}")
    print(f"  ETF 모니터링 결과 ({today})")
    print(f"{'='*50}")
    print(f"  타겟 ETF: {len(TARGET_ETFS)}개")
    print(f"  비활성   : {len(report['inactive'])}개")
    print(f"  신규 후보: {len(report['new_candidates'])}개")
    print(f"  데이터부족: {len(report['low_data'])}개")

    if report['inactive']:
        print(f"\n  [비활성화된 ETF]")
        for code, name, reason in report['inactive']:
            print(f"    {code} {name} ({reason})")

    if report['new_candidates']:
        print(f"\n  [신규 ETF 후보]")
        for code, name, cnt, listed in report['new_candidates']:
            print(f"    {code} {name} ({cnt}일, 상장={listed})")

    if report['low_data']:
        print(f"\n  [데이터 부족 ETF]")
        for code, name, cnt in report['low_data']:
            print(f"    {code} {name} ({cnt}일)")

    # ── Slack 알람 ──
    if alerts and not dry_run:
        _send_alerts(alerts, today)
    elif alerts and dry_run:
        print(f"\n  [DRY RUN] Slack 알람 {len(alerts)}건 미발송")
    else:
        print(f"\n  변경 사항 없음 ✓")

    print(f"{'='*50}\n")
    return report


def _send_alerts(alerts: list, date: str):
    """Slack 알람 발송"""
    header = f"*[ETF 모니터링] {date}*\n"
    body = "\n".join(alerts)
    message = header + body

    try:
        import requests
        webhook = os.environ.get("TRAINER_SLACK_WEBHOOK_URL")
        if not webhook:
            print("\n  Slack 알람 생략: TRAINER_SLACK_WEBHOOK_URL이 설정되지 않았습니다.")
            return
        requests.post(webhook, json={"text": message}, timeout=10).raise_for_status()
        print(f"\n  Slack 알람 발송 완료 ({len(alerts)}건)")
    except Exception as e:
        print(f"\n  Slack 알람 발송 실패: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='타겟 ETF 리스트 모니터링')
    parser.add_argument('--dry-run', action='store_true',
                        help='알람 미발송 (체크만)')
    args = parser.parse_args()

    check_etf_changes(dry_run=args.dry_run)
