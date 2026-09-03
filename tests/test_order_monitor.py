import unittest
from datetime import datetime, timedelta

import pandas as pd

from quantylab.trainer.etf_single_swing.swing_trading import _limit_price, monitor_limit_orders


class FakeClock:
    def __init__(self, value):
        self.value = value

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += timedelta(seconds=seconds)


class FakeClient:
    def __init__(self):
        self.open_calls = 0
        self.cancelled = []
        self.buys = []

    def get_open_orders(self):
        self.open_calls += 1
        if self.open_calls <= 2:
            return pd.DataFrame([{'주문번호': 'A1', '미체결수량': 4}])
        return pd.DataFrame()

    def cancel_order(self, ord_no, code, cncl_qty='0'):
        self.cancelled.append((ord_no, code, cncl_qty))
        return {'return_code': '0'}

    def buy_order(self, code, qty, price=0, trde_tp='3'):
        self.buys.append((code, qty, price, trde_tp))
        return {'return_code': '0', 'ord_no': 'A2'}


class StickyClient(FakeClient):
    def get_open_orders(self):
        return pd.DataFrame([{'주문번호': 'A1', '미체결수량': 7}])


class OrderMonitorTest(unittest.TestCase):
    def test_prices_stay_in_band_and_move_toward_fill(self):
        self.assertEqual(_limit_price(10_000, 'buy', 0.005, 0), 9_950)
        self.assertEqual(_limit_price(10_000, 'buy', 0.005, 1), 10_050)
        self.assertEqual(_limit_price(10_000, 'sell', 0.005, 0), 10_050)
        self.assertEqual(_limit_price(10_000, 'sell', 0.005, 1), 9_950)

    def test_partial_fill_is_cancelled_and_replaced_as_one_batch(self):
        clock = FakeClock(datetime(2026, 8, 19, 15, 19, 0))
        client = FakeClient()
        orders = [{
            'success': True, 'ord_no': 'A1', 'code': '069500', 'qty': 10,
            'side': 'buy', 'reference_price': 10_000, 'order_price': 9_950,
        }]

        monitor_limit_orders(
            client, orders, band_pct=0.005, poll_seconds=20, end_time='15:20',
            now_fn=clock.now, sleep_fn=clock.sleep,
        )

        self.assertEqual(client.cancelled, [('A1', '069500', '4')])
        self.assertEqual(client.buys, [('069500', 4, 9_980, '0')])
        self.assertEqual(client.open_calls, 4)
        self.assertTrue(orders[0]['filled'])
        self.assertEqual(orders[0]['remaining_qty'], 0)
        self.assertEqual(orders[0]['final_ord_no'], 'A2')
        self.assertEqual(orders[0]['final_order_price'], 9_980)

    def test_signal_invalidation_cancels_only_current_remaining_quantity(self):
        clock = FakeClock(datetime(2026, 8, 19, 10, 0, 0))
        client = StickyClient()
        orders = [{
            'success': True, 'ord_no': 'A1', 'code': '069500', 'qty': 10,
            'side': 'buy', 'reference_price': 10_000, 'order_price': 9_950,
        }]

        monitor_limit_orders(
            client, orders, band_pct=0.005, poll_seconds=20, end_time='10:30',
            signal_check_fn=lambda order, now: False,
            now_fn=clock.now, sleep_fn=clock.sleep,
        )

        self.assertEqual(client.cancelled, [('A1', '069500', '7')])
        self.assertEqual(orders[0]['cancel_reason'], 'signal_invalid')


if __name__ == '__main__':
    unittest.main()
