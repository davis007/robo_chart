#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RCADX戦略 - RCI（短期）とADXを使用したトレンドフォロー戦略
短期RCIの傾きとADXの強度で売買シグナルを生成
"""

import pandas as pd
import numpy as np
import os
from typing import Dict


class Strategy:
    """RCADX戦略クラス"""

    def __init__(self,
                 rci_short_period: int = 9,        # 短期RCI期間
                 rci_slope_threshold: float = 50.0, # RCI傾き判定閾値（度）
                 adx_period: int = 14,             # ADX期間
                 adx_threshold: float = 30.0,      # ADX強度閾値
                 stop_loss_pct: float = 0.04,      # 損切り率（4%）
                 position_size: int = 100,         # トレードサイズ（100株固定）
                 adx_decline_threshold: float = 5.0, # ADX低下判定閾値
                 profit_target_amount: float = 5000.0, # 利益目標額（円）
                 profit_target_pct: float = 0.0,   # 利益目標率（%）
                 trailing_stop_pct: float = 0.04,  # トレーリングストップ率（2%）
                 partial_profit_pct: float = 0.03, # 部分利確率（3%）
                 partial_profit_ratio: float = 0.5, # 部分利確比率（50%）
                 rci_exit_threshold: float = -10.0): # RCI利確閾値（度）
        """
        初期化

        Args:
            rci_short_period: 短期RCIの期間
            rci_slope_threshold: RCI傾き判定の閾値（度）
            adx_period: ADX計算期間
            adx_threshold: ADX強度判定閾値
            stop_loss_pct: 損切り率（%）
            position_size: トレードサイズ（株数）
            adx_decline_threshold: ADX低下判定閾値（ポイント）
            profit_target_amount: 利益目標額（円）
            profit_target_pct: 利益目標率（%）
            trailing_stop_pct: トレーリングストップ率（%）
            partial_profit_pct: 部分利確率（%）
            partial_profit_ratio: 部分利確比率（0.0-1.0）
            rci_exit_threshold: RCI利確閾値（度）
        """
        self.rci_short_period = rci_short_period
        self.rci_slope_threshold = rci_slope_threshold
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.stop_loss_pct = stop_loss_pct
        self.position_size = position_size
        self.adx_decline_threshold = adx_decline_threshold
        self.profit_target_amount = profit_target_amount
        self.profit_target_pct = profit_target_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.partial_profit_pct = partial_profit_pct
        self.partial_profit_ratio = partial_profit_ratio
        self.rci_exit_threshold = rci_exit_threshold

        # 状態管理
        self.last_adx_value = None
        self.max_profit_price = None  # 最大利益価格（トレーリングストップ用）
        self.partial_profit_taken = False  # 部分利確済みフラグ

    def calculate_signal(
        self,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        ctx: dict
    ) -> dict:
        """
        売買シグナルを計算（RCADX戦略）

        Args:
            data: 株価データ（当日を含む直近90日分）
            current_date: 現在の日付
            ctx: コンテキスト情報 {
                "position": "LONG" or "FLAT",
                "entry_price": float or None,
                "size": int or 0
            }

        Returns:
            Dict: {'signal': 'BUY'/'SELL'/'HOLD', 'reason': '文字列'}
        """
        # 現在のインデックスを取得
        current_idx = data.index.get_loc(current_date)

        # 十分なデータがない場合はHOLD
        min_data_required = max(self.rci_short_period, self.adx_period)
        if current_idx < min_data_required:
            return {'signal': 'HOLD', 'reason': 'insufficient_data'}

        # カラム名の大文字小文字を考慮してOHLCを取得
        open_col = 'open' if 'open' in data.columns else 'Open'
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'
        close_col = 'close' if 'close' in data.columns else 'Close'

        # 現在の終値を取得
        current_close = data.iloc[current_idx][close_col]

        # 指標計算
        indicators = self._calculate_indicators(data, open_col, high_col, low_col, close_col, current_idx)
        if indicators is None:
            return {'signal': 'HOLD', 'reason': 'indicator_error'}

        # デバッグ出力
        if os.getenv('DEBUG_STRATEGY') == '1':
            self._debug_output(current_date, current_close, indicators)

        # ポジション状態に基づいてシグナル判定
        position = ctx.get("position", "FLAT")
        entry_price = ctx.get("entry_price")

        # FLAT状態（未保有）の場合
        if position == "FLAT":
            return self._evaluate_flat_position(current_date, current_close, indicators, ctx)

        # LONG状態（保有中）の場合
        elif position == "LONG":
            return self._evaluate_long_position(current_date, current_close, indicators, ctx, entry_price)

        # SHORT状態（保有中）の場合
        elif position == "SHORT":
            return self._evaluate_short_position(current_date, current_close, indicators, ctx, entry_price)

        # その他の状態（想定外）
        return {'signal': 'HOLD', 'reason': 'no_signal'}

    def _calculate_indicators(self, data: pd.DataFrame, open_col: str, high_col: str,
                            low_col: str, close_col: str, current_idx: int) -> Dict:
        """
        指標を計算（未来データ禁止）

        Args:
            data: 株価データ
            open_col: 始値カラム名
            high_col: 高値カラム名
            low_col: 安値カラム名
            close_col: 終値カラム名
            current_idx: 現在のインデックス

        Returns:
            Dict: 指標値の辞書
        """
        try:
            # 現在のデータ範囲（未来データ禁止）
            today_data = data.iloc[:current_idx+1]

            # RCI（短期）計算
            rci_short_current = self._calculate_rci(today_data, close_col, self.rci_short_period, current_idx)

            # 前日のRCI（短期）計算
            if current_idx > 0:
                yesterday_data = data.iloc[:current_idx]
                rci_short_yesterday = self._calculate_rci(yesterday_data, close_col, self.rci_short_period, current_idx-1)
            else:
                rci_short_yesterday = rci_short_current

            # RCI傾き計算（度数）
            rci_slope_degrees = self._calculate_rci_slope(rci_short_yesterday, rci_short_current)

            # ADX計算
            adx_dict = self._calculate_adx(today_data, high_col, low_col, close_col, self.adx_period, current_idx)
            adx_current, plus_di, minus_di = adx_dict["adx"], adx_dict["plus_di"], adx_dict["minus_di"]

            # 前日のADX
            if current_idx > 0:
                adx_dict_y = self._calculate_adx(data.iloc[:current_idx], high_col, low_col, close_col, self.adx_period, current_idx-1)
                adx_yesterday = adx_dict_y["adx"]
            else:
                adx_yesterday = adx_current

            # ADXの変化
            adx_change = adx_current - adx_yesterday if adx_yesterday is not None else 0

            # 現在のADXを状態として保存
            self.last_adx_value = adx_current

            return {
                'rci_short_current': rci_short_current,
                'rci_short_yesterday': rci_short_yesterday,
                'rci_slope_degrees': rci_slope_degrees,
                'adx_current': adx_current,
                'adx_yesterday': adx_yesterday,
                'adx_change': adx_change,
                'plus_di': plus_di,
                'minus_di': minus_di
            }

        except Exception as e:
            print(f"指標計算エラー: {e}")
            return None

    def _calculate_rci(self, data: pd.DataFrame, close_col: str, period: int, current_idx: int) -> float:
        """
        RCI（Rank Correlation Index）を計算

        Args:
            data: 株価データ
            close_col: 終値カラム名
            period: RCI期間
            current_idx: 現在のインデックス

        Returns:
            float: RCI値（-100〜100）
        """
        start_idx = current_idx - period + 1
        if start_idx < 0:
            start_idx = 0

        period_data = data.iloc[start_idx:current_idx + 1]
        closes = period_data[close_col].values

        if len(closes) <= 1:
            return 0.0

        # 日付順位（新しいほど高い）
        date_ranks = np.arange(1, len(closes) + 1)

        # 価格順位（高いほど高い）
        price_ranks = np.argsort(np.argsort(closes)) + 1

        # スピアマンの順位相関係数を計算
        n = len(closes)
        d_squared = np.sum((date_ranks - price_ranks) ** 2)
        rci = (1 - (6 * d_squared) / (n * (n**2 - 1))) * 100

        return rci

    def _calculate_rci_slope(self, rci_yesterday: float, rci_current: float) -> float:
        """
        RCIの傾きを度数で計算

        Args:
            rci_yesterday: 前日のRCI値
            rci_current: 現在のRCI値

        Returns:
            float: 傾き（度）
        """
        if rci_yesterday == rci_current:
            return 0.0

        # RCIの変化を度数に変換（最大変化±100を90度に対応）
        rci_change = rci_current - rci_yesterday
        slope_degrees = (rci_change / 100.0) * 90.0

        return slope_degrees

    def _calculate_adx(self, data: pd.DataFrame, high_col: str, low_col: str,
                      close_col: str, period: int, current_idx: int) -> dict:
        """
        ADX（Average Directional Index）、+DI、-DIを計算

        Args:
            data: 株価データ
            high_col: 高値カラム名
            low_col: 安値カラム名
            close_col: 終値カラム名
            period: ADX期間
            current_idx: 現在のインデックス

        Returns:
            dict: ADX、+DI、-DIの辞書
        """
        try:
            start_idx = max(0, current_idx - period * 2)
            adx_data = data.iloc[start_idx:current_idx+1]

            highs = adx_data[high_col].values
            lows = adx_data[low_col].values
            closes = adx_data[close_col].values

            if len(highs) < period + 1:
                return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

            plus_dm, minus_dm, tr_values = [], [], []
            for i in range(1, len(highs)):
                up_move = highs[i] - highs[i-1]
                down_move = lows[i-1] - lows[i]
                if up_move > down_move and up_move > 0:
                    plus_dm.append(up_move); minus_dm.append(0)
                elif down_move > up_move and down_move > 0:
                    plus_dm.append(0); minus_dm.append(down_move)
                else:
                    plus_dm.append(0); minus_dm.append(0)
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr_values.append(max(tr1, tr2, tr3))

            if len(plus_dm) >= period and len(minus_dm) >= period and len(tr_values) >= period:
                plus_dm_smooth = sum(plus_dm[:period]) / period
                minus_dm_smooth = sum(minus_dm[:period]) / period
                tr_smooth = sum(tr_values[:period]) / period

                for i in range(period, len(plus_dm)):
                    plus_dm_smooth = (plus_dm_smooth * (period-1) + plus_dm[i]) / period
                    minus_dm_smooth = (minus_dm_smooth * (period-1) + minus_dm[i]) / period
                    tr_smooth = (tr_smooth * (period-1) + tr_values[i]) / period

                plus_di = 100 * (plus_dm_smooth / tr_smooth) if tr_smooth != 0 else 0
                minus_di = 100 * (minus_dm_smooth / tr_smooth) if tr_smooth != 0 else 0

                dx_values = []
                for i in range(period, len(plus_dm)):
                    current_plus_dm_smooth = sum(plus_dm[i-period+1:i+1]) / period
                    current_minus_dm_smooth = sum(minus_dm[i-period+1:i+1]) / period
                    current_tr_smooth = sum(tr_values[i-period+1:i+1]) / period
                    current_plus_di = 100 * (current_plus_dm_smooth / current_tr_smooth) if current_tr_smooth != 0 else 0
                    current_minus_di = 100 * (current_minus_dm_smooth / current_tr_smooth) if current_tr_smooth != 0 else 0
                    current_dx = 100 * abs(current_plus_di - current_minus_di) / (current_plus_di + current_minus_di) if (current_plus_di + current_minus_di) != 0 else 0
                    dx_values.append(current_dx)

                adx = sum(dx_values) / len(dx_values) if dx_values else 0
                return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}

            return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}
        except Exception as e:
            print(f"ADX計算エラー: {e}")
            return {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

    def _evaluate_flat_position(self, current_date: pd.Timestamp, current_close: float,
                               indicators: Dict, ctx: dict) -> dict:
        """
        FLAT状態でのシグナル評価（RCI強度・ゾーン分離版）

        Args:
            current_date: 現在の日付
            current_close: 現在の終値
            indicators: 指標値
            ctx: コンテキスト情報

        Returns:
            dict: シグナル
        """
        rci_slope_degrees = indicators['rci_slope_degrees']
        adx_current = indicators['adx_current']
        plus_di = indicators['plus_di']
        minus_di = indicators['minus_di']
        rci_short_current = indicators['rci_short_current']
        rci_strength = abs(rci_short_current)

        # ADX強度チェック
        if adx_current < self.adx_threshold:
            return {'signal': 'HOLD', 'reason': 'ADX弱'}

        # ロングエントリー条件：RCI短期が正、強度が閾値以上、+DI > -DI かつ +DI が -DI の1.2倍以上
        if (
            rci_short_current > 0
            and rci_strength >= self.rci_slope_threshold
            and plus_di > minus_di
            and plus_di >= minus_di * 1.2
        ):
            ratio = plus_di / minus_di if minus_di != 0 else float('inf')
            return {
                'signal': 'BUY',
                'reason': f'RCI+:BUY(strength={rci_strength:.1f}, +DI={plus_di:.1f}, -DI={minus_di:.1f}, +DI/-DI={ratio:.2f} >= 1.20)',
                'rci_value': rci_short_current,
                'rci_strength': rci_strength,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'adx_value': adx_current
            }

        # ショートエントリー条件：RCI短期が負、強度が閾値以上、-DI > +DI
        if (
            rci_short_current < 0
            and rci_strength >= self.rci_slope_threshold
            and minus_di > plus_di
        ):
            return {
                'signal': 'SELL',
                'reason': f'RCI-:SELL(strength={rci_strength:.1f}, +DI={plus_di:.1f}, -DI={minus_di:.1f})',
                'rci_value': rci_short_current,
                'rci_strength': rci_strength,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'adx_value': adx_current
            }

        return {'signal': 'HOLD', 'reason': 'no_signal'}

    def _evaluate_long_position(self, current_date: pd.Timestamp, current_close: float,
                               indicators: Dict, ctx: dict, entry_price: float) -> dict:
        """
        LONG状態でのシグナル評価

        Args:
            current_date: 現在の日付
            current_close: 現在の終値
            indicators: 指標値
            ctx: コンテキスト情報
            entry_price: エントリー価格

        Returns:
            dict: シグナル
        """
        # 損切り判定（最優先）
        if current_close <= entry_price * (1 - self.stop_loss_pct):
            return {'signal': 'SELL', 'reason': 'stop_loss'}

        # 利益目標額による利確（+5000円以上）
        profit_amount = (current_close - entry_price) * self.position_size
        if profit_amount >= self.profit_target_amount:
            return {'signal': 'SELL', 'reason': f'利益目標額達成({profit_amount:.0f}円)'}

        # 利益目標率による利確（設定されている場合）
        if self.profit_target_pct > 0:
            profit_pct = (current_close - entry_price) / entry_price
            if profit_pct >= self.profit_target_pct:
                return {'signal': 'SELL', 'reason': f'利益目標率達成({profit_pct*100:.1f}%)'}

        # 部分利確（+3%で半分利確）
        if not self.partial_profit_taken:
            profit_pct = (current_close - entry_price) / entry_price
            if profit_pct >= self.partial_profit_pct:
                self.partial_profit_taken = True
                return {'signal': 'SELL', 'reason': f'部分利確({profit_pct*100:.1f}%)', 'partial_profit': True}

        # トレーリングストップ（最大利益から-2%で決済）
        if self.max_profit_price is None or current_close > self.max_profit_price:
            self.max_profit_price = current_close

        if self.max_profit_price is not None:
            trailing_stop_price = self.max_profit_price * (1 - self.trailing_stop_pct)
            if current_close <= trailing_stop_price:
                return {'signal': 'SELL', 'reason': f'トレーリングストップ({self.trailing_stop_pct*100:.1f}%)'}

        rci_slope_degrees = indicators['rci_slope_degrees']
        adx_current = indicators['adx_current']
        adx_change = indicators['adx_change']

        # RCI傾き判定による利確（RCIが下向きに転じたら早めに利確）
        if rci_slope_degrees <= self.rci_exit_threshold:
            return {'signal': 'SELL', 'reason': f'RCI傾き利確({rci_slope_degrees:.1f}度)'}

        # ADX低下によるトレンド消失検知
        if (self.last_adx_value is not None and
            adx_current < self.last_adx_value - self.adx_decline_threshold):
            return {'signal': 'SELL', 'reason': f'ADX低下({adx_current:.1f})'}

        # 逆方向RCIシグナルによる利確
        if rci_slope_degrees <= -self.rci_slope_threshold:
            return {'signal': 'SELL', 'reason': f'RCI:SELL({rci_slope_degrees:.1f}度)'}

        return {'signal': 'HOLD', 'reason': 'no_signal'}

    def _evaluate_short_position(self, current_date: pd.Timestamp, current_close: float,
                                indicators: Dict, ctx: dict, entry_price: float) -> dict:
        """
        SHORT状態でのシグナル評価

        Args:
            current_date: 現在の日付
            current_close: 現在の終値
            indicators: 指標値
            ctx: コンテキスト情報
            entry_price: エントリー価格

        Returns:
            dict: シグナル
        """
        # 損切り判定（最優先）
        if current_close >= entry_price * (1 + self.stop_loss_pct):
            return {'signal': 'BUY', 'reason': 'stop_loss'}

        # 利益目標額による利確（+5000円以上）
        profit_amount = (entry_price - current_close) * self.position_size
        if profit_amount >= self.profit_target_amount:
            return {'signal': 'BUY', 'reason': f'利益目標額達成({profit_amount:.0f}円)'}

        # 利益目標率による利確（設定されている場合）
        if self.profit_target_pct > 0:
            profit_pct = (entry_price - current_close) / entry_price
            if profit_pct >= self.profit_target_pct:
                return {'signal': 'BUY', 'reason': f'利益目標率達成({profit_pct*100:.1f}%)'}

        # 部分利確（+3%で半分利確）
        if not self.partial_profit_taken:
            profit_pct = (entry_price - current_close) / entry_price
            if profit_pct >= self.partial_profit_pct:
                self.partial_profit_taken = True
                return {'signal': 'BUY', 'reason': f'部分利確({profit_pct*100:.1f}%)', 'partial_profit': True}

        # トレーリングストップ（最大利益から-2%で決済）
        if self.max_profit_price is None or current_close < self.max_profit_price:
            self.max_profit_price = current_close

        if self.max_profit_price is not None:
            trailing_stop_price = self.max_profit_price * (1 + self.trailing_stop_pct)
            if current_close >= trailing_stop_price:
                return {'signal': 'BUY', 'reason': f'トレーリングストップ({self.trailing_stop_pct*100:.1f}%)'}

        rci_slope_degrees = indicators['rci_slope_degrees']
        adx_current = indicators['adx_current']
        adx_change = indicators['adx_change']

        # RCI傾き判定による利確（RCIが上向きに転じたら早めに利確）
        if rci_slope_degrees >= -self.rci_exit_threshold:
            return {'signal': 'BUY', 'reason': f'RCI傾き利確({rci_slope_degrees:.1f}度)'}

        # ADX低下によるトレンド消失検知
        if (self.last_adx_value is not None and
            adx_current < self.last_adx_value - self.adx_decline_threshold):
            return {'signal': 'BUY', 'reason': f'ADX低下({adx_current:.1f})'}

        # 逆方向RCIシグナルによる利確
        if rci_slope_degrees >= self.rci_slope_threshold:
            return {'signal': 'BUY', 'reason': f'RCI:BUY({rci_slope_degrees:.1f}度)'}

        return {'signal': 'HOLD', 'reason': 'no_signal'}

    def _debug_output(self, current_date: pd.Timestamp, current_close: float, indicators: Dict):
        """
        デバッグ出力

        Args:
            current_date: 現在の日付
            current_close: 現在の終値
            indicators: 指標値
        """
        rci_short_current = indicators['rci_short_current']
        rci_short_yesterday = indicators['rci_short_yesterday']
        rci_slope_degrees = indicators['rci_slope_degrees']
        adx_current = indicators['adx_current']
        adx_yesterday = indicators['adx_yesterday']
        adx_change = indicators['adx_change']

        print(f"DEBUG: {current_date.strftime('%Y-%m-%d')} "
              f"close={current_close:.0f} "
              f"RCI={rci_short_current:.1f} "
              f"RCI傾き={rci_slope_degrees:.1f}度 "
              f"ADX={adx_current:.1f} "
              f"+DI={indicators['plus_di']:.1f} "
              f"-DI={indicators['minus_di']:.1f} "
              f"ADX変化={adx_change:+.1f}")


# テスト用
if __name__ == "__main__":
    # テストデータの作成
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    test_data = pd.DataFrame({
        'Open': [1000, 1010, 1020, 1030, 1040, 1030, 1020, 1010, 1000, 990,
                 980, 970, 960, 950, 940, 950, 960, 970, 980, 990,
                 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090],
        'High': [1010, 1020, 1030, 1040, 1050, 1040, 1030, 1020, 1010, 1000,
                 990, 980, 970, 960, 950, 960, 970, 980, 990, 1000,
                 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100],
        'Low': [990, 1000, 1010, 1020, 1030, 1020, 1010, 1000, 990, 980,
                970, 960, 950, 940, 930, 940, 950, 960, 970, 980,
                990, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080],
        'Close': [1005, 1015, 1025, 1035, 1045, 1035, 1025, 1015, 1005, 995,
                  985, 975, 965, 955, 945, 955, 965, 975, 985, 995,
                  1005, 1015, 1025, 1035, 1045, 1055, 1065, 1075, 1085, 1095],
        'Volume': [1000000] * 30
    }, index=dates)

    print("=== 基本テスト ===")
    strategy = Strategy()
    last_date = dates[-1]
    ctx = {"position": "FLAT", "entry_price": None, "size": 0}
    signal = strategy.calculate_signal(test_data, last_date, ctx)
    print(f"最終日のシグナル: {signal}")

    print("\n=== 利益目標額テスト ===")
    # 利益目標額が5000円を超えるケースをテスト
    strategy_profit = Strategy(profit_target_amount=5000.0)
    ctx_long = {"position": "LONG", "entry_price": 1000.0, "size": 100}
    # 現在価格が1050円の場合、利益額 = (1050-1000)*100 = 5000円
    test_data_profit = test_data.copy()
    test_data_profit.iloc[-1, test_data_profit.columns.get_loc('Close')] = 1050.0
    signal_profit = strategy_profit.calculate_signal(test_data_profit, last_date, ctx_long)
    print(f"利益目標額テスト: {signal_profit}")

    print("\n=== 利益目標率テスト ===")
    # 利益目標率が5%を超えるケースをテスト
    strategy_pct = Strategy(profit_target_pct=0.05)
    ctx_long_pct = {"position": "LONG", "entry_price": 1000.0, "size": 100}
    # 現在価格が1050円の場合、利益率 = (1050-1000)/1000 = 5%
    signal_pct = strategy_pct.calculate_signal(test_data_profit, last_date, ctx_long_pct)
    print(f"利益目標率テスト: {signal_pct}")

    print("\n=== 部分利確テスト ===")
    # 部分利確が3%を超えるケースをテスト
    strategy_partial = Strategy(partial_profit_pct=0.03)
    ctx_long_partial = {"position": "LONG", "entry_price": 1000.0, "size": 100}
    # 現在価格が1030円の場合、利益率 = (1030-1000)/1000 = 3%
    test_data_partial = test_data.copy()
    test_data_partial.iloc[-1, test_data_partial.columns.get_loc('Close')] = 1030.0
    signal_partial = strategy_partial.calculate_signal(test_data_partial, last_date, ctx_long_partial)
    print(f"部分利確テスト: {signal_partial}")

    print("\n=== RCI傾き利確テスト ===")
    # RCI傾きが-10度を下回るケースをテスト
    strategy_rci = Strategy(rci_exit_threshold=-10.0)
    ctx_long_rci = {"position": "LONG", "entry_price": 1000.0, "size": 100}
    # RCI傾きが-15度の場合
    test_data_rci = test_data.copy()
    test_data_rci.iloc[-1, test_data_rci.columns.get_loc('Close')] = 1020.0
    # RCI傾きを強制的に-15度に設定（テスト用）
    strategy_rci.rci_slope_degrees = -15.0
    signal_rci = strategy_rci.calculate_signal(test_data_rci, last_date, ctx_long_rci)
    print(f"RCI傾き利確テスト: {signal_rci}")
