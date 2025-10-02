#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グランビルの法則戦略 - 月間2〜4回トレードを目指す
基準EMAと価格の関係に基づく売買シグナル生成
"""

import pandas as pd
import numpy as np
import os
from typing import Dict


class Strategy:
    """グランビルの法則戦略クラス"""

    def __init__(self,
                 baseline_ma_period: int = 12,  # 15 → 12に短縮
                 short_ma_period: int = 5,
                 rci_short_period: int = 9,
                 touch_band_pct: float = 0.05,  # 3.0% → 5.0%に緩和
                 slope_lookback: int = 2,
                 stop_loss_pct: float = 0.03,
                 cooldown_days: int = 1,        # 2 → 1に短縮
                 min_days_in_trade: int = 2):
        """
        初期化

        Args:
            baseline_ma_period: グランビル判定の基準MA（EMA）
            short_ma_period: ノイズ抑制用の補助MA
            rci_short_period: 補助指標RCIの期間
            touch_band_pct: MA接触判定の許容帯（±%）
            slope_lookback: MA傾き判定の期間
            stop_loss_pct: 損切り率（%）
            cooldown_days: 決済後のクールダウン日数
            min_days_in_trade: 最低保有日数
        """
        self.baseline_ma_period = baseline_ma_period
        self.short_ma_period = short_ma_period
        self.rci_short_period = rci_short_period
        self.touch_band_pct = touch_band_pct
        self.slope_lookback = slope_lookback
        self.stop_loss_pct = stop_loss_pct
        self.cooldown_days = cooldown_days
        self.min_days_in_trade = min_days_in_trade

        # 状態管理
        self.last_sell_date = None
        self.entry_date = None

    def calculate_signal(
        self,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        ctx: dict
    ) -> dict:
        """
        売買シグナルを計算（グランビルの法則）

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
        if current_idx < max(self.baseline_ma_period, self.slope_lookback, self.rci_short_period):
            return {'signal': 'HOLD', 'reason': 'insufficient_data'}

        # カラム名の大文字小文字を考慮して終値を取得
        close_col = 'close' if 'close' in data.columns else 'Close'

        # 現在の終値を取得
        current_close = data.iloc[current_idx][close_col]

        # 指標計算
        indicators = self._calculate_indicators(data, close_col, current_idx)
        if indicators is None:
            return {'signal': 'HOLD', 'reason': 'indicator_error'}

        # デバッグ出力
        if os.getenv('DEBUG_STRATEGY') == '1':
            self._debug_output(current_date, current_close, indicators)

        # ポジション状態に基づいてシグナル判定
        position = ctx.get("position", "FLAT")
        entry_price = ctx.get("entry_price")
        size = ctx.get("size", 0)

        # 状態更新
        if position == "LONG" and self.entry_date is None:
            self.entry_date = current_date
        elif position == "FLAT":
            self.entry_date = None

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

    def _calculate_indicators(self, data: pd.DataFrame, close_col: str, current_idx: int) -> Dict:
        """
        指標を計算（未来データ禁止）

        Args:
            data: 株価データ
            close_col: 終値カラム名
            current_idx: 現在のインデックス

        Returns:
            Dict: 指標値の辞書
        """
        try:
            # 現在と前日のデータ範囲
            today_data = data.iloc[:current_idx+1]
            yesterday_data = data.iloc[:current_idx] if current_idx > 0 else None

            # 基準EMA（baseline_ma_period）
            baseline_ema_today = today_data[close_col].ewm(
                span=self.baseline_ma_period, adjust=False
            ).mean().iloc[-1]

            # 前日の基準EMA
            baseline_ema_yesterday = yesterday_data[close_col].ewm(
                span=self.baseline_ma_period, adjust=False
            ).mean().iloc[-1] if yesterday_data is not None and len(yesterday_data) >= self.baseline_ma_period else None

            # 傾き判定用の過去EMA
            if current_idx >= self.slope_lookback:
                slope_data = data.iloc[:current_idx+1]
                baseline_ema_current = slope_data[close_col].ewm(
                    span=self.baseline_ma_period, adjust=False
                ).mean().iloc[-1]

                slope_past_idx = current_idx - self.slope_lookback
                slope_past_data = data.iloc[:slope_past_idx+1]
                baseline_ema_past = slope_past_data[close_col].ewm(
                    span=self.baseline_ma_period, adjust=False
                ).mean().iloc[-1] if len(slope_past_data) >= self.baseline_ma_period else baseline_ema_current
            else:
                baseline_ema_past = baseline_ema_today

            # 傾き判定
            ma_slope = "UP" if baseline_ema_today > baseline_ema_past else "DOWN"

            # 現在と前日の終値
            current_close = data.iloc[current_idx][close_col]
            prev_close = data.iloc[current_idx-1][close_col] if current_idx > 0 else current_close

            # 接触判定
            touch_today = self._is_touching_ma(current_close, baseline_ema_today)
            touch_yesterday = self._is_touching_ma(prev_close, baseline_ema_yesterday) if baseline_ema_yesterday else False

            # 補助指標（オプション）
            rci_short = self._calculate_rci(data, close_col, self.rci_short_period, current_idx)

            return {
                'baseline_ema_today': baseline_ema_today,
                'baseline_ema_yesterday': baseline_ema_yesterday,
                'ma_slope': ma_slope,
                'current_close': current_close,
                'prev_close': prev_close,
                'touch_today': touch_today,
                'touch_yesterday': touch_yesterday,
                'rci_short': rci_short
            }

        except Exception as e:
            return None

    def _is_touching_ma(self, close: float, ma: float) -> bool:
        """
        MA接触判定

        Args:
            close: 終値
            ma: 移動平均値

        Returns:
            bool: 接触帯内ならTrue
        """
        if ma == 0:
            return False
        deviation = abs(close - ma) / ma
        return deviation <= self.touch_band_pct

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

    def _evaluate_flat_position(self, current_date: pd.Timestamp, current_close: float,
                               indicators: Dict, ctx: dict) -> dict:
        """
        FLAT状態でのシグナル評価

        Args:
            current_date: 現在の日付
            current_close: 現在の終値
            indicators: 指標値
            ctx: コンテキスト情報

        Returns:
            dict: シグナル
        """
        # クールダウン判定
        if self.last_sell_date is not None:
            days_since_sell = (current_date - self.last_sell_date).days
            if days_since_sell < self.cooldown_days:
                return {'signal': 'HOLD', 'reason': 'cooldown'}

        # グランビル買いシグナル判定（優先順位順）
        baseline_ema_today = indicators['baseline_ema_today']
        baseline_ema_yesterday = indicators['baseline_ema_yesterday']
        ma_slope = indicators['ma_slope']
        prev_close = indicators['prev_close']
        touch_yesterday = indicators['touch_yesterday']

        # G1: 下から上への明確ブレイク
        if (baseline_ema_yesterday is not None and
            prev_close <= baseline_ema_yesterday and
            current_close > baseline_ema_today and
            ma_slope == "UP"):
            return {'signal': 'BUY', 'reason': 'granville_buy_1'}

        # G2: MA付近での反発（押し目買い）
        if (baseline_ema_yesterday is not None and
            prev_close >= baseline_ema_yesterday and
            touch_yesterday and
            current_close > prev_close and
            ma_slope == "UP"):
            return {'signal': 'BUY', 'reason': 'granville_buy_2'}

        # G3: MA上方維持中の反発強化
        if (baseline_ema_yesterday is not None and
            prev_close > baseline_ema_yesterday and
            current_close > baseline_ema_today and
            current_close > prev_close and
            ma_slope == "UP"):
            return {'signal': 'BUY', 'reason': 'granville_buy_3'}

        # G4: 一時割れ後の早期回復
        if (baseline_ema_yesterday is not None and
            prev_close < baseline_ema_yesterday and
            current_close >= baseline_ema_today and
            ma_slope == "UP"):
            return {'signal': 'BUY', 'reason': 'granville_buy_4'}

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
            self.last_sell_date = current_date
            return {'signal': 'SELL', 'reason': 'stop_loss'}

        # 最低保有日数チェック
        if self.entry_date is not None:
            days_in_trade = (current_date - self.entry_date).days
            if days_in_trade < self.min_days_in_trade:
                return {'signal': 'HOLD', 'reason': 'min_days_protection'}

        baseline_ema_today = indicators['baseline_ema_today']
        baseline_ema_yesterday = indicators['baseline_ema_yesterday']
        ma_slope = indicators['ma_slope']
        prev_close = indicators['prev_close']
        touch_yesterday = indicators['touch_yesterday']

        # S5: 上から下への明確ブレイク
        if (baseline_ema_yesterday is not None and
            prev_close >= baseline_ema_yesterday and
            current_close < baseline_ema_today and
            ma_slope == "DOWN"):
            self.last_sell_date = current_date
            return {'signal': 'SELL', 'reason': 'granville_sell_5'}

        # S6: MA付近での反落（戻り売り）
        if (baseline_ema_yesterday is not None and
            prev_close <= baseline_ema_yesterday and
            touch_yesterday and
            current_close < prev_close and
            ma_slope == "DOWN"):
            self.last_sell_date = current_date
            return {'signal': 'SELL', 'reason': 'granville_sell_6'}

        # S7: MA下方維持中の反落継続
        if (baseline_ema_yesterday is not None and
            prev_close < baseline_ema_yesterday and
            current_close < baseline_ema_today and
            current_close < prev_close and
            ma_slope == "DOWN"):
            self.last_sell_date = current_date
            return {'signal': 'SELL', 'reason': 'granville_sell_7'}

        # S8: 一時上抜け後の失速
        if (baseline_ema_yesterday is not None and
            prev_close > baseline_ema_yesterday and
            current_close <= baseline_ema_today and
            ma_slope == "DOWN"):
            self.last_sell_date = current_date
            return {'signal': 'SELL', 'reason': 'granville_sell_8'}

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
        # 損切り判定（最優先） - ショートの場合は価格上昇で損失
        if current_close >= entry_price * (1 + self.stop_loss_pct):
            self.last_sell_date = current_date
            return {'signal': 'BUY', 'reason': 'stop_loss'}

        # 最低保有日数チェック
        if self.entry_date is not None:
            days_in_trade = (current_date - self.entry_date).days
            if days_in_trade < self.min_days_in_trade:
                return {'signal': 'HOLD', 'reason': 'min_days_protection'}

        baseline_ema_today = indicators['baseline_ema_today']
        baseline_ema_yesterday = indicators['baseline_ema_yesterday']
        ma_slope = indicators['ma_slope']
        prev_close = indicators['prev_close']
        touch_yesterday = indicators['touch_yesterday']

        # ショート解消シグナル（グランビル買いシグナルを流用）
        # G1: 下から上への明確ブレイク
        if (baseline_ema_yesterday is not None and
            prev_close <= baseline_ema_yesterday and
            current_close > baseline_ema_today and
            ma_slope == "UP"):
            self.last_sell_date = current_date
            return {'signal': 'BUY', 'reason': 'granville_buy_1'}

        # G2: MA付近での反発（押し目買い）
        if (baseline_ema_yesterday is not None and
            prev_close >= baseline_ema_yesterday and
            touch_yesterday and
            current_close > prev_close and
            ma_slope == "UP"):
            self.last_sell_date = current_date
            return {'signal': 'BUY', 'reason': 'granville_buy_2'}

        # G3: MA上方維持中の反発強化
        if (baseline_ema_yesterday is not None and
            prev_close > baseline_ema_yesterday and
            current_close > baseline_ema_today and
            current_close > prev_close and
            ma_slope == "UP"):
            self.last_sell_date = current_date
            return {'signal': 'BUY', 'reason': 'granville_buy_3'}

        # G4: 一時割れ後の早期回復
        if (baseline_ema_yesterday is not None and
            prev_close < baseline_ema_yesterday and
            current_close >= baseline_ema_today and
            ma_slope == "UP"):
            self.last_sell_date = current_date
            return {'signal': 'BUY', 'reason': 'granville_buy_4'}

        return {'signal': 'HOLD', 'reason': 'no_signal'}

    def _debug_output(self, current_date: pd.Timestamp, current_close: float, indicators: Dict):
        """
        デバッグ出力

        Args:
            current_date: 現在の日付
            current_close: 現在の終値
            indicators: 指標値
        """
        baseline_ema_today = indicators['baseline_ema_today']
        baseline_ema_yesterday = indicators['baseline_ema_yesterday']
        ma_slope = indicators['ma_slope']
        prev_close = indicators['prev_close']
        touch_today = indicators['touch_today']
        touch_yesterday = indicators['touch_yesterday']

        print(f"DEBUG: {current_date.strftime('%Y-%m-%d')} "
              f"close={current_close:.0f} "
              f"MA{self.baseline_ma_period}={baseline_ema_today:.0f} "
              f"slope={ma_slope} "
              f"touch_today={touch_today} "
              f"touch_yesterday={touch_yesterday}")
