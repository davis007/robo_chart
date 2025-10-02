#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
チャート分析スクリプト - 244項目の特徴量抽出とJSON出力
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import jsonschema
from pathlib import Path

# 既存のDataManagerをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest import DataManager


class ChartAnalyzer:
    """チャート分析クラス - 244項目の特徴量を抽出"""

    def __init__(self, years: int = 5, min_years: int = 3):
        self.years = years
        self.min_years = min_years
        self.data_manager = DataManager()

        # 乱数シード固定（再現性確保）
        np.random.seed(42)

    def analyze_symbol(self, symbol: str, start_date: str = None, end_date: str = None, force: bool = False) -> bool:
        """
        銘柄を分析して特徴量を抽出

        Args:
            symbol: 銘柄コード
            start_date: 開始日（指定なしの場合は過去years年）
            end_date: 終了日（指定なしの場合は今日）
            force: 強制実行フラグ

        Returns:
            bool: 成功時True、失敗時False
        """
        print(f"銘柄分析開始: {symbol}")

        # 日付範囲の設定
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=self.years * 365)
            start_date = start_dt.strftime('%Y-%m-%d')

        print(f"期間: {start_date} 〜 {end_date}")

        # 株価データ取得
        data = self.data_manager.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            print(f"データが取得できませんでした: {symbol}")
            return False

        # データ期間チェック
        data_years = (data.index[-1] - data.index[0]).days / 365.0
        if data_years < self.min_years:
            print(f"学習できません！データ期間が{data_years:.1f}年（最低{self.min_years}年必要）")
            return False

        print(f"取得データ: {len(data)}日分 ({data.index[0].strftime('%Y-%m-%d')} 〜 {data.index[-1].strftime('%Y-%m-%d')})")

        # 期間分割（学習4年＋検証1年）
        split_info = self._split_data(data)
        if not split_info:
            print("期間分割に失敗しました")
            return False

        # 特徴量計算
        print("特徴量計算中...")
        features_data = self._calculate_all_features(data, split_info)

        if not features_data:
            print("特徴量計算に失敗しました")
            return False

        # JSON出力
        output_path = self._save_to_json(symbol, features_data, split_info)

        if output_path:
            print(f"分析完了: {output_path}")
            return True
        else:
            print("JSON出力に失敗しました")
            return False

    def _split_data(self, data: pd.DataFrame) -> Optional[Dict]:
        """データを学習期間と検証期間に分割"""
        if len(data) < 252 * self.min_years:  # 最低営業日数チェック
            return None

        # 学習期間: 過去4年、検証期間: 直近1年
        split_idx = int(len(data) * 0.8)  # 80%を学習期間
        split_date = data.index[split_idx]

        train_data = data.iloc[:split_idx]
        valid_data = data.iloc[split_idx:]

        if len(train_data) < 252 * 3 or len(valid_data) < 252 * 0.5:  # 最低期間チェック
            return None

        return {
            "train_start": train_data.index[0].strftime('%Y-%m-%d'),
            "train_end": train_data.index[-1].strftime('%Y-%m-%d'),
            "valid_start": valid_data.index[0].strftime('%Y-%m-%d'),
            "valid_end": valid_data.index[-1].strftime('%Y-%m-%d')
        }

    def _calculate_all_features(self, data: pd.DataFrame, split_info: Dict) -> Optional[Dict]:
        """全244項目の特徴量を計算"""
        try:
            # 学習期間データ（インデックスで直接抽出）
            train_start = pd.to_datetime(split_info["train_start"])
            train_end = pd.to_datetime(split_info["train_end"])
            train_data = data[(data.index >= train_start) & (data.index <= train_end)]

            if len(train_data) < 252:  # 最低1年分のデータ
                return None

            features = {
                "returns": self._calculate_return_features(train_data),
                "trend": self._calculate_trend_features(train_data),
                "vol": self._calculate_volatility_features(train_data),
                "osc": self._calculate_oscillator_features(train_data),
                "candle": self._calculate_candle_features(train_data),
                "volume": self._calculate_volume_features(train_data),
                "streak": self._calculate_streak_features(train_data),
                "dd": self._calculate_drawdown_features(train_data),
                "regime": self._calculate_regime_features(train_data),
                "pricepos": self._calculate_price_position_features(train_data)
            }

            # グローバル統計
            stats = self._calculate_global_stats(train_data)

            # 観察メモ
            notes = self._generate_notes(features, stats)

            return {
                "stats": stats,
                "features": features,
                "notes": notes
            }

        except Exception as e:
            print(f"特徴量計算エラー: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_global_stats(self, data: pd.DataFrame) -> Dict:
        """グローバル統計指標を計算"""
        close_prices = data['close']

        # 基本統計
        total_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
        cagr = (1 + total_return) ** (1 / (len(data) / 252)) - 1

        # 最大ドローダウン
        rolling_max = close_prices.expanding().max()
        drawdowns = (close_prices - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        max_dd_idx = drawdowns.idxmin()
        max_dd_start_idx = rolling_max[rolling_max == rolling_max.loc[:max_dd_idx].max()].index[-1]

        # 回復日数
        recovery_mask = close_prices.loc[max_dd_idx:] >= rolling_max.loc[max_dd_start_idx]
        if recovery_mask.any():
            recovery_idx = recovery_mask.idxmax()
            recovery_days = (recovery_idx - max_dd_idx).days
        else:
            recovery_days = len(data)

        # 年次リターン
        yearly_returns = []
        for year in range(data.index.year.min(), data.index.year.max() + 1):
            year_data = data[data.index.year == year]
            if len(year_data) > 0:
                year_return = (year_data['close'].iloc[-1] / year_data['close'].iloc[0]) - 1
                yearly_returns.append(year_return)

        return {
            "growth_5y": total_return,
            "cagr_5y": cagr,
            "max_dd_5y": max_dd,
            "max_dd_start_date": max_dd_start_idx.strftime('%Y-%m-%d'),
            "max_dd_end_date": max_dd_idx.strftime('%Y-%m-%d'),
            "dd_recovery_days_max": recovery_days,
            "yearly_return_mean": np.mean(yearly_returns) if yearly_returns else 0,
            "yearly_return_std": np.std(yearly_returns) if yearly_returns else 0
        }

    def _calculate_return_features(self, data: pd.DataFrame) -> Dict:
        """リターン系特徴量（54項目）"""
        close_prices = data['close']
        returns = {}

        # 期間設定
        windows = [1, 5, 10, 20, 60, 120, 252]

        for window in windows:
            if len(data) >= window:
                # リターン計算
                returns_series = close_prices.pct_change(window)

                # 基本統計
                returns[f"r_mean_w{window}"] = returns_series.mean()
                returns[f"r_std_w{window}"] = returns_series.std()
                returns[f"r_min_w{window}"] = returns_series.min()
                returns[f"r_max_w{window}"] = returns_series.max()

                # 分位点（w1以外）
                if window > 1:
                    for p in [10, 25, 50, 75, 90]:
                        returns[f"r_p{p}_w{window}"] = returns_series.quantile(p/100)
                else:
                    # w1は特別扱い
                    returns["r_p50_w1"] = returns_series.quantile(0.5)
                    returns["r_p90_w1"] = returns_series.quantile(0.9)

        return returns

    def _calculate_trend_features(self, data: pd.DataFrame) -> Dict:
        """トレンド系特徴量（40項目）"""
        close_prices = data['close']
        trend = {}

        # 移動平均期間
        ma_windows = [5, 10, 20, 50, 100, 200]

        for window in ma_windows:
            if len(data) >= window:
                # SMA
                sma = close_prices.rolling(window).mean()
                sma_slope = sma.diff()

                trend[f"sma_slope_mean_w{window}"] = sma_slope.mean()
                trend[f"sma_slope_std_w{window}"] = sma_slope.std()

                # EMA
                ema = close_prices.ewm(span=window).mean()
                ema_slope = ema.diff()

                trend[f"ema_slope_mean_w{window}"] = ema_slope.mean()
                trend[f"ema_slope_std_w{window}"] = ema_slope.std()

                # 滞在率
                trend[f"above_sma_ratio_w{window}"] = (close_prices > sma).mean()
                trend[f"above_ema_ratio_w{window}"] = (close_prices > ema).mean()

        # クロス回数
        cross_pairs = [(5, 20), (10, 20), (20, 50), (50, 200)]
        for short, long in cross_pairs:
            if len(data) >= long:
                sma_short = close_prices.rolling(short).mean()
                sma_long = close_prices.rolling(long).mean()

                golden_cross = ((sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))).sum()
                death_cross = ((sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))).sum()

                trend[f"golden_cross_count_{short}_{long}"] = golden_cross
                trend[f"death_cross_count_{short}_{long}"] = death_cross

        return trend

    def _calculate_volatility_features(self, data: pd.DataFrame) -> Dict:
        """ボラティリティ系特徴量（24項目）"""
        vol = {}

        # ATR
        for period in [14, 20, 60]:
            if len(data) >= period:
                atr = ta.atr(data['high'], data['low'], data['close'], length=period)
                vol[f"atr_mean_w{period}"] = atr.mean()
                vol[f"atr_std_w{period}"] = atr.std()

                # True Range to Close Ratio
                tr = ta.true_range(data['high'], data['low'], data['close'])
                tr_to_close = tr / data['close']
                vol[f"true_range_to_close_mean_w{period}"] = tr_to_close.rolling(period).mean().mean()
                vol[f"true_range_to_close_std_w{period}"] = tr_to_close.rolling(period).std().mean()

        # ヒストリカルボラティリティ
        for period in [20, 60, 120]:
            if len(data) >= period:
                returns = data['close'].pct_change()
                hv = returns.rolling(period).std() * np.sqrt(252)
                vol[f"hv_w{period}"] = hv.mean()

        # ボリンジャーバンド
        if len(data) >= 20:
            bb = ta.bbands(data['close'], length=20, std=2)
            bb_width = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
            vol["bb_width_mean_w20"] = bb_width.mean()
            vol["bb_width_p90_w20"] = bb_width.quantile(0.9)

            # 収縮/拡大比率
            vol["bb_squeeze_ratio_w20"] = (bb_width < bb_width.quantile(0.2)).mean()
            vol["bb_expand_ratio_w20"] = (bb_width > bb_width.quantile(0.8)).mean()

        # ギャップ
        gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        vol["gap_abs_mean"] = gap.abs().mean()
        vol["gap_down_p95"] = gap[gap < 0].quantile(0.05) if len(gap[gap < 0]) > 0 else 0
        vol["gap_up_p95"] = gap[gap > 0].quantile(0.95) if len(gap[gap > 0]) > 0 else 0

        return vol

    def _calculate_oscillator_features(self, data: pd.DataFrame) -> Dict:
        """オシレーター系特徴量（22項目）"""
        osc = {}
        close_prices = data['close']

        # RSI
        if len(data) >= 14:
            rsi = ta.rsi(close_prices, length=14)
            osc["rsi14_mean"] = rsi.mean()
            osc["rsi14_std"] = rsi.std()
            osc["rsi14_gt70_ratio"] = (rsi > 70).mean()
            osc["rsi14_lt30_ratio"] = (rsi < 30).mean()
            osc["rsi14_stay80_3d_count"] = (rsi.rolling(3).min() > 80).sum()

        # RCI (簡易実装)
        for period in [9, 26]:
            if len(data) >= period:
                # 簡易RCI計算
                rci_values = []
                for i in range(period, len(data)):
                    window_data = close_prices.iloc[i-period:i]
                    if len(window_data) == period:
                        # スピアマンの順位相関
                        date_ranks = np.arange(1, period + 1)
                        price_ranks = np.argsort(np.argsort(window_data.values)) + 1
                        d_squared = np.sum((date_ranks - price_ranks) ** 2)
                        rci = (1 - (6 * d_squared) / (period * (period**2 - 1))) * 100
                        rci_values.append(rci)

                if rci_values:
                    rci_series = pd.Series(rci_values, index=data.index[period:])
                    osc[f"rci{period}_mean"] = rci_series.mean()
                    osc[f"rci{period}_std"] = rci_series.std()
                    osc[f"rci{period}_gt80_ratio"] = (rci_series > 80).mean()
                    osc[f"rci{period}_lt_80_ratio"] = (rci_series < -80).mean()
                    osc[f"rci{period}_cross_zero_count"] = ((rci_series > 0) & (rci_series.shift(1) <= 0)).sum()

        # ストキャスティクス
        if len(data) >= 14:
            stoch = ta.stoch(data['high'], data['low'], close_prices)
            osc["stoch_k_mean"] = stoch['STOCHk_14_3_3'].mean()
            osc["stoch_k_std"] = stoch['STOCHk_14_3_3'].std()
            osc["stoch_k_gt80_ratio"] = (stoch['STOCHk_14_3_3'] > 80).mean()
            osc["stoch_k_lt20_ratio"] = (stoch['STOCHk_14_3_3'] < 20).mean()

        # MACD
        if len(data) >= 26:
            macd = ta.macd(close_prices)
            hist = macd['MACDh_12_26_9']
            osc["macd_hist_mean"] = hist.mean()
            osc["macd_hist_std"] = hist.std()

            # 連続正負日数
            hist_pos = (hist > 0)
            hist_neg = (hist < 0)

            # 連続正日数
            pos_streaks = []
            current_streak = 0
            for val in hist_pos:
                if val:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        pos_streaks.append(current_streak)
                    current_streak = 0
            osc["macd_hist_pos_streak_mean"] = np.mean(pos_streaks) if pos_streaks else 0

            # 連続負日数
            neg_streaks = []
            current_streak = 0
            for val in hist_neg:
                if val:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        neg_streaks.append(current_streak)
                    current_streak = 0
            osc["macd_hist_neg_streak_mean"] = np.mean(neg_streaks) if neg_streaks else 0

        return osc

    def _calculate_candle_features(self, data: pd.DataFrame) -> Dict:
        """ローソク足系特徴量（30項目）"""
        candle = {}

        # 主要な酒田パターン
        patterns = [
            'engulfing', 'harami', 'morning_star', 'evening_star',
            'three_white_soldiers', 'three_black_crows', 'hammer',
            'shooting_star', 'doji', 'piercing', 'dark_cloud'
        ]

        for pattern in patterns:
            try:
                # パターン検出
                if pattern == 'engulfing':
                    result = ta.cdl_pattern(data['open'], data['high'], data['low'], data['close'], name=pattern)
                else:
                    result = getattr(ta, f"cdl_{pattern}")(data['open'], data['high'], data['low'], data['close'])

                if result is not None:
                    # 正負のヒット数
                    pos_count = (result == 100).sum()
                    neg_count = (result == -100).sum()

                    candle[f"{pattern}_pos_count"] = pos_count
                    candle[f"{pattern}_neg_count"] = neg_count

                    # 翌3日勝率（正パターン）
                    if pos_count > 0:
                        pos_dates = result[result == 100].index
                        next3d_returns = []
                        for date in pos_dates:
                            if date + pd.Timedelta(days=3) in data.index:
                                entry_price = data.loc[date, 'close']
                                exit_price = data.loc[date + pd.Timedelta(days=3), 'close']
                                ret = (exit_price - entry_price) / entry_price
                                next3d_returns.append(ret)

                        if next3d_returns:
                            wins = sum(1 for r in next3d_returns if r > 0)
                            candle[f"{pattern}_pos_next3d_winrate"] = wins / len(next3d_returns)
                            candle[f"{pattern}_pos_next3d_mean"] = np.mean(next3d_returns)

                    # 翌5日平均リターン（負パターン）
                    if neg_count > 0:
                        neg_dates = result[result == -100].index
                        next5d_returns = []
                        for date in neg_dates:
                            if date + pd.Timedelta(days=5) in data.index:
                                entry_price = data.loc[date, 'close']
                                exit_price = data.loc[date + pd.Timedelta(days=5), 'close']
                                ret = (exit_price - entry_price) / entry_price
                                next5d_returns.append(ret)

                        if next5d_returns:
                            candle[f"{pattern}_neg_next5d_mean"] = np.mean(next5d_returns)

            except Exception as e:
                # パターンが利用できない場合はスキップ
                continue

        return candle

    def _calculate_volume_features(self, data: pd.DataFrame) -> Dict:
        """出来高系特徴量（20項目）"""
        volume = {}
        vol_series = data['volume']
        close_prices = data['close']

        # 基本統計
        volume["vol_mean"] = vol_series.mean()
        volume["vol_std"] = vol_series.std()

        # トレンド
        if len(data) >= 20:
            volume["vol_trend_slope_w20"] = np.polyfit(range(len(vol_series)), vol_series.values, 1)[0]

        # 移動平均比
        for window in [5, 20, 60]:
            if len(data) >= window:
                vol_ma = vol_series.rolling(window).mean()
                vol_ratio = vol_series / vol_ma
                volume[f"vol_ma_ratio_mean_w{window}"] = vol_ratio.mean()
                volume[f"vol_ma_ratio_p90_w{window}"] = vol_ratio.quantile(0.9)

        # Zスコア
        vol_z = (vol_series - vol_series.mean()) / vol_series.std()
        volume["vol_z_mean"] = vol_z.mean()
        volume["vol_z_std"] = vol_z.std()
        volume["vol_z_p95"] = vol_z.quantile(0.95)
        volume["vol_spike_count_z1_5"] = (vol_z > 1.5).sum()

        # スパイク後のリターン
        spike_dates = vol_z[vol_z > 1.5].index
        spike_returns = []
        for date in spike_dates:
            if date + pd.Timedelta(days=3) in data.index:
                entry_price = data.loc[date, 'close']
                exit_price = data.loc[date + pd.Timedelta(days=3), 'close']
                ret = (exit_price - entry_price) / entry_price
                spike_returns.append(ret)

        if spike_returns:
            volume["vol_spike_next3d_mean"] = np.mean(spike_returns)

        # OBV
        if len(data) > 0:
            obv = ta.obv(close_prices, vol_series)
            if len(obv) >= 20:
                volume["on_balance_volume_slope_w20"] = np.polyfit(range(len(obv)), obv.values, 1)[0]

        # マネーフロー
        price_up = close_prices > close_prices.shift(1)
        money_flow_pos_ratio = (vol_series[price_up].sum() / vol_series.sum()) if vol_series.sum() > 0 else 0
        volume["money_flow_pos_ratio"] = money_flow_pos_ratio

        return volume

    def _calculate_streak_features(self, data: pd.DataFrame) -> Dict:
        """連続性系特徴量（18項目）"""
        streak = {}
        close_prices = data['close']
        returns = close_prices.pct_change()

        # 連騰/連敗
        up_streaks = []
        down_streaks = []
        current_up = 0
        current_down = 0

        for ret in returns:
            if ret > 0:
                current_up += 1
                if current_down > 0:
                    down_streaks.append(current_down)
                    current_down = 0
            elif ret < 0:
                current_down += 1
                if current_up > 0:
                    up_streaks.append(current_up)
                    current_up = 0

        if current_up > 0:
            up_streaks.append(current_up)
        if current_down > 0:
            down_streaks.append(current_down)

        streak["up_streak_max"] = max(up_streaks) if up_streaks else 0
        streak["down_streak_max"] = max(down_streaks) if down_streaks else 0
        streak["up_streak_mean"] = np.mean(up_streaks) if up_streaks else 0
        streak["down_streak_mean"] = np.mean(down_streaks) if down_streaks else 0
        streak["up7_plus_count"] = sum(1 for s in up_streaks if s >= 7)
        streak["down7_plus_count"] = sum(1 for s in down_streaks if s >= 7)

        # 急騰/急落イベント
        r20 = close_prices.pct_change(20)
        streak["r20_gt_25pct_count"] = (r20 > 0.25).sum()
        streak["r20_lt_neg20pct_count"] = (r20 < -0.20).sum()

        # 急騰後のリターン
        spike_dates = r20[r20 > 0.25].index
        spike_returns = []
        for date in spike_dates:
            if date + pd.Timedelta(days=10) in data.index:
                entry_price = data.loc[date, 'close']
                exit_price = data.loc[date + pd.Timedelta(days=10), 'close']
                ret = (exit_price - entry_price) / entry_price
                spike_returns.append(ret)

        if spike_returns:
            streak["r20_spike_next10d_mean"] = np.mean(spike_returns)

        # 急落後のリターン
        crash_dates = r20[r20 < -0.20].index
        crash_returns = []
        for date in crash_dates:
            if date + pd.Timedelta(days=10) in data.index:
                entry_price = data.loc[date, 'close']
                exit_price = data.loc[date + pd.Timedelta(days=10), 'close']
                ret = (exit_price - entry_price) / entry_price
                crash_returns.append(ret)

        if crash_returns:
            streak["crash_next10d_mean"] = np.mean(crash_returns)

        # 新高値/新安値
        rolling_20d_high = close_prices.rolling(20).max()
        rolling_20d_low = close_prices.rolling(20).min()

        streak["new_20d_high_count"] = (close_prices == rolling_20d_high).sum()
        streak["new_20d_low_count"] = (close_prices == rolling_20d_low).sum()

        # ブレイク後の勝率
        break_high_dates = ((close_prices > rolling_20d_high.shift(1)) & (close_prices.shift(1) <= rolling_20d_high.shift(1))).index
        break_high_returns = []
        for date in break_high_dates:
            if date + pd.Timedelta(days=3) in data.index:
                entry_price = data.loc[date, 'close']
                exit_price = data.loc[date + pd.Timedelta(days=3), 'close']
                ret = (exit_price - entry_price) / entry_price
                break_high_returns.append(ret)

        if break_high_returns:
            wins = sum(1 for r in break_high_returns if r > 0)
            streak["break_20d_high_next3d_winrate"] = wins / len(break_high_returns)

        break_low_dates = ((close_prices < rolling_20d_low.shift(1)) & (close_prices.shift(1) >= rolling_20d_low.shift(1))).index
        break_low_returns = []
        for date in break_low_dates:
            if date + pd.Timedelta(days=3) in data.index:
                entry_price = data.loc[date, 'close']
                exit_price = data.loc[date + pd.Timedelta(days=3), 'close']
                ret = (exit_price - entry_price) / entry_price
                break_low_returns.append(ret)

        if break_low_returns:
            wins = sum(1 for r in break_low_returns if r > 0)
            streak["break_20d_low_next3d_winrate"] = wins / len(break_low_returns)

        # 年初来高値/安値タッチ
        ytd_high = close_prices.expanding().max()
        ytd_low = close_prices.expanding().min()

        streak["ytd_high_touch_count"] = (close_prices == ytd_high).sum()
        streak["ytd_low_touch_count"] = (close_prices == ytd_low).sum()

        return streak

    def _calculate_drawdown_features(self, data: pd.DataFrame) -> Dict:
        """ドローダウン系特徴量（20項目）"""
        dd = {}
        close_prices = data['close']

        # 日次DD（前日終値→当日終値）
        daily_returns = close_prices.pct_change()
        daily_dd = daily_returns.copy()
        daily_dd[daily_returns > 0] = 0  # 上昇日は0

        dd["daily_dd_max"] = daily_dd.min()
        dd["daily_dd_min"] = daily_dd.max()  # これは実際には最小の下落（最大の負の値）
        dd["daily_dd_mean"] = daily_dd.mean()
        dd["daily_dd_std"] = daily_dd.std()

        # 分位点
        for p in [50, 75, 90, 95, 99]:
            dd[f"daily_dd_p{p}"] = daily_dd.quantile(p/100)

        dd["daily_dd_nonzero_ratio"] = (daily_dd < 0).mean()

        # 連続下落クラスター
        dd_clusters = []
        current_cluster = 0
        for val in (daily_dd < 0):
            if val:
                current_cluster += 1
            else:
                if current_cluster > 0:
                    dd_clusters.append(current_cluster)
                current_cluster = 0

        if current_cluster > 0:
            dd_clusters.append(current_cluster)

        dd["daily_dd_cluster_mean_days"] = np.mean(dd_clusters) if dd_clusters else 0

        # 日中DD（高値→終値）
        intraday_dd = (data['close'] - data['high']) / data['high']
        dd["intraday_dd_max"] = intraday_dd.min()
        dd["intraday_dd_mean"] = intraday_dd.mean()
        dd["intraday_dd_p95"] = intraday_dd.quantile(0.95)

        # 暴落期DD分布
        for window in [5, 20, 60]:
            if len(data) >= window:
                window_dd = daily_dd.rolling(window).min()
                dd[f"crash_window_dd_p95_w{window}"] = window_dd.quantile(0.95)

        # 大下落後の反発
        big_dd_dates = daily_dd[daily_dd < -0.05].index
        rebound_returns = []
        for date in big_dd_dates:
            if date + pd.Timedelta(days=5) in data.index:
                entry_price = data.loc[date, 'close']
                exit_price = data.loc[date + pd.Timedelta(days=5), 'close']
                ret = (exit_price - entry_price) / entry_price
                rebound_returns.append(ret)

        if rebound_returns:
            dd["rebound_after_dd_gt5pct_next5d_mean"] = np.mean(rebound_returns)
            wins = sum(1 for r in rebound_returns if r > 0)
            dd["rebound_winrate_after_dd_gt5pct"] = wins / len(rebound_returns)

        return dd

    def _calculate_regime_features(self, data: pd.DataFrame) -> Dict:
        """レジーム/季節系特徴量（16項目）"""
        regime = {}
        close_prices = data['close']

        # ADX
        if len(data) >= 14:
            adx = ta.adx(data['high'], data['low'], close_prices)
            regime["adx14_mean"] = adx['ADX_14'].mean()
            regime["adx14_gt25_ratio"] = (adx['ADX_14'] > 25).mean()

        # ボリンジャー収縮/拡大（既存の特徴量を活用）
        if len(data) >= 20:
            bb = ta.bbands(close_prices, length=20, std=2)
            bb_width = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
            regime["bb_squeeze_days_ratio"] = (bb_width < bb_width.quantile(0.2)).mean()
            regime["bb_expand_days_ratio"] = (bb_width > bb_width.quantile(0.8)).mean()

        # 曜日別リターン
        returns = close_prices.pct_change()
        weekdays = returns.index.weekday
        for i, day_name in enumerate(['mon', 'tue', 'wed', 'thu', 'fri']):
            day_returns = returns[weekdays == i]
            if len(day_returns) > 0:
                regime[f"weekday_return_mean_{day_name}"] = day_returns.mean()

        # 月別リターン
        monthly_returns = []
        month_stats = {}
        for month in range(1, 13):
            month_returns = returns[returns.index.month == month]
            if len(month_returns) > 0:
                month_stats[month] = month_returns.mean()
                monthly_returns.append(month_returns.mean())

        if monthly_returns:
            best_month = np.argmax(monthly_returns) + 1
            worst_month = np.argmin(monthly_returns) + 1
            regime["best_month"] = best_month
            regime["worst_month"] = worst_month

        return regime

    def _calculate_price_position_features(self, data: pd.DataFrame) -> Dict:
        """価格位置系特徴量（12項目）"""
        pricepos = {}
        close_prices = data['close']

        # 52週高値/安値からの距離
        if len(data) >= 252:
            rolling_52w_high = close_prices.rolling(252).max()
            rolling_52w_low = close_prices.rolling(252).min()

            dist_to_52w_high = (rolling_52w_high - close_prices) / close_prices
            dist_to_52w_low = (close_prices - rolling_52w_low) / close_prices

            pricepos["dist_to_52w_high_mean"] = dist_to_52w_high.mean()
            pricepos["dist_to_52w_low_mean"] = dist_to_52w_low.mean()
            pricepos["new_high_ratio_52w"] = (close_prices == rolling_52w_high).mean()
            pricepos["new_low_ratio_52w"] = (close_prices == rolling_52w_low).mean()

        # 歪度と尖度
        returns = close_prices.pct_change()
        for window in [20, 60]:
            if len(data) >= window:
                rolling_returns = returns.rolling(window)
                pricepos[f"return_skew_w{window}"] = rolling_returns.skew().mean()
                pricepos[f"return_kurt_w{window}"] = rolling_returns.kurt().mean()

        # ジャンプとテールリスク
        if len(data) >= 1:
            pricepos["jump_p95_w1"] = returns.quantile(0.95)
            pricepos["tail_risk_p99_w1"] = returns.quantile(0.01)

        return pricepos

    def _generate_notes(self, features: Dict, stats: Dict) -> List[str]:
        """観察メモを生成"""
        notes = []

        # 最大DDに関するメモ
        max_dd = stats.get("max_dd_5y", 0)
        if max_dd < -0.3:
            notes.append(f"最大ドローダウンが{-max_dd*100:.1f}%と大きく、暴落耐性が重要")
        elif max_dd > -0.1:
            notes.append(f"最大ドローダウンが{-max_dd*100:.1f}%と小さく、安定した銘柄")

        # ボラティリティに関するメモ
        vol_features = features.get("vol", {})
        hv_20 = vol_features.get("hv_w20", 0)
        if hv_20 > 0.3:
            notes.append(f"ヒストリカルボラティリティが{hv_20*100:.1f}%と高く、大きな値動きを期待")
        elif hv_20 < 0.1:
            notes.append(f"ヒストリカルボラティリティが{hv_20*100:.1f}%と低く、安定した値動き")

        # トレンド強度
        trend_features = features.get("trend", {})
        above_sma_20 = trend_features.get("above_sma_ratio_w20", 0)
        if above_sma_20 > 0.7:
            notes.append("20日SMAを上回る日が多く、強い上昇トレンド")
        elif above_sma_20 < 0.3:
            notes.append("20日SMAを下回る日が多く、弱い下落トレンド")

        # RSI特性
        osc_features = features.get("osc", {})
        rsi_gt70 = osc_features.get("rsi14_gt70_ratio", 0)
        if rsi_gt70 > 0.2:
            notes.append("RSIが70超えの日が多く、買われすぎ警戒が必要")

        return notes

    def _save_to_json(self, symbol: str, features_data: Dict, split_info: Dict) -> Optional[str]:
        """JSONファイルに保存"""
        try:
            # 完全なデータ構造を構築
            output_data = {
                "symbol": symbol,
                "span_years": self.years,
                "split": split_info,
                "stats": self._convert_numpy_types(features_data["stats"]),
                "features": self._convert_numpy_types(features_data["features"]),
                "notes": features_data["notes"],
                "version": "params-v1"
            }

            # 出力ディレクトリの確認
            output_dir = Path("params")
            output_dir.mkdir(exist_ok=True)

            # ファイルパス
            output_path = output_dir / f"{symbol}.json"

            # JSONとして保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            # スキーマ検証
            schema_path = Path("schemas/params_schema.json")
            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                try:
                    jsonschema.validate(output_data, schema)
                    print("スキーマ検証: OK")
                except jsonschema.ValidationError as e:
                    print(f"スキーマ検証警告: {e}")

            return str(output_path)

        except Exception as e:
            print(f"JSON保存エラー: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_numpy_types(self, obj):
        """numpy型をPython基本型に変換"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int8, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            # Seriesは最初の要素を返す（単一値として扱う）
            if len(obj) > 0:
                return self._convert_numpy_types(obj.iloc[0])
            else:
                return None
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        else:
            return obj


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='チャート分析スクリプト - 244項目の特徴量抽出')
    parser.add_argument('symbol', help='銘柄コード（例: 9984.T）')
    parser.add_argument('--years', type=int, default=5, help='分析年数（デフォルト: 5年）')
    parser.add_argument('--min-years', type=int, default=3, help='最低必要年数（デフォルト: 3年）')
    parser.add_argument('--start', help='開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='終了日（YYYY-MM-DD）')
    parser.add_argument('--force', action='store_true', help='強制実行')

    args = parser.parse_args()

    analyzer = ChartAnalyzer(years=args.years, min_years=args.min_years)
    success = analyzer.analyze_symbol(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        force=args.force
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
