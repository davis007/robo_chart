#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デバッグ用チャート分析スクリプト
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 既存のDataManagerをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest import DataManager


class DebugChartAnalyzer:
    """デバッグ用チャート分析クラス"""

    def __init__(self, years: int = 5, min_years: int = 3):
        self.years = years
        self.min_years = min_years
        self.data_manager = DataManager()

    def debug_analyze(self, symbol: str):
        """デバッグ分析を実行"""
        print(f"デバッグ分析開始: {symbol}")

        # 日付範囲の設定
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=self.years * 365)
        start_date = start_dt.strftime('%Y-%m-%d')

        print(f"期間: {start_date} 〜 {end_date}")

        # 株価データ取得
        data = self.data_manager.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            print(f"データが取得できませんでした: {symbol}")
            return

        print(f"取得データ: {len(data)}日分 ({data.index[0].strftime('%Y-%m-%d')} 〜 {data.index[-1].strftime('%Y-%m-%d')})")

        # 期間分割（学習4年＋検証1年）
        split_info = self._split_data(data)
        if not split_info:
            print("期間分割に失敗しました")
            return

        print(f"分割情報: {split_info}")

        # 学習期間データ（インデックスで直接抽出）
        train_start = pd.to_datetime(split_info["train_start"])
        train_end = pd.to_datetime(split_info["train_end"])
        train_data = data[(data.index >= train_start) & (data.index <= train_end)]

        print(f"学習データ: {len(train_data)}日分")

        # 各特徴量計算を個別にテスト
        print("\n=== 特徴量計算テスト ===")

        try:
            print("1. リターン特徴量計算中...")
            returns = self._calculate_return_features(train_data)
            print(f"  リターン特徴量: {len(returns)}項目")
        except Exception as e:
            print(f"  リターン特徴量エラー: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            print("2. トレンド特徴量計算中...")
            trend = self._calculate_trend_features(train_data)
            print(f"  トレンド特徴量: {len(trend)}項目")
        except Exception as e:
            print(f"  トレンド特徴量エラー: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            print("3. ボラティリティ特徴量計算中...")
            vol = self._calculate_volatility_features(train_data)
            print(f"  ボラティリティ特徴量: {len(vol)}項目")
        except Exception as e:
            print(f"  ボラティリティ特徴量エラー: {e}")
            import traceback
            traceback.print_exc()
            return

        print("特徴量計算完了")

    def _split_data(self, data: pd.DataFrame) -> Optional[Dict]:
        """データを学習期間と検証期間に分割"""
        if len(data) < 252 * self.min_years:
            return None

        split_idx = int(len(data) * 0.8)
        split_date = data.index[split_idx]

        train_data = data.iloc[:split_idx]
        valid_data = data.iloc[split_idx:]

        if len(train_data) < 252 * 3 or len(valid_data) < 252 * 0.5:
            return None

        return {
            "train_start": train_data.index[0].strftime('%Y-%m-%d'),
            "train_end": train_data.index[-1].strftime('%Y-%m-%d'),
            "valid_start": valid_data.index[0].strftime('%Y-%m-%d'),
            "valid_end": valid_data.index[-1].strftime('%Y-%m-%d')
        }

    def _calculate_return_features(self, data: pd.DataFrame) -> Dict:
        """リターン系特徴量（54項目）"""
        close_prices = data['close']
        returns = {}

        windows = [1, 5, 10, 20, 60, 120, 252]

        for window in windows:
            if len(data) >= window:
                returns_series = close_prices.pct_change(window)

                returns[f"r_mean_w{window}"] = returns_series.mean()
                returns[f"r_std_w{window}"] = returns_series.std()
                returns[f"r_min_w{window}"] = returns_series.min()
                returns[f"r_max_w{window}"] = returns_series.max()

                if window > 1:
                    for p in [10, 25, 50, 75, 90]:
                        returns[f"r_p{p}_w{window}"] = returns_series.quantile(p/100)
                else:
                    returns["r_p50_w1"] = returns_series.quantile(0.5)
                    returns["r_p90_w1"] = returns_series.quantile(0.9)

        return returns

    def _calculate_trend_features(self, data: pd.DataFrame) -> Dict:
        """トレンド系特徴量（40項目）"""
        close_prices = data['close']
        trend = {}

        ma_windows = [5, 10, 20, 50, 100, 200]

        for window in ma_windows:
            if len(data) >= window:
                sma = close_prices.rolling(window).mean()
                sma_slope = sma.diff()

                trend[f"sma_slope_mean_w{window}"] = sma_slope.mean()
                trend[f"sma_slope_std_w{window}"] = sma_slope.std()

                ema = close_prices.ewm(span=window).mean()
                ema_slope = ema.diff()

                trend[f"ema_slope_mean_w{window}"] = ema_slope.mean()
                trend[f"ema_slope_std_w{window}"] = ema_slope.std()

                trend[f"above_sma_ratio_w{window}"] = (close_prices > sma).mean()
                trend[f"above_ema_ratio_w{window}"] = (close_prices > ema).mean()

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

        for period in [14, 20, 60]:
            if len(data) >= period:
                atr = ta.atr(data['high'], data['low'], data['close'], length=period)
                vol[f"atr_mean_w{period}"] = atr.mean()
                vol[f"atr_std_w{period}"] = atr.std()

        for period in [20, 60, 120]:
            if len(data) >= period:
                returns = data['close'].pct_change()
                hv = returns.rolling(period).std() * np.sqrt(252)
                vol[f"hv_w{period}"] = hv.mean()

        if len(data) >= 20:
            bb = ta.bbands(data['close'], length=20, std=2)
            bb_width = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
            vol["bb_width_mean_w20"] = bb_width.mean()
            vol["bb_width_p90_w20"] = bb_width.quantile(0.9)

        return vol


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='デバッグ用チャート分析スクリプト')
    parser.add_argument('symbol', help='銘柄コード（例: 9984.T）')

    args = parser.parse_args()

    analyzer = DebugChartAnalyzer(years=5, min_years=3)
    analyzer.debug_analyze(args.symbol)


if __name__ == "__main__":
    main()
