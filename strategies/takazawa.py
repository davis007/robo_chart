#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高澤式（BB+RCI）戦略
ボリンジャーバンドとRCI（順位相関指数）を使用した売買シグナル生成
"""

import pandas as pd
import numpy as np
from typing import Optional


class Strategy:
    """高澤式戦略クラス"""

    def __init__(self, bb_period: int = 20, bb_std: int = 2, rci_period: int = 9):
        """
        初期化

        Args:
            bb_period: ボリンジャーバンドの期間
            bb_std: ボリンジャーバンドの標準偏差
            rci_period: RCIの期間
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rci_period = rci_period

    def calculate_signal(self, data: pd.DataFrame, current_date: pd.Timestamp, ctx: dict) -> dict:
        """
        売買シグナルを計算

        Args:
            data: 株価データ
            current_date: 現在の日付
            ctx: コンテキスト情報（ポジション状態など）

        Returns:
            dict: {'signal': 'BUY'|'SELL'|'HOLD', 'reason': str}
        """
        # 現在のインデックスを取得
        current_idx = data.index.get_loc(current_date)

        # 十分なデータがない場合はHOLD
        if current_idx < max(self.bb_period, self.rci_period):
            return {'signal': 'HOLD', 'reason': 'データ不足'}

        # ボリンジャーバンドを計算
        bb_signal = self._calculate_bb_signal(data, current_idx)

        # RCIを計算
        rci_signal = self._calculate_rci_signal(data, current_idx)

        # シグナルを統合（最も柔軟な条件）
        # ポジション状態に基づいてシグナル解釈
        position = ctx.get("position", "FLAT")

        if position == "FLAT":
            # FLAT状態：BUY→ロングエントリー、SELL→ショートエントリー
            if bb_signal == 'BUY' or rci_signal == 'BUY':
                return {'signal': 'BUY', 'reason': f'BB:{bb_signal}+RCI:{rci_signal}'}
            elif bb_signal == 'SELL' or rci_signal == 'SELL':
                return {'signal': 'SELL', 'reason': f'BB:{bb_signal}+RCI:{rci_signal}'}
            else:
                return {'signal': 'HOLD', 'reason': 'シグナルなし'}

        elif position == "LONG":
            # LONG状態：SELL→ロング解消
            if bb_signal == 'SELL' or rci_signal == 'SELL':
                return {'signal': 'SELL', 'reason': f'BB:{bb_signal}+RCI:{rci_signal}'}
            else:
                return {'signal': 'HOLD', 'reason': 'シグナルなし'}

        elif position == "SHORT":
            # SHORT状態：BUY→ショート解消
            if bb_signal == 'BUY' or rci_signal == 'BUY':
                return {'signal': 'BUY', 'reason': f'BB:{bb_signal}+RCI:{rci_signal}'}
            else:
                return {'signal': 'HOLD', 'reason': 'シグナルなし'}

        else:
            return {'signal': 'HOLD', 'reason': 'シグナルなし'}

    def _calculate_bb_signal(self, data: pd.DataFrame, current_idx: int) -> str:
        """ボリンジャーバンドシグナルを計算"""
        # 期間内のデータを取得
        start_idx = current_idx - self.bb_period + 1
        bb_data = data.iloc[start_idx:current_idx + 1]

        # カラム名の大文字小文字を考慮して終値を取得
        close_col = 'close' if 'close' in bb_data.columns else 'Close'

        # 移動平均と標準偏差を計算
        ma = bb_data[close_col].mean()
        std = bb_data[close_col].std()

        # バンドを計算
        upper_band = ma + (self.bb_std * std)
        lower_band = ma - (self.bb_std * std)

        current_price = data.iloc[current_idx][close_col]

        # シグナル判定（厳格な条件に戻す）
        if current_price <= lower_band:
            return 'BUY'  # 下バンドを割り込んだら買い
        elif current_price >= upper_band:
            return 'SELL'  # 上バンドを超えたら売り
        else:
            return 'NONE'

    def _calculate_rci_signal(self, data: pd.DataFrame, current_idx: int) -> str:
        """RCIシグナルを計算"""
        # 期間内のデータを取得
        start_idx = current_idx - self.rci_period + 1
        rci_data = data.iloc[start_idx:current_idx + 1]

        # カラム名の大文字小文字を考慮して終値を取得
        close_col = 'close' if 'close' in rci_data.columns else 'Close'

        # 日付の順位と価格の順位を計算
        dates = list(range(1, len(rci_data) + 1))  # 1, 2, 3, ...
        prices = rci_data[close_col].tolist()

        # 価格の順位を計算（高い順）
        price_ranks = self._calculate_ranks(prices)

        # RCIを計算
        n = len(dates)
        sum_d_squared = sum((d - p) ** 2 for d, p in zip(dates, price_ranks))
        rci = (1 - (6 * sum_d_squared) / (n * (n ** 2 - 1))) * 100

        # シグナル判定（条件を緩和）
        if rci <= -60:  # 売られすぎ（-70 → -60に緩和）
            return 'BUY'
        elif rci >= 60:  # 買われすぎ（70 → 60に緩和）
            return 'SELL'
        else:
            return 'NONE'

    def _calculate_ranks(self, values: list) -> list:
        """値の順位を計算（高い順）"""
        sorted_values = sorted(values, reverse=True)
        ranks = [sorted_values.index(v) + 1 for v in values]
        return ranks


# テスト用
if __name__ == "__main__":
    # テストデータの作成
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    test_data = pd.DataFrame({
        'Open': np.random.normal(1000, 50, 30),
        'High': np.random.normal(1020, 50, 30),
        'Low': np.random.normal(980, 50, 30),
        'Close': np.random.normal(1000, 50, 30),
        'Volume': np.random.randint(1000000, 5000000, 30)
    }, index=dates)

    strategy = Strategy()

    # 最後の日付でテスト
    last_date = dates[-1]
    ctx = {"position": "FLAT", "entry_price": None, "size": 0}
    signal = strategy.calculate_signal(test_data, last_date, ctx)
    print(f"最終日のシグナル: {signal}")
