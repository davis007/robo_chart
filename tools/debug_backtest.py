#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デバッグ用バックテストスクリプト - エントリー条件を緩和
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 既存のDataManagerをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest import DataManager


class DebugBacktest:
    """デバッグ用バックテストクラス"""

    def __init__(self, initial_cash: float = 1000000, position_size: int = 100):
        self.data_manager = DataManager()
        self.initial_cash = initial_cash
        self.position_size = position_size

        # ポジション状態
        self.cash = initial_cash
        self.side = "FLAT"
        self.entry_price = None
        self.size = 0
        self.entry_date = None

        # パフォーマンス記録
        self.trades: List[Dict] = []
        self.equity_list: List[int] = []
        self.closed_trades: List[Dict] = []
        self.wins = 0

        # パラメータ
        self.params = None

        # 新規パラメータ
        self.min_holding_days = 3  # 最低保有日数
        self.daily_dd_p95_threshold = -0.05  # 日次ドローダウン95%閾値

    def run(self, symbol: str, params_file: str, start_date: str = None, end_date: str = None):
        """バックテストを実行"""
        print(f"デバッグバックテスト開始: {symbol}")

        # パラメータ読み込み
        if not self._load_params(params_file):
            return

        # 日付範囲の設定
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365*3)).strftime('%Y-%m-%d')

        print(f"期間: {start_date} 〜 {end_date}")

        # 株価データ取得
        data = self.data_manager.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            print(f"データが取得できませんでした: {symbol}")
            return

        print(f"データ取得完了: {len(data)}日分")

        # バックテスト実行
        self._run_debug_backtest(data, symbol)

        # パフォーマンス出力
        self._output_performance(symbol, data)

    def _load_params(self, params_file: str) -> bool:
        """パラメータファイルを読み込み"""
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                self.params = json.load(f)

            print(f"パラメータ読み込み完了: {params_file}")
            return True

        except Exception as e:
            print(f"パラメータ読み込みエラー: {e}")
            return False

    def _run_debug_backtest(self, data: pd.DataFrame, symbol: str):
        """デバッグ用バックテストを実行"""
        features = self.params.get("features", {})
        stats = self.params.get("stats", {})

        print(f"\n=== デバッグ情報 ===")
        print(f"特徴量数: {sum(len(f) for f in features.values())}")
        print(f"エントリー条件チェック開始...")

        entry_count = 0
        for i in range(20, len(data)):  # 最低20日分のデータが必要
            date = data.index[i]

            # データ欠損チェック
            if pd.isna(data.iloc[i]['close']):
                continue

            # 過去データの取得
            past_data = data.iloc[max(0, i-60):i+1]

            # 簡易エントリー条件チェック
            signal = self._check_debug_entry(past_data, features)

            if signal != "HOLD":
                entry_count += 1
                print(f"{date.strftime('%Y-%m-%d')}: {signal}シグナル検出")

                # 取引実行
                current_close = data.iloc[i]['close']
                print(f"  → 現在のポジション状態: {self.side}")
                if signal == "BUY" and self.side == "FLAT":
                    self._execute_buy(current_close, date, "debug_entry")
                elif signal == "SELL" and self.side == "LONG":
                    # 最低保有日数チェック
                    if self._check_min_holding_days(date):
                        self._execute_sell(current_close, date, "debug_exit")
                    else:
                        print(f"  → 最低保有日数未満のためHOLD: {self._get_holding_days(date)}日")
                else:
                    print(f"  → ポジション状態不一致: signal={signal}, side={self.side}")

            # 損切り条件チェック
            if self.side == "LONG":
                stop_loss_signal = self._check_stop_loss_conditions(data, i, date)
                if stop_loss_signal:
                    current_close = data.iloc[i]['close']
                    self._execute_sell(current_close, date, stop_loss_signal)

            # 評価損益計算
            total_equity = self._calculate_total_equity(data.iloc[i]['close'])
            self.equity_list.append(round(total_equity))

        print(f"エントリーシグナル検出数: {entry_count}回")
        print(f"実際の取引回数: {len(self.closed_trades)}回")

        # 最終ポジション清算
        self._close_final_position(data)

    def _check_debug_entry(self, data: pd.DataFrame, features: Dict) -> str:
        """デバッグ用エントリー条件チェック - 複数シグナル一致方式"""
        current_close = data.iloc[-1]['close']

        # 特徴量の取得
        returns_feat = features.get("returns", {})
        trend_feat = features.get("trend", {})
        vol_feat = features.get("vol", {})
        osc_feat = features.get("osc", {})
        candle_feat = features.get("candle", {})
        volume_feat = features.get("volume", {})
        streak_feat = features.get("streak", {})

        # 複数シグナル一致チェック
        buy_signals = 0
        sell_signals = 0

        # 条件1: キャンドル系シグナル
        candle_signal = self._check_candle_signals(data, candle_feat)
        if candle_signal == "BUY":
            buy_signals += 1
        elif candle_signal == "SELL":
            sell_signals += 1

        # 条件2: RSI系シグナル
        rsi_signal = self._check_rsi_signals(data, osc_feat)
        if rsi_signal == "BUY":
            buy_signals += 1
        elif rsi_signal == "SELL":
            sell_signals += 1

        # 条件3: トレンド系シグナル
        trend_signal = self._check_trend_signals(data, trend_feat)
        if trend_signal == "BUY":
            buy_signals += 1
        elif trend_signal == "SELL":
            sell_signals += 1

        # 条件4: ボラティリティ系シグナル
        vol_signal = self._check_volatility_signals(data, vol_feat, volume_feat)
        if vol_signal == "BUY":
            buy_signals += 1
        elif vol_signal == "SELL":
            sell_signals += 1

        # 条件5: ストリーク系シグナル（勝率が低い場面の抑制）
        streak_signal = self._check_streak_signals(data, streak_feat)
        if streak_signal == "BUY":
            buy_signals += 1
        elif streak_signal == "SELL":
            sell_signals += 1

        # 非対称エントリー条件: BUYは緩和(2/3以上)、SELLは厳格(3/3以上)
        total_signals = 5  # 全シグナル数

        if self.side == "FLAT" and buy_signals >= 2:  # BUY: 2/3以上で許可
            print(f"    → 複数シグナル一致: BUYシグナル{buy_signals}個, SELLシグナル{sell_signals}個")
            return "BUY"
        elif self.side == "LONG" and sell_signals >= 3:  # SELL: 3/3以上で許可
            print(f"    → 複数シグナル一致: BUYシグナル{buy_signals}個, SELLシグナル{sell_signals}個")
            return "SELL"

        if buy_signals > 0 or sell_signals > 0:
            print(f"    → シグナル不足: BUYシグナル{buy_signals}個, SELLシグナル{sell_signals}個")

        return "HOLD"

    def _check_candle_signals(self, data: pd.DataFrame, candle_feat: Dict) -> str:
        """キャンドル系シグナルチェック"""
        current_close = data.iloc[-1]['close']
        current_open = data.iloc[-1]['open']
        current_high = data.iloc[-1]['high']
        current_low = data.iloc[-1]['low']

        # 前日のデータ
        prev_close = data.iloc[-2]['close'] if len(data) >= 2 else current_close
        prev_open = data.iloc[-2]['open'] if len(data) >= 2 else current_open

        # 強気のキャンドルパターン
        bullish_patterns = 0

        # 陽線チェック
        if current_close > current_open:
            bullish_patterns += 1

        # 前日陰線からの反発
        if prev_close < prev_open and current_close > current_open:
            bullish_patterns += 1

        # 弱気のキャンドルパターン
        bearish_patterns = 0

        # 陰線チェック
        if current_close < current_open:
            bearish_patterns += 1

        # 前日陽線からの下落
        if prev_close > prev_open and current_close < current_open:
            bearish_patterns += 1

        if bullish_patterns >= 2:
            return "BUY"
        elif bearish_patterns >= 2:
            return "SELL"

        return "HOLD"

    def _check_rsi_signals(self, data: pd.DataFrame, osc_feat: Dict) -> str:
        """RSI系シグナルチェック"""
        rsi = self._calculate_rsi(data['close'])

        # RSIオーバーソールド/オーバーボート
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"

        return "HOLD"

    def _check_trend_signals(self, data: pd.DataFrame, trend_feat: Dict) -> str:
        """トレンド系シグナルチェック"""
        # 移動平均クロス
        sma_5 = data['close'].rolling(5).mean().iloc[-1]
        sma_20 = data['close'].rolling(20).mean().iloc[-1]

        if sma_5 > sma_20:
            return "BUY"
        elif sma_5 < sma_20:
            return "SELL"

        return "HOLD"

    def _check_volatility_signals(self, data: pd.DataFrame, vol_feat: Dict, volume_feat: Dict) -> str:
        """ボラティリティ系シグナルチェック"""
        current_close = data.iloc[-1]['close']

        # ボリンジャーバンド
        bb_upper = data['close'].rolling(20).mean() + 2 * data['close'].rolling(20).std()
        bb_lower = data['close'].rolling(20).mean() - 2 * data['close'].rolling(20).std()

        # ボラティリティスパイク検出
        vol_spike = self._detect_vol_spike(data)

        if current_close < bb_lower.iloc[-1] and not vol_spike:
            return "BUY"
        elif current_close > bb_upper.iloc[-1] and not vol_spike:
            return "SELL"

        return "HOLD"

    def _check_streak_signals(self, data: pd.DataFrame, streak_feat: Dict) -> str:
        """ストリーク系シグナルチェック（勝率が低い場面の抑制）"""
        # 連続上昇/下落ストリーク
        up_streak = self._calculate_up_streak(data)
        down_streak = self._calculate_down_streak(data)

        # 過度な連続上昇は逆張りシグナル
        if up_streak >= 5:  # 5日連続上昇は過熱
            return "SELL"
        elif down_streak >= 5:  # 5日連続下落は反発期待
            return "BUY"

        return "HOLD"

    def _detect_vol_spike(self, data: pd.DataFrame) -> bool:
        """ボラティリティスパイク検出"""
        if len(data) < 21:
            return False

        # 20日間のボラティリティ平均
        vol_20d = data['close'].pct_change().rolling(20).std().iloc[-1]
        # 当日のボラティリティ
        current_vol = abs(data['close'].iloc[-1] / data['close'].iloc[-2] - 1) if len(data) >= 2 else 0

        # ボラティリティが平均の2倍以上ならスパイクと判定
        return current_vol > vol_20d * 2 if vol_20d > 0 else False

    def _calculate_up_streak(self, data: pd.DataFrame) -> int:
        """連続上昇日数を計算"""
        if len(data) < 2:
            return 0

        streak = 0
        for i in range(len(data)-1, 0, -1):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                streak += 1
            else:
                break
        return streak

    def _calculate_down_streak(self, data: pd.DataFrame) -> int:
        """連続下落日数を計算"""
        if len(data) < 2:
            return 0

        streak = 0
        for i in range(len(data)-1, 0, -1):
            if data['close'].iloc[i] < data['close'].iloc[i-1]:
                streak += 1
            else:
                break
        return streak

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI計算"""
        if len(prices) < period + 1:
            return 50

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def _check_min_holding_days(self, current_date: pd.Timestamp) -> bool:
        """最低保有日数チェック"""
        if self.entry_date is None:
            return True

        holding_days = (current_date - self.entry_date).days
        return holding_days >= self.min_holding_days

    def _get_holding_days(self, current_date: pd.Timestamp) -> int:
        """保有日数を取得"""
        if self.entry_date is None:
            return 0
        return (current_date - self.entry_date).days

    def _check_stop_loss_conditions(self, data: pd.DataFrame, current_index: int, current_date: pd.Timestamp) -> Optional[str]:
        """損切り条件チェック - 二段階方式"""
        if self.entry_price is None:
            return None

        current_close = data.iloc[current_index]['close']
        loss_rate = (current_close - self.entry_price) / self.entry_price

        # 日次ドローダウン95%閾値チェック
        if self._check_daily_dd_p95(data, current_index):
            return "daily_dd_p97_stop_loss"

        # 二段階損切り
        if loss_rate <= -0.08:  # -8%で全額損切り
            return "full_stop_loss"
        elif loss_rate <= -0.05:  # -5%で半分損切り
            return "partial_stop_loss"

        return None

    def _check_daily_dd_p95(self, data: pd.DataFrame, current_index: int) -> bool:
        """日次ドローダウン95%閾値チェック - 閾値を緩和"""
        if current_index < 20:
            return False

        # 過去20日間の日次リターン
        daily_returns = data['close'].pct_change().iloc[current_index-19:current_index+1]

        # 97%分位点の計算（閾値を緩和: p95 → p97相当）
        if len(daily_returns) >= 20:
            p97_threshold = daily_returns.quantile(0.03)  # 3%分位点（下側）- p97相当
            current_return = data['close'].iloc[current_index] / data['close'].iloc[current_index-1] - 1

            # 現在のリターンが97%閾値を下回った場合（より厳しい条件）
            if current_return < p97_threshold:
                print(f"  → daily_dd_p97損切り発動: {current_return:.2%} < {p97_threshold:.2%}")
                return True

        return False

    def _execute_buy(self, price: float, date: pd.Timestamp, reason: str) -> bool:
        """買い注文実行"""
        if self.side == "FLAT":
            cost = price * self.position_size
            print(f"  → BUY条件チェック: cost={cost}, cash={self.cash}, cost<=cash={cost <= self.cash}")
            if cost <= self.cash:
                self.cash -= cost
                self.side = "LONG"
                self.entry_price = price
                self.size = self.position_size
                self.entry_date = date

                trade = {
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'BUY',
                    'price': price,
                    'quantity': self.position_size,
                    'pnl': 0,
                    'reason': reason
                }
                self.trades.append(trade)
                print(f"  → BUY実行: {price}円")
                return True
            else:
                print(f"  → 資金不足: cost={cost}, cash={self.cash}")
        else:
            print(f"  → ポジション状態がFLATではありません: {self.side}")
        return False

    def _execute_sell(self, price: float, date: pd.Timestamp, reason: str) -> Tuple[bool, int]:
        """売り注文実行"""
        pnl = 0
        if self.side == "LONG":
            revenue = price * self.size
            pnl = round((price - self.entry_price) * self.size)
            self.cash += revenue

            if pnl > 0:
                self.wins += 1

            trade = {
                'date': date.strftime('%Y-%m-%d'),
                'action': 'SELL',
                'price': price,
                'quantity': self.size,
                'pnl': pnl,
                'reason': reason
            }
            self.trades.append(trade)

            closed_trade = {
                'entry_price': self.entry_price,
                'exit_price': price,
                'pnl': pnl,
                'reason': reason
            }
            self.closed_trades.append(closed_trade)

            self.side = "FLAT"
            self.entry_price = None
            self.size = 0
            self.entry_date = None

            print(f"  → SELL実行: {price}円, PnL: {pnl:+}円")
            return True, pnl

        return False, pnl

    def _calculate_total_equity(self, current_price: float) -> float:
        """総資産を計算"""
        if self.side == "LONG":
            return self.cash + current_price * self.size
        else:
            return self.cash

    def _close_final_position(self, data: pd.DataFrame):
        """最終ポジションを清算"""
        if self.side != "FLAT" and len(data) > 0:
            final_price = data.iloc[-1]['close']
            if self.side == "LONG":
                self._execute_sell(final_price, data.index[-1], "final_close")

    def _output_performance(self, symbol: str, data: pd.DataFrame):
        """パフォーマンス出力"""
        total_trades = len(self.closed_trades)
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(trade['pnl'] for trade in self.closed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # 最大ドローダウン計算
        max_equity = max(self.equity_list) if self.equity_list else self.initial_cash
        min_equity = min(self.equity_list) if self.equity_list else self.initial_cash
        max_dd = ((min_equity - max_equity) / max_equity * 100) if max_equity > 0 else 0

        final_equity = self.equity_list[-1] if self.equity_list else self.initial_cash
        total_return = ((final_equity - self.initial_cash) / self.initial_cash * 100)

        print("\n" + "="*60)
        print("=== デバッグバックテスト結果 ===")
        print("="*60)
        print(f"銘柄: {symbol}")
        print(f"期間: {data.index[0].strftime('%Y-%m-%d')} 〜 {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"初期資金: {self.initial_cash:,}円")
        print(f"最終資金: {final_equity:,}円")
        print(f"総損益: {total_pnl:+,}円 ({total_return:+.1f}%)")
        print(f"取引回数: {total_trades}回")
        print(f"勝率: {win_rate:.1f}%")
        print(f"平均損益: {avg_pnl:+,.0f}円")
        print(f"最大ドローダウン: {max_dd:.1f}%")

        # 取引詳細
        if total_trades > 0:
            print(f"\n--- 取引詳細 ---")
            for i, trade in enumerate(self.closed_trades, 1):
                print(f"{i:2d}. {trade['entry_price']:6} → {trade['exit_price']:6} | {trade['pnl']:+6}円 | {trade['reason']}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='デバッグ用バックテストスクリプト')
    parser.add_argument('--symbol', required=True, help='銘柄コード（例: 9984.T）')
    parser.add_argument('--params', required=True, help='パラメータJSONファイル')
    parser.add_argument('--start', help='開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='終了日（YYYY-MM-DD）')

    args = parser.parse_args()

    # 株価に応じてposition_sizeを調整
    backtest = DebugBacktest(initial_cash=1000000, position_size=10)  # 10株単位に変更
    backtest.run(
        symbol=args.symbol,
        params_file=args.params,
        start_date=args.start,
        end_date=args.end
    )


if __name__ == "__main__":
    main()
