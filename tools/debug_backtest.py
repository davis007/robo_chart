#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デバッグ用バックテストスクリプト - ATRトレーリング・ストップの役割変更
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

        # トレーリング・ストップ用変数
        self.peak_price = 0.0  # ポジション保有中の最高価格

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

            # パラメータ読み込み後の強制上書き (デバッグおよび最終調整用)
            # ATRトレーリング・ストップを純粋なリスク管理として再導入
            try:
                print("【FINAL TUNE】ATRトレーリング・ストップを純粋なリスク管理として再導入")

                # エントリー感度と防御の調整（変更なし）
                self.params['MIN_BUY_SIGNAL_COUNT'] = 1
                self.params['MAX_ENTRY_RSI'] = 85.0
                self.params['MAX_ENTRY_SMA_DEVIATION'] = 0.50

                # 決済基準の定義 - 純粋なリスク管理アプローチ
                # 1. 固定利確の復活（+15%）
                self.params['FIXED_TAKE_PROFIT'] = 0.15
                # 2. ATRベースの固定損切り（購入価格からの損失追跡）
                self.params['ATR_LOSS_MULTIPLIER'] = 4.0
                # 3. 時間制限（変更なし）
                self.params['MAX_HOLD_DAYS_EXIT'] = 20

            except Exception as e:
                print(f"警告: JSON強制上書きエラー: {e}")
                # エラーが発生しても、プログラムが停止しないように継続

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
                    # 安全性フィルター適用
                    safety_rejected = self._check_entry_safety(past_data, current_close)
                    if safety_rejected:
                        print(f"  → 安全性フィルターによりエントリー却下")
                        continue
                    self._execute_buy(current_close, date, "debug_entry")
                elif signal == "SELL" and self.side == "LONG":
                    # 最低保有日数チェック
                    if self._check_min_holding_days(date):
                        self._execute_sell(current_close, date, "debug_exit")
                    else:
                        print(f"  → 最低保有日数未満のためHOLD: {self._get_holding_days(date)}日")
                else:
                    print(f"  → ポジション状態不一致: signal={signal}, side={self.side}")

            # 決済条件チェック
            if self.side == "LONG":
                current_close = data.iloc[i]['close']

                # 優先度 1: SELLシグナル（純粋トレンド終焉）
                if signal == "SELL":
                    # SELLシグナルは debug_entry 内でチェック済みだが、ここで最終決定
                    self._execute_sell(current_close, date, "pure_trend_exit") # SELLシグナルで決済

                # 優先度 2: 緊急避難ストップロスチェック
                stop_loss_reason = self._check_stop_loss_conditions(data, i, date)
                if stop_loss_reason:
                    self._execute_sell(current_close, date, stop_loss_reason)
                # 優先度 3: 時間切れ決済は無効化（緊急避難のみ）
                elif not self._check_min_holding_days(date):
                    # 最低保有日数を満たさない場合はHOLDを維持
                    print(f"  → 最低保有日数未満のためHOLD: {self._get_holding_days(date)}日")

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

        # 最小シグナル数チェック（JSONパラメータから取得）
        min_buy_signal_count = self.params.get("MIN_BUY_SIGNAL_COUNT", 2)

        # 非対称エントリー条件: BUYはJSONパラメータで制御、SELLは厳格(3/3以上)
        total_signals = 5  # 全シグナル数

        if self.side == "FLAT" and buy_signals >= min_buy_signal_count:  # BUY: JSONパラメータで制御
            print(f"    → 複数シグナル一致: BUYシグナル{buy_signals}個 >= {min_buy_signal_count}, SELLシグナル{sell_signals}個")
            return "BUY"
        elif self.side == "LONG" and sell_signals >= 3:  # SELL: 3/3以上で許可
            print(f"    → 複数シグナル一致: BUYシグナル{buy_signals}個, SELLシグナル{sell_signals}個")
            return "SELL"

        if buy_signals > 0 or sell_signals > 0:
            if self.side == "FLAT" and buy_signals < min_buy_signal_count:
                print(f"    → 【最小シグナル数却下】シグナル数 {buy_signals} < {min_buy_signal_count}")
            else:
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
        """RSIを計算"""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def _check_stop_loss_conditions(self, data: pd.DataFrame, current_index: int, current_date: pd.Timestamp) -> Optional[str]:
        """損切り条件チェック - 緊急避難ロスカット"""
        if self.entry_price is None:
            return None

        # 固定パーセント損切り（緊急避難用）
        stop_loss_pct = 0.10  # -10%を最大許容損失とする (JSONから読み込む値に置き換えてください)

        stop_loss_price = self.entry_price * (1 - stop_loss_pct)
        current_low = data.iloc[current_index]['low']

        if current_low <= stop_loss_price:
            print(f"  → 緊急避難発動: 安値{current_low} <= 損切りライン{stop_loss_price:.1f} ({stop_loss_pct*100:.1f}%)")
            return "emergency_fixed_sl"

        return None

    def _check_exit_conditions(self, data: pd.DataFrame, current_index: int, current_date: datetime) -> Optional[str]:
        """損切り条件チェック - 全ての固定/ATR損切りロジックを無効化（純粋トレンド決済戦略）"""
        # この戦略では、損切りは「時間切れ」または「SELLシグナル」のみで実行されます。
        return None

    def _calculate_current_atr(self, data: pd.DataFrame, current_index: int, period: int = 14) -> float:
        """現在のATRを計算"""
        if current_index < period:
            return 0.0

        # 過去period日分のデータを取得
        recent_data = data.iloc[current_index-period:current_index+1]

        # TR (True Range) の計算
        tr1 = recent_data['high'] - recent_data['low']
        tr2 = abs(recent_data['high'] - recent_data['close'].shift(1))
        tr3 = abs(recent_data['low'] - recent_data['close'].shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATRの計算
        atr = true_range.rolling(window=period).mean()

        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0

    def _get_holding_days(self, current_date: datetime) -> int:
        """保有日数を計算"""
        if self.entry_date is None:
            return 0
        return (current_date - self.entry_date).days

    def _check_entry_safety(self, data: pd.DataFrame, current_price: float) -> bool:
        """エントリー安全性チェック"""
        # RSIが高すぎる場合はエントリーを却下
        rsi = self._calculate_rsi(data['close'])
        if rsi > self.params.get('MAX_ENTRY_RSI', 85.0):
            return True

        # 移動平均からの乖離が大きすぎる場合は却下
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        deviation = abs(current_price - sma_20) / sma_20
        if deviation > self.params.get('MAX_ENTRY_SMA_DEVIATION', 0.50):
            return True

        return False

    def _check_min_holding_days(self, current_date: datetime) -> bool:
        """最低保有日数チェック"""
        holding_days = self._get_holding_days(current_date)
        return holding_days >= self.min_holding_days

    def _execute_buy(self, price: float, date: datetime, reason: str):
        """買い注文実行"""
        cost = price * self.position_size
        if self.cash >= cost:
            self.cash -= cost
            self.side = "LONG"
            self.entry_price = price
            self.size = self.position_size
            self.entry_date = date
            self.peak_price = price  # 最高価格を初期化

            trade = {
                'date': date,
                'side': 'BUY',
                'price': price,
                'size': self.position_size,
                'reason': reason
            }
            self.trades.append(trade)
            print(f"  → 買い注文実行: {price}円, 理由: {reason}")

    def _execute_sell(self, price: float, date: datetime, reason: str):
        """売り注文実行"""
        if self.side == "LONG" and self.size > 0:
            revenue = price * self.size
            self.cash += revenue
            profit = (price - self.entry_price) * self.size
            profit_rate = (price - self.entry_price) / self.entry_price

            trade = {
                'date': date,
                'side': 'SELL',
                'price': price,
                'size': self.size,
                'reason': reason,
                'profit': profit,
                'profit_rate': profit_rate,
                'entry_price': self.entry_price
            }
            self.closed_trades.append(trade)

            if profit > 0:
                self.wins += 1

            print(f"  → 売り注文実行: {price}円, 利益: {profit:,.0f}円 ({profit_rate:.1%}), 理由: {reason}")

            # ポジション状態リセット
            self.side = "FLAT"
            self.entry_price = None
            self.size = 0
            self.entry_date = None
            self.peak_price = 0.0

    def _calculate_total_equity(self, current_price: float) -> float:
        """総資産評価額を計算"""
        position_value = self.size * current_price if self.side == "LONG" else 0
        return self.cash + position_value

    def _close_final_position(self, data: pd.DataFrame):
        """最終ポジション清算"""
        if self.side == "LONG" and self.size > 0:
            last_close = data.iloc[-1]['close']
            self._execute_sell(last_close, data.index[-1], "final_close")

    def _output_performance(self, symbol: str, data: pd.DataFrame):
        """パフォーマンス出力"""
        if not self.closed_trades:
            print("取引がありませんでした")
            return

        total_trades = len(self.closed_trades)
        win_rate = self.wins / total_trades if total_trades > 0 else 0
        total_profit = sum(trade['profit'] for trade in self.closed_trades)
        total_return = total_profit / self.initial_cash

        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        avg_profit_rate = sum(trade['profit_rate'] for trade in self.closed_trades) / total_trades if total_trades > 0 else 0

        max_profit = max(trade['profit'] for trade in self.closed_trades) if self.closed_trades else 0
        max_loss = min(trade['profit'] for trade in self.closed_trades) if self.closed_trades else 0

        final_equity = self._calculate_total_equity(data.iloc[-1]['close'])
        total_return_percent = (final_equity - self.initial_cash) / self.initial_cash * 100

        print(f"\n=== パフォーマンスレポート ({symbol}) ===")
        print(f"総取引回数: {total_trades}回")
        print(f"勝率: {win_rate:.1%} ({self.wins}/{total_trades})")
        print(f"総利益: {total_profit:,.0f}円")
        print(f"総リターン: {total_return:.1%}")
        print(f"平均利益: {avg_profit:,.0f}円")
        print(f"平均利益率: {avg_profit_rate:.1%}")
        print(f"最大利益: {max_profit:,.0f}円")
        print(f"最大損失: {max_loss:,.0f}円")
        print(f"最終資産: {final_equity:,.0f}円")
        print(f"総リターン率: {total_return_percent:.1f}%")

        # 詳細な取引履歴
        print(f"\n=== 取引履歴 ===")
        for i, trade in enumerate(self.closed_trades[-10:], 1):  # 直近10件のみ表示
            print(f"{i}. {trade['date'].strftime('%Y-%m-%d')}: {trade['side']} {trade['price']}円, "
                  f"利益: {trade['profit']:,.0f}円 ({trade['profit_rate']:.1%}), 理由: {trade['reason']}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='デバッグ用バックテスト')
    parser.add_argument('symbol', help='銘柄コード (例: 9984.T)')
    parser.add_argument('params_file', help='パラメータファイルのパス')
    parser.add_argument('--start', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', help='終了日 (YYYY-MM-DD)')

    args = parser.parse_args()

    backtest = DebugBacktest()
    backtest.run(args.symbol, args.params_file, args.start, args.end)


if __name__ == "__main__":
    main()
