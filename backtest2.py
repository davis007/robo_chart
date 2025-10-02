#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
パラメータ駆動型バックテストシステム - backtest2.py
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


class ParametricBacktest:
    """パラメータ駆動型バックテストクラス"""

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
        self.rules = None

    def run(self, symbol: str, params_file: str, rules_file: str = None,
            start_date: str = None, end_date: str = None, report_file: str = None):
        """
        バックテストを実行

        Args:
            symbol: 銘柄コード
            params_file: パラメータJSONファイル
            rules_file: ルールセットファイル（任意）
            start_date: 開始日
            end_date: 終了日
            report_file: レポートファイル
        """
        print(f"パラメータ駆動型バックテスト開始: {symbol}")

        # パラメータ読み込み
        if not self._load_params(params_file):
            return

        # ルールセット読み込み（任意）
        if rules_file:
            self._load_rules(rules_file)

        # 日付範囲の設定
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            # 検証期間を使用
            start_date = self.params.get("split", {}).get("valid_start")
            if not start_date:
                start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')

        print(f"期間: {start_date} 〜 {end_date}")

        # 株価データ取得
        data = self.data_manager.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            print(f"データが取得できませんでした: {symbol}")
            return

        print(f"データ取得完了: {len(data)}日分")

        # バックテスト実行
        self._run_backtest(data, symbol)

        # パフォーマンス出力
        self._output_performance(symbol, data, report_file)

    def _load_params(self, params_file: str) -> bool:
        """パラメータファイルを読み込み"""
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                self.params = json.load(f)

            print(f"パラメータ読み込み完了: {params_file}")
            print(f"銘柄: {self.params.get('symbol')}, 期間: {self.params.get('span_years')}年")
            return True

        except Exception as e:
            print(f"パラメータ読み込みエラー: {e}")
            return False

    def _load_rules(self, rules_file: str):
        """ルールセットファイルを読み込み（YAML対応予定）"""
        # 現在はJSONのみ対応
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            print(f"ルールセット読み込み完了: {rules_file}")
        except Exception as e:
            print(f"ルールセット読み込み警告: {e}")

    def _run_backtest(self, data: pd.DataFrame, symbol: str):
        """バックテストを実行"""
        features = self.params.get("features", {})
        stats = self.params.get("stats", {})

        # デフォルトルールの設定
        default_rules = self._get_default_rules(features, stats)

        for i in range(len(data)):
            date = data.index[i]

            # データ欠損チェック
            if pd.isna(data.iloc[i]['close']):
                continue

            # 当日の終値を取得
            current_close = data.iloc[i]['close']

            # 過去データの取得（指標計算用）
            if i >= 20:  # 最低20日分のデータが必要
                past_data = data.iloc[max(0, i-60):i+1]  # 直近60日分

                # シグナル判定
                signal_result = self._calculate_signal(past_data, date, features, stats, default_rules)
                signal = signal_result.get('signal', 'HOLD')
                reason = signal_result.get('reason', 'no_signal')

                # 取引実行
                trade_executed = False
                pnl = 0

                # BUYシグナル
                if signal == "BUY":
                    trade_executed = self._execute_buy(current_close, date, reason)

                # SELLシグナル
                elif signal == "SELL":
                    trade_executed, pnl = self._execute_sell(current_close, date, reason)

                # 評価損益計算
                total_equity = self._calculate_total_equity(current_close)
                self.equity_list.append(round(total_equity))

                # ログ出力
                self._output_daily_log(
                    date.strftime('%Y-%m-%d'),
                    round(current_close),
                    signal if trade_executed else None,
                    self.size,
                    pnl if trade_executed else self._calculate_unrealized_pnl(current_close),
                    round(total_equity),
                    reason if trade_executed else None
                )

        # 最終ポジション清算
        self._close_final_position(data)

    def _get_default_rules(self, features: Dict, stats: Dict) -> Dict:
        """デフォルトルールを生成"""
        returns = features.get("returns", {})
        trend = features.get("trend", {})
        vol = features.get("vol", {})
        osc = features.get("osc", {})
        candle = features.get("candle", {})
        volume = features.get("volume", {})
        streak = features.get("streak", {})
        dd = features.get("dd", {})

        rules = {
            # 損切りルール
            "stop_loss": {
                "long": dd.get("daily_dd_mean", -0.01) + 2 * dd.get("daily_dd_std", 0.02),
                "short": -(dd.get("daily_dd_mean", -0.01) + 2 * dd.get("daily_dd_std", 0.02))
            },

            # 利確ルール
            "take_profit": {
                "long": returns.get("r_p90_w20", 0.05),
                "short": -returns.get("r_p90_w20", 0.05)
            },

            # エントリー条件
            "entry_conditions": {
                # トレンド条件
                "trend_strength": trend.get("above_sma_ratio_w20", 0.5) > 0.6,
                "volatility_condition": vol.get("bb_width_p90_w20", 0.03) > 0.02,

                # オシレーター条件
                "rsi_condition": osc.get("rsi14_mean", 50) < 70,

                # ボラティリティ条件
                "vol_condition": vol.get("hv_w20", 0.2) > 0.15
            },

            # 保有日数制限
            "max_hold_days": min(20, streak.get("up_streak_max", 10)),

            # 暴落例外ルール
            "crash_exception": {
                "threshold": dd.get("crash_window_dd_p95_w20", -0.08),
                "action": "time_weighted_exit"
            }
        }

        return rules

    def _calculate_signal(self, data: pd.DataFrame, date: pd.Timestamp,
                         features: Dict, stats: Dict, rules: Dict) -> Dict:
        """売買シグナルを計算"""
        current_close = data.iloc[-1]['close']

        # ポジション状態に基づいて判定
        if self.side == "FLAT":
            return self._evaluate_entry_signal(data, date, features, rules)
        else:
            return self._evaluate_exit_signal(data, date, features, rules)

    def _evaluate_entry_signal(self, data: pd.DataFrame, date: pd.Timestamp,
                              features: Dict, rules: Dict) -> Dict:
        """エントリーシグナル評価"""
        current_close = data.iloc[-1]['close']
        returns_feat = features.get("returns", {})
        trend_feat = features.get("trend", {})
        vol_feat = features.get("vol", {})
        osc_feat = features.get("osc", {})
        candle_feat = features.get("candle", {})
        volume_feat = features.get("volume", {})

        entry_conditions = rules.get("entry_conditions", {})

        # 基本条件チェック
        if not entry_conditions.get("trend_strength", True):
            return {'signal': 'HOLD', 'reason': 'trend_weak'}

        if not entry_conditions.get("volatility_condition", True):
            return {'signal': 'HOLD', 'reason': 'low_volatility'}

        if not entry_conditions.get("rsi_condition", True):
            return {'signal': 'HOLD', 'reason': 'rsi_overbought'}

        # ローソク足パターン
        engulfing_winrate = candle_feat.get("engulfing_pos_next3d_winrate", 0)
        if engulfing_winrate > 0.6:
            return {'signal': 'BUY', 'reason': 'engulfing_pattern'}

        # ボリンジャーバンド反発
        bb_width = vol_feat.get("bb_width_p90_w20", 0)
        if bb_width > 0.03:
            # 簡易的なBB反発判定
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            if current_close < sma_20 * 0.98:
                return {'signal': 'BUY', 'reason': 'bb_bounce'}

        # 出来高スパイク
        vol_spike_return = volume_feat.get("vol_spike_next3d_mean", 0)
        if vol_spike_return > 0.01:
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            if current_volume > avg_volume * 1.5:
                return {'signal': 'BUY', 'reason': 'volume_spike'}

        return {'signal': 'HOLD', 'reason': 'no_entry_signal'}

    def _evaluate_exit_signal(self, data: pd.DataFrame, date: pd.Timestamp,
                             features: Dict, rules: Dict) -> Dict:
        """エグジットシグナル評価"""
        current_close = data.iloc[-1]['close']
        dd_feat = features.get("dd", {})
        returns_feat = features.get("returns", {})

        stop_loss_rules = rules.get("stop_loss", {})
        take_profit_rules = rules.get("take_profit", {})

        # 損切りチェック
        if self.side == "LONG":
            stop_loss_level = self.entry_price * (1 + stop_loss_rules.get("long", -0.03))
            if current_close <= stop_loss_level:
                return {'signal': 'SELL', 'reason': 'stop_loss'}

            take_profit_level = self.entry_price * (1 + take_profit_rules.get("long", 0.05))
            if current_close >= take_profit_level:
                return {'signal': 'SELL', 'reason': 'take_profit'}

        elif self.side == "SHORT":
            stop_loss_level = self.entry_price * (1 + stop_loss_rules.get("short", 0.03))
            if current_close >= stop_loss_level:
                return {'signal': 'BUY', 'reason': 'stop_loss'}

            take_profit_level = self.entry_price * (1 + take_profit_rules.get("short", -0.05))
            if current_close <= take_profit_level:
                return {'signal': 'BUY', 'reason': 'take_profit'}

        # 保有日数制限
        if self.entry_date:
            hold_days = (date - self.entry_date).days
            max_hold = rules.get("max_hold_days", 20)
            if hold_days >= max_hold:
                return {'signal': 'SELL' if self.side == "LONG" else 'BUY',
                        'reason': 'max_hold_days'}

        # 暴落例外ルール
        crash_threshold = rules.get("crash_exception", {}).get("threshold", -0.08)
        recent_dd = self._calculate_recent_drawdown(data)
        if recent_dd <= crash_threshold:
            return {'signal': 'SELL' if self.side == "LONG" else 'BUY',
                    'reason': 'crash_exception'}

        return {'signal': 'HOLD', 'reason': 'no_exit_signal'}

    def _calculate_recent_drawdown(self, data: pd.DataFrame) -> float:
        """直近のドローダウンを計算"""
        if len(data) < 5:
            return 0

        recent_data = data.iloc[-5:]
        peak = recent_data['close'].max()
        current = recent_data['close'].iloc[-1]
        return (current - peak) / peak

    def _execute_buy(self, price: float, date: pd.Timestamp, reason: str) -> bool:
        """買い注文実行"""
        if self.side == "FLAT":
            # ロングエントリー
            cost = price * self.position_size
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
                    'position_type': 'LONG_ENTRY',
                    'reason': reason
                }
                self.trades.append(trade)
                return True

        elif self.side == "SHORT":
            # ショート解消
            cost = price * self.size
            pnl = round((self.entry_price - price) * self.size)
            self.cash -= cost

            if pnl > 0:
                self.wins += 1

            trade = {
                'date': date.strftime('%Y-%m-%d'),
                'action': 'BUY',
                'price': price,
                'quantity': self.size,
                'pnl': pnl,
                'reason': reason,
                'position_type': 'SHORT_EXIT'
            }
            self.trades.append(trade)

            closed_trade = {
                'entry_price': self.entry_price,
                'exit_price': price,
                'pnl': pnl,
                'reason': reason,
                'position_type': 'SHORT'
            }
            self.closed_trades.append(closed_trade)

            self.side = "FLAT"
            self.entry_price = None
            self.size = 0
            self.entry_date = None
            return True

        return False

    def _execute_sell(self, price: float, date: pd.Timestamp, reason: str) -> Tuple[bool, int]:
        """売り注文実行"""
        pnl = 0

        if self.side == "FLAT":
            # ショートエントリー
            margin_required = price * self.position_size * 0.1
            if margin_required <= self.cash:
                self.cash += price * self.position_size
                self.side = "SHORT"
                self.entry_price = price
                self.size = self.position_size
                self.entry_date = date

                trade = {
                    'date': date.strftime('%Y-%m-%d'),
                    'action': 'SELL',
                    'price': price,
                    'quantity': self.position_size,
                    'pnl': 0,
                    'position_type': 'SHORT_ENTRY',
                    'reason': reason
                }
                self.trades.append(trade)
                return True, pnl

        elif self.side == "LONG":
            # ロング解消
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
                'reason': reason,
                'position_type': 'LONG_EXIT'
            }
            self.trades.append(trade)

            closed_trade = {
                'entry_price': self.entry_price,
                'exit_price': price,
                'pnl': pnl,
                'reason': reason,
                'position_type': 'LONG'
            }
            self.closed_trades.append(closed_trade)

            self.side = "FLAT"
            self.entry_price = None
            self.size = 0
            self.entry_date = None
            return True, pnl

        return False, pnl

    def _calculate_total_equity(self, current_price: float) -> float:
        """総資産を計算"""
        if self.side == "LONG":
            return self.cash + current_price * self.size
        elif self.side == "SHORT":
            return self.cash + (self.entry_price - current_price) * self.size
        else:
            return self.cash

    def _calculate_unrealized_pnl(self, current_price: float) -> int:
        """評価損益を計算"""
        if self.side == "LONG":
            return round((current_price - self.entry_price) * self.size)
        elif self.side == "SHORT":
            return round((self.entry_price - current_price) * self.size)
        else:
            return 0

    def _close_final_position(self, data: pd.DataFrame):
        """最終ポジションを清算"""
        if self.side != "FLAT" and len(data) > 0:
            final_price = data.iloc[-1]['close']
            if self.side == "LONG":
                self._execute_sell(final_price, data.index[-1], "final_close")
            elif self.side == "SHORT":
                self._execute_buy(final_price, data.index[-1], "final_close")

    def _output_daily_log(self, date: str, price: int, signal: str, size: int,
                         pnl: int, equity: int, reason: str):
        """日次ログ出力"""
        if signal:
            print(f"{date} | {signal:4} | {price:6} | {size:3} | {pnl:+6} | {equity:8} | {reason}")

    def _output_performance(self, symbol: str, data: pd.DataFrame, report_file: str = None):
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
        print("=== パラメータ駆動型バックテスト結果 ===")
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
                print(f"{i:2d}. {trade['position_type']:6} | {trade['pnl']:+6}円 | {trade['reason']}")

        # レポートファイル出力
        if report_file:
            self._save_report(report_file, symbol, total_pnl, total_trades, win_rate, max_dd)

    def _save_report(self, report_file: str, symbol: str, total_pnl: int,
                    total_trades: int, win_rate: float, max_dd: float):
        """レポートファイルを保存"""
        try:
            report_dir = Path("reports2")
            report_dir.mkdir(exist_ok=True)

            if not report_file.startswith("reports2/"):
                report_file = report_dir / report_file

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"=== パラメータ駆動型バックテストレポート ===\n")
                f.write(f"銘柄: {symbol}\n")
                f.write(f"日付: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"パラメータファイル: {self.params.get('symbol', 'N/A')}.json\n")
                f.write(f"期間: {self.params.get('split', {}).get('valid_start', 'N/A')} 〜 {self.params.get('split', {}).get('valid_end', 'N/A')}\n\n")

                f.write(f"初期資金: {self.initial_cash:,}円\n")
                f.write(f"最終資金: {self.equity_list[-1] if self.equity_list else self.initial_cash:,}円\n")
                f.write(f"総損益: {total_pnl:+,}円\n")
                f.write(f"取引回数: {total_trades}回\n")
                f.write(f"勝率: {win_rate:.1f}%\n")
                f.write(f"最大ドローダウン: {max_dd:.1f}%\n\n")

                f.write("--- 使用パラメータ ---\n")
                notes = self.params.get("notes", [])
                for note in notes:
                    f.write(f"- {note}\n")

                f.write("\n--- 取引履歴 ---\n")
                for trade in self.trades:
                    f.write(f"{trade['date']} | {trade['action']:4} | {trade['price']:6} | {trade['quantity']:3} | {trade['pnl']:+6} | {trade['reason']}\n")

            print(f"レポート保存: {report_file}")

        except Exception as e:
            print(f"レポート保存エラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='パラメータ駆動型バックテスト')
    parser.add_argument('--symbol', required=True, help='銘柄コード（例: 9984.T）')
    parser.add_argument('--params', required=True, help='パラメータJSONファイル')
    parser.add_argument('--rules', help='ルールセットファイル（任意）')
    parser.add_argument('--cash', type=float, default=1000000, help='初期資金（デフォルト: 1,000,000円）')
    parser.add_argument('--unit', type=int, default=100, help='取引単位（デフォルト: 100株）')
    parser.add_argument('--start', help='開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='終了日（YYYY-MM-DD）')
    parser.add_argument('--report', help='レポートファイル')

    args = parser.parse_args()

    backtest = ParametricBacktest(initial_cash=args.cash, position_size=args.unit)
    backtest.run(
        symbol=args.symbol,
        params_file=args.params,
        rules_file=args.rules,
        start_date=args.start,
        end_date=args.end,
        report_file=args.report
    )


if __name__ == "__main__":
    main()
