#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
株の仮想売買バックテストシステム - 厳密版
"""

import sqlite3
import yfinance as yf
import pandas as pd
import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class DataManager:
    """株価データ管理クラス"""

    def __init__(self, db_path: str = "datas.sqlite"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """データベースの初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            )
        ''')
        # キャッシュメタ情報テーブルを追加
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_metadata (
                symbol TEXT,
                date_range TEXT,
                price_source TEXT DEFAULT 'close',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date_range)
            )
        ''')
        conn.commit()
        conn.close()

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        株価データを取得（キャッシュがあれば使用）

        Args:
            symbol: 銘柄コード（例: 9984.T）
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）

        Returns:
            pandas.DataFrame: 株価データ
        """
        # キャッシュの整合性チェックと必要に応じて再取得
        self._check_and_refresh_cache(symbol, start_date, end_date)

        # キャッシュからデータを取得
        cached_data = self._get_cached_data(symbol, start_date, end_date)
        if cached_data is not None and len(cached_data) > 0:
            print(f"キャッシュからデータを取得: {symbol} ({start_date} 〜 {end_date}) - {len(cached_data)}日分")
            return cached_data

        # yfinanceからデータを取得（auto_adjust=Falseで実際のOHLCVを取得）
        print(f"yfinanceからデータを取得中: {symbol} ({start_date} 〜 {end_date})")
        try:
            # end_dateの翌日を指定（非包含のため）
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            end_date_next = (end_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')

            data = yf.download(symbol, start=start_date, end=end_date_next,
                              auto_adjust=False, threads=False)

            if data.empty:
                raise ValueError(f"データが取得できませんでした: {symbol}")

            # MultiIndexの列を単純な列名に変換
            data.columns = data.columns.droplevel(1)  # ティッカー名のレベルを削除
            # 列名を小文字に正規化
            data.columns = [c.lower() for c in data.columns]

            # 'adj close'列が存在する場合は削除（使用禁止）
            if 'adj close' in data.columns:
                data = data.drop('adj close', axis=1)

            print(f"取得したデータ: {len(data)}日分 ({data.index[0].strftime('%Y-%m-%d')} 〜 {data.index[-1].strftime('%Y-%m-%d')})")

            # キャッシュに保存
            self._save_to_cache(symbol, data)

            return data

        except Exception as e:
            print(f"データ取得エラー: {e}")
            sys.exit(1)

    def _get_cached_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """キャッシュからデータを取得"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT date, open, high, low, close, volume
            FROM stock_prices
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
        conn.close()

        if df.empty:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _check_and_refresh_cache(self, symbol: str, start_date: str, end_date: str):
        """キャッシュの整合性チェックと必要に応じて再取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 対象期間のキャッシュデータを確認
        cursor.execute('''
            SELECT DISTINCT date FROM stock_prices
            WHERE symbol = ? AND date BETWEEN ? AND ?
        ''', (symbol, start_date, end_date))

        cached_dates = [row[0] for row in cursor.fetchall()]

        if not cached_dates:
            # キャッシュがなければ何もしない
            conn.close()
            return

        # キャッシュメタ情報を確認
        date_range = f"{start_date}_{end_date}"
        cursor.execute('''
            SELECT price_source FROM cache_metadata
            WHERE symbol = ? AND date_range = ?
        ''', (symbol, date_range))

        meta_result = cursor.fetchone()

        # メタ情報がないか、price_sourceが'adj_close'の場合は再取得
        if meta_result is None or meta_result[0] == 'adj_close':
            print(f"キャッシュの整合性チェック: {symbol} ({start_date} 〜 {end_date}) を再取得します")
            self._refresh_cache_data(symbol, start_date, end_date)

        conn.close()

    def _refresh_cache_data(self, symbol: str, start_date: str, end_date: str):
        """キャッシュデータを再取得して更新"""
        try:
            # 既存のキャッシュを削除
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM stock_prices
                WHERE symbol = ? AND date BETWEEN ? AND ?
            ''', (symbol, start_date, end_date))
            conn.commit()
            conn.close()

            # yfinanceからデータを再取得（auto_adjust=Falseで実際のOHLCVを取得）
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            end_date_next = (end_date_dt + timedelta(days=1)).strftime('%Y-%m-%d')

            data = yf.download(symbol, start=start_date, end=end_date_next,
                              auto_adjust=False, threads=False)

            if not data.empty:
                # MultiIndexの列を単純な列名に変換
                data.columns = data.columns.droplevel(1)  # ティッカー名のレベルを削除
                # 列名を小文字に正規化
                data.columns = [c.lower() for c in data.columns]

                # 'adj close'列が存在する場合は削除（使用禁止）
                if 'adj close' in data.columns:
                    data = data.drop('adj close', axis=1)

                # キャッシュに保存
                self._save_to_cache(symbol, data)

                # メタ情報を更新
                self._update_cache_metadata(symbol, start_date, end_date, 'close')

                print(f"キャッシュを更新しました: {symbol} ({start_date} 〜 {end_date}) - {len(data)}日分")
            else:
                print(f"再取得データが空でした: {symbol}")

        except Exception as e:
            print(f"キャッシュ再取得エラー: {e}")

    def _update_cache_metadata(self, symbol: str, start_date: str, end_date: str, price_source: str):
        """キャッシュメタ情報を更新"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        date_range = f"{start_date}_{end_date}"
        cursor.execute('''
            INSERT OR REPLACE INTO cache_metadata
            (symbol, date_range, price_source)
            VALUES (?, ?, ?)
        ''', (symbol, date_range, price_source))

        conn.commit()
        conn.close()

    def _save_to_cache(self, symbol: str, data: pd.DataFrame):
        """データをキャッシュに保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for date, row in data.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO stock_prices
                (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, date.strftime('%Y-%m-%d'),
                  row['open'], row['high'], row['low'],
                  row['close'], row['volume']))

        conn.commit()
        conn.close()


class Backtest:
    """バックテスト実行クラス（厳密版）"""

    def __init__(self, initial_cash: float = 1000000, position_size: int = 100):
        self.data_manager = DataManager()
        self.initial_cash = initial_cash
        self.position_size = position_size

        # ポジション状態（一元管理）
        self.cash = initial_cash
        self.side = "FLAT"  # "FLAT" or "LONG" or "SHORT"
        self.entry_price = None
        self.size = 0

        # パフォーマンス記録
        self.trades: List[Dict] = []  # 取引履歴
        self.equity_list: List[int] = []  # 資産曲線
        self.closed_trades: List[Dict] = []  # クローズした取引
        self.trade_pairs: List[Dict] = []  # エントリー・決済ペア
        self.wins = 0  # 勝ちトレード数
        self.current_trade: Dict = None  # 現在のトレード情報

    def run(self, strategy_name: str, symbol: str, start_date: str, end_date: str):
        """
        バックテストを実行

        Args:
            strategy_name: 戦略名
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
        """
        print(f"バックテスト開始: {strategy_name} - {symbol}")
        print(f"期間: {start_date} 〜 {end_date}")

        # 株価データの取得（過去90日分のデータも含めるため、開始日を90日前に設定）
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        extended_start_date = (start_dt - timedelta(days=90)).strftime('%Y-%m-%d')

        data = self.data_manager.get_stock_data(symbol, extended_start_date, end_date)

        # 列名の小文字化
        data.columns = [c.lower() for c in data.columns]

        print(f"データ取得完了: {len(data)}日分")

        # 戦略の読み込み
        strategy = self._load_strategy(strategy_name)
        if not strategy:
            print(f"戦略が見つかりません: {strategy_name}")
            return

        # performance.txtを新規作成（既存内容を上書き）
        with open('performance.txt', 'w', encoding='utf-8') as f:
            f.write(f"=== バックテスト開始 ===\n")
            f.write(f"戦略: {strategy_name}\n")
            f.write(f"銘柄: {symbol}\n")
            f.write(f"期間: {start_date} 〜 {end_date}\n\n")

        # 日次ループ（未来データ禁止）
        for i in range(len(data)):
            date = data.index[i]

            # 開始日以前のデータはスキップ
            if date.strftime('%Y-%m-%d') < start_date:
                continue

            # データ欠損チェック
            if pd.isna(data.iloc[i]['close']):
                print(f"WARNING: {date.strftime('%Y-%m-%d')} の終値データが欠損しています。スキップします。")
                continue

            # 評価開始インデックス >= 90 から開始（過去90日分のデータが必要）
            if i < 90:
                # データが不足している日はFLATで出力
                current_close = data.iloc[i]['close']
                total_equity = self.cash
                self.equity_list.append(round(total_equity))

                self._output_daily_log(
                    date.strftime('%Y-%m-%d'),
                    round(current_close),
                    None,
                    0,
                    0,
                    round(total_equity),
                    None
                )
                continue

            # 当日の終値を取得
            current_close = data.iloc[i]['close']

            # 過去90日分のデータを取得（当日を含む）
            past_90 = data.iloc[i-90:i+1]

            # コンテキストを構築
            ctx = {
                "position": self.side,
                "entry_price": self.entry_price,
                "size": self.size
            }

            # 戦略からシグナルを取得
            signal_result = strategy.calculate_signal(past_90, date, ctx)
            signal = signal_result.get('signal', 'HOLD')
            reason = signal_result.get('reason', 'no_signal')
            rci_value = signal_result.get('rci_value')
            adx_value = signal_result.get('adx_value')

            # 取引実行
            trade_executed = False
            pnl = 0

            # BUYシグナル解釈
            if signal == "BUY":
                # FLAT状態 → ロングエントリー
                if self.side == "FLAT":
                    # 資金チェック
                    cost = current_close * self.position_size
                    if cost <= self.cash:
                        # 買い注文実行
                        self.cash -= cost
                        self.side = "LONG"
                        self.entry_price = current_close
                        self.size = self.position_size
                        trade_executed = True

                        # 取引履歴に記録
                        trade = {
                            'date': date.strftime('%Y-%m-%d'),
                            'action': 'BUY',
                            'price': current_close,
                            'quantity': self.position_size,
                            'pnl': 0,
                            'position_type': 'LONG_ENTRY',
                            'rci_value': rci_value,
                            'adx_value': adx_value
                        }
                        self.trades.append(trade)

                        # 現在のトレード情報を記録
                        self.current_trade = {
                            'entry_date': date.strftime('%Y-%m-%d'),
                            'entry_price': current_close,
                            'position_type': 'LONG',
                            'quantity': self.position_size,
                            'rci_value': rci_value,
                            'adx_value': adx_value,
                            'signal_reason': reason
                        }

                # SHORT状態 → ショート解消
                elif self.side == "SHORT":
                    # 買い戻し注文実行
                    cost = current_close * self.size
                    pnl = round((self.entry_price - current_close) * self.size)
                    self.cash -= cost

                    # 勝敗判定
                    if pnl > 0:
                        self.wins += 1

                    # 取引履歴に記録
                    trade = {
                        'date': date.strftime('%Y-%m-%d'),
                        'action': 'BUY',
                        'price': current_close,
                        'quantity': self.size,
                        'pnl': pnl,
                        'reason': reason,
                        'position_type': 'SHORT_EXIT'
                    }
                    self.trades.append(trade)

                    # クローズした取引を記録
                    closed_trade = {
                        'entry_price': self.entry_price,
                        'exit_price': current_close,
                        'pnl': pnl,
                        'reason': reason,
                        'position_type': 'SHORT'
                    }
                    self.closed_trades.append(closed_trade)

                    # トレードペアを作成
                    if self.current_trade:
                        trade_pair = {
                            'entry_date': self.current_trade['entry_date'],
                            'entry_price': self.current_trade['entry_price'],
                            'exit_date': date.strftime('%Y-%m-%d'),
                            'exit_price': current_close,
                            'position_type': 'SHORT',
                            'quantity': self.size,
                            'pnl': pnl,
                            'entry_rci': self.current_trade.get('rci_value'),
                            'entry_adx': self.current_trade.get('adx_value'),
                            'signal_reason': self.current_trade.get('signal_reason'),
                            'exit_reason': reason
                        }
                        self.trade_pairs.append(trade_pair)
                        self.current_trade = None

                    # ポジション状態をリセット
                    self.side = "FLAT"
                    self.entry_price = None
                    self.size = 0
                    trade_executed = True

            # SELLシグナル解釈
            elif signal == "SELL":
                # FLAT状態 → ショートエントリー
                if self.side == "FLAT":
                    # 資金チェック（ショートは証拠金が必要）
                    margin_required = current_close * self.position_size * 0.1  # 10%証拠金に緩和
                    if margin_required <= self.cash:
                        # 売り注文実行
                        self.cash += current_close * self.position_size  # 売却代金を受け取る
                        self.side = "SHORT"
                        self.entry_price = current_close
                        self.size = self.position_size
                        trade_executed = True

                        # 取引履歴に記録
                        trade = {
                            'date': date.strftime('%Y-%m-%d'),
                            'action': 'SELL',
                            'price': current_close,
                            'quantity': self.position_size,
                            'pnl': 0,
                            'position_type': 'SHORT_ENTRY',
                            'rci_value': rci_value,
                            'adx_value': adx_value
                        }
                        self.trades.append(trade)

                        # 現在のトレード情報を記録
                        self.current_trade = {
                            'entry_date': date.strftime('%Y-%m-%d'),
                            'entry_price': current_close,
                            'position_type': 'SHORT',
                            'quantity': self.position_size,
                            'rci_value': rci_value,
                            'adx_value': adx_value,
                            'signal_reason': reason
                        }

                # LONG状態 → ロング解消
                elif self.side == "LONG":
                    # 売り注文実行
                    sell_quantity = self.size  # 現在の保有数量を保存
                    revenue = current_close * sell_quantity
                    pnl = round((current_close - self.entry_price) * sell_quantity)

                    self.cash += revenue

                    # 勝敗判定
                    if pnl > 0:
                        self.wins += 1

                    # 取引履歴に記録
                    trade = {
                        'date': date.strftime('%Y-%m-%d'),
                        'action': 'SELL',
                        'price': current_close,
                        'quantity': sell_quantity,
                        'pnl': pnl,
                        'reason': reason,
                        'position_type': 'LONG_EXIT'
                    }
                    self.trades.append(trade)

                    # クローズした取引を記録
                    closed_trade = {
                        'entry_price': self.entry_price,
                        'exit_price': current_close,
                        'pnl': pnl,
                        'reason': reason,
                        'position_type': 'LONG'
                    }
                    self.closed_trades.append(closed_trade)

                    # トレードペアを作成
                    if self.current_trade:
                        trade_pair = {
                            'entry_date': self.current_trade['entry_date'],
                            'entry_price': self.current_trade['entry_price'],
                            'exit_date': date.strftime('%Y-%m-%d'),
                            'exit_price': current_close,
                            'position_type': 'LONG',
                            'quantity': sell_quantity,
                            'pnl': pnl,
                            'entry_rci': self.current_trade.get('rci_value'),
                            'entry_adx': self.current_trade.get('adx_value'),
                            'signal_reason': self.current_trade.get('signal_reason'),
                            'exit_reason': reason
                        }
                        self.trade_pairs.append(trade_pair)
                        self.current_trade = None

                    # ポジション状態をリセット
                    self.side = "FLAT"
                    self.entry_price = None
                    self.size = 0
                    trade_executed = True

            # 累計資産を計算
            total_equity = self.cash
            if self.side == "LONG":
                total_equity += current_close * self.size
            elif self.side == "SHORT":
                # ショートポジションの評価損益を加算
                short_pnl = (self.entry_price - current_close) * self.size
                total_equity += short_pnl

            # 資産曲線に記録
            self.equity_list.append(round(total_equity))

            # 評価損益を計算（保有中のみ）
            unrealized_pnl = 0
            if self.side == "LONG":
                unrealized_pnl = round((current_close - self.entry_price) * self.size)
            elif self.side == "SHORT":
                unrealized_pnl = round((self.entry_price - current_close) * self.size)

            # ログ出力
            log_size = self.size
            if signal == "SELL" and trade_executed and self.side == "FLAT":
                log_size = self.size  # ロング解消時の数量
            self._output_daily_log(
                date.strftime('%Y-%m-%d'),
                round(current_close),
                signal if trade_executed else None,
                log_size,
                pnl if trade_executed else unrealized_pnl,
                round(total_equity),
                reason if trade_executed else None  # エントリー時にもシグナル情報を記録
            )

        # 最終的なポジションを清算
        if self.side == "LONG":
            last_date = data.index[-1].strftime('%Y-%m-%d')
            last_price = data.iloc[-1]['close']

            revenue = last_price * self.size
            pnl = round((last_price - self.entry_price) * self.size)

            self.cash += revenue

            # 勝敗判定
            if pnl > 0:
                self.wins += 1

            # 取引履歴に記録
            trade = {
                'date': last_date,
                'action': 'SELL',
                'price': last_price,
                'quantity': self.size,
                'pnl': pnl,
                'reason': '清算',
                'position_type': 'LONG_EXIT'
            }
            self.trades.append(trade)

            # クローズした取引を記録
            closed_trade = {
                'entry_price': self.entry_price,
                'exit_price': last_price,
                'pnl': pnl,
                'reason': '清算',
                'position_type': 'LONG'
            }
            self.closed_trades.append(closed_trade)

            # 最終資産を計算
            total_equity = self.cash
            self.equity_list.append(round(total_equity))

            # 清算ログを出力
            self._output_daily_log(
                last_date,
                round(last_price),
                "SELL",
                self.size,
                pnl,
                round(total_equity),
                "清算"
            )

            # ポジション状態をリセット
            self.side = "FLAT"
            self.entry_price = None
            self.size = 0

        elif self.side == "SHORT":
            last_date = data.index[-1].strftime('%Y-%m-%d')
            last_price = data.iloc[-1]['close']

            cost = last_price * self.size
            pnl = round((self.entry_price - last_price) * self.size)
            self.cash -= cost

            # 勝敗判定
            if pnl > 0:
                self.wins += 1

            # 取引履歴に記録
            trade = {
                'date': last_date,
                'action': 'BUY',
                'price': last_price,
                'quantity': self.size,
                'pnl': pnl,
                'reason': '清算',
                'position_type': 'SHORT_EXIT'
            }
            self.trades.append(trade)

            # クローズした取引を記録
            closed_trade = {
                'entry_price': self.entry_price,
                'exit_price': last_price,
                'pnl': pnl,
                'reason': '清算',
                'position_type': 'SHORT'
            }
            self.closed_trades.append(closed_trade)

            # 最終資産を計算
            total_equity = self.cash
            self.equity_list.append(round(total_equity))

            # 清算ログを出力
            self._output_daily_log(
                last_date,
                round(last_price),
                "BUY",
                self.size,
                pnl,
                round(total_equity),
                "清算"
            )

            # ポジション状態をリセット
            self.side = "FLAT"
            self.entry_price = None
            self.size = 0

        # パフォーマンス出力
        self._output_performance(strategy_name, symbol)

        # トレードペアを自動エクスポート
        if self.trade_pairs:
            self.export_trade_pairs('csv')
            self.export_trade_pairs('json')

    def _load_strategy(self, strategy_name: str):
        """戦略を読み込む"""
        try:
            # 戦略モジュールの動的インポート
            strategy_path = f"strategies.{strategy_name}"
            strategy_module = __import__(strategy_path, fromlist=['Strategy'])
            return strategy_module.Strategy()
        except Exception as e:
            print(f"戦略読み込みエラー: {e}")
            return None

    def _output_daily_log(self, date: str, close_price: int, signal: str,
                         size: int, pnl: int, total_equity: int, reason: str = None):
        """日次ログを出力（厳密なフォーマット）"""
        with open('performance.txt', 'a', encoding='utf-8') as f:
            # 行頭は必ず：YYYY-MM-DD 終値=XXXX
            base_output = f"{date} 終値={close_price}"

            if signal == "BUY":
                # BUYシグナル：ロング建て or ショート解消
                if self.side == "LONG":
                    # ロング建て（エントリー時）
                    if reason:
                        f.write(f"{base_output} BUY 数量={size} ロング建て 累計資産={total_equity} シグナル={reason}\n")
                    else:
                        f.write(f"{base_output} BUY 数量={size} ロング建て 累計資産={total_equity}\n")
                elif self.side == "FLAT":
                    # ショート解消
                    f.write(f"{base_output} BUY 数量={size} ショート決済 損益={pnl:+} 累計資産={total_equity} 理由={reason}\n")
                else:
                    # その他のケース
                    f.write(f"{base_output} BUY 数量={size} 損益=0 累計資産={total_equity}\n")

            elif signal == "SELL":
                # SELLシグナル：ショート建て or ロング解消
                if self.side == "SHORT":
                    # ショート建て（エントリー時）
                    if reason:
                        f.write(f"{base_output} SELL 数量={size} ショート建て 累計資産={total_equity} シグナル={reason}\n")
                    else:
                        f.write(f"{base_output} SELL 数量={size} ショート建て 累計資産={total_equity}\n")
                elif self.side == "FLAT":
                    # ロング解消
                    f.write(f"{base_output} SELL 数量={size} ロング決済 損益={pnl:+} 累計資産={total_equity} 理由={reason}\n")
                else:
                    # その他のケース
                    f.write(f"{base_output} SELL 数量={size} 損益={pnl:+} 累計資産={total_equity} 理由={reason}\n")
            else:
                # 非取引日
                if self.side == "LONG":
                    # side=="LONG" → "HOLD ロングNNN株保持中 評価損益=±PPP 累計資産=XXXXX"
                    f.write(f"{base_output} HOLD ロング{size}株保持中 評価損益={pnl:+} 累計資産={total_equity}\n")
                elif self.side == "SHORT":
                    # side=="SHORT" → "HOLD ショートNNN株保持中 評価損益=±PPP 累計資産=XXXXX"
                    f.write(f"{base_output} HOLD ショート{size}株保持中 評価損益={pnl:+} 累計資産={total_equity}\n")
                else:
                    # side=="FLAT" → "FLAT ポジションなし 累計資産=XXXXX"
                    f.write(f"{base_output} FLAT ポジションなし 累計資産={total_equity}\n")

    def _output_performance(self, strategy_name: str, symbol: str):
        """パフォーマンスを出力"""
        # 最大ドローダウンを計算
        max_dd = self._calculate_max_drawdown()

        # 集計統計
        total_trades = len(self.closed_trades)
        total_pnl = sum(trade['pnl'] for trade in self.closed_trades)
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0.0
        final_cash = round(self.cash)

        # ポジションタイプ別の集計
        long_trades = [t for t in self.closed_trades if t.get('position_type') == 'LONG']
        short_trades = [t for t in self.closed_trades if t.get('position_type') == 'SHORT']
        long_pnl = sum(t['pnl'] for t in long_trades)
        short_pnl = sum(t['pnl'] for t in short_trades)

        with open('performance.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n=== バックテスト結果 ===\n")
            f.write(f"戦略: {strategy_name}\n")
            f.write(f"銘柄: {symbol}\n")
            f.write(f"取引回数: {total_trades}\n")
            f.write(f"ロング取引: {len(long_trades)}\n")
            f.write(f"ショート取引: {len(short_trades)}\n")
            f.write(f"勝ちトレード: {self.wins}\n")
            f.write(f"負けトレード: {total_trades - self.wins}\n")
            f.write(f"勝率: {win_rate:.1f}%\n")
            f.write(f"総損益: {total_pnl:+}\n")
            f.write(f"ロング損益: {long_pnl:+}\n")
            f.write(f"ショート損益: {short_pnl:+}\n")
            f.write(f"最終資金: {final_cash}\n")
            f.write(f"最大ドローダウン: -{max_dd:.1f}%\n")

            # サマリ出力
            f.write(f"\n=== サマリ ===\n")
            f.write(f"総損益: {total_pnl:+}\n")
            f.write(f"勝率: {win_rate:.1f}%\n")
            f.write(f"取引回数: {total_trades}\n")
            f.write(f"最大ドローダウン: -{max_dd:.1f}%\n")

        print("\n=== バックテスト結果 ===")
        print(f"取引回数: {total_trades}")
        print(f"ロング取引: {len(long_trades)}")
        print(f"ショート取引: {len(short_trades)}")
        print(f"勝ちトレード: {self.wins}")
        print(f"負けトレード: {total_trades - self.wins}")
        print(f"勝率: {win_rate:.1f}%")
        print(f"総損益: {total_pnl:+}")
        print(f"ロング損益: {long_pnl:+}")
        print(f"ショート損益: {short_pnl:+}")
        print(f"最終資金: {final_cash}")
        print(f"最大ドローダウン: -{max_dd:.1f}%")

        # コンソールにもサマリを出力
        print(f"\n=== サマリ ===")
        print(f"総損益: {total_pnl:+}")
        print(f"勝率: {win_rate:.1f}%")
        print(f"取引回数: {total_trades}")
        print(f"最大ドローダウン: -{max_dd:.1f}%")

    def _calculate_max_drawdown(self) -> float:
        """最大ドローダウンを計算（資産曲線ベース）"""
        if not self.equity_list:
            return 0.0

        peak = self.equity_list[0]
        max_drawdown = 0.0

        for equity in self.equity_list:
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100 if peak != 0 else 0.0
            max_drawdown = max(max_drawdown, dd)

        return max_drawdown

    def export_trade_pairs(self, format_type: str = 'csv'):
        """
        トレードペアをエクスポート

        Args:
            format_type: 出力形式 ('csv' または 'json')
        """
        if not self.trade_pairs:
            print("エクスポートするトレードデータがありません")
            return

        if format_type == 'csv':
            self._export_to_csv()
        elif format_type == 'json':
            self._export_to_json()
        else:
            print(f"サポートされていない形式: {format_type}")

    def _export_to_csv(self):
        """CSV形式でエクスポート"""
        import csv

        filename = 'trade_pairs.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # ヘッダー行
            writer.writerow([
                'エントリー日', 'エントリー価格', '決済日', '決済価格',
                'ポジションタイプ', '数量', '損益', 'エントリーRCI', 'エントリーADX',
                'シグナル理由', '決済理由'
            ])

            # データ行
            for trade in self.trade_pairs:
                writer.writerow([
                    trade['entry_date'],
                    trade['entry_price'],
                    trade['exit_date'],
                    trade['exit_price'],
                    trade['position_type'],
                    trade['quantity'],
                    trade['pnl'],
                    trade.get('entry_rci', ''),
                    trade.get('entry_adx', ''),
                    trade.get('signal_reason', ''),
                    trade.get('exit_reason', '')
                ])

        print(f"トレードペアをCSV形式でエクスポートしました: {filename}")

    def _export_to_json(self):
        """JSON形式でエクスポート"""
        import json

        filename = 'trade_pairs.json'

        # データをJSON形式で整形
        export_data = {
            'trade_pairs': self.trade_pairs,
            'summary': {
                'total_trades': len(self.trade_pairs),
                'total_pnl': sum(trade['pnl'] for trade in self.trade_pairs),
                'long_trades': len([t for t in self.trade_pairs if t['position_type'] == 'LONG']),
                'short_trades': len([t for t in self.trade_pairs if t['position_type'] == 'SHORT']),
                'winning_trades': len([t for t in self.trade_pairs if t['pnl'] > 0]),
                'losing_trades': len([t for t in self.trade_pairs if t['pnl'] < 0])
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"トレードペアをJSON形式でエクスポートしました: {filename}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='株の仮想売買バックテスト')
    parser.add_argument('--strategy', required=True, help='戦略名（strategiesフォルダ内のファイル名）')
    parser.add_argument('--symbol', required=True, help='銘柄コード（例: 9984.T）')
    parser.add_argument('--start', required=True, help='開始日（YYYY-MM-DD）')
    parser.add_argument('--end', required=True, help='終了日（YYYY-MM-DD）')
    parser.add_argument('--position_size', type=int, default=100, help='ポジションサイズ（デフォルト: 100株）')

    args = parser.parse_args()

    backtest = Backtest(position_size=args.position_size)
    backtest.run(args.strategy, args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
