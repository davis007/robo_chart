#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
監視銘柄リスト管理CLIユーティリティ
"""

import sqlite3
import yfinance as yf
import argparse
import sys
import re
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import importlib.util


class WatchlistManager:
    """監視銘柄リスト管理クラス"""

    def __init__(self, db_path: str = "datas.sqlite"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """データベースの初期化（新規テーブルのみ追加）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 監視銘柄テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # ポジション履歴テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                position TEXT NOT NULL,
                size INTEGER NOT NULL,
                entry_price REAL,
                pnl INTEGER,
                reason TEXT,
                PRIMARY KEY (date, symbol, strategy)
            )
        ''')

        conn.commit()
        conn.close()

    def normalize_symbol(self, symbol: str) -> Tuple[str, bool]:
        """
        銘柄コードを正規化

        Args:
            symbol: 入力された銘柄コード

        Returns:
            Tuple[正規化された銘柄コード, 有効かどうか]
        """
        # 数字のみの場合は4桁チェック
        if symbol.isdigit():
            if len(symbol) == 4:
                return f"{symbol}.T", True
            else:
                return symbol, False

        # 4桁.T形式のチェック
        pattern = r'^(\d{4})\.T$'
        match = re.match(pattern, symbol)
        if match:
            return symbol, True

        return symbol, False

    def get_company_name(self, symbol: str) -> Optional[str]:
        """
        yfinanceから企業名を取得

        Args:
            symbol: 銘柄コード

        Returns:
            企業名（取得できない場合はNone）
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # 優先順位で企業名を取得
            company_name = (
                info.get('longName') or
                info.get('shortName') or
                info.get('name')
            )

            return company_name
        except Exception as e:
            print(f"企業名取得エラー: {e}")
            return None

    def list_watchlist(self):
        """監視銘柄リストを表示"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, symbol, name FROM watchlist ORDER BY id
        ''')

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("監視銘柄リストは空です")
            return

        print("\n監視銘柄リスト:")
        print("-" * 60)
        for row in rows:
            print(f"ID: {row[0]}, 銘柄コード: {row[1]}, 企業名: {row[2]}")
        print("-" * 60)
        print(f"合計: {len(rows)}件")

    def add_symbol(self, symbol: str):
        """銘柄を監視リストに追加"""
        # 銘柄コードの正規化と検証
        normalized_symbol, is_valid = self.normalize_symbol(symbol)
        if not is_valid:
            print(f"エラー: 銘柄コード '{symbol}' は無効な形式です")
            print("有効な形式: 4桁の数字（例: 9984）または 4桁.T（例: 9984.T）")
            return

        # 既存チェック
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM watchlist WHERE symbol = ?', (normalized_symbol,))
        existing = cursor.fetchone()
        conn.close()

        if existing:
            print(f"銘柄 '{normalized_symbol}' は既に登録済みです（企業名: {existing[0]}）")
            return

        # 企業名取得
        company_name = self.get_company_name(normalized_symbol)
        if not company_name:
            company_name = "企業名不明"

        # 確認プロンプト
        print(f"\n追加対象: {company_name} ({normalized_symbol})")
        response = input("この銘柄を監視リストに追加しますか？（Y/N）: ").strip().upper()

        if response == 'Y':
            # 登録実行
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(
                    'INSERT INTO watchlist (symbol, name) VALUES (?, ?)',
                    (normalized_symbol, company_name)
                )
                conn.commit()
                print(f"銘柄 '{normalized_symbol}' を監視リストに追加しました")
            except sqlite3.IntegrityError:
                print(f"エラー: 銘柄 '{normalized_symbol}' は既に登録済みです")
            except Exception as e:
                print(f"データベースエラー: {e}")
            finally:
                conn.close()
        else:
            print("追加をキャンセルしました")

    def remove_symbol(self, symbol: str):
        """銘柄を監視リストから削除"""
        # 銘柄コードの正規化
        normalized_symbol, is_valid = self.normalize_symbol(symbol)
        if not is_valid:
            print(f"エラー: 銘柄コード '{symbol}' は無効な形式です")
            print("有効な形式: 4桁の数字（例: 9984）または 4桁.T（例: 9984.T）")
            return

        # 存在チェック
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM watchlist WHERE symbol = ?', (normalized_symbol,))
        existing = cursor.fetchone()
        conn.close()

        if not existing:
            print(f"銘柄 '{normalized_symbol}' は監視リストに登録されていません")
            return

        # 確認プロンプト
        print(f"\n削除対象: {existing[0]} ({normalized_symbol})")
        response = input("この銘柄を監視リストから削除しますか？（Y/N）: ").strip().upper()

        if response == 'Y':
            # 削除実行
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute('DELETE FROM watchlist WHERE symbol = ?', (normalized_symbol,))
                conn.commit()
                print(f"銘柄 '{normalized_symbol}' を監視リストから削除しました")
            except Exception as e:
                print(f"データベースエラー: {e}")
            finally:
                conn.close()
        else:
            print("削除をキャンセルしました")


class TestCommand:
    """testコマンド実行クラス"""

    def __init__(self, db_path: str = "datas.sqlite"):
        self.db_path = db_path

    def get_watchlist(self) -> List[Tuple[str, str]]:
        """監視銘柄リストを取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT symbol, name FROM watchlist ORDER BY symbol')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_stock_data(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        株価データを取得（直近90営業日分）

        Args:
            symbol: 銘柄コード
            days: 取得日数

        Returns:
            株価データ（取得失敗時はNone）
        """
        try:
            # 終了日を今日、開始日を90営業日前に設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)  # 休日を考慮して多めに取得

            data = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                auto_adjust=False,
                threads=False
            )

            if data.empty:
                print(f"データが取得できませんでした: {symbol}")
                return None

            # MultiIndexの列を単純な列名に変換
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)  # ティッカー名のレベルを削除

            # 列名を小文字に正規化
            data.columns = [c.lower() for c in data.columns]

            # 'adj close'列が存在する場合は削除
            if 'adj close' in data.columns:
                data = data.drop('adj close', axis=1)

            # 欠損日を除外し、直近90行を取得
            data = data.dropna()
            if len(data) > days:
                data = data.tail(days)

            print(f"データ取得完了: {symbol} - {len(data)}日分")
            return data

        except Exception as e:
            print(f"データ取得エラー ({symbol}): {e}")
            return None

    def get_previous_position(self, symbol: str, strategy: str) -> Dict:
        """
        前日のポジション状態を取得

        Args:
            symbol: 銘柄コード
            strategy: 戦略名

        Returns:
            前日のポジション状態
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT position, size, entry_price, pnl, reason
            FROM positions
            WHERE symbol = ? AND strategy = ?
            ORDER BY date DESC
            LIMIT 1
        ''', (symbol, strategy))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'position': row[0],
                'size': row[1],
                'entry_price': row[2],
                'pnl': row[3],
                'reason': row[4]
            }
        else:
            # 前回記録がない場合は初期状態
            return {
                'position': 'FLAT',
                'size': 0,
                'entry_price': None,
                'pnl': 0,
                'reason': '初期状態'
            }

    def load_strategy(self, strategy_name: str):
        """
        戦略を読み込む

        Args:
            strategy_name: 戦略名

        Returns:
            戦略インスタンス（失敗時はNone）
        """
        try:
            strategy_path = f"strategies/{strategy_name}.py"
            if not os.path.exists(strategy_path):
                print(f"戦略ファイルが見つかりません: {strategy_path}")
                return None

            spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            return strategy_module.Strategy()

        except Exception as e:
            print(f"戦略読み込みエラー ({strategy_name}): {e}")
            return None

    def interpret_signal(self, signal: str, current_position: str) -> Tuple[str, str, bool]:
        """
        シグナルを解釈して実行判定

        Args:
            signal: 戦略からのシグナル（BUY/SELL/HOLD）
            current_position: 現在のポジション状態

        Returns:
            Tuple[新しいポジション状態, エントリー種別, 取引実行有無]
        """
        if signal == "BUY":
            if current_position == "FLAT":
                return "LONG", "買い", True
            elif current_position == "SHORT":
                return "FLAT", "買い", True
            else:  # LONG
                return "LONG", "なし", False

        elif signal == "SELL":
            if current_position == "FLAT":
                return "SHORT", "売り", True
            elif current_position == "LONG":
                return "FLAT", "売り", True
            else:  # SHORT
                return "SHORT", "なし", False

        else:  # HOLD
            return current_position, "なし", False

    def calculate_pnl(self, position: str, entry_price: float, current_price: float, size: int) -> int:
        """
        損益を計算

        Args:
            position: ポジション状態
            entry_price: 建値
            current_price: 現在価格
            size: 保有株数

        Returns:
            損益（整数）
        """
        if position == "LONG" and entry_price is not None:
            return int((current_price - entry_price) * size)
        elif position == "SHORT" and entry_price is not None:
            return int((entry_price - current_price) * size)
        else:
            return 0

    def format_notification_message(self, results: List[Dict], date: str, is_test: bool = True) -> str:
        """
        LINE通知用メッセージを整形

        Args:
            results: 各銘柄の結果
            date: 日付
            is_test: テスト通知かどうか

        Returns:
            通知メッセージ
        """
        if is_test:
            message = f"【テスト通知 {date}】\n\n"
        else:
            message = f"【監視銘柄レポート {date}】\n\n"

        for result in results:
            symbol = result['symbol']
            company_name = result['company_name']
            close_price = result['close_price']
            signal = result['signal']
            entry_type = result['entry_type']
            position = result['position']
            pnl = result['pnl']
            reason = result['reason']

            # トレンドマッピング
            trend_map = {'BUY': '上昇', 'SELL': '下降', 'HOLD': 'なし'}
            trend = trend_map.get(signal, 'なし')

            # ポジション表示
            position_display = "FLAT"
            if position == "LONG":
                position_display = "100株ロング"
            elif position == "SHORT":
                position_display = "100株ショート"

            # 損益表示（カンマ区切り）
            pnl_display = f"{pnl:+,}" if pnl != 0 else "0"

            # 銘柄セクション
            message += f"{company_name} ({symbol})\n"
            message += f"終値: {close_price:,}円\n"
            message += f"トレンド: {trend}\n"
            message += f"状況: {reason}\n"
            message += f"エントリー: {entry_type}\n"
            message += f"ポジション: {position_display}\n"
            message += f"損益: {pnl_display}\n\n"

        return message

    def write_test_log(self, results: List[Dict], args, log_file: str):
        """
        テストログファイルを出力

        Args:
            results: 各銘柄の結果
            args: コマンドライン引数
            log_file: ログファイル名
        """
        current_date = datetime.now().strftime('%Y-%m-%d')

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== テスト実行レポート ({current_date}) ===\n")
            f.write(f"戦略: {args.strategy}\n")
            f.write(f"銘柄数: {len(results)}\n")
            if args.symbol:
                f.write(f"対象銘柄: {args.symbol}\n")
            if args.date:
                f.write(f"対象日付: {args.date}\n")
            if args.limit:
                f.write(f"銘柄制限: {args.limit}\n")
            f.write("-" * 50 + "\n\n")

            message = self.format_notification_message(results, current_date, is_test=True)
            f.write(message)

    def execute(self, args):
        """
        testコマンドを実行

        Args:
            args: コマンドライン引数
        """
        print(f"テスト実行開始: 戦略={args.strategy}")

        # 対象日付の決定
        if args.date:
            current_date = args.date
        else:
            current_date = datetime.now().strftime('%Y-%m-%d')

        # 監視銘柄リストを取得
        if args.symbol:
            # 単一銘柄指定の場合
            watchlist = [(args.symbol, "テスト銘柄")]
        else:
            # 全銘柄の場合
            watchlist = self.get_watchlist()
            if not watchlist:
                print("監視対象銘柄がありません")
                return

            if args.limit:
                watchlist = watchlist[:args.limit]
                print(f"処理銘柄数を制限: {args.limit}件")

        print(f"監視銘柄数: {len(watchlist)}件")

        # 戦略を読み込み
        strategy = self.load_strategy(args.strategy)
        if not strategy:
            print(f"戦略 '{args.strategy}' の読み込みに失敗しました")
            return

        results = []
        errors = []

        # 銘柄ごとに処理
        for symbol, company_name in watchlist:
            try:
                print(f"処理中: {company_name} ({symbol})")

                # 株価データを取得
                data = self.get_stock_data(symbol)
                if data is None or data.empty:
                    errors.append(f"{symbol}: データ取得失敗")
                    continue

                # 当日のデータを取得（最後の行）
                if len(data) == 0:
                    errors.append(f"{symbol}: データが空")
                    continue

                current_date_idx = data.index[-1]
                current_close = data.iloc[-1]['close']

                # 前日のポジション状態を取得
                prev_position = self.get_previous_position(symbol, args.strategy)
                ctx = {
                    'position': prev_position['position'],
                    'entry_price': prev_position['entry_price'],
                    'size': prev_position['size']
                }

                # 戦略からシグナルを取得
                signal_result = strategy.calculate_signal(data, current_date_idx, ctx)
                signal = signal_result.get('signal', 'HOLD')
                reason = signal_result.get('reason', 'no_signal')

                # シグナル解釈
                new_position, entry_type, trade_executed = self.interpret_signal(
                    signal, prev_position['position']
                )

                # ポジション状態の更新
                size = 100  # 常に100株
                entry_price = prev_position['entry_price']

                # 新規建ての場合
                if trade_executed and new_position != "FLAT":
                    entry_price = current_close
                # 決済の場合
                elif trade_executed and new_position == "FLAT":
                    entry_price = None

                # 損益計算
                if trade_executed and new_position == "FLAT":
                    # 決済日は確定損益
                    pnl = self.calculate_pnl(
                        prev_position['position'],
                        prev_position['entry_price'],
                        current_close,
                        size
                    )
                else:
                    # 保有中は評価損益
                    pnl = self.calculate_pnl(
                        new_position,
                        entry_price,
                        current_close,
                        size
                    )

                # 結果を記録
                results.append({
                    'symbol': symbol,
                    'company_name': company_name,
                    'close_price': int(current_close),
                    'signal': signal,
                    'entry_type': entry_type,
                    'position': new_position,
                    'pnl': pnl,
                    'reason': reason
                })

                print(f"処理完了: {symbol} - {signal} -> {new_position}")

            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                print(f"エラー: {error_msg}")
                errors.append(error_msg)

        # 通知メッセージを生成
        if results:
            message = self.format_notification_message(results, current_date, is_test=True)

            # エラーがある場合は末尾に追加
            if errors:
                message += "\n--- エラー ---\n"
                for error in errors:
                    message += f"{error}\n"

            # 標準出力に表示
            print("\n" + message)

            # ログファイル出力
            if args.symbol:
                log_file = f"{current_date}_test_{args.symbol.replace('.', '_')}.log"
            else:
                log_file = f"{current_date}_test.log"

            self.write_test_log(results, args, log_file)
            print(f"ログファイルを保存: {log_file}")

            # --notifyオプションが指定された場合のみ通知を送信
            if hasattr(args, 'notify') and args.notify:
                try:
                    from utils.notify import send_notification
                    success = send_notification(message)
                    if success:
                        print("通知を送信しました")
                    else:
                        print("通知送信に失敗しました")
                except Exception as e:
                    print(f"通知送信エラー: {e}")

        print(f"テスト実行完了: 処理銘柄 {len(results)}件, エラー {len(errors)}件")

    def list_strategies(self):
        """利用可能な戦略リストを表示"""
        strategies_dir = "strategies"
        if not os.path.exists(strategies_dir):
            print(f"戦略ディレクトリ '{strategies_dir}' が見つかりません")
            return

        strategy_files = []
        for file in os.listdir(strategies_dir):
            if file.endswith('.py') and file != '__init__.py':
                strategy_name = file[:-3]  # .pyを除去
                strategy_files.append(strategy_name)

        if not strategy_files:
            print("利用可能な戦略がありません")
            return

        print("\n利用可能な戦略リスト:")
        print("-" * 40)
        for i, strategy in enumerate(strategy_files, 1):
            print(f"{i}. {strategy}")
        print("-" * 40)
        print(f"合計: {len(strategy_files)}件")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='監視銘柄リスト管理ユーティリティ')
    subparsers = parser.add_subparsers(dest='command', help='サブコマンド')

    # listコマンド
    subparsers.add_parser('list', help='監視銘柄リストを表示')

    # addコマンド
    add_parser = subparsers.add_parser('add', help='銘柄を監視リストに追加')
    add_parser.add_argument('symbol', help='銘柄コード（例: 9984 または 9984.T）')

    # removeコマンド
    remove_parser = subparsers.add_parser('remove', help='銘柄を監視リストから削除')
    remove_parser.add_argument('symbol', help='銘柄コード（例: 9984 または 9984.T）')

    # testコマンド
    test_parser = subparsers.add_parser('test', help='シミュレーション実行')
    test_parser.add_argument('--strategy', default='takazawa', help='戦略名')
    test_parser.add_argument('--symbol', help='単一銘柄指定')
    test_parser.add_argument('--date', help='対象日付（YYYY-MM-DD）')
    test_parser.add_argument('--limit', type=int, help='処理銘柄数上限')
    test_parser.add_argument('--notify', action='store_true', help='通知送信（--notify指定時のみ）')

    # test strategy listコマンド
    test_subparsers = test_parser.add_subparsers(dest='test_subcommand', help='testサブコマンド')
    test_subparsers.add_parser('strategy-list', help='利用可能な戦略リストを表示')

    args = parser.parse_args()

    manager = WatchlistManager()

    if args.command == 'list':
        manager.list_watchlist()
    elif args.command == 'add':
        manager.add_symbol(args.symbol)
    elif args.command == 'remove':
        manager.remove_symbol(args.symbol)
    elif args.command == 'test':
        test_command = TestCommand()
        if hasattr(args, 'test_subcommand') and args.test_subcommand == 'strategy-list':
            test_command.list_strategies()
        else:
            test_command.execute(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
