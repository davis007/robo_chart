#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日次監視実行スクリプト
"""

import sqlite3
import yfinance as yf
import pandas as pd
import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import importlib.util


class DailyWatch:
    """日次監視実行クラス"""

    def __init__(self, db_path: str = "datas.sqlite"):
        self.db_path = db_path
        self.setup_logging()

    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('daily_watch.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

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
                self.logger.warning(f"データが取得できませんでした: {symbol}")
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

            self.logger.info(f"データ取得完了: {symbol} - {len(data)}日分")
            return data

        except Exception as e:
            self.logger.error(f"データ取得エラー ({symbol}): {e}")
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
                self.logger.error(f"戦略ファイルが見つかりません: {strategy_path}")
                return None

            spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            return strategy_module.Strategy()

        except Exception as e:
            self.logger.error(f"戦略読み込みエラー ({strategy_name}): {e}")
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

    def save_position(self, date: str, symbol: str, strategy: str, position: str,
                     size: int, entry_price: Optional[float], pnl: int, reason: str):
        """
        ポジション状態を保存

        Args:
            date: 日付
            symbol: 銘柄コード
            strategy: 戦略名
            position: ポジション状態
            size: 保有株数
            entry_price: 建値
            pnl: 損益
            reason: 理由
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO positions
                (date, symbol, strategy, position, size, entry_price, pnl, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, symbol, strategy, position, size, entry_price, pnl, reason))
            conn.commit()
            self.logger.info(f"ポジション保存: {symbol} - {position}")
        except Exception as e:
            self.logger.error(f"ポジション保存エラー ({symbol}): {e}")
        finally:
            conn.close()

    def format_notification_message(self, results: List[Dict], date: str) -> str:
        """
        LINE通知用メッセージを整形

        Args:
            results: 各銘柄の結果
            date: 日付

        Returns:
            通知メッセージ
        """
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

    def send_line_notify(self, message: str, dry_run: bool = False):
        """
        LINE通知を送信

        Args:
            message: 送信メッセージ
            dry_run: ドライランモード
        """
        if dry_run:
            self.logger.info("ドライラン: LINE通知は送信されません")
            self.logger.info(f"通知メッセージ:\n{message}")
            return

        # LINEトークンは環境変数から取得
        line_token = os.getenv('LINE_NOTIFY_TOKEN')
        if not line_token:
            self.logger.warning("LINE_NOTIFY_TOKEN環境変数が設定されていません")
            return

        try:
            import requests
            headers = {'Authorization': f'Bearer {line_token}'}
            data = {'message': message}
            response = requests.post('https://notify-api.line.me/api/notify',
                                   headers=headers, data=data)
            if response.status_code == 200:
                self.logger.info("LINE通知を送信しました")
            else:
                self.logger.error(f"LINE通知送信エラー: {response.status_code}")
        except Exception as e:
            self.logger.error(f"LINE通知送信エラー: {e}")

    def run(self, strategy_name: str = "takazawa", dry_run: bool = False,
           target_date: Optional[str] = None, limit: Optional[int] = None):
        """
        日次監視を実行

        Args:
            strategy_name: 戦略名
            dry_run: ドライランモード
            target_date: 対象日付（未指定時は今日）
            limit: 処理銘柄数の上限
        """
        start_time = datetime.now()
        self.logger.info(f"日次監視開始: 戦略={strategy_name}, ドライラン={dry_run}")

        # 対象日付の決定
        if target_date:
            current_date = target_date
        else:
            current_date = datetime.now().strftime('%Y-%m-%d')

        # 監視銘柄リストを取得
        watchlist = self.get_watchlist()
        if not watchlist:
            self.logger.info("監視対象銘柄がありません")
            return

        if limit:
            watchlist = watchlist[:limit]
            self.logger.info(f"処理銘柄数を制限: {limit}件")

        self.logger.info(f"監視銘柄数: {len(watchlist)}件")

        # 戦略を読み込み
        strategy = self.load_strategy(strategy_name)
        if not strategy:
            self.logger.error(f"戦略 '{strategy_name}' の読み込みに失敗しました")
            return

        results = []
        errors = []

        # 銘柄ごとに処理
        for symbol, company_name in watchlist:
            try:
                self.logger.info(f"処理中: {company_name} ({symbol})")

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
                prev_position = self.get_previous_position(symbol, strategy_name)
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

                # ポジション保存（ドライランでなければ）
                if not dry_run:
                    self.save_position(
                        current_date, symbol, strategy_name, new_position,
                        size, entry_price, pnl, reason
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

                self.logger.info(f"処理完了: {symbol} - {signal} -> {new_position}")

            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        # 通知メッセージを生成
        if results:
            message = self.format_notification_message(results, current_date)

            # エラーがある場合は末尾に追加
            if errors:
                message += "\n--- エラー ---\n"
                for error in errors:
                    message += f"{error}\n"

            # LINE通知送信
            self.send_line_notify(message, dry_run)

            # テキストファイルにも保存
            if not dry_run:
                report_file = f"{current_date}_daily_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(message)
                self.logger.info(f"レポートを保存: {report_file}")

        # 実行時間を記録
        execution_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"日次監視完了: {execution_time:.1f}秒, 処理銘柄: {len(results)}件")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='日次監視実行スクリプト')
    parser.add_argument('--strategy', default='takazawa', help='使用する戦略名（デフォルト: takazawa）')
    parser.add_argument('--dry-run', action='store_true', help='ドライランモード（DB書き込み・通知なし）')
    parser.add_argument('--date', help='対象日付（YYYY-MM-DD形式、未指定時は今日）')
    parser.add_argument('--limit', type=int, help='処理銘柄数の上限（テスト用）')

    args = parser.parse_args()

    watch = DailyWatch()
    watch.run(
        strategy_name=args.strategy,
        dry_run=args.dry_run,
        target_date=args.date,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
