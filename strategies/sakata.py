#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒田五法戦略
pandas-taのcandlestick関数と自前実装を使用した酒田五法パターン判定
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, Tuple


class Strategy:
    """酒田五法戦略クラス"""

    def __init__(self):
        """
        初期化
        """
        # 利用可能なpandas-ta candlestick関数（実際に存在するもののみ）
        self.available_patterns = [
            'cdl_doji',           # 十字線
            'cdl_inside',         # インサイドバー（はらみ足に類似）
            'cdl_pattern',        # 一般的なパターン
            'cdl_z',              # Zパターン
            'cdl_engulfing',      # 包み足
            'cdl_harami',         # はらみ足
            'cdl_morningstar',    # 明けの明星
            'cdl_eveningstar',    # 宵の明星
            'cdl_hammer',         # ハンマー
            'cdl_shootingstar',   # 流れ星
            'cdl_3whitesoldiers', # 三兵
            'cdl_3blackcrows',    # 三羽烏
            'cdl_3outside',       # 外側三法
            'cdl_3inside',        # 内側三法
            'cdl_3linestrike',    # 三線押し
            'cdl_abandonedbaby',  # 捨て子
            'cdl_advanceblock',   # 前進三法
            'cdl_belthold',       # ベルトホールド
            'cdl_breakaway',      # ブレイクアウェイ
            'cdl_closingmarubozu', # 終値坊主
            'cdl_counterattack',  # 反撃線
            'cdl_darkcloudcover', # 曇り空
            'cdl_dragonflydoji',  # トンボ
            'cdl_gravestonedoji', # 墓石
            'cdl_hangingman',     # 首吊り
            'cdl_identical3crows', # 三羽烏
            'cdl_inneck',         # 首付き
            'cdl_kicking',        # 蹴り
            'cdl_ladderbottom',   # はしご底
            'cdl_longleggeddoji', # 大引け十字線
            'cdl_longline',       # 長い実体
            'cdl_marubozu',       # 坊主
            'cdl_matchinglow',    # マッチングロウ
            'cdl_mathold',        # マットホールド
            'cdl_morningdojistar', # 明けの明星十字
            'cdl_onneck',         # 首付き
            'cdl_piercing',       # 貫き
            'cdl_rickshawman',    # 人力車
            'cdl_risefall3methods', # 上げ三法・下げ三法
            'cdl_separatinglines', # 分離線
            'cdl_shortline',      # 短い実体
            'cdl_spinningtop',    # コマ
            'cdl_stalledpattern', # 停滞パターン
            'cdl_sticksandwich',  # スティックサンドイッチ
            'cdl_takuri',         # たくり
            'cdl_tasukigap',      # たすき
            'cdl_thrusting',      # 突き上げ
            'cdl_tristar',        # 三ツ星
            'cdl_unique3river',   # ユニーク三川
            'cdl_upsidegap2crows', # 上放れ二羽烏
            'cdl_xsidegap3methods', # ギャップ三法
        ]

        # 酒田五法パターンの優先順位（高い順） - 三空 > 三兵 > 明けの明星/宵の明星 > 包み足 > はらみ足
        self.pattern_priority = [
            # 三空（自前実装）
            'three_gaps',         # 三空
            # 三兵（pandas-ta）
            'cdl_3whitesoldiers', # 三兵（強気）
            'cdl_3blackcrows',    # 三羽烏（弱気）
            # 明けの明星・宵の明星
            'morning_star',       # 明けの明星（自前実装）
            'evening_star',       # 宵の明星（自前実装）
            'cdl_morningstar',    # 明けの明星（pandas-ta）
            'cdl_eveningstar',    # 宵の明星（pandas-ta）
            # 包み足
            'engulfing',          # 包み足（自前実装）
            'cdl_engulfing',      # 包み足（pandas-ta）
            # はらみ足
            'harami',             # はらみ足（自前実装）
            'cdl_harami',         # はらみ足（pandas-ta）
            # その他
            'hammer',             # ハンマー（自前実装）
            'shooting_star',      # 流れ星（自前実装）
            'cdl_hammer',         # ハンマー（pandas-ta）
            'cdl_shootingstar',   # 流れ星（pandas-ta）
            'cdl_doji',           # 十字線
            'cdl_inside',         # インサイドバー
            'cdl_pattern',        # 一般的なパターン
            'cdl_z',              # Zパターン
        ]

    def calculate_signal(
        self,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        ctx: dict
    ) -> dict:
        """
        売買シグナルを計算（酒田五法）

        Args:
            data: 株価データ（当日を含む直近データ）
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

        # 十分なデータがない場合はHOLD（最低3日分必要）
        if current_idx < 2:
            return {'signal': 'HOLD', 'reason': 'データ不足'}

        # カラム名の大文字小文字を考慮してOHLCを取得
        open_col = 'open' if 'open' in data.columns else 'Open'
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'
        close_col = 'close' if 'close' in data.columns else 'Close'

        # 酒田五法パターンを検出
        detected_patterns = self._detect_sakata_patterns(
            data, open_col, high_col, low_col, close_col, current_idx
        )

        # デバッグ: 検出されたパターンを出力
        if detected_patterns:
            print(f"DEBUG: 検出されたパターン: {detected_patterns}")

        # ポジション状態に基づいてシグナル判定
        position = ctx.get("position", "FLAT")
        signal, reason = self._evaluate_signals(detected_patterns, position)

        # デバッグ: 最終的なシグナルを出力
        if signal != 'HOLD':
            print(f"DEBUG: 最終シグナル: {signal}, 理由: {reason}, ポジション: {position}")

        return {'signal': signal, 'reason': reason}

    def _detect_sakata_patterns(
        self,
        data: pd.DataFrame,
        open_col: str,
        high_col: str,
        low_col: str,
        close_col: str,
        current_idx: int
    ) -> Dict[str, int]:
        """
        酒田五法パターンを検出

        Args:
            data: 株価データ
            open_col: 始値カラム名
            high_col: 高値カラム名
            low_col: 安値カラム名
            close_col: 終値カラム名
            current_idx: 現在のインデックス

        Returns:
            Dict: 検出されたパターンとそのシグナル値
        """
        patterns = {}

        # 必要なデータ範囲を取得（直近3日分）
        start_idx = max(0, current_idx - 2)
        pattern_data = data.iloc[start_idx:current_idx+1]

        # pandas-taのcandlestick関数を実行
        detected_count = 0
        for pattern_name in self.available_patterns:
            try:
                # 各パターンの関数を実行
                if hasattr(ta, pattern_name):
                    pattern_func = getattr(ta, pattern_name)

                    # 関数の実行（OHLCデータを渡す）
                    result = pattern_func(
                        open=pattern_data[open_col],
                        high=pattern_data[high_col],
                        low=pattern_data[low_col],
                        close=pattern_data[close_col]
                    )

                    # 結果がSeriesの場合、最後の値を取得
                    if isinstance(result, pd.Series):
                        signal_value = result.iloc[-1] if len(result) > 0 else 0
                    else:
                        signal_value = 0

                    # pandas-taのシグナル値を厳密に解釈
                    # +100 → 強気パターン → BUYシグナル
                    # -100 → 弱気パターン → SELLシグナル
                    # 0 → パターンなし → HOLD
                    if signal_value == 100:
                        patterns[pattern_name] = 100  # 強気シグナル
                        detected_count += 1
                    elif signal_value == -100:
                        patterns[pattern_name] = -100  # 弱気シグナル
                        detected_count += 1
                    # その他の値は無視（厳密に+100/-100のみを採用）

            except Exception as e:
                # エラーが発生した場合はスキップ
                continue

        # デバッグ: pandas-taの検出結果を出力
        if detected_count > 0:
            print(f"DEBUG: pandas-taで{detected_count}個のパターンを検出")

        # 自前実装の酒田五法パターンを検出
        custom_patterns = self._detect_custom_patterns(
            pattern_data, open_col, high_col, low_col, close_col
        )
        patterns.update(custom_patterns)

        return patterns

    def _detect_custom_patterns(
        self,
        data: pd.DataFrame,
        open_col: str,
        high_col: str,
        low_col: str,
        close_col: str
    ) -> Dict[str, int]:
        """
        自前実装の酒田五法パターンを検出

        Args:
            data: 株価データ（直近3日分）
            open_col: 始値カラム名
            high_col: 高値カラム名
            low_col: 安値カラム名
            close_col: 終値カラム名

        Returns:
            Dict: 検出されたパターンとシグナル値
        """
        patterns = {}

        # データが3日分以上ある場合のみ検出
        if len(data) < 3:
            return patterns

        # 各日のOHLCを取得
        today = data.iloc[-1]
        yesterday = data.iloc[-2]
        day_before = data.iloc[-3]

        today_open = today[open_col]
        today_high = today[high_col]
        today_low = today[low_col]
        today_close = today[close_col]

        yesterday_open = yesterday[open_col]
        yesterday_high = yesterday[high_col]
        yesterday_low = yesterday[low_col]
        yesterday_close = yesterday[close_col]

        day_before_open = day_before[open_col]
        day_before_high = day_before[high_col]
        day_before_low = day_before[low_col]
        day_before_close = day_before[close_col]

        # 包み足（Engulfing Pattern）
        if (yesterday_close < yesterday_open and  # 前日は陰線
            today_close > today_open and          # 当日は陽線
            today_open < yesterday_close and      # 当日始値 < 前日終値
            today_close > yesterday_open):        # 当日終値 > 前日始値
            patterns['engulfing'] = 100  # 強気の包み足

        elif (yesterday_close > yesterday_open and  # 前日は陽線
              today_close < today_open and          # 当日は陰線
              today_open > yesterday_close and      # 当日始値 > 前日終値
              today_close < yesterday_open):        # 当日終値 < 前日始値
            patterns['engulfing'] = -100  # 弱気の包み足

        # はらみ足（Harami Pattern）
        if (yesterday_close > yesterday_open and  # 前日は陽線
            today_close < today_open and          # 当日は陰線
            today_open < yesterday_close and      # 当日始値 < 前日終値
            today_close > yesterday_open and      # 当日終値 > 前日始値
            today_high < yesterday_high and       # 当日高値 < 前日高値
            today_low > yesterday_low):           # 当日安値 > 前日安値
            patterns['harami'] = -100  # 弱気のはらみ足

        elif (yesterday_close < yesterday_open and  # 前日は陰線
              today_close > today_open and          # 当日は陽線
              today_open > yesterday_close and      # 当日始値 > 前日終値
              today_close < yesterday_open and      # 当日終値 < 前日始値
              today_high < yesterday_high and       # 当日高値 < 前日高値
              today_low > yesterday_low):           # 当日安値 > 前日安値
            patterns['harami'] = 100  # 強気のはらみ足

        # 明けの明星（Morning Star）
        if (day_before_close < day_before_open and  # 一昨日は陰線
            yesterday_high < day_before_low and     # 前日は下落ギャップ
            today_close > today_open and            # 当日は陽線
            today_close > (day_before_open + day_before_close) / 2):  # 前日の実体中央以上
            patterns['morning_star'] = 100  # 強気の明けの明星

        # 宵の明星（Evening Star）
        if (day_before_close > day_before_open and  # 一昨日は陽線
            yesterday_low > day_before_high and     # 前日は上昇ギャップ
            today_close < today_open and            # 当日は陰線
            today_close < (day_before_open + day_before_close) / 2):  # 前日の実体中央以下
            patterns['evening_star'] = -100  # 弱気の宵の明星

        # ハンマー（Hammer）
        body = abs(today_close - today_open)
        lower_shadow = min(today_open, today_close) - today_low
        upper_shadow = today_high - max(today_open, today_close)

        if (lower_shadow > 2 * body and  # 下ヒゲが実体の2倍以上
            upper_shadow < body * 0.1 and  # 上ヒゲがほとんどない
            today_close > today_open):      # 陽線
            patterns['hammer'] = 100  # 強気のハンマー

        # 流れ星（Shooting Star）
        if (upper_shadow > 2 * body and  # 上ヒゲが実体の2倍以上
            lower_shadow < body * 0.1 and  # 下ヒゲがほとんどない
            today_close < today_open):      # 陰線
            patterns['shooting_star'] = -100  # 弱気の流れ星

        # 三空（Three Gaps）パターン - 強気の三空
        if (day_before_close < day_before_open and  # 一昨日は陰線
            yesterday_close < yesterday_open and     # 前日は陰線
            today_close > today_open and             # 当日は陽線
            today_open > yesterday_high and          # 当日始値 > 前日高値（ギャップアップ）
            yesterday_open > day_before_high):       # 前日始値 > 一昨日高値（ギャップアップ）
            patterns['three_gaps'] = 100  # 強気の三空

        # 三空（Three Gaps）パターン - 弱気の三空
        elif (day_before_close > day_before_open and  # 一昨日は陽線
              yesterday_close > yesterday_open and     # 前日は陽線
              today_close < today_open and             # 当日は陰線
              today_open < yesterday_low and           # 当日始値 < 前日安値（ギャップダウン）
              yesterday_open < day_before_low):        # 前日始値 < 一昨日安値（ギャップダウン）
            patterns['three_gaps'] = -100  # 弱気の三空

        return patterns

    def _evaluate_signals(
        self,
        patterns: Dict[str, int],
        position: str
    ) -> Tuple[str, str]:
        """
        検出されたパターンからシグナルを評価

        Args:
            patterns: 検出されたパターンとシグナル値
            position: 現在のポジション状態

        Returns:
            Tuple: (signal, reason)
        """
        # デバッグ: 検出された全パターンを出力
        if patterns:
            print(f"DEBUG: 評価対象パターン: {patterns}")

        # 優先順位に従ってパターンを評価
        for pattern_name in self.pattern_priority:
            if pattern_name in patterns:
                signal_value = patterns[pattern_name]

                # シグナル値を厳密に解釈
                if signal_value == 100:
                    signal = self._map_signal_for_position('BUY', position)
                    reason = f"{pattern_name}:BUY"
                    print(f"DEBUG: 採用パターン: {pattern_name}, シグナル値: {signal_value}, 最終シグナル: {signal}")
                    return signal, reason
                elif signal_value == -100:
                    signal = self._map_signal_for_position('SELL', position)
                    reason = f"{pattern_name}:SELL"
                    print(f"DEBUG: 採用パターン: {pattern_name}, シグナル値: {signal_value}, 最終シグナル: {signal}")
                    return signal, reason

        # どのパターンも検出されなかった場合
        print("DEBUG: 採用可能なパターンなし")
        return 'HOLD', 'no_pattern'

    def _map_signal_for_position(self, pattern_signal: str, position: str) -> str:
        """
        ポジション状態に基づいてシグナルをマッピング

        Args:
            pattern_signal: パターンから検出されたシグナル
            position: 現在のポジション状態

        Returns:
            str: 実際の売買シグナル
        """
        # デバッグ: シグナルマッピングを出力
        print(f"DEBUG: シグナルマッピング: パターンシグナル={pattern_signal}, ポジション={position}")

        if position == "FLAT":
            # FLAT状態：BUY→ロングエントリー、SELL→ショートエントリー
            return pattern_signal
        elif position == "LONG":
            # LONG状態：SELL→ロング解消、BUY→HOLD（トレンド継続の場合はHOLD）
            return 'SELL' if pattern_signal == 'SELL' else 'HOLD'
        elif position == "SHORT":
            # SHORT状態：BUY→ショート解消、SELL→HOLD（トレンド継続の場合はHOLD）
            return 'BUY' if pattern_signal == 'BUY' else 'HOLD'
        else:
            return 'HOLD'


# テスト用
if __name__ == "__main__":
    # テストデータの作成（包み足パターンを含む）
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    test_data = pd.DataFrame({
        'Open': [1000, 1010, 1020, 1030, 1040, 1030, 1020, 1010, 1000, 990,
                 980, 970, 960, 950, 940, 950, 960, 970, 980, 990,
                 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090],
        'High': [1010, 1020, 1030, 1040, 1050, 1040, 1030, 1020, 1010, 1000,
                 990, 980, 970, 960, 950, 960, 970, 980, 990, 1000,
                 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100],
        'Low': [990, 1000, 1010, 1020, 1030, 1020, 1010, 1000, 990, 980,
                970, 960, 950, 940, 930, 940, 950, 960, 970, 980,
                990, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080],
        'Close': [1005, 1015, 1025, 1035, 1045, 1035, 1025, 1015, 1005, 995,
                  985, 975, 965, 955, 945, 955, 965, 975, 985, 995,
                  1005, 1015, 1025, 1035, 1045, 1055, 1065, 1075, 1085, 1095],
        'Volume': [1000000] * 30
    }, index=dates)

    strategy = Strategy()

    # 最後の日付でテスト
    last_date = dates[-1]
    ctx = {"position": "FLAT", "entry_price": None, "size": 0}
    signal = strategy.calculate_signal(test_data, last_date, ctx)
    print(f"最終日のシグナル: {signal}")
