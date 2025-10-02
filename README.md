# 株の仮想売買バックテストシステム

yfinanceを使用して株価データを取得し、指定した戦略に基づいて仮想売買を行うバックテストシステムです。

## 特徴

- yfinanceから株価データ（日足）を自動取得
- SQLiteによるデータキャッシュ機能
- 戦略カセット方式で様々な売買アルゴリズムを実装可能
- 売買履歴とパフォーマンス統計を自動出力

## インストール

```bash
# 必要なパッケージをインストール
pip install -r requirements.txt
```

## 使用方法

### 基本的な実行方法

```bash
python backtest.py --strategy takazawa --symbol 9984.T --start 2024-01-01 --end 2024-12-31
```

### 引数の説明

- `--strategy`: 戦略名（strategiesフォルダ内のファイル名）
- `--symbol`: 銘柄コード（例: 9984.T, 7203.T, 6758.T）
- `--start`: 開始日（YYYY-MM-DD形式）
- `--end`: 終了日（YYYY-MM-DD形式）

### 出力ファイル

- `performance.txt`: バックテスト結果のサマリー
- `datas.sqlite`: 株価データのキャッシュデータベース

## 戦略の追加方法

`strategies`フォルダに新しい戦略ファイルを作成します。

### 戦略ファイルのテンプレート

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

class Strategy:
    def __init__(self):
        # パラメータの初期化
        pass

    def calculate_signal(self, data: pd.DataFrame, current_date: pd.Timestamp) -> str:
        """
        売買シグナルを計算

        Args:
            data: 株価データ（DataFrame）
            current_date: 現在の日付

        Returns:
            str: 'BUY', 'SELL', 'HOLD'
        """
        # ここに売買ロジックを実装
        return 'HOLD'
```

## 利用可能な戦略

### takazawa（高澤式）
- ボリンジャーバンドとRCI（順位相関指数）を使用
- 両方の指標が一致した場合のみ売買シグナルを出力

## 出力例

```
バックテスト開始: takazawa - 9984.T
期間: 2024-01-01 〜 2024-12-31
yfinanceからデータを取得中: 9984.T (2024-01-01 〜 2024-12-31)
データ取得完了: 245日分
2024-01-15 BUY 価格=1450 数量=100 損益=0
2024-02-10 SELL 価格=1520 損益=+70
...

=== バックテスト結果 ===
total_trades: 12
winning_trades: 8
losing_trades: 4
win_rate: 66.7%
total_pnl: +320
final_cash: 1032000
return_rate: 3.2%
```

## 注意事項

- このシステムは教育・研究目的のためのものです
- 実際の投資には使用しないでください
- 株価データはyfinanceから取得するため、利用規約に従ってください
- データの正確性は保証されません
