# CLIコマンド設計書（test/history/report）

## 1. 全体設計方針

### 基本原則
- 既存の `backtest.py`・`strategies/` は改変禁止
- `manage.py` にサブコマンドを追加
- DB書き込み・通知送信は一切行わない（シミュレーション専用）
- ログファイルは毎回上書き（追記なし）

### 共通機能
- ログファイル命名規則：`YYYY-MM-DD_{コマンド種別}_{symbol}.log`
- 標準出力と同じ内容をログファイルに保存
- ヘッダー付与で実行条件を明示

---

## 2. test コマンド設計

### 目的
daily_watch.pyの処理フローをシミュレーション実行し、結果を標準出力＋ログファイルに保存

### コマンド形式
```bash
python manage.py test [--strategy <name>] [--symbol <code>] [--date YYYY-MM-DD] [--limit N] [--dry-run]
```

### 引数仕様
- `--strategy`：戦略カセット指定（デフォルト: takazawa）
- `--symbol`：単一銘柄のみ対象（未指定時は全銘柄）
- `--date`：シミュレーション日付上書き（未指定時はシステム日付）
- `--limit`：処理銘柄数上限（全件対象時のみ有効）
- `--dry-run`：明示指定可能だが、testコマンドは常にDB/通知なし

### 実行フロー
1. **銘柄リスト取得**
   - `--symbol` 指定あり：その1件のみ
   - `--symbol` 指定なし：watchlist全件
   - `--limit` 指定あり：先頭N件に制限

2. **株価データ取得**
   - 当日を含む直近90日の日足データをyfinanceから取得
   - daily_watch.pyの`get_stock_data()`メソッドを再利用

3. **ポジション状態復元**
   - positionsテーブルから前回のポジション状態を取得
   - 存在しなければFLAT状態で初期化

4. **戦略実行**
   - strategies/内の戦略カセットをそのまま呼び出し
   - `calculate_signal()`メソッドでシグナル判定

5. **シグナル解釈**
   - daily_watch.pyの`interpret_signal()`メソッドを再利用
   - BUY/SELL/HOLDの解釈ルールを同一適用

6. **損益計算**
   - 当日の評価損益を計算
   - `calculate_pnl()`メソッドを再利用

7. **結果出力**
   - 標準出力：daily_watch.pyと同じ通知メッセージ形式
   - ログファイル：ヘッダー付きで保存

### 出力フォーマット

#### 標準出力
```
【監視銘柄レポート 2025-10-02】
ソフトバンクグループ (9984.T)
終値: 8,500円
トレンド: 上昇
状況: BB:BUY+RCI:BUY
エントリー: 買い
ポジション: 100株ロング
損益: +15,000
```

#### ログファイル
- 複数銘柄対象: `2025-10-02_test.log`
- 単一銘柄対象: `2025-10-02_test_9984.T.log`

```
=== テスト実行レポート (2025-10-02) ===
戦略: takazawa
銘柄数: 3
--------------------------------------------------
【監視銘柄レポート 2025-10-02】
...
```

---

## 3. history コマンド設計

### 目的
指定銘柄の日次スナップショット履歴を表示・保存

### コマンド形式
```bash
python manage.py history <symbol> [--strategy <name>] [--limit N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
```

### 引数仕様
- `<symbol>`：必須、対象銘柄コード
- `--strategy`：戦略指定（デフォルト: takazawa）
- `--limit`：表示件数上限
- `--start`：期間開始日
- `--end`：期間終了日

### 表示項目
```
date | position | size | entry_price | pnl | reason | action
```

### action判定ロジック
- 新規建て：前日position=FLAT、当日position=LONG/SHORT
- 決済：前日position=LONG/SHORT、当日position=FLAT
- 継続：前日と当日のpositionが同じ

### 出力フォーマット

#### 標準出力
```
date       | position | size | entry_price | pnl    | reason           | action
2025-10-01 | FLAT     | 0    | -           | 0      | 初期状態         | -
2025-10-02 | LONG     | 100  | 8,500       | +15,000| BB:BUY+RCI:BUY   | 新規建て
2025-10-03 | LONG     | 100  | 8,500       | +12,000| シグナルなし     | 継続
```

#### ログファイル
`2025-10-02_history_9984.T.log`

```
=== 銘柄履歴 (2025-10-02) ===
銘柄: 9984.T (ソフトバンクグループ)
戦略: takazawa
--------------------------------------------------
date       | position | size | entry_price | pnl    | reason           | action
...
```

---

## 4. report コマンド設計

### 目的
指定期間の成績サマリを集計（全銘柄合算＋任意で銘柄別）

### コマンド形式
```bash
python manage.py report [--period <7d|30d|90d|ytd|all>] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--strategy <name>] [--equity <int>] [--detail by-symbol]
```

### 引数仕様
- `--period`：期間指定（7d, 30d, 90d, ytd, all）
- `--start/--end`：カスタム期間指定
- `--strategy`：戦略指定（デフォルト: takazawa）
- `--equity`：初期資金（デフォルト: 1,000,000円）
- `--detail`：銘柄別詳細表示（by-symbol指定時）

### 集計項目
- **総確定損益**：決済済みの損益合計
- **評価損益**：期間末時点の保有ポジション評価損益
- **取引回数**：新規建て＋決済の合計回数
- **勝率**：利益確定した取引の割合
- **平均損益**：取引1回あたりの平均損益
- **最大ドローダウン**：期間中の最大評価損益下落幅

### 出力フォーマット

#### 標準出力（全体サマリ）
```
=== 集計レポート (2025-10-02) ===
期間: 2025-09-01 ～ 2025-09-30
戦略: takazawa
初期資金: 1,000,000円

【全体成績】
総確定損益: +125,000円
評価損益: +45,000円
取引回数: 15回
勝率: 60.0%
平均損益: +8,333円
最大ドローダウン: -25,000円
```

#### 銘柄別詳細（--detail by-symbol指定時）
```
【銘柄別成績】
9984.T (ソフトバンクグループ)
  確定損益: +35,000円, 取引回数: 3回, 勝率: 66.7%

7203.T (トヨタ自動車)
  確定損益: +90,000円, 取引回数: 5回, 勝率: 80.0%
```

#### ログファイル
`2025-10-02_report.log`

---

## 5. 実装クラス設計

### TestCommandクラス
```python
class TestCommand:
    def __init__(self, db_path="datas.sqlite"):
        self.db_path = db_path
        self.watch = DailyWatch(db_path)  # daily_watch.pyの機能再利用

    def execute(self, args):
        # testコマンドのメイン処理
        pass

    def _simulate_daily_watch(self, symbol, strategy, date):
        # daily_watch.pyの処理をシミュレーション
        pass

    def _write_test_log(self, results, args):
        # テストログファイル出力
        pass
```

### HistoryCommandクラス
```python
class HistoryCommand:
    def __init__(self, db_path="datas.sqlite"):
        self.db_path = db_path

    def execute(self, args):
        # historyコマンドのメイン処理
        pass

    def _get_position_history(self, symbol, strategy, start_date, end_date):
        # ポジション履歴取得
        pass

    def _calculate_action(self, prev_position, current_position):
        # アクション判定
        pass

    def _format_history_output(self, history_data):
        # 履歴出力整形
        pass
```

### ReportCommandクラス
```python
class ReportCommand:
    def __init__(self, db_path="datas.sqlite"):
        self.db_path = db_path

    def execute(self, args):
        # reportコマンドのメイン処理
        pass

    def _calculate_performance_metrics(self, positions_data, equity):
        # 成績指標計算
        pass

    def _calculate_max_drawdown(self, daily_equity):
        # 最大ドローダウン計算
        pass

    def _generate_summary_report(self, metrics, args):
        # サマリレポート生成
        pass
```

### 共通ユーティリティ
```python
class CommandUtils:
    @staticmethod
    def setup_logging(log_file):
        # ログ設定
        pass

    @staticmethod
    def write_log_file(content, filename):
        # ログファイル書き込み（上書きモード）
        pass

    @staticmethod
    def format_currency(value):
        # 通貨フォーマット
        pass

    @staticmethod
    def parse_date_range(args):
        # 期間解析
        pass
```

---

## 6. manage.py拡張設計

### サブコマンド追加
```python
def main():
    parser = argparse.ArgumentParser(description='監視銘柄リスト管理ユーティリティ')
    subparsers = parser.add_subparsers(dest='command', help='サブコマンド')

    # 既存コマンド
    subparsers.add_parser('list', help='監視銘柄リストを表示')
    add_parser = subparsers.add_parser('add', help='銘柄を監視リストに追加')
    add_parser.add_argument('symbol', help='銘柄コード')
    remove_parser = subparsers.add_parser('remove', help='銘柄を監視リストから削除')
    remove_parser.add_argument('symbol', help='銘柄コード')

    # 新規コマンド：test
    test_parser = subparsers.add_parser('test', help='シミュレーション実行')
    test_parser.add_argument('--strategy', default='takazawa', help='戦略名')
    test_parser.add_argument('--symbol', help='単一銘柄指定')
    test_parser.add_argument('--date', help='対象日付（YYYY-MM-DD）')
    test_parser.add_argument('--limit', type=int, help='処理銘柄数上限')
    test_parser.add_argument('--dry-run', action='store_true', help='ドライランモード')

    # 新規コマンド：history
    history_parser = subparsers.add_parser('history', help='銘柄履歴表示')
    history_parser.add_argument('symbol', help='銘柄コード')
    history_parser.add_argument('--strategy', default='takazawa', help='戦略名')
    history_parser.add_argument('--limit', type=int, help='表示件数上限')
    history_parser.add_argument('--start', help='期間開始日')
    history_parser.add_argument('--end', help='期間終了日')

    # 新規コマンド：report
    report_parser = subparsers.add_parser('report', help='成績レポート生成')
    report_parser.add_argument('--period', choices=['7d', '30d', '90d', 'ytd', 'all'],
                              help='期間指定')
    report_parser.add_argument('--start', help='期間開始日')
    report_parser.add_argument('--end', help='期間終了日')
    report_parser.add_argument('--strategy', default='takazawa', help='戦略名')
    report_parser.add_argument('--equity', type=int, default=1000000, help='初期資金')
    report_parser.add_argument('--detail', choices=['by-symbol'], help='詳細表示')
```

---

## 7. データフロー設計

### testコマンドデータフロー
```
watchlist取得 → 株価データ取得 → 前回ポジション取得 → 戦略実行 →
シグナル解釈 → 損益計算 → 結果出力（標準出力＋ログファイル）
```

### historyコマンドデータフロー
```
positionsテーブル検索 → 日付昇順ソート → アクション判定 →
表形式整形 → 出力（標準出力＋ログファイル）
```

### reportコマンドデータフロー
```
全銘柄positions取得 → 期間フィルタリング → 成績指標計算 →
最大ドローダウン計算 → レポート生成 → 出力（標準出力＋ログファイル）
```

---

## 8. エラーハンドリング設計

### 共通エラーケース
- データベース接続エラー
- 戦略ファイル読み込みエラー
- 株価データ取得エラー
- 無効な日付フォーマット
- 存在しない銘柄コード

### エラー対応方針
- エラーメッセージを標準エラー出力
- ログファイルにもエラー内容を記録
- 致命的なエラーのみ終了、その他は継続処理
- ユーザーフレンドリーなエラーメッセージ

---

この設計書に基づいて、既存のmanage.pyに3つの新規サブコマンドを安全に追加できます。既存のbacktest.py・strategies/ディレクトリは一切変更せず、daily_watch.pyの機能を再利用することで効率的な実装が可能です。
