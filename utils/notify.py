import requests
import sys
from secret_config import NOTIFY_SECRET


def send_notification(message: str) -> bool:
    """
    通知を送信する共通関数

    Args:
        message (str): 送信メッセージ

    Returns:
        bool: 成功時 True、失敗時 False
    """
    try:
        # APIエンドポイント
        url = "https://api.9diz.com/api.php"

        # リクエストデータ
        data = {
            "secret": NOTIFY_SECRET,
            "message": message
        }

        # POSTリクエスト送信
        response = requests.post(url, data=data)

        # HTTP 200 なら成功
        if response.status_code == 200:
            return True
        else:
            print(f"通知送信エラー: HTTP {response.status_code}", file=sys.stderr)
            return False

    except Exception as e:
        print(f"通知送信例外: {e}", file=sys.stderr)
        return False
