import os
import requests
from config.settings import PUSHPLUS_TOKEN


class WeChatPusher:
    """Pushes messages to WeChat via pushplus.plus service."""

    def __init__(self, token: str = None):
        if token is not None:
            self.token = token
        else:
            self.token = os.environ.get("PUSHPLUS_TOKEN", PUSHPLUS_TOKEN)
        if not self.token:
            raise ValueError(
                "Pushplus token is required. "
                "Set PUSHPLUS_TOKEN environment variable or pass token."
            )
        self.url = "http://www.pushplus.plus/send"

    def _build_payload(self, content: str, title: str = "股票信号") -> dict:
        """Build pushplus message payload."""
        return {
            "token": self.token,
            "title": title,
            "content": content,
            "template": "txt",
        }

    def send(self, content: str, title: str = "股票信号") -> bool:
        """Send a message via pushplus.

        Args:
            content: Message text
            title: Message title

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            payload = self._build_payload(content, title)
            resp = requests.post(self.url, json=payload, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                return data.get("code", -1) == 200

            return False

        except Exception:
            return False

    def send_recommendation(self, report: str) -> bool:
        """Send daily recommendation report."""
        return self.send(report, title="📈 今日荐股")

    def send_alert(self, alert_content: str) -> bool:
        """Send risk alert message."""
        return self.send(alert_content, title="⚠️ 风险警示")

    def send_verification(self, verification_report: str) -> bool:
        """Send 5-day verification report."""
        return self.send(verification_report, title="📋 荐股印证")
