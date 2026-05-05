import os
import requests
from config.settings import WECHAT_WEBHOOK_URL


class WeChatPusher:
    """Pushes messages to WeChat Work group via webhook robot."""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get("WECHAT_WEBHOOK_URL", WECHAT_WEBHOOK_URL)
        if not self.webhook_url:
            raise ValueError(
                "WeChat webhook URL is required. "
                "Set WECHAT_WEBHOOK_URL environment variable or pass webhook_url."
            )

    def _build_payload(self, content: str) -> dict:
        """Build WeChat Work markdown message payload."""
        return {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
            },
        }

    def send(self, content: str) -> bool:
        """Send a markdown message to WeChat Work group."""
        try:
            payload = self._build_payload(content)
            resp = requests.post(self.webhook_url, json=payload, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                return data.get("errcode", -1) == 0

            return False

        except Exception:
            return False

    def send_recommendation(self, report: str) -> bool:
        """Send daily recommendation report."""
        return self.send(report)

    def send_alert(self, alert_content: str) -> bool:
        """Send risk alert message."""
        return self.send(alert_content)

    def send_verification(self, verification_report: str) -> bool:
        """Send 5-day verification report."""
        return self.send(verification_report)
