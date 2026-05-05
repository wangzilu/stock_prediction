import pytest
from unittest.mock import patch, MagicMock
from push.wechat import WeChatPusher


def test_format_markdown_message():
    """Should format text as WeChat markdown message payload."""
    pusher = WeChatPusher(webhook_url="https://example.com/hook")
    payload = pusher._build_payload("测试消息\n第二行")
    assert payload["msgtype"] == "markdown"
    assert payload["markdown"]["content"] == "测试消息\n第二行"


@patch("push.wechat.requests.post")
def test_send_success(mock_post):
    """Successful push should return True."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"errcode": 0, "errmsg": "ok"}
    mock_post.return_value = mock_resp

    pusher = WeChatPusher(webhook_url="https://example.com/hook")
    result = pusher.send("测试消息")
    assert result is True
    mock_post.assert_called_once()


@patch("push.wechat.requests.post")
def test_send_failure_returns_false(mock_post):
    """Failed push should return False."""
    mock_post.side_effect = Exception("Network error")

    pusher = WeChatPusher(webhook_url="https://example.com/hook")
    result = pusher.send("测试消息")
    assert result is False


def test_empty_webhook_url_raises():
    """Empty webhook URL should raise ValueError."""
    with pytest.raises(ValueError):
        WeChatPusher(webhook_url="")
