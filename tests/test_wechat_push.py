import pytest
from unittest.mock import patch, MagicMock
from push.wechat import WeChatPusher


def test_format_message_payload():
    """Should format text as pushplus message payload."""
    pusher = WeChatPusher(token="test_token")
    payload = pusher._build_payload("测试消息\n第二行", title="测试")
    assert payload["token"] == "test_token"
    assert payload["title"] == "测试"
    assert payload["content"] == "测试消息\n第二行"
    assert payload["template"] == "txt"


@patch("push.wechat.requests.post")
def test_send_success(mock_post):
    """Successful push should return True."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"code": 200, "msg": "ok"}
    mock_post.return_value = mock_resp

    pusher = WeChatPusher(token="test_token")
    result = pusher.send("测试消息")
    assert result is True
    mock_post.assert_called_once()


@patch("push.wechat.requests.post")
def test_send_failure_returns_false(mock_post):
    """Failed push should return False."""
    mock_post.side_effect = Exception("Network error")

    pusher = WeChatPusher(token="test_token")
    result = pusher.send("测试消息")
    assert result is False


def test_empty_token_raises():
    """Empty token should raise ValueError."""
    with pytest.raises(ValueError):
        WeChatPusher(token="")
