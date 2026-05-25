"""Network profile defaults for cron job wrapper."""

PROXY_URL = "http://127.0.0.1:10808"
PROXY_PORT = 10808
PROXY_START_CMD = ["zsh", "-ic", "ssproxy"]
LLM_NETWORK = "domestic"
PUSH_NETWORK = "domestic"
