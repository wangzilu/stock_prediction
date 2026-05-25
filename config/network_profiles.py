"""Network profile defaults for cron job wrapper."""

# ShadowsocksX listens on SOCKS5 port 10808.
# shadowsocks-http-auto.js bridges to HTTP port 10818.
# Python requests needs HTTP proxy, not SOCKS5.
PROXY_URL = "http://127.0.0.1:10818"
PROXY_PORT = 10818
PROXY_START_CMD = ["zsh", "-ic", "ssproxy"]
LLM_NETWORK = "domestic"
PUSH_NETWORK = "domestic"
