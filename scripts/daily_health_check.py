"""Daily health check: verify all crontab jobs ran successfully + push summary.

Checks:
  1. Morning recommendation: did it push stocks?
  2. Model training: did it succeed? RankIC improved?
  3. Shadow/Champion paper trading: positions + returns
  4. LLM event pipeline: events collected?
  5. Data updates: Qlib data fresh?

Pushes a single summary message with all statuses.

Usage:
    python scripts/daily_health_check.py              # check today
    python scripts/daily_health_check.py --date 2026-05-22
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
LOGS_DIR = PROJECT_ROOT / "logs"


def check_log_for_today(log_name: str, date: str, success_keywords: list,
                        fail_keywords: list = None,
                        job_id: str = None) -> tuple[str, str]:
    """Check a log file for today's run status.

    Also checks job_status.json as a secondary source (some scripts
    like baostock don't write dated log lines).

    Returns (status, detail) where status is '✅', '⚠️', or '❌'.
    """
    # First check job_status.json (most reliable — written by run_with_status)
    if job_id:
        try:
            import json
            status_path = PROJECT_ROOT / "data" / "storage" / "job_status.json"
            if status_path.exists():
                d = json.loads(status_path.read_text())
                job = d.get("jobs", {}).get(job_id, {})
                started = job.get("started_at", "")
                if date in started:
                    if job.get("status") == "success":
                        duration = job.get("duration_seconds", 0)
                        return "✅", f"成功 ({duration:.0f}s)"
                    elif job.get("status") == "failed":
                        error = job.get("error", "")[:60]
                        return "⚠️", f"运行但有错误: {error}"
                    elif job.get("status") == "running":
                        return "⚠️", "仍在运行中"
        except Exception:
            pass

    log_path = LOGS_DIR / log_name
    if not log_path.exists():
        return "❌", "日志文件不存在"

    # Read last 50 lines
    lines = []
    with open(log_path) as f:
        lines = f.readlines()[-50:]

    today_lines = [l for l in lines if date in l]
    if not today_lines:
        return "❌", "今天没有运行"

    text = "".join(today_lines)

    if fail_keywords:
        for kw in fail_keywords:
            if kw in text:
                return "⚠️", f"运行但有错误: {kw}"

    for kw in success_keywords:
        if kw in text:
            return "✅", kw

    return "⚠️", "运行了但未确认成功"


def check_training(date: str) -> tuple[str, str]:
    """Check if model training succeeded and compare with previous."""
    log_path = LOGS_DIR / "lgb_after_close_train.log"
    if not log_path.exists():
        return "❌", "训练日志不存在"

    lines = open(log_path).readlines()[-30:]
    today_lines = [l for l in lines if date in l]

    if not today_lines:
        return "❌", "今天没训练"

    text = "".join(today_lines)
    if "Training complete" in text and "health passed" in text.lower():
        # Extract prediction count
        for l in today_lines:
            if "prediction_count" in l:
                return "✅", l.strip().split("prediction_count:")[-1].strip()[:30]
        return "✅", "训练成功"
    elif "failed" in text.lower() or "error" in text.lower():
        return "❌", "训练失败"
    return "⚠️", "训练状态不明"


def check_paper_trading(date: str) -> tuple[str, str]:
    """Check paper trading status."""
    state_path = DATA_DIR / "paper" / "oms_state.json"
    if not state_path.exists():
        return "❌", "OMS 状态文件不存在"

    state = json.loads(state_path.read_text())
    value = state.get("total_value", 0)
    n_pos = len(state.get("positions", {}))
    history = state.get("daily_pnl_history", [])

    if history:
        last = history[-1]
        last_date = last.get("date", "")
        ret = last.get("daily_return", 0)
        return "✅", f"持仓{n_pos}只 市值{value:,.0f} {last_date}收益{ret:+.2%}"
    return "⚠️", f"持仓{n_pos}只 市值{value:,.0f} 无历史记录"


def check_shadow(date: str) -> tuple[str, str]:
    """Check shadow optimizer status."""
    state_path = DATA_DIR / "paper_shadow" / "oms_state.json"
    if not state_path.exists():
        return "❌", "Shadow 状态不存在"

    state = json.loads(state_path.read_text())
    value = state.get("total_value", 0)
    n_pos = len(state.get("positions", {}))
    history = state.get("daily_pnl_history", [])

    if history:
        last = history[-1]
        last_date = last.get("date", "")
        ret = last.get("daily_return", 0)
        return "✅", f"持仓{n_pos}只 市值{value:,.0f} {last_date}收益{ret:+.2%}"
    return "⚠️", f"持仓{n_pos}只 市值{value:,.0f}"


def check_llm_events(date: str) -> tuple[str, str]:
    """Check LLM event pipeline."""
    events_path = DATA_DIR / "llm_events" / f"{date}.jsonl"
    news_path = DATA_DIR / "daily_news" / f"{date}.jsonl"

    if not news_path.exists():
        return "❌", "新闻未采集"

    n_news = sum(1 for _ in open(news_path))
    if not events_path.exists():
        return "⚠️", f"新闻{n_news}条 事件未提取"

    n_events = sum(1 for _ in open(events_path))
    return "✅", f"新闻{n_news}条 事件{n_events}条"


def check_guba(date: str) -> tuple[str, str]:
    """Check guba popularity."""
    path = DATA_DIR / "guba" / f"{date}.jsonl"
    if not path.exists():
        return "❌", "未采集"
    n = sum(1 for _ in open(path))
    return "✅", f"人气榜{n}只"


def check_promotion_eligibility() -> tuple[str, str]:
    """Check if shadow has enough data for promotion."""
    compare_path = DATA_DIR / "paper_shadow" / "daily_compare.jsonl"
    shadow_state_path = DATA_DIR / "paper_shadow" / "oms_state.json"
    champion_state_path = DATA_DIR / "paper" / "oms_state.json"

    if not shadow_state_path.exists():
        return "❌", "Shadow 未运行"

    shadow = json.loads(shadow_state_path.read_text())
    champion = json.loads(champion_state_path.read_text()) if champion_state_path.exists() else {}

    sh_days = len(shadow.get("daily_pnl_history", []))
    ch_days = len(champion.get("daily_pnl_history", []))

    sh_ret = shadow.get("total_value", 1e6) / 1e6 - 1
    ch_ret = champion.get("total_value", 1e6) / 1e6 - 1
    excess = sh_ret - ch_ret

    if sh_days < 20:
        return "⏳", f"观察{sh_days}/20天 shadow{sh_ret:+.2%} vs champion{ch_ret:+.2%} excess{excess:+.2%}"

    # 20+ days: check if shadow beats champion
    if excess > 0:
        return "🟢", f"可晋升！{sh_days}天 shadow{sh_ret:+.2%} > champion{ch_ret:+.2%} excess{excess:+.2%}"
    else:
        return "🔴", f"{sh_days}天 shadow{sh_ret:+.2%} < champion{ch_ret:+.2%} excess{excess:+.2%}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    date = args.date

    logger.info(f"=== Daily Health Check: {date} ===")

    checks = []

    # 1. Morning recommendation
    status, detail = check_log_for_today(
        "cron_morning.log", date,
        success_keywords=["recommendations"],
        fail_keywords=["timed out", "failed", "0 recommendations"],
        job_id="morning_recommendation",
    )
    checks.append(("早盘推荐", status, detail))

    # 2. Model training (only Wed + Sat)
    from datetime import datetime as _dt
    day_of_week = _dt.strptime(date, "%Y-%m-%d").weekday()
    if day_of_week in (2, 5):  # Wed=2, Sat=5
        status, detail = check_training(date)
        checks.append(("模型训练", status, detail))
    else:
        checks.append(("模型训练", "⏭️", f"今天不训练(周{day_of_week+1})"))

    # 3. Paper trading (champion)
    status, detail = check_paper_trading(date)
    checks.append(("Champion", status, detail))

    # 4. Shadow trading
    status, detail = check_shadow(date)
    checks.append(("Shadow", status, detail))

    # 5. LLM events
    status, detail = check_llm_events(date)
    checks.append(("LLM事件", status, detail))

    # 6. Guba popularity
    status, detail = check_guba(date)
    checks.append(("人气榜", status, detail))

    # 7. Data update
    status, detail = check_log_for_today(
        "data_update.log", date,
        success_keywords=["successfully", "instruments"],
        fail_keywords=["failed", "error"],
        job_id="qlib_data_update",
    )
    checks.append(("数据更新", status, detail))

    # 8. Shadow promotion eligibility
    status, detail = check_promotion_eligibility()
    checks.append(("Shadow晋升", status, detail))

    # 9. Regime controller
    try:
        from signals.regime_controller import RegimeController
        rc = RegimeController()
        regime = rc.compute()
        alert = regime["alert_level"]
        risk_on = regime["risk_on_score"]
        adj = regime.get("suggested_adjustments", {})
        reason = adj.get("reason", "")
        icon = {"normal": "🟢", "watch": "🟡", "warning": "🟠", "critical": "🔴"}.get(alert, "?")
        checks.append(("Regime", icon, f"{alert} risk={risk_on:+.2f} {reason}"))
        if alert in ("warning", "critical"):
            has_issue = True
    except Exception as e:
        checks.append(("Regime", "⚠️", f"计算失败: {e}"))

    # 10. Registry status
    try:
        from models.registry import ModelRegistry
        reg = ModelRegistry()
        rs = reg.status()
        ch = rs.get("champion", "?")
        sh = rs.get("shadow", "?")
        ch_exec = rs.get("champion_execution", {}).get("mode", "?")
        sh_exec = rs.get("shadow_execution", {}).get("mode", "?")
        checks.append(("Registry", "✅", f"champion={ch}({ch_exec}) shadow={sh}({sh_exec})"))
    except Exception:
        checks.append(("Registry", "⚠️", "无法读取"))

    # Print all checks
    msg_lines = [f"📊 系统健康检查 {date}"]
    has_issue = False
    for name, status, detail in checks:
        line = f"{status} {name}: {detail}"
        msg_lines.append(line)
        logger.info(f"  {line}")
        if status not in ("✅", "🟢", "⏭️", "⏳"):
            has_issue = True

    if has_issue:
        msg_lines.append("\n⚠️ 有异常项需要关注")
    else:
        msg_lines.append("\n✅ 全部正常")

    msg = "\n".join(msg_lines)

    # Push
    if not args.no_push:
        try:
            from push.wechat import WeChatPusher
            WeChatPusher().send(msg, title="系统健康检查")
            logger.info("  Push sent")
        except Exception as e:
            logger.warning(f"  Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
