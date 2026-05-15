"""Push training comparison results to WeChat via pushplus.

Reads baseline + ablation + latest results and sends a formatted summary.

Usage:
    python scripts/push_training_results.py
    python scripts/push_training_results.py --title "174维训练结果"
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from push.wechat import WeChatPusher

DATA_DIR = PROJECT_ROOT / "data" / "storage"


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_metric(val, pct=False):
    if val is None:
        return "—"
    if pct:
        return f"{val*100:+.3f}%"
    return f"{val:+.4f}"


def build_message():
    lines = []
    lines.append("📊 模型训练结果对比报告")
    lines.append("=" * 40)

    # 1. PIT-safe baseline comparison
    baseline = load_json(DATA_DIR / "pit_baseline_comparison.json")
    if baseline and "results" in baseline:
        lines.append("")
        lines.append("一、PIT-safe 基线对比")
        lines.append(f"  评估时间: {baseline.get('evaluated_at', '?')}")
        lines.append("")
        lines.append(f"{'模型':<10} {'维度':<5} {'IC':>8} {'ICIR':>7} {'RankIC':>8} {'Spread':>9} {'RIC>0':>6}")
        lines.append("-" * 55)
        for r in baseline["results"]:
            if "error" in r:
                lines.append(f"{r['model']:<10} {r['dim_mode']:<5} ERROR")
                continue
            lines.append(
                f"{r['model']:<10} {r['dim_mode']:<5} "
                f"{format_metric(r.get('ic_mean')):>8} "
                f"{format_metric(r.get('icir')):>7} "
                f"{format_metric(r.get('rank_ic_mean')):>8} "
                f"{format_metric(r.get('top20_spread'), pct=True):>9} "
                f"{r.get('rank_ic_pos_ratio', 0):.0%}".rjust(6)
            )

    # 2. Factor ablation
    ablation = load_json(DATA_DIR / "factor_ablation_v2.json")
    if ablation and "results" in ablation:
        lines.append("")
        lines.append("二、因子消融实验")
        baseline_ric = ablation.get("baseline_rank_ic", 0)
        lines.append(f"  基线 RankIC: {format_metric(baseline_ric)}")
        lines.append("")
        lines.append(f"{'因子组':<28} {'维度':>4} {'RankIC':>8} {'vs基线':>8} {'Spread':>9} {'负控?':>5}")
        lines.append("-" * 65)
        for r in ablation["results"]:
            delta = r.get("rank_ic_mean", 0) - baseline_ric if baseline_ric else 0
            shuf = "⚠" if r.get("shuffled") else ""
            lines.append(
                f"{r['name']:<28} {r['n_features']:>4} "
                f"{format_metric(r.get('rank_ic_mean')):>8} "
                f"{format_metric(delta):>8} "
                f"{format_metric(r.get('top20_spread'), pct=True):>9} "
                f"{shuf:>5}"
            )

    # 3. Latest enhanced XGB results
    enhanced = load_json(DATA_DIR / "xgb_enhanced_results.json")
    if enhanced:
        lines.append("")
        lines.append("三、最新增强 XGB 结果")
        lines.append(f"  特征维度: {enhanced.get('features', '?')}")
        lines.append(f"  IC:       {format_metric(enhanced.get('ic_mean'))}")
        lines.append(f"  ICIR:     {format_metric(enhanced.get('icir'))}")
        lines.append(f"  RankIC:   {format_metric(enhanced.get('rank_ic_mean'))}")
        lines.append(f"  Spread:   {format_metric(enhanced.get('top20_spread'), pct=True)}")
        lines.append(f"  标签:     {enhanced.get('label', '?')}")
        lines.append(f"  测试期:   {enhanced.get('test_period', '?')}")

    # 4. Key conclusions
    lines.append("")
    lines.append("四、关键结论")
    if ablation and "results" in ablation:
        passed = [r["name"] for r in ablation["results"]
                  if not r.get("shuffled") and r["name"] != "base_158"
                  and r.get("rank_ic_mean", 0) > baseline_ric + 0.005]
        failed = [r["name"] for r in ablation["results"]
                  if not r.get("shuffled") and r["name"] != "base_158" and r["name"] != "base_all_202"
                  and r.get("rank_ic_mean", 0) <= baseline_ric + 0.003]
        if passed:
            lines.append(f"  ✅ 通过消融: {', '.join(passed)}")
        if failed:
            lines.append(f"  ❌ 未通过: {', '.join(failed)}")

    lines.append("")
    lines.append("五、数据状态")
    for f in ["st_margin_detail", "st_top_list", "st_limit_list_d", "st_moneyflow_hsgt"]:
        p = DATA_DIR / f"{f}.parquet"
        if p.exists():
            import pandas as pd
            df = pd.read_parquet(p)
            dates = df["date"].nunique() if "date" in df.columns else "?"
            lines.append(f"  {f}: {len(df)} rows, {dates} dates ✅")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="模型训练结果对比")
    args = parser.parse_args()

    msg = build_message()
    print(msg)
    print()

    try:
        pusher = WeChatPusher()
        ok = pusher.send(msg, title=args.title)
        if ok:
            print("✅ 推送成功")
        else:
            print("❌ 推送失败")
    except Exception as e:
        print(f"❌ 推送异常: {e}")


if __name__ == "__main__":
    main()
