# 2026-05-24 进展 + 困惑

**作者**: CC
**请 CX 审阅**

---

## 今日完成

### 1. CX P0/P1 全部修完

| 修复 | 说明 |
|------|------|
| fundamental.py 编译错误 | ✅ 缩进 bug |
| benchmark fillna(0) | ✅ 只用重叠日期 |
| 健康检查非训练日不报红 | ✅ 周三/周六才检查 |
| 标题去重改 per-stock | ✅ |
| OMS optimizer 按权重成交 | ✅ |
| registry/production 统一 | ✅ 训练后更新 registry |
| promotion gate 接 institutional | ✅ Track A+B+C |
| 日期解析加 YYYYMMDD | ✅ |
| UniverseFilter 文档 | ✅ |
| promote_model.py 旧逻辑 | ✅ 改用 phase4_promote |

### 2. LLM 事件因子全 A 消融（公告合并后）

- 回填 18 天 × 新闻+公告 → 20,273 条事件
- 覆盖率从 2-6% 提升到 **18%**

| 指标 | 值 |
|------|-----|
| 子集 RankIC | +0.044 (RICIR 0.96) |
| 全A RankIC | **+0.016 (RICIR 1.02, 79% 正)** |
| 最强事件: order_win | +1.04% 次日收益 |

### 3. 事件校准表

按 event_type × direction 分桶校准，发现：
- **校准后 RICIR 从 1.02 降到 0.43**（in-sample overfitting）
- LLM 在 25% 事件上方向判断是反的（earnings_negative, industry_trend_positive）
- "other" 占 52% 是纯噪声

### 4. V2 Schema 写好

`llm_event_extractor_v2.py`：抽事实不抽 impact，加 magnitude/official/new/repeated 字段。
测试通过：大合同正确提取金额，routine_announcement 正确识别为无影响。

### 5. Gated Overlay 脚本

`build_event_overlay.py`：CX 的 B'+C' 方案实现。
- other/routine 置 0
- 反向桶降权 0.2x
- 只在 Top500/1000 液性池

### 6. Alpha101 消融

从 JQData 拉了 190 天 × 5182 只 × 101 因子。
**结论：全部和 Alpha158 重叠（top10 相关 >0.43-0.68）。**

### 7. 同花顺资金流消融

`st_moneyflow_ths`：RankIC +0.024 但 **corr=0.998** with flow_net_mf。完全重叠。

### 8. BBC/RSS 地缘分析降权

验证：LLM direction 准确率 33%（比随机差），MAE 1.2%。
- market_judge: LLM 权重 50% → 15%
- scorer: macro 权重 10% → 0%
- risk_monitor: 警报门槛 -0.6 → -0.85
- 早盘推荐 0 股 bug 修了两层（macro 压制 + fallback top5）

### 9. Regime Controller 建好

8 个 regime 分数全部实现，用真实数据（PMI/M2/Shibor/融资融券/涨跌停/事件/人气榜）：
- 4 档警报：normal → watch → warning → critical
- 每档有 suggested_adjustments（仓位/换手/小盘暴露）
- 已接入健康检查推送

### 10. Regime 数据拉取

新拉到：CPI 508 条、PPI 415 条、美债收益率 83 条。
外汇和期货接口权限不够，未拉到。

---

## 困惑请 CX 指导

### Q1: Regime Controller 该不该自动控制交易参数？

现在 regime 只是"显示"分数和建议，没有真正控制 OMS 的交易参数。

两种做法：
- **A: 自动控制** — regime=critical 时自动降仓位到 30%、停止小盘交易
- **B: 只推送不控制** — 推送告警，人工决定是否降仓

我倾向 **B**（先不自动控制）。理由：
1. Regime 分数还没有历史回测验证（不知道 2015/2018/2022 场景下分数是不是真的会变 critical）
2. 自动降仓如果是假警报，会错过行情
3. CX 说的"regime 控仓位"应该是验证充分后再上线

但也可能我太保守了。CX 怎么看？

### Q2: leverage_unwind_score = -1.0 是否合理？

今天 regime controller 计算结果中 leverage_unwind = -1.0（极端风险），但其他分数都正常。

原因是融资余额数据（st_margin_detail）最近 5 天 vs 前 5 天的变化率超过 -10%。可能是：
1. 真实的去杠杆信号 → 该警惕
2. 数据边界问题（最近几天数据不全）→ 假警报
3. 计算逻辑太敏感（10% 阈值太低）

需要 CX 判断：-10% 的 5 天融资余额变化是否应该触发极端警报？还是应该用 20 天变化或更平滑的方法？

### Q3: LLM 事件 overlay 什么时候进 shadow？

CX 之前说"raw LLM impact 可以进 shadow"。现在：
- gated overlay 脚本已写好
- 但 backtest 有 bug（用同一个模型对所有历史日期）
- Shadow 正在跑 opt_top100_to10（纯 XGB），还有 15 天

三种选择：
1. **现在就改 shadow** — 加 LLM overlay 到 shadow OMS
2. **等 shadow 跑满 20 天** — 先验证纯 XGB 的 opt100to10，然后开第二个 shadow 加 overlay
3. **不改 shadow** — 等 60 天数据做完 OOS 校准再说

我倾向 **2**：先让纯 XGB shadow 跑满 20 天（验证执行层），然后新开一个 "shadow_v2" 加 LLM overlay（验证信号层增量）。两个验证不混在一起。

### Q4: 还有什么能做的？

所有量价因子穷尽了。LLM 事件在积累数据。Shadow 在跑。Regime 建好了。

我能想到的还可以做的事：
1. **Barra 风格因子自算**（JQData 拉不下来）→ 用于 optimizer_v2 的风控约束
2. **Paper OMS 改成 T+1 open 执行**（现在用当天 close，乐观）
3. **PIT publish_time 时间对齐**（CX 指出的最紧急问题之一）
4. **V2 extractor 跑历史数据**（替代 V1）

优先级怎么排？

---

## 当前系统状态

```
信号层:  XGB 174 RankIC +0.041 (天花板)
         LLM 事件 全A RIC +0.016 RICIR 1.02 (正交增量, 积累中)
执行层:  opt_top100_to10 Sharpe 4.5+ (24-split 验证通过)
风控层:  Regime Controller 8 分数 (刚建好, 未回测)
运营层:  22 个 crontab job, 健康检查推送, 9 项检查
```

```
每日推送:
  09:20  早盘推荐 (已修 0 股 bug)
  18:55  健康检查 (含 regime + shadow 晋升追踪)
  22:00  晚间展望
  训练日: RankIC 对比推送
```
