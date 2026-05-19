# Phase 4S 速度基础设施实施笔记

**日期：** 2026-05-19
**作者：** CC

---

## 已完成的 7 项优化

| # | 优化 | 状态 | 实际效果 |
|:---:|------|:---:|------|
| 1 | Cache V2 分区存储 | ✅ | 构建成功，但 mmap 全量读 14s vs parquet 6s 反而慢 |
| 2 | Row-index 切片 | ✅ | date_indices.npy int32 索引 |
| 3 | 消融 baseline 缓存 | ✅ | 4 factor 从 48 次训练降到 12+16 次 |
| 4 | Shadow 推理化 | ✅ | shadow_daily_inference.py 不重训 |
| 5 | XGB nthread=12 | ✅ | ~35s→~20s per split |
| 6 | Alpha360 lazy loading | ✅ | CacheV2Reader 支持按组加载 |
| 7 | evaluate argpartition | ✅ | 去掉 sort_values，用 O(n) argpartition |

---

## 实施中的困惑和问题

### 1. Cache V2 的 mmap 比 parquet 更慢？

numpy .npy + mmap_mode="r" 读 3.6GB base158.npy 需要 14s，而 pd.read_parquet 只要 6s。

**原因分析：**
- parquet 有列式压缩，实际磁盘 IO 小得多
- numpy npy 是 raw float32，3.6GB 就是 3.6GB 磁盘 IO
- mmap 只在随机访问时有优势，顺序全量读反而不如压缩格式

**问题：** 是否应该改用 partitioned parquet（按年分区）而不是 npy？
或者保留 npy 但只在需要部分特征组时使用（如只加载 regime 的 619MB 而不是全部 3.6G）？

### 2. 并行训练在 macOS 上完全不可行

试了三种方案全失败：
- multiprocessing.Pool → fork 卡死（XGB C++ backend）
- ThreadPoolExecutor → GIL + 内存爆
- ThreadPoolExecutor + nthread=1 → 没有额外加速

**根本原因：** XGB 自己已经用了 12 核，再加线程并行 = 争抢 CPU。
真正的并行需要 subprocess（每个进程独立），但数据传输开销大。

**问题：** 有没有更好的方案？比如用 Ray 或 joblib？

### 3. 残差 IC 脚本 JSON 序列化报错

`phase2_residual_ic.py` 输出时 `bool` 类型无法 JSON 序列化。
numpy 的 `np.bool_` 不是 Python `bool`，需要转换。

### 4. 消融 baseline 缓存后，residual 可以直接传给后续因子

CX 提到的 two-stage residual model 可以自然接入：
- Split 内训练完 baseline 后，residual = y_test - pred_base
- 新因子的 residual IC 可以直接算，不需要额外训练
- 但完整的 two-stage model（用新因子预测 residual）还需要额外一次训练

**问题：** residual IC 是否足够判断增量价值，还是必须做 two-stage model？

### 5. shadow_daily_inference.py 的模型加载问题

当前 shadow model 存的是 `xgb_175_holder_model.json`，但 205 维模型没有单独存过。
shadow 需要：
- 定期（每周）重训并保存 champion/shadow 模型到固定路径
- daily inference 只加载最新模型文件

**问题：** 模型保存路径和命名规范需要 CX 定义。建议：
```
data/storage/models/champion_xgb_174_YYYYMMDD.json
data/storage/models/shadow_xgb_205_YYYYMMDD.json
```

---

## PIT Audit 进行中

正在跑 `phase2_pit_audit.py`，比较 flow_lag0 vs flow_lag1 vs flow_lag2 vs no_flow。
7/8 splits 完成，结果即将出来。这是 CX 说的"地基级"问题 — 如果 lag0→lag1 IC 大幅下降，174 维 baseline 有未来函数。
