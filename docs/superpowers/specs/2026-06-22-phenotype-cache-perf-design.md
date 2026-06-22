# 表型缓存重建性能修复设计

**日期:** 2026-06-22
**状态:** 设计 — 待用户 review
**作者:** hub (Claude) + 3 路 subagent(性能工程 / 设计保真审计 / ponytail)

## 1. 问题陈述

仲裁性能优化轮(`2026-06-21-arbitration-perf-design.md`)落地后,实测发现 PHASE3 仲裁**已不是瓶颈**(逐 tick ≤69ms,满世界 50ms,整跑 T=450 的 arb 部分仅 21.2s)。真正卡死 T=450×4 跑批的瓶颈在**别处** —— `StrainTable.phenotype_arrays`。

### 1.1 实测诊断(measured + 代码确认,非臆测)

逐 tick 计时曲线(`scripts/diag_arb_curve.py`,纯只读,只计 phase3 本体):arb 全程 8–69ms,与"占用增长"无关。但 phase-probe 的 arb 桶报 1129ms/tick —— 两者差 ~20×。根因:`run_batch.py` 的 `phase_probe`(L90-97)把 `e._refresh_phe()` 计进了 arb 桶。真正吃 ~1080ms/tick 的是 `_refresh_phe()` 调用的 `phenotype_arrays`,**不在仲裁 kernel 里**。

op 级剖析(`scripts/diag_arb_sections.py`)+ 读码确认了机制:

- `phenotype_arrays`(`src/des/phenotype_cache.py:44-83)每次"脏"重建时,用 python `for sid in range(1, n)` 循环,逐行做 ~10 个**标量 GPU 写**(`f[sid] = phe.f` …共 10 字段)。每个标量写进 CUDA 张量 = 一次 CPU→GPU 同步。
- `Engine._refresh_phe()`(`engine.py:29-32`)在表增长时调它;突变几乎每 tick mint 新 strain → 表单调增长 → **每 tick 全量重建**。第 n tick 要 python 循环 n 次、做 ~10n 次标量 GPU 写。
- tick449 表里 ~万级 strain → 每 tick ~10 万次标量 GPU 写 ≈ ~1080ms/tick。

**核心浪费(决定性观察):** 瓶颈**不是**"每 tick 重建全部 n 行",而是那 **10n 次标量 GPU 同步**。strain id 追加单调、`_id_to_phe[sid]` mint 后不可变 → 那些 Phenotype 对象早已在 CPU 侧算好缓存,重建只是把它们**逐个标量**搬进 GPU。
