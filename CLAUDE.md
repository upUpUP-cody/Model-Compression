# Claude Code 项目规范

## 已知环境限制

### Windows GBK 编码问题
**规则**: 所有 `print()` 输出中严禁使用 emoji 字符（🎯📊✅💪🏆📉⚠ 等）。

**原因**: 本项目运行在 Windows 环境下，终端默认使用 GBK 编码，无法输出 emoji，会触发 `UnicodeEncodeError` 导致实验崩溃。即使实验计算本身已经全部完成，崩溃发生在打印阶段，结果文件同样不会被保存。

**正确做法**:
```python
# 错误
print("🎯 最佳权衡点:")
print("✅ 完成")
print("⚠ 警告")

# 正确
print("[Best Tradeoff]")
print("[OK] Done")
print("[WARNING]")
```

**排查范围**: 每次新增实验脚本或模块，用以下命令检查：
```bash
grep -rn "[\U0001F000-\U0001FFFF]" src/ experiments/
```

---

## 长时间实验规范

**规则**: 每次启动预计超过 5 分钟的后台实验，必须在启动后明确告知用户以下信息：

1. **实验名称** — 做什么
2. **预计时长** — 基于 epoch 数 × 每 epoch 耗时估算
3. **输出文件** — 完成后会生成哪些文件

**估算方法**（CPU 环境下 MNIST MLP）:
- 每个训练 epoch ≈ 10-15 秒
- Frontier Profiling（9 个比例 × 10 epoch）≈ 15-20 分钟
- Lottery Ticket Proposal（3 个压缩率 × 3 策略 × 10 epoch）≈ 20-30 分钟

**示例通知格式**:
```
已启动后台实验: Frontier Profiling
预计时长: 约 20 分钟（9 个剪枝比例 × 10 epoch 恢复训练）
完成后生成: results/frontier_profiling_results.json, results/frontier_curve.png
```

### LLM / GPU 与配置优化告知

**硬件基线（写死）**：当前只有 **1 张 GPU**（RTX 4090）。用户**仅在 Agent 明确说「需要两张卡」时才加第二张**。平时不催促、不默认规划双卡。

**Agent 职责**：启动或被问进度时，按下列标准判断；需要 2 卡时主动告知；不需要时在回复里一句带过「仍单卡即可」。配置问题与加卡分开说。不擅自改正在跑的 job。

#### A. 通知「请加第二张卡」（默认不触发）

仅当下列条件**同时**满足时，主动说：请加第二张卡；加好后按 target/task 双进程拆跑（给出命令）。若本趟已在跑则继续单卡，下趟再用双卡。

1. 计划或剩余工作可拆成 **≥2 个独立 cell 组**（不同 `target` / `task` / seed），且
2. 单卡预估墙钟 **≥ 约 3 小时**（或正式矩阵：多 seed × 多 target，明显长于 K6 小扫），且
3. 加速方式是 **多进程拆分**（不是模型并行），且
4. 拆分不会破坏同预算对照协议

**明确不通知加卡**：

- 当前 K6 小矩阵（1.5x/2x/4x × 四方法、小样本帽）——单卡继续
- 显存未打满、利用率间歇 0% —— 串行换模/小 batch，不是加卡理由
- 1.5B 模型并行 / 双卡训练 —— **永不建议**
- 评测坏了、超剪、OOM —— 先改配置，不因此要求加卡

加卡到位后的推荐用法（多进程拆分，以实际 CLI 为准）：

```bash
# Example: split independent Stage C cells across two GPUs (adjust configs/CLI to actual E job)
CUDA_VISIBLE_DEVICES=0 python experiments/stage_c/run_e8_random_recovery.py --config configs/stage_c/e8_random_recovery.yaml
CUDA_VISIBLE_DEVICES=1 python experiments/stage_c/run_e9_high_gap_recovery.py --config configs/stage_c/e9_high_gap_recovery.yaml
```

#### B. 建议优化配置（与加卡独立；单卡也可触发）

出现任一条即告知，**不擅自改正在跑的 job**：

| 信号 | 建议 |
|------|------|
| 同方法单格墙钟 **> 25 分钟** 且样本帽仍很小 | 查重载权重 / 恢复过重 |
| OOM 或显存持续 **> 90%** | 降 `max_seq_len` / batch / 恢复；先单卡救活，不因此要求加卡 |
| GPU 利用率长期 ~0% 且墙钟很长 | 减每方法重载或提高有效 batch，**不是**加卡 |
| iterative 相对目标 **超剪 >15%** | 修 stage 预算对齐 |
| dense 指标异常（~0 / CE 非有限） | 先修评测，再谈加速 |
| 下趟规划总时长 **> 3 小时** 且可拆 cell | **升级为 A：请加第二张卡** |

通知模板：

```text
[资源配置建议]
实验: <名称>
硬件基线: 1 卡（仅在需要时请你加第 2 卡）
判定: 需要加卡 | 不需要加卡 | 仅需改配置
依据: <触发条件>
建议: <具体动作；本趟是否继续单卡>
```

进度类回答不沉默：要么「请加卡」，要么「仍单卡即可」（可附配置建议）。

---

## 任务完成与自动提交

**规则**: 一个开发任务完成后，必须先完成验证，再自动提交本次任务涉及的代码、配置、测试和文档变更。

提交前必须执行：

1. 运行与本次改动相关的聚焦测试。
2. 运行完整测试：`python -m pytest tests -q`。
3. 运行 `git diff --check`。
4. 检查 `git status --short` 和 `git diff`，确认没有误包含临时产物、数据集、缓存、大型 checkpoint 或他人未授权的修改。
5. 只有所有检查通过后，才执行 Git commit，并在提交后检查 `git status --short` 和 `git log -1 --oneline`。

如果测试、差异检查或变更范围检查失败，禁止自动提交，必须先修复问题或明确报告阻塞原因。`.claude/settings.json` 属于本地用户配置，默认不得加入项目提交。

提交信息应简洁描述本次任务，并以以下署名结尾：

```
Co-Authored-By: Claude <noreply@anthropic.com>
```
