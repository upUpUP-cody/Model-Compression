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
