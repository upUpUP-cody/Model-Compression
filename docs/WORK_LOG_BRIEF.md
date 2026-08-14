# 工作日志（精简）

> 详细版（结论与创新点）：[WORK_LOG.md](WORK_LOG.md)
> 日期：2026-08-13（压缩比 / sweep / 搜索门禁：2026-08-14）

## 已做

1. **MNIST P1.2 GPU 统计补全**：修好 dense_small；3 seed × 6 压缩率；迭代/搜索高压缩仍约 97–98%，one-shot 过 2x 崩溃。
2. **P2 代码**：CIFAR 协议、ResNet-18、块内 conv1 剪枝、CNN Wanda、六方法对照、LoRA/自蒸馏接口。
3. **CIFAR 基线 + 目标压缩比**：基线 test **87.88%**；conv1-only 上限约 **15.1x**；YAML 由 `target_compression_ratio` 推导全部 8 个 conv1。
4. **CIFAR 6×3 sweep**：oneshot/iterative 压缩达标；iterative 10x test 约 **83.4%**；当时 search 因门禁错误全程 **1.00x**。
5. **修复 CNN 搜索门禁**
   - Cheap Critic 只排序；恢复后再做 2 点能力门禁
   - 2x formal：`cifar_p12_gpu_study_514cff755ed7`
   - search 实际压缩 **7.66x**，test **85.13%**（Critic 8–22% 仍可 accept）
   - 同 run iterative 2.01x / test 88.01%

## 现在处于哪一步

搜索已能真正压缩。旧 sweep 的 search=1.00x 结论作废。下一步是恢复消融。

## 下一步

恢复消融 Level 1/2/3 → 同压缩预算下 search vs iterative 对照（或修复后门禁后的 sweep）。
