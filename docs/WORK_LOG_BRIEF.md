# 工作日志（精简）

> 详细版（结论与创新点）：[WORK_LOG.md](WORK_LOG.md)
> 日期：2026-08-13（压缩比 + sweep：2026-08-14）

## 已做

1. **MNIST P1.2 GPU 统计补全**：修好 dense_small 从零训练；跑完 3 seed × 6 压缩率；迭代/自主搜索在高压缩下仍约 97–98% test acc，one-shot 超过 2x 后崩溃。
2. **P2 代码**：CIFAR train/val/test 协议、ResNet-18、块内 conv1 通道剪枝、CNN Wanda、六方法对照 runner、LoRA/自蒸馏接口。
3. **CIFAR 基线 + smoke**：20 epoch 基线 test **87.88%**；早期轻剪 smoke / report-test 修复。
4. **旧 formal/multiseed**：当时 YAML 只剪 1–2 层，实际约 **1.00x**（作废对比用）。
5. **CNN 目标压缩比**：conv1-only 上限约 **15.1x**；配置由 `target_compression_ratio` 推导全部 8 个 conv1。
6. **CIFAR 6×3 sweep（已跑通，~22 min）**
   - 目录：`results/cifar_p12_comparison_gpu_sweep/`，报告 `AGGREGATE_REPORT.md`
   - oneshot/iterative 实际压缩全档达标（2.01x … 10.16x）
   - iterative + Level1 恢复：10x test 仍约 **83.4%**；oneshot ≥2x 崩溃
   - autonomous_search **全程 1.00x**（Cheap Critic 在恢复前拒绝候选）

## 现在处于哪一步

压缩率扫描完成。方法主证据是 **迭代恢复**；搜索在当前门禁下未真正压缩。

## 下一步

调搜索/Cheap Critic 门禁（或单独 search 对照）→ 恢复消融 Level 1/2/3。
