可以，而且我觉得“模型压缩”其实是一个很适合 Harness 化的领域。因为模型压缩天然就是一个反复试验的闭环：

选择压缩策略 → 执行压缩 → 部署模型 → 测性能/精度 → 定位退化来源 → 修改压缩策略 → 再实验

所以我建议不要把它做成一个“模型压缩工具箱”，而是定义成：

Compression Harness = 压缩工具 + 实验环境 + 评测器 + 诊断器 + 实验记忆 + 自动决策 Controller

最终它应该能够从：

“Qwen-8B，希望显存降低 50%，MMLU 下降 <1%，同时 TPOT 至少快 20%”

自动走到一套 Pareto 最优压缩方案。

1. 整体架构

我会把系统设计成下面这个闭环：

                    ┌──────────────────────────┐
                    │     Compression Goal     │
                    │ model / hardware / SLO   │
                    │ quality budget / tasks   │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │      Planner / Agent     │
                    │   下一步做什么实验？      │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │     Recipe Compiler      │
                    │ bit / layer / method /   │
                    │ calibration / sparsity   │
                    └────────────┬─────────────┘
                                 ↓
             ┌────────────────────────────────────┐
             │         Compression Executor       │
             │                                    │
             │ Quantization | Pruning | Distill  │
             │ KV Compress  | Low-rank | Hybrid  │
             └────────────────┬───────────────────┘
                              ↓
                    ┌──────────────────────────┐
                    │      Deployment Env      │
                    │ vLLM / torch / SGLang   │
                    └────────────┬─────────────┘
                                 ↓
               ┌────────────────────────────────┐
               │            Evaluator           │
               │ quality / latency / memory /   │
               │ throughput / long-context      │
               └──────────────┬─────────────────┘
                              ↓
                    ┌──────────────────────────┐
                    │        Diagnoser         │
                    │ 哪里压坏了？为什么？       │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │    Experiment Memory     │
                    │ recipe / result / failure│
                    │ sensitivity / knowledge  │
                    └────────────┬─────────────┘
                                 │
                                 └──────→ Planner

这里真正体现 Harness 的地方，是中间那个闭环，而不是任何某一种压缩算法。

2. 第一层：Goal Specification

这个部分非常重要。

不要让用户直接说：

帮我跑 AWQ 4-bit。

而应该让用户描述目标：

model:
  name: Qwen/Qwen3-8B
  dtype: bf16

hardware:
  gpu: RTX4090
  num_gpu: 1

objective:
  memory:
    max_gb: 10

  quality:
    max_drop: 0.01

  latency:
    p95_tpot_ms: 25

  throughput:
    min_tokens_per_second: 80

evaluation:
  tasks:
    - mmlu
    - gsm8k
    - humaneval

constraints:
  training_allowed: false
  max_gpu_hours: 20

然后 Harness 自己决定：

应该先尝试 FP8？INT8？W4A16？AutoRound？mixed precision？

这和一般模型压缩工具的区别非常大。

3. 第二层：Compression Action Space

不要让 Agent 自由写 Python。

给 Agent 一个有限、结构化的 Action Space。

例如：

profile_model()

quantize(
    method="autoround",
    weight_bits=4,
    activation_bits=16,
    group_size=128,
    targets=["linear"]
)

quantize(
    method="gptq",
    layer_range=[4, 28]
)

sparsify(
    method="wanda",
    sparsity=0.5,
    structure="2:4"
)

exclude_layers([
    "model.layers.0",
    "model.layers.31"
])

set_layer_precision(
    layers=[0, 1, 30, 31],
    bits=8
)

calibrate(
    dataset="pile",
    samples=256
)

serve(
    backend="vllm"
)

evaluate()

compare("run_001", "run_002")

diagnose("run_002")

Agent 只能组合这些 primitive。

这样整个 Harness 才是可复现、可审计、可控的。

4. 压缩算法本身不要自己重新实现

这一层应该做成 Adapter。

现在已经有很多可以直接复用的轮子。

Quantization

优先接：

LLM Compressor

它目前已经支持 RTN、GPTQ、AWQ、SmoothQuant、SpinQuant、QuIP、AutoRound，以及 FP8/INT8/INT4/FP4/KV Cache quantization 等，并且输出可以直接给 vLLM 部署。

例如：

harness
   ↓
LLMCompressorAdapter
   ↓
GPTQ / AWQ / AutoRound / FP8 / INT4

AutoRound 也非常适合纳入搜索空间。目前它重点支持 2–4 bit 低比特压缩，并且已经能和 vLLM 等推理环境衔接。

Torch-native Compression

第二个值得重点支持的是：

torchao

因为它已经把：

INT8
FP8
weight-only quantization
W+A quantization
QAT
2:4 sparsity
Wanda sparsification

逐渐统一到 PyTorch 的接口下面。

尤其有意思的是它支持：

quantize_(model, config)

以及：

sparsify_(model, config)

这种接口天然适合作为 Harness primitive。

Pruning

Pruning 最好单独做 Adapter：

WandaAdapter
SparseGPTAdapter
TorchAOPruningAdapter

Wanda 的官方实现本身已经提供 magnitude / Wanda / SparseGPT，以及 unstructured / 2:4 / 4:8 等不同设置。

这里我反而不建议 Harness 绑定 LLM Compressor。最新 LLM Compressor 文档已经明确说明，它们正在弱化/取消 sparse compression 支持，主要原因是硬件支持和使用需求不足。

这正好说明为什么 Harness 要有 Adapter Layer：

上层 Compression Harness 不关心底层具体是谁实现 Wanda/GPTQ。

5. 真正关键的模块：Compression Diagnoser

如果只是：

GPTQ 4bit → eval → score=68.3
AutoRound 4bit → eval → score=69.7

这其实只是 AutoML。

我觉得你这个系统最值得研究的地方，是：

让 Harness 理解“为什么压坏了”。

例如 AutoRound W4A16 之后：

Baseline MMLU       74.3
Compressed MMLU     72.1

GSM8K
82.4 → 76.3

HumanEval
71.2 → 70.9

Harness 应该继续分析：

Diagnosis

Main regression:
    mathematical reasoning

Layer sensitivity:
    L0-L3     high
    L4-L21    low
    L22-L27   medium
    L28-L31   high

Activation outliers:
    L2 MLP.down_proj
    L29 attention.o_proj

Error concentration:
    long CoT > short answers

Hypothesis:
    early/late layers are over-quantized

然后自动生成下一实验：

Experiment #17

Keep:
L0-L3   INT8
L28-L31 INT8

Quantize:
L4-L27  INT4

Expected:
+1.2 MMLU
+0.7 GB memory

这一下就从：

compression benchmark

变成：

compression reasoning harness。

6. 给每个模型生成 Compressibility Profile

这个我非常建议做。

Harness 第一次拿到一个模型后，先不要直接大规模搜索。

先运行一组 cheap probes。

得到：

Qwen3-8B Compressibility Profile

                   sensitivity
Layer 0       █████████
Layer 1       ████████
Layer 2       ███████
...
Layer 12      ██
Layer 13      █
...
Layer 29      ███████
Layer 30      █████████
Layer 31      ██████████

以及：

                     INT8      INT4      INT3
Attention QKV        low       low       medium
Attention O          low       medium    high
MLP gate             low       medium    high
MLP up               low       low       medium
MLP down             medium    high      high
Embedding             low       medium    -
LM Head               low       high      -

这就是模型的一个：

Compression Fingerprint

以后 Planner 搜索的时候就不用从零开始。

7. 这样搜索，而不是 Grid Search

我会设计成四阶段。

Phase 0：Baseline

先测：

Quality
Memory
TTFT
TPOT
Throughput
Peak VRAM
KV memory

尤其不能只报告 model size。

因为：

文件从 16 GB → 4 GB

不等于：

serving latency 一定下降 4 倍。

Phase 1：粗粒度探索

跑少量全局 recipe：

BF16
FP8
W8A8
W4A16 RTN
W4A16 GPTQ
W4A16 AWQ
W4A16 AutoRound

快速获得第一条 Pareto frontier。

Phase 2：Sensitivity Probe

例如：

只压 layer 0-7
只压 layer 8-15
只压 layer 16-23
只压 layer 24-31

再细分：

attention
MLP
embedding
LM head

Harness 自动定位：

哪部分最不耐压。

Phase 3：Mixed Compression

这一步才真正开始 autonomous optimization。

例如：

Layer 0-3      W8A8
Layer 4-26     W4A16
Layer 27-31    W8A8

QKV            INT4
O projection   INT8

MLP up         INT4
MLP down       INT8

甚至：

Quantization
+
KV cache FP8
+
2:4 sparsity

形成组合搜索。

Phase 4：Recovery

如果距离 constraint 只差一点：

quality drop = 1.6%
requirement   = <1%

Planner 再决定是否值得：

QAT
LoRA recovery
distillation
re-calibration
outlier exception

而不是从头换一种算法。

8. Evaluation Harness

可以直接把现成的 evaluation harness 嵌进来。

例如 EleutherAI 的 lm-evaluation-harness 本身就支持大量 zero/few-shot benchmark，而且提供 task versioning，适合保证不同压缩实验之间的可复现比较。

但是 Compression Harness 不能只测 benchmark accuracy。

我会至少分成四类。

Dimension	指标
Quality	MMLU / GSM8K / HumanEval / task-specific
Memory	weights / KV / peak VRAM
Latency	TTFT / TPOT / p50 / p95
Throughput	tok/s / req/s / concurrency

vLLM 目前已经直接提供 vllm bench latency / serve / throughput / sweep，所以系统性能这一层也没有必要自己造轮子。

9. 不要设计成单一 Score

这一点对模型压缩特别重要。

例如：

Recipe A

MMLU      73.5
VRAM      8.2G
TPOT      18ms

Recipe B

MMLU      74.0
VRAM      10.1G
TPOT      16ms

谁好？

没有唯一答案。

所以系统内部应该维护：

Pareto Frontier

例如二维：

Quality
  ↑
75|              ● BF16
74|         ● INT8
73|     ● mixed
72|  ● INT4
71|
  +------------------------→ Memory

实际上可以是：

$$ \mathcal P = Pareto(Q,-M,-L,T) $$

其中：

\(Q\)：quality
\(M\)：memory
\(L\)：latency
\(T\)：throughput

然后用户给 constraint：

$$ Q\ge Q_0-\epsilon $$ $$ Memory \le B $$ $$ TPOT\le L_{max} $$

Harness 去寻找满足约束的 recipe。

这比：

$$ Score=accuracy-\lambda memory $$

科学得多。

10. Experiment Memory 是 Harness 和普通 Pipeline 最大的区别

每一次运行都要存成一个结构化 artifact：

{
  "run_id": "run_018",
  "parent": "run_011",

  "model": "Qwen3-8B",

  "hypothesis": "early layers are quantization sensitive",

  "recipe": {
    "layers_0_3": "int8",
    "layers_4_27": "int4",
    "layers_28_31": "int8"
  },

  "results": {
    "mmlu": 0.738,
    "gsm8k": 0.812,
    "vram": 9.2,
    "tpot": 19.3
  },

  "diagnosis": {
    "remaining_regression": "math_reasoning",
    "sensitive_modules": [
      "layer29.mlp.down_proj"
    ]
  }
}

慢慢地你会得到：

Model
   │
   ├── INT8
   │
   ├── INT4
   │    ├── calibration=128
   │    ├── calibration=512
   │    └── mixed precision
   │          ├── ...
   │
   └── pruning

这实际上就是一棵：

Compression Research Tree

11. 再往前一步：跨模型经验迁移

这个方向我觉得特别适合做成论文。

例如 Harness 已经压过：

Llama-3-8B
Qwen2.5-7B
Qwen3-8B
Mistral-7B

它逐渐总结出：

Knowledge #17

Decoder-only transformers:
first/last layers usually show higher
low-bit quantization sensitivity.

Evidence:
17 / 21 runs

Models:
Llama / Qwen / Mistral

然后新来：

Qwen3.5-14B

Planner 不需要完全从零探索。

这时候就出现真正意义上的：

Compression Harness RSI
模型压缩经验
      ↓
改进 Planner
      ↓
更少的实验找到更好的压缩方案
      ↓
产生新的压缩经验
      ↓
进一步改进 Planner

所以你之前提到的 RSI，在这个场景里其实非常自然。

12. UI 我反而建议做成“研究工作台”

不要做：

[模型]
[算法]
[bit]
[运行]

这种普通工具界面。

可以设计成：

┌──────────────────────────────────────────────────────────────┐
│ Qwen3-8B     RTX4090       Target: <10GB / ΔAcc<1%          │
├───────────────┬───────────────────────────┬──────────────────┤
│ Goal          │ Experiment Graph          │ Current Agent    │
│               │                           │                  │
│ VRAM < 10GB   │ BF16                      │ Hypothesis       │
│ ΔQ < 1%       │  ├─ FP8                   │ early layers     │
│ TPOT < 20ms   │  ├─ INT8                  │ are sensitive    │
│               │  └─ INT4                  │                  │
│ Hardware      │       ├─ GPTQ              │ Next experiment  │
│ RTX4090       │       ├─ AWQ               │ L0-3 → INT8      │
│               │       └─ AutoRound         │ others → INT4    │
│               │              └─ mixed ★    │                  │
├───────────────┴───────────────────────────┴──────────────────┤
│                      Pareto Frontier                         │
│                                                              │
│ Quality ↑              ● BF16                                │
│                        ● INT8                                 │
│                   ★ mixed                                    │
│              ● INT4                     → Memory              │
├──────────────────────────────────────────────────────────────┤
│ Regression diagnosis | Layer sensitivity | experiment logs   │
└──────────────────────────────────────────────────────────────┘

这个界面会非常有 Harness 感。

它展示的是：

模型压缩正在如何被完成。

而不是只展示最后结果。

13. 代码结构可以非常清晰

第一版甚至不用很复杂：

compression-harness/

├── harness/
│   ├── controller.py
│   ├── planner.py
│   ├── executor.py
│   ├── evaluator.py
│   ├── diagnoser.py
│   └── memory.py
│
├── adapters/
│   ├── llm_compressor/
│   ├── autoround/
│   ├── torchao/
│   ├── wanda/
│   ├── vllm/
│   └── lm_eval/
│
├── actions/
│   ├── quantize.py
│   ├── prune.py
│   ├── distill.py
│   ├── serve.py
│   └── evaluate.py
│
├── recipes/
│   ├── fp8.yaml
│   ├── int8.yaml
│   ├── int4.yaml
│   └── mixed_precision.yaml
│
├── experiments/
│   ├── run_001/
│   ├── run_002/
│   └── ...
│
├── knowledge/
│   ├── layer_sensitivity.json
│   ├── failure_patterns.json
│   └── model_profiles.json
│
└── ui/
14. 我建议第一版千万别把量化、剪枝、蒸馏全部做了

最适合的 MVP 是：

Compression Harness v0.1

只支持：

Quantization

接三个 backend：

LLM Compressor
AutoRound
torchao

Evaluation：

lm-eval
+
vLLM bench

然后自己重点做三个东西：

① Compression Recipe DSL
method: autoround

layers:
  default:
    bits: 4
    group_size: 128

  "0-3":
    bits: 8

  "28-31":
    bits: 8

exclude:
  - lm_head
② Sensitivity Diagnoser

自动找到：

哪些 layer/module 导致性能损失。

③ Autonomous Planner

自动完成：

Baseline
 ↓
global INT8
 ↓
global INT4
 ↓
发现 INT4 精度掉 3%
 ↓
Sensitivity profiling
 ↓
找到 L0-L3、L29-L31
 ↓
mixed precision
 ↓
精度下降 0.7%
 ↓
满足 constraint
 ↓
停止

这一条链跑通，Compression Harness 就成立了。

15. 这件事情甚至可以进一步形式化成一个研究问题

我会把核心问题定义为：

Budgeted Autonomous Model Compression

给定：

$$ M,\ H,\ D,\ C,\ B $$

其中：

\(M\)：原始模型
\(H\)：目标硬件
\(D\)：任务分布
\(C\)：部署约束
\(B\)：实验预算

Harness 寻找：

$$ r^* = \arg\max_{r\in\mathcal R} Q(M_r,D) $$

subject to：

$$ Memory(M_r,H)\le C_m $$ $$ Latency(M_r,H)\le C_l $$ $$ Cost(r)\le B $$

这里真正有意思的问题已经不是：

什么量化算法最好？

而是：

在有限实验预算下，Agent 能否通过实验反馈越来越快地找到模型的 Compressibility Frontier？

这就同时连上了你之前关注的：

模型压缩 + Harness + Agent + RSI + 自动科研。

我甚至觉得这比单独再做一种 W4A16 算法更有潜力形成你未来一条比较完整的研究主线。

如果让我现在开始搭

我会直接定一个 v0.1 目标：

输入任意 HuggingFace LLM + 一块目标 GPU + 精度/显存/延迟约束，Harness 在不人工干预的情况下，自动完成 10～30 次压缩实验，并给出一组 Pareto-optimal compression recipes，同时解释每一次 recipe 为什么被选择。

技术栈可以先固定成：

Controller        GPT / DeepSeek / local LLM
       ↓
Compression       LLM Compressor + torchao + AutoRound
       ↓
Serving           vLLM
       ↓
Quality Eval      lm-evaluation-harness
       ↓
System Eval       vLLM bench
       ↓
Storage           SQLite + JSON artifacts
       ↓
UI                React / Next.js

第一篇真正值得研究的 contribution，我会放在 Diagnose → Plan 这条边，而不是 Compression Executor。

换句话说：

现有工作解决“怎么压”，你的 Harness 解决“下一步应该怎么压”。

这会是整个系统最核心的定位。