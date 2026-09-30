# MemSifter 在 LoCoMo 与 LongMemEval 上的复现手册

本文档用于复现论文 **MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning** 在以下测试集上的结果：

| 论文中的名称 | 仓库数据集名称 | 测试样本数 | 原始记忆规模 |
| --- | --- | ---: | ---: |
| LoCoMo 32K | `locomo` | 400 | 约 32K tokens |
| LongMemEval 128K | `split_longmemeval_s` | 150 | 约 128K tokens |
| LongMemEval 1M | `split_longmemeval_m` | 150 | 约 1M tokens |

复现分为两个阶段：

1. **MemSifter 排序与 Table 2 检索指标**：本地运行 BGE 预筛选和 MemSifter-4B 排序，计算 NDCG。作者发布的测试文件已经包含 BGE-M3 的 `similarities`，因此无需重新编码。
2. **Table 1 最终回答指标**：把 MemSifter 选出的 top-10 session 交给 Qwen3-30B-A3B-Instruct 或 DeepSeek V3.2，计算回答 F1。这一步需要 OpenAI 兼容的工作 LLM API。

论文地址：<https://arxiv.org/abs/2603.03379>

## 1. 复现流程

```text
全部历史 session
        │
        │ 使用作者数据中预计算的 BGE-M3 similarities
        ▼
按相似度从高到低排列 session
        │
        │ 依次加入完整 session，MemSifter 输入限制约 128K
        ▼
MemSifter-4B Think-and-Rank
        │
        ├── top-10 排序 ──► NDCG@1/NDCG@5（Table 2）
        │
        └── top-10 session ──► 工作 LLM ──► 回答 F1（Table 1）
```

`LongMemEval-1M` 的 1M 表示原始历史规模。它不会把 1M tokens 全部输入 MemSifter。程序会使用所有 session 已有的 BGE 分数进行排序，然后选取累计不超过约 128K tokens 的完整 session 输入 MemSifter。

`--context_limit 128k` 限制的是 MemSifter 的整体输入，包括检索指令、问题、session 标签、日期和历史内容。留给历史内容的空间会略低于 128K。

## 2. 当前工作区的兼容修改

当前工作区已经包含两项复现所需的修改：

1. `utils/ray_gen_utils.py` 同时兼容旧版 Ray 的 `build_llm_processor` 和新版 Ray 的 `build_processor`，并排除 `models/`、`data/` 和 `.git/`，避免 Ray 打包大型文件。
2. `memsifter/ray_chat_infer_async.py` 的 MemSifter 分支严格使用生成排序的前 10 个 session，不再继续用 BGE session 填满剩余上下文。

检查修改是否存在：

```bash
cd /home/hongyshen/MemSifter

rg -n 'build_processor' utils/ray_gen_utils.py
rg -n 'pred_ranking_indexes\[:10\]' memsifter/ray_chat_infer_async.py
```

不要直接执行发布仓库中的 `scripts/infer/session_ranking.sh`。该脚本仍引用不存在的 `memretriever/ray_vllm_infer.py`，并包含本次未下载的其他数据集。使用本文后面的 Python 命令。

## 3. 激活环境并检查依赖

所有命令均从仓库根目录执行：

```bash
cd /home/hongyshen/MemSifter
conda activate memsifter
```

检查主要依赖和 GPU：

```bash
python -c \
  "import ray, vllm, torch; print('ray', ray.__version__); print('vllm', vllm.__version__); print('torch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('GPU count', torch.cuda.device_count())"

nvidia-smi
```

如果服务器使用 R580 或更新驱动，可以使用：

```text
ray 2.58.0
vllm 0.26.0
torch 2.11.0+cu130
```

当前服务器是 NVIDIA 565.57.01 驱动，只支持 CUDA 12.x，不能运行上述 `cu130` 环境。为该服务器创建独立的 CUDA 12.6 推理环境：

```bash
ray stop --force || true

conda create -n memsifter-cu126 python=3.11 -y
conda activate memsifter-cu126
python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.7.0 的 CUDA 12.6 构建
python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126

# 必须使用 vLLM 的 cu126 wheel；PyPI 默认 CUDA wheel 不适合 565 驱动
python -m pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.9.2/vllm-0.9.2+cu126-cp38-abi3-manylinux1_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu126

# 固定 Ray 版本，避免 ray[llm] 再把 vLLM 升级到 cu130 版本
python -m pip install -r requirements-infer-cu126.txt
python -m pip check
```

不要在这个环境中再执行仓库未固定版本的 `pip install -r requirements.txt`。其中的 `ray[llm]` 会解析到当前最新版，并再次安装 CUDA 13.0 的 vLLM/PyTorch。`verl` 是训练依赖，本次测试推理不需要安装。

验证环境：

```bash
python - <<'PY'
import torch, ray, vllm, transformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("ray:", ray.__version__)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
PY
```

预期核心版本：

```text
torch 2.7.0+cu126
torch CUDA 12.6
ray 2.48.0
vllm 0.9.2+cu126
transformers 4.53.3
CUDA available True
```

用于运行推理的终端必须满足 `torch.cuda.is_available() == True`。

## 4. 检查模型文件

检查 MemSifter 两个权重分片：

```bash
ls -lh models/zstanjj/MemSifter-4B-Thinking/model-*.safetensors
```

预期至少包含：

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
```

检查 BGE-M3：

```bash
ls -lh models/bge-m3/pytorch_model.bin
```

本次测试使用作者 Parquet 中已有的 `similarities`，不会重新加载 BGE-M3 计算 embedding。保留 BGE 模型用于后续从原始数据重新生成相似度或测试新数据。

## 5. 检查测试数据

原始文件应位于：

```text
data/locomo/test.parquet
data/split_longmemeval_s/test.parquet
data/split_longmemeval_m/test.parquet
```

检查行数和 `similarities` 字段：

```bash
python - <<'PY'
import pandas as pd

datasets = {
    "locomo": "data/locomo/test.parquet",
    "split_longmemeval_s": "data/split_longmemeval_s/test.parquet",
    "split_longmemeval_m": "data/split_longmemeval_m/test.parquet",
}

for name, path in datasets.items():
    df = pd.read_parquet(path)
    assert "similarities" in df.columns, f"{name} 缺少 similarities"
    print(name, "rows=", len(df), "columns=", len(df.columns))
PY
```

预期：

```text
locomo rows=400
split_longmemeval_s rows=150
split_longmemeval_m rows=150
```

## 6. 准备排序程序的输入路径

`memsifter/ray_vllm_infer.py` 固定读取：

```text
data/results/bge-m3/{dataset}_test_embed.parquet
```

作者数据已经带有相似度，因此建立软链接即可：

```bash
cd /home/hongyshen/MemSifter

mkdir -p data/results/bge-m3

ln -sfn ../../locomo/test.parquet \
  data/results/bge-m3/locomo_test_embed.parquet

ln -sfn ../../split_longmemeval_s/test.parquet \
  data/results/bge-m3/split_longmemeval_s_test_embed.parquet

ln -sfn ../../split_longmemeval_m/test.parquet \
  data/results/bge-m3/split_longmemeval_m_test_embed.parquet

readlink -f data/results/bge-m3/*_test_embed.parquet
```

## 7. 配置 GPU

单卡运行：

```bash
export CUDA_VISIBLE_DEVICES=0
export TP=1
```

两卡张量并行：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export TP=2
```

128K 输入和最长 16K 输出需要较大的 KV cache。如果显存不足，优先增加 GPU 数并同步增大 `TP`。降低 `context_limit` 或 `max_model_len` 可用于排错，但会偏离论文设置。

使用 `--local` 时程序会自行初始化本地 Ray，无需提前执行 `ray start --head`。

## 8. 先运行 10 条 LoCoMo 样本

调试运行只处理前 10 条：

```bash
cd /home/hongyshen/MemSifter
conda activate memsifter

export CUDA_VISIBLE_DEVICES=0
export TP=1

MODEL_PATH="$PWD/models/zstanjj/MemSifter-4B-Thinking"
OUTPUT_DIR="$PWD/data/results"
METHOD="DAPO-GenRank-MemSifter-4B-Thinking"

python -B memsifter/ray_vllm_infer.py \
  --local \
  --debug \
  --prompt_name gen_retrieve_instruct_prompt.yaml \
  --dataset_name locomo \
  --dataset_split test \
  --embed_model bge-m3 \
  --output_dir "$OUTPUT_DIR" \
  --model_path "$MODEL_PATH" \
  --model_name "$METHOD" \
  --override_num 1 \
  --concurrency 1 \
  --tp "$TP" \
  --pp 1 \
  --batch_size 1 \
  --context_limit 128k \
  --filter_col similarities \
  --max_num_batched_tokens 133120 \
  --max_model_len 149504 \
  --max_output_token 16384 \
  --temperature 0.6
```

预期输出：

```text
data/results/DAPO-GenRank-MemSifter-4B-Thinking/locomo_test_ranking.parquet
```

该文件只有 10 行。后续完整运行会覆盖它。

## 9. 完整运行三个测试集

```bash
cd /home/hongyshen/MemSifter
conda activate memsifter

export CUDA_VISIBLE_DEVICES=0
export TP=1

MODEL_PATH="$PWD/models/zstanjj/MemSifter-4B-Thinking"
OUTPUT_DIR="$PWD/data/results"
METHOD="DAPO-GenRank-MemSifter-4B-Thinking"

mkdir -p "$PWD/logs"
set -o pipefail

for DATASET in \
  locomo \
  split_longmemeval_s \
  split_longmemeval_m
do
  LOG_FILE="$PWD/logs/${DATASET}_full_$(date +%Y%m%d_%H%M%S).log"

  python -u -B memsifter/ray_vllm_infer.py \
    --local \
    --prompt_name gen_retrieve_instruct_prompt.yaml \
    --dataset_name "$DATASET" \
    --dataset_split test \
    --embed_model bge-m3 \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --model_name "$METHOD" \
    --override_num 1 \
    --concurrency 1 \
    --tp "$TP" \
    --pp 1 \
    --batch_size 1 \
    --context_limit 128k \
    --filter_col similarities \
    --max_num_batched_tokens 133120 \
    --max_model_len 149504 \
    --max_output_token 16384 \
    --temperature 0.6 \
    2>&1 | tee "$LOG_FILE"

  # 当前数据集运行失败时停止循环
  if [ $? -ne 0 ]; then
    echo "运行失败，日志：$LOG_FILE"
    break
  fi
done
```

输出文件：

```text
data/results/DAPO-GenRank-MemSifter-4B-Thinking/locomo_test_ranking.parquet
data/results/DAPO-GenRank-MemSifter-4B-Thinking/split_longmemeval_s_test_ranking.parquet
data/results/DAPO-GenRank-MemSifter-4B-Thinking/split_longmemeval_m_test_ranking.parquet
```

此脚本没有断点续跑功能。如果某个数据集失败，只重新运行该数据集即可，不必重新运行已经完成的数据集。

## 10. 检查排序输出

```bash
python - <<'PY'
import pandas as pd

method = "DAPO-GenRank-MemSifter-4B-Thinking"
expected = {
    "locomo": 400,
    "split_longmemeval_s": 150,
    "split_longmemeval_m": 150,
}

for dataset, expected_rows in expected.items():
    path = f"data/results/{method}/{dataset}_test_ranking.parquet"
    df = pd.read_parquet(path)
    assert len(df) == expected_rows, (dataset, len(df), expected_rows)
    empty = sum(len(x) == 0 for x in df["haystack_session_gen_rankings"])
    lengths = df["haystack_session_gen_rankings"].map(len)
    print(
        dataset,
        "rows=", len(df),
        "empty_rankings=", empty,
        "mean_ranking_len=", round(lengths.mean(), 2),
    )
PY
```

理想情况下 `empty_rankings=0`，大部分排序长度应为 10。若大量为空，检查模型输出是否包含完整的 `</think><ranking>...</ranking>`。

## 11. 复现 Table 2：检索 NDCG

```bash
python - <<'PY'
import pandas as pd
from utils.eval_utils import eval_ranking_data

method = "DAPO-GenRank-MemSifter-4B-Thinking"
datasets = {
    "LoCoMo-32K": "locomo",
    "LongMemEval-128K": "split_longmemeval_s",
    "LongMemEval-1M": "split_longmemeval_m",
}

for label, dataset in datasets.items():
    path = f"data/results/{method}/{dataset}_test_ranking.parquet"
    df = pd.read_parquet(path)
    result = eval_ranking_data(df, verbose=False)
    print(
        f"{label:20s} "
        f"NDCG@1={result[1]['ndcg'] * 100:.2f}  "
        f"NDCG@5={result[5]['ndcg'] * 100:.2f}  "
        f"NDCG@10={result[10]['ndcg'] * 100:.2f}"
    )
PY
```

论文 Table 2 中 MemSifter 的参考值：

| 数据集 | NDCG@1 | NDCG@5 |
| --- | ---: | ---: |
| LoCoMo 32K | 70.00 | 78.11 |
| LongMemEval 128K | 67.33 | 89.67 |
| LongMemEval 1M | 60.00 | 75.63 |

完成本节即可复现 MemSifter 本身的检索效果，不需要工作 LLM API。

## 12. 复现 Table 1：工作 LLM 回答 F1

### 12.1 前提

下载的 MemSifter Qwen3-4B 是排序代理，不负责最终回答。Table 1 还需要以下工作 LLM 之一：

- DeepSeek V3.2
- Qwen3-30B-A3B-Instruct

准备 OpenAI 兼容接口：

```bash
export API_KEY="替换为实际密钥"
export BASE_URL="https://替换为实际服务地址/v1"
```

不要把 API 密钥写入本文档或提交到 Git。

`--model_path` 在回答阶段只用于加载工作 LLM 对应的 tokenizer，可以是本地 tokenizer 目录，不要求下载完整 30B 模型权重。严格复现时应使用与远程模型版本一致的 tokenizer。

如果缺少 NLTK 分词资源：

```bash
python -m nltk.downloader punkt punkt_tab
```

### 12.2 使用 Qwen3-30B-A3B-Instruct

将模型 ID 和 tokenizer 路径改成 API 服务商实际使用的精确版本：

```bash
cd /home/hongyshen/MemSifter
conda activate memsifter

METHOD="DAPO-GenRank-MemSifter-4B-Thinking"
WORKING_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
WORKING_TOKENIZER="/path/to/Qwen3-30B-A3B-Instruct-tokenizer"

for DATASET in \
  locomo \
  split_longmemeval_s \
  split_longmemeval_m
do
  python -B memsifter/ray_chat_infer_async.py \
    --input_dir data/results \
    --output_dir data/results/chat \
    --dataset_name "$DATASET" \
    --dataset_split test \
    --method "$METHOD" \
    --model_name "$WORKING_MODEL" \
    --model_path "$WORKING_TOKENIZER" \
    --max_input_tokens 131072 \
    --max_output_tokens 4096 \
    --temperature 0.6 \
    --max_concurrent 8 \
    --infer_batch_size 32
done
```

### 12.3 使用 DeepSeek V3.2

使用相同命令，仅替换：

```bash
WORKING_MODEL="服务商提供的 DeepSeek-V3.2 精确模型 ID"
WORKING_TOKENIZER="/path/to/compatible-deepseek-tokenizer"
```

如果 API 出现限流，降低：

```bash
--max_concurrent 2 \
--infer_batch_size 16
```

### 12.4 Table 1 参考值

脚本在每个数据集完成后打印 `exact_match`、`f1_score` 和 `bleu_score`。输出 `f1_score=0.4639` 对应论文表格中的 `46.39`。

| 工作 LLM | LoCoMo 32K | LongMemEval 128K | LongMemEval 1M |
| --- | ---: | ---: | ---: |
| DeepSeek V3.2 | 41.79 | 35.38 | 33.32 |
| Qwen3-30B-A3B-Instruct | 46.39 | 47.26 | 49.58 |

远程工作 LLM 的模型版本、服务商实现和采样随机性都可能影响 Table 1。复现记录中应保存 API 模型的精确 ID、服务商、日期、temperature 和输出文件。

## 13. 推理与训练的区别

本手册执行的是推理：

```text
BGE 预筛选 → MemSifter 一次生成 top-10 → 计算检索指标或调用一次工作 LLM
```

它不会执行训练阶段的 task reward。训练时，每个问题会采样多条完整的 top-10 排序，并使用工作 LLM 分别评估同一排序的 `top-{0,1,2,3,5,8,10}` 前缀，根据回答质量计算位置敏感的奖励，再用 DAPO/GRPO 更新 MemSifter 参数。

## 14. 常见问题

### CUDA 不可见

症状：

```text
torch.cuda.is_available() == False
Triton ... 0 active driver(s)
```

需要切换到能访问 NVIDIA 驱动和 GPU 的终端或计算节点，并确认 `CUDA_VISIBLE_DEVICES` 设置正确。

### Ray 的 processor API 导入失败

确认当前工作区使用了新旧 API 兼容导入：

```python
try:
    from ray.data.llm import build_processor
except ImportError:
    from ray.data.llm import build_llm_processor as build_processor
```

### vLLM 显存不足

先降低并发并保持：

```text
--concurrency 1
--batch_size 1
```

仍然不足时使用更多 GPU 并增大 `--tp`。为了保持论文设置，不应把正式结果建立在降低 `--context_limit` 的运行上。

### `CUDA error: uncorrectable ECC error encountered`

2026-09-30 15:19 的运行在 vLLM 初始化 Qwen3 的 RoPE 缓存时出现此错误，随后发生 `WorkerProc initialization failed` 和 `Engine core initialization failed`。此前 `nvidia-smi` 已显示 `Volatile Uncorr. ECC: 3098`。这表明 GPU 报告了不可纠正的 ECC 错误，需要检查设备状态；仅凭该报错不能认定 GPU 永久损坏。

在 GPU 所在主机上保存诊断信息：

```bash
mkdir -p logs
nvidia-smi -q -i 0 > logs/gpu0_health.txt 2>&1
nvidia-smi -q -i 0 -d ECC
nvidia-smi -q -i 0 -d ROW_REMAPPER
# 若没有内核日志读取权限，请由管理员执行。
journalctl -k --since '2026-09-30 15:15:00' --until '2026-09-30 15:25:00' \
  | rg -i 'NVRM|Xid|ECC'
```

由管理员根据 Xid 和 GPU Recovery Action 决定恢复操作。Xid 48、94、95 的恢复要求不同；需要 GPU reset 时，应先确认该 GPU 上所有任务和服务都已停止，再由管理员执行 `sudo nvidia-smi --gpu-reset -i 0`，或按要求重启节点。如果恢复后同类错误仍反复出现，应换用健康 GPU，并进行 NVIDIA/硬件厂商诊断。

恢复后先验证基础 CUDA 运算，再重新运行 MemSifter：

```bash
python -B - <<'PY'
import torch
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('CUDA matmul OK:', y.shape)
PY
```

参考 [NVIDIA Xid 错误目录](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)。

### `TokenizeUDF` 报 `There is no current event loop`

在 Ray 2.48 + Python 3.11 + uvloop 环境中，Ray worker 会安装 uvloop，但旧版 Ray Data 初始化异步 UDF 时假定当前线程已有事件循环。这会导致 `TokenizeUDF` 创建失败，并出现：

```text
RuntimeError: no running event loop
RuntimeError: There is no current event loop in thread 'MainThread'.
ray.exceptions.ActorDiedError
```

当前工作区通过 worker 启动钩子 `utils.ray_runtime.setup_worker_event_loop` 显式创建缺失的循环，保留 uvloop。`start_local_ray()` 和 `configs/runtime_env.yaml` 已配置此钩子；不需要卸载 uvloop 或更改模型推理参数。

清理旧的本地 Ray worker，再用原命令重新运行：

```bash
ray stop --force
```

如果异常堆栈指向 `ds.to_pandas()`，那是因为它触发了整条惰性数据处理流水线，不代表错误发生在 Parquet 写入阶段。应继续查看堆栈里的具体 worker，例如 `MapBatches(TokenizeUDF)`。

### 排序文件行数只有 10

这是先前 `--debug` 运行产生的文件。移除 `--debug` 后重新运行完整数据集。

### `AggregateNumRows` 和整体进度一直是 `0/1`

这表示程序正在执行 `ds.count()`。`AggregateNumRows` 只输出一条最终计数结果，所以它和整体进度会在计数完成之前保持 `0/1`。这里的 `1` 是一个计数结果，不是一个测试样本；也不代表模型推理进度。

旧代码在 `--debug` 截取 10 条之前，先对全部 400 条历史执行 JSON 解析并计数，可能启动大量 Ray worker。当前代码改为从 Parquet 元数据读取总条数，先截取 debug 数据，再执行解析。`--override_num` 控制读取分块数及 JSON 解析并发；单卡调试保持 `--override_num 1`。

当前代码保存结果后直接使用 DataFrame 的行数记录完成情况，避免继续调用 `ds.take_all()`、`ds.count()` 再次执行模型推理。

更新代码后，先按 Ctrl+C 停止原程序，执行 `ray stop --force` 清理该程序启动的本地 Ray，再用原命令重新运行。启动时应看到 `Input contains 400 entries` 和 `Debug mode enabled, only processing 10 entries`。

### Ray 报输入 Parquet 不存在

如果 Ray worker 报错：

```text
FileNotFoundError: data/results/bge-m3/locomo_test_embed.parquet
```

通常是因为 Ray worker 在 runtime 工作目录中解析了相对路径。确认软链接有效，并向程序传递绝对输出目录：

```bash
readlink -f data/results/bge-m3/locomo_test_embed.parquet
OUTPUT_DIR="$PWD/data/results"
```

排序命令使用：

```bash
--output_dir "$OUTPUT_DIR"
```

### Table 1 指标与论文有差异

检查：

- 是否严格使用 MemSifter top-10，而非继续填充 BGE session；
- 工作 LLM 是否是论文对应的精确版本；
- API 服务是否修改了默认 system prompt、thinking 模式或采样参数；
- F1 是否乘以 100 后再与论文表格比较；
- 是否存在失败、空回答或被限流的请求。

## 15. 推荐的复现记录

每次正式运行至少记录：

```bash
git rev-parse HEAD
python --version
python -c "import ray, vllm, torch; print(ray.__version__, vllm.__version__, torch.__version__)"
nvidia-smi
```

同时保留：

- 完整命令行；
- `CUDA_VISIBLE_DEVICES` 和 `TP`；
- 三个 `*_ranking.parquet`；
- Table 2 指标输出；
- Table 1 使用的工作 LLM 精确 ID 和三个 `*_chat.parquet`；
- API 请求失败数量和运行日期。
