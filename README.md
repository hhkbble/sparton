# Sparton

Fast and memory-efficient Triton kernel for learned sparse retrieval.

Sparton replaces the MLM projection head in SPLADE-style models with a CUDA-only fused projection head (`SpartonHead`) that avoids materializing the full `[B, S, V]` logits tensor.

## Installation

Requirements:

- Python >= 3.10
- CUDA-capable PyTorch runtime
- `torch>=2.7.1`
- `triton>=3.3.1`

```bash
pip install "torch>=2.7.1" "triton>=3.3.1"

# Install the sparton package from a local checkout
cd /path/to/sparton
pip install -e .
```

## Quick Start

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sparton import SpartonHead

# Load a SPLADE model
llm = AutoModelForMaskedLM.from_pretrained("naver/splade-v3").cuda()
tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")

# Create SpartonHead and tie weights to the pretrained decoder
decoder = llm.cls.predictions.decoder
head = SpartonHead(decoder.out_features, decoder.in_features, use_bias=True).cuda()
head.tie_weights(decoder)

# Encode
inputs = tokenizer("What is sparse retrieval?", return_tensors="pt").to("cuda")
hidden = llm.bert(**inputs).last_hidden_state
hidden = llm.cls.predictions.transform(hidden)
sparse_reps = head(hidden, inputs["attention_mask"])  # [1, vocab_size]
```

See [model.py](training/model.py) for a full example integrating Sparton into a SPLADE model.

### Backend Selection

`SpartonHead` defaults to the `optimized` Gluon fused-forward backend wherever
it is available (CUDA sm_80+ plus a Triton with importable
`triton.experimental.gluon`; validated with Triton 3.6.0). On platforms where
it is unavailable, the default resolves to the `hybrid` backend with a
one-time `RuntimeWarning`. This availability-adaptive behavior applies only
to the *default*: an explicitly selected backend never falls back and raises
with the reason when unavailable.

The `optimized` forward is a single TMA + `mma_v2` Gluon kernel with bounded
autotune (a fixed production policy universe pruned at launch from the actual
CUDA device profile and problem shape). It was promoted to the default after
the M10 gates in
[docs/sparton_remaining_work_design_v2.md](docs/sparton_remaining_work_design_v2.md):
it is faster than hybrid on every measured shape, uses output-only memory
(~14x less peak than hybrid on the dev shape), and matches the reference on
the full correctness matrix; evidence in
[docs/sparton_milestone10_promotion_memo.md](docs/sparton_milestone10_promotion_memo.md).

To pin a backend explicitly (rollback path), pass the constructor argument or
set the environment variable before importing `sparton`:

```python
head = SpartonHead(vocab_size, hidden_dim, use_bias=True, backend="hybrid")
```

```bash
SPARTON_BACKEND=hybrid python train.py ...
```

Available backends: `optimized` (default where available), `hybrid` (the
previous default — compiled tiled matmul + Triton reduction; the
compatibility path), and `naive` (a Triton-only `tl.dot` debug baseline).
All three share the same Triton backward and the same numerics contract.

Mixed precision: the forward wrappers mirror `torch.autocast` semantics —
under an active CUDA autocast region, fp32 master parameters are cast to the
autocast dtype (fp16/bf16) like any autocast-aware matmul, so standard AMP
training works with every backend.

Index semantics across backends: the returned index for a vocabulary entry is
meaningful only where its score is greater than zero (zero-baseline policy).
Within one backend, ties resolve to the lowest sequence index. Across
backends, `naive` and `optimized` accumulate logits in fp32 while `hybrid`
produces input-dtype logits, so at near-ties (logit gaps within input-dtype
rounding) different backends may legitimately return different winners. The
guaranteed contract, enforced by the test suite, is that the chosen position's
masked logit stays within score tolerance of the true maximum.

## How It Works

Standard SPLADE computes sparse representations in multiple steps:

```
hidden [B,S,D] → matmul with decoder [D,V] → logits [B,S,V] → mask → ReLU → log1p → max → reps [B,V]
```

The `[B,S,V]` logits tensor (e.g. 512 x 512 x 30522 = ~16GB in fp16) is the memory bottleneck.

The current Sparton implementation is a hybrid path: it uses compiled PyTorch/TorchInductor matmul over vocabulary tiles, then a Triton reduction kernel for masking, sequence max/argmax, ReLU, and log1p. It materializes per-tile `[B,S,V_tile]` logits, but not the full `[B,S,V]` tensor:

```
hidden [B,S,D] → tiled matmul → Triton reduction → reps [B,V]
```

Backward uses the saved max scores and indices to accumulate gradients for hidden states, decoder weights, and optional bias. Since M11 it is a segmented design shared by all backends: decoder-weight and bias gradients are written by exclusive owners (no atomics — exactly deterministic), and hidden-state gradients are sorted by destination row and reduced segment-wise before a handful of partial-sum atomics, which makes the backward 1.35–2.4x faster on real tokenized batches (where many vocabulary entries share one argmax position) and slightly faster on uniform synthetic inputs. The mask contract is the standard binary `{0,1}` tokenizer `attention_mask`, under which gradients are exact; non-binary mask values weight logits in the forward as an implementation property but are outside the contract and are not differentiated by the backward.

## Training

A sample training script is provided to benchmark Sparton against PyTorch and `torch.compile` baselines on a real training workload.

### Setup

```bash
pip install -U transformers datasets accelerate
```

### Usage

Run from the `training/` directory:

```bash
cd training

# Sparton head
python train.py \
    --model_name_or_path FacebookAI/xlm-roberta-base \
    --head sparton \
    --output_dir ../output/splade-sparton \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --fp16

# PyTorch implementation
python train.py --head torch ...

# torch.compile implementation
python train.py --head compiled ...
```

Sparton is faster than PyTorch and `torch.compile` at the same batch size, and its memory efficiency also allows training with larger batch sizes, longer sequence lengths, and larger vocabularies. Feel free to experiment with different `--per_device_train_batch_size` values to find the best throughput for your GPU.

> **Note:** At the beginning of training, Sparton may appear slow due to Triton's autotuning of kernel configurations. Once autotuning completes (after the first few steps), the kernel runs at full speed.

The three `--head` modes use identical model weights and produce the same gradients — only the head implementation differs:

| Head | Description |
|------|-------------|
| `sparton` | Hybrid Sparton kernel (`SpartonHead`) |
| `torch` | Standard PyTorch ops |
| `compiled` | `torch.compile`'d PyTorch head |


### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--head` | `torch` | `torch`, `compiled`, or `sparton` |
| `--model_name_or_path` | `FacebookAI/xlm-roberta-base` | Base MLM model |
| `--languages` | `de,es,fr` | Dataset language subsets |
| `--temperature` | `1.0` | InfoNCE temperature |
| `--lambda_l1` | `1e-4` | L1 regularization weight |
| `--reg_warmup_steps` | `10000` | Regularization warmup steps |

## Testing

The test suite is configured in `pyproject.toml` for pytest 9.

```bash
python -m pip install "pytest>=9.0,<10"
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas python -m pytest -v
```

CUDA kernel tests require a CUDA device. In this workspace, Triton/TorchInductor probes also require the `TRITON_PTXAS_PATH` shown above.

## Project Notes

Recent repository changes are tracked in [CHANGELOG.md](CHANGELOG.md). Design and audit notes for the Gluon/backend refactor live in [docs/](docs/), including [sparton_gluon_current_platform_design.md](docs/sparton_gluon_current_platform_design.md), [sparton_gluon_design_review.md](docs/sparton_gluon_design_review.md), and [sparton_milestone2_bias_none_backward_memo.md](docs/sparton_milestone2_bias_none_backward_memo.md).

## Citation

```bibtex
@article{sparton2026,
  title={Sparton: Fast and Memory-Efficient Triton Kernel for Learned Sparse Retrieval},
  author={Thong Nguyen, Cosimo Rulli, Franco Maria Nardini, Rossano Venturini, Andrew Yates},
  year={2026},
  url={https://arxiv.org/pdf/2603.25011}
}
```
