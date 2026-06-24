"""Capture real Sparton index distributions to a .pt bundle (M11 T2).

Loads the cached tier-2 stack (`FacebookAI/xlm-roberta-base` through
``training/model.py`` with ``head="sparton"``, swim-ir batches through
``training/train.py``'s collator), optionally fine-tunes for ``--train-steps``
optimizer steps (default 150 — the M10-validated tier-2 recipe: bf16 Trainer
AMP, per-device batch 16), then runs no-grad forward passes over real
tokenized batches and records, per batch and per side (query, document):

  ``{side, hidden_shape (B, S, D), max_scores fp16 [B, V], max_idx int16
  [B, V], mask uint8 [B, S]}``

plus tokenizer/model metadata and distribution stats (active fraction =
share of ``scores > 0``, mask density, per-row index-collision summary).
``hidden``/``embed`` values are NOT stored: the backward's memory-access
pattern depends only on ``max_scores``'s sparsity pattern, ``max_idx``, and
the shapes, so ``bench_backward.py`` synthesizes operand values at the
recorded shapes. Capture-to-disk exists so backward benchmarking never pays
a backbone forward per measurement and runs reproduce across sessions
(DEVELOPMENT.md M11 §4, the M11-T2 capture).

Bundles default to ``tests/data/bundles/swimir_de_steps{N}.pt`` (gitignored
— large generated data per ``tests/data/README.md``). If the output file
already exists the script **reuses it**: it prints the stored summary and
exits 0 without touching the GPU, because regenerated bundles contain
*different records* (training nondeterminism) and silently replacing the
file would break comparability with every transcript measured against it.
Pass ``--force`` to regenerate deliberately.

``--lambda-l1`` / ``--lambda-flops`` / ``--reg-warmup-steps`` (M13-T0) pass
through to ``LSRTrainingArguments`` so a shortened-warmup fine-tune can probe
the sparse regime (``f << 1``) without the 10000-step default warmup; unset,
the fine-tune recipe is byte-identical to the M11 captures. The values used
are recorded in the bundle metadata.

Indices are recovered by calling the head's bound forward wrapper directly —
``SpartonHead.forward`` discards them. Capture runs under
``torch.autocast("cuda", bfloat16)`` to mirror the tier-2 training regime.

Use ``--quick`` (2 batches, no fine-tune) during development. Exits non-zero
listing every failed capture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TRAINING = REPO_ROOT / "training"
for _path in (SRC, TRAINING):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def compute_record_stats(
    scores: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Distribution stats that feed the M11 analytic traffic model."""

    active = scores.float() > 0
    active_fraction = active.float().mean().item()
    mask_density = mask.float().mean().item()
    top1_shares = []
    for row in range(scores.shape[0]):
        row_active = active[row]
        count = int(row_active.sum().item())
        if count == 0:
            continue
        winners = idx[row][row_active]
        top1 = int(torch.bincount(winners).max().item())
        top1_shares.append(top1 / count)
    top1_share = float(sum(top1_shares) / len(top1_shares)) if top1_shares else 0.0
    return {
        "active_fraction": active_fraction,
        "mask_density": mask_density,
        "idx_top1_share": top1_share,
    }


def capture_records(model, collator, dataset, *, num_batches: int, batch_size: int):
    head = model.projection.sparton_head
    records = []
    failures: list[tuple[str, str]] = []
    for batch_index in range(num_batches):
        start = batch_index * batch_size
        features = [dataset[int(i)] for i in range(start, start + batch_size)]
        batch = collator(features)
        for side in ("query", "doc"):
            label = f"batch{batch_index}-{side}"
            try:
                input_ids = batch[f"{side}_input_ids"].to("cuda")
                attention_mask = batch[f"{side}_attention_mask"].to("cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.backbone(input_ids, attention_mask).last_hidden_state
                    hidden = model.projection.transform(hidden)
                    scores, idx = head._forward_op(
                        hidden, head.weight, head.bias, attention_mask
                    )
                seq_len = hidden.shape[1]
                if seq_len >= 32768:
                    raise ValueError(f"S={seq_len} overflows int16 max_idx storage")
                records.append(
                    {
                        "side": side,
                        "hidden_shape": tuple(hidden.shape),
                        "max_scores": scores.to(device="cpu", dtype=torch.float16),
                        "max_idx": idx.to(device="cpu", dtype=torch.int16),
                        "mask": attention_mask.to(device="cpu", dtype=torch.uint8),
                        "stats": compute_record_stats(scores, idx, attention_mask),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — gate script reports and continues
                failures.append((label, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL {label}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    return records, failures


def maybe_finetune(model, tokenizer, dataset, args) -> None:
    """150-step tier-2 recipe from training/train.py (M10 gate 6 tier 2)."""

    from train import (
        ContrastiveCollator,
        LSRDataArguments,
        LSRModelArguments,
        LSRTrainer,
        LSRTrainingArguments,
    )

    # Regularizer overrides are forwarded only when set, so the default
    # recipe stays byte-identical to the M11 captures (M13-T0 sparse probe).
    reg_overrides = {
        key: value
        for key, value in (
            ("lambda_l1", args.lambda_l1),
            ("lambda_flops", args.lambda_flops),
            ("reg_warmup_steps", args.reg_warmup_steps),
        )
        if value is not None
    }
    training_args = LSRTrainingArguments(
        output_dir=args.work_dir,
        max_steps=args.train_steps,
        per_device_train_batch_size=args.batch_size,
        bf16=True,
        seed=args.seed,
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        **reg_overrides,
    )
    trainer = LSRTrainer(
        model_args=LSRModelArguments(model_name_or_path=args.model, head="sparton"),
        data_args=LSRDataArguments(dataset_name=args.dataset, languages=args.languages),
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ContrastiveCollator(
            tokenizer=tokenizer,
            query_max_length=args.query_max_length,
            document_max_length=args.document_max_length,
        ),
    )
    trainer.train()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None,
                        help="bundle output path (default: "
                             "tests/data/bundles/swimir_de_steps{train-steps}.pt)")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if the output bundle exists "
                             "(regenerated bundles contain different records)")
    parser.add_argument("--model", type=str, default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--dataset", type=str, default="nthakur/swim-ir-cross-lingual")
    parser.add_argument("--languages", type=str, default="de")
    parser.add_argument("--train-steps", type=int, default=150,
                        help="fine-tune steps before capture (0 = untrained)")
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--document-max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-l1", type=float, default=None,
                        help="override LSRTrainingArguments.lambda_l1 (default: trainer default)")
    parser.add_argument("--lambda-flops", type=float, default=None,
                        help="override LSRTrainingArguments.lambda_flops (default: trainer default)")
    parser.add_argument("--reg-warmup-steps", type=int, default=None,
                        help="override LSRTrainingArguments.reg_warmup_steps (default: trainer default)")
    parser.add_argument("--work-dir", type=str,
                        default=str(REPO_ROOT / "tests" / "data" / "bundles" / "work"))
    parser.add_argument("--quick", action="store_true",
                        help="2 batches, no fine-tune")
    args = parser.parse_args()
    if args.quick:
        args.num_batches = 2
        args.train_steps = 0
    if args.out is None:
        args.out = str(REPO_ROOT / "tests" / "data" / "bundles"
                       / f"swimir_de_steps{args.train_steps}.pt")

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        bundle = torch.load(out_path, weights_only=False)
        records = bundle.get("records", [])
        mean_active = (sum(r["stats"]["active_fraction"] for r in records)
                       / len(records)) if records else float("nan")
        print(
            f"capture summary: reused existing bundle ({len(records)} records, "
            f"train_steps={bundle.get('train_steps')}, "
            f"backend={bundle.get('backend')}) -> {out_path} | "
            f"mean active {mean_active:.4f} | pass --force to regenerate "
            f"(regenerated bundles contain different records)",
            flush=True,
        )
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("capture_index_distributions.py requires CUDA")
    try:
        from transformers import AutoTokenizer, set_seed
        from model import SpladeModel
        from train import ContrastiveCollator, load_swim_ir_dataset
    except ImportError as exc:
        raise RuntimeError(
            "capture_index_distributions.py requires the tier-2 training stack "
            f"(transformers, datasets; cached since M10): {exc}"
        ) from exc

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = SpladeModel(model_name_or_path=args.model, head="sparton")
    model.to("cuda")
    dataset = load_swim_ir_dataset(args.dataset, args.languages)

    if args.train_steps > 0:
        maybe_finetune(model, tokenizer, dataset, args)

    model.eval()
    collator = ContrastiveCollator(
        tokenizer=tokenizer,
        query_max_length=args.query_max_length,
        document_max_length=args.document_max_length,
    )
    records, failures = capture_records(
        model, collator, dataset,
        num_batches=args.num_batches, batch_size=args.batch_size,
    )

    head = model.projection.sparton_head
    bundle = {
        "version": 1,
        "model": args.model,
        "backend": head.backend,
        "dataset": args.dataset,
        "languages": args.languages,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "lambda_l1": args.lambda_l1,
        "lambda_flops": args.lambda_flops,
        "reg_warmup_steps": args.reg_warmup_steps,
        "tokenizer": args.model,
        "query_max_length": args.query_max_length,
        "document_max_length": args.document_max_length,
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)

    if records:
        mean_active = sum(r["stats"]["active_fraction"] for r in records) / len(records)
        mean_density = sum(r["stats"]["mask_density"] for r in records) / len(records)
        mean_top1 = sum(r["stats"]["idx_top1_share"] for r in records) / len(records)
    else:
        mean_active = mean_density = mean_top1 = float("nan")
    print(
        f"capture summary: {len(records)} records "
        f"({args.num_batches} batches x {args.batch_size}, train_steps={args.train_steps}, "
        f"backend={head.backend}) -> {out_path} | "
        f"mean active {mean_active:.4f}, mean mask density {mean_density:.4f}, "
        f"mean idx top1 share {mean_top1:.4f}",
        flush=True,
    )
    if failures:
        print(f"{len(failures)} FAILURES:")
        for label, message in failures:
            print(f"  {label}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
