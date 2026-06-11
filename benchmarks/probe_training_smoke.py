"""Tier-1 training-integration smoke for the M10 promotion gate (gate 6).

Head-only synthetic training with no Hub downloads: a fixed dataset of
(query, document) hidden-state pairs, an in-batch contrastive cross-entropy
loss plus a FLOPS-style sparsity regularizer, and AdamW for ``--steps``
optimizer steps. Runs the hybrid and optimized backends from identical fp32
master parameters under fp16 AMP (autocast + GradScaler) and bf16 autocast.

Gate assertions per mode:
  - every per-step loss is finite for both backends;
  - bf16 (no GradScaler): every sampled gradient is finite;
  - fp16 (GradScaler): scaler-skipped steps are bounded (early overflow while
    the scale calibrates is expected AMP behavior) and none occur in the
    final quarter of training;
  - the final loss is lower than the initial loss (the head overfits the
    fixed synthetic set, so training demonstrably works);
  - the hybrid and optimized loss curves agree within tolerance (same seed,
    same data, same init).

Exits non-zero with the failing assertions listed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def flops_regularizer(reps: torch.Tensor) -> torch.Tensor:
    """SPLADE FLOPS regularizer: sum over vocab of squared mean activation."""

    return (reps.float().mean(dim=0) ** 2).sum()


def make_dataset(
    *,
    pairs: int,
    seq_len: int,
    dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    queries = torch.randn((pairs, seq_len, dim), device="cuda", generator=generator) * 0.05
    documents = torch.randn((pairs, seq_len, dim), device="cuda", generator=generator) * 0.05
    query_mask = (torch.rand((pairs, seq_len), device="cuda", generator=generator) > 0.25).to(
        torch.int32
    )
    document_mask = (torch.rand((pairs, seq_len), device="cuda", generator=generator) > 0.25).to(
        torch.int32
    )
    return queries, documents, query_mask, document_mask


def run_training(
    backend: str,
    *,
    autocast_dtype: torch.dtype,
    use_scaler: bool,
    steps: int,
    batch: int,
    dim: int,
    vocab: int,
    dataset,
    init_state: dict,
    lr: float,
    temperature: float,
    lambda_flops: float,
    grad_check_every: int,
) -> list[float]:
    import sparton.sparton_kernel as sk

    queries, documents, query_mask, document_mask = dataset
    pairs = queries.shape[0]

    head = sk.SpartonHead(vocab, dim, use_bias=True, backend=backend).to("cuda")
    head.load_state_dict(init_state)
    head = head.float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    losses: list[float] = []
    skipped_steps: list[int] = []
    for step in range(steps):
        start = (step * batch) % pairs
        sel = torch.arange(start, start + batch, device="cuda") % pairs
        hidden_q = queries[sel]
        hidden_d = documents[sel]
        mask_q = query_mask[sel]
        mask_d = document_mask[sel]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            reps_q = head(hidden_q, mask_q)
            reps_d = head(hidden_d, mask_d)
        # Cosine similarity keeps the logit scale sane for in-batch CE.
        sim = (
            torch.nn.functional.normalize(reps_q.float(), dim=-1)
            @ torch.nn.functional.normalize(reps_d.float(), dim=-1).T
            / temperature
        )
        labels = torch.arange(batch, device="cuda")
        ramp = min(1.0, (step + 1) / max(1, steps // 4))
        loss = (
            torch.nn.functional.cross_entropy(sim, labels)
            + lambda_flops * ramp * (flops_regularizer(reps_q) + flops_regularizer(reps_d))
        )
        assert torch.isfinite(loss), f"{backend} step {step}: non-finite loss"

        scaler.scale(loss).backward()
        if not use_scaler and step % grad_check_every == 0:
            # No scaler: gradients must be finite, no excuses.
            for name, param in head.named_parameters():
                assert param.grad is not None, f"{backend} step {step}: no grad for {name}"
                assert bool(torch.isfinite(param.grad).all()), (
                    f"{backend} step {step}: non-finite grad for {name}"
                )
        scale_before = scaler.get_scale() if use_scaler else None
        scaler.step(optimizer)
        scaler.update()
        if use_scaler and scaler.get_scale() < scale_before:
            skipped_steps.append(step)  # standard AMP overflow handling
        losses.append(loss.item())

    if use_scaler:
        assert len(skipped_steps) <= max(2, steps // 4), (
            f"{backend}: GradScaler skipped {len(skipped_steps)}/{steps} steps"
        )
        last_quarter = steps - max(1, steps // 4)
        assert all(step < last_quarter for step in skipped_steps), (
            f"{backend}: GradScaler still skipping in the final quarter "
            f"({[s for s in skipped_steps if s >= last_quarter]})"
        )
    return losses


def run_mode(
    *,
    mode: str,
    steps: int,
    batch: int,
    seq_len: int,
    dim: int,
    vocab: int,
    seed: int,
    lr: float,
    temperature: float,
    lambda_flops: float,
    parity_tolerance: float,
) -> list[str]:
    import sparton.sparton_kernel as sk

    autocast_dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    use_scaler = mode == "fp16"

    torch.manual_seed(seed)
    reference_head = sk.SpartonHead(vocab, dim, use_bias=True, backend="hybrid").to("cuda")
    init_state = {key: value.detach().clone() for key, value in reference_head.state_dict().items()}
    dataset = make_dataset(pairs=64, seq_len=seq_len, dim=dim, seed=seed)

    curves: dict[str, list[float]] = {}
    for backend in ("hybrid", "optimized"):
        curves[backend] = run_training(
            backend,
            autocast_dtype=autocast_dtype,
            use_scaler=use_scaler,
            steps=steps,
            batch=batch,
            dim=dim,
            vocab=vocab,
            dataset=dataset,
            init_state=init_state,
            lr=lr,
            temperature=temperature,
            lambda_flops=lambda_flops,
            grad_check_every=max(1, steps // 10),
        )

    failures: list[str] = []
    hybrid = curves["hybrid"]
    optimized = curves["optimized"]
    mean_rel = sum(
        abs(h - o) / max(abs(h), 1e-6) for h, o in zip(hybrid, optimized)
    ) / len(hybrid)
    final_rel = abs(hybrid[-1] - optimized[-1]) / max(abs(hybrid[-1]), 1e-6)
    print(
        f"{mode}: hybrid loss {hybrid[0]:.4f} -> {hybrid[-1]:.4f}, "
        f"optimized loss {optimized[0]:.4f} -> {optimized[-1]:.4f}, "
        f"mean rel diff {mean_rel:.4f}, final rel diff {final_rel:.4f}",
        flush=True,
    )
    for backend, losses in curves.items():
        if not losses[-1] < losses[0]:
            failures.append(
                f"{mode}/{backend}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"
            )
    if final_rel > parity_tolerance:
        failures.append(
            f"{mode}: final loss parity {final_rel:.4f} exceeds {parity_tolerance}"
        )
    if mean_rel > parity_tolerance:
        failures.append(
            f"{mode}: mean loss parity {mean_rel:.4f} exceeds {parity_tolerance}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--vocab", type=int, default=30522)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lambda-flops", type=float, default=1e-3)
    parser.add_argument("--parity-tolerance", type=float, default=0.05)
    parser.add_argument("--modes", default="fp16,bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("probe_training_smoke.py requires CUDA")
    from sparton._gluon_runtime import is_gluon_backend_available

    available, reason = is_gluon_backend_available()
    if not available:
        raise RuntimeError(
            "probe_training_smoke.py requires the optimized Gluon backend "
            f"(CUDA sm_80+ and importable triton.experimental.gluon): {reason}"
        )

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if "bf16" in modes and not torch.cuda.is_bf16_supported():
        modes.remove("bf16")
        print("bf16 unsupported on this device; skipping that mode", flush=True)

    print(
        f"tier-1 training smoke: steps={args.steps} batch={args.batch} "
        f"S={args.seq_len} D={args.dim} V={args.vocab} modes={modes} "
        f"parity tolerance {args.parity_tolerance}",
        flush=True,
    )
    failures: list[str] = []
    for mode in modes:
        failures.extend(
            run_mode(
                mode=mode,
                steps=args.steps,
                batch=args.batch,
                seq_len=args.seq_len,
                dim=args.dim,
                vocab=args.vocab,
                seed=args.seed,
                lr=args.lr,
                temperature=args.temperature,
                lambda_flops=args.lambda_flops,
                parity_tolerance=args.parity_tolerance,
            )
        )

    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("tier-1 training smoke: passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
