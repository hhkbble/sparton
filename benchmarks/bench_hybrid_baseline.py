"""Hybrid-path baseline on the SPLADE dev shape.

Measures, on B=32 S=128 D=768 V=30522 fp16:
  - fused_sparton_fwd_op latency (bias / no-bias), after autotune warmup
  - full-V cuBLAS GEMM latency (the GEMM lower bound for any fused forward)
  - peak extra device memory during one hybrid forward
  - forward+backward latency for the bias case

Needs PYTHONPATH=src and the hardened env prefix. Recorded baselines:
docs/sparton_gluon_remaining_work_design.md §3.2.
"""

import torch
import triton

import sparton.sparton_kernel as sk

B, S, D, V = 32, 128, 768, 30522


def main():
    torch.manual_seed(0)
    hidden = torch.randn(B, S, D, device="cuda", dtype=torch.float16) * 0.05
    embed = torch.randn(V, D, device="cuda", dtype=torch.float16) * 0.05
    bias = torch.randn(V, device="cuda", dtype=torch.float16) * 0.05
    mask = (torch.rand(B, S, device="cuda") > 0.25).to(torch.int32)

    def fwd(bias_):
        return sk.fused_sparton_fwd_op(hidden, embed, bias_, mask)

    for _ in range(3):  # trigger torch.compile + reduction autotune
        fwd(bias)
        fwd(None)
    torch.cuda.synchronize()

    ms_bias = triton.testing.do_bench(lambda: fwd(bias), warmup=25, rep=100)
    ms_nobias = triton.testing.do_bench(lambda: fwd(None), warmup=25, rep=100)

    hidden_flat = hidden.reshape(B * S, D)
    ms_gemm = triton.testing.do_bench(lambda: torch.matmul(hidden_flat, embed.T), warmup=25, rep=100)
    flops = 2.0 * (B * S) * V * D
    print(f"tile size from v_tile_from_bs: {sk.v_tile_from_bs(B, S, V)}")
    print(f"hybrid fwd bias:    {ms_bias:.3f} ms")
    print(f"hybrid fwd no-bias: {ms_nobias:.3f} ms")
    print(f"full-V cuBLAS GEMM: {ms_gemm:.3f} ms ({flops / ms_gemm * 1e-9:.1f} TFLOP/s)")
    print(f"hybrid overhead over pure GEMM (bias): {(ms_bias - ms_gemm) / ms_gemm * 100:.0f}%")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    scores, idx = fwd(bias)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    out_bytes = scores.numel() * scores.element_size() + idx.numel() * idx.element_size()
    print(f"peak extra memory during fwd: {(peak - base) / 2**20:.1f} MiB "
          f"(outputs alone: {out_bytes / 2**20:.1f} MiB)")

    def fwd_bwd():
        h = hidden.detach().requires_grad_(True)
        e = embed.detach().requires_grad_(True)
        b2 = bias.detach().requires_grad_(True)
        s, _ = sk.fused_sparton_fwd_op(h, e, b2, mask)
        s.float().sum().backward()

    fwd_bwd()  # autotune backward kernel
    torch.cuda.synchronize()
    ms_fwd_bwd = triton.testing.do_bench(fwd_bwd, warmup=10, rep=50)
    print(f"hybrid fwd+bwd bias: {ms_fwd_bwd:.3f} ms")


if __name__ == "__main__":
    main()
