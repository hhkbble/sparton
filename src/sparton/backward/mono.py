"""Mono backward: the original M2-era fully-atomic scatter (the unoptimized baseline).

Restored verbatim from commit 6e19af3 (`fused_sparton_bwd_kernel_with_bias`). It loads the
saved argmax index, computes `grad * exp(-scores)` where `scores > 0` (the same exact
gradient as the optimized backward), and scatters hidden/embed/bias gradients with
`tl.atomic_add` into zero-initialized fp32 buffers. Selected by the hybrid and naive
forwards, and the original-vs-optimized A/B baseline. Slow by design.
"""

import triton
import triton.language as tl
import torch


def get_mono_bwd_configs():
    return [
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 16, 'BLOCK_D': 32, 'GROUP_SIZE': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 16, 'BLOCK_D': 32, 'GROUP_SIZE': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 32, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 64, 'BLOCK_V': 32, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 64, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 32, 'BLOCK_D': 128, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 32, 'BLOCK_D': 128, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 32, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 64, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_B': 128, 'BLOCK_V': 16, 'BLOCK_D': 16, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    ]


@triton.autotune(
    configs=get_mono_bwd_configs(),
    key=['batch_size', 'vocab_size', 'hidden_dim'],
    reset_to_zero=['hidden_grad_ptr', 'embed_grad_ptr', 'bias_grad_ptr']
)
@triton.jit
def mono_bwd_kernel(
    grad_out_ptr, # [B, V], float32)
    max_scores_ptr, # [B, V], (float32)
    max_idx_ptr, # [B, V], (int64)
    hidden_ptr, # [B*S, D] (float32)
    embed_ptr, # [V, D] (float32)
    hidden_grad_ptr, # [B*S, D] (float32)
    embed_grad_ptr, # [V, D] (float32)
    bias_grad_ptr, # [V] (float32)
    batch_size, # B, 
    seq_len, # S, 
    hidden_dim: tl.constexpr, # D, 
    vocab_size: tl.constexpr, # V, 
    BLOCK_B: tl.constexpr, # batch-block
    BLOCK_V: tl.constexpr, # seq-block
    BLOCK_D: tl.constexpr, # hidden-dim-block
    GROUP_SIZE: tl.constexpr, # group size
):
    v_block_id = tl.program_id(0)
    b_block_id = tl.program_id(1)
    
    num_v_blocks = tl.num_programs(0)
    num_b_blocks = tl.num_programs(1)
    # this is probabaly not needed
    b_block_id, v_block_id = tl.swizzle2d(b_block_id, v_block_id, num_b_blocks, num_v_blocks, GROUP_SIZE)

    start_b = b_block_id * BLOCK_B
    start_v = v_block_id * BLOCK_V
    offs_v = start_v + tl.arange(0, BLOCK_V).to(tl.int64)
    offs_b = start_b + tl.arange(0, BLOCK_B).to(tl.int64)

    # might not be needed
    offs_v = tl.max_contiguous(tl.multiple_of(offs_v, BLOCK_V), BLOCK_V)
    offs_b = tl.max_contiguous(tl.multiple_of(offs_b, BLOCK_B), BLOCK_B)

    offs_d = tl.arange(0, BLOCK_D)

    b_mask = (offs_b < batch_size)
    bv_mask = (offs_b[:, None] < batch_size) & (offs_v[None, :] < vocab_size)
    bv_offs = offs_b[:, None] * vocab_size + offs_v[None, :]
    
    # load max logits (block)
    block_max_logits = tl.load(max_scores_ptr + bv_offs, mask = bv_mask, other=0.0).to(tl.float32) # BLOCK_B x BLOCK_V

    if tl.sum(block_max_logits) == 0:
        return 

    # load gradient with regard to the max logits (block)
    grad_out = tl.load(grad_out_ptr + bv_offs, mask = bv_mask, other=0.0).to(tl.float32) # BLOCK_B x BLOCK_V

    # load max indices (block)
    block_max_idx = tl.load(max_idx_ptr + bv_offs, mask = bv_mask, other=0) # BLOCK_B x BLOCK_V
    # if b_block_id ==0 and v_block_id == 0:
    #     tl.device_print("block_max_idx")

    # calculate the gradient of log(1 + relu(x)) with regard to x
    # log(x) = 1/x = 1/ exp(log(x))
    # relu_log1p_grad = (block_max_logits > 0).to(tl.float32) * grad_out * tl.exp(-block_max_logits)
    relu_log1p_grad = tl.where(block_max_logits > 0, grad_out * tl.exp(-block_max_logits), 0.0).to(tl.float32)

    
    # calculate gradient with regard to bias:
    mask_v = offs_v < vocab_size
    tl.atomic_add(bias_grad_ptr + offs_v, tl.sum(relu_log1p_grad, axis=0), mask = mask_v, sem = "relaxed")

    # address of the vocab emb block
    e_ptrs = embed_ptr + offs_v[:, None] * hidden_dim + offs_d[None, :]

    # address of the hidden states with selected max logit values. note: this one is non-continuous, might lead to slow data loading
    h_ptrs = hidden_ptr + offs_b[:, None, None] * seq_len * hidden_dim + block_max_idx[:, :, None] * hidden_dim + offs_d[None, None, :]

    # address of the output embedding gradient 
    e_grad_ptrs = embed_grad_ptr + offs_v[:, None] * hidden_dim + offs_d[None, :]
    # address of the output hidden state gradient
    h_grad_ptrs = hidden_grad_ptr + offs_b[:, None, None] * seq_len * hidden_dim + block_max_idx[:, :, None] * hidden_dim + offs_d[None, None, :]
    
    valid_logit_mask = block_max_logits > 0

    for start_d in range(0, hidden_dim, BLOCK_D):
        mask_d =  start_d +  offs_d < hidden_dim
        #  b x v x hidden_size: non-coalesed reading 
        hidden_state = tl.load(h_ptrs, mask= b_mask[:, None, None] & mask_d[None, None, :] & valid_logit_mask[:, :, None], other=0.0).to(tl.float32) # BLOCK_B x BLOCK_V x BLOCK_D                
        embed_grad_update =  tl.sum(hidden_state * relu_log1p_grad[:, :, None], axis=0) # V x E
        # this seems to be a non-coalesed writing? output at scattered addresses
        tl.atomic_add(e_grad_ptrs, embed_grad_update, mask =  mask_v[:, None] & mask_d[None, :], sem = "relaxed")
        # loading v embedidngs, to compate gradient for hidden states
        embed = tl.load(e_ptrs, mask= mask_v[:, None] & mask_d[None, :], other=0.0, eviction_policy="evict_last").to(tl.float32)# BLOCK_V x BLOCK_D 
        hidden_grad_update = (embed[None, :, :] * relu_log1p_grad[:, :, None]).to(tl.float32) # BLOCK_B x BLOCK_V x BLOCK_D 
        tl.atomic_add(h_grad_ptrs, hidden_grad_update, mask  = bv_mask[:, :, None] & mask_d[None, None, :] & valid_logit_mask[:,:, None], sem = "relaxed")    
        e_ptrs += BLOCK_D
        h_ptrs += BLOCK_D
        e_grad_ptrs += BLOCK_D
        h_grad_ptrs += BLOCK_D


def mono_bwd(grad_out, max_scores, max_idx, hidden, embed, bias):
    """Launch the mono backward; returns fp32 (hidden_grad, embed_grad, bias_grad)."""

    B, S, D = hidden.shape
    V, D_e = embed.shape
    assert D == D_e
    assert max_scores.shape == (B, V)
    assert max_idx.shape == (B, V)

    # Mono accumulates ALL three gradients via atomic_add and only touches the
    # argmax-winner rows, so the buffers MUST be zero-initialized (NOT
    # torch.empty): the autotuner's reset_to_zero covers only its sweep, and
    # untouched rows must read 0. This is the OPPOSITE of the optimized
    # backward's exclusive-owner plain stores -- do not cross the strategies.
    hidden_grad = torch.zeros_like(hidden, dtype=torch.float32)
    embed_grad = torch.zeros_like(embed, dtype=torch.float32)
    if bias is not None:
        bias_grad = torch.zeros_like(bias, dtype=torch.float32)
    else:
        # The kernel unconditionally writes bias_grad; give it a scratch buffer
        # (the op discards it and returns None when bias is None).
        bias_grad = torch.zeros((V,), device=grad_out.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_V']), triton.cdiv(B, meta['BLOCK_B']))
    mono_bwd_kernel[grid](
        grad_out_ptr=grad_out,
        max_scores_ptr=max_scores,
        max_idx_ptr=max_idx,
        hidden_ptr=hidden,
        embed_ptr=embed,
        hidden_grad_ptr=hidden_grad,
        embed_grad_ptr=embed_grad,
        bias_grad_ptr=bias_grad,
        batch_size=B,
        seq_len=S,
        hidden_dim=D,
        vocab_size=V,
    )
    return hidden_grad, embed_grad, bias_grad
