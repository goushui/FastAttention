import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def attention_forward_minimal(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    nheads, seq_len, head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Minimal FlashAttention - just enough to verify correctness"""

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)  # batch * head index

    # Batch and head indices
    off_h = off_hz % nheads
    off_b = off_hz // nheads

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute Q block pointer
    q_offset = off_b * stride_qb + off_h * stride_qh
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    # Compute K, V block pointers
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh

    # Load Q block
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Scale Q
    sm_scale = 1.0 / tl.sqrt(float(head_dim))
    q = q * sm_scale

    # Loop over K, V blocks
    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        # Load K block
        k_ptrs = K + k_offset + offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        mask_n = offs_n_curr < seq_len
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        # Apply causal mask
        mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        # Update running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Scale previous accumulator
        acc = acc * alpha[:, None]

        # Load V block
        v_ptrs = V + v_offset + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Update accumulator
        acc += tl.dot(p.to(v.dtype), v)

        # Update m_i
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Write output
    out_offset = off_b * stride_ob + off_h * stride_oh
    out_ptrs = Out + out_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


class MinimalFlashAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        # Shapes
        batch, nheads, seq_len, head_dim = q.shape

        # Allocate output
        out = torch.empty_like(q)

        # Grid
        grid = lambda META: (
            triton.cdiv(seq_len, META['BLOCK_M']),
            batch * nheads,
        )

        # Launch kernel
        attention_forward_minimal[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            nheads, seq_len, head_dim,
            BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=head_dim,
        )

        return out


# Test correctness
def test_correctness():
    torch.manual_seed(0)

    batch, nheads, seq_len, head_dim = 2, 32, 2048, 128
    q = torch.randn(batch, nheads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Reference
    ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Your implementation
    my_attn = MinimalFlashAttention()
    my_out = my_attn(q, k, v)

    # Check
    diff = (ref_out - my_out).abs().max().item()
    print(f"Max difference: {diff}")
    assert diff < 1e-3, f"Correctness check failed: {diff}"
    print("Correctness test passed!")


if __name__ == "__main__":
    test_correctness()
