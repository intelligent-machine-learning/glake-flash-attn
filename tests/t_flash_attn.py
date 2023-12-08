import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import _get_block_size
from flash_attn.layers.rotary import apply_rotary_emb
import vmmAllocator
import flash_attn_2_cuda as flash_attn_cuda

MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)




def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)




dtype = torch.float16
num_splits = 1
mha_type = "mha"
new_kv = False
local = False
causal = False
seqlen_new_eq_seqlen_q = True
rotary_interleaved = False
rotary_fraction = 0.0
has_batch_idx = False
d = 64
seqlen_q, seqlen_k = (1, 1024)

# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    mha_type,
    num_splits,
    dtype,
):
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    print(window_size)
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None
    k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    print(k_cache.stride(-1))
    v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    phy_gran = 2 * 1024 * 1024
    context_size = seqlen_k * nheads_k * d * k_cache.element_size()
    context_size = phy_gran * ((context_size + phy_gran - 1) // phy_gran)
    num_blocks = context_size // phy_gran
    vmmAllocator.init_physical_handle(2*num_blocks*batch_size_cache)
    key_cache = vmmAllocator.KVcacheSegment(batch_size_cache, context_size)
    value_cache = vmmAllocator.KVcacheSegment(batch_size_cache, context_size)
    

    cache_seqlens = torch.randint(
        0,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
        if new_kv
        else (seqlen_k + 1),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    print(cache_seqlens)
    for i in range(batch_size_cache):
        b_seq_len = cache_seqlens[i]
        #b_seq_len = seqlen_k
        b_context_size = b_seq_len * nheads_k * d * k_cache.element_size()
        b_context_size = phy_gran * ((b_context_size + phy_gran - 1) // phy_gran)
        b_num_blocks = b_context_size // phy_gran
        for _ in range(b_num_blocks - 1):
            key_cache.expandSegment(i)
            value_cache.expandSegment(i)
        b_k_cache = k_cache[i, :, :, :]
        b_v_cache = v_cache[i, :, :, :]
        offset = 0
        for j in range(b_seq_len):
            j_k_cache = b_k_cache[j, :, :]
            j_v_cache = b_v_cache[j, :, :]
            import numpy as np
            k_n = j_k_cache.cpu().numpy()
            k_t = np.ascontiguousarray(k_n)
            vmmAllocator.copy_kv_cache(key_cache.getDevicePtr(i), k_t.__array_interface__['data'][0], nheads_k * d * k_cache.element_size(), offset, 0)
            v_n = j_v_cache.cpu().numpy()
            v_t = np.ascontiguousarray(v_n)
            vmmAllocator.copy_kv_cache(value_cache.getDevicePtr(i), v_t.__array_interface__['data'][0], nheads_k * d * k_cache.element_size(), offset, 0)
            offset += nheads_k * d * k_cache.element_size()

        #import numpy as np
        #k_n = b_k_cache.cpu().numpy()
        #k_t = np.ascontiguousarray(k_n)
        #vmmAllocator.copy_kv_cache(key_cache.getDevicePtr(i), k_t.__array_interface__['data'][0], b_seq_len * nheads_k * d * k_cache.element_size(), 0, 0)
        #v_n = b_v_cache.cpu().numpy()
        #v_t = np.ascontiguousarray(v_n)
        #vmmAllocator.copy_kv_cache(value_cache.getDevicePtr(i), v_t.__array_interface__['data'][0], b_seq_len * nheads_k * d * k_cache.element_size(), 0, 0)
    
    if has_batch_idx:
        cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[:batch_size]
    else:
        cache_batch_idx = None
    # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    if rotary_dim > 0:
        angle = torch.rand(seqlen_k, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        if causal or local:
            q_ro = apply_rotary_emb(
                q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            q_ro = rearrange(
                apply_rotary_emb(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=seqlen_q,
            )
        # q_ro = q
        k_ro = apply_rotary_emb(
            k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (k_cache if not has_batch_idx else k_cache[cache_batch_idx]).clone()
    v_cache_ref = (v_cache if not has_batch_idx else v_cache[cache_batch_idx]).clone()
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    softmax_scale=None
    rotary_cos=None
    rotary_sin=None
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    slot_m = [0, 1]
    slot_mapping = torch.tensor(slot_m, dtype=torch.int, device="cuda")
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        key_cache.getDevicePtr(0),
        value_cache.getDevicePtr(0),
        seqlen_k,
        nheads_k,
        batch_size_cache,
        context_size,
        context_size,
        slot_mapping,
        k,
        v,
        cache_seqlens,
        cos,
        sin,
        cache_batch_idx,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        rotary_interleaved,
        num_splits,
    )

    #out = flash_attn_with_kvcache(
    #    q,
    #    k_cache,
    #    v_cache,
    #    k,
    #    v,
    #    cos,
    #    sin,
    #    cache_seqlens,
    #    cache_batch_idx,
    #    causal=causal,
    #    window_size=window_size,
    #    rotary_interleaved=rotary_interleaved,
    #    num_splits=num_splits,
    #)
    # out = flash_attn_with_kvcache(
    #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
    # )
    # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
    # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # probs = torch.softmax(qk, dim=-1)
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        k_cache_select = k_cache if not has_batch_idx else k_cache[cache_batch_idx]
        v_cache_select = v_cache if not has_batch_idx else v_cache[cache_batch_idx]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    assert (out - out_ref).abs().max().item() <= 3 * (out_pt - out_ref).abs().max().item() + 1e-5


def main():
    test_flash_attn_kvcache(seqlen_q, seqlen_k, d, has_batch_idx, rotary_fraction, rotary_interleaved, seqlen_new_eq_seqlen_q, causal, local, new_kv, mha_type, num_splits, dtype)

if __name__ == "__main__":
    main()
