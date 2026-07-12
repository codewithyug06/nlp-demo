import torch
import warnings

def flash_attention_var_len(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seq_lens_q: torch.Tensor, cu_seq_lens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, dropout_p: float=0.0, softmax_scale: float=None, causal: bool=False) -> torch.Tensor:
    try:
        from flash_attn import flash_attn_var_len_func
        return flash_attn_var_len_func(q, k, v, cu_seqlens_q=cu_seq_lens_q, cu_seqlens_k=cu_seq_lens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    except ImportError:
        warnings.warn('flash_attn is not installed. Falling back to mock implementation (PyTorch SDPA).')
        B = cu_seq_lens_q.shape[0] - 1
        nheads, headdim = (q.shape[1], q.shape[2])
        out = torch.zeros_like(q)
        for i in range(B):
            start_q, end_q = (cu_seq_lens_q[i], cu_seq_lens_q[i + 1])
            start_k, end_k = (cu_seq_lens_k[i], cu_seq_lens_k[i + 1])
            q_i = q[start_q:end_q].transpose(0, 1)
            k_i = k[start_k:end_k].transpose(0, 1)
            v_i = v[start_k:end_k].transpose(0, 1)
            attn_output = torch.nn.functional.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=dropout_p, is_causal=causal)
            out[start_q:end_q] = attn_output.transpose(0, 1)
        return out