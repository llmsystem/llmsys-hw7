from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig


def init_model_params(
    config: ModelConfig,
    rng_key: jax.Array,
    dtype: jnp.dtype = jnp.float32,
) -> dict[str, Any]:
    """Initialize a LLaMA-style parameter PyTree with deterministic random weights."""
    keys = jax.random.split(rng_key, 2 + config.n_layers * 7)
    idx = 0

    def next_key() -> jax.Array:
        nonlocal idx
        k = keys[idx]
        idx += 1
        return k

    def rand_weight(shape: tuple[int, ...], std: float = 0.02) -> jax.Array:
        return (jax.random.normal(next_key(), shape, dtype=jnp.float32) * std).astype(dtype)

    kv_dim = config.n_kv_heads * config.head_dim
    layers = []
    for _ in range(config.n_layers):
        layers.append(
            {
                "attn_norm": jnp.ones((config.d_model,), dtype=dtype),
                "w_q": rand_weight((config.d_model, config.d_model)),
                "w_k": rand_weight((config.d_model, kv_dim)),
                "w_v": rand_weight((config.d_model, kv_dim)),
                "w_o": rand_weight((config.d_model, config.d_model)),
                "mlp_norm": jnp.ones((config.d_model,), dtype=dtype),
                "w_gate": rand_weight((config.d_model, config.d_ff)),
                "w_up": rand_weight((config.d_model, config.d_ff)),
                "w_down": rand_weight((config.d_ff, config.d_model)),
            }
        )

    embed_tokens = rand_weight((config.vocab_size, config.d_model))
    params: dict[str, Any] = {
        "embed_tokens": embed_tokens,
        "layers": tuple(layers),
        "final_norm": jnp.ones((config.d_model,), dtype=dtype),
    }
    if not config.tie_word_embeddings:
        params["lm_head"] = rand_weight((config.d_model, config.vocab_size))
    return params


def left_pad_prompts(
    token_lists: list[list[int]],
    pad_token_id: int = 0,
    pad_to_multiple: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Left-pad a batch of variable-length token lists into static JAX arrays.

    Returns:
        prompt_ids: [B, P] int32
        prompt_mask: [B, P] bool (True for real tokens, False for left-pad)
    """
    max_len = max(len(seq) for seq in token_lists)
    if pad_to_multiple > 1:
        max_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    batch_size = len(token_lists)
    ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int32)
    mask = np.zeros((batch_size, max_len), dtype=np.bool_)
    for i, seq in enumerate(token_lists):
        n = len(seq)
        if n > 0:
            ids[i, max_len - n :] = seq
            mask[i, max_len - n :] = True
    return jnp.asarray(ids), jnp.asarray(mask)


def estimate_tpu_v5e_roofline(
    config: ModelConfig,
    batch_size: int,
    context_len: int,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate single-step decode bytes, FLOPs, and TPU v5e roofline bounds.

    TPU v5e single-chip specs:
      - Peak BF16 MXU compute: 197 TFLOP/s (1.97e14 FLOP/s)
      - Peak HBM bandwidth:    819 GB/s    (8.19e11 bytes/s)
      - Critical intensity:    ~240.5 FLOP/byte
    """
    kv_dim = config.n_kv_heads * config.head_dim
    params_per_layer = (
        config.d_model * config.d_model
        + 2 * config.d_model * kv_dim
        + config.d_model * config.d_model
        + 3 * config.d_model * config.d_ff
    )
    total_linear_params = config.n_layers * params_per_layer + config.d_model * config.vocab_size
    weight_bytes = total_linear_params * dtype_bytes

    kv_cache_bytes = (
        2
        * config.n_layers
        * batch_size
        * context_len
        * config.n_kv_heads
        * config.head_dim
        * dtype_bytes
    )
    total_hbm_bytes = weight_bytes + kv_cache_bytes

    linear_flops = 2 * batch_size * total_linear_params
    attn_flops = 4 * config.n_layers * batch_size * context_len * config.d_model
    total_flops = linear_flops + attn_flops

    tpu_v5e_bw = 819e9  # bytes/s
    tpu_v5e_flops = 197e12  # FLOP/s
    t_mem_ms = (total_hbm_bytes / tpu_v5e_bw) * 1e3
    t_math_ms = (total_flops / tpu_v5e_flops) * 1e3

    return {
        "weight_bytes": float(weight_bytes),
        "kv_cache_bytes": float(kv_cache_bytes),
        "total_hbm_bytes": float(total_hbm_bytes),
        "total_flops": float(total_flops),
        "arithmetic_intensity": float(total_flops / max(total_hbm_bytes, 1)),
        "t_mem_ms": float(t_mem_ms),
        "t_math_ms": float(t_math_ms),
        "roofline_tpot_ms": float(max(t_mem_ms, t_math_ms)),
    }


def load_hf_llama_weights(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[ModelConfig, dict[str, Any], Any]:
    """Load an ungated HuggingFace LLaMA-family checkpoint into JAX.

    Requires `transformers` and `torch`.
    Returns:
        (config, params, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_cfg = hf_model.config
    sd = hf_model.state_dict()

    tie_emb = bool(getattr(hf_cfg, "tie_word_embeddings", True))
    # transformers>=5 moves `rope_theta` into `config.rope_parameters`.
    rope_params = getattr(hf_cfg, "rope_parameters", None) or {}
    rope_theta = getattr(hf_cfg, "rope_theta", None) or rope_params.get("rope_theta", 10000.0)
    config = ModelConfig(
        vocab_size=int(hf_cfg.vocab_size),
        n_layers=int(hf_cfg.num_hidden_layers),
        d_model=int(hf_cfg.hidden_size),
        n_heads=int(hf_cfg.num_attention_heads),
        n_kv_heads=int(getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads)),
        d_ff=int(hf_cfg.intermediate_size),
        rms_norm_eps=float(getattr(hf_cfg, "rms_norm_eps", 1e-6)),
        rope_theta=float(rope_theta),
        tie_word_embeddings=tie_emb,
    )

    def to_jax(tensor: torch.Tensor, transpose: bool = False) -> jax.Array:
        arr = tensor.detach().cpu().numpy()
        if transpose:
            arr = arr.T
        return jnp.asarray(arr, dtype=dtype)

    layers = []
    for i in range(config.n_layers):
        p = f"model.layers.{i}"
        layers.append(
            {
                "attn_norm": to_jax(sd[f"{p}.input_layernorm.weight"]),
                "w_q": to_jax(sd[f"{p}.self_attn.q_proj.weight"], transpose=True),
                "w_k": to_jax(sd[f"{p}.self_attn.k_proj.weight"], transpose=True),
                "w_v": to_jax(sd[f"{p}.self_attn.v_proj.weight"], transpose=True),
                "w_o": to_jax(sd[f"{p}.self_attn.o_proj.weight"], transpose=True),
                "mlp_norm": to_jax(sd[f"{p}.post_attention_layernorm.weight"]),
                "w_gate": to_jax(sd[f"{p}.mlp.gate_proj.weight"], transpose=True),
                "w_up": to_jax(sd[f"{p}.mlp.up_proj.weight"], transpose=True),
                "w_down": to_jax(sd[f"{p}.mlp.down_proj.weight"], transpose=True),
            }
        )

    params: dict[str, Any] = {
        "embed_tokens": to_jax(sd["model.embed_tokens.weight"]),
        "layers": tuple(layers),
        "final_norm": to_jax(sd["model.norm.weight"]),
    }
    if not tie_emb and "lm_head.weight" in sd:
        params["lm_head"] = to_jax(sd["lm_head.weight"], transpose=True)

    return config, params, tokenizer
