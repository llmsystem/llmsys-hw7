"""
Run all tests:
    pytest -v tests/test_inference.py

Run a single part (or skip the TPU-only speedup check on CPU):
    pytest -v tests/test_inference.py -k part_1
    pytest -v tests/test_inference.py -k "not tpu"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model

DATA_PATH = REPO_ROOT / "data" / "test_data.npz"
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.fixture(scope="session", autouse=True)
def _clear_jit_caches():
    """Start each test run from a fresh trace.

    `model.generate` is jitted at import time, so after editing a function it calls
    (and letting `%autoreload` reload it) the old compiled program would still be used.
    """
    jax.clear_caches()


@pytest.fixture(scope="module")
def _npz():
    with np.load(DATA_PATH) as npz:
        yield npz


@pytest.fixture
def golden(_npz):
    """Golden test arrays with IEEE float32 matmul precision (needed on TPU)."""
    with jax.default_matmul_precision("highest"):
        yield _npz


def _load_weights(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    n_layers: int,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    """Reconstruct a parameter PyTree from saved golden weights."""
    layer_keys = ("attn_norm", "w_q", "w_k", "w_v", "w_o", "mlp_norm", "w_gate", "w_up", "w_down")
    layers = tuple(
        {k: jnp.asarray(data[f"{prefix}_l{i}_{k}"]) for k in layer_keys}
        for i in range(n_layers)
    )
    params: dict[str, Any] = {
        "embed_tokens": jnp.asarray(data[f"{prefix}_embed_tokens"]),
        "layers": layers,
        "final_norm": jnp.asarray(data[f"{prefix}_final_norm"]),
    }
    if not tie_word_embeddings:
        params["lm_head"] = jnp.asarray(data[f"{prefix}_lm_head"])
    return params


def test_part_1_1_rms_norm(golden) -> None:
    """Checks RMSNorm numerical accuracy, shape/dtype, and bfloat16 preservation."""
    x = jnp.asarray(golden["a11_rms_x"])
    w = jnp.asarray(golden["a11_rms_w"])
    out = model.rms_norm(x, w, eps=1e-6)
    assert out.shape == x.shape, f"rms_norm shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == x.dtype, f"rms_norm dtype mismatch: {out.dtype} vs {x.dtype}"
    np.testing.assert_allclose(np.asarray(out), golden["a11_rms_expected"], rtol=1e-5, atol=1e-5)

    out_bf16 = model.rms_norm(x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), eps=1e-6)
    assert out_bf16.dtype == jnp.bfloat16, f"rms_norm should preserve bfloat16, got {out_bf16.dtype}"


def test_part_1_2_apply_rope(golden) -> None:
    """Checks RoPE rotation accuracy, norm preservation, and relative-position invariance."""
    q = jnp.asarray(golden["a11_rope_q"])
    pos = jnp.asarray(golden["a11_rope_pos"])
    cos = jnp.asarray(golden["a11_rope_cos"])
    sin = jnp.asarray(golden["a11_rope_sin"])

    q_rot = model.apply_rope(q, pos, cos, sin)
    assert q_rot.shape == q.shape, f"apply_rope shape mismatch: {q_rot.shape}"
    assert q_rot.dtype == q.dtype
    np.testing.assert_allclose(np.asarray(q_rot), golden["a11_rope_expected"], rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(q_rot, axis=-1)),
        np.asarray(jnp.linalg.norm(q, axis=-1)),
        rtol=1e-5,
        atol=1e-5,
    )

    q0, k0 = q[:1, :1], q[1:2, :1]
    dot_a = jnp.sum(
        model.apply_rope(q0, jnp.array([[5]]), cos, sin)
        * model.apply_rope(k0, jnp.array([[2]]), cos, sin),
        axis=-1,
    )
    dot_b = jnp.sum(
        model.apply_rope(q0, jnp.array([[15]]), cos, sin)
        * model.apply_rope(k0, jnp.array([[12]]), cos, sin),
        axis=-1,
    )
    np.testing.assert_allclose(np.asarray(dot_a), np.asarray(dot_b), rtol=1e-4, atol=1e-4)


def test_part_1_3_swiglu_mlp(golden) -> None:
    """Checks SwiGLU MLP output shape and numerical accuracy."""
    mlp_out = model.swiglu_mlp(
        jnp.asarray(golden["a11_mlp_x"]),
        jnp.asarray(golden["a11_mlp_wg"]),
        jnp.asarray(golden["a11_mlp_wu"]),
        jnp.asarray(golden["a11_mlp_wd"]),
    )
    assert mlp_out.shape == golden["a11_mlp_expected"].shape
    np.testing.assert_allclose(np.asarray(mlp_out), golden["a11_mlp_expected"], rtol=1e-4, atol=1e-4)


def test_part_2_1_init_kv_cache() -> None:
    """Checks static shapes and zero-initialization across all layers."""
    cfg = model.ModelConfig(n_layers=3, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
    bsz, max_len = 2, 16
    cache = model.init_kv_cache(cfg, bsz, max_len, dtype=jnp.float32)

    assert len(cache) == cfg.n_layers, f"Expected {cfg.n_layers} layers, got {len(cache)}"
    for layer_cache in cache:
        assert layer_cache.k.shape == (bsz, max_len, cfg.n_kv_heads, cfg.head_dim)
        assert layer_cache.v.shape == (bsz, max_len, cfg.n_kv_heads, cfg.head_dim)
        assert jnp.all(layer_cache.k == 0.0) and jnp.all(layer_cache.v == 0.0)


def test_part_2_2_update_kv_cache() -> None:
    """Checks functional slice updates, immutability, and single-trace @jax.jit."""
    cfg = model.ModelConfig(n_layers=3, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
    bsz, max_len = 2, 16
    cache = model.init_kv_cache(cfg, bsz, max_len, dtype=jnp.float32)

    key = jax.random.PRNGKey(12)
    k_pre = jax.random.normal(key, (bsz, 4, cfg.n_kv_heads, cfg.head_dim))
    v_pre = jax.random.normal(jax.random.fold_in(key, 1), (bsz, 4, cfg.n_kv_heads, cfg.head_dim))
    lc1 = model.update_kv_cache(cache[0], k_pre, v_pre, 0)

    np.testing.assert_allclose(np.asarray(lc1.k[:, :4]), np.asarray(k_pre))
    np.testing.assert_allclose(np.asarray(lc1.v[:, :4]), np.asarray(v_pre))
    assert jnp.all(lc1.k[:, 4:] == 0.0)
    assert jnp.all(cache[0].k == 0.0), "Original KV cache must remain immutable"

    trace_count = 0

    @jax.jit
    def jitted_update(lc, kn, vn, idx):
        nonlocal trace_count
        trace_count += 1
        return model.update_kv_cache(lc, kn, vn, idx)

    lc = lc1
    for step in range(4):
        k_tok = jnp.full((bsz, 1, cfg.n_kv_heads, cfg.head_dim), float(step + 1))
        v_tok = jnp.full((bsz, 1, cfg.n_kv_heads, cfg.head_dim), float(-(step + 1)))
        lc = jitted_update(lc, k_tok, v_tok, jnp.asarray(4 + step, dtype=jnp.int32))

    assert trace_count == 1, f"Expected 1 XLA trace for update_kv_cache, got {trace_count}"
    for step in range(4):
        np.testing.assert_allclose(np.asarray(lc.k[:, 4 + step]), float(step + 1))
        np.testing.assert_allclose(np.asarray(lc.v[:, 4 + step]), float(-(step + 1)))


def test_part_3_1_make_attention_mask(golden) -> None:
    """Checks causal + valid-token boolean mask for prefill and decode."""
    mask_pre = model.make_attention_mask(jnp.asarray(golden["a13_valid_pre"]), 0, 4)
    assert mask_pre.shape == golden["a13_mask_pre_expected"].shape
    assert mask_pre.dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(mask_pre), golden["a13_mask_pre_expected"])

    mask_dec = model.make_attention_mask(jnp.asarray(golden["a13_valid_dec"]), 4, 1)
    assert mask_dec.shape == golden["a13_mask_dec_expected"].shape
    np.testing.assert_array_equal(np.asarray(mask_dec), golden["a13_mask_dec_expected"])


def test_part_3_2_gqa_attention(golden) -> None:
    """Checks GQA accuracy, NaN-free left-padding, and invariance to masked KV slots."""
    q = jnp.asarray(golden["a13_gqa_q"])
    k_cache = jnp.asarray(golden["a13_gqa_k"])
    v_cache = jnp.asarray(golden["a13_gqa_v"])
    mask = jnp.asarray(golden["a13_mask_pre_expected"])

    out = model.gqa_attention(q, k_cache, v_cache, mask)
    assert out.shape == q.shape
    assert jnp.all(jnp.isfinite(out)), "gqa_attention produced NaN/Inf (check all-masked left-pad rows!)"
    np.testing.assert_allclose(np.asarray(out), golden["a13_gqa_expected"], rtol=1e-4, atol=1e-4)

    out_dirty = model.gqa_attention(q, k_cache.at[:, 4:].set(1e5), v_cache.at[:, 4:].set(-1e5), mask)
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(out_dirty[0]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(out[1, 2:]), np.asarray(out_dirty[1, 2:]), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tag,tie", [("tied", True), ("untied", False)])
def test_part_4_2_prefill(golden, tag: str, tie: bool) -> None:
    """Checks prefill (and transformer_block) logits for tied and untied embeddings."""
    cos = jnp.asarray(golden["m_cos"])
    sin = jnp.asarray(golden["m_sin"])
    prompt_ids = jnp.asarray(golden["a14_prompt_ids"])
    prompt_mask = jnp.asarray(golden["a14_prompt_mask"])

    cfg = model.ModelConfig.from_json(CONFIGS_DIR / "model_test.json", tie_word_embeddings=tie)
    params = _load_weights(golden, f"m_{tag}", cfg.n_layers, tie)

    logits_pre, _, _, _ = model.prefill(params, prompt_ids, prompt_mask, cfg, 20, cos, sin)
    expected_pre = golden[f"a14_{tag}_prefill_logits"]
    assert logits_pre.shape == expected_pre.shape
    np.testing.assert_allclose(np.asarray(logits_pre[0]), expected_pre[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(logits_pre[1, 2:]), expected_pre[1, 2:], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tag,tie", [("tied", True), ("untied", False)])
def test_part_4_3_decode_step(golden, tag: str, tie: bool) -> None:
    """Checks 5 sequential jitted decode steps (logits + single XLA trace)."""
    cos = jnp.asarray(golden["m_cos"])
    sin = jnp.asarray(golden["m_sin"])
    prompt_ids = jnp.asarray(golden["a14_prompt_ids"])
    prompt_mask = jnp.asarray(golden["a14_prompt_mask"])
    prompt_len = prompt_ids.shape[1]

    cfg = model.ModelConfig.from_json(CONFIGS_DIR / "model_test.json", tie_word_embeddings=tie)
    params = _load_weights(golden, f"m_{tag}", cfg.n_layers, tie)

    _, cache, valid_mask, cur_pos = model.prefill(params, prompt_ids, prompt_mask, cfg, 20, cos, sin)

    trace_count = 0

    @jax.jit
    def jitted_decode(tok, pos, c, vm, idx):
        nonlocal trace_count
        trace_count += 1
        return model.decode_step(params, tok, pos, c, vm, idx, cos, sin, cfg)

    dec_inputs = golden[f"a14_{tag}_decode_input_tokens"]
    dec_expected = golden[f"a14_{tag}_decode_logits"]

    for step in range(5):
        tok = jnp.asarray(dec_inputs[step][:, None], dtype=jnp.int32)
        dec_logits, cache, valid_mask = jitted_decode(
            tok, cur_pos, cache, valid_mask, jnp.asarray(prompt_len + step, dtype=jnp.int32)
        )
        assert dec_logits.shape == (2, 1, cfg.vocab_size)
        np.testing.assert_allclose(np.asarray(dec_logits), dec_expected[step], rtol=1e-4, atol=1e-4)
        cur_pos = cur_pos + 1

    assert trace_count == 1, f"decode_step recompiled across steps! trace_count={trace_count}"


def test_part_5_1_sample_next_token(golden) -> None:
    """Checks greedy argmax, deterministic sampling, and top-k filtering."""
    logits = jnp.asarray(golden["a15_sample_logits"])

    greedy = model.sample_next_token(logits, jax.random.PRNGKey(0), temperature=0.0)
    np.testing.assert_array_equal(np.asarray(greedy), golden["a15_sample_greedy_expected"])

    s1 = model.sample_next_token(logits, jax.random.PRNGKey(7), temperature=0.8, top_k=5)
    s2 = model.sample_next_token(logits, jax.random.PRNGKey(7), temperature=0.8, top_k=5)
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))

    top5_idx = golden["a15_sample_top5_indices"]
    for b in range(logits.shape[0]):
        assert int(s1[b]) in set(top5_idx[b].tolist()), "top_k sampled outside top-k!"


def test_part_5_2_generate(golden) -> None:
    """Checks greedy generation, left-padded batch invariance, max_new=1, and stochastic sampling."""
    cfg = model.ModelConfig.from_json(CONFIGS_DIR / "model_test.json", tie_word_embeddings=True)
    params = _load_weights(golden, "m_tied", cfg.n_layers, tie_word_embeddings=True)
    prompt_ids = jnp.asarray(golden["a15_prompt_ids"])
    prompt_mask = jnp.asarray(golden["a15_prompt_mask"])
    expected_gen = golden["a15_gen_expected"]
    max_new = expected_gen.shape[1]

    gen = model.generate(params, prompt_ids, prompt_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
    assert gen.shape == expected_gen.shape, f"Expected shape {expected_gen.shape}, got {gen.shape}"
    np.testing.assert_array_equal(np.asarray(gen), expected_gen)

    for i in range(prompt_ids.shape[0]):
        real_tokens = prompt_ids[i][prompt_mask[i]]
        single_ids = real_tokens[None, :]
        single_mask = jnp.ones_like(single_ids, dtype=jnp.bool_)
        single_gen = model.generate(params, single_ids, single_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
        np.testing.assert_array_equal(
            np.asarray(gen[i]),
            np.asarray(single_gen[0]),
            err_msg=f"Variable-length batch item {i} diverged from unpadded B=1 generation!",
        )

    gen_one = model.generate(params, prompt_ids, prompt_mask, cfg, 1, jax.random.PRNGKey(0), temperature=0.0)
    assert gen_one.shape == (3, 1)
    np.testing.assert_array_equal(np.asarray(gen_one[:, 0]), expected_gen[:, 0])

    gen_s1 = model.generate(params, prompt_ids, prompt_mask, cfg, 8, jax.random.PRNGKey(42), temperature=0.9, top_k=10)
    gen_s2 = model.generate(params, prompt_ids, prompt_mask, cfg, 8, jax.random.PRNGKey(42), temperature=0.9, top_k=10)
    np.testing.assert_array_equal(np.asarray(gen_s1), np.asarray(gen_s2))
    assert jnp.all((gen_s1 >= 0) & (gen_s1 < cfg.vocab_size))


def test_tpu_kv_cache_speedup() -> None:
    """On TPU v5e, JIT+scan KV-cached generate must be >= 100x faster than uncached prefill."""
    cfg = model.ModelConfig.from_json(CONFIGS_DIR / "model_tiny.json")
    params = model.init_model_params(cfg, jax.random.PRNGKey(100), dtype=jnp.float32)

    bsz, prompt_len, max_new = 2, 16, 12
    prompt_ids = jax.random.randint(jax.random.PRNGKey(101), (bsz, prompt_len), 1, cfg.vocab_size, dtype=jnp.int32)
    prompt_mask = jnp.ones((bsz, prompt_len), dtype=jnp.bool_)
    cos, sin = model.precompute_rope_freqs(prompt_len + max_new, cfg.head_dim, cfg.rope_theta)

    warm = model.generate(params, prompt_ids, prompt_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
    warm.block_until_ready()

    t0 = time.perf_counter()
    cached_out = model.generate(params, prompt_ids, prompt_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
    cached_out.block_until_ready()
    t_cached = time.perf_counter() - t0

    t0 = time.perf_counter()
    cur_ids = prompt_ids
    cur_mask = prompt_mask
    uncached_tokens = []
    for _ in range(max_new):
        logits, _, _, _ = model.prefill(params, cur_ids, cur_mask, cfg, cur_ids.shape[1], cos, sin)
        next_tok = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)
        uncached_tokens.append(next_tok)
        cur_ids = jnp.concatenate([cur_ids, next_tok[:, None]], axis=1)
        cur_mask = jnp.concatenate([cur_mask, jnp.ones((bsz, 1), dtype=jnp.bool_)], axis=1)
    uncached_out = jnp.stack(uncached_tokens, axis=1)
    uncached_out.block_until_ready()
    t_uncached = time.perf_counter() - t0

    np.testing.assert_array_equal(np.asarray(cached_out), np.asarray(uncached_out))

    speedup = t_uncached / max(t_cached, 1e-9)
    print(f"\nKV-cache speedup: {speedup:.1f}x (cached={t_cached*1e3:.1f} ms, uncached={t_uncached*1e3:.1f} ms)")
    assert speedup >= 100.0, (
        f"Expected KV-cached JIT+scan generate to be >= 100.0x faster than uncached recomputation on TPU v5e, "
        f"but got {speedup:.2f}x (cached={t_cached*1e3:.1f} ms, uncached={t_uncached*1e3:.1f} ms)."
    )
