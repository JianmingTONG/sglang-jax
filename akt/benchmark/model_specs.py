"""Architecture-derived workload specs for the AKT loop.

Builders that turn an architecture description (HF-config-style parameters, as
consumed by the sglang-jax model stack in `python/sgl_jax/srt/models/` and
`python/sgl_jax/srt/configs/`) into a formal `WorkloadSpec`, plus a library of
concrete instances (Qwen3, Qwen3-MoE, Kimi-Linear).

Provenance discipline: every instance names where its numbers come from. Where
a public checkpoint's exact config is not vendored in this repo, the instance
is parameterized from the sgl_jax config class defaults and says so — pass the
real HF config dict to the builder to get the exact workload.

Sequence lengths are workload knobs, not architecture facts: each builder takes
`seqs=(short, long)` and emits one prefill block per length, mirroring how the
frozen tiny-* traces pair a short and a long envelope. `tokens_decode` sizes the
decode ops.

Run as a CLI (repo root, PYTHONPATH=python:.):
    python akt/benchmark/model_specs.py --list
    python akt/benchmark/model_specs.py --describe qwen3-8b
    python akt/benchmark/model_specs.py --lower qwen3-8b     # trace + coverage
    python akt/benchmark/model_specs.py --verify-legacy
"""

from __future__ import annotations

import argparse
import json

from akt.benchmark.workload_spec import (
    BlockSpec,
    OpSpec,
    WorkloadSpec,
    legacy_specs,
    lower,
    materialize_cases,
    spec_fingerprint,
    verify_legacy_roundtrip,
)


# ------------------------------------------------------------- builders
def dense_transformer_spec(
    workload_id: str,
    *,
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    inter: int,
    seqs: tuple[int, ...] = (512, 2048),
    tokens_decode: int = 256,
    cache_tokens: int = 4096,
    provenance: str = "hand-authored",
) -> WorkloadSpec:
    """Dense GQA transformer (Qwen3 / Llama shape family).

    Per block: qkv projection + o projection (matmul), SwiGLU MLP, full-attn
    prefill (reported unsupported — coverage gap), paged decode attention and
    KV update on the decode side.
    """

    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    blocks = []
    for seq in seqs:
        phase = "long-prefill" if seq >= 1024 else "prefill"
        blocks.append(BlockSpec(
            block_id=f"prefill{seq}", repeat=layers,
            ops=(
                OpSpec.make("matmul_grouped", phase=phase,
                            m=seq, k=hidden, n=q_out + 2 * kv_out, groups=1),
                OpSpec.make("attn_prefill_full", phase=phase,
                            seq=seq, heads=heads, kv_heads=kv_heads,
                            head_dim=head_dim),
                OpSpec.make("matmul_grouped", phase=phase,
                            m=seq, k=q_out, n=hidden, groups=1),
                OpSpec.make("mlp_swiglu", phase=phase,
                            seq=seq, hidden=hidden, inter=inter),
            ),
        ))
    blocks.append(BlockSpec(
        block_id="decode", repeat=layers,
        ops=(
            OpSpec.make("attn_decode_paged", phase="decode",
                        heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                        ctx=cache_tokens),
            OpSpec.make("kv_update", phase="decode", seed=42,
                        heads=kv_heads, cache=cache_tokens, new=tokens_decode),
        ),
    ))
    return WorkloadSpec.make(
        workload_id, "dense", blocks,
        description=f"dense GQA transformer, {layers}L h{hidden} "
                    f"{heads}q/{kv_heads}kv hd{head_dim} i{inter}",
        provenance=provenance,
        hidden=hidden, layers=layers, heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, inter=inter,
    )


def hybrid_linear_spec(
    workload_id: str,
    *,
    hidden: int,
    layers: int,
    heads: int,
    kda_head_dim: int,
    kda_ratio: int = 3,          # kda layers per full-attention layer (Kimi-Linear 3:1)
    inter: int,
    seqs: tuple[int, ...] = (512, 2048),
    tokens_decode: int = 256,
    cache_tokens: int = 4096,
    provenance: str = "hand-authored",
) -> WorkloadSpec:
    """Hybrid linear-attention transformer (Kimi-Linear shape family).

    Per `kda_ratio+1` layers: kda_ratio KDA prefill layers + one full-attention
    layer (prefill side unsupported → coverage gap; decode side = rpa) + shared
    projections/MLP per layer.
    """

    kda_layers = layers * kda_ratio // (kda_ratio + 1)
    full_layers = layers - kda_layers
    blocks = []
    for seq in seqs:
        phase = "long-prefill" if seq >= 1024 else "prefill"
        blocks.append(BlockSpec(
            block_id=f"kda-block{seq}", repeat=kda_layers,
            ops=(
                OpSpec.make("matmul_grouped", phase=phase,
                            m=seq, k=hidden, n=heads * kda_head_dim, groups=1),
                OpSpec.make("kda_prefill", phase=phase,
                            seq=seq, heads=heads, head_dim=kda_head_dim),
                OpSpec.make("mlp_swiglu", phase=phase,
                            seq=seq, hidden=hidden, inter=inter),
            ),
        ))
        blocks.append(BlockSpec(
            block_id=f"full-attn-block{seq}", repeat=full_layers,
            ops=(
                OpSpec.make("attn_prefill_full", phase=phase,
                            seq=seq, heads=heads, head_dim=kda_head_dim),
            ),
        ))
    blocks.append(BlockSpec(
        block_id="decode", repeat=full_layers,
        ops=(
            OpSpec.make("attn_decode_paged", phase="decode", heads=heads,
                        head_dim=kda_head_dim, ctx=cache_tokens),
            OpSpec.make("kv_update", phase="decode", seed=42,
                        heads=heads, cache=cache_tokens, new=tokens_decode),
        ),
    ))
    return WorkloadSpec.make(
        workload_id, "hybrid-linear", blocks,
        description=f"hybrid linear-attention (KDA:{kda_ratio}:1 full), {layers}L "
                    f"h{hidden} {heads}h kd{kda_head_dim} i{inter}",
        provenance=provenance,
        hidden=hidden, layers=layers, heads=heads,
        kda_head_dim=kda_head_dim, kda_ratio=kda_ratio, inter=inter,
    )


def moe_transformer_spec(
    workload_id: str,
    *,
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    experts: int,
    top_k: int,
    moe_inter: int,
    seqs: tuple[int, ...] = (512,),
    tokens_decode: int = 256,
    cache_tokens: int = 8192,
    provenance: str = "hand-authored",
) -> WorkloadSpec:
    """MoE transformer (Qwen3-MoE / DeepSeek shape family).

    Router projection is a dense matmul; the expert FFN is expressed BOTH as the
    fused-MoE op (unsupported wiring today → coverage gap) and as the grouped
    expert matmuls (supported via gmm with groups=experts), mirroring how the
    Kimi EPMoE route can dispatch grouped GMM.
    """

    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    blocks = []
    for seq in seqs:
        phase = "long-prefill" if seq >= 1024 else "prefill"
        blocks.append(BlockSpec(
            block_id=f"prefill{seq}", repeat=layers,
            ops=(
                OpSpec.make("matmul_grouped", phase=phase,
                            m=seq, k=hidden, n=q_out + 2 * kv_out, groups=1),
                OpSpec.make("attn_prefill_full", phase=phase,
                            seq=seq, heads=heads, kv_heads=kv_heads,
                            head_dim=head_dim),
                OpSpec.make("matmul_grouped", phase=phase, call_id=f"router{seq}",
                            m=seq, k=hidden, n=experts, groups=1),
                OpSpec.make("moe_ffn", phase=phase,
                            tokens=seq, experts=experts, top_k=top_k,
                            hidden=hidden, inter=moe_inter),
                OpSpec.make("matmul_grouped", phase=phase, call_id=f"experts{seq}",
                            m=seq * top_k, k=hidden, n=moe_inter, groups=experts),
            ),
        ))
    blocks.append(BlockSpec(
        block_id="decode", repeat=layers,
        ops=(
            OpSpec.make("attn_decode_paged", phase="decode", heads=heads,
                        kv_heads=kv_heads, head_dim=head_dim, ctx=cache_tokens),
            OpSpec.make("kv_update", phase="decode", seed=42,
                        heads=kv_heads, cache=cache_tokens, new=tokens_decode),
        ),
    ))
    return WorkloadSpec.make(
        workload_id, "moe", blocks,
        description=f"MoE transformer, {layers}L h{hidden} {experts}E top{top_k} "
                    f"moe_i{moe_inter}",
        provenance=provenance,
        hidden=hidden, layers=layers, heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, experts=experts, top_k=top_k, moe_inter=moe_inter,
    )


# ------------------------------------------------------------- instances
def library() -> dict[str, WorkloadSpec]:
    """Concrete architecture instances. Provenance is stated per instance."""

    specs = list(legacy_specs())

    specs.append(dense_transformer_spec(
        "qwen3-8b",
        hidden=4096, layers=36, heads=32, kv_heads=8, head_dim=128, inter=12288,
        provenance="public Qwen3-8B HF config (hidden 4096, 36L, 32q/8kv GQA, "
                   "head_dim 128, intermediate 12288); served by "
                   "python/sgl_jax/srt/models/qwen3.py",
    ))
    specs.append(moe_transformer_spec(
        "qwen3-30b-a3b",
        hidden=2048, layers=48, heads=32, kv_heads=4, head_dim=128,
        experts=128, top_k=8, moe_inter=768,
        provenance="public Qwen3-30B-A3B HF config (hidden 2048, 48L, 32q/4kv, "
                   "128 experts top-8, moe_intermediate 768); served by "
                   "python/sgl_jax/srt/models/qwen3_moe.py",
    ))
    specs.append(hybrid_linear_spec(
        "kimi-linear-sgl-default",
        hidden=4096, layers=32, heads=32, kda_head_dim=128, kda_ratio=3,
        inter=11008,
        provenance="sgl_jax KimiLinearConfig class DEFAULTS "
                   "(python/sgl_jax/srt/configs/kimi_linear.py: hidden 4096, "
                   "32L, 32h, intermediate 11008; kda_ratio 3:1 per the "
                   "Kimi-Linear architecture); pass the real "
                   "Kimi-Linear-48B-A3B HF config to hybrid_linear_spec for "
                   "the exact checkpoint workload",
    ))
    # Testbench-scale siblings: same architecture families at shapes small
    # enough for exhaustive per-config measurement on this host.
    specs.append(dense_transformer_spec(
        "qwen3-tiny-testbench",
        hidden=512, layers=2, heads=8, kv_heads=2, head_dim=64, inter=512,
        seqs=(256,), tokens_decode=256, cache_tokens=4096,
        provenance="qwen3 family scaled to testbench-measurable shapes",
    ))
    return {spec.workload_id: spec for spec in specs}


# ------------------------------------------------------------------ CLI
def _describe(spec: WorkloadSpec) -> str:
    lines = [
        f"workload_id : {spec.workload_id}",
        f"family      : {spec.family}",
        f"description : {spec.description}",
        f"provenance  : {spec.provenance}",
        f"fingerprint : {spec_fingerprint(spec)}",
        f"arch        : {dict(spec.arch)}",
        "blocks:",
    ]
    for block in spec.blocks:
        lines.append(f"  {block.block_id} (x{block.repeat}):")
        for op in block.ops:
            lines.append(f"    {op.op:20s} {dict(op.params)} [{op.phase}]")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--describe")
    parser.add_argument("--lower")
    parser.add_argument("--verify-legacy", action="store_true")
    args = parser.parse_args()
    lib = library()

    if args.verify_legacy:
        verify_legacy_roundtrip()
        print("legacy roundtrip OK: the formal specs reproduce "
              "MODEL_WORKLOADS call-for-call; frozen fingerprint untouched")
    if args.list:
        for wid, spec in lib.items():
            print(f"{wid:26s} {spec.family:14s} {spec.description[:70]}")
    if args.describe:
        print(_describe(lib[args.describe]))
    if args.lower:
        lowered = lower(lib[args.lower])
        print(f"# lowered trace for {args.lower} "
              f"({len(lowered.workload.calls)} measurable calls)")
        for call in lowered.workload.calls:
            print(f"  {call.call_id:34s} -> {call.case_id:34s} [{call.phase}]")
        new = lowered.new_case_ids
        print(f"# new (non-frozen) cases required: {len(new)}")
        for case_id in new:
            print(f"  {case_id}")
        gaps = [row for row in lowered.coverage if row["status"] == "unsupported"]
        print(f"# coverage gaps: {len(gaps)}")
        for row in gaps:
            print(f"  {row['call']:34s} {row['op']:22s} {row['why'][:90]}")
        print(json.dumps({"AKT_WORKLOAD_LOWER": {
            "workload": args.lower,
            "measurable_calls": len(lowered.workload.calls),
            "new_cases": new,
            "unsupported_ops": sorted({row['op'] for row in gaps}),
        }}))


if __name__ == "__main__":
    main()
