"""Tests for the formal workload representation and its frozen-contract safety."""

import pytest

from akt.benchmark.model_specs import (
    dense_transformer_spec,
    hybrid_linear_spec,
    library,
    moe_transformer_spec,
)
from akt.benchmark.model_workloads import MODEL_WORKLOADS, contract_fingerprint
from akt.benchmark.workload_spec import (
    OpSpec,
    UNSUPPORTED_OPS,
    legacy_specs,
    lower,
    materialize_cases,
    spec_fingerprint,
    verify_legacy_roundtrip,
)


def test_legacy_roundtrip_reproduces_the_frozen_contract():
    """The formal layer must reproduce MODEL_WORKLOADS call-for-call and must
    not disturb the pinned campaign fingerprint."""
    before = contract_fingerprint()
    assert verify_legacy_roundtrip() is True
    for spec, frozen in zip(legacy_specs(), MODEL_WORKLOADS):
        lowered = lower(spec)
        assert lowered.workload.calls == frozen.calls
        assert lowered.new_case_ids == []
    assert contract_fingerprint() == before


def test_spec_fingerprint_is_deterministic_and_content_sensitive():
    spec_a = dense_transformer_spec("wl", hidden=512, layers=2, heads=8,
                                    kv_heads=2, head_dim=64, inter=512)
    spec_b = dense_transformer_spec("wl", hidden=512, layers=2, heads=8,
                                    kv_heads=2, head_dim=64, inter=512)
    spec_c = dense_transformer_spec("wl", hidden=1024, layers=2, heads=8,
                                    kv_heads=2, head_dim=64, inter=512)
    assert spec_fingerprint(spec_a) == spec_fingerprint(spec_b)
    assert spec_fingerprint(spec_a) != spec_fingerprint(spec_c)


def test_lowering_reports_unsupported_ops_instead_of_dropping_them_silently():
    spec = dense_transformer_spec("wl", hidden=512, layers=2, heads=8,
                                  kv_heads=2, head_dim=64, inter=512,
                                  seqs=(256,))
    lowered = lower(spec)
    gaps = [row for row in lowered.coverage if row["status"] == "unsupported"]
    assert gaps, "dense prefill attention must surface as a coverage gap"
    assert all(row["op"] in UNSUPPORTED_OPS for row in gaps)
    # every op instance is accounted for: measurable + unsupported == expanded
    assert len(lowered.coverage) == sum(len(b.ops) for b in spec.blocks)


def test_lowering_reuses_frozen_cases_and_names_only_new_shapes():
    spec = dense_transformer_spec("wl", hidden=512, layers=2, heads=8,
                                  kv_heads=2, head_dim=64, inter=512,
                                  seqs=(256,))
    lowered = lower(spec)
    case_ids = [c.case_id for c in lowered.workload.calls]
    # fused_mlp:s256_h512_i512 is a FROZEN case (tiny-moe expert-mlp) — the
    # lowering must resolve to it rather than materializing a duplicate.
    assert "fused_mlp:s256_h512_i512" in case_ids
    assert "fused_mlp:s256_h512_i512" not in lowered.new_case_ids
    assert "gmm:m256_k512_n768_g1" in lowered.new_case_ids


def test_materialized_case_inherits_the_frozen_refs_contract():
    from akt.benchmark.refs import gmm as gmm_refs

    spec = dense_transformer_spec("wl", hidden=512, layers=2, heads=8,
                                  kv_heads=2, head_dim=64, inter=512,
                                  seqs=(256,))
    cases = materialize_cases(spec)
    case = cases["gmm:m256_k512_n768_g1"]
    assert case.atol == gmm_refs.ATOL and case.rtol == gmm_refs.RTOL
    assert case.native_test == gmm_refs.NATIVE_TEST
    assert "NOT part of the frozen objective" in case.note


def test_unknown_operator_family_fails_closed():
    from akt.benchmark.workload_spec import BlockSpec, WorkloadSpec

    spec = WorkloadSpec.make(
        "wl", "dense",
        [BlockSpec("b", (OpSpec.make("warp_drive", flux=7),))],
    )
    with pytest.raises(ValueError, match="unknown operator family"):
        lower(spec)


def test_library_instances_lower_without_errors():
    for workload_id, spec in library().items():
        lowered = lower(spec)
        assert lowered.workload.model_id == workload_id
        if spec.family == "legacy":
            assert not lowered.new_case_ids
        else:
            # architecture-derived workloads must produce at least one
            # measurable call and keep provenance non-empty
            assert lowered.workload.calls
            assert spec.provenance and spec.provenance != "hand-authored"


def test_builders_cover_the_three_architecture_families():
    dense = dense_transformer_spec("d", hidden=512, layers=2, heads=8,
                                   kv_heads=2, head_dim=64, inter=512)
    hybrid = hybrid_linear_spec("h", hidden=512, layers=4, heads=4,
                                kda_head_dim=64, inter=512)
    moe = moe_transformer_spec("m", hidden=512, layers=2, heads=8, kv_heads=2,
                               head_dim=64, experts=8, top_k=2, moe_inter=512)
    assert {dense.family, hybrid.family, moe.family} == {"dense", "hybrid-linear", "moe"}
    hybrid_cases = [c.case_id for c in lower(hybrid).workload.calls]
    assert any(cid.startswith("kda:") for cid in hybrid_cases)
    moe_cases = [c.case_id for c in lower(moe).workload.calls]
    assert any(cid.endswith("_g8") for cid in moe_cases), "grouped expert matmul"


def test_all_kernels_spec_covers_the_complete_frozen_inventory():
    from akt.benchmark.model_specs import all_kernels_spec
    from akt.benchmark.suites import load_cases

    lowered = lower(all_kernels_spec())
    assert not lowered.new_case_ids
    assert {c.case_id for c in lowered.workload.calls} == {
        case.case_id for case in load_cases("full")
    }
    # phases/seeds inherited from the frozen callsites, not invented
    frozen = {c.case_id: c for w in MODEL_WORKLOADS for c in w.calls}
    for call in lowered.workload.calls:
        assert call.phase == frozen[call.case_id].phase
        assert call.seed == frozen[call.case_id].seed


def test_registry_coverage_maps_every_served_model_file():
    from akt.benchmark.model_specs import registry_coverage

    rows = registry_coverage()
    files = {row["model_file"] for row in rows}
    assert {"qwen3", "kimi_linear", "deepseek_v3", "llama"} <= files
    assert any("UNCOVERED" in row["coverage"] for row in rows), \
        "coverage report must stay honest about uncovered model families"


def test_extended_workloads_env_gate_default_off_and_fail_closed():
    import subprocess, sys, os, json as _json

    code = (
        "import json\n"
        "from akt.benchmark.model_workloads import MODEL_WORKLOADS, "
        "contract_fingerprint, validate_workloads\n"
        "validate_workloads()\n"
        "print(json.dumps({'n': len(MODEL_WORKLOADS), 'fp': contract_fingerprint()}))\n"
    )
    env = dict(os.environ, PYTHONPATH="python:.", PALLAS_INTERPRET="1")
    env.pop("AKT_EXTENDED_WORKLOADS", None)
    base = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, check=True)
    base_row = _json.loads(base.stdout.strip().splitlines()[-1])
    assert base_row["n"] == 3

    env["AKT_EXTENDED_WORKLOADS"] = "all-kernels"
    ext = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    ext_row = _json.loads(ext.stdout.strip().splitlines()[-1])
    assert ext_row["n"] == 4
    assert ext_row["fp"] != base_row["fp"], "adoption must change the fingerprint (forces rebaseline)"

    env["AKT_EXTENDED_WORKLOADS"] = "no-such-workload"
    bad = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    assert bad.returncode != 0, "unknown extended workload must fail closed"
