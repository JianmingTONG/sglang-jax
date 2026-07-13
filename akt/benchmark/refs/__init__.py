"""FROZEN authoritative correctness references for the kernel suite.

Each module here sources its reference implementation + tolerance + canonical
inputs from sglang-jax's OWN kernel/test code (imported or faithfully mirrored
with a citation), and names the repo pytest nodeid (NATIVE_TEST) that verifies the
Pallas kernel in interpret. The EDITABLE runners (akt/core/runners/) import from
here and own only the DesignSpace + run-mapping, so a capability can add tuning
knobs but cannot weaken the correctness contract (this dir is FROZEN).
"""
