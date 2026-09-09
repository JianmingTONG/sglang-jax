"""Single-block decode guard and arithmetic checks without TPU/model execution.

Extract only the changed helpers; compare with independent softmax equations.
These source checks do not replace hardware numerical/performance validation.
"""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
from enum import Enum
import numpy as np
import ml_dtypes



def test_single_block_decode_guard_and_math():
    path = Path(__file__).resolve().parents[2] / 'python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_ragged_paged_attention_kernel_loop')
    guard = next(n.value for n in kernel.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'single_kv_block' for t in n.targets))
    class RpaCase(Enum):
        DECODE = 0
        PREFILL = 1
        MIXED = 2
    expr = compile(ast.Expression(guard), str(path), 'eval')
    guard_cases = 0
    for case, sizes, pages, page_size, sliding, sink in itertools.product(
        RpaCase, [(770,770),(1024,512),(384,384)], [1,769,771], [1,16], [None,128], [None,object()]
    ):
        bkv_sz, bkv_csz = sizes
        env = dict(RpaCase=RpaCase, case=case, bkv_sz=bkv_sz, bkv_csz=bkv_csz, num_page_indices=pages, page_size=page_size, sliding_window=sliding, attention_sink_ref=sink)
        enabled = eval(expr, env)
        if enabled:
            assert case is RpaCase.DECODE and sliding is None and sink is None
            for kv_len in [1, pages*page_size]:
                assert (kv_len+bkv_sz-1)//bkv_sz == 1
                assert (kv_len+bkv_csz-1)//bkv_csz == 1
        else:
            assert case is not RpaCase.DECODE or sliding is not None or sink is not None or bkv_sz != bkv_csz or pages*page_size > bkv_csz
        guard_cases += 1
    captured = dict(RpaCase=RpaCase, case=RpaCase.DECODE, bkv_sz=770, bkv_csz=770, num_page_indices=769, page_size=1, sliding_window=None, attention_sink_ref=None)
    assert eval(expr, captured)

    names = {'flash_attention_step1_qk_softmax','flash_attention_step2_pv','broadcast_minor','mask_and'}
    functions = [n for n in kernel.body if isinstance(n, ast.FunctionDef) and n.name in names]
    class ArrayAPI:
        def __init__(self): self.exp_calls = 0
        def __getattr__(self, name): return getattr(np, name)
        def concat(self, *args, **kwargs): return np.concatenate(*args, **kwargs)
        def matmul(self, a, b, preferred_element_type=None):
            return np.matmul(a.astype(preferred_element_type), b.astype(preferred_element_type))
        def exp(self, x):
            self.exp_calls += 1
            return np.exp(x)
    class Ref:
        def __init__(self, value):
            self.value = value.copy()
            self.shape = value.shape
            self.reads = 0
        def __getitem__(self, key):
            self.reads += 1
            return self.value[key]
        def __setitem__(self, key, value): self.value[key] = value

    def iota(dtype, shape, axis):
        view = [1]*len(shape)
        view[axis] = shape[axis]
        return np.broadcast_to(np.arange(shape[axis],dtype=dtype).reshape(view), shape)

    api = ArrayAPI()
    env = dict(jnp=api, lax=SimpleNamespace(broadcasted_iota=iota), single_kv_block=True,
               num_q_heads_per_kv_head=4, head_dim=128, bkv_csz=770,
               q_scale=None, k_scale=None, v_scale=None, sm_scale=1/np.sqrt(128),
               soft_cap=None, softmax_dtype=None, use_causal_mask=False,
               skip_kv_mask=False, sliding_window=None, tpu_version=7,
               mask_value=-0.7*float(np.finfo(np.float32).max),
               get_dtype_packing=lambda dt: 32//(np.dtype(dt).itemsize*8),
               align_to=lambda x,y: (x+y-1)//y*y)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),env)
    rng = np.random.default_rng(1729)
    checked = 0
    for dtype, kv_len, poison_padding in itertools.product([np.float32,ml_dtypes.bfloat16], [1,127,128,511,512,513,767,768,769], [False,True]):
        env['out_dtype'] = dtype
        q = rng.normal(size=(4,128)).astype(dtype)
        k = rng.normal(size=(770,128)).astype(dtype)
        v = rng.normal(size=(770,128)).astype(dtype)
        if poison_padding:
            k[kv_len:] = np.nan
            v[kv_len:] = np.nan
        l = Ref(np.zeros((4,128),dtype=dtype))
        m = Ref(np.full((4,128),-np.inf,dtype=dtype))
        acc = Ref(np.zeros((4,128),dtype=dtype))
        api.exp_calls = 0
        p, masked_v, correction = env['flash_attention_step1_qk_softmax'](
            q,k,v,l,m,processed_q_len=np.int32(kv_len-1),processed_kv_len=np.int32(0),effective_kv_len=np.int32(kv_len))
        env['flash_attention_step2_pv'](p,masked_v,correction,acc)
        assert correction is None and api.exp_calls == 1
        assert (l.reads,m.reads,acc.reads) == (0,0,0)
        scores = np.matmul(q.astype(np.float32),k.astype(np.float32).T)
        scores *= env['sm_scale']
        scores[:,kv_len:] = env['mask_value']
        expected_p = np.exp(scores-np.max(scores,axis=1,keepdims=True))
        expected_v = np.where(np.arange(770)[:,None] < kv_len,v,0)
        expected_l = np.broadcast_to(expected_p.sum(axis=1,keepdims=True),(4,128)).astype(dtype)
        expected_acc = (expected_p @ expected_v.astype(np.float32)).astype(dtype)
        np.testing.assert_array_equal(p,expected_p)
        np.testing.assert_array_equal(l.value,expected_l)
        np.testing.assert_array_equal(acc.value,expected_acc)
        # Existing first-block recurrence has exp(-inf - finite_max)=0;
        # its additions to zero do not alter finite row sums or numerators.
        checked += 1
