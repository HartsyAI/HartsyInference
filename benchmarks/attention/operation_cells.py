"""Concrete source-derived K0 probes. Runtime execution, rejection, and numerical review remain separate facts."""
import itertools


def cells(cohort, declarations):
    result = []
    def add(name, api, feature, parameters, native):
        source = 'comfy_kitchen/backends/eager/sol_attn.py' if api == 'sol' else (
            'comfy_kitchen/sage_attention.py' if api in ('sage', 'prequantized') else 'comfy_kitchen/flash_attention.py')
        result.append({'id': name, 'cohort': cohort, 'api': api, 'feature': feature, 'parameters': parameters,
                       'source': {'path': source, 'sha256': declarations[source]['sha256']},
                       'public_admission': 'Pinned public validation and upstream tests; runtime probe required',
                       'native_admission': native, 'native_available': None, 'executed_path': None,
                       'k0_outcome': 'Not executed', 'oracle_result': None,
                       'hartsy_required': {'cuda': 'correct execution', 'vulkan': 'correct execution'},
                       'oracle': 'independent float64 dense/merge/block-tail reference',
                       'numerical_acceptance': {'reviewed': False, 'policy': parameters.get('precision', 'bf16')},
                       'matched_quality_eligible': False,
                       'timing_boundaries': ['resident', 'public']})
    sage_native = 'CUDA extension; SM75+; matching F32/F16/BF16; D 1..256; contiguous last dimension'
    for dtype, dim, grouping, layout in itertools.product(('f32', 'f16', 'bf16'), (64, 128, 256),
                                                         ((4, 4), (4, 2), (4, 1)), ('head-major', 'strided')):
        h, hk = grouping
        add(f'sage-{dtype}-d{dim}-h{h}-{hk}-{layout}', 'sage', 'sage-inputs/grouping',
            dict(batch=1, heads=h, kv_heads=hk, sq=32, skv=48, dim=dim, dtype=dtype, layout=layout,
                 precision='int8', output_dtype=dtype), sage_native)
    for dim in (1, 40, 80, 96, 112, 160, 255, 257):
        add(f'sage-padding-d{dim}', 'sage', 'sage-padding',
            dict(batch=1, heads=4, kv_heads=2, sq=16, skv=48, dim=dim, dtype='f16', precision='int8'), sage_native)
        if dim == 257:
            result[-1]['expected_admission'] = 'K0-rejected: public maximum head dimension is 256'
            result[-1]['hartsy_required'] = {'cuda': 'not an execution-parity cell', 'vulkan': 'not an execution-parity cell'}
    for sq, skv in ((16, 48), (48, 16)):
        add(f'sage-length-{sq}-{skv}', 'sage', 'sage-grouping',
            dict(batch=1, heads=4, kv_heads=1, sq=sq, skv=skv, dim=64, dtype='f16', precision='int8'), sage_native)
    for dim, length, shifted in itertools.product((64, 80, 128, 160, 256), (257, 1025), (False, True)):
        add(f'sage-rotation-d{dim}-kv{length}-shift{int(shifted)}', 'sage', 'sage-rotation',
            dict(batch=1, heads=4, kv_heads=2, sq=16, skv=length, dim=dim, dtype='f16', precision='int8',
                 key_shift=25. if shifted else 0., representative_key_probe=shifted), sage_native)
    for dtype, broadcast in itertools.product(('bool', 'f16', 'bf16', 'f32'), ('key', 'query', 'head', 'batch', 'full')):
        add(f'sage-mask-{dtype}-{broadcast}', 'sage', 'sage-masks',
            dict(batch=2, heads=4, kv_heads=2, sq=16, skv=48, dim=64, dtype='f16', precision='int8',
                 mask_dtype=dtype, mask_broadcast=broadcast, excluded_row=True), sage_native)
    for dim, reuse in itertools.product((64, 128, 256), (1, 2, 4, 8)):
        add(f'sage-prequant-d{dim}-reuse{reuse}', 'prequantized', 'sage-rotation/prequantized',
            dict(batch=1, heads=4, kv_heads=2, sq=32, skv=48, dim=dim, dtype='bf16', precision='int8', reuse=reuse), sage_native)
        result[-1]['timing_boundaries'] += ['preparation', 'reuse']
    for api, dim, hk, causal in itertools.product(('decode', 'decode-gqa'), (64, 128, 256), (4, 2, 1), (False, True)):
        if api == 'decode' and causal: continue
        add(f'{api}-d{dim}-hk{hk}-causal{int(causal)}', api, 'decode-bf16/lse',
            dict(batch=3, heads=4, kv_heads=hk, sq=1 if api == 'decode' else 4, skv=73, dim=dim,
                 dtype='bf16', precision='bf16', causal=causal, kv_lengths=[0, 53, 73], return_lse=api == 'decode-gqa'),
            'CUDA extension SM80+; BF16; D64/128/256 for decode; D256 for native GQA, other GQA uses torch GPU fallback')
    for api in ('tree-merge', 'step-merge'):
        add(api, api, 'decode-' + api, dict(batch=1, heads=4, kv_heads=2, sq=4, skv=53, dim=256,
            dtype='bf16', precision='bf16', tree_mask=[1, 3, 5, 13], prefix_counted_once=True),
            'CUDA extension SM80+; BF16 D256, otherwise torch GPU fallback')
    for routing, tail, partial in itertools.product(('tau', 'topk'), (False, True), (False, True)):
        add(f'sol-{routing}-tail{int(tail)}-partial{int(partial)}', 'sol', 'sol-general',
            dict(batch=1, heads=2, kv_heads=2, sq=257 if partial else 256, skv=257 if partial else 256,
                 dim=128, dtype='bf16', precision='bf16', routing=routing, tail=tail),
            'Pinned registry: head dimension 128; record native versus eager GPU fallback independently')
    for option in ('key_bias_bool', 'key_bias_additive', 'sink_blocks', 'sink_q', 'block_len', 'coarse_gate', 'token_aug'):
        add('sol-' + option, 'sol', 'sol-general', dict(batch=1, heads=2, kv_heads=2, sq=257, skv=257,
            dim=128, dtype='bf16', precision='bf16', option=option),
            'Pinned registry: head dimension 128; option-specific public/native restrictions require runtime probes')
    add('sol-negative-d64', 'sol', 'sol-general', dict(batch=1, heads=2, kv_heads=2, sq=257, skv=257,
        dim=64, dtype='f16', precision='f16'), 'Common public call rule requires head dimension 128')
    result[-1]['expected_admission'] = 'K0-rejected: sol_attn_common_call_rule'
    result[-1]['hartsy_required'] = {'cuda': 'not an execution-parity cell', 'vulkan': 'not an execution-parity cell'}
    return result
