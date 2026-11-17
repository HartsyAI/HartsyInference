"""Materialize frozen K0 probe definitions into shared bytes, including complete merge and sparse metadata."""
import json
import math
import pathlib
import random
from fixtures import create, decode, encode, load, sha256


def materialize(root, cell):
    p = cell['parameters']
    fields = {key: p[key] for key in ('batch', 'heads', 'kv_heads', 'sq', 'skv', 'dim', 'dtype')}
    path = create(root, cell['id'], **fields, precision=p['precision'])
    document = load(path)
    spec = document['call']
    spec['causal'] = p.get('causal', False)
    spec['kv_lengths'] = p.get('kv_lengths', [p['skv']] * p['batch'])
    spec['query_offset'] = p['skv'] - p['sq'] if spec['causal'] else 0
    spec['api'] = cell['api']
    spec['reuse_count'] = p.get('reuse', 1)
    root = pathlib.Path(root)

    def tensor(name, values, shape, dtype):
        file = root / (name + '.bin')
        with file.open('wb') as output:
            for value in values: output.write(encode(value, dtype))
        stride = 1
        strides = []
        for n in reversed(shape): strides.insert(0, stride); stride *= n
        document['tensors'][name] = {'file': file.name, 'sha256': sha256(file), 'dtype': dtype,
                                      'shape': shape, 'storage_shape': shape, 'strides': strides, 'offset': 0}

    if p.get('key_shift'):
        descriptor = document['tensors']['k']
        values = decode((root / descriptor['file']).read_bytes(), descriptor['dtype'])
        tensor('k', (value + p['key_shift'] for value in values), descriptor['shape'], descriptor['dtype'])

    if p.get('layout') == 'strided':
        for name in ('q', 'k', 'v'):
            descriptor = document['tensors'][name]
            b, h, s, d = descriptor['shape']
            old = decode((root / descriptor['file']).read_bytes(), descriptor['dtype'])
            values = [0.] * (b*s*h*d)
            for batch in range(b):
                for head in range(h):
                    for row in range(s):
                        begin = ((batch*h+head)*s+row)*d
                        target = ((batch*s+row)*h+head)*d
                        values[target:target+d] = old[begin:begin+d]
            tensor(name, values, [b, h, s, d], descriptor['dtype'])
            document['tensors'][name].update(storage_shape=[b, s, h, d], strides=[s*h*d, d, h*d, 1], layout='TokenMajor')
    if 'mask_dtype' in p:
        shapes = {'key': [1, 1, 1, p['skv']], 'query': [1, 1, p['sq'], p['skv']],
                  'head': [1, p['heads'], 1, p['skv']], 'batch': [p['batch'], 1, 1, p['skv']],
                  'full': [p['batch'], p['heads'], p['sq'], p['skv']]}
        shape = shapes[p['mask_broadcast']]
        values = []
        for index in range(math.prod(shape)):
            row = (index // shape[-1]) % shape[-2]
            excluded = (shape[-2] > 1 and row == 0) or index % shape[-1] >= p['skv']-3
            values.append(not excluded if p['mask_dtype'] == 'bool' else (-math.inf if excluded else 0.125))
        tensor('bias', values, shape, p['mask_dtype'])
        document['coverage_note'] = 'Key-only broadcasts exclude keys; query-varying masks also exclude a complete row.'
    if cell['api'] in ('tree-merge', 'step-merge'):
        import torch
        from fixtures import logical_values
        from operation_oracle import dense
        data = {name: torch.tensor(logical_values(root, document['tensors'][name])).reshape(
            document['tensors'][name]['shape']) for name in ('q', 'k', 'v')}
        prefix, lse = dense(data['q'], data['k'], data['v'], spec['scale'])
        # Prefix public output is token-major and rounded exactly once to its declared output dtype.
        tensor('prefix_output', prefix.transpose(1, 2).contiguous().flatten().tolist(),
               [p['batch'], p['sq'], p['heads'] * p['dim']], p['dtype'])
        tensor('prefix_lse', lse.flatten().tolist(), list(lse.shape), 'f32')
        rng = random.Random(274019)
        shape = [p['batch'], p['kv_heads'], p['sq'], p['dim']]
        for name in ('step_k', 'step_v'):
            tensor(name, (rng.uniform(-1, 1) for _ in range(math.prod(shape))), shape, p['dtype'])
        spec['tree_mask'] = p['tree_mask']
    if cell['api'] == 'sol':
        options = {'tau': 1., 'tail': p.get('tail', True), 'topk_ratio': .25 if p.get('routing') == 'topk' else 0.}
        option = p.get('option')
        blocks = (p['sq'] + 63) // 64
        if option in ('key_bias_bool', 'key_bias_additive'):
            # Bias affects one sink-covered block, preserving the pooled-tail contract.
            options['sink_blocks'] = [0, 1]
            dtype = 'bool' if option.endswith('bool') else 'f32'
            tensor('bias', [j >= 3 if dtype == 'bool' else (.125 if j < 64 else 0.)
                            for j in range(p['skv'])], [p['skv']], dtype)
        if option == 'sink_blocks': options['sink_blocks'] = [0, 2]
        if option == 'sink_q': options['sink_q'] = [1, 3]
        if option == 'block_len': options['block_len'] = [37] * (blocks-1) + [p['sq'] % 64 or 64]
        if option == 'coarse_gate': options['coarse_gate'] = .25
        if option == 'token_aug': options['token_aug'] = 1
        spec['sol'] = options
        document['coverage_note'] = 'Dead query rows for explicit block_len are unspecified and excluded from quality comparisons.'
    document['cell_id'] = cell['id']
    document['cohort'] = cell['cohort']
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    load(path)
    return path
