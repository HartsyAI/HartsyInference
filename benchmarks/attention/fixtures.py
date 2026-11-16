"""Shared raw tensor fixtures. A fixture is written once; both engines consume its exact storage bytes."""
import hashlib
import itertools
import json
import math
import pathlib
import random
import struct


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def encode(value, dtype):
    if dtype == 'bool':
        return struct.pack('<?', value)
    if dtype == 'i32':
        return struct.pack('<i', value)
    if dtype == 'f32':
        return struct.pack('<f', value)
    if dtype == 'f16':
        return struct.pack('<e', value)
    if dtype == 'bf16':
        bits, = struct.unpack('<I', struct.pack('<f', value))
        return struct.pack('<H', ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff)
    raise ValueError('Unsupported fixture dtype: ' + dtype)


def decode(data, dtype):
    if dtype in ('bool', 'i32'):
        return [value[0] for value in struct.iter_unpack('<?' if dtype == 'bool' else '<i', data)]
    if dtype in ('f32', 'f16'):
        return [value[0] for value in struct.iter_unpack('<f' if dtype == 'f32' else '<e', data)]
    if dtype == 'bf16':
        return [struct.unpack('<f', struct.pack('<I', value[0] << 16))[0]
                for value in struct.iter_unpack('<H', data)]
    raise ValueError('Unsupported fixture dtype: ' + dtype)


def tensor_path(root, descriptor):
    name = descriptor['file']
    if not name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError('Tensor files must be direct children of their fixture directory')
    path = pathlib.Path(root) / name
    if path.is_symlink():
        raise ValueError('Fixture tensor symlinks are not allowed')
    return path


def load(path):
    path = pathlib.Path(path)
    document = json.loads(path.read_text())
    if document.get('schema') != 1:
        raise ValueError('Unsupported fixture schema')
    for name, descriptor in document['tensors'].items():
        file = tensor_path(path.parent, descriptor)
        if sha256(file) != descriptor['sha256']:
            raise ValueError('Fixture hash mismatch: ' + name)
        shape, strides = descriptor['shape'], descriptor['strides']
        if not 1 <= len(shape) <= 6 or len(strides) != len(shape) or any(n <= 0 for n in shape) or any(n < 0 for n in strides):
            raise ValueError('Invalid fixture geometry: ' + name)
        physical = descriptor['storage_shape']
        if not physical or any(n <= 0 for n in physical):
            raise ValueError('Invalid physical storage: ' + name)
        count = math.prod(physical)
        extent = descriptor['offset'] + sum((n - 1) * s for n, s in zip(shape, strides)) + 1
        if descriptor['offset'] < 0 or extent > count:
            raise ValueError('Fixture view exceeds storage: ' + name)
        size = {'f32': 4, 'f16': 2, 'bf16': 2, 'bool': 1, 'i32': 4}.get(descriptor['dtype'])
        if size is None or file.stat().st_size != count * size:
            raise ValueError('Invalid fixture storage: ' + name)
    return document


def create(root, case_id, batch=1, heads=4, kv_heads=4, sq=32, skv=48, dim=64, dtype='f16', seed=1234, precision=None,
           route='Dense', kv_dtype=None, causal=False, query_offset=None, legacy_asserted_bounded=False):
    if batch <= 0 or heads <= 0 or kv_heads <= 0 or heads % kv_heads or min(sq, skv, dim) <= 0:
        raise ValueError('Invalid attention geometry')
    kv_dtype = kv_dtype or dtype
    if route not in ('Dense', 'Flash') or dtype not in ('f32', 'f16', 'bf16') or kv_dtype not in ('f32', 'f16', 'bf16'):
        raise ValueError('Unsupported fixture route or dtype')
    if query_offset is None: query_offset = skv - sq if causal else 0
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    tensors = {}
    for name, shape in (('q', [batch, heads, sq, dim]), ('k', [batch, kv_heads, skv, dim]),
                        ('v', [batch, kv_heads, skv, dim])):
        storage_dtype = dtype if name == 'q' else kv_dtype
        file = root / (name + '.bin')
        with file.open('wb') as output:
            for _ in range(math.prod(shape)):
                output.write(encode(rng.uniform(-1, 1), storage_dtype))
        b, h, s, d = shape
        tensors[name] = {'file': file.name, 'sha256': sha256(file), 'dtype': storage_dtype,
                         'shape': shape, 'storage_shape': shape, 'strides': [h*s*d, s*d, d, 1], 'offset': 0}
    precision = precision or dtype
    tolerances = {'f32': 2e-5, 'tf32': 2e-3, 'f16': 3e-3, 'bf16': 8e-3, 'int8': 1e-2}
    if precision not in tolerances:
        raise ValueError('Unknown frozen numerical policy')
    document = {'schema': 1, 'case_id': case_id, 'tensors': tensors,
                'call': {'route': route, 'scale': 1 / math.sqrt(dim), 'causal': causal,
                         'kv_length': skv, 'query_offset': query_offset, 'legacy_asserted_bounded': legacy_asserted_bounded,
                         'precision': precision},
                'acceptance': {'source': 'docs/ATTENTION_CONTRACT.md section 8',
                               'rel_l2': tolerances[precision],
                               'mean_abs': 2e-3 if precision == 'int8' else None,
                               'reviewed': False},
                'generator': {'seed': seed, 'implementation': 'Python random.Random; persisted bytes are authoritative'}}
    path = root / 'fixture.json'
    # Written as UTF-8 with LF line endings on every platform: the published manifest is pinned by its bytes.
    path.write_bytes((json.dumps(document, indent=2) + '\n').encode('utf-8'))
    return path


def logical_values(root, descriptor):
    storage = decode(tensor_path(root, descriptor).read_bytes(), descriptor['dtype'])
    return [storage[descriptor['offset'] + sum(i*s for i, s in zip(indices, descriptor['strides']))]
            for indices in itertools.product(*(range(n) for n in descriptor['shape']))]
