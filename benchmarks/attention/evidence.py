#!/usr/bin/env python3
"""Attention evidence commands. Incomplete resources remain explicit and cannot certify a campaign."""
import argparse
import ast
import importlib.metadata
import json
import pathlib
import platform
import re
import subprocess
import sys
from fixtures import create, load, sha256
from statistics_protocol import analyze

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((pathlib.Path(__file__).parent / 'protocol.json').read_text())


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def write(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('Evidence is immutable; choose a fresh output path: ' + str(path))
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def inventory(repo):
    repo = pathlib.Path(repo).resolve()
    patterns = {
        'attention-call': r'\.(?:Attention|ScaledDotProductAttention(?:TokenMajor)?|FlashAttention(?:Dev)?)\s*\(',
        'shared-layer': r'AttentionLayer|JointAttention',
        'head-repeat': r'RepeatInterleave|RepeatKv',
        'manual-softmax-review': r'Softmax|softmax',
        'cache-prefix': r'\.(?:KeyPrefix|ValuePrefix|KeyBuffer|ValueBuffer)\(',
    }
    rows = []
    files = command('git', '-C', str(repo), 'ls-files', 'src').splitlines()
    for file in files:
        if not file.endswith('.cs'):
            continue
        for line, text in enumerate((repo / file).read_text(encoding='utf-8-sig').splitlines(), 1):
            kinds = [kind for kind, pattern in patterns.items() if re.search(pattern, text)]
            if kinds:
                rows.append({'file': file, 'line': line, 'kinds': kinds, 'source': text.strip(),
                             'classification': 'source candidate; runtime attribution and review required'})
    return {'schema': 1, 'commit': command('git', '-C', str(repo), 'rev-parse', 'HEAD'), 'callers': rows}


def environment(kitchen, wheelhouse):
    import site
    import torch
    kitchen = pathlib.Path(kitchen).resolve()
    commit = command('git', '-C', str(kitchen), 'rev-parse', 'HEAD')
    if commit != PROTOCOL['kitchen_commit']:
        raise ValueError('Kitchen commit differs from K0')
    if command('git', '-C', str(kitchen), 'diff', 'HEAD', '--'):
        raise ValueError('Kitchen source has tracked modifications; this is not K0')
    extensions = {}
    reasons = []
    capabilities = {}
    installed_sources = {}
    native_ready = False
    try:
        import comfy_kitchen
        from comfy_kitchen.backends import cuda
        import comfy_kitchen.sage_attention as sage
        import comfy_kitchen.flash_attention as flash
        installed = pathlib.Path(comfy_kitchen.__file__).resolve().parent
        extensions = {str(path): sha256(path) for pattern in ('*.pyd', '*.so', '*.dll') for path in installed.rglob(pattern)}
        capabilities = {'extension_available': cuda._EXT_AVAILABLE,
                        'sage_native': sage.is_available(), 'decode_native': flash.is_available(),
                        'gqa_bf16_native': flash.flash_attention_decode_gqa_is_available()}
        native_ready = bool(cuda._EXT_AVAILABLE and extensions)
        if cuda._EXT_AVAILABLE and not extensions: reasons.append('native extension hashes are missing')
        for relative in command('git', '-C', str(kitchen), 'ls-files', 'comfy_kitchen').splitlines():
            if not relative.endswith('.py'): continue
            file = installed / pathlib.Path(relative).relative_to('comfy_kitchen')
            if not file.exists() or sha256(file) != sha256(kitchen / relative):
                reasons.append('installed Python source differs from K0: ' + relative)
            else: installed_sources[relative] = sha256(file)
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
        reasons.append('Kitchen import/capability query failed: ' + str(error))
    wheels = {path.name: sha256(path) for path in pathlib.Path(wheelhouse).glob('*.whl')}
    packages = sorted({(dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()})
    config = pathlib.Path(sys.prefix) / 'pyvenv.cfg'
    isolated = sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE and config.exists()
    isolated = isolated and 'include-system-site-packages = false' in config.read_text().lower()
    if not isolated: reasons.append('interpreter is not an isolated user-site-disabled virtual environment')
    if not wheels: reasons.append('dependency wheel hashes are missing')
    if not torch.cuda.is_available(): reasons.append('CUDA unavailable')
    identities = {}
    for name, args in {'nvcc': ['nvcc', '--version'],
                       'driver': ['nvidia-smi', '--query-gpu=name,driver_version,pci.bus_id', '--format=csv,noheader']}.items():
        try: identities[name] = command(*args)
        except (OSError, subprocess.CalledProcessError) as error: identities[name] = 'unavailable: ' + str(error)
    native = {str(path): sha256(path) for path in (pathlib.Path(torch.__file__).parent / 'lib').glob('*')
              if path.suffix.lower() in ('.dll', '.so') or '.so.' in path.name}
    return {'schema': 1, 'kind': 'K0', 'kitchen_commit': commit, 'python': sys.version,
            'executable': sys.executable, 'python_sha256': sha256(sys.executable), 'platform': platform.platform(),
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'packages': [{'name': name, 'version': version} for name, version in packages],
            'wheel_hashes': wheels, 'extension_hashes': extensions, 'isolated': isolated,
            'python_source_hashes': installed_sources, 'native_ready': native_ready,
            'readiness_scope': 'public GPU API probes; native availability is evaluated separately for each cell',
            'toolchain_driver': identities, 'torch_native_hashes': native,
            'cuda_available': torch.cuda.is_available(), 'capabilities': capabilities,
            'gpu': [{'name': torch.cuda.get_device_name(i), 'capability': list(torch.cuda.get_device_capability(i)),
                     'vram_bytes': torch.cuda.get_device_properties(i).total_memory}
                    for i in range(torch.cuda.device_count())],
            'ready': not reasons, 'blocking_reasons': reasons}


def source_matrix(kitchen, cohort):
    kitchen = pathlib.Path(kitchen)
    if command('git', '-C', str(kitchen), 'rev-parse', 'HEAD') != PROTOCOL['kitchen_commit']:
        raise ValueError('Feature support must come from pinned Kitchen source')
    declarations = {}
    for relative in ('comfy_kitchen/sage_attention.py', 'comfy_kitchen/flash_attention.py',
                     'comfy_kitchen/backends/eager/sol_attn.py'):
        source = (kitchen / relative).read_text()
        declarations[relative] = {'sha256': sha256(kitchen / relative), 'functions': [
            {'name': node.name, 'line': node.lineno, 'doc': ast.get_docstring(node)}
            for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]}
    features = json.loads((pathlib.Path(__file__).parent / 'features.json').read_text())
    source_files = command('git', '-C', str(kitchen), 'ls-files', 'comfy_kitchen').splitlines()
    native_sources = {path: sha256(kitchen / path) for path in source_files
                      if path.endswith(('.cu', '.cuh', '.cpp', '.h'))
                      and any(word in path.lower() for word in ('attention', 'attn', 'flash', 'hadamard', 'convrot'))}
    from operation_cells import cells
    return {'schema': 1, 'kitchen_commit': PROTOCOL['kitchen_commit'], 'cohort': cohort,
            'source_declarations': declarations, 'features': features,
            'native_source_hashes': native_sources,
            'cells': cells(cohort, declarations),
            'upstream_tests': {str(path.relative_to(kitchen)): sha256(path)
                               for path in sorted((kitchen / 'tests').rglob('*attention*.py'))},
            'cell_requirement': 'Each concrete runtime cell needs source admission, native/public execution, oracle and timing eligibility.',
            'runtime_status': 'Not executed', 'exceptions_reviewed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fixture')
    p.add_argument('--out', required=True)
    p.add_argument('--id', required=True)
    for name, default in (('batch', 1), ('heads', 4), ('kv-heads', 4), ('sq', 32), ('skv', 48), ('dim', 64)):
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--dtype', choices=('f32', 'f16', 'bf16'), default='f16')
    p.add_argument('--precision', choices=('f32', 'tf32', 'f16', 'bf16', 'int8'))
    p = sub.add_parser('verify-fixture'); p.add_argument('path')
    p = sub.add_parser('cell-fixture')
    p.add_argument('--matrix', required=True); p.add_argument('--cell', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('inventory'); p.add_argument('--repo', default=str(ROOT)); p.add_argument('--out', required=True)
    p = sub.add_parser('lock-environment')
    p.add_argument('--kitchen', required=True); p.add_argument('--wheelhouse', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('matrix')
    p.add_argument('--kitchen', required=True); p.add_argument('--cohort', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('statistics')
    p.add_argument('--pairs', required=True); p.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'fixture':
        print(create(args.out, args.id, args.batch, args.heads, args.kv_heads, args.sq, args.skv, args.dim, args.dtype,
                     precision=args.precision))
    elif args.command == 'verify-fixture':
        print(json.dumps({'case_id': load(args.path)['case_id'], 'verified': True}))
    elif args.command == 'cell-fixture':
        from cell_fixture import materialize
        matrix = json.loads(pathlib.Path(args.matrix).read_text())
        cell, = [cell for cell in matrix['cells'] if cell['id'] == args.cell]
        print(materialize(args.out, cell))
    elif args.command == 'inventory': write(args.out, inventory(args.repo))
    elif args.command == 'lock-environment':
        result = environment(args.kitchen, args.wheelhouse)
        write(args.out, result)
        if not result['ready']: return 2
    elif args.command == 'matrix': write(args.out, source_matrix(args.kitchen, args.cohort))
    elif args.command == 'statistics':
        pairs = json.loads(pathlib.Path(args.pairs).read_text())
        result = analyze(pairs['sessions'], pairs['confirmation'], pairs['claim'])
        write(args.out, result)
        if not result['passed']: return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
