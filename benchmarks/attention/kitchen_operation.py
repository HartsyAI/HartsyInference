"""Pinned Kitchen public-operation execution, with separate device and host timing cells."""
import argparse
import json
import math
import pathlib
import sys
import time
from fixtures import load, sha256, tensor_path
from operation_oracle import dense, merge, sol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--api', choices=('sage', 'prequantized', 'decode', 'decode-gqa', 'tree-merge', 'step-merge', 'sol'), default='sage')
    parser.add_argument('--boundary', choices=('public', 'resident', 'preparation', 'reuse'), default='public')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mandatory', action='store_true')
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args(argv)
    if args.trials <= 0: raise ValueError('At least one trial is required')
    output = pathlib.Path(args.out)
    if output.exists(): raise ValueError('Results are immutable; choose a fresh output path')
    result = {'schema': 1, 'k0_status': 'Not executed', 'api': args.api, 'boundary': args.boundary,
              'diagnostics': args.diagnostics, 'timing_eligible': not args.diagnostics,
              'fixture_sha256': sha256(args.fixture), 'environment_sha256': sha256(args.environment)}
    directory = pathlib.Path(__file__).resolve().parent
    result['harness_source_hashes'] = {name: sha256(directory / name) for name in (
        'kitchen_operation.py', 'fixtures.py', 'operation_oracle.py', 'protocol.json')}
    try:
        environment = json.loads(pathlib.Path(args.environment).read_text())
        if not environment.get('ready') or environment['python_sha256'] != sha256(sys.executable):
            raise RuntimeError('The current isolated native environment is not locked and ready')
        for path, expected in environment['extension_hashes'].items():
            if sha256(path) != expected: raise RuntimeError('A locked Kitchen extension changed')
        import torch
        import comfy_kitchen as ck
        import comfy_kitchen.sage_attention as sa
        import comfy_kitchen.flash_attention as fa
        if torch.__version__ != environment['torch']: raise RuntimeError('PyTorch differs from the lock')
        fixture = load(args.fixture)
        if args.mandatory and not fixture['acceptance']['reviewed']:
            raise RuntimeError('Mandatory measurement requires reviewed numerical acceptance')
        spec = fixture['call']
        if args.api == 'prequantized' and 'reuse_count' in spec:
            result['timing_eligible'] = False
            result['measurement_limitation'] = 'Finite reuse-count groups are not modeled; this pilot retains one prepared state across warmups/trials'
            if args.mandatory:
                raise NotImplementedError('Finite reuse-count timing needs validated group-state accounting')
        tensors = {}
        for name, descriptor in fixture['tensors'].items():
            dtype = {'f32': torch.float32, 'f16': torch.float16, 'bf16': torch.bfloat16,
                     'bool': torch.bool, 'i32': torch.int32}[descriptor['dtype']]
            storage = torch.frombuffer(bytearray(tensor_path(pathlib.Path(args.fixture).parent, descriptor).read_bytes()), dtype=dtype).to('cuda')
            tensors[name] = torch.as_strided(storage, descriptor['shape'], descriptor['strides'], descriptor['offset'])
        q, k, v = (tensors[name] for name in ('q', 'k', 'v'))
        result['cuda_extension_available'] = environment['capabilities'].get('extension_available', False)
        if args.api in ('sage', 'prequantized'):
            result['native_available'] = sa.is_available(q.device)
        elif args.api == 'decode': result['native_available'] = fa.is_available(q.device)
        elif args.api in ('decode-gqa', 'tree-merge', 'step-merge'):
            result['native_available'] = q.shape[-1] == 256 and fa.flash_attention_decode_gqa_is_available(q.device, q.dtype)
        bias = tensors.get('bias')
        lengths = spec.get('kv_lengths', [spec['kv_length']] * q.shape[0])
        length_tensor = torch.tensor(lengths, device='cuda', dtype=torch.int32)
        prepared = None
        if args.api in ('sage', 'prequantized'):
            if spec['causal']: raise NotImplementedError('Sage needs a frozen explicit mask for causal comparison')
            prepare = lambda: sa.prequantize_int8_attention(q, k, v, scale=spec['scale'], attn_mask=bias)
            if args.api == 'prequantized' and args.boundary != 'preparation': prepared = prepare()
            operation = (lambda: sa.int8_attention(q, k, v, scale=spec['scale'], attn_mask=bias)) if prepared is None else (
                lambda: sa.int8_attention_from_prequantized(prepared))
            canonical = lambda value: value
            reference, reference_lse = dense(q, k, v, spec['scale'], bias=bias)
        elif args.api in ('decode', 'decode-gqa'):
            if not math.isclose(spec['scale'], q.shape[3] ** -.5, rel_tol=1e-6):
                result.update(k0_status='K0-rejected', reason='Pinned decode API has no custom scale parameter')
                return 2
            if args.api == 'decode':
                if q.shape[2] != 1: raise ValueError('decode requires one query row')
                operation = lambda: fa.flash_attention_decode(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), length_tensor)
            else:
                operation = lambda: fa.flash_attention_decode_gqa(q, k, v, length_tensor, return_lse=True, causal=spec['causal'])
            canonical = lambda value: value.reshape(q.shape[0], q.shape[2], q.shape[1], q.shape[3]).transpose(1, 2)
            clamped = [max(0, min(k.shape[2], length)) for length in lengths]
            reference, reference_lse = dense(q, k, v, spec['scale'], clamped,
                                             spec['causal'] if args.api == 'decode-gqa' else False,
                                             [length-q.shape[2] for length in clamped])
        elif args.api in ('tree-merge', 'step-merge'):
            if not math.isclose(spec['scale'], q.shape[3] ** -.5, rel_tol=1e-6):
                result.update(k0_status='K0-rejected', reason='Pinned merge API has no custom scale parameter')
                return 2
            prefix = tensors['prefix_output'].reshape(q.shape[0], q.shape[2], -1)
            lse = tensors['prefix_lse'].reshape(q.shape[:3])
            destination = torch.empty_like(prefix)
            steps_k, steps_v = tensors['step_k'], tensors['step_v']
            mask = spec.get('tree_mask', [(1 << (i+1))-1 for i in range(q.shape[2])]) if args.api == 'tree-merge' else [
                (1 << (i+1))-1 for i in range(q.shape[2])]
            mask_tensor = torch.tensor(mask, dtype=torch.int32, device='cuda')
            operation = (lambda: fa.flash_attention_decode_tree_merge(prefix, lse, q, steps_k, steps_v, mask_tensor, destination)) if args.api == 'tree-merge' else (
                lambda: fa.flash_attention_decode_step_merge(prefix, lse, q, steps_k, steps_v, destination))
            canonical = lambda value: value.reshape(q.shape[0], q.shape[2], q.shape[1], q.shape[3]).transpose(1, 2)
            reference = merge(canonical(prefix), lse, q, steps_k, steps_v, spec['scale'], mask)
            reference_lse = None
        else:
            kwargs = dict(spec.get('sol', {}))
            if 'block_len' in kwargs: kwargs['block_len'] = torch.tensor(kwargs['block_len'], device='cuda', dtype=torch.int32)
            if 'coarse_gate' in kwargs:
                gate = torch.tensor(kwargs['coarse_gate'], device='cuda', dtype=q.dtype)
                kwargs['coarse_gate'] = gate.expand_as(q.transpose(1, 2)).contiguous()
            operation = lambda: ck.sol_attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           scale=spec['scale'], key_bias=bias, **kwargs)
            canonical = lambda value: value.transpose(1, 2)
            reference = sol(q, k, v, spec['scale'], spec.get('sol', {}), bias)
            reference_lse = None
        if args.boundary == 'preparation':
            if args.api != 'prequantized': raise ValueError('Preparation cell requires prequantized API')
            operation = prepare
        if args.boundary == 'reuse' and args.api != 'prequantized': raise ValueError('Reuse requires prequantized API')
        # Validation precedes timing; backend failures after successful admission are not negative admission cases.
        result['k0_status'] = 'K0-invalid'
        try:
            observed = operation()
            torch.cuda.synchronize()
        except (ValueError, TypeError) as error:
            result.update(k0_status='K0-rejected', reason=str(error))
            return 2
        observed = None
        for _ in range(9): operation()
        torch.cuda.synchronize()
        called = set()
        def trace_calls(frame, event, argument):
            module = frame.f_globals.get('__name__', '')
            if event == 'call' and module.startswith('comfy_kitchen'):
                called.add(module + ':' + frame.f_code.co_qualname)
        if args.diagnostics:
            previous_profile = sys.getprofile()
            try:
                sys.setprofile(trace_calls)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as trace:
                    operation()
                    torch.cuda.synchronize()
            finally:
                sys.setprofile(previous_profile)
            result['executed_python_functions'] = sorted(called)
            result['gpu_kernels'] = sorted({event.name for event in trace.events() if 'CUDA' in str(event.device_type)})
            result['aten_operations'] = sorted({event.name for event in trace.events() if event.name.startswith('aten::')})
            result['kernel_trace_available'] = bool(result['gpu_kernels'])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        elapsed = []
        for _ in range(args.trials):
            observed = None
            torch.cuda.synchronize()
            if args.boundary == 'resident':
                start.record()
                for _ in range(32):
                    observed = None
                    observed = operation()
                end.record(); end.synchronize()
                elapsed.append(start.elapsed_time(end) / 32)
            else:
                before = time.perf_counter_ns()
                observed = operation(); torch.cuda.synchronize()
                elapsed.append((time.perf_counter_ns() - before) / 1e6)
        state = observed if args.boundary == 'preparation' else prepared
        if args.boundary == 'preparation': observed = sa.int8_attention_from_prequantized(state)
        result['output_device'] = str((observed[0] if isinstance(observed, tuple) else observed).device)
        if state is not None:
            result['prepared_state'] = {name: {'dtype': str(value.dtype), 'shape': list(value.shape),
                                              'stride': list(value.stride())}
                                        if isinstance(value, torch.Tensor) else str(value)
                                        for name, value in vars(state).items()}
        actual = canonical(observed[0] if isinstance(observed, tuple) else observed).cpu().double()
        if args.api == 'sol' and 'block_len' in spec.get('sol', {}):
            live = torch.tensor(spec['sol']['block_len']).clamp(1, 64)
            rows = torch.arange(q.shape[2])
            selected = rows % 64 < live[rows // 64]
            result['unspecified_query_rows_excluded'] = int((~selected).sum())
            actual, reference = actual[:, :, selected], reference[:, :, selected]
        relative = float((actual-reference).norm() / reference.norm().clamp_min(1e-30))
        quality = math.isfinite(relative) and relative <= fixture['acceptance']['rel_l2']
        mean_absolute = float((actual-reference).abs().mean())
        if fixture['acceptance'].get('mean_abs') is not None:
            quality &= mean_absolute <= fixture['acceptance']['mean_abs']
        if isinstance(observed, tuple) and reference_lse is not None:
            actual_lse = observed[1].cpu().double()
            finite = torch.isfinite(reference_lse) & torch.isfinite(actual_lse)
            difference = (actual_lse - reference_lse)[finite]
            result['lse_max_abs'] = float(difference.abs().max()) if difference.numel() else 0.0
            quality &= bool(torch.allclose(actual_lse, reference_lse, rtol=2e-5, atol=2e-5))
        result.update(case_id=fixture['case_id'], k0_status='K0-valid' if quality else 'K0-invalid',
                      quality_passed=quality, rel_l2=relative if math.isfinite(relative) else None,
                      mean_abs=mean_absolute if math.isfinite(mean_absolute) else None,
                      precision=spec.get('precision', 'f32'),
                      acceptance_reviewed=fixture['acceptance']['reviewed'], milliseconds=elapsed,
                      warmup=10, iterations=32 if args.boundary == 'resident' else 1,
                      dtype=str(actual.dtype), output_storage_dtype=str((observed[0] if isinstance(observed, tuple) else observed).dtype))
        return 0 if quality else 1
    except Exception as error:
        if type(error).__name__ in ('NoCapableBackendError', 'BackendNotFoundError'):
            result['k0_status'] = 'K0-rejected'
        elif type(error).__name__ in ('OutOfMemoryError', 'MemoryError'):
            result['k0_status'] = 'Not executed'
        elif args.api in ('sage', 'prequantized', 'decode') and result.get('native_available') is False \
                and isinstance(error, RuntimeError) and 'requires' in str(error):
            result['k0_status'] = 'K0-rejected'
        result['reason'] = type(error).__name__ + ': ' + str(error)
        return 2 if result['k0_status'] == 'Not executed' else 1
    finally:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    sys.exit(main())
