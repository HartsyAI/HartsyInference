"""Exact equivalence check for pristine, telemetry-disabled, and telemetry-enabled operation runs."""
import argparse
import json
import pathlib
import statistics
from fixtures import sha256


def compare(pristine, disabled, enabled, observations):
    arms = (pristine, disabled, enabled)
    for arm in arms:
        if not arm.get('quality_passed') or not arm.get('milliseconds'):
            raise ValueError('Every arm must execute and pass its independent oracle')
    for field in ('fixture_sha256', 'output_sha256', 'backend', 'device', 'branch', 'boundary',
                  'resolved_settings', 'ptx_sha256', 'spirv_sha256'):
        if any(arm.get(field) != pristine.get(field) for arm in arms[1:]):
            raise ValueError('Telemetry equivalence failed: ' + field)
    # Earlier observation builds included managed Vulkan assemblies in the native-module list.
    def native(arm):
        return {path: digest for path, digest in arm.get('native_libraries', {}).items()
                if not path.replace('\\', '/').rsplit('/', 1)[-1].lower().startswith('hartsyinference.')}
    if any(native(arm) != native(pristine) for arm in arms[1:]):
        raise ValueError('Telemetry equivalence failed: native libraries')
    if disabled['core_sha256'] != enabled['core_sha256'] or disabled['backend_sha256'] != enabled['backend_sha256']:
        raise ValueError('Enabled and disabled runs must use identical production binaries')
    if pristine['core_sha256'] == disabled['core_sha256']:
        raise ValueError('Pristine and diagnostic assemblies were mislabeled')
    if [arm['telemetry'] for arm in arms] != ['disabled', 'disabled', 'enabled']:
        raise ValueError('Incorrect telemetry configuration labels')
    if not observations or not any(row.get('Implementation') for row in observations):
        raise ValueError('No executed-candidate observations')
    means = [statistics.fmean(arm['milliseconds']) for arm in arms]
    return {'exact_output_identity': True, 'top_level_route_identity': True,
            'observed_executions': sum(bool(row.get('Implementation')) for row in observations),
            'mean_ms': dict(zip(('pristine', 'disabled', 'enabled'), means)),
            'disabled_over_pristine': means[1] / means[0], 'enabled_over_disabled': means[2] / means[1],
            'claim': 'descriptive overhead only; not a registered paired statistical campaign',
            'limitations': ['B0 exposes its top-level route only; this does not prove every nested path',
                            'This case does not establish memory-pressure or real-generation equivalence']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pristine', 'disabled', 'enabled', 'observations', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    paths = [args.pristine, args.disabled, args.enabled, args.observations]
    result = compare(*(json.loads(pathlib.Path(path).read_text()) for path in paths))
    result['inputs_sha256'] = {str(path): sha256(path) for path in paths}
    with pathlib.Path(args.out).open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
