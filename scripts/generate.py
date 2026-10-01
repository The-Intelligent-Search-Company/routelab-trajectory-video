"""Generate locally with pretrained experts or a verified adapted expert pair."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from nfl_conditioning import FILES, validate_bundle
from wan_move_cloud_storage import digest, read_manifest

VENDOR_REVISION = 'c458cb42ab1ee838bff85c6546e14bb01c3571e9'


def verified_pair(root, identity, step):
    paths = {}
    for expert in ['high_noise', 'low_noise']:
        directory = Path(root)/expert
        files = read_manifest(directory, verify_hashes=True)
        if set(files) != {'model.safetensors', 'export.json'}:
            raise ValueError('Unexpected export inventory')
        report = json.loads((directory/'export.json').read_text())
        if (report['identity'], report['expert'], report['step'], report['original_world_size']) != (identity, expert, step, 8):
            raise ValueError('Expert identity, step, or world size mismatch')
        if report['state_sha256'] != files['model.safetensors']['sha256'] or not all(
            report.get(k) is True for k in ['all_rank_sha256_verified', 'all_tensors_finite', 'all_tensors_serialization_roundtrip_equal']):
            raise ValueError('Export integrity evidence is incomplete')
        paths[expert] = directory/'model.safetensors'
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'model', 'vendor', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/wan22_inference.json'))
    p.add_argument('--candidate', type=Path)
    p.add_argument('--identity')
    p.add_argument('--step', type=int, default=32)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing generation attempt')
    files = {path.name: path.read_bytes() for path in args.bundle.iterdir() if path.is_file()}
    request = validate_bundle(files)
    config = json.loads(args.config.read_text())
    if request['video'] != config['video']:
        raise ValueError('Input and inference geometry differ')
    revision = subprocess.check_output(['git', '-C', str(args.vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != VENDOR_REVISION:
        raise ValueError('DiffSynth revision differs from the audited pin')
    if subprocess.check_output(['git', '-C', str(args.vendor), 'status', '--porcelain'], text=True).strip():
        raise ValueError('DiffSynth checkout has local modifications')
    paths = verified_pair(args.candidate, args.identity, args.step) if args.candidate else None
    if args.identity and not args.candidate:
        raise ValueError('Identity requires a candidate directory')
    sys.path.insert(0, str(args.vendor.resolve()))
    from nfl_rollout_offload import OffloadedWan
    from nfl_route_candidate import install_components
    args.output.mkdir(parents=True, exist_ok=False)
    # A claim remains if the process fails. No automatic generation retry.
    (args.output/'attempt.json').write_text(json.dumps({'bundle_sha256': digest(args.bundle/'manifest.json'),
        'candidate_identity': args.identity, 'step': args.step if paths else 0, 'config': config}, indent=2)+'\n')
    engine = OffloadedWan(args.model, config)
    if paths:
        install_components(engine.models, paths)
    result = engine.generate(args.bundle, args.output, request,
                            lambda stage, **fields: print(json.dumps({'stage': stage, **fields}), flush=True))
    result.update({'seed': request['seed'], 'candidate_identity': args.identity,
        'candidate_step': args.step if paths else 0, 'config': config,
        'input_bundle_sha256': digest(args.bundle/'manifest.json'), 'quality_validated': False})
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'result': str(args.output/'generated.mp4'), 'sha256': result['video_sha256']}))


if __name__ == '__main__':
    main()
