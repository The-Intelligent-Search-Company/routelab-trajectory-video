"""Full-parameter audit, RNG state, and checkpoint contracts."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from wan_move_cloud_storage import MARKER, digest, read_manifest

def identity(config, dataset_manifest, code=None):
    scripts = Path(__file__).parent
    payload = {"config": config, "dataset_sha256": digest(dataset_manifest),
        "code": {str(p.relative_to(scripts)): digest(p) for p in sorted(scripts.rglob("*.py"))}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def full_parameter_audit(model, optimizer=None):
    params = dict(model.named_parameters())
    frozen = [n for n, p in params.items() if not p.requires_grad]
    lora = [n for n in params if 'lora' in n.lower()]
    if not params or frozen or lora:
        raise ValueError(f'Incomplete full-backbone coverage: frozen={frozen[:5]}, lora={lora[:5]}')
    if optimizer is not None:
        actual = [p for group in optimizer.param_groups for p in group['params']]
        if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != {id(p) for p in params.values()}:
            raise ValueError('Optimizer does not cover each backbone parameter exactly once')
    return {'scope': 'full_backbone', 'parameter_count': sum(p.numel() for p in params.values()),
            'tensor_count': len(params), 'names': list(params), 'frozen': frozen, 'lora': lora}



def expert_range(expert, count=1000):
    if expert == 'high_noise':
        return 0, int(.358 * count)
    if expert == 'low_noise':
        return int(.358 * count), count
    raise ValueError('Unknown expert')



def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}



def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])



def verify_checkpoint(root, expected_identity, world_size, expert, rank=None):
    manifest = read_manifest(root)
    meta_path = root / 'training_state.json'
    if digest(meta_path) != manifest['training_state.json']['sha256']:
        raise ValueError('Checkpoint metadata checksum mismatch')
    state = json.loads(meta_path.read_text())
    if (state['identity'], state['world_size'], state['expert']) != (expected_identity, world_size, expert):
        raise ValueError('Checkpoint identity, world size or expert mismatch')
    required = {f'rank-{i:05d}.pt' for i in range(world_size)} | {'training_state.json'}
    if set(manifest) != required or state['step'] < 1:
        raise ValueError('Incomplete checkpoint inventory')
    if rank is not None:
        name = f'rank-{rank:05d}.pt'
        if digest(root / name) != manifest[name]['sha256']:
            raise ValueError('Checkpoint rank checksum mismatch')
    return state



def latest_checkpoint(root, expected_identity, world_size, expert):
    for p in sorted(root.glob('step-*'), reverse=True):
        if (p / MARKER).is_file():
            verify_checkpoint(p, expected_identity, world_size, expert)
            return p
    return None



def sampled_parameters(model):
    """Local-shard samples; aggregated across ranks by the trainer."""
    result = {}
    for name, p in model.named_parameters():
        if p.numel():
            stride = max(1, p.numel() // 128)
            result[name] = p.detach().reshape(-1)[::stride][:128].float().cpu().clone()
    return result



def parameter_changes(model, before):
    after = sampled_parameters(model)
    return {n: float((after[n] - v).abs().max()) for n, v in before.items()}



def gradient_samples(model):
    result = {}
    for name, p in model.named_parameters():
        if p.numel() and p.grad is not None:
            grad = p.grad.detach()
            # Every local parameter tensor is checked, not only the samples.
            if not torch.isfinite(grad).all():
                raise FloatingPointError('Nonfinite backbone gradient: ' + name)
            result[name] = float(torch.linalg.vector_norm(grad))
    return result
