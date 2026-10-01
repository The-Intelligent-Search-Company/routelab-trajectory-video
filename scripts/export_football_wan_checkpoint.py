"""Restore the original FSDP rank topology and export complete BF16 experts."""
from __future__ import annotations

import argparse
from datetime import timedelta
from functools import partial
import gc
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig,
    MixedPrecision, ShardingStrategy, StateDictType)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from football_full_backbone_common import verify_checkpoint
from football_wan_eval_common import canonical_weights
from train_football_full_backbone import shard_context
from football_trajectory_control import write_json
from wan_move_cloud_storage import digest


def export_tensor_sample(value, count=256):
    """Sample using exact integer indices, including large tensor endpoints.

    Float32 linspace can round numel-1 up to numel above 2**24. Keep this
    export-only diagnostic separate from the immutable training runtime.
    Full-tensor finiteness and serialization checks remain mandatory below.
    """
    if type(count) is not int or count < 1:
        raise ValueError('Sample count must be a positive integer')
    x = value.detach().reshape(-1)
    n = min(count, x.numel())
    indices = torch.tensor([i*(x.numel()-1)//max(1, n-1) for i in range(n)],
                           dtype=torch.int64, device=x.device)
    return x[indices].float().cpu().clone()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'model', 'vendor', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--identity', required=True)
    p.add_argument('--expert', choices=['high_noise', 'low_noise'], required=True)
    p.add_argument('--expected-step',type=int,default=32)
    p.add_argument('--world-size',type=int,default=8)
    args = p.parse_args()
    rank, world, local_rank = (int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK'])
    if world != args.world_size or torch.__version__.split('+')[0] != '2.8.0':
        raise ValueError('Export must use the original checkpoint rank count and Torch 2.8 runtime')
    torch.set_num_threads(8)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    sys.path.insert(0, str(args.vendor.resolve()))
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.models.wan_video_dit import DiTBlock
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cpu', tokenizer_config=None,
        model_configs=[ModelConfig(path=str(args.model/f'{args.expert}_model/diffusion_pytorch_model.safetensors'))])
    dit = pipe.dit
    if dit.in_dim != 52 or len(dit.blocks) != 40 or not hasattr(dit, 'ref_conv'):
        raise ValueError('Wrong Wan architecture')
    wrapper = torch.nn.Module()
    wrapper.add_module('dit', dit)
    shapes = {n: list(v.shape) for n, v in wrapper.state_dict().items()}
    base_samples = {n: export_tensor_sample(v) for n, v in wrapper.state_dict().items()} if rank == 0 else {}
    parameter_count = sum(p.numel() for p in wrapper.parameters())
    if parameter_count != 14_289_561_664:
        raise ValueError('Unexpected full-backbone parameter count')
    wrapper.float().requires_grad_(True)
    del pipe, dit
    model = FSDP(wrapper, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={DiTBlock}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16),
        device_id=local_rank, use_orig_params=True, limit_all_gathers=True)
    state = verify_checkpoint(args.checkpoint, args.identity, world, args.expert, rank=rank)
    if state['step'] != args.expected_step:
        raise ValueError('Checkpoint is not at the requested export step')
    payload = torch.load(args.checkpoint/f'rank-{rank:05d}.pt', map_location='cpu', weights_only=False)
    if (payload['identity'], payload['rank'], payload['world_size'], payload['step']) != (args.identity, rank, world, args.expected_step):
        raise ValueError('Rank payload disagrees with committed checkpoint')
    with shard_context(model):
        model.load_state_dict(payload['model'], strict=True)
    del payload
    gc.collect()
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        full_state = model.state_dict()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        master_samples = {n: export_tensor_sample(v) for n, v in full_state.items()}
        weights = canonical_weights(full_state, shapes)
        del full_state
        path = args.output/'model.safetensors'
        save_file(weights, str(path), metadata={'identity': args.identity, 'expert': args.expert,
            'step': str(args.expected_step), 'format': 'canonical_diffsynth_dit_bf16'})
        # Compare exact bytes for every tensor after the serialization round trip.
        loaded = load_file(str(path), device='cpu')
        if set(loaded) != set(weights) or any(not torch.equal(loaded[n], v) for n, v in weights.items()):
            raise ValueError('Safetensors round-trip mismatch')
        del loaded
        changes = {}
        for name, before in base_samples.items():
            rounded = export_tensor_sample(weights[name[4:]])
            master = master_samples[name]
            changes[name] = {'samples': before.numel(),
                'changed_fp32_samples': int(torch.count_nonzero(master != before)),
                'changed_bf16_samples': int(torch.count_nonzero(rounded != before)),
                'max_fp32_delta': float((master-before).abs().max()),
                'max_bf16_delta': float((rounded-before).abs().max())}
        write_json(args.output/'export.json', {'identity': args.identity, 'expert': args.expert, 'step': args.expected_step,
            'original_world_size': world, 'all_rank_sha256_verified': True,
            'parameter_count': parameter_count, 'tensor_count': len(weights),
            'dtype': 'bfloat16', 'original_master_dtype': 'float32',
            'all_tensors_finite': True, 'all_tensors_serialization_roundtrip_equal': True,
            'state_sha256': digest(path), 'sampled_changes': changes,
            'quantization_note': 'BF16 matches training compute precision; tiny FP32 updates may round away.'})
        from wan_move_cloud_storage import commit_directory
        commit_directory(args.output)
        print(json.dumps({'export_complete': args.expert, 'identity': args.identity,
            'changed_bf16_samples': sum(x['changed_bf16_samples'] for x in changes.values())}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
