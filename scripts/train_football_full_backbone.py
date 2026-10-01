"""Full Wan expert fine-tuning with FSDP and complete, committed resume state.

Launch with torchrun on one node. Frozen encoders run in a separate cache stage.
No PEFT modules, inference quantization, or pretrained layers remain frozen here.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
    StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from football_full_backbone_common import (
    expert_range, full_parameter_audit, gradient_samples, identity, latest_checkpoint,
    parameter_changes, rng_state, sampled_parameters, set_rng_state, verify_checkpoint,
)
from football_full_backbone_data import validate_config, verify_dataset
from football_trajectory_control import sha256, write_json
from wan_move_cloud_storage import MARKER, publish, restore
from nfl_posttrain_sampling import clip_index, expert_indices, sampler_contract
from nfl_training_loss import future_velocity_loss
from nfl_training_status import update_model_status


def shard_context(model):
    return FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True), ShardedOptimStateDictConfig(offload_to_cpu=True))


def save_checkpoint(model, optimizer, scheduler, step, args, run_id, rank, world):
    from nfl_distributed_checkpoint_publish import run_rank_zero_io
    directory = args.output / f'step-{step:08d}'
    directory.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    with shard_context(model):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    payload = {'model': model_state, 'optimizer': optimizer_state,
               'scheduler': scheduler.state_dict(), 'rng': rng_state(),
               'step': step, 'identity': run_id, 'rank': rank, 'world_size': world}
    payload['sampler']={**args.sampler_contract,'next_update':step}
    temporary = directory / f'rank-{rank:05d}.pt.partial'
    torch.save(payload, temporary)
    temporary.replace(directory / f'rank-{rank:05d}.pt')
    del payload, model_state, optimizer_state
    dist.barrier()
    def commit():
        write_json(directory/'training_state.json', {'step':step, 'identity':run_id,
                   'world_size':world, 'expert':args.expert,
                   'sampler':{**args.sampler_contract,'next_update':step},
                   'resume_contract':'same world size, config, data, code and expert; optimizer/scheduler/RNG restored'})
        # Marker is published only after every rank shard is complete.
        publish(directory, args.state / f'step-{step:08d}', workers=8)
        update_model_status('h3' if args.expert=='h3' else 'wan22', checkpoint={
            'step':step,'expert':args.expert,'status':'committed','identity':run_id},
            message=f'{args.expert}: checkpoint {step} is committed to local storage.')
        print(json.dumps({'checkpoint_committed': step, 'expert':args.expert}), flush=True)
    run_rank_zero_io(commit)


def load_checkpoint(model, optimizer, scheduler, checkpoint, args, run_id, rank, world):
    # Each rank checks its own large shard; together they hash the entire state.
    state = verify_checkpoint(checkpoint, run_id, world, args.expert, rank=rank)
    payload = torch.load(checkpoint/f'rank-{rank:05d}.pt', map_location='cpu', weights_only=False)
    if (payload['step'],payload['identity'],payload['rank'],payload['world_size']) != (state['step'],run_id,rank,world):
        raise ValueError('Rank payload disagrees with committed metadata')
    if args.require_sampler and payload.get('sampler') != {**args.sampler_contract,'next_update':state['step']}:
        raise ValueError('Distributed clip sampler was not restored')
    with shard_context(model):
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, payload['optimizer']))
    scheduler.load_state_dict(payload['scheduler'])
    set_rng_state(payload['rng'])
    if scheduler.last_epoch != state['step']:
        raise ValueError('Scheduler step was not restored')
    if not optimizer.state or any(int(v['step']) != state['step'] for v in optimizer.state.values()):
        raise ValueError('Adam momentum/step state was not restored')
    return state['step']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config','dataset','cache','model','vendor','output','state']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expert',choices=['high_noise','low_noise'],required=True)
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify-only',action='store_true',help='Recover a report after a final-checkpoint interruption')
    args=parser.parse_args()
    config=json.loads(args.config.read_text()); validate_config(config)
    settings=config['training']
    rank,world,local_rank=(int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    if world != settings['world_size']:
        raise ValueError('Unexpected distributed world size')
    torch.cuda.set_device(local_rank)
    device=torch.device('cuda',local_rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    torch.set_num_threads(8)
    seed=settings['seed']+rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    dataset=verify_dataset(args.dataset)
    cache=json.loads((args.cache/'manifest.json').read_text())
    from football_wan_codec_precision import policy as vae_policy
    if vae_policy(cache)!=vae_policy(config):
        raise ValueError('Encoder cache precision differs from training configuration')
    if cache['dataset_sha256'] != sha256(args.dataset/'manifest.json') or cache['models'] != config['models'] or cache['sources'] != config['sources']:
        raise ValueError('Stale encoder cache')
    for row in cache['rows']:
        if sha256(args.cache/row['cache']) != cache['files'][row['cache']]:
            raise ValueError('Corrupt cached features')
    train=[r for r in cache['rows'] if r['split'] in {'train_debug','train'}]
    dev=[r for r in cache['rows'] if r['split']=='development']
    if not train or not dev:raise ValueError('Empty training or development cache')
    args.sampler_contract=sampler_contract(config,[r['sample_id'] for r in train],world)
    args.require_sampler=config.get('schema_version') in {'nfl-full-backbone-v2','nfl-route-full-backbone-v3'}
    accumulation=settings.get('gradient_accumulation_steps',1)
    run_id=identity(config,args.dataset/'manifest.json',list(Path('scripts').glob('*full_backbone*.py')))
    args.output.mkdir(parents=True,exist_ok=True)
    args.state.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.vendor.resolve()))
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, model_fn_wan_video
    from diffsynth.models.wan_video_dit import DiTBlock
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    pipe=WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device='cpu', tokenizer_config=None,
        model_configs=[ModelConfig(path=str(args.model/f'{args.expert}_model/diffusion_pytorch_model.safetensors'))])
    dit=pipe.dit
    if dit is None or len(dit.blocks)!=40 or dit.in_dim!=52 or not hasattr(dit,'ref_conv'):
        raise ValueError('Unexpected control backbone')
    if dit.require_clip_embedding:
        raise ValueError('Cache contract requires the no-CLIP Wan2.2 control model')
    class Backbone(torch.nn.Module):
        def __init__(self,dit):
            super().__init__(); self.dit=dit
        def forward(self,latents,timestep,context,y,reference_latents):
            return model_fn_wan_video(dit=self.dit,latents=latents,timestep=timestep,context=context,
                y=y,reference_latents=reference_latents,use_gradient_checkpointing=settings['gradient_checkpointing'])
    # FP32 master parameters; FSDP casts gathered compute weights to BF16.
    backbone=Backbone(dit).float().requires_grad_(True).train()
    audit=full_parameter_audit(backbone)
    if audit['parameter_count'] < 13_000_000_000:
        raise ValueError('Full 14B backbone not loaded')
    del pipe,dit
    model=FSDP(backbone,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={DiTBlock}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16),
        device_id=local_rank,use_orig_params=True,limit_all_gathers=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],
        weight_decay=settings['weight_decay'],foreach=False)
    full_parameter_audit(model,optimizer)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/max(1,settings['warmup_steps'])))
    flow=FlowMatchScheduler('Wan');flow.set_timesteps(1000,training=True)
    lo,hi=expert_range(args.expert,len(flow.timesteps))
    allowed_indices=(expert_indices(flow.sigmas.cpu().numpy(),args.expert,settings['sigma_boundary'])
                     if 'sigma_boundary' in settings else np.arange(lo,hi))
    lo,hi=int(allowed_indices.min()),int(allowed_indices.max())+1
    if rank==0:
        write_json(args.output/'parameter_audit.json',{**audit,'identity':run_id,'world_size':world,
            'expert':args.expert,'sharding':'FULL_SHARD','master_dtype':'float32','compute_dtype':'bfloat16',
            'timestep_indices':[lo,hi], 'timestep_values':[float(flow.timesteps[lo]),float(flow.timesteps[hi-1])]})
        print(json.dumps({'full_backbone_parameters':audit['parameter_count'],'expert':args.expert,'identity':run_id}),flush=True)
    start=0
    if args.resume:
        checkpoint=latest_checkpoint(args.state,run_id,world,args.expert)
        if checkpoint is None: raise ValueError('Explicit resume requested but no committed checkpoint exists')
        local=args.output/('restore-'+checkpoint.name)
        if rank==0:
            restored=restore(checkpoint,local,workers=8)
            write_json(local/MARKER,restored)
        dist.barrier()
        start=load_checkpoint(model,optimizer,scheduler,local,args,run_id,rank,world)
        if rank==0:
            print(json.dumps({'resumed_optimizer_step':start,'expert':args.expert}),flush=True)
    elif latest_checkpoint(args.state,run_id,world,args.expert) is not None:
        raise ValueError('Existing training state requires explicit resume')
    def features(row):
        return {k:x.to(device=device,dtype=torch.bfloat16) for k,x in torch.load(
            args.cache/row['cache'],weights_only=True,map_location='cpu').items()}
    def loss_for(batch, index=None, noise_seed=None):
        index=int(allowed_indices[int(torch.randint(len(allowed_indices),(1,)))]) if index is None else index
        timestep=flow.timesteps[index:index+1].to(device=device,dtype=torch.bfloat16)
        generator=None if noise_seed is None else torch.Generator(device=device).manual_seed(noise_seed)
        noise=torch.randn(batch['input_latents'].shape,device=device,dtype=torch.bfloat16,generator=generator)
        z=flow.add_noise(batch['input_latents'],noise,timestep)
        target=flow.training_target(batch['input_latents'],noise,timestep)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(z,timestep,batch['context'],batch['y'],batch['reference_latents'])
        prefix=1 if dataset.get('loss_mask_policy')=='native_first_frame_prefix' else 0
        loss=future_velocity_loss(pred,target,prefix)*flow.training_weight(timestep)
        return loss,index
    def evaluate():
        model.eval()
        selected=sorted(dev,key=lambda r:r['sample_id'])[:settings.get('validation_examples',64)]
        sums=torch.zeros(2,device=device,dtype=torch.float64)
        with torch.no_grad():
            for offset in range(0,len(selected),world):
                index=offset+rank
                row=selected[index%len(selected)]
                value,_=loss_for(features(row),index=int(allowed_indices[len(allowed_indices)//2]),noise_seed=1847+index%len(selected))
                if index<len(selected):sums+=torch.stack([value.double(),value.new_ones(()).double()])
            dist.all_reduce(sums)
        model.train()
        return float(sums[0]/sums[1])
    baseline=evaluate()
    stop=min(settings['max_steps'],args.stop_after or settings['max_steps'])
    if not start < stop and not (args.verify_only and args.resume and start==stop):
        raise ValueError('No remaining steps for this invocation')
    records=[]
    last_save=time.monotonic()
    for step in range(start,stop):
        begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
        before=sampled_parameters(model)
        total_loss=torch.zeros((),device=device)
        sampled=[]
        for micro in range(accumulation):
            selected=clip_index(len(train),settings['seed'],step,micro,rank,world,accumulation)
            batch=features(train[selected]);sampled.append(train[selected]['sample_id'])
            if settings['control_dropout'] and random.random()<settings['control_dropout']:
                batch['y'][:,:16]=0
            if settings.get('text_dropout',0) and random.random()<settings['text_dropout']:
                batch['context']=batch['context_null']
            loss,index=loss_for(batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            (loss/accumulation).backward()
            total_loss+=loss.detach()/accumulation
        gradients=gradient_samples(model)
        grad_norm=model.clip_grad_norm_(settings['max_grad_norm'])
        if not torch.isfinite(grad_norm):raise FloatingPointError('Nonfinite full gradient norm')
        optimizer.step();scheduler.step()
        changes=parameter_changes(model,before)
        local={'gradients':gradients,'changes':changes}
        gathered=[None]*world
        dist.all_gather_object(gathered,local)
        merged_grad={};merged_change={}
        for shard in gathered:
            for k,v in shard['gradients'].items():merged_grad[k]=max(v,merged_grad.get(k,0))
            for k,v in shard['changes'].items():merged_change[k]=max(v,merged_change.get(k,0))
        groups=[f'blocks.{i}.' for i in range(40)]+['patch_embedding.','ref_conv.','head.']
        for group in groups:
            if not any(group in k and v>0 for k,v in merged_grad.items()) or not any(group in k and v>0 for k,v in merged_change.items()):
                raise RuntimeError('No verified gradient and weight update in '+group)
        value=total_loss;dist.all_reduce(value);value/=world
        record={'step':step+1,'loss':float(value),'grad_norm':float(grad_norm),
            'seconds':time.monotonic()-begin,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'lr':optimizer.param_groups[0]['lr'],'timestep_index_rank0':index,'verified_groups':groups,
            'sample_ids_rank0':sampled,'effective_batch':world*accumulation,
            'clip_exposures':(step+1)*world*accumulation,
            'epochs_completed':(step+1)*world*accumulation/len(train)}
        records.append(record)
        if (step+1)%settings.get('validation_every',250)==0 or step+1==stop:
            record['validation_loss']=evaluate()
        if rank==0:
            write_json(args.output/f'update-audit-{step+1:08d}.json',{'gradients':merged_grad,'changes':merged_change})
            print(json.dumps(record),flush=True)
            update_model_status('wan22',record={**record,'expert':args.expert,'expert_step':step+1,
                'step':step+1+(settings['max_steps'] if args.expert=='low_noise' else 0)},phase='training_pilot',
                total_steps=2*settings['max_steps'],gpu_count=world,
                message=f'{args.expert}: bounded full-backbone pilot; no visual quality claim.')
        due=torch.tensor(int(time.monotonic()-last_save>=settings.get('checkpoint_seconds',1800)),device=device)
        dist.all_reduce(due,op=dist.ReduceOp.MAX)
        if (step+1)%settings['save_every']==0 or step+1==stop or due.item():
            save_checkpoint(model,optimizer,scheduler,step+1,args,run_id,rank,world)
            last_save=time.monotonic()
    after=evaluate()
    if rank==0:
        report={'status':'bounded_training_completed','expert':args.expert,'identity':run_id,
            'start_step':start,'stop_step':stop,'resume_verified':start>0,
            'development_fixed_noise_loss_before':baseline,'development_fixed_noise_loss_after':after,
            'steps':records,'training_windows':len(train),'sampler':args.sampler_contract,
            'camera_policy':dataset.get('camera'),
            'limitations':['bounded_training_check','no_video_quality_claim']}
        write_json(args.output/f'training-report-{start:08d}-{stop:08d}.json',report)
        logs=args.output/'report-upload';logs.mkdir(exist_ok=True)
        import shutil
        for path in args.output.glob('*.json'):shutil.copy2(path,logs/path.name)
        publish(logs,args.state/f'report-{start:08d}-{stop:08d}')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    main()
