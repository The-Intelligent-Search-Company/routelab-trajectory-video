"""Canonical expert export and the pinned Wan CFG denoising loop.

The loop follows DiffSynth-Studio (Apache-2.0); see NOTICE.
"""
import hashlib
import torch
EXPERTS = ("high_noise", "low_noise")
CONDITION_KEYS = {"context", "y", "reference_latents"}

def tensor_digest(value):
    x = value.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()



def canonical_weights(state, expected_shapes):
    if set(state) != set(expected_shapes):
        raise ValueError('Exported state does not cover the complete backbone')
    result = {}
    for key, value in state.items():
        if not key.startswith('dit.') or tuple(value.shape) != tuple(expected_shapes[key]):
            raise ValueError('Unexpected tensor in exported state: '+key)
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite exported tensor: '+key)
        result[key[4:]] = value.detach().to(dtype=torch.bfloat16, device='cpu').contiguous().clone()
    return result



def denoise(models, model_fn, scheduler, latents, conditions, negative, *, cfg_scale, boundary,
            activate, progress=lambda *args: None, batch_cfg=False):
    """The pinned Wan pipeline's CFG and expert-switch loop, with cached inputs."""
    if set(conditions) != CONDITION_KEYS:
        raise ValueError('Generation accepts only whitelisted conditioning')
    active = None
    counts = {name: 0 for name in EXPERTS}
    for index, timestep in enumerate(scheduler.timesteps):
        name = 'low_noise' if timestep.item() < boundary*1000 else 'high_noise'
        if active != name:
            activate(active, name)
            active = name
        t = timestep.unsqueeze(0).to(dtype=latents.dtype, device=latents.device)
        shared = {'latents': latents, 'timestep': t, 'y': conditions['y'],
                  'reference_latents': conditions['reference_latents']}
        if batch_cfg and cfg_scale != 1:
            if negative.shape != conditions['context'].shape:
                raise ValueError('Guidance context shapes differ.')
            paired = {key: torch.cat([value, value], dim=0) for key, value in shared.items()}
            output = model_fn(dit=models[name],
                context=torch.cat([conditions['context'], negative], dim=0), **paired)
            positive, unconditional = output.chunk(2, dim=0)
            prediction = unconditional + cfg_scale*(positive-unconditional)
        else:
            positive = model_fn(dit=models[name], context=conditions['context'], **shared)
            if cfg_scale != 1:
                unconditional = model_fn(dit=models[name], context=negative, **shared)
                prediction = unconditional + cfg_scale*(positive-unconditional)
            else:
                prediction = positive
        latents = scheduler.step(prediction, timestep, latents)
        if not torch.isfinite(latents).all():
            raise ValueError('Nonfinite generated latents')
        counts[name] += 1
        progress(index+1, name)
    return latents, counts

