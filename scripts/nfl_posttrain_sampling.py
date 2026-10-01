"""Deterministic distributed clip order and shared sigma expert selection."""
import numpy as np


def clip_index(size, seed, update, microstep, rank, world_size, accumulation):
    if min(size,world_size,accumulation)<1 or min(update,microstep,rank)<0 or rank>=world_size or microstep>=accumulation:
        raise ValueError('Invalid distributed sampling state')
    offset=(update*accumulation+microstep)*world_size+rank
    epoch,position=divmod(offset,size)
    return int(np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(size)[position])


def expert_indices(sigmas, expert, boundary):
    values=np.asarray(sigmas,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not 0<boundary<1:
        raise ValueError('Invalid sigma schedule')
    if expert not in {'high_noise','low_noise'}:
        raise ValueError('Unknown expert')
    mask=values>=boundary
    indices=np.flatnonzero(mask if expert=='high_noise' else ~mask)
    if not len(indices):raise ValueError('Expert has no training timesteps')
    return indices


def sampler_contract(config, sample_ids, world_size):
    settings=config['training']
    return {'version':'shuffled-global-clips-v1','sample_ids':list(sample_ids),
        'seed':settings['seed'],'world_size':world_size,
        'gradient_accumulation_steps':settings.get('gradient_accumulation_steps',1)}
