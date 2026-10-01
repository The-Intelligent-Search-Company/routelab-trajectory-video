"""Versioned WAN codec arithmetic shared by caches, audits and rollouts.

The FP32 policy loads the original VAE directly at FP32; upcasting rounded
BF16 weights is not equivalent. Backbone inputs remain BF16. Legacy requests
retain their original policy, and changing policy creates a new run identity.
"""
from contextlib import contextmanager,nullcontext
import torch

LEGACY='native_bf16_v1'
FP32='fp32_vae_bf16_latents_v1'


def policy(config):
    value=config.get('vae_precision_policy',LEGACY)
    if value not in {LEGACY,FP32}:raise ValueError('Unknown WAN VAE precision policy')
    return value


def compute_dtype(value):
    if value not in {LEGACY,FP32}:raise ValueError('Unknown WAN VAE precision policy')
    return torch.float32 if value==FP32 else torch.bfloat16


@contextmanager
def arithmetic(value,device):
    compute_dtype(value)
    if value==LEGACY:
        yield
        return
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        context=torch.autocast('cuda',enabled=False) if torch.device(device).type=='cuda' else nullcontext()
        with context,torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=False,allow_tf32=False):
            yield
    finally:torch.backends.cuda.matmul.allow_tf32=old


def install_original_fp32_vae(pipe,path,value):
    if value==LEGACY:return
    compute_dtype(value)
    from diffsynth.pipelines.wan_video import WanVideoPipeline,ModelConfig
    native=WanVideoPipeline.from_pretrained(torch_dtype=torch.float32,device=pipe.device,
        model_configs=[ModelConfig(path=str(path))],tokenizer_config=None)
    if native.dit is not None or native.text_encoder is not None:raise ValueError('Expected VAE-only FP32 reload')
    pipe.vae=native.vae.requires_grad_(False).eval()
    assert_vae_dtype(pipe.vae,value)


def assert_vae_dtype(vae,value):
    expected=compute_dtype(value)
    if any(p.dtype!=expected for p in vae.parameters() if p.is_floating_point()):
        raise ValueError('VAE weights differ from declared compute precision')


@torch.no_grad()
def encode_pixels(pipe,pixels,value):
    assert_vae_dtype(pipe.vae,value)
    with arithmetic(value,pipe.device):
        return pipe.vae.encode(pixels.to(device=pipe.device,dtype=compute_dtype(value)),
            device=pipe.device,tiled=True).to(dtype=torch.bfloat16)


@torch.no_grad()
def encode_frames(pipe,frames,value):
    # Normalize RGB at compute precision; do not round pixels before FP32 VAE.
    pixels=pipe.preprocess_video(frames,torch_dtype=compute_dtype(value))
    return encode_pixels(pipe,pixels,value)


@torch.no_grad()
def decode(pipe,latents,value):
    assert_vae_dtype(pipe.vae,value)
    with arithmetic(value,pipe.device):
        return pipe.vae.decode(latents.to(device=pipe.device,dtype=compute_dtype(value)),
            device=pipe.device,tiled=True)
