"""Cache frozen Wan encoders and publish a real-video VAE reconstruction audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from football_full_backbone_data import read_video, save_video, verify_dataset
from football_trajectory_control import sha256, write_json


def control_conditioning(control_latents, target_latents, in_channels=52):
    """Match the encoder output device; tiled Wan VAE encoding returns CPU tensors."""
    if control_latents.shape != target_latents.shape or control_latents.ndim != 5:
        raise ValueError('Control and target latent geometry must match')
    empty_channels = in_channels - 2 * target_latents.shape[1]
    if empty_channels < 0:
        raise ValueError('Backbone has too few conditioning channels')
    zeros = control_latents.new_zeros((control_latents.shape[0], empty_channels, *control_latents.shape[2:]))
    return torch.cat([control_latents, zeros], dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'dataset', 'model', 'vendor', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    manifest = verify_dataset(args.dataset)
    sys.path.insert(0, str(args.vendor.resolve()))
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, WanVideoUnit_PromptEmbedder
    if not torch.cuda.is_available():
        raise RuntimeError('Real encoder cache requires CUDA')
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=str(args.model/'Wan2.1_VAE.pth')),
                       ModelConfig(path=str(args.model/'models_t5_umt5-xxl-enc-bf16.pth'))],
        tokenizer_config=ModelConfig(path=str(args.model/'google/umt5-xxl')))
    import football_wan_codec_precision as codec_precision
    vae_policy=codec_precision.policy(config)
    codec_precision.install_original_fp32_vae(pipe,args.model/'Wan2.1_VAE.pth',vae_policy)
    pipe.requires_grad_(False)
    pipe.eval()
    if pipe.dit is not None:
        raise ValueError('Cache stage must load only frozen encoders')
    architecture = json.loads((args.model/'high_noise_model/config.json').read_text())
    if architecture['in_dim'] != 52 or architecture['add_ref_conv'] is not True:
        raise ValueError('Unsupported control architecture')
    v = config['video']
    times = np.arange(v['num_frames']) / v['fps']
    records = []
    token_budget = config.get('caption_policy', {}).get('max_tokens', 512)
    if not 1 <= token_budget <= 512:
        raise ValueError('Caption budget exceeds the pinned Wan tokenizer context')
    with torch.no_grad():
        for row in manifest['rows']:
            token_count = len(pipe.tokenizer.tokenizer(row['prompt'], add_special_tokens=True,
                                                       truncation=False)['input_ids'])
            if token_count > token_budget:
                raise ValueError(f"Caption would overflow: {row['sample_id']}: {token_count}>{token_budget}")
            directory = args.output / row['sample_id']
            directory.mkdir()
            real, _ = read_video(args.dataset/row['video'], times, (v['width'], v['height']))
            control, _ = read_video(args.dataset/row['control_video'], times, (v['width'], v['height']))
            if manifest.get('schema_version') == 'nfl-native-route-dataset-v1':
                with np.load(args.dataset/row['control_frames'],allow_pickle=False) as archive:
                    control = list(archive['rgb'][...,::-1].copy())
            def encode(frames):
                rgb = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
                return codec_precision.encode_frames(pipe,rgb,vae_policy)
            target = encode(real)
            control_latents = encode(control)
            clean_reference = cv2.imread(str(args.dataset/row['reference_image']))
            reference = encode([clean_reference])
            context = WanVideoUnit_PromptEmbedder().encode_prompt(pipe, row['prompt'])
            # Exact FunControl format: 16 control channels + 20 empty I2V channels.
            y = control_conditioning(control_latents, target, architecture['in_dim'])
            tensors = {'input_latents': target.cpu(), 'context': context.cpu(),
                       'y': y.cpu(), 'reference_latents': reference.cpu()}
            if config['training'].get('text_dropout',0):
                tensors['context_null']=WanVideoUnit_PromptEmbedder().encode_prompt(pipe,'').cpu()
            if any(not torch.isfinite(x).all() for x in tensors.values()):
                raise ValueError('Nonfinite encoded training input')
            torch.save(tensors, directory/'features.pt')
            decoded = codec_precision.decode(pipe,target,vae_policy)
            rgb = ((decoded[0].permute(1,2,3,0).float().clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
            recon = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb]
            if len(recon) != len(real):
                raise ValueError('VAE temporal reconstruction mismatch')
            save_video(directory/'vae_reconstruction.mp4', recon, v['fps'])
            cv2.imwrite(str(directory/'vae_preview.jpg'), np.vstack([
                np.hstack([real[i], recon[i]]) for i in (0,len(real)//2,len(real)-1)]))
            mse = float(np.mean((np.asarray(real,dtype=np.float32)/255-np.asarray(recon,dtype=np.float32)/255)**2))
            record = {**row, 'cache': f"{row['sample_id']}/features.pt", 'mse': mse,
                      'caption_token_count': token_count,
                      'psnr_db': float(-10*np.log10(max(mse,1e-12))),
                      'shapes': {k:list(x.shape) for k,x in tensors.items()}}
            records.append(record)
            print(json.dumps({'cache_complete':row['sample_id'],'shapes':record['shapes'],'psnr_db':record['psnr_db']}),flush=True)
    write_json(args.output/'manifest.json', {
        'dataset_sha256':sha256(args.dataset/'manifest.json'), 'models':config['models'],
        'vae_precision_policy':vae_policy,
        'sources':config['sources'], 'rows':records,
        'files':{str(p.relative_to(args.output)):sha256(p) for p in sorted(args.output.rglob('*')) if p.is_file()}})


if __name__ == '__main__':
    main()
