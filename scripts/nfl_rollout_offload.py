"""Two-input rollout engine for one 80 GB GPU; experts swap at the sigma boundary."""
import json
from pathlib import Path
import time

import cv2
from PIL import Image
import torch
import football_wan_codec_precision as codec_precision

from football_full_backbone_data import save_video
from football_wan_eval_common import denoise,tensor_digest
from wan_move_cloud_storage import digest


def offload_conditioning_encoder(pipe):
    """Move encoder weights without changing the pipeline's execution device.

    Native BasePipeline.to('cpu') also changes pipe.device. The later VAE-only
    reload would then send latent inputs to CPU while its weights are on CUDA.
    """
    if torch.device(pipe.device).type != 'cuda':
        raise ValueError('Offloaded WAN conditioning pipeline must target CUDA')
    pipe.vae.to('cpu')
    pipe.text_encoder.to('cpu')


def input_control_frames(bundle, request):
    """V2 uses the canonical raw renderer; MP4 is a legacy/display artifact."""
    if request.get('schema') == 'nfl-route-request-v2':
        from nfl_conditioning import control_rgb_frames
        return [Image.fromarray(frame) for frame in control_rgb_frames(bundle, request)]
    capture = cv2.VideoCapture(str(Path(bundle)/'control.mp4'))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return frames


class OffloadedWan:
    def __init__(self,model_path,config,profile='baseline'):
        from diffsynth.pipelines.wan_video import WanVideoPipeline,ModelConfig
        if profile!='baseline' or torch.cuda.device_count()!=1:
            raise ValueError('Offloaded evaluation requires one visible CUDA GPU and baseline execution')
        self.config=config;self.calls=0;self.models={};started=time.monotonic()
        torch.set_num_threads(8);path=Path(model_path)
        for expert in ('high_noise','low_noise'):
            pipe=WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cpu',tokenizer_config=None,
                model_configs=[ModelConfig(path=str(path/f'{expert}_model/diffusion_pytorch_model.safetensors'))])
            if pipe.dit is None or pipe.dit.in_dim!=52 or len(pipe.dit.blocks)!=40 or pipe.dit.require_clip_embedding:
                raise ValueError('Unexpected Wan control architecture')
            self.models[expert]=pipe.dit.requires_grad_(False).eval();del pipe
        # Encoder is on GPU only during conditioning and reconstruction. Its
        # pipeline device remains CUDA, matching every encode/decode invocation.
        self.encoder=WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda',
            model_configs=[ModelConfig(path=str(path/'Wan2.1_VAE.pth')),
                ModelConfig(path=str(path/'models_t5_umt5-xxl-enc-bf16.pth'))],
            tokenizer_config=ModelConfig(path=str(path/'google/umt5-xxl')))
        self.encoder.requires_grad_(False).eval()
        self.vae_policy=codec_precision.policy(config)
        codec_precision.install_original_fp32_vae(self.encoder,path/'Wan2.1_VAE.pth',self.vae_policy)
        self.load_seconds=time.monotonic()-started

    @torch.inference_mode()
    def generate(self,bundle,output,request,stage):
        from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder,model_fn_wan_video
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        bundle,output=Path(bundle),Path(output);output.mkdir(parents=True,exist_ok=True)
        settings=self.config['inference'];v=request['video'];started=time.monotonic()
        torch.cuda.reset_peak_memory_stats();stage('encoding_controls',step=0)
        for model in self.models.values():model.to('cpu')
        self.encoder.to('cuda')
        embed=WanVideoUnit_PromptEmbedder()
        context=embed.encode_prompt(self.encoder,request['prompt'])
        negative=embed.encode_prompt(self.encoder,settings['negative_prompt'])
        frames=input_control_frames(bundle,request)
        if len(frames)!=v['num_frames'] or any(f.size!=(v['width'],v['height']) for f in frames):
            raise ValueError('Control video geometry mismatch')
        image=Image.open(bundle/'reference.png').convert('RGB')
        if image.size!=(v['width'],v['height']):raise ValueError('Reference geometry mismatch')
        control=codec_precision.encode_frames(self.encoder,frames,self.vae_policy).to(device='cuda')
        reference=codec_precision.encode_frames(self.encoder,[image],self.vae_policy).to(device='cuda')
        conditions={'context':context,'y':torch.cat([control,control.new_zeros((1,20,*control.shape[2:]))],dim=1),
            'reference_latents':reference}
        conditioning_sha={key:tensor_digest(x) for key,x in conditions.items()}
        offload_conditioning_encoder(self.encoder);torch.cuda.empty_cache()
        shape=(1,16,(v['num_frames']-1)//4+1,v['height']//8,v['width']//8)
        noise=torch.randn(shape,generator=torch.Generator(device='cpu').manual_seed(request['seed']),dtype=torch.float32)
        noise_sha=tensor_digest(noise);latents=noise.to(device='cuda',dtype=torch.bfloat16)
        scheduler=FlowMatchScheduler('Wan')
        scheduler.set_timesteps(settings['num_inference_steps'],denoising_strength=1,shift=settings['sigma_shift'])
        def activate(previous,current):
            if previous is not None:self.models[previous].to('cpu')
            torch.cuda.empty_cache();self.models[current].to('cuda')
        def progress(step,expert):
            if step==1 or step%5==0:stage('generating',step=step,expert=expert)
        latents,counts=denoise(self.models,model_fn_wan_video,scheduler,latents,conditions,negative,
            cfg_scale=settings['cfg_scale'],boundary=settings['switch_DiT_boundary'],activate=activate,progress=progress)
        for model in self.models.values():model.to('cpu')
        torch.cuda.empty_cache();self.encoder.vae.to('cuda')
        stage('decoding',step=settings['num_inference_steps'])
        decoded=codec_precision.decode(self.encoder,latents,self.vae_policy)
        rgb=((decoded[0].permute(1,2,3,0).float().clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
        self.encoder.vae.to('cpu')
        if rgb.shape!=(v['num_frames'],v['height'],v['width'],3) or not all(counts.values()):
            raise ValueError('Incomplete rollout')
        save_video(output/'generated.mp4',[cv2.cvtColor(f,cv2.COLOR_RGB2BGR) for f in rgb],v['fps'])
        cv2.imwrite(str(output/'preview.jpg'),cv2.cvtColor(rgb[len(rgb)//2],cv2.COLOR_RGB2BGR))
        self.calls+=1
        return {'initial_noise_sha256':noise_sha,'conditioning_sha256':conditioning_sha,'expert_steps':counts,
            'video_sha256':digest(output/'generated.mp4'),'decoded_frames':len(rgb),
            'gpu_name':torch.cuda.get_device_name(0),'execution':{'profile':'cpu_offloaded_experts'},
            'vae_precision_policy':self.vae_policy,
            'container_generation_index':self.calls,'peak_gpu_memory_gb':torch.cuda.max_memory_allocated()/1e9,
            'timings':{'model_load_seconds':self.load_seconds,'generation_seconds':time.monotonic()-started}}
