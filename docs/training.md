# Training protocol and reproduction

## Audited experiment

The base is `alibaba-pai/Wan2.2-Fun-A14B-Control`, revision
`297be6d520f54908e124be0298e2317ee50ffdf4`. DiffSynth-Studio is pinned to
`c458cb42ab1ee838bff85c6546e14bb01c3571e9`. Each high- and low-noise expert has
14,289,561,664 parameters. Each expert was fully updated for 32 optimizer steps;
the two experts were trained separately, not as a single 32-step shared model.
VAE and text encoders were frozen and cached. This is not LoRA training.

| Setting | Audited value |
|---|---|
| Windows | 37 train; 7 validation; 7 sealed test |
| Plays / games | Train 21 / 19; validation 4 / 4; test 4 / 4 |
| Video | 49 frames, 16 fps, 832×480; timestamps 0–3 s |
| Parallelism | Eight H100 80GB GPUs; FSDP FULL_SHARD |
| Batch | One window per rank; accumulation 1; effective batch 8 |
| Optimizer | AdamW, learning rate 3e-6, weight decay 0.01 |
| Schedule | Four-step warmup; then constant learning rate |
| Gradient clipping | Global norm 1.0 |
| Seed | 42 + rank for training RNG |
| Precision | FP32 master parameters; BF16 compute; FP32 reductions |
| Encoders | Original FP32 VAE; BF16 output latents; frozen text encoder |
| Control / text dropout | 0 / 0 |
| Checkpoints | Step 8 resume proof; periodic step 16, final step 32; elapsed-time trigger 1800 s |
| Validation | Every 8 updates, before each invocation, and after each phase |
| Expert boundary | Sigma 0.875; 1,000-step training flow schedule |

Distributed clip order shuffles all training windows using a deterministic
per-epoch permutation. Each expert receives 256 clip exposures, about 6.9189
passes through the 37 windows. This does not mean 256 unique plays.

## Conditioning and loss

The input trajectories are rendered to geometry-only RGB frames with stable
identity colors, team rings and 0.25-second trails. Controls are VAE encoded.
The model concatenates 16 noisy video-latent channels with 36 conditioning
channels: 16 control-latent channels plus 20 zero image-to-video channels,
giving the control backbone's 52-channel input. A separately encoded starting
image supplies reference latents through the native reference convolution/token
path. No future RGB target is permitted in an inference bundle.

The training objective is the scheduler-weighted mean squared velocity error
on temporal latents after the first latent time slice. This is a loss mask;
the released denoising loop does not hard-clamp a generated frame to the input
image. Inspect `nfl_training_loss.py`, the cache constructor, and the training
`loss_for` function for the exact computation.

## Prepare an environment (explicit user action)

Use Linux, Python 3.13, Torch 2.8.0/cu128, FFmpeg, and enough host RAM/storage for
both 14B experts and sharded optimizer state. The historical final checkpoints
totaled 343,038,683,553 bytes across both experts; working copies, caches and
exports require additional capacity. Eight 80GB GPUs were used for training
and export. No cloud provisioning script is included.

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
git clone https://github.com/modelscope/DiffSynth-Studio.git vendor/DiffSynth-Studio
git -C vendor/DiffSynth-Studio checkout --detach c458cb42ab1ee838bff85c6546e14bb01c3571e9
```

Review upstream terms before acquiring model assets. Obtain the pinned Hugging
Face snapshot locally (the following downloads weights and requires storage):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('alibaba-pai/Wan2.2-Fun-A14B-Control',
    revision='297be6d520f54908e124be0298e2317ee50ffdf4',
    local_dir='models/wan22', allow_patterns=[
        'high_noise_model/*', 'low_noise_model/*', 'google/umt5-xxl/*',
        'Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth'])
PY
```

`requirements-gpu.txt` pins the release's renderer and support packages. This
is a portable dependency specification, not a claim of bitwise recreation of
the full historical operating-system environment. The original campaign used
NumPy 2.4.6/OpenCV 5.0.0.93/Pillow 12.3.0 for some local preparation; the serving
bundle renderer used NumPy 2.2.6/OpenCV 4.12.0.88/Pillow 11.3.0. A pixel digest
mismatch was resolved by preparing with the serving versions. This release
pins the latter, and never weakens the digest check.

## Package and cache authorized data

Prepare aligned source windows according to [data.md](data.md). The packaging
adapter is new release code; it starts from curated, already-aligned inputs
and does not reproduce the private ingestion/calibration pipeline.

```bash
python scripts/prepare_dataset.py --index data/windows.json --output data/native
python scripts/cache_football_full_backbone.py --config configs/wan22_train.json \
  --dataset data/native --model models/wan22 --vendor vendor/DiffSynth-Studio \
  --output cache/wan22
```

Caching uses a CUDA device but no gradients. It saves frozen latent features
and VAE reconstruction diagnostics. Reconstruction PSNR is a codec diagnostic,
not a generated-video quality result. Test windows must remain outside the
training/validation dataset and encoder cache.

## Full-backbone training and resume proof

Run from the repository root. The shell loop is sequential: one expert at a
time. These commands allocate no machines; they use an already provisioned node.

```bash
for expert in high_noise low_noise; do
  torchrun --standalone --nproc_per_node=8 scripts/train_football_full_backbone.py \
    --config configs/wan22_train.json --dataset data/native --cache cache/wan22 \
    --model models/wan22 --vendor vendor/DiffSynth-Studio --expert "$expert" \
    --output "runs/train/$expert" --state "runs/state/$expert" --stop-after 8
  torchrun --standalone --nproc_per_node=8 scripts/train_football_full_backbone.py \
    --config configs/wan22_train.json --dataset data/native --cache cache/wan22 \
    --model models/wan22 --vendor vendor/DiffSynth-Studio --expert "$expert" \
    --output "runs/train/$expert" --state "runs/state/$expert" --resume
done
```

Keep the world size, config, source files, cached data and expert unchanged
across resume. The loader restores model, optimizer, scheduler, Python/NumPy/
Torch/CUDA RNG and sampler cursor, and checks Adam and scheduler step counters.
Local artifacts are committed only after every shard is copied and hashed.
Partial output is preserved and rejected on reuse; it is not silently retried.
The public local-storage adapter differs from the historical GCS publisher.

## Export adapted experts

Read the reproduction identity from either final `training_state.json`.
It differs from the historical identity because this extraction changes source
files and storage integration. Supply that new identity as `RUN_ID`:

```bash
RUN_ID=$(python -c 'import json; print(json.load(open("runs/state/high_noise/step-00000032/training_state.json"))["identity"])')
for expert in high_noise low_noise; do
  torchrun --standalone --nproc_per_node=8 scripts/export_football_wan_checkpoint.py \
    --checkpoint "runs/state/$expert/step-00000032" --model models/wan22 \
    --vendor vendor/DiffSynth-Studio --output "exports/$expert" \
    --identity "$RUN_ID" --expert "$expert" --world-size 8 --expected-step 32
done
```

Export restores the original FSDP topology, checks the complete backbone,
finiteness and serialization roundtrip, and emits BF16 safetensors plus a
checksummed manifest. Only load trusted local PyTorch checkpoints: native FSDP
resume uses Python serialization. The public repository contains no checkpoint.
