# Inference

Use the environment and pinned upstream checkout described in [training](training.md).
Inference uses one visible CUDA GPU; experts are offloaded to host RAM and
swapped at the noise boundary. The original serving experiment ran on a B200;
an 80GB-class CUDA device is the intended minimum profile, not a verified
performance guarantee for this extracted release. CPU/MPS generation is not
implemented. No API key or proprietary service is needed.

1. Supply an authorized first-frame PNG and 22 player trajectories.
2. Apply the same aspect-preserving letterbox transform to the image and all
   image coordinates before bundle creation. The helpers are in
   `nfl_preprocessing/media.py`. Raw input dimensions must not be stretched.
3. Run `prepare_inputs.py`. It validates geometry, writes the image, trajectories,
   request, control preview and hashes, and preserves the supplied timing.
4. Run `generate.py`, selecting either base experts or a verified exported pair.

```bash
python scripts/prepare_inputs.py --first-frame data/start.png \
  --trajectories data/routes.json --output runs/input-01 --seed 42

# Pretrained base.
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --bundle runs/input-01 \
  --model models/wan22 --vendor vendor/DiffSynth-Studio --output runs/base-01

# Adapted expert pair from a reproduction run; RUN_ID is the training identity.
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --bundle runs/input-01 \
  --model models/wan22 --vendor vendor/DiffSynth-Studio --output runs/adapted-01 \
  --candidate exports --identity "$RUN_ID" --step 32
```

The defaults are 49 frames, 832×480, 16fps, seed 42, 50 denoising steps,
CFG 5, sigma shift 5, expert boundary 0.875, empty negative text, original
FP32 VAE/BF16 latents, and the fixed football caption. Native generation in
the audited serving receipts used 21 high-noise and 29 low-noise steps.

Only the starting frame and trajectories enter the bundle. Fixed prompt text
is configuration, not per-play tactical annotation. There is no target-video,
future camera matrix, ball trajectory, or future-image argument. The control
MP4 is a display artifact; model pixels are rendered directly from the JSON
and checked against the input digest to avoid chroma-compression differences.

Image-space routes may encode a recorded camera plan in historical training
or matched replay inputs. The predicted-futures experiment instead fixed the
authenticated starting homography for the entire rollout. The model-input
boundary excludes future camera files in both cases; these are different
upstream trajectory-construction policies and should not be conflated.

Candidate installation hashes all export files, checks both experts' shared
identity/step/world size, validates all parameter names/shapes before changing
either model, and loads strictly. `result.json` records the seed, input/config
and output hashes, expert steps, precision, memory and timing. An existing
output directory is rejected, preventing accidental overwrite/replay.

These receipts prove what ran, not whether the requested player stayed on its
route. Route overlays represent conditioning, not recovered output tracks.
