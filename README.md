# RouteLab: trajectory-conditioned football video

RouteLab studies short football video generation from a starting frame and
22 time-indexed player trajectories. This repository releases the Wan2.2
conditioning implementation, full-backbone adaptation protocol, training and
inference entry points, and a bounded evidence report for SSAC 2027 research.

**Status: exploratory research.** Fixed-noise validation loss decreased after
32 updates per expert. Eight matched generated clips show limited appearance
gains alongside serious identity failures. Calibrated route accuracy, broad
video-quality improvement, tactical validity, and superiority to other methods
have not been established.

| Expert | Pretrained, step 0 | Adapted, step 32 | Relative reduction |
|---|---:|---:|---:|
| High noise | 0.030484072331871306 | 0.020289108423250064 | 33.4436% |
| Low noise | 0.10639166831970215 | 0.09817920412336077 | 7.7191% |

These are means over **seven fixed-noise validation windows**, not trajectory
errors or perceptual scores. The train/validation/sealed-test release contains
37/7/7 windows from 21/4/4 plays and 19/4/4 games. Only train and validation
targets were exposed to training; the sealed test was not evaluated here.

## What is included

- The actual identity-marker, team-ring, and trail renderer and target-free
  first-frame/trajectory bundle contract extracted from RouteLab.
- Frozen-encoder caching, full FSDP training of both 14B control experts,
  complete optimizer/RNG/sampler checkpoint state, strict export, and inference.
- A local-only storage adapter and standalone CLIs, with no cloud deployment,
  credentials, production service, or automatic compute launch.
- [Training protocol](docs/training.md), [inference instructions](docs/inference.md),
  [data contract](docs/data.md), [evaluation and limits](docs/evaluation.md), and
  [provenance/reproducibility](docs/provenance.md).

The release does **not** contain football footage or tracking, adapted weights,
upstream model code, internal transcripts, or empirical figure images whose
redistribution rights remain unverified. Bring your own authorized inputs and
obtain the pinned base weights under their own terms. Without the original
private dataset and checkpoints, the reported scientific results cannot be
independently regenerated from this repository alone.

## Quick start: CPU contracts

Python 3.13 and FFmpeg/ffprobe are required. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install torch==2.8.0
PYTHONPATH=scripts python -m pytest -q
python scripts/prepare_inputs.py --help
```

Tests use artificial tensors and neutral test images solely to exercise software
contracts. They are not empirical video results. No model weights are downloaded
and no GPU work is launched by these checks.

## Generation

Prepare a 832×480 PNG and a matching normalized trajectory JSON with 22 unique
player IDs, as described in [the data contract](docs/data.md):

```bash
python scripts/prepare_inputs.py --first-frame data/start.png \
  --trajectories data/routes.json --output runs/input-01 --seed 42
```

After explicitly preparing the GPU environment and base weights, generate with
one visible CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py \
  --bundle runs/input-01 --model models/wan22 \
  --vendor vendor/DiffSynth-Studio --output runs/base-01
```

This command runs inference when invoked. The release was tested on CPU during
publication; the extracted GPU paths were not rerun for this release. Read
[inference](docs/inference.md) for strict adapted-checkpoint installation and
[training](docs/training.md) before allocating hardware.

## Citation and license

This is a research software release, not an accepted SSAC paper. Cite the
repository URL and the exact commit used. Code is Apache-2.0; see [LICENSE](LICENSE)
and [NOTICE](NOTICE). Input data, weights, and third-party dependencies retain
their separate terms.
