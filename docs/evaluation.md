# Measured evidence and proposed evaluation

## Completed measurements

The exact values and source-report SHA256 digests are preserved in
[`results/measured_validation.json`](../results/measured_validation.json).

| Expert | Step 0 | Step 8 | Step 32 |
|---|---:|---:|---:|
| High noise | 0.030484072331871306 | 0.02276325784623623 | 0.020289108423250064 |
| Low noise | 0.10639166831970215 | 0.10308640556676048 | 0.09817920412336077 |

These are scheduler-weighted velocity MSE values after masking the first
temporal latent slice. The step-0 number is the pretrained expert's evaluation
before any optimizer update. Step-8 resumed evaluations exactly match the
previous phase's final evaluation in the source reports.

The native dataset has 37 training and 7 development rows. The sealed trainer's
`evaluate()` sorts all development sample IDs, selects seven, and uses the
midpoint index among the expert's allowed training timesteps. The noise seed is
`1847 + sorted_window_index`. Every rank participates in the FSDP forward pass;
with eight ranks and seven windows, the eighth rank computes a duplicate that
is **excluded** from the accumulated sum/count. A float64 all-reduce followed
by division yields the arithmetic mean over the seven distinct windows.
It is one fixed timestep and one fixed noise draw per window per expert, not
an average over the full diffusion noise distribution.

Relative reductions from step 0 to 32 are 33.4436% and 7.7191%. Windows are
correlated within plays and games. No confidence interval or significance
claim is made. The separate sealed test set of seven windows was not scored.

## Completed generated-video review

Four matched cases (train/validation × recorded/edited routes), each with
pretrained and adapted outputs, produced eight clips under the same inference
settings: 49 frames, 832×480, 16fps, seed 42, 50 steps, CFG 5, shift 5, and expert
boundary 0.875. The recorded cases have separate factual targets; edits do not.
The train edit changed O05; the validation edit changed O03.

Full-frame views at frames 0/16/32/48 and selected-player crops at
0/8/16/24/32/40/48 showed more recognizable bodies/limbs in some adapted views,
but contact, merging bodies, camera drift and identity loss remained. In the
edited validation clip, the selected tracker moved onto grass at frame 24 and
later followed a different uniform. This invalidates a semantic route-error
interpretation. This bounded inspection establishes neither overall quality
gain nor generalization.

One RGB-only CoTracker3 selected-point diagnostic was run per clip, with points
chosen from frame zero before viewing later frames; requested future positions
were not tracker inputs. The mean screen-space separation between recorded-
and edited-route outputs over 49 frames was:

| Split | Base | Adapted |
|---|---:|---:|
| Train | 107.925 px | 106.896 px |
| Validation | 174.359 px | 183.898 px |

These are sensitivity diagnostics, **not ADE, endpoint error, route accuracy,
or improvement scores**. Base/candidate seeds differ to match generated
appearances; drift and wrong identities confound comparisons. Predicted
visibility is not an identity validation. Do not rank models using this table.

A later ten-hypothesis demonstration used one common starting snap, the same
adapted experts and seed, and external predicted trajectories. Video and input
hashes verify completion. The game is held out for WAN but is training data
for the trajectory predictor, so the whole pipeline is not an unseen-play
evaluation. No new quantitative route or quality score was obtained there.

## Proposed evaluation — not yet run

Compare adapted WAN, pretrained WAN, and the closest public trajectory-video
baseline, [Wan-Move / MoveBench](https://github.com/ali-vilab/Wan-Move), with
matched starting frames, prompts, paths, duration, resolution and seed sets.
Wan-Move's [paper](https://arxiv.org/html/2512.08765v1) reports video/trajectory
evaluation including FVD and endpoint error. Other useful task-specific
references are [MagicMotion](https://arxiv.org/html/2503.16421v2) for multi-object
mask/box correspondence and [Tora](https://arxiv.org/html/2407.21705v3) for
trajectory-conditioned generation. These papers are context, not completed
RouteLab comparisons; no cross-dataset ranking is justified.

Freeze a larger game-held-out football test set and all evaluation choices
before generation. Use multiple seeds and paired play-level statistics rather
than treating correlated frames/windows as independent samples. Measure:

- Per-player average trajectory displacement and endpoint error, in a declared
  coordinate frame with camera compensation, visibility/occlusion rules and
  validated correspondence. Report denominators and tracking failures.
- Identity switches, missing/duplicated players, and all-player retention.
- No-route and shuffled-route ablations to distinguish route responsiveness
  from first-frame/appearance reliance.
- Blinded human ratings of visual plausibility and route adherence, with
  rater agreement and paired uncertainty estimates.
- Distributional video metrics only on a sufficiently large, appropriate
  sample. Seven sealed test windows are too few for a headline FVD claim.

Generated alternatives need not resemble an unobserved factual future; the
metric must evaluate supplied-condition adherence and plausibility separately.
Neither small denoising-loss reductions nor attractive examples establish SOTA.
