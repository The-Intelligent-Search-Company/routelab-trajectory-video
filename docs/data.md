# Data and trajectory contracts

No video, tracking dataset, calibration annotation, or model weights are
distributed. Supply data you are authorized to process and publish. This code
license does not grant media or dataset rights.

`prepare_inputs.py` accepts a first-frame RGB image already letterboxed to
832×480 and a JSON object containing:

| Field | Contract |
|---|---|
| `schema` | `nfl-first-frame-tracks-v1` |
| `coordinate_system` | `image_normalized` for model bundles |
| `times_seconds` | Strictly increasing, starting at 0; covering at least 3 seconds |
| `player_ids` | Exactly 22 unique strings; keep stable identity and ordering |
| `positions` | Finite `[T,22,2]` normalized image coordinates |
| `instruction_available` | Binary `[T,22]` supplied-route availability, not visual visibility |
| `team_ids` | Optional length-22 list: 0/1, or 2 for unknown |

Positions can leave the frame; do not clip them into false stationary edge
tracks. Sampling is linear at 16Hz with availability checked at both bracketing
samples. No extrapolation beyond the supplied duration is allowed. Identity
colors use sorted player IDs and team rings; rendering order is deterministic.
The renderer excludes the ball. Unknown fields such as future camera matrices
are rejected. A field-coordinate calibration helper exists, but canonical
bundles require image-normalized routes created before inference.

For training, `windows.json` contains `rows`, each with:

```json
{
  "sample_id": "your_unique_window_id",
  "game_id": "your_game_id",
  "play_id": "your_play_id",
  "split": "train",
  "first_frame": "window/start.png",
  "trajectories": "window/routes.json",
  "target_video": "window/target.mp4"
}
```

All three paths are resolved relative to the index file. Use `train` and
`validation` splits; validation is mapped to the trainer's historical name
`development`. Keep all windows from a game in one split, including alternate
angles and repeated plays. Duplicate target bytes must not cross splits.
Test data are rejected by the packaging adapter and must remain sealed elsewhere.

Targets must already have real aligned timestamps 0 through 3 seconds at 16fps,
49 frames, 832×480, and the matching start image. The packaging adapter checks
geometry, timing, split consistency, duplicate-source leakage, and approximate
frame-zero agreement. These are mechanical checks; they do not establish
correct player identity, accurate calibration, or legal permission.

For the historical predicted-futures experiment, the supplied 10Hz model paths
were linearly resampled to 16Hz. Player slots were matched at the start using
positions and team roles, with a uniqueness margin and a role-ID cross-check.
The supplied ball was retained for visualization but excluded from WAN controls.
That external trajectory predictor is not implemented or redistributed here.
