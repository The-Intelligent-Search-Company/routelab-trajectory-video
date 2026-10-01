# Provenance and release scope

This is a sanitized standalone extraction of first-party RouteLab research
code, prepared 2026-10-01. It contains actual renderer, conditioning, caching,
FSDP training, export and inference implementations, not pseudocode.
`results/source_provenance.json` records source filenames and SHA256 hashes,
extracted functions, and release hashes. No internal git history or transcripts
are included. Source hashes allow the owners to cross-check private evidence;
they do not make withheld evidence publicly inspectable.

The historical training identity is
`1442794c9fe6491c936e1ea7778a53ae78d17453013a7f3c3d43f118efc08b1f`.
The native dataset manifest SHA256 is
`785c7807c038aacee3c7da33c352b362bfb6edde8cedb7486a5d27c0aaec979b`.
The current and sealed historical trainer source were byte-identical at audit.
Reported losses were read from the original 0–8 and 8–32 phase reports, rather
than estimated from plots or inferred from completion status.

## Changes in the public extraction

- Cloud object storage and dashboard status are replaced with local-only
  checksummed storage and structured standard-output progress.
- Source identity hashes the complete released Python tree. It will not match
  the historical source/config identity; the public code must not impersonate
  the old experiment's receipts.
- Unrelated models, production web services, cloud launchers, credentials,
  bucket names, user paths and operational schedules are omitted.
- The standalone dataset packager, CLI wiring and release tests are new.
  The packager expects already aligned authorized input windows.
- Expert export defaults are corrected to eight ranks and step 32 for this
  release, and a local manifest is committed after export.
- A historical report field named `unique_training_plays` counted windows.
  The release labels it `training_windows`; actual play counts were audited
  separately and are recorded in the evidence report.

The original training publisher did not independently reread large remote
checkpoint shards. A separate completed readback verified all 18 final files,
343,038,683,553 bytes, before evaluation/export. This distinction is preserved;
the training process exit alone was not the full-byte integrity gate.

## Reproducibility limits

Public CPU tests cover software contracts and release wiring. No GPU training,
export, model download, inference, tracker probe or paid compute was rerun while
preparing this publication. Consequently, the standalone GPU path is extracted
from the executed implementation but is not independently end-to-end certified
in this new layout. See `results/release_validation.json` for actual release
checks and environment.

The proprietary input data and adapted checkpoints are not distributed.
Readers can reproduce the procedure with authorized data, but cannot exactly
regenerate the historical reported outputs using this repository alone.
PyTorch/CUDA/FFmpeg builds and renderer versions can affect numerical or encoded
bytes. Preserve the configuration, package inventory, model revision, source
commit, input hashes, full RNG/checkpoint state and output receipts for new runs.

## Media and third-party rights

No football footage, tracking files, generated rollout images/videos, model
weights, or vendor source trees are published here. The empirical figure was
prepared privately from existing results and is withheld pending an explicit
media/data rights grant. Prior link-sharing of a demonstration does not itself
establish redistribution rights. No generated illustration substitutes for an
empirical result. The code license covers only the released code; upstream
dependencies and model assets retain their own terms. See NOTICE.
