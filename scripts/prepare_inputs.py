"""Prepare a canonical bundle from a user-owned frame and normalized routes."""
import argparse
import json
from pathlib import Path
from nfl_conditioning import prepare_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first-frame', type=Path, required=True)
    p.add_argument('--trajectories', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    result = prepare_bundle(args.first_frame, json.loads(args.trajectories.read_text()),
                            args.output, model='wan22', seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
