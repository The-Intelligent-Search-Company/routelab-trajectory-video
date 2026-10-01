"""Package already-aligned, user-owned windows; never use test targets."""
import argparse
import json
from pathlib import Path
import re
import shutil
import cv2
import numpy as np
from PIL import Image
from football_full_backbone_data import split_guard, verify_dataset
from football_trajectory_control import sha256, write_json
from nfl_conditioning import prepare_bundle
from nfl_preprocessing.media import PROFILES, probe_video, render_control


def build(index, destination):
    index, destination = Path(index), Path(destination)
    rows = json.loads(index.read_text())['rows']
    rows = [dict(row, split={'validation': 'development'}.get(row['split'], row['split'])) for row in rows]
    if not rows or any(row['split'] not in {'train', 'development'} for row in rows):
        raise ValueError('Supply only training and validation rows; sealed test pixels must remain separate')
    split_guard(rows)
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', row['sample_id']) for row in rows):
        raise ValueError('Sample IDs must be safe directory names')
    destination.mkdir(parents=True, exist_ok=False)
    video = PROFILES['wan22']
    output = []
    source_splits = {}
    for row in rows:
        paths = {k: (index.parent/row[k]).resolve() for k in ['first_frame', 'trajectories', 'target_video']}
        source_hash = sha256(paths['target_video'])
        if source_splits.setdefault(source_hash, row['split']) != row['split']:
            raise ValueError('Identical target videos cross data splits')
        times, size = probe_video(paths['target_video'])
        if size != (832, 480) or len(times) != 49 or not np.allclose(times, np.arange(49)/16, atol=.001):
            raise ValueError('Targets must already be aligned 49-frame 832x480 clips at 16 fps, starting at zero')
        tracks = json.loads(paths['trajectories'].read_text())
        folder = destination/row['sample_id']
        request = prepare_bundle(paths['first_frame'], tracks, folder, model='wan22')
        # Do not derive the reference from future footage here. Check supplied start alignment.
        cap = cv2.VideoCapture(str(paths['target_video']))
        ok, start = cap.read()
        cap.release()
        reference = np.asarray(Image.open(paths['first_frame']).convert('RGB'))
        if not ok or np.mean(np.abs(start[..., ::-1].astype(float)-reference.astype(float))) > 8:
            raise ValueError('Supplied reference is inconsistent with target frame zero')
        shutil.copyfile(paths['target_video'], folder/'target.mp4')
        rgb = np.asarray(render_control(tracks, video))[..., ::-1].copy()
        np.savez_compressed(folder/'control_frames.npz', rgb=rgb)
        prefix = row['sample_id']+'/'
        output.append({'sample_id': row['sample_id'], 'game_id': str(row['game_id']),
            'play_id': str(row['play_id']), 'split': row['split'], 'prompt': request['prompt'],
            'source_video_sha256': source_hash, 'video': prefix+'target.mp4',
            'reference_image': prefix+'reference.png', 'control_video': prefix+'control.mp4',
            'trajectories': prefix+'trajectories.json', 'control_frames': prefix+'control_frames.npz'})
    manifest = {'schema_version': 'nfl-native-route-dataset-v1', 'model_id': 'wan22', 'video': video,
        'future_rgb_in_controls': False, 'loss_mask_policy': 'native_first_frame_prefix',
        'camera': 'implicit_in_image_routes', 'test_examples_available_to_trainer': False,
        'rows': output, 'files': {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob('*')) if p.is_file()}}
    write_json(destination/'manifest.json', manifest)
    verify_dataset(destination)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = build(args.index, args.output)
    print(json.dumps({'windows': len(result['rows']), 'manifest_sha256': sha256(args.output/'manifest.json')}))


if __name__ == '__main__':
    main()
