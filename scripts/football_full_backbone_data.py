"""Training data contracts and video I/O from RouteLab."""
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from football_trajectory_control import sha256

def validate_config(config):
    v, t = config['video'], config['training']
    if t['scope'] != 'full_backbone' or any('lora' in k.lower() for k in t):
        raise ValueError('This implementation requires full backbone training without LoRA')
    if set(t['experts']) != {'high_noise', 'low_noise'}:
        raise ValueError('Both Wan experts are required')
    if v['width'] % 16 or v['height'] % 16 or v['num_frames'] % 4 != 1:
        raise ValueError('Invalid Wan video dimensions')
    if min(v[k] for k in ['width', 'height', 'fps', 'num_frames']) <= 0 or v.get('start_seconds', 0) < 0 or t['max_steps'] < 1 or t['save_every'] < 1:
        raise ValueError('Invalid dimensions or training length')
    if not 0 <= t['control_dropout'] < 1:
        raise ValueError('Invalid control dropout')
    if t.get('gradient_accumulation_steps',1)<1 or not 0<=t.get('text_dropout',0)<1:
        raise ValueError('Invalid accumulation or text dropout')
    if 'sigma_boundary' in t and not 0<t['sigma_boundary']<1:
        raise ValueError('Invalid expert sigma boundary')



def split_guard(rows):
    games = {}
    samples = set()
    for row in rows:
        if row['sample_id'] in samples:
            raise ValueError('Duplicate sample')
        samples.add(row['sample_id'])
        game = str(row['game_id'])
        if game in games and games[game] != row['split']:
            raise ValueError('Game leakage across splits')
        games[game] = row['split']
    if not any(r['split'] in {'train_debug','train'} for r in rows):
        raise ValueError('Missing training split')
    if not any(r['split'] == 'development' for r in rows):
        raise ValueError('Missing development split')



def read_video(path, times, size):
    cap = cv2.VideoCapture(str(path))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count < 1:
        raise ValueError('Unreadable video')
    frames, actual = [], []
    try:
        for t in times:
            index = int(np.rint(t * fps))
            if index >= count:
                raise ValueError('Training video requires padding; choose a shorter window')
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Video decode failure')
            if abs(frame.shape[1] / frame.shape[0] - size[0] / size[1]) > .001:
                raise ValueError('Aspect ratio change would corrupt projection')
            frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            actual.append(index / fps)
    finally:
        cap.release()
    return frames, actual



def save_video(path, frames, fps):
    height, width = frames[0].shape[:2]
    writer = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '0', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            writer.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError('Training video encode failed')



def verify_dataset(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, checksum in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != checksum:
            raise ValueError('Dataset integrity failure: ' + name)
    split_guard(manifest['rows'])
    if manifest['future_rgb_in_controls'] is not False:
        raise ValueError('Future RGB leakage')
    return manifest

