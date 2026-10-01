"""Real timestamp sampling, geometry, and identity-trails-v1 controls."""
import colorsys
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from football_full_backbone_data import save_video
from nfl_rollout_contract import validate_tracks, sample_tracks
PROFILES = {"wan22": {"fps": 16, "num_frames": 49, "width": 832, "height": 480}}

def validate_profile(model, profile):
    if model not in PROFILES:
        raise ValueError('Unsupported model profile')
    w, h, n, fps = (profile[k] for k in ('width', 'height', 'num_frames', 'fps'))
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (w, h, n, fps)):
        raise ValueError('Video dimensions and frame rate must be integers')
    divisor = 32 if model == 'h3' else 16
    if min(w, h, n, fps) <= 0 or w % divisor or h % divisor:
        raise ValueError('Video dimensions violate model profile')
    if model == 'h3' and (n < 22 or (n - 5) % 17):
        raise ValueError('H3 needs 17n+5 native frames')
    if model != 'h3' and (n < 5 or (n - 1) % 4):
        raise ValueError('Video profile needs 4n+1 native frames')
    if model == 'cosmos3_nano' and n < 61:
        raise ValueError('Cosmos transfer loader needs at least 61 frames')



def probe_video(path):
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height:frame=best_effort_timestamp_time', '-of', 'json', str(path)])
    data = json.loads(raw)
    times = np.asarray([float(f['best_effort_timestamp_time']) for f in data.get('frames', [])])
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Missing or nonmonotonic decoded presentation timestamps')
    stream = data['streams'][0]
    return times, (int(stream['width']), int(stream['height']))



def nearest_frames(pts, requested, max_error=.025):
    pts, requested = np.asarray(pts), np.asarray(requested)
    if not np.isfinite(requested).all() or np.any(np.diff(requested) <= 0):
        raise ValueError('Requested timestamps must be finite and increasing')
    if requested[0] < pts[0] - 1e-8 or requested[-1] > pts[-1] + 1e-8:
        raise ValueError('Window exceeds real footage; padding is forbidden')
    right = np.searchsorted(pts, requested).clip(0, len(pts)-1)
    left = (right-1).clip(0, len(pts)-1)
    indices = np.where(np.abs(pts[left]-requested) <= np.abs(pts[right]-requested), left, right)
    if np.any(np.diff(indices) <= 0):
        raise ValueError('Resampling would duplicate a source frame')
    if np.max(np.abs(pts[indices]-requested)) > max_error + 1e-8:
        raise ValueError('Source timing error exceeds configured tolerance')
    return indices



def letterbox_transform(source_size, target_size):
    sw, sh = source_size
    tw, th = target_size
    scale = min(tw/sw, th/sh)
    rw, rh = round(sw*scale), round(sh*scale)
    left, top = (tw-rw)//2, (th-rh)//2
    return {'source_size': list(source_size), 'output_size': list(target_size),
            'resized_size': [rw, rh], 'padding_left_top': [left, top],
            'matrix': [[rw/sw, 0., left], [0., rh/sh, top], [0., 0., 1.]],
            'policy': 'aspect_preserving_letterbox'}



def transform_frame(frame, transform):
    tw, th = transform['output_size']
    rw, rh = transform['resized_size']
    x, y = transform['padding_left_top']
    result = np.zeros((th, tw, 3), np.uint8)
    result[y:y+rh, x:x+rw] = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
    return result



def decode_frames(path, indices, transform):
    wanted = set(map(int, indices))
    capture, found = cv2.VideoCapture(str(path)), {}
    try:
        for i in range(int(max(indices))+1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Decoder ended before the timestamp inventory')
            if (frame.shape[1], frame.shape[0]) != tuple(transform['source_size']):
                raise ValueError('Source dimensions changed inside a clip')
            if i in wanted:
                found[i] = transform_frame(frame, transform)
    finally:
        capture.release()
    return [found[int(i)] for i in indices]



def player_color(player_id, ids):
    hue = sorted(ids).index(player_id) / len(ids)
    return tuple(round(v*255) for v in colorsys.hsv_to_rgb(hue, .72, 1)[::-1])



def render_control(tracks, profile, trail_seconds=.25):
    """No RGB, camera matrices, labels, or target-video path is accepted here."""
    validate_tracks(tracks)
    if tracks['coordinate_system'] != 'image_normalized':
        raise ValueError('This renderer accepts image-space tracks only')
    n, fps, w, h = (profile[k] for k in ('num_frames', 'fps', 'width', 'height'))
    points, valid = sample_tracks(tracks, np.arange(n)/fps)
    points *= [w, h]
    ids, teams = tracks['player_ids'], tracks.get('team_ids', [2]*22)
    frames, tail = [], max(1, round(trail_seconds*fps))
    for t in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        for j in sorted(range(len(ids)), key=lambda j: ids[j]):
            color = player_color(ids[j], ids)
            for k in range(max(1, t-tail+1), t+1):
                if valid[k-1, j] and valid[k, j] and np.abs(points[k-1:k+1, j]).max() < 1e6:
                    a, b = np.rint(points[k-1:k+1, j]).astype(int)
                    cv2.line(frame, tuple(a), tuple(b), tuple(int(v*.5) for v in color), 2, cv2.LINE_AA)
            x, y = points[t, j]
            if valid[t, j] and 0 <= x < w and 0 <= y < h:
                point = tuple(np.rint([x, y]).astype(int))
                ring = (240,240,240) if teams[j] == 0 else (100,100,100)
                cv2.circle(frame, point, 7, ring, 2, cv2.LINE_AA)
                cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)
        frames.append(frame)
    return frames

