"""Canonical target-free RouteLab bundle and route edits."""
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from football_trajectory_control import project, sha256
from nfl_rollout_contract import validate_tracks
from nfl_preprocessing import RENDERER
from nfl_preprocessing.common import atomic_json
from nfl_preprocessing.media import PROFILES, render_control, save_video, validate_profile
SCHEMA = "nfl-route-request-v2"
FILES = {"reference.png", "control.mp4", "trajectories.json", "request.json"}
MODELS = {"wan22": "Wan2.2-Fun-A14B-Control"}

def model_context(model, video):
    if model != "wan22":
        raise ValueError("This release implements WAN only")
    return "Continuous real American football footage. Players move naturally on a consistent football field."

def prepare_bundle(first_frame, tracks, destination, *, model, video=None, seed=42):
    video = dict(video or PROFILES[model])
    if model not in MODELS or type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Invalid model or seed')
    validate_profile(model, video)
    validate_tracks(tracks)
    image = Image.open(first_frame).convert('RGB')
    if image.size != (video['width'], video['height']):
        raise ValueError('First-frame and trajectory geometry must match')
    controls = render_control(tracks, video)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    image.save(destination/'reference.png')
    save_video(destination/'control.mp4', controls, video['fps'])
    atomic_json(destination/'trajectories.json', tracks)
    request = {'schema': SCHEMA, 'model_id': model, 'model': MODELS[model],
               'video': video, 'seed': seed, 'prompt': model_context(model, video),
               'renderer_version': RENDERER, 'future_rgb_is_input': False,
               'required_inputs': ['first_frame', 'trajectories'],
               'control_video_usage': 'preview_only',
               'model_control_source': 'render_public_trajectories',
               'camera_policy': 'implicit_in_image_routes', 'future_camera_is_input': False,
               'trajectory_sha256': sha256(destination/'trajectories.json'),
               'control_pixel_sha256': __import__('hashlib').sha256(np.asarray(controls).tobytes()).hexdigest()}
    atomic_json(destination/'request.json', request)
    manifest = {name: sha256(destination/name) for name in sorted(FILES)}
    atomic_json(destination/'manifest.json', manifest)
    return {**request, 'bundle_sha256': sha256(destination/'manifest.json')}



def edit_routes(tracks, strokes, *, video, editor=None, timing='original'):
    """Strokes are normalized positions drawn on the first-frame canvas.

    With calibrated editor data, preserve the source camera plan. Without it,
    edits are explicitly screen-space paths and carry no yard-speed claim.
    """
    times, original, available = validate_tracks(tracks)
    ids = tracks['player_ids']
    if tracks['coordinate_system'] != 'image_normalized' or timing not in {'original', 'uniform'}:
        raise ValueError('Invalid route coordinate system or timing')
    if not isinstance(strokes, dict) or not set(strokes).issubset(ids):
        raise ValueError('Unknown edited player')
    size = np.array([video['width'], video['height']])
    positions = original.copy()
    field, cameras = None, None
    if editor is not None:
        field = np.asarray(editor['field_xy_yards'], dtype=float)
        cameras = np.asarray(editor['annotation_camera_matrices'], dtype=float)
        if field.shape != original.shape or cameras.shape != (len(times), 3, 3):
            raise ValueError('Editor annotation geometry differs from public tracks')
        projected = np.stack([project(h, p) for h, p in zip(cameras, field)]) / size
        if not np.isfinite(projected).all() or not np.allclose(projected, original, atol=1e-7):
            raise ValueError('Editor annotations disagree with supplied image trajectories')
    reports = {}
    for pid, stroke in strokes.items():
        j = ids.index(pid)
        if not available[:, j].all():
            raise ValueError('Edited route requires continuous annotation coverage')
        path = np.asarray(stroke, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or not 2 <= len(path) <= 2000:
            raise ValueError('Each route needs 2–2000 normalized points')
        if not np.isfinite(path).all() or (path < 0).any() or (path > 1).any():
            raise ValueError('Draw the route inside the first-frame canvas')
        source = original[:, j] if field is None else field[:, j]
        path = path if field is None else project(np.linalg.inv(cameras[0]), path * size)
        path = np.vstack([source[0], path])
        path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-8]]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        old_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
        progress = old_arc/old_arc[-1] if timing == 'original' and old_arc[-1] > 1e-8 else times/times[-1]
        edited = np.stack([np.interp(progress * arc[-1], arc, path[:, axis]) for axis in range(2)], axis=1)
        positions[:, j] = edited if field is None else np.stack([
            project(h, p[None])[0] for h, p in zip(cameras, edited)]) / size
        positions[0, j] = original[0, j]
        speed = np.linalg.norm(np.diff(edited, axis=0), axis=1)/np.diff(times)
        acceleration = np.abs(np.diff(speed))/np.diff(times)[1:]
        reports[pid] = {'distance': float(arc[-1]), 'peak_speed': float(speed.max()),
            'peak_acceleration': float(acceleration.max()) if len(acceleration) else 0.,
            'units': 'yards' if field is not None else 'normalized_image',
            'high_speed_warning': bool(field is not None and speed.max() > 12)}
    untouched = [j for j, pid in enumerate(ids) if pid not in strokes]
    if not np.array_equal(positions[:, untouched], original[:, untouched]) or not np.array_equal(positions[0], original[0]):
        raise AssertionError('Route edit changed an unrelated path or starting position')
    result = copy.deepcopy(tracks)
    result['positions'] = positions.tolist()
    validate_tracks(result)
    return result, {'changed_players': sorted(strokes), 'timing': timing, 'metrics': reports,
                    'camera_policy': 'recorded_replay' if editor is not None else 'screen_space',
                    'unchanged_players_exact': True, 'starting_positions_exact': True}



def validate_bundle(files):
    if set(files) != FILES | {'manifest.json'} or any(not isinstance(v, bytes) for v in files.values()):
        raise ValueError('Unexpected v2 input bundle')
    if sum(map(len, files.values())) > 64_000_000:
        raise ValueError('Oversized v2 input bundle')
    import hashlib
    manifest = json.loads(files['manifest.json'])
    if set(manifest) != FILES or any(hashlib.sha256(files[k]).hexdigest() != v for k, v in manifest.items()):
        raise ValueError('Input checksum mismatch')
    r = json.loads(files['request.json'])
    if r.get('schema') != SCHEMA or r.get('renderer_version') != RENDERER or r.get('future_rgb_is_input') is not False:
        raise ValueError('Incompatible conditioning version')
    if r.get('model_id') not in MODELS or r.get('model') != MODELS[r['model_id']]:
        raise ValueError('Unknown model identity')
    validate_profile(r['model_id'], r['video'])
    if r.get('prompt') != model_context(r['model_id'], r['video']):
        raise ValueError('Context must match the trained two-input contract')
    if type(r.get('seed')) is not int or not 0 <= r['seed'] < 2**31:
        raise ValueError('Invalid seed')
    tracks = json.loads(files['trajectories.json'])
    validate_tracks(tracks)
    if tracks['coordinate_system'] != 'image_normalized' or r.get('trajectory_sha256') != manifest['trajectories.json']:
        raise ValueError('Trajectory identity mismatch')
    if r.get('future_camera_is_input') is not False or r.get('required_inputs') != ['first_frame', 'trajectories']:
        raise ValueError('Unexpected conditioning inputs')
    import io
    image = Image.open(io.BytesIO(files['reference.png']))
    if image.size != (r['video']['width'], r['video']['height']):
        raise ValueError('First frame has incompatible geometry')
    expected = np.asarray(render_control(tracks, r['video']))
    if hashlib.sha256(expected.tobytes()).hexdigest() != r.get('control_pixel_sha256'):
        raise ValueError('Control pixels disagree with public trajectories')
    if r.get('model_control_source') != 'render_public_trajectories' or r.get('control_video_usage') != 'preview_only':
        raise ValueError('Model controls must be rendered directly from public trajectories')
    return r



def control_rgb_frames(bundle, request):
    """Exact model pixels; the browser MP4 has chroma subsampling and is preview-only."""
    if request.get('schema') != SCHEMA:
        raise ValueError('This renderer requires the v2 request contract')
    bundle = Path(bundle)
    if sha256(bundle/'trajectories.json') != request['trajectory_sha256']:
        raise ValueError('Trajectory file changed')
    tracks = json.loads((bundle/'trajectories.json').read_text())
    frames = np.asarray(render_control(tracks, request['video']))
    if __import__('hashlib').sha256(frames.tobytes()).hexdigest() != request['control_pixel_sha256']:
        raise ValueError('Trajectory renderer changed')
    return frames[...,::-1].copy()
