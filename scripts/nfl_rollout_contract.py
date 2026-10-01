"""Twenty-two-player trajectory contract; no future camera lookup."""
import cv2
import numpy as np
from football_trajectory_control import project
SCHEMA = "nfl-first-frame-tracks-v1"
ALLOWED = {"schema", "coordinate_system", "times_seconds", "player_ids", "positions", "instruction_available", "initial_pixel_xy", "initial_binding_valid", "team_ids"}

def validate_tracks(tracks):
    if set(tracks) - ALLOWED or tracks.get('schema') != SCHEMA:
        raise ValueError('Unexpected trajectory fields or schema; no future camera/text inputs are accepted')
    if tracks.get('coordinate_system') not in {'image_normalized', 'field_yards'}:
        raise ValueError('Specify image_normalized or field_yards coordinates')
    ids = tracks['player_ids']
    if len(ids) != 22 or len(set(ids)) != 22 or any(not isinstance(x, str) for x in ids):
        raise ValueError('Expected 22 unique string player IDs')
    times = np.asarray(tracks['times_seconds'], dtype=np.float64)
    xy = np.asarray(tracks['positions'], dtype=np.float64)
    available = np.asarray(tracks['instruction_available'])
    if (times.ndim != 1 or len(times) < 2 or abs(times[0]) > 1e-8 or
        not np.isfinite(times).all() or not (np.diff(times) > 0).all()):
        raise ValueError('Trajectory timestamps must start at zero and increase strictly')
    if xy.shape != (len(times),22,2) or not np.isfinite(xy).all():
        raise ValueError('Invalid player coordinates')
    if available.shape != xy.shape[:2] or not np.isin(available, [0,1]).all():
        raise ValueError('Invalid instruction availability; it is distinct from visibility')
    if 'team_ids' in tracks and (len(tracks['team_ids']) != 22 or
        any(x not in (0,1,2) for x in tracks['team_ids'])):
        raise ValueError('Team IDs must be 0/1 or 2 for unknown')
    if tracks['coordinate_system'] == 'field_yards':
        initial = np.asarray(tracks.get('initial_pixel_xy'), dtype=float)
        valid = np.asarray(tracks.get('initial_binding_valid'))
        if initial.shape != (22,2) or valid.shape != (22,) or not np.isin(valid,[0,1]).all():
            raise ValueError('Field routes require initial image bindings as normalized trajectory origins')
        if not np.isfinite(initial).all() or valid.sum() < 6 or not available[0,valid.astype(bool)].all():
            raise ValueError('At least six valid initial player bindings are required')
        if np.any(initial[valid.astype(bool)] < 0) or np.any(initial[valid.astype(bool)] > 1):
            raise ValueError('Visible initial bindings must be within the starting frame')
    elif 'initial_pixel_xy' in tracks or 'initial_binding_valid' in tracks:
        raise ValueError('Image-space trajectory origins already supply the initial bindings')
    return times, xy, available.astype(bool)



def sample_tracks(tracks, frame_times):
    times, xy, available = validate_tracks(tracks)
    target = np.asarray(frame_times, dtype=float)
    if target.ndim != 1 or not np.isfinite(target).all() or target.min() < 0 or target.max() > times[-1]+1e-8:
        raise ValueError('Requested video exceeds supplied trajectory duration')
    result = np.empty((len(target),22,2), dtype=float)
    for player in range(22):
        for axis in range(2):
            result[:,player,axis] = np.interp(target,times,xy[:,player,axis])
    right = np.searchsorted(times,target,side='left').clip(0,len(times)-1)
    left = (right-1).clip(0,len(times)-1)
    exact = np.isclose(times[right],target,atol=1e-8,rtol=0)
    supported = available[left] & available[right]
    supported[exact] = available[right[exact]]
    return result, supported



def initial_camera(tracks, size):
    """Fit from route origins only; future routes cannot change initial calibration."""
    _, xy, _ = validate_tracks(tracks)
    if tracks['coordinate_system'] != 'field_yards':
        return None
    valid = np.asarray(tracks['initial_binding_valid'], bool)
    points = np.asarray(tracks['initial_pixel_xy'], float)*np.asarray(size)
    h, inliers = cv2.findHomography(xy[0,valid],points[valid],cv2.RANSAC,3.0)
    if h is None or inliers.sum() < 6 or not np.isfinite(h).all() or np.linalg.cond(h) > 1e10:
        raise ValueError('Initial-frame calibration failed; do not fall back to future camera data')
    errors = np.linalg.norm(project(h,xy[0,valid])-points[valid],axis=1)
    if np.quantile(errors,.9) > 5:
        raise ValueError('Initial player bindings disagree with field calibration')
    return h

