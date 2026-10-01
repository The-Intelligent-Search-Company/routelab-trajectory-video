"""Atomic JSON and integrity primitives."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from football_trajectory_control import sha256

class IntegrityError(ValueError):
    """A completed artifact changed; never silently turn this into a rejected sample."""



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()



def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)



def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes dataset root')
    return path



def file_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'COMPLETE.json'}



def verify_files(root, files):
    for name, info in files.items():
        path = inside(root, name)
        if not path.is_file() or path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise IntegrityError(f'Artifact integrity failure: {name}')
