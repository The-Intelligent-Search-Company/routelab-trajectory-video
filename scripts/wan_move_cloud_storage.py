"""Local-only replacement for the original storage adapter; no cloud access.

The historical module name keeps the extracted training imports stable.
Committed artifacts are immutable, inventory checked, and fully hashed.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

MARKER = 'artifact_manifest.json'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks are not allowed in committed artifacts')
        if path.is_file() and path.name != MARKER:
            files[str(path.relative_to(root))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return files


def read_manifest(root, verify_hashes=False):
    root = Path(root).resolve()
    files = json.loads((root/MARKER).read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != MARKER}
    if actual != set(files):
        raise ValueError('Artifact inventory mismatch')
    for name, info in files.items():
        path = root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe artifact path')
        if path.stat().st_size != info['bytes'] or (verify_hashes and digest(path) != info['sha256']):
            raise ValueError('Artifact integrity failure: '+name)
    return files


def commit_directory(root):
    root = Path(root)
    files = inventory(root)
    if (root/MARKER).exists():
        if read_manifest(root, True) != files:
            raise ValueError('Committed directory changed')
        return files
    with (root/MARKER).open('x') as stream:
        json.dump(files, stream, indent=2)
        stream.write('\n')
    return files


def publish(source, destination, workers=8):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate trees')
    expected = inventory(source)
    if (destination/MARKER).exists():
        if read_manifest(destination, True) != expected:
            raise ValueError('Refusing to replace a different committed artifact')
        return expected
    # Preserve incomplete output rather than silently restarting it.
    destination.mkdir(parents=True, exist_ok=False)
    def transfer(name):
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
        if digest(target) != expected[name]['sha256']:
            raise ValueError('Copy checksum mismatch')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(transfer, expected))
    if inventory(source) != expected:
        raise ValueError('Source changed during publication')
    committed = commit_directory(destination)
    if committed != expected:
        raise ValueError('Publication inventory mismatch')
    return committed


def restore(source, destination, workers=8):
    read_manifest(source, True)
    return publish(source, destination, workers)
