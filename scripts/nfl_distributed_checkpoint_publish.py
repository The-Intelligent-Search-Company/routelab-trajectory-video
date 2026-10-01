"""Keep GPU collectives short while rank zero performs slow checkpoint I/O."""
import queue
import threading
import time

import torch.distributed as dist


def run_rank_zero_io(operation, *, poll_seconds=5):
    """All ranks participate; publish success only after the operation returns.

    Bulk cloud I/O runs in a daemon thread without touching tensors or process
    groups. Each status broadcast completes promptly, instead of leaving a
    barrier/broadcast outstanding throughout a potentially long upload. The
    worker's existing outer execution timeout still bounds the entire job.
    """
    if poll_seconds <= 0:
        raise ValueError('Polling interval must be positive')
    results = queue.Queue()
    rank = dist.get_rank()
    if rank == 0:
        def work():
            try:
                operation()
                results.put({'done': True, 'error': None})
            except BaseException as error:
                results.put({'done': True, 'error': f'{type(error).__name__}: {error}'})
        threading.Thread(target=work, name='checkpoint-publication', daemon=True).start()
    while True:
        status = [{'done': False, 'error': None}]
        if rank == 0:
            try:
                status[0] = results.get_nowait()
            except queue.Empty:
                pass
        dist.broadcast_object_list(status, src=0)
        if status[0]['done']:
            if status[0]['error']:
                raise RuntimeError('Checkpoint publication failed: ' + status[0]['error'])
            return
        time.sleep(poll_seconds)
