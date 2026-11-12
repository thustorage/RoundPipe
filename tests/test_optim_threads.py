import os
import subprocess
import sys

# A thread that calls torch.set_num_threads(1) before roundpipe is imported, as a
# DataLoader(pin_memory=True) pin-memory thread does, must not leave the optimizer
# stream (CPU optimizer kernels, gradient copies) with a single thread.
SCRIPT = """
import threading
import torch
expected = torch.get_num_threads()
t = threading.Thread(target=torch.set_num_threads, args=(1,))
t.start(); t.join()
from roundpipe.device import get_num_devices
from roundpipe.optim_stream import launch_optim_kernel, synchronize_optim
num_gpu = get_num_devices()
if expected > num_gpu * 4:
    expected -= num_gpu
seen = {}
launch_optim_kernel(lambda: seen.update(n=torch.get_num_threads()))
synchronize_optim()
assert seen["n"] == expected, (seen["n"], expected)
"""


def test_optim_stream_thread_count_ignores_earlier_set_num_threads():
    env = dict(os.environ, OMP_NUM_THREADS="8")
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
