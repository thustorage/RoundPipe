import os
import subprocess
import sys


def test_import_without_visible_gpu():
    # Planner defaults are evaluated at import; a host with no visible GPU must still import.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", "import roundpipe"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
