"""Launch the embedded EVA execution/recording client for the DSRL --config.

Input: source-bound client configuration, isolated interpreter and ROS setup.
Output: EVA console and native recordings. Setup/START remain operator actions.
"""

import argparse
import os
from pathlib import Path
import socket
import subprocess

from process.runtime_checks import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    path = parser.parse_args().config.resolve()
    cfg = load_settings(path)
    client = cfg['client']
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', client['web_port']))
    env = dict(os.environ, DSRL_CONFIG=str(path), JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    root = str(Path(__file__).resolve().parents[1])
    command = ['bash', '-c',
               'set -eo pipefail; source "$1"; source "$2"; '
               'export PYTHONPATH="$3:$4/src:$4${PYTHONPATH:+:$PYTHONPATH}"; '
               'exec "$5" -m main --config "$6" --web-port "$7"',
               'dsrl-eva-client', client['ros_setup'], client['ros_env_file'], root,
               client['root'], client['python'], client['config'], str(client['web_port'])]
    subprocess.run(command, cwd=client['root'], env=env, check=True)


if __name__ == '__main__':
    main()
