"""Expose the frozen IQL actor through EVA's existing OpenPI wire protocol."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    if settings["format"] != "r1lite_iql_eval_v1":
        raise ValueError("Expected an R1 Lite IQL evaluation configuration")
    for source, expected in settings["source_sha256"].items():
        if sha256(source) != expected:
            raise ValueError(f"Evaluation dependency changed: {source}")
    metadata = json.loads((Path(settings["policy_directory"]) / "policy.json").read_text())
    if metadata["checkpoint_step"] != 3000 or metadata["execution_horizon"] != 1:
        raise ValueError("This evaluation requires the fixed step-3000, single-step actor")
    if settings["eval_cfg"]["tasks"][0]["prompt_en"] != metadata["task"]:
        raise ValueError("Evaluation instruction differs from the policy bundle")
    return settings, metadata


def build_policy(settings):
    import torch
    from process.r1lite_policy import R1LitePolicy

    torch.set_num_threads(settings["torch_num_threads"])
    torch.manual_seed(settings["seed"])
    return R1LitePolicy(settings["policy_directory"], device=settings["device"])


class IQLPolicy:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation):
        if observation["prompt"] != self.policy.metadata["task"]:
            raise ValueError("The IQL actor is bound to the PackTask instruction")
        # Msgpack arrays borrow immutable message bytes; give Torch owned observation buffers.
        inputs = {"state": np.array(observation["state"], copy=True),
                  **{key: np.array(frame, copy=True) for key, frame in observation["images"].items()}}
        action = self.policy.predict(inputs, deterministic=True)
        return {"actions": action[np.newaxis], "action_mode": "qpos"}


def serve(settings):
    import logging
    import threading
    import websockets.exceptions
    import websockets.sync.server
    from openpi_client import msgpack_numpy

    policy = IQLPolicy(build_policy(settings))
    metadata = {"server_name": "robometer-iql-step-3000", "action_mode": "qpos",
                "chunk_size": 1, "action_dim": 14, "supports_reset": False,
                "actor_sha256": sha256(Path(settings["policy_directory"]) / "actor.pt")}
    lock = threading.Lock()

    def handler(connection):
        packer = msgpack_numpy.Packer()
        connection.send(packer.pack(metadata))
        try:
            for message in connection:
                observation = msgpack_numpy.unpackb(message)
                with lock:
                    result = policy.infer(observation)
                connection.send(packer.pack(result))
        except websockets.exceptions.ConnectionClosed:
            pass

    logging.basicConfig(level=logging.INFO)
    with websockets.sync.server.serve(handler, "127.0.0.1", settings["serving_port"],
                                      compression=None, max_size=None) as server:
        print(f"IQL actor ready on 127.0.0.1:{settings['serving_port']}", flush=True)
        server.serve_forever()


def evaluate(config_path, settings):
    import socket
    import time

    # Refuse occupied ports; the native console otherwise attempts to reclaim its port.
    for port in (settings["serving_port"], settings["client_web_port"]):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
    env = os.environ.copy()
    env["IQL_EVAL_CONFIG"] = str(config_path)
    project = Path(__file__).resolve().parents[1]
    client = Path(settings["client_root"])
    server = subprocess.Popen([sys.executable, "-m", "process.eval_iql", "--config",
                               str(config_path), "--serve-only"], cwd=project, env=env)
    try:
        deadline = time.monotonic() + settings["startup_timeout_s"]
        while True:
            if server.poll() is not None:
                raise RuntimeError("IQL policy server stopped during startup")
            try:
                with socket.create_connection(("127.0.0.1", settings["serving_port"]), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("IQL policy server startup timed out")
                time.sleep(0.1)
        # The embedded client runs in DenseRL's environment; IQL stays in this project's environment.
        client_env = dict(env, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                          OPENBLAS_NUM_THREADS="1")
        command = ["bash", "-c", 'set -eo pipefail; source "$1"; source "$2"; export PYTHONPATH="$3/src:$3${PYTHONPATH:+:$PYTHONPATH}"; exec "$4" -m main --config "$5" --web-port "$6"',
                   "iql-eva-client", settings["ros_setup"], settings["ros_env_file"], str(client),
                   settings["client_python"], settings["client_config"], str(settings["client_web_port"])]
        subprocess.run(command, cwd=client, env=client_env, check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
