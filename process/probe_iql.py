"""在录制观测上检查 IQL actor 和 EVA WebSocket 接口，不连接机器人。

从显式登记的 parquet/三路视频按同一帧号读取真实观测，核对直接推理与 RPC 输出并测量延迟。
输入：--config 指定评测配置、回放文件与采样帧。输出：包含源 SHA、动作和耗时的 JSON 报告。
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import socket
import sys
import time

import numpy as np

from process.iql_evaluation import IQLPolicy, build_policy, load_settings, sha256


def probe(config_path):
    import cv2
    import pyarrow.parquet as pq
    import websockets.sync.client
    from openpi_client import msgpack_numpy

    config = json.loads(config_path.read_text())
    settings_path = Path(config["evaluation_config"])
    settings, _ = load_settings(settings_path)
    for path, expected in config["source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Replay input changed: {path}")
    table = pq.read_table(config["parquet"])
    rows = table.to_pylist()
    policy = IQLPolicy(build_policy(settings))
    captures = {key: cv2.VideoCapture(path) for key, path in config["videos"].items()}
    observations = []
    try:
        for index in config["frame_indices"]:
            row = rows[index]
            if row["frame_index"] != index or not row["observation.images_valid"]:
                raise ValueError(f"Replay frame is not valid and aligned: {index}")
            images = {}
            for key, capture in captures.items():
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot read recorded camera frame: {key}/{index}")
                images[key] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            observations.append({"state": np.asarray(row[config["state_key"]], dtype=np.float32),
                                 "images": images, "prompt": policy.policy.metadata["task"]})
    finally:
        for capture in captures.values():
            capture.release()
    for _ in range(config["warmup_requests"]):
        policy.infer(observations[0])
    records = []
    for index, observation in zip(config["frame_indices"], observations):
        started = time.perf_counter()
        action = policy.infer(observation)["actions"]
        records.append({"frame_index": index, "action": action.tolist(),
                        "camera_source_times": {key: rows[index][f"observation.image_source_time.{key}"]
                                                for key in captures},
                        "inference_ms": (time.perf_counter() - started) * 1000})

    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", settings["serving_port"]))
    server = subprocess.Popen([sys.executable, "-m", "process.eval_iql", "--config",
                               str(settings_path), "--serve-only"], cwd=Path(__file__).resolve().parents[1])
    try:
        deadline = time.monotonic() + settings["startup_timeout_s"]
        with ExitStack() as stack:
            while True:
                if server.poll() is not None:
                    raise RuntimeError("IQL RPC server stopped before the probe")
                try:
                    connection = stack.enter_context(websockets.sync.client.connect(
                        f"ws://127.0.0.1:{settings['serving_port']}", compression=None,
                        max_size=None, open_timeout=0.5))
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("IQL RPC server startup timed out")
                    time.sleep(0.1)
            metadata = msgpack_numpy.unpackb(connection.recv())
            if metadata["chunk_size"] != 1 or metadata["action_dim"] != 14:
                raise ValueError("EVA handshake changed the single-step action contract")
            if metadata["actor_sha256"] != sha256(Path(settings["policy_directory"]) / "actor.pt"):
                raise ValueError("The RPC service loaded a different actor")
            packer = msgpack_numpy.Packer()
            for observation, record in zip(observations, records):
                started = time.perf_counter()
                connection.send(packer.pack(observation))
                rpc = msgpack_numpy.unpackb(connection.recv())["actions"]
                record["rpc_ms"] = (time.perf_counter() - started) * 1000
                np.testing.assert_allclose(rpc, record["action"], rtol=0, atol=1e-6)
                record["rpc_matches_direct"] = True
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    report = {"format": "r1lite_iql_replay_probe_v1", "robot_connected": False,
              "config_sha256": sha256(config_path), "evaluation_config_sha256": sha256(settings_path),
              "source_sha256": config["source_sha256"], "records": records, "handshake": metadata}
    output = Path(config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    probe(parser.parse_args().config.resolve())


if __name__ == "__main__":
    main()
