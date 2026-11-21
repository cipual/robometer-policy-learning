"""Probe frozen VLA/noise and reward RPC on recorded RGB/qpos, without a robot.

Input: --config with explicit parquet/video files and service addresses.
Output: source hashes, finite model outputs and measured RPC times in JSON.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyarrow.parquet as pq

from process.frozen_vla_client import FrozenVLAClient
from process.runtime_checks import sha256
from process.reward_client import RewardClient


def probe(config_path):
    cfg = json.loads(config_path.read_text())
    rows = pq.read_table(cfg["parquet"]).to_pylist()
    frames = {}
    for key, path in cfg["videos"].items():
        capture = cv2.VideoCapture(path)
        images = []
        try:
            for _ in range(cfg["frames"]):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Incomplete probe video: {key}")
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        frames[key] = np.asarray(images)
    vla = FrozenVLAClient(cfg["vla_host"], cfg["vla_port"])
    observation = {"state": np.asarray(rows[0]["observations.state.qpos"], np.float32),
                   "images": {key: images[0] for key, images in frames.items()}, "prompt": cfg["task"]}
    start = time.perf_counter()
    features = vla.features(observation)
    feature_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    repeated_features = vla.features(observation)
    warm_features_ms = (time.perf_counter() - start) * 1000
    for key in features:
        np.testing.assert_array_equal(features[key], repeated_features[key])
    if not all(np.isfinite(value).all() for value in features.values()):
        raise ValueError("Non-finite frozen features")
    shape = (vla.metadata["model_horizon"], vla.metadata["noise_dim"])
    noise = np.random.default_rng(cfg["seed"]).normal(size=shape).astype(np.float32)
    actions, timings = [], []
    for value in (np.zeros(shape, np.float32), np.zeros(shape, np.float32), noise):
        start = time.perf_counter()
        _, token = vla.prepare(observation)
        result = np.asarray(vla.infer({"dsrl_context_id": token}, noise=value, baseline_noise=value)["actions"])
        timings.append((time.perf_counter() - start) * 1000)
        if result.shape != (8, 14) or not np.isfinite(result).all():
            raise ValueError("Invalid frozen VLA action chunk")
        actions.append(result)
    np.testing.assert_array_equal(actions[0], actions[1])
    if np.array_equal(actions[0], actions[2]):
        raise ValueError("Explicit diffusion noise did not change the VLA actions")
    prepared_timings = []
    for _ in range(3):
        start = time.perf_counter()
        prepared_features, context_id = vla.prepare(observation)
        prepare_ms = (time.perf_counter() - start) * 1000
        for key in features:
            np.testing.assert_array_equal(features[key], prepared_features[key])
        start = time.perf_counter()
        prepared_actions = vla.infer({"dsrl_context_id": context_id}, noise=noise, baseline_noise=noise)["actions"]
        sample_ms = (time.perf_counter() - start) * 1000
        np.testing.assert_array_equal(actions[2], prepared_actions)
        prepared_timings.append(dict(prepare_ms=prepare_ms, sample_ms=sample_ms,
                                     total_ms=prepare_ms + sample_ms))
    report = {"format": "r1lite_dsrl_service_probe_v1", "robot_connected": False,
              "config_sha256": sha256(config_path),
              "sources_sha256": {path: sha256(path) for path in [cfg["parquet"], *cfg["videos"].values()]},
              "vla_metadata": vla.metadata, "features_shape": {key: list(value.shape) for key, value in features.items()},
              "features_ms": feature_ms, "vla_request_ms": timings,
              "warm_features_ms": warm_features_ms,
              "prepared_context_repeated_identical": True, "prepared_request_ms": prepared_timings,
              "same_noise_identical": True, "different_noise_max_action_delta": float(np.max(abs(actions[0] - actions[2]))),
              "action_chunks": [value.tolist() for value in actions],
              "status": "vla_passed_reward_pending",
              "scope": "Recorded-observation engineering check; no executed DSRL transitions or success-rate evidence."}
    output = Path(cfg["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with RewardClient(cfg["reward_address"], timeout=cfg["reward_timeout_s"],
                      context_frames=cfg["reward_context_frames"]) as reward:
        start = time.perf_counter()
        progress, success = reward.predict(frames["cam_high"], cfg["reward_indices"], cfg["task"],
                                           cfg["episode_id"])
        reward_ms = (time.perf_counter() - start) * 1000
    with RewardClient(cfg["reward_address"], timeout=cfg["reward_timeout_s"]) as reference:
        start = time.perf_counter()
        full_prefix = reference.predict(frames["cam_high"], cfg["reward_indices"], cfg["task"], cfg["episode_id"])
        full_prefix_ms = (time.perf_counter() - start) * 1000
    np.testing.assert_allclose(np.asarray([progress, success]), full_prefix, rtol=0, atol=1e-6)
    report.update(status="passed", reward_indices=cfg["reward_indices"], progress=progress.tolist(),
                  success=success.tolist(), reward_ms=reward_ms, full_prefix_ms=full_prefix_ms,
                  bounded_context_matches_full_prefix=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    probe(parser.parse_args().config.resolve())


if __name__ == "__main__":
    main()
