"""Exercise the DSRL noise, real reward RPC, replay, and SAC update without robot motion."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from process.r1lite_dsrl import NoiseTransition, R1LiteDSRL
from process.reward_client import RewardClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--address", default="127.0.0.1:50052")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vla-port", type=int, help="Use the local frozen OpenPI service in place of the fixture")
    args = parser.parse_args()
    config = {"features": {"state": 14, "dino_wrist": 384, "vlm_features": 2048}, "capacity": 16,
              "noise_bound": 2.0, "noise_dim": 32, "chunk_size": 8, "model_horizon": 10,
              "success_threshold": 0.6, "reward_batch_size": 2,
              "actor": {"d_model": 32, "nhead": 4, "num_encoder_layers": 1, "feature_hidden_dims": [32]},
              "critic": {"d_model": 32, "nhead": 4, "num_encoder_layers": 1, "feature_hidden_dims": [32],
                         "action_embedding_dim": 32, "pooling_strategy": "first"},
              "sac": {"gamma": 0.995, "learning_starts": 0, "batch_size": 2, "target_entropy": 0,
                      "num_critic_updates_per_actor_update": 1, "train_actor_with_kl_divergence": False}}
    torch.manual_seed(42)

    def frozen_fixture(obs, *, noise):
        assert noise.shape == (10, 32)
        np.testing.assert_array_equal(noise[8:], noise[:2])
        # A fixed local fixture verifies transport and never connects to a robot or a real VLA.
        return {"actions": obs["state"][None] + noise[:, :14] * 0.01}

    vla = None
    if args.vla_port is not None:
        from process.frozen_vla_client import FrozenVLAClient

        vla = FrozenVLAClient("127.0.0.1", args.vla_port)
        config["model_horizon"] = vla.metadata["model_horizon"]
        config["noise_dim"] = vla.metadata["noise_dim"]
        if not vla.metadata["dino_model"]:
            config["features"].pop("dino_wrist")

    with h5py.File(args.replay, "r") as replay, RewardClient(args.address) as client:
        episode = replay["data"][args.episode]
        task = episode["language_instruction"].asstr()[()]
        frames = episode["frames/cam_high"][:17]
        states = episode["frames/state"][:17]
        if len(frames) != 17:
            raise ValueError("The smoke replay must contain at least 16 transitions")
        with R1LiteDSRL(vla.infer if vla is not None else frozen_fixture, client, config, args.device) as agent:
            before = next(agent.algorithm.actor.parameters()).detach().clone()
            transitions = []
            features = {"state": states[0], "dino_wrist": np.zeros(384, dtype=np.float32),
                        "vlm_features": np.zeros(2048, dtype=np.float32)}
            def observation(step, state):
                cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
                return {"images": {key: episode["frames"][key][step] for key in cameras}, "state": state, "prompt": task}

            current = observation(0, states[0])
            if vla is not None:
                features = vla.features(current)
            for start in (0, 8):
                proposed, noise = agent.select_chunk(current, features)
                current = observation(start + 8, proposed[-1])
                successor = vla.features(current) if vla is not None else {**features, "state": proposed[-1]}
                transitions.append(NoiseTransition(features, noise, successor, list(range(start + 1, start + 9)),
                                                   done=False, truncated=start == 8, success=False))
                features = successor
            success = agent.succeeded(frames, task, args.episode)
            agent.submit_episode(frames, transitions, task, args.episode)
            assert agent.collect_rewards(wait=True) == len(transitions)
            metrics = {key: float(value) for key, value in agent.train_after_episode(16).items()}
            if not metrics or not np.isfinite(list(metrics.values())).all():
                raise ValueError("The local SAC update did not return finite losses")
            assert not torch.equal(before, next(agent.algorithm.actor.parameters()))
            args.output.mkdir(parents=True, exist_ok=False)
            torch.save(agent.algorithm.actor.state_dict(), args.output / "noise_actor.pt")
            (args.output / "smoke.json").write_text(json.dumps({"transitions": len(transitions), "metrics": metrics,
                                                                "success_query": success, "config": config,
                                                                "vla": vla.metadata if vla is not None else "fixture"},
                                                               indent=2) + "\n")
            print(f"Verified {len(transitions)} relabeled chunks and one SAC update")


if __name__ == "__main__":
    main()
