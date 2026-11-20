"""Build and run the Robometer Transformer actor with the R1 Lite input contract."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from PIL import Image

from robometer_policy_learning.modules.transformer import TransformerActor, TransformerActorConfig


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
QPOS_DIM = 14


def observation_space(metadata):
    return gym.spaces.Dict({
        "state": gym.spaces.Box(-np.inf, np.inf, shape=(QPOS_DIM,), dtype=np.float32),
        **{f"image_{key}": gym.spaces.Box(0, 255, shape=tuple(metadata["image_shape"]), dtype=np.uint8)
           for key in CAMERAS},
    })


def observation_tensors(obs, metadata, device):
    state = torch.as_tensor(obs["state"], dtype=torch.float32, device=device)
    center = torch.as_tensor(metadata["state_center"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(metadata["state_scale"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return {"state": (state - center) / scale,
            **{f"image_{key}": torch.as_tensor(obs[key], dtype=torch.float32, device=device) for key in CAMERAS}}


def actor_from_metadata(metadata, device):
    actor_config = TransformerActorConfig(
        observation_space=observation_space(metadata),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(QPOS_DIM,), dtype=np.float32),
        dinov2_model=None, dinov2_processor=None,
        featurizer={"state": [384]},
        **metadata["actor"],
    )
    return TransformerActor(actor_config).to(device)


class R1LitePolicy:
    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        self.metadata = json.loads((directory / "policy.json").read_text())
        if self.metadata.get("format") != "r1lite_robometer_iql_v1":
            raise ValueError("Unsupported policy bundle format")
        self.device = torch.device(device)
        self.actor = actor_from_metadata(self.metadata, self.device).eval()
        self.actor.load_state_dict(torch.load(directory / "actor.pt", map_location=self.device, weights_only=True))
        self.center = np.asarray(self.metadata["action_center"], dtype=np.float32)
        self.scale = np.asarray(self.metadata["action_scale"], dtype=np.float32)

    @torch.inference_mode()
    def predict(self, observation, deterministic=True):
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (QPOS_DIM,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite R1 Lite 14-dimensional qpos state")
        obs = {"state": state[None]}
        height, width, _ = self.metadata["image_shape"]
        for key in CAMERAS:
            frame = np.asarray(observation[key])
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"Expected a uint8 RGB frame for {key}")
            if frame.shape != (height, width, 3):
                frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)).copy()
            obs[key] = frame[None]
        action, _ = self.actor.act(observation_tensors(obs, self.metadata, self.device), deterministic=deterministic)
        output = action[0, 0].cpu().numpy() * self.scale + self.center
        if output.shape != (QPOS_DIM,) or not np.isfinite(output).all():
            raise ValueError("The actor returned an invalid qpos command")
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    policy = R1LitePolicy(args.policy, args.device)
    with h5py.File(args.replay, "r") as file:
        obs = file["data"][args.episode]["obs"]
        command = policy.predict({key: obs[key][args.step] for key in ("state", *CAMERAS)})
    print(json.dumps({"action.qpos": command.tolist()}))


if __name__ == "__main__":
    main()
