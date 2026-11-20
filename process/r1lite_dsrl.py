"""R1 Lite DSRL learner, frozen-VLA noise contract, and asynchronous episode relabeling."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from robometer_policy_learning.algorithms.sac.configuration_sac import SACConfig
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.modules.transformer import (
    TransformerActor, TransformerActorConfig, TransformerCritic, TransformerCriticConfig,
)


@dataclass
class NoiseTransition:
    observation: dict
    noise: np.ndarray
    next_observation: dict
    frame_indices: list[int]
    done: bool
    truncated: bool
    success: bool


def expand_noise(noise, model_horizon):
    """Repeat the learned chunk to the frozen model horizon as in the DSRL baseline."""
    if noise.ndim != 2 or not 0 < len(noise) <= model_horizon or not np.isfinite(noise).all():
        raise ValueError("Expected a finite noise chunk no longer than the VLA horizon")
    return np.tile(noise, (int(np.ceil(model_horizon / len(noise))), 1))[:model_horizon]


class R1LiteDSRL:
    def __init__(self, frozen_infer: Callable, reward_client, config, device):
        self.frozen_infer = frozen_infer
        self.reward_client = reward_client
        self.config = config
        self.device = torch.device(device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robometer")
        self.pending = []
        self.buffer = ReplayBuffer(capacity=config["capacity"])
        features = gym.spaces.Dict({key: gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
                                    for key, size in config["features"].items()})
        bound = config["noise_bound"]
        noise = gym.spaces.Box(-bound, bound, shape=(config["noise_dim"],), dtype=np.float32)
        common = {"observation_space": features, "action_space": noise, "chunk_size": config["chunk_size"],
                  "use_language_embeddings": False, "dinov2_model": None, "dinov2_processor": None,
                  "featurizer": {key: [384] for key in config["features"]}}
        actor = TransformerActor(TransformerActorConfig(**common, **config["actor"])).to(self.device)
        critic = TransformerCritic(TransformerCriticConfig(**common, **config["critic"], use_action=True)).to(self.device)
        self.algorithm = SACConfig(actor=actor, critic=critic, action_space=noise, buffer=self.buffer,
                                   **config["sac"]).create()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.executor.shutdown(wait=True)

    def select_chunk(self, vla_observation, features, deterministic=False):
        obs = {}
        for key, size in self.config["features"].items():
            feature = np.asarray(features[key], dtype=np.float32)
            if feature.shape != (size,) or not np.isfinite(feature).all():
                raise ValueError(f"Invalid DSRL observation feature: {key}")
            obs[key] = torch.tensor(feature[None], device=self.device)
        noise, _ = self.algorithm.actor.act(obs, deterministic=deterministic)
        raw_noise = noise[0].cpu().numpy()
        model_noise = expand_noise(raw_noise, self.config["model_horizon"])
        result = self.frozen_infer(vla_observation, noise=model_noise)
        actions = np.asarray(result["actions"], dtype=np.float32)[:self.config["chunk_size"]]
        if actions.shape != (self.config["chunk_size"], 14) or not np.isfinite(actions).all():
            raise ValueError("Frozen VLA must return a finite chunk of R1 Lite qpos commands")
        return actions, raw_noise / self.config["noise_bound"]

    def succeeded(self, frames, task, episode_id):
        _, success = self.reward_client.predict(frames, [len(frames) - 1], task, episode_id)
        return bool(success[0] > self.config["success_threshold"])

    def submit_episode(self, frames, transitions, task, episode_id):
        if not transitions or not (transitions[-1].done or transitions[-1].truncated):
            raise ValueError("Submit an episode after termination or time limit")
        self.pending.append((episode_id, transitions, self.executor.submit(
            self._scores, frames, transitions, task, episode_id)))

    def _scores(self, frames, transitions, task, episode_id):
        indices = [index for transition in transitions for index in transition.frame_indices]
        if indices != list(range(1, len(frames))):
            raise ValueError("One post-action RGB frame is required for each executed control step")
        # Bound per-request activation memory while retaining the episode prefix.
        batch_size = self.config["reward_batch_size"]
        parts = []
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            progress, _ = self.reward_client.predict(frames[:batch[-1] + 1], batch, task, episode_id)
            parts.append(progress)
        return np.concatenate(parts)

    def collect_rewards(self, wait=False):
        count = 0
        while self.pending and (wait or self.pending[0][2].done()):
            episode_id, transitions, future = self.pending[0]
            progress = future.result()
            offset = 0
            for step, transition in enumerate(transitions):
                length = len(transition.frame_indices)
                rewards = progress[offset:offset + length].copy() - 1.0
                if transition.success:
                    rewards[-1] += 1.0
                reward = float(rewards @ (self.config["sac"]["gamma"] ** np.arange(length)))
                self.buffer.add(transition.observation, transition.noise, reward, transition.next_observation,
                                transition.done, transition.truncated, episode_id=episode_id, step_in_episode=step)
                offset += length
                count += 1
            self.pending.pop(0)
        return count

    def train_after_episode(self, environment_steps):
        self.collect_rewards()
        if self.buffer.size() < self.algorithm.batch_size:
            return {}
        return self.algorithm.train_step(rollout_step=environment_steps) or {}
