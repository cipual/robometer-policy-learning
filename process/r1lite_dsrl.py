"""R1 Lite DSRL learner, frozen-VLA noise contract, and asynchronous episode relabeling."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

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


def compose_qpos_actions(result, chunk_size, *, frozen_grippers):
    """Keep the arm proposal and replace only the two R1 Lite gripper commands."""
    branches = {}
    for key in ('actions', 'baseline_actions') if frozen_grippers else ('actions',):
        value = np.asarray(result[key], dtype=np.float32)[:chunk_size]
        if value.shape != (chunk_size, 14) or not np.isfinite(value).all():
            raise ValueError('Frozen VLA must return finite chunks of R1 Lite qpos commands')
        branches['dsrl_actions' if key == 'actions' else key] = value.copy()
    actions = branches['dsrl_actions'].copy()
    if frozen_grippers:
        actions[:, [6, 13]] = branches['baseline_actions'][:, [6, 13]]
    return actions, branches


class R1LiteDSRL:
    def __init__(self, frozen_infer: Callable, reward_client, config, device):
        self.frozen_infer = frozen_infer
        self.reward_client = reward_client
        self.config = config
        self.device = torch.device(device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robometer")
        self.pending = []
        self.buffer = ReplayBuffer(capacity=config["capacity"])
        self.prior_rng = torch.Generator(device='cpu').manual_seed(config['warmup_seed'])
        features = gym.spaces.Dict({key: gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
                                    for key, size in config["features"].items()})
        bound = config["noise_bound"]
        noise = gym.spaces.Box(-bound, bound, shape=(config["noise_dim"],), dtype=np.float32)
        common = {"observation_space": features, "action_space": noise, "chunk_size": config["model_horizon"],
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

    def select_chunk(self, vla_observation, features, deterministic=False, *, warmup=False):
        obs = {}
        for key, size in self.config["features"].items():
            feature = np.asarray(features[key], dtype=np.float32)
            if feature.shape != (size,) or not np.isfinite(feature).all():
                raise ValueError(f"Invalid DSRL observation feature: {key}")
            obs[key] = torch.tensor(feature[None], device=self.device)
        # Exactly the independent CPU Gaussian generator used by Dense base.
        # Never clip/repeat this prior: H50 iid Gaussian is the warmup policy.
        prior = torch.randn((self.config['model_horizon'], self.config['noise_dim']),
                            generator=self.prior_rng).numpy()
        if warmup:
            raw_noise = prior
        else:
            noise, _ = self.algorithm.actor.act(obs, deterministic=deterministic)
            raw_noise = noise[0].cpu().numpy()
        result = self.frozen_infer(vla_observation, noise=raw_noise, baseline_noise=prior)
        actions, proposals = compose_qpos_actions(result, self.config['chunk_size'],
                                                 frozen_grippers=True)
        proposals['baseline_noise'] = prior.copy()
        proposals['noise_source'] = np.asarray('base_gaussian_warmup' if warmup else 'dsrl_actor')
        return actions, raw_noise / self.config["noise_bound"], proposals

    def succeeded(self, frames, task, episode_id):
        if self.config["success_threshold"] is None:
            raise ValueError("Automatic success termination is disabled; use the operator outcome")
        _, success = self.reward_client.predict(frames, [len(frames) - 1], task, episode_id)
        return bool(success[0] > self.config["success_threshold"])

    def submit_episode(self, frames, transitions, task, episode_id):
        if not transitions or not (transitions[-1].done or transitions[-1].truncated):
            raise ValueError("Submit an episode after termination or time limit")
        future = self.executor.submit(self._scores, frames, transitions, task, episode_id)
        self.pending.append((episode_id, transitions, future))
        return future

    def _scores(self, frames, transitions, task, episode_id):
        executed = [index for transition in transitions for index in transition.frame_indices]
        if executed != list(range(executed[0], len(frames))):
            raise ValueError("One post-action RGB frame is required for each executed control step")
        # A DSRL environment step is one executed noise chunk. Query the dense
        # progress model at that step's actual endpoint, retaining its causal
        # context from all intervening control-source frames.
        indices = [transition.frame_indices[-1] for transition in transitions]
        # Bound per-request activation memory while retaining the episode prefix.
        batch_size = self.config["reward_batch_size"]
        parts = []
        for start in tqdm(range(0, len(indices), batch_size), desc=f"Robometer {episode_id[:8]}",
                          mininterval=10):
            batch = indices[start:start + batch_size]
            progress, success = self.reward_client.predict(frames[:batch[-1] + 1], batch, task, episode_id)
            parts.append(np.asarray([progress, success], dtype=np.float32))
        return np.concatenate(parts, axis=1)

    def collect_rewards(self, wait=False):
        count = 0
        while self.pending and (wait or self.pending[0][2].done()):
            episode_id, transitions, future = self.pending[0]
            progress, _ = future.result()
            for step, transition in enumerate(transitions):
                reward = float(progress[step] - self.config["step_penalty"]
                               + self.config["success_bonus"] * transition.success)
                self.buffer.add(transition.observation, transition.noise, reward, transition.next_observation,
                                transition.done, transition.truncated, episode_id=episode_id, step_in_episode=step)
                count += 1
            self.pending.pop(0)
        return count

    def train_after_episode(self, decision_steps):
        self.collect_rewards()
        if self.buffer.size() < self.algorithm.batch_size:
            return {}
        return self.algorithm.train_step(rollout_step=decision_steps) or {}

    def save_checkpoint(self, path, *, environment_steps, episodes, metadata=None):
        """Save learner, replay and RNG together at a completed-episode boundary."""
        self.collect_rewards(wait=True)
        components = {}
        for key in self.algorithm.component_names:
            value = getattr(self.algorithm, key, None)
            if value is None:
                continue
            components[key] = (value.detach().cpu() if isinstance(value, torch.Tensor)
                               else value.state_dict())
        state = {"format": "r1lite_dsrl_checkpoint_v1", "config": self.config,
                 "components": components, "replay": self.buffer.get_all_transitions(),
                 "updates": self.algorithm._n_updates,
                 "environment_steps": environment_steps, "episodes": episodes,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "prior_rng": self.prior_rng.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}
        if metadata is not None:
            state["metadata"] = metadata
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path):
        """Restore an owned local checkpoint into the existing parameter objects.

        Replay contains NumPy arrays and Transition records, so this loader is
        for this project's trusted local outputs, not arbitrary downloaded files.
        """
        if self.pending:
            raise ValueError("Cannot resume while reward jobs are pending")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["format"] != "r1lite_dsrl_checkpoint_v1" or state["config"] != self.config:
            raise ValueError("DSRL checkpoint runtime configuration changed")
        for key, value in state["components"].items():
            component = getattr(self.algorithm, key)
            if isinstance(component, torch.Tensor):
                with torch.no_grad():
                    component.copy_(value)
            else:
                component.load_state_dict(value)
        self.buffer.clear()
        for transition in state["replay"]:
            self.buffer.add(transition.obs, transition.action, transition.reward, transition.next_obs,
                            transition.done, transition.truncated, episode_id=transition.episode_id,
                            step_in_episode=transition.step_in_episode, timestamp=transition.timestamp)
        self.algorithm._n_updates = state["updates"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        self.prior_rng.set_state(state['prior_rng'].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        result = {key: state[key] for key in ("environment_steps", "episodes", "updates")}
        if "metadata" in state:
            result["metadata"] = state["metadata"]
        return result
