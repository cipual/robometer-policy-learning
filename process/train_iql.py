"""Train the official Robometer IQL models on labeled R1 Lite replay without a rollout environment."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

from process.r1lite_policy import CAMERAS, actor_from_metadata, observation_space, observation_tensors
from robometer_policy_learning.algorithms.iql.configuration_iql import IQLConfig
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.modules.transformer import TransformerCritic, TransformerCriticConfig


def load_replay(path, labels, split):
    with h5py.File(path, "r") as replay, h5py.File(labels, "r") as rewards:
        if replay.attrs["split"] != split or rewards.attrs["split"] != split:
            raise ValueError(f"Expected {split} replay and reward labels")
        if replay.attrs["manifest_sha256"] != rewards.attrs["manifest_sha256"] or set(replay["data"]) != set(rewards["data"]):
            raise ValueError("Reward labels disagree with the registered replay episodes")
        data = replay["data"]
        image_shapes = {tuple(episode[f"obs/{camera}"].shape[1:]) for episode in data.values() for camera in CAMERAS}
        if len(image_shapes) != 1:
            raise ValueError("Expected common RGB image shape across replay episodes")
        action, state, scores = [], [], {}
        for key, episode in data.items():
            reward = rewards["data"][key]["rewards"][:]
            if reward.shape != (len(episode["actions"]),) or not np.isfinite(reward).all() or not ((0 <= reward) & (reward <= 1)).all():
                raise ValueError(f"Invalid absolute progress reward for {key}")
            action.append(episode["actions"][:])
            state.append(episode["obs/state"][:])
            scores[key] = reward
        return {
            "actions": np.concatenate(action), "states": np.concatenate(state), "scores": scores,
            "image_shape": list(image_shapes.pop()), "manifest": replay.attrs["manifest_sha256"],
            "reward_bundle": rewards.attrs["reward_bundle_sha256"],
            "reward_method": rewards.attrs.get("method"),
            "task": next(iter(data.values()))["language_instruction"].asstr()[()],
        }


class LabeledReplay(H5ReplayBuffer):
    def __init__(self, replay_path, scores, metadata, cache_images=False):
        self.scores = scores
        self.metadata = metadata
        self.frames = {}
        super().__init__(str(replay_path), obs_keys=["state", *CAMERAS], hdf5_use_swmr=False, num_image_loading_threads=1)
        if set(self.image_keys) != set(CAMERAS):
            raise ValueError(f"Replay camera keys not recognized: {self.image_keys}")
        if cache_images:
            with self._get_hdf5_file(str(replay_path)) as replay:
                for episode_id, episode in replay["data"].items():
                    self.frames[episode_id] = {key: episode["frames"][key][:] for key in CAMERAS}

    def _load_obs_from_file(self, h5_path, demo, obs_key, timestep):
        if self.frames:
            return self.frames[demo][obs_key][timestep]
        return super()._load_obs_from_file(h5_path, demo, obs_key, timestep)

    def _get_rewards_array(self, demo_group):
        return self.scores[demo_group.name.rsplit("/", 1)[-1]]

    def _construct_transitions_from_indices(self, indices):
        transitions = super()._construct_transitions_from_indices(indices)
        for index, transition in zip(indices, transitions, strict=True):
            entry = self._index[index]
            if entry["is_last"] and not transition.done:
                with self._get_hdf5_file(entry["h5_path"]) as replay:
                    group = self._get_demo_group(replay, entry["original_demo_name"])["next_obs"]
                    transition.next_obs = {key: group[key][entry["t"]] for key in transition.obs}
        return transitions

    def batch_from_indices(self, indices, device=None, dtype=None):
        batch = super().batch_from_indices(indices, device, dtype)
        batch["obs"] = observation_tensors(batch["obs"], self.metadata, device)
        batch["next_obs"] = observation_tensors(batch["next_obs"], self.metadata, device)
        action = batch["action"]
        center = torch.as_tensor(self.metadata["action_center"], dtype=action.dtype, device=device)
        scale = torch.as_tensor(self.metadata["action_scale"], dtype=action.dtype, device=device).clamp_min(1e-6)
        batch["action"] = ((action - center) / scale).clamp(-0.999, 0.999).unsqueeze(1)
        return batch


def midpoint_and_scale(array):
    low, high = array.min(axis=0), array.max(axis=0)
    return ((high + low) / 2).tolist(), ((high - low) / 2).tolist()


@torch.no_grad()
def validate(actor, replay, device, batch_size, samples):
    indices = np.linspace(0, len(replay) - 1, min(len(replay), samples), dtype=int)
    total = 0.0
    for start in range(0, len(indices), batch_size):
        batch = replay.batch_from_indices(indices[start:start + batch_size], device=device)
        predicted, _ = actor.act(batch["obs"], deterministic=True)
        total += (predicted - batch["action"]).abs().mean(dim=(1, 2)).sum().item()
    return total / len(indices)


def save_checkpoint(path, state, retained_path=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        torch.save(state, staged)
        if retained_path is not None:
            retained = staged.with_name(retained_path.name)
            retained.hardlink_to(staged)
            retained.replace(retained_path)
        staged.replace(path)


def train(config_path, output, resume, stop_after):
    config_bytes = config_path.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg["device"])
    keep_steps = set(cfg.get("keep_steps", []))
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "highest"))
    training = load_replay(cfg["train_replay"], cfg["train_rewards"], "train")
    validation = load_replay(cfg["val_replay"], cfg["val_rewards"], "val")
    if (set(training["scores"]) & set(validation["scores"]) or any(
            training[key] != validation[key] for key in ("manifest", "image_shape", "reward_bundle", "reward_method", "task"))):
        raise ValueError("Train and validation must be disjoint splits of one replay and reward snapshot")
    metadata = {
        "format": "r1lite_robometer_iql_v1", "actor": cfg["actor"], "image_shape": training["image_shape"],
        "cameras": list(CAMERAS), "task": training["task"], "reward_bundle_sha256": training["reward_bundle"],
        "manifest_sha256": training["manifest"], "action": "absolute action.qpos in original EVA recording units",
        "qpos_order": ["left_arm_1_to_6", "left_gripper", "right_arm_1_to_6", "right_gripper"],
        "execution_horizon": 1, "reference_hz": cfg["reference_hz"],
    }
    if "reward_method" in cfg:
        if cfg["reward_method"] != training["reward_method"]:
            raise ValueError("Replay labels were produced by a different reward method")
        metadata["reward_method"] = training["reward_method"]
        metadata["reward_labels_sha256"] = {
            split: hashlib.sha256(Path(cfg[f"{split}_rewards"]).read_bytes()).hexdigest()
            for split in ("train", "val")
        }
    metadata["state_center"], metadata["state_scale"] = midpoint_and_scale(training["states"])
    metadata["action_center"], metadata["action_scale"] = midpoint_and_scale(training["actions"])
    train_replay = LabeledReplay(cfg["train_replay"], training["scores"], metadata, cfg.get("cache_training_images", False))
    val_replay = LabeledReplay(cfg["val_replay"], validation["scores"], metadata)
    actor = actor_from_metadata(metadata, device)
    networks = {"observation_space": observation_space(metadata), "action_space": actor.action_space,
                "dinov2_model": None, "dinov2_processor": None, "featurizer": {"state": [384]}, **cfg["critic"]}
    critic = TransformerCritic(TransformerCriticConfig(**networks, use_action=True)).to(device)
    value = TransformerCritic(TransformerCriticConfig(**networks, use_action=False)).to(device)
    algo = IQLConfig(actor=actor, critic=critic, v_net=value, buffer=train_replay, **cfg["iql"]).create()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    checkpoint = output / "ckpt/last.pt"
    start, best, run_id = 0, float("inf"), None
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config_sha256"] != config_hash or saved["metadata"] != metadata:
            raise ValueError("IQL checkpoint configuration or input metadata changed")
        for key, state in saved["components"].items():
            getattr(algo, key).load_state_dict(state)
        start, best, run_id = saved["step"], saved["best"], saved["wandb_run_id"]
        algo._n_updates = saved["updates"]
        rng = saved["numpy_rng"]
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    elif checkpoint.exists():
        raise FileExistsError(f"Use --resume to continue {checkpoint}")

    output.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg["wandb"]:
        import wandb

        run = wandb.init(project=cfg["wandb_project"], name=cfg.get("run_name", output.name), id=run_id,
                         resume="must" if resume else None, config=cfg, dir=str(output))
        run_id = run.id
    end = min(cfg["steps"], start + stop_after) if stop_after is not None else cfg["steps"]
    with (output / "metrics.jsonl").open("a", buffering=1) as log:
        for step in trange(start + 1, end + 1, initial=start, total=cfg["steps"], mininterval=10, unit="step"):
            tick = perf_counter()
            metrics = {key: float(value) for key, value in algo.train_step().items()}
            if not metrics or not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"Non-finite or empty IQL metrics at step {step}")
            metrics.update({"step": step, "actor_lr": algo.actor_optimizer.param_groups[0]["lr"],
                            "iter_per_second": 1.0 / (perf_counter() - tick)})
            if step % cfg["save_every"] == 0 or step == end or step in keep_steps:
                loss = validate(actor, val_replay, device, cfg["val_batch_size"], cfg["val_samples"])
                if not math.isfinite(loss):
                    raise ValueError(f"Invalid IQL validation loss at step {step}")
                metrics["val_normalized_action_mae"] = loss
                improved = loss < best
                best = min(loss, best)
                np_rng = np.random.get_state()
                state = {
                    "components": {key: getattr(algo, key).state_dict() for key in algo.component_names},
                    "step": step, "updates": algo._n_updates, "best": best, "metadata": metadata,
                    "config_sha256": config_hash, "wandb_run_id": run_id,
                    "numpy_rng": [np_rng[0], np_rng[1].tolist(), *np_rng[2:]], "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                }
                retained = checkpoint.with_name(f"step-{step}.pt") if step in keep_steps else None
                save_checkpoint(checkpoint, state, retained)
                if improved:
                    save_checkpoint(output / "ckpt/best.pt", {"actor": state["components"]["actor"],
                                                               "metadata": metadata, "step": step, "val_mae": loss})
            log.write(json.dumps(metrics) + "\n")
            if step % cfg.get("log_every", 20) == 0 or "val_normalized_action_mae" in metrics:
                print(json.dumps(metrics), flush=True)
            if run is not None:
                run.log(metrics, step=step)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Save and stop after this many steps in this invocation")
    args = parser.parse_args()
    train(args.config, args.output, args.resume, args.stop_after)
