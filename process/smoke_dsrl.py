"""Check DSRL noise, reward backfill, SAC and exact resume without robot motion.

Input: --config with recorded RGB/qpos, explicit fixture/RPC choices and a tiny
learner budget. Output: an engineering report and an owned checkpoint; proposed
noise actions are never executed and this replay is not online training data.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from process.r1lite_dsrl import NoiseTransition, R1LiteDSRL
from process.reward_client import RewardClient


class FixtureReward:
    def predict(self, frames, indices, task, episode_id):
        return np.asarray(indices, np.float32) / max(len(frames), 1), np.zeros(len(indices), np.float32)


def smoke(config_path):
    cfg = json.loads(config_path.read_text())
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['torch_num_threads'])
    learner = cfg['learner']
    steps = 2 * learner['chunk_size']
    rows = pq.read_table(cfg['parquet']).to_pylist()[:steps + 1]
    if len(rows) != steps + 1:
        raise ValueError('The engineering replay needs two complete chunks')
    frames = {}
    for camera, path in cfg['videos'].items():
        capture = cv2.VideoCapture(path)
        try:
            values = []
            for _ in rows:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Incomplete recorded video: {camera}')
                values.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames[camera] = np.asarray(values)
        finally:
            capture.release()

    def observation(step):
        return {'state': np.asarray(rows[step]['observations.state.qpos'], np.float32),
                'images': {key: value[step] for key, value in frames.items()}, 'prompt': cfg['task']}

    def fixture(obs, *, noise, baseline_noise):
        if noise.shape != (learner['model_horizon'], learner['noise_dim']):
            raise ValueError('Fixture received the wrong diffusion noise shape')
        return {'actions': obs['state'][None] + noise[:8, :14] * 0.01,
                'baseline_actions': obs['state'][None] + baseline_noise[:8, :14] * 0.01}

    vla = None
    if cfg['vla_port'] is not None:
        from process.frozen_vla_client import FrozenVLAClient
        vla = FrozenVLAClient(cfg['vla_host'], cfg['vla_port'])
        if (learner['model_horizon'], learner['noise_dim']) != (vla.metadata['model_horizon'], vla.metadata['noise_dim']):
            raise ValueError('Smoke configuration disagrees with frozen VLA')

    def features(step):
        obs = observation(step)
        if vla is not None:
            return vla.features(obs)
        return {key: obs['state'] if key == 'state' else np.zeros(size, np.float32)
                for key, size in learner['features'].items()}

    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        reward = (stack.enter_context(RewardClient(cfg['reward_address'], timeout=cfg['reward_timeout_s'],
                                                  context_frames=cfg['reward_context_frames']))
                  if cfg['reward_address'] is not None else FixtureReward())
        agent = stack.enter_context(R1LiteDSRL(vla.infer if vla else fixture, reward, learner, cfg['device']))
        transitions = []
        for start in range(0, steps, learner['chunk_size']):
            stop = start + learner['chunk_size']
            if vla is not None:
                current_features, context_id = vla.prepare(observation(start))
                vla_input = {'dsrl_context_id': context_id}
            else:
                current_features, vla_input = features(start), observation(start)
            actions, noise, proposals = agent.select_chunk(vla_input, current_features, warmup=True)
            np.testing.assert_array_equal(actions, proposals['baseline_actions'])
            transitions.append(NoiseTransition(current_features, noise, features(stop),
                                               list(range(start + 1, stop + 1)), False, stop == steps, False))
        agent.submit_episode(frames['cam_high'], transitions, cfg['task'], cfg['episode_id'])
        if agent.collect_rewards(wait=True) != 2:
            raise ValueError('Reward backfill did not produce two chunks')
        times = []
        all_metrics = []
        for _ in range(2):
            started = time.perf_counter()
            metrics = {key: float(value) for key, value in agent.train_after_episode(steps).items()}
            if not metrics or not np.isfinite(list(metrics.values())).all():
                raise ValueError('Non-finite or missing SAC metrics')
            times.append((time.perf_counter() - started) * 1000)
            all_metrics.append(metrics)
            if len(times) == 1:
                agent.save_checkpoint(output / 'ckpt/engineering-resume.pt', environment_steps=steps, episodes=1)
        expected = {key: value.detach().clone() for key, value in agent.algorithm.actor.state_dict().items()}
        restored = stack.enter_context(R1LiteDSRL(vla.infer if vla else fixture, reward, learner, cfg['device']))
        counters = restored.load_checkpoint(output / 'ckpt/engineering-resume.pt')
        resumed_metrics = {key: float(value) for key, value in restored.train_after_episode(steps).items()}
        for key, value in restored.algorithm.actor.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=cfg['resume_atol'])
        loss_keys = [key for key in all_metrics[-1] if 'loss' in key]
        for key in loss_keys:
            np.testing.assert_allclose(resumed_metrics[key], all_metrics[-1][key], rtol=0, atol=cfg['resume_atol'])
        report = {'format': 'r1lite_dsrl_smoke_v1', 'robot_connected': False, 'online_training_data': False,
                  'vla_backend': vla.metadata if vla else 'fixture',
                  'action_control': cfg['action_control'],
                  'reward_backend': cfg['reward_address'] or 'fixture', 'transitions': 2,
                  'updates_before_save': counters['updates'], 'restored_counters': counters,
                  'resume_actor_and_loss_match': True, 'update_ms': times, 'metrics': all_metrics,
                  'scope': 'Two tiny SAC updates plus replay of the second update after resume. Recorded next states are engineering fixtures, not outcomes of the proposed DSRL actions.'}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(output / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    smoke(parser.parse_args().config.resolve())


if __name__ == '__main__':
    main()
