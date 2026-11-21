"""Learn a DSRL noise policy from explicit EVA execution and operator outcomes.

Input: --config with frozen services, learner, source clocks and output paths.
Output: source-step replay, progress rewards, SAC metrics and checkpoints. Robot
commands are owned by the separately launched embedded EVA client.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import os
import random
from pathlib import Path
import threading
import time
import traceback
import uuid

import cv2
import numpy as np
from tqdm import tqdm


def materialize(observation):
    """Decode original camera pixels; the frozen VLA owns its own resize/norm."""
    images = {}
    transform = observation.get('_reference_image_transform')
    for key, packet in observation['images'].items():
        if isinstance(packet, dict):
            if transform is None or not transform['convert_bgr_to_rgb']:
                raise ValueError('Raw EVA images must declare their BGR-to-RGB conversion')
            if packet['encoding'] == 'compressed':
                bgr = cv2.imdecode(np.frombuffer(packet['data'], np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f'Invalid JPEG source: {key}')
            elif packet['encoding'] == 'array':
                bgr = np.asarray(packet['data'])
            else:
                raise ValueError('Unsupported raw source encoding')
            image = np.ascontiguousarray(bgr[..., ::-1])
        else:
            image = np.asarray(packet)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f'Expected uint8 RGB source: {key}')
        images[key] = image
    state = np.asarray(observation['state'], np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Invalid R1 Lite source state')
    return dict(state=state, images=images, prompt=observation['prompt'])


@dataclass
class Episode:
    episode_id: str
    recording_index: int
    task: str
    policy_version: int = 0
    action_control: dict | None = None
    observations: dict = field(default_factory=dict)
    decision: dict | None = None
    proposals: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)
    transitions: list = field(default_factory=list)
    chunk: dict | None = None
    observation: dict | None = None
    source_time: float | None = None
    image_source_times: dict = field(default_factory=dict)
    pending: dict | None = None
    published: int = 0
    closed: bool = False
    reason: str = ''
    score: dict | None = None
    learned: bool = False
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    io_jobs: deque = field(default_factory=deque)
    directory: Path | None = None


class OnlineDSRL:
    def __init__(self, settings, frozen, reward, *, device):
        from process.r1lite_dsrl import R1LiteDSRL
        self.settings, self.frozen, self.reward = settings, frozen, reward
        self.config_sha256 = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        self.agent = R1LiteDSRL(frozen.infer, reward, settings['learner'], device)
        self.output = Path(settings['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.episodes = {}
        self.active = None
        self.environment_steps = 0
        self.completed = 0
        self.version = 0
        self.lock = threading.RLock()
        self.worker = None
        self.training_error = None
        self.training_pending = False
        self.validation_lock = threading.Lock()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dsrl-recording')

    def _directory(self, episode):
        if episode.directory is None:
            episode.directory = self.output / 'replay' / episode.episode_id
            episode.directory.mkdir(parents=True, exist_ok=True)
        return episode.directory

    def _check_io(self, episode, *, wait=False):
        while episode.io_jobs and (wait or episode.io_jobs[0].done()):
            episode.io_jobs.popleft().result()

    @staticmethod
    def _write_record(path, content):
        temporary = path.with_suffix('.partial')
        temporary.write_text(content)
        temporary.replace(path)

    def _record(self, episode):
        record = dict(episode_id=episode.episode_id, recording_episode_index=episode.recording_index,
                      task=episode.task, published_steps=episode.published, closed=episode.closed,
                      end_reason=episode.reason, score=episode.score, learned=episode.learned,
                      error=episode.error, policy_version=episode.policy_version,
                      action_control=episode.action_control)
        path = self._directory(episode) / 'episode.json'
        episode.io_jobs.append(self.writer.submit(self._write_record, path,
                               json.dumps(record, ensure_ascii=False, indent=2) + '\n'))

    def _episode(self, message):
        ep = self.active
        if ep is None or ep.episode_id != message['episode_id'] or ep.closed:
            raise ValueError('No matching active DSRL episode')
        return ep

    def _observe(self, ep, raw):
        self._check_io(ep)
        t = float(raw['_residual_capture_time'])
        if not np.isfinite(t) or (ep.source_time is not None and t <= ep.source_time):
            raise ValueError('DSRL source time must strictly advance')
        obs = materialize(raw)
        if obs['prompt'] != ep.task:
            raise ValueError('Task changed within an episode')
        if ep.published and t - ep.source_time > self.settings['max_source_gap_s']:
            raise ValueError('Discontinuous DSRL source observations')
        if ep.pending is not None:
            ep.frames.append(obs['images']['cam_high'].copy())
            pending = ep.pending
            ep.io_jobs.append(self.writer.submit(np.savez_compressed,
                                self._directory(ep) / f'step-{ep.published - 1:06d}.npz',
                                action=pending['action'], state=pending['state'], next_state=obs['state'],
                                source_time=pending['source_time'], next_source_time=t,
                                publication_completed_at=pending['publication_completed_at'],
                                camera_keys=np.asarray(list(raw['_image_source_times'])),
                                image_source_times=np.asarray([pending['image_source_times'][key]
                                                               for key in raw['_image_source_times']]),
                                next_image_source_times=np.asarray(list(raw['_image_source_times'].values())),
                                post_frame=obs['images']['cam_high'], chunk_id=pending['chunk_id'],
                                chunk_row=pending['row']))
            ep.pending = None
        elif not ep.published:
            # Initial VLA preparation happens while holding. Replace the
            # unpublished endpoint with the fresh source of the first command.
            ep.frames[:] = [obs['images']['cam_high'].copy()]
            ep.io_jobs.append(self.writer.submit(np.savez_compressed,
                                                self._directory(ep) / 'initial.npz', frame=ep.frames[0]))
        ep.observation, ep.source_time = obs, t
        ep.image_source_times = dict(raw['_image_source_times'])
        ep.observations[ep.published] = raw
        ep.observations = {step: value for step, value in ep.observations.items() if step >= ep.published-5}

    def _close_decision(self, ep, next_features, endpoint, *, terminal=False):
        from process.r1lite_dsrl import NoiseTransition
        decision = ep.decision
        if decision is not None:
            edges = list(range(decision['source_step']+1, endpoint+1))
            # Startup 0->3 precedes the regular 3,11,19... decision clock.
            # Preserve it in raw facts; never assign an 8-step discount to it.
            if edges and (len(edges) == self.settings['learner']['chunk_size'] or terminal):
                ep.transitions.append(NoiseTransition(decision['features'], decision['noise'], next_features,
                                                      edges, False, False, False))

    def _prefetch(self, message):
        with self.lock:
            ep = self._episode(message)
            step = int(message['source_step'])
            raw = ep.observations[step]
            if float(raw['_residual_capture_time']) != message['source_time']:
                raise ValueError('RTC generation source was not observed on this episode')
            prefix = np.asarray(message['prefix'], np.float32).reshape(-1, 14)
            if step == 0:
                if len(prefix) or ep.decision is not None:
                    raise ValueError('RTC initial generation must have prefix0')
            else:
                if step % 8 != 3 or len(prefix) != 5:
                    raise ValueError('RTC prefetch requires the five committed commands at H8 minus5')
                np.testing.assert_array_equal(prefix, ep.chunk['actions'][3:8])
        # Slow RPC must never hold the observation/publication lock. Control
        # continues through the already committed prefix while this runs.
        features, context_id = self.frozen.prepare(raw, prefix=prefix)
        warmup = self.agent.buffer.size() < self.settings['learner']['sac']['learning_starts']
        actions, noise, proposals = self.agent.select_chunk(
            {'dsrl_context_id': context_id}, features, warmup=warmup)
        with self.lock:
            if ep.closed or self.active is not ep:
                raise ValueError('RTC proposal completed after the episode was closed')
            self._close_decision(ep, features, step)
            token = uuid.uuid4().hex
            ep.decision = dict(source_step=step, features=features, noise=noise)
            ep.proposals[token] = dict(actions=actions, source_step=step, prefix=prefix,
                                      source_time=message['source_time'], start_step=step+len(prefix))
            ep.io_jobs.append(self.writer.submit(np.savez_compressed,
                self._directory(ep)/f'chunk-{token}.npz', actions=actions, noise=noise,
                source_step=step, execution_start_step=step+len(prefix), prefix=prefix,
                source_time=message['source_time'],
                **{f'feature_{key}': value for key, value in features.items()}, **proposals))
            return dict(ok=True, proposal_id=token, source_step=step, actions=actions)

    def request(self, message):
        if message.get('kind') == 'prefetch':
            return self._prefetch(message)
        if message.get('kind') == 'validate':
            # Setup inference is discarded by EVA and owns no episode, replay,
            # actor sample or control credit. It may run while the idle learner
            # updates; it only calls the independently frozen service.
            if self.active is not None:
                raise ValueError('Cannot run setup validation during an active trajectory')
            with self.validation_lock:
                obs = materialize(message['observation'])
                features, context_id = self.frozen.prepare(message['observation'], prefix=np.empty((0,14),np.float32))
                for key, size in self.settings['learner']['features'].items():
                    if np.asarray(features[key]).shape != (size,):
                        raise ValueError(f'Unexpected frozen feature shape: {key}')
                cfg = self.settings['learner']
                import torch
                generator = torch.Generator(device='cpu').manual_seed(cfg['warmup_seed'])
                noise = torch.randn((cfg['model_horizon'], cfg['noise_dim']), generator=generator).numpy()
                from process.r1lite_dsrl import compose_qpos_actions
                result = self.frozen.infer({'dsrl_context_id': context_id}, noise=noise, baseline_noise=noise)
                actions, _ = compose_qpos_actions(result, cfg['chunk_size'],
                                                  frozen_grippers=True)
                if actions.shape != (cfg['chunk_size'], 14) or not np.isfinite(actions).all():
                    raise ValueError('Invalid frozen setup output')
                return dict(ok=True, actions=actions, action_mode='qpos', validation_only=True)
        # Updating an idle learner must not block the robot client's command
        # loop behind a whole episode of gradient steps.
        if self.training_pending:
            kind = message.get('kind')
            if kind in ('score', 'score_check'):
                ep = self.episodes[int(message['recording_episode_index'])]
                fields = message['fields']
                if any(fields.get(key) != ep.score.get(key) for key in ('score', 'max_score')):
                    raise ValueError('Outcome is already being used for training')
                return dict(ok=True, processing=True)
            raise ValueError('Wait for the previous trajectory reward/update before starting')
        with self.lock:
            if self.training_error:
                raise RuntimeError(f'Learner stopped: {self.training_error}')
            kind = message.get('kind')
            if kind == 'begin':
                if self.active is not None:
                    raise ValueError('Finish the active DSRL episode first')
                if self.training_pending:
                    raise ValueError('Wait for the previous trajectory reward/update before starting')
                if len(self.episodes) >= self.settings['trajectories']:
                    raise ValueError('The approved trajectory budget is exhausted')
                index = int(message['recording_episode_index'])
                if index in self.episodes:
                    raise ValueError('Native recording episode index was already used')
                if message['task'] != self.settings['task']:
                    raise ValueError('Unexpected PackTask instruction')
                ep = Episode(uuid.uuid4().hex, index, message['task'], self.version,
                             dict(self.settings['action_control']))
                self.episodes[index] = self.active = ep
                self._record(ep)
                return dict(ok=True, episode_id=ep.episode_id, policy_version=self.version)
            if kind in ('score', 'score_check'):
                ep = self.episodes[int(message['recording_episode_index'])]
                fields = message['fields']
                score, maximum = int(fields['score']), int(fields['max_score'])
                if maximum != self.settings['outcome_max_score'] or not 0 <= score <= maximum:
                    raise ValueError('Outcome score disagrees with the configured protocol')
                target_changed = (ep.score is not None and
                                  (score, maximum) != (ep.score['score'], ep.score['max_score']))
                if target_changed and (ep.learned or self.training_pending):
                    raise ValueError('Cannot silently change a label already used for training')
                if kind == 'score_check':
                    return dict(ok=True)
                ep.score = fields
                self._record(ep)
                if ep.error or not ep.transitions:
                    return dict(ok=True, training_excluded=ep.error or 'No valid executed transitions')
                self._schedule(ep)
                return dict(ok=True)
            ep = self._episode(message)
            try:
                if kind == 'abort':
                    ep.closed, ep.reason, ep.error = True, 'unknown', message['error']
                    self.active = None
                    self._record(ep)
                    return dict(ok=True, training_excluded=True)
                if kind == 'observe':
                    self._observe(ep, message['observation'])
                    return dict(ok=True)
                if kind == 'activate':
                    proposal = ep.proposals.pop(message['proposal_id'])
                    if ep.published != proposal['start_step']:
                        raise ValueError('RTC proposal is not aligned with the executed source clock')
                    if ep.chunk is not None and ep.chunk['published'] != 8:
                        raise ValueError('Cannot replace an unacknowledged H8')
                    ep.chunk = dict(chunk_id=message['proposal_id'], actions=proposal['actions'], published=0)
                    return dict(ok=True, actions=proposal['actions'], chunk_id=message['proposal_id'])
                if kind == 'published':
                    chunk = ep.chunk
                    row = int(message['row'])
                    if chunk is None or message['chunk_id'] != chunk['chunk_id'] or row != chunk['published']:
                        raise ValueError('Publication acknowledgement does not match the proposal')
                    if ep.pending is not None or message['source_time'] != ep.source_time:
                        raise ValueError('Publication acknowledgement source is invalid')
                    action = np.asarray(message['action'], np.float32)
                    expected = chunk['actions'][row].copy()
                    for index in (6, 13):
                        expected[index] = 100 if expected[index] >= self.settings['gripper_threshold'] else 0
                    np.testing.assert_allclose(action, expected, rtol=0, atol=1e-6)
                    ep.pending = dict(action=action.copy(), state=ep.observation['state'].copy(),
                                      image_source_times=dict(ep.image_source_times),
                                      source_time=ep.source_time, chunk_id=chunk['chunk_id'], row=row,
                                      publication_completed_at=float(message['publication_completed_at']))
                    chunk['published'] += 1
                    ep.published += 1
                    self.environment_steps += 1
                    self._record(ep)
                    return dict(ok=True, stop_requested=(ep.published >= self.settings['max_episode_steps']
                                or time.monotonic() - ep.started_at >= self.settings['max_episode_seconds']))
                if kind == 'end':
                    if message['observation'] is None:
                        raise ValueError('Missing final post-action observation; preserve as unknown')
                    self._observe(ep, message['observation'])
                    features = self.frozen.features(message['observation'], prefix=message.get('prefix'))
                    self._close_decision(ep, features, ep.published, terminal=True)
                    self._check_io(ep, wait=True)
                    ep.closed, ep.reason = True, message['reason']
                    self._save_transitions(ep)
                    self.active = None
                    self._record(ep)
                    self._schedule(ep)
                    return dict(ok=True)
                raise ValueError(f'Unsupported DSRL request: {kind}')
            except Exception as error:
                ep.error = str(error)
                if kind == 'end':
                    ep.closed, ep.reason = True, 'unknown'
                    self.active = None
                self._record(ep)
                raise

    def _schedule(self, ep):
        if not ep.closed or ep.score is None or ep.learned or self.training_pending:
            return
        if ep.error or not ep.transitions:
            raise ValueError('An invalid or empty source trajectory cannot train the learner')
        success = ep.score['score'] == ep.score['max_score']
        truncated = ep.reason == 'time_limit' and not success
        # A partial time-limit chunk has a shorter discount than SAC's fixed K.
        # Preserve its raw facts, but exclude that incomplete transition from SAC.
        if truncated and len(ep.transitions[-1].frame_indices) != self.settings['learner']['chunk_size']:
            ep.transitions.pop()
        if not ep.transitions:
            raise ValueError('No complete transitions before the time limit')
        last = ep.transitions[-1]
        last.done, last.truncated, last.success = not truncated, truncated, success
        frames = ep.frames[:last.frame_indices[-1] + 1]
        self.training_pending = True
        self.worker = threading.Thread(target=self._learn, args=(ep, frames), daemon=False)
        self.worker.start()

    def _save_transitions(self, ep):
        import torch
        path = self._directory(ep) / 'transitions.pt'
        temporary = path.with_suffix('.partial')
        torch.save(ep.transitions, temporary)
        temporary.replace(path)

    def _wait_recording(self, ep):
        deadline = time.monotonic() + self.settings['save_timeout_s']
        manifest = Path(self.settings['recording_manifest'])
        while time.monotonic() < deadline:
            try:
                rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            except (FileNotFoundError, json.JSONDecodeError):
                rows = []
            row = next((row for row in rows if row.get('episode_index') == ep.recording_index), None)
            if row is not None:
                if row.get('policy_episode_id') != ep.episode_id:
                    raise ValueError('Native recording and learner episode identity disagree')
                if row.get('save_status') == 'failed':
                    raise RuntimeError('Native trajectory saving failed; no training credit')
                # Native RL appends its row only after parquet/videos finish;
                # unlike EVAL, it does not write queued/saved placeholder rows.
                if int(row.get('length', 0)) > 0:
                    if any(row.get(key) != ep.score.get(key) for key in ('score', 'max_score')):
                        raise ValueError('Native recording and learner outcome disagree')
                    return
            time.sleep(0.1)
        raise TimeoutError('Native recording did not finish before reward/learning')

    def _learn(self, ep, frames):
        try:
            self._wait_recording(ep)
            self._check_io(ep, wait=True)
            self._save_transitions(ep)
            labels = self.agent.submit_episode(frames, ep.transitions, ep.task, ep.episode_id)
            self.agent.collect_rewards(wait=True)
            progress, success = labels.result()
            rewards = progress - self.settings['learner']['step_penalty'] + np.asarray(
                [item.success for item in ep.transitions]) * self.settings['learner']['success_bonus']
            np.savez_compressed(self._directory(ep) / 'rewards.npz', progress=progress,
                                success_prediction=success, reward=rewards,
                                endpoint_indices=[item.frame_indices[-1] for item in ep.transitions])
            outcome = dict(episode_id=ep.episode_id, recording_episode_index=ep.recording_index,
                           published_steps=ep.published, rewarded_chunks=len(ep.transitions),
                           progress_mean=float(progress.mean()), success_prediction_mean=float(success.mean()),
                           reward_mean=float(rewards.mean()), operator_success=bool(ep.transitions[-1].success))
            print(json.dumps(outcome), flush=True)
            if self.settings.get('wandb_run'):
                self.settings['wandb_run'].log({f'episode/{key}': value for key, value in outcome.items()
                                                 if key != 'episode_id'})
            updates = len(ep.transitions) * self.settings['updates_per_transition']
            metrics_path = self.output / 'metrics.jsonl'
            with self.lock:
                for _ in tqdm(range(updates), desc=f'SAC episode {ep.recording_index}', mininterval=10):
                    metrics = self.agent.train_after_episode(self.agent.buffer.size())
                    if metrics:
                        values = {key: float(value) for key, value in metrics.items()}
                        if not np.isfinite(list(values.values())).all():
                            raise ValueError('Non-finite online SAC metrics')
                        row = dict(episode_id=ep.episode_id, environment_steps=self.environment_steps,
                                   updates=self.agent.algorithm._n_updates,
                                   actor_lr=self.agent.algorithm.actor_optimizer.param_groups[0]['lr'],
                                   critic_lr=self.agent.algorithm.critic_optimizer.param_groups[0]['lr'], **values)
                        with metrics_path.open('a') as stream:
                            stream.write(json.dumps(row) + '\n')
                        if row['updates'] % self.settings['log_every'] == 0:
                            print(json.dumps(row), flush=True)
                        if self.settings.get('wandb_run'):
                            self.settings['wandb_run'].log({f'train/{key}': value for key, value in row.items()
                                                              if key != 'episode_id'})
                self.completed += 1
                self.version += 1
                ep.learned = True
                self.agent.save_checkpoint(self.output / 'ckpt' / f'episode-{self.completed:04d}.pt',
                                           environment_steps=self.environment_steps, episodes=self.completed,
                                           metadata=dict(config_sha256=self.config_sha256,
                                               learned_episode_ids=[item.episode_id for item in self.episodes.values()
                                                                    if item.learned],
                                               policy_version=self.version,
                                               wandb_run_id=self.settings['wandb_run'].id
                                                   if self.settings.get('wandb_run') else None))
                self._record(ep)
                summary = dict(collected_trajectories=len(self.episodes), learned_trajectories=self.completed,
                               trajectory_budget=self.settings['trajectories'], environment_steps=self.environment_steps,
                               replay_chunks=self.agent.buffer.size(), actor_updates=self.agent.algorithm._n_updates,
                               latest_checkpoint=str(self.output / 'ckpt' / f'episode-{self.completed:04d}.pt'),
                               complete=self.completed == self.settings['trajectories'])
                (self.output / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                ep.frames.clear()
        except Exception as error:
            traceback.print_exc()
            self.training_error = str(error)
            ep.error = str(error)
            self._record(ep)
        finally:
            self.training_pending = False

    def close(self):
        if self.worker:
            self.worker.join()
        self.agent.__exit__(None, None, None)
        self.writer.shutdown(wait=True)

    def disconnect(self, episode_id):
        with self.lock:
            if self.active is not None and self.active.episode_id == episode_id:
                self.active.closed, self.active.reason = True, 'unknown'
                self.active.error = 'Control client disconnected before a clean trajectory end'
                self._record(self.active)
                self.active = None

    def resume(self, checkpoint):
        import torch
        counters = self.agent.load_checkpoint(checkpoint)
        metadata = counters['metadata']
        if metadata['config_sha256'] != self.config_sha256:
            raise ValueError('Online DSRL configuration changed; cannot continue the same run')
        learned = set(metadata['learned_episode_ids'])
        self.completed, self.version = counters['episodes'], metadata['policy_version']
        for path in sorted((self.output / 'replay').glob('*/episode.json')):
            row = json.loads(path.read_text())
            ep = Episode(row['episode_id'], row['recording_episode_index'], row['task'])
            ep.policy_version = row['policy_version']
            ep.action_control = row.get('action_control')
            ep.published, ep.closed, ep.reason = row['published_steps'], row['closed'], row['end_reason']
            ep.score, ep.error, ep.learned = row['score'], row['error'], ep.episode_id in learned
            self.episodes[ep.recording_index] = ep
            if not ep.closed:
                ep.closed, ep.reason = True, 'unknown'
                ep.error = 'Interrupted before the final post-action observation'
            if not ep.learned and ep.closed and not ep.error:
                transitions = path.parent / 'transitions.pt'
                if transitions.exists():
                    ep.transitions = torch.load(transitions, map_location='cpu', weights_only=False)
                    if ep.transitions:
                        with np.load(path.parent / 'initial.npz', allow_pickle=False) as data:
                            ep.frames = [data['frame'].copy()]
                        for index in range(ep.transitions[-1].frame_indices[-1]):
                            with np.load(path.parent / f'step-{index:06d}.npz', allow_pickle=False) as data:
                                ep.frames.append(data['post_frame'].copy())
            self._record(ep)
        self.environment_steps = sum(ep.published for ep in self.episodes.values())
        return metadata['wandb_run_id']

    def resume_pending(self):
        for ep in self.episodes.values():
            if ep.closed and ep.score is not None and not ep.learned and not ep.error:
                self._schedule(ep)
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    from process.runtime_checks import load_settings
    cfg = load_settings(args.config)
    if cfg['format'] != 'r1lite_dsrl_online_v1':
        raise ValueError('Expected an R1 Lite DSRL online configuration')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['physical_gpu'])
    from process.frozen_vla_client import FrozenVLAClient
    from process.reward_client import RewardClient
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve
    import torch
    import wandb

    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(cfg['torch_num_threads'])
    frozen = FrozenVLAClient(cfg['vla_host'], cfg['vla_port'])
    if not frozen.metadata['paired_prepared'] or frozen.metadata['rtc_delay'] != 5:
        raise ValueError('Frozen grippers require paired decoding against one prepared observation')
    if (frozen.metadata['model_horizon'], frozen.metadata['noise_dim']) != (
            cfg['learner']['model_horizon'], cfg['learner']['noise_dim']):
        raise ValueError('Learner and frozen VLA noise contracts disagree')
    with RewardClient(cfg['reward_address'], timeout=cfg['reward_timeout_s'],
                      context_frames=cfg['reward_context_frames']) as reward:
        learner = OnlineDSRL(cfg, frozen, reward, device='cuda:0')
        # Warm inference and sampling kernels without advancing collection RNG.
        with torch.random.fork_rng(devices=[0]):
            learner.agent.algorithm.actor.act({key: torch.zeros((1, size), device='cuda:0')
                                              for key, size in cfg['learner']['features'].items()},
                                             deterministic=False)
            torch.cuda.synchronize()
        run_id = learner.resume(args.resume) if args.resume else None
        run = wandb.init(project=cfg['wandb_project'], name=cfg['run_name'], group=cfg['run_group'],
                         config=cfg, dir=cfg['output'], mode=cfg['wandb_mode'],
                         id=run_id, resume='must' if run_id else None)
        run.define_metric('train/updates')
        run.define_metric('train/*', step_metric='train/updates')
        run.define_metric('episode/recording_episode_index')
        run.define_metric('episode/*', step_metric='episode/recording_episode_index')
        cfg['wandb_run'] = run
        learner.resume_pending()
        metadata = dict(server_name='robometer-dsrl-online', action_mode='qpos',
                        chunk_size=cfg['learner']['chunk_size'], action_dim=14, supports_reset=False,
                        control_hz=cfg['control_hz'], rtc_delay=5, model_horizon=50,
                        max_source_gap_s=cfg['max_source_gap_s'],
                        actor_deadline_s=cfg['rtc_deadline_s'])
        def handler(connection):
            packer = msgpack_numpy.Packer()
            connection.send(packer.pack(metadata))
            owned_episode = None
            try:
                for message in connection:
                    try:
                        request = msgpack_numpy.unpackb(message)
                        result = learner.request(request)
                        if request['kind'] == 'begin':
                            owned_episode = result['episode_id']
                        elif request['kind'] == 'end':
                            owned_episode = None
                    except Exception as error:
                        result = dict(ok=False, error=str(error))
                    connection.send(packer.pack(result))
            finally:
                learner.disconnect(owned_episode)
        try:
            with serve(handler, '127.0.0.1', cfg['serving_port'], compression=None, max_size=None) as server:
                server.serve_forever()
        finally:
            learner.close()
            run.finish()


if __name__ == '__main__':
    main()
