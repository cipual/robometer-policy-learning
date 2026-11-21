"""DSRL RPC and generic Dense source-step RTC execution in the native EVA client."""
import threading
import time
import numpy as np

from core.registry import POLICY_REGISTRY
from policy_client.base import PolicyRequestError
from policy_client.openpi import OpenPiPolicyClient
from policy_client._dense_execution import DenseExecution
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


@POLICY_REGISTRY.register_client('dsrl_online')
class DSRLPolicyClient(OpenPiPolicyClient):
    requires_execution_feedback = True
    requires_source_rtc = True
    needs_source_time = True
    needs_raw_images = True

    def __init__(self, host, port, **kwargs):
        super().__init__(host, port, **kwargs)
        self._rpc_lock = threading.RLock()
        self._prefetch_client = WebsocketClientPolicy(host, port)
        self.episode_id = None
        self.stop_requested = False
        self.execution_error = None
        self._observation_provider = None
        self._chunk_id = None
        self._row = 0
        self.observation_preprocessor = None
        self.execution = DenseExecution(self._generate, self._activate, horizon=8, delay=5,
            max_interval=self.metadata['max_source_gap_s'], actor_deadline=self.metadata['actor_deadline_s'])

    def bind_observation_provider(self, provider):
        self._observation_provider = provider

    def _raw(self, observation):
        value = self.observation_preprocessor(observation) if self.observation_preprocessor else observation
        value = dict(value)
        if '_residual_image_source_times' in value:
            value['_image_source_times'] = value['_residual_image_source_times']
        return value

    def _request(self, message):
        try:
            with self._rpc_lock:
                self._ws.send(self._packer.pack(message))
                raw = self._ws.recv()
                if isinstance(raw, str):
                    raise ValueError(raw)
                reply = msgpack_numpy.unpackb(raw)
                if not reply.get('ok', True):
                    raise ValueError(reply['error'])
                return reply
        except Exception as error:
            raise PolicyRequestError(str(error)) from error

    def _generate(self, frame, prefix):
        # A separate socket prevents slow VLA generation from blocking factual
        # observe/publish acknowledgements on the control connection.
        reply = self._prefetch_client.infer(dict(kind='prefetch', episode_id=self.episode_id,
            source_step=frame['frame_index'], source_time=frame['capture_time'], prefix=prefix))
        if not reply.get('ok', True):
            raise PolicyRequestError(reply['error'])
        return reply

    def _activate(self, frame, history, bundle, *, deadline):
        reply = self._request(dict(kind='activate', episode_id=self.episode_id,
                                   proposal_id=bundle['proposal_id']))
        self._chunk_id, self._row = reply['chunk_id'], 0
        return reply

    def begin_episode(self, task, recording_episode_index):
        reply = self._request(dict(kind='begin', task=task, recording_episode_index=recording_episode_index))
        self.episode_id = reply['episode_id']
        self.recording_episode_index = recording_episode_index
        self.stop_requested = False
        self.execution_error = None
        self.execution.reset()
        return dict(policy_episode_id=self.episode_id, policy_version=reply['policy_version'],
                    policy_sample_contract='rtc_generation_clock_prefix5_h8_v1')

    def observe(self, observation):
        raw = self._raw(observation)
        try:
            self._request(dict(kind='observe', episode_id=self.episode_id, observation=raw))
            self.execution.observe(raw, raw['_residual_capture_time'])
        except Exception as error:
            self.execution_error = str(error)
            raise

    def prepare_initial(self, observation):
        """Generate while holding, then apply on the next fresh source, as Dense base does."""
        raw = self._raw(observation)
        self._request(dict(kind='observe', episode_id=self.episode_id, observation=raw))
        return self.execution.prepare_initial(raw, raw['_residual_capture_time'])

    def infer(self, observation, prev_action=None, rtc_params=None):
        if self.episode_id is not None:
            raise PolicyRequestError('RTC owns active generation; do not issue synchronous chunks')
        return self._request(dict(kind='validate', observation=self._raw(observation)))

    def record_execution(self, action, observation):
        reply = self._request(dict(kind='published', episode_id=self.episode_id,
            chunk_id=self._chunk_id, row=self._row, source_time=observation['_residual_capture_time'],
            publication_completed_at=time.time(), action=np.asarray(action, np.float32)))
        self._row += 1
        self.stop_requested = bool(reply['stop_requested'])

    def end_episode(self):
        if self.episode_id is None:
            return
        # A running prefetch must drain before the terminal state is sampled.
        try:
            if self.execution_error:
                self._request(dict(kind='abort', episode_id=self.episode_id, error=self.execution_error))
                return
            if self.execution.future is not None:
                self.execution.future.result(timeout=self.metadata['actor_deadline_s'])
            time.sleep(1.0 / self.metadata['control_hz'])
            observation = self._observation_provider() if self._observation_provider else None
            remaining = self.execution.plan
            prefix = np.stack(remaining) if len(remaining) == 5 else np.empty((0,14), np.float32)
            self._request(dict(kind='end', episode_id=self.episode_id,
                observation=self._raw(observation) if observation is not None else None,
                prefix=prefix, reason='time_limit' if self.stop_requested else 'operator_stop'))
        except Exception as error:
            self._request(dict(kind='abort', episode_id=self.episode_id, error=str(error)))
            raise
        finally:
            self.episode_id = None
            self.execution.reset()

    def freeze_execution(self):
        self.execution_error = 'RTC source or command alignment failed; exclude the interrupted trajectory'
        self.execution.reset()

    def submit_episode_score(self, recording_episode_index, fields):
        return self._request(dict(kind='score', recording_episode_index=recording_episode_index, fields=fields))

    def validate_episode_score(self, recording_episode_index, fields):
        return self._request(dict(kind='score_check', recording_episode_index=recording_episode_index, fields=fields))

    def submit_operator_event(self, event):
        if event not in ('learner_success', 'learner_fail'):
            raise PolicyRequestError(f'Unsupported DSRL operator event: {event}')
        return self.submit_episode_score(self.recording_episode_index,
                                         dict(score=int(event == 'learner_success'), max_score=1))

    def reset(self):
        self.end_episode()

    def close(self):
        self.execution.close()
        self._prefetch_client._ws.close()
        super().close()
