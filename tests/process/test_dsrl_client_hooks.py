"""Run in the embedded EVA environment; no ROS participant or publisher is created."""
import json
from types import SimpleNamespace

import numpy as np

from core.app.handlers import control, recording
from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger
from core.types import Observation
from robots.base import ActuatorGroup, ObservationSchema, Robot


def test_publication_ack_is_after_transport_and_uses_the_same_source(monkeypatch):
    calls = []
    source = dict(state=np.zeros(2, np.float32), _residual_capture_time=10.)
    policy = SimpleNamespace(requires_execution_feedback=True, stop_requested=False,
                             record_execution=lambda action, obs: calls.append(('ack', obs)))
    transport = SimpleNamespace(has_sim_publishers=lambda: False,
                                publish_action=lambda action, target: calls.append(('publish', target)))
    runtime = SimpleNamespace(policy=policy, transport=transport)
    monkeypatch.setattr(control, 'record_executed_action',
                        lambda *args, **kwargs: calls.append(('record', kwargs['source_observation'])))
    control.publish_real_action_with_preview(ConfigDict({}), runtime, np.zeros(2),
                                             SimpleNamespace(step_index=0), source_observation=source)
    assert [row[0] for row in calls] == ['publish', 'record', 'ack']
    assert calls[-1][1] is source


def test_episode_lifecycle_link_survives_native_save_and_readonly_clip_lookup(tmp_path, monkeypatch):
    robot = Robot(name='feedback_fixture', actuator_groups=(ActuatorGroup('arm', 2, ('a', 'b')),),
                  initial_qpos=np.zeros(2, np.float32),
                  observation_schema=ObservationSchema(cameras=(), state_composition=('arm',)))
    logger = EpisodeLogger(tmp_path, robot, fps=15, eval_mode=True,
                           dataset_keys=ConfigDict(state_key='state', eef_key='eef',
                                                    action_key='action', video_keys={}))
    calls = []
    policy = SimpleNamespace(requires_execution_feedback=True,
        begin_episode=lambda task, index: dict(policy_episode_id='fixture-episode'),
        end_episode=lambda: calls.append('end'))
    runtime = SimpleNamespace(episode_logger=logger, policy=policy,
                              transport=SimpleNamespace(start_policy_collection=lambda: None))
    for function in ('prepare_collection_capture', 'stop_collection_capture',
                     'suspend_evaluation_gc', 'resume_evaluation_gc'):
        monkeypatch.setattr(recording, function, lambda *args: None)
    recording.start_episode(runtime, SimpleNamespace(selected_task='test'))
    logger.set_episode_meta(clip_id='fixture-clip', score=3, max_score=3)
    for index in range(2):
        logger.record_step(Observation(images={}, state_qpos=np.zeros(2, np.float32)),
                           np.zeros(2, np.float32), timestamp=10 + index / 15)
    recording.end_episode(runtime)
    logger.wait_for_saves()
    path = tmp_path / 'meta/episodes.jsonl'
    before = path.read_bytes()
    rows = [json.loads(line) for line in before.decode().splitlines()]
    assert rows[0]['policy_episode_id'] == 'fixture-episode'
    assert logger.episode_index_for_clip_id('fixture-clip') == 0
    assert path.read_bytes() == before
    assert calls == ['end']
    logger.finalize()


def test_training_config_rejects_stale_eval_start_requests():
    from core.app.console.server import ConsoleRequestHandler
    replies, queued = [], []
    handler = SimpleNamespace(ctx=SimpleNamespace(config=SimpleNamespace(eval=None)),
                              _send_json=lambda status, body: replies.append((status, body)),
                              _enqueue=queued.append)
    ConsoleRequestHandler._eval_start(handler, dict(clip_id='old-iql',prompt='test',trial=1))
    assert replies[0][0] == 409 and not queued


def test_feedback_failure_stops_control_without_exiting_client(monkeypatch):
    from core.app.handlers import io
    from core.app.state import SessionMode, SessionStatus
    from policy_client.base import PolicyRequestError
    calls=[]
    def failed(obs):
        raise PolicyRequestError('source gap')
    runtime=SimpleNamespace(infer_strategy=None,policy=SimpleNamespace(
        requires_execution_feedback=True,observe=failed))
    session=SimpleNamespace(mode=SessionMode.REAL,selected_task='test',last_error='')
    monkeypatch.setattr(control,'is_replay',lambda runtime:False)
    monkeypatch.setattr(io,'_loop_observation',lambda *args:dict(state=np.zeros(14)))
    monkeypatch.setattr(control,'stop_collection_capture',lambda runtime:calls.append('stop'))
    monkeypatch.setattr(control,'mark_rollout_save_ready',lambda *args:calls.append('retain'))
    monkeypatch.setattr(control,'set_status',lambda session,status,**kw:setattr(session,'status',status))
    assert control.publish_next_action(ConfigDict({}),runtime,session) is False
    assert calls==['stop','retain'] and session.status is SessionStatus.READY


def test_rtc_fault_aborts_learning_but_leaves_native_recording_saveable():
    from process.dsrl_client import DSRLPolicyClient
    calls=[]
    client=object.__new__(DSRLPolicyClient)
    client.episode_id='faulted-fixture'
    client.execution_error='prefetch missed committed prefix'
    client.execution=SimpleNamespace(reset=lambda:calls.append('reset'))
    client._request=lambda message:calls.append(message)
    client.end_episode()
    assert calls[0]['kind']=='abort' and calls[0]['episode_id']=='faulted-fixture'
    assert calls[1]=='reset' and client.episode_id is None
