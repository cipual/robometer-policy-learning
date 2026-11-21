"""Native EVA half of the isolated cross-environment feedback test; no ROS."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger
from core.types import Observation
from robots.base import ActuatorGroup, CameraSpec, ObservationSchema, Robot
from process.dsrl_client import DSRLPolicyClient
from core.app.handlers import recording


def observation(step):
    stamp = 100 + step / 15
    cameras = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
    return dict(state=np.zeros(14, np.float32), prompt='test',
                images={key: np.full((32, 32, 3), step, np.uint8) for key in cameras},
                _residual_capture_time=stamp, _image_source_times={key: stamp for key in cameras})


def main():
    port, output = int(sys.argv[1]), Path(sys.argv[2])
    cameras = tuple(observation(0)['images'])
    robot = Robot(name='dsrl_fixture', actuator_groups=(ActuatorGroup('arm', 14,
                  tuple(f'q{i}' for i in range(14))),), initial_qpos=np.zeros(14, np.float32),
                  observation_schema=ObservationSchema(cameras=tuple(CameraSpec(key, key) for key in cameras),
                                                       state_composition=('arm',)))
    logger = EpisodeLogger(output, robot, fps=15, convert_bgr_to_rgb=False,
                          dataset_keys=ConfigDict(state_key='state', eef_key='eef', action_key='action',
                              video_keys={key: f'observation.images.{key}' for key in cameras}))
    client = DSRLPolicyClient('127.0.0.1', port, retry_until_connected=False)
    runtime = SimpleNamespace(policy=client, rollout_episode_logger=logger, rollout_vr=None,
        transport=SimpleNamespace(start_policy_collection=lambda:None, clear_collection_backlog=lambda:None,
                                  stop_collection=lambda:None))
    session = SimpleNamespace(selected_task='test',last_error='')
    try:
        validated = client.infer(observation(0))
        assert validated['validation_only'] and client.episode_id is None
        with patch.object(recording,'stop_collection_capture'), patch.object(recording,'start_collection_capture'), \
             patch.object(recording,'maybe_build_rollout_episode_logger'):
            recording.begin_rollout_save_episode(ConfigDict(inference_cfg=dict(publish_rate=15)), runtime, session)
        logger.set_episode_meta(clip_id='cross-env-fixture')
        for step in range(16):
            obs = observation(step)
            client.observe(obs)
            action = client.execution.pop()
            action[[6, 13]] = np.where(action[[6, 13]] >= 80, 100, 0)
            logger.record_step(Observation(images=obs['images'], state_qpos=obs['state']),
                               action, timestamp=obs['_residual_capture_time'])
            # A fake publication is acknowledged only inside this isolated fixture.
            client.record_execution(action, obs)
            client.execution.published(action)
        assert client.stop_requested
        client.bind_observation_provider(lambda: observation(16))
        recording.mark_rollout_save_ready(runtime, 'stop')
        assert recording.score_rollout_episode(runtime,session,1)
        with patch.object(recording,'stop_collection_capture'):
            assert recording.save_rollout_episode(runtime,session)
        logger.wait_for_saves()
        assert len(list(output.glob('videos/chunk-000/*/episode_000000.mp4'))) == 3
        assert logger.episode_index_for_clip_id('cross-env-fixture') == 0
    finally:
        client.close()
        logger.finalize()


if __name__ == '__main__':
    main()
