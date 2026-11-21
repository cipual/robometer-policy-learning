"""DSRL noise-policy execution through the embedded EVA client."""

_base_ = ['../../../DenseRL/client/configs/01_deploy/r1lite/_base.py']

import os
from pathlib import Path
from process.runtime_checks import load_settings as _load_settings
import process.dsrl_client as _dsrl_client
import process.dsrl_strategy as _dsrl_strategy

_settings = _load_settings(Path(os.environ['DSRL_CONFIG']))

policy = dict(type='dsrl_online', host='127.0.0.1', port=_settings['serving_port'],
              backend_options=dict(_delete_=True))
# The feedback backend forwards original source pixels; VLA and reward each
# apply their own frozen preprocessing in their independent services.
transport = dict(image_height=360, image_width=640, resize_pad=False,
                 image_layout='hwc', convert_bgr_to_rgb=True, receive_in_subprocess=True)
inference_cfg = dict(publish_rate=_settings['control_hz'], inference_rate=_settings['control_hz'],
                     setup_warmup_chunks=0, debug_tasks=[_settings['task']])
inference_strategies = dict(_delete_=True, rtc=dict(type='DSRLRTCStrategy', args={}))
operator_control = dict(enabled=False)
rollout = dict(storage=dict(enabled=False))
robot = dict(gripper_open=100., gripper_close=0., gripper_open_threshold=_settings['gripper_threshold'])
console = dict(initial_tab='rl')
work_dir = _settings['work_dir']
