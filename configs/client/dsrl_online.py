"""Collect DSRL training trajectories in EVA's native RL workspace."""

_base_ = ['dsrl_deploy.py', '../../../DenseRL/client/configs/00_base/rl.py']

import os
from pathlib import Path
from process.runtime_checks import load_settings as _load_settings

_settings = _load_settings(Path(os.environ['DSRL_CONFIG']))
rl_cfg = dict(_settings['rl_cfg'])
rl_cfg['policies'] = [dict(name='DSRL online — 43 training trajectories', kind='vla',
    config=_settings['client']['deploy_config'], host='127.0.0.1', port=_settings['serving_port'])]
console = dict(initial_tab='rl')
eval_cfg = None
