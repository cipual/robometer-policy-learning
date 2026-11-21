"""Single-step IQL qpos deployment using DenseRL's embedded EVA client kernel."""

import json
import os
from pathlib import Path

_base_ = ["../../../DenseRL/client/configs/01_deploy/r1lite/_base.py"]

_settings = json.loads(Path(os.environ["IQL_EVAL_CONFIG"]).read_text())
_metadata = json.loads((Path(_settings["policy_directory"]) / "policy.json").read_text())

policy = dict(type="openpi", host="127.0.0.1", port=_settings["serving_port"],
              backend_options=dict(_delete_=True))
transport = dict(image_height=_metadata["image_shape"][0], image_width=_metadata["image_shape"][1],
                 resize_pad=False, image_layout="hwc", convert_bgr_to_rgb=True,
                 receive_in_subprocess=True)
inference_cfg = dict(publish_rate=_metadata["reference_hz"],
                     inference_rate=_metadata["reference_hz"], debug_tasks=[_metadata["task"]])
inference_strategies = dict(_delete_=True, sync=dict(type="BaseInferStrategy", args=dict(
    execute_horizon=_metadata["execution_horizon"], sync_wait_until_reached=False)))
operator_control = dict(enabled=False)
rollout = dict(storage=dict(enabled=False))
robot = dict(_settings["robot"])
console = dict(initial_tab="eval")
work_dir = _settings["work_dir"]
