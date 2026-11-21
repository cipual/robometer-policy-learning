"""EVA EVAL workflow with the IQL experiment's scoring and recording configuration."""

import json
import os
from pathlib import Path

_base_ = ["iql_deploy.py", "../../../DenseRL/client/configs/00_base/evaluation.py"]

_settings = json.loads(Path(os.environ["IQL_EVAL_CONFIG"]).read_text())
eval_cfg = dict(_settings["eval_cfg"])
eval_cfg["checkpoints"] = [dict(name=_settings["model_name"],
    config=_settings["client_deploy_config"], host="127.0.0.1", port=_settings["serving_port"])]
