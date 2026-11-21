"""Serve the frozen Robometer from an explicit ordinary --config file.

Run with the reward repository's Python environment. Input: local model and
runtime settings. Output: loopback reward RPC; no robot or training connection.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    from process.runtime_checks import load_settings
    settings = load_settings(parser.parse_args().config)
    if settings["format"] != "r1lite_reward_service_v1":
        raise ValueError("Expected an R1 Lite frozen reward service configuration")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["physical_gpu"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(Path(settings["robometer_repo"]).resolve(strict=True)))

    import torch
    from robometer_policy_learning.distributed.servers.reward_relabel_server import RewardRelabelServer

    torch.set_num_threads(settings["torch_num_threads"])
    server = RewardRelabelServer(
        model_path=settings["model_path"], host="127.0.0.1", port=settings["port"],
        max_workers=settings["max_workers"], max_msg_mb=settings["max_msg_mb"],
        batch_size=settings["batch_size"],
        image_keys=settings["image_keys"], language_key=settings["language_key"],
        sentence_model_name=None, device=torch.device("cuda:0"), reward_model_type="robometer",
    )
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
