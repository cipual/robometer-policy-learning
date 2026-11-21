"""使用 EVA 的 EVAL 评分与录制流程评测冻结 IQL actor。

在独立策略进程提供单步绝对 qpos 服务，由 EVA 原生客户端管理回合、里程碑评分和保存。
输入：--config 指定模型、客户端、ROS 环境及评测参数。输出：EVA 逐回合视频、动作和评分。
"""

import argparse
from pathlib import Path

from process.iql_evaluation import evaluate, load_settings, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--serve-only", action="store_true", help="仅启动本地 actor 服务，不连接机器人")
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings, _ = load_settings(config_path)
    if args.serve_only:
        serve(settings)
    else:
        evaluate(config_path, settings)


if __name__ == "__main__":
    main()
