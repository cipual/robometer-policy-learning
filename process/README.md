# R1 Lite 策略流程

本仓库与 `../robometer/` 并列，分别使用 Python 3.11 和 Python 3.10 的 `.venv`。上游实现基于 `4fe3476230befaea6ec6fd84c5a8a24eae049669`。原始数据通过链接共享，策略产物放在本仓库的 `work_dirs/`。两条策略线消费 [Robometer 的固定奖励包](../../robometer/process/README.md)。

## 工作站部署入口

- 离线 IQL 首测使用 `20261119-1937_robometer/actors/step-3000/`。部署端通过 [`R1LitePolicy.predict`](r1lite_policy.py) 传入同步的三路 RGB 和 14 维 qpos，取得单步绝对 `action.qpos`。发布包中的 `DEPLOYMENT.md` 给出了输入顺序和录制回放检查命令。[IQL 权重与交接](https://modelscope.cn/models/cipual/robometer-iqr-1120/files)
- 在线 DSRL 使用 [`r1lite_dsrl.py`](r1lite_dsrl.py) 对接现有真机采集。固定奖励模型的权重、部署说明和 Robometer 加载器补丁在 `robometer-261119/`。[奖励模型发布包](https://modelscope.cn/models/cipual/robometer-process-1120/files)

## 1. 本机离线 IQL

`prepare_iql.py` 从 Robometer 同一份配置和 episode UUID 清单生成 `train.h5`、`val.h5`。视频为前视和双腕的 RGB，状态和动作分别取 14 维 `observations.state.qpos` 和日志中实际执行的 `action.qpos`。数据保留原始时间戳、来源和 VR 接管区间。C-sparse 留给独立测试。

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    uv pip install --python .venv/bin/python -r process/requirements-r1lite.txt \
    ../EVA-VLA-R1lite/packages/openpi-client

reward_run_dir="$PWD/../robometer/work_dirs/debug/robometer-261119"
for split in train val; do
    .venv/bin/python -m process.prepare_iql \
        --prepare-config "$reward_run_dir/config/prepare.json" \
        --data-root work_dirs/data/denserl-c-only-1118 \
        --split "$split" --height "$image_height" --width "$image_width" \
        --output "$policy_run_dir/data/$split.h5"
done
```

图像尺寸由本轮配置明确给出，部署沿用相同尺寸和 RGB 顺序。HDF5 原样保留 qpos，按 `obs[t]`、`action[t]` 和真实 `obs[t+1]` 配对。末帧只作相邻转换的后继观测，最后一条可用转换标记终止。当前输出为待重标的 replay，`reward_status` 为 `pending_robometer_relabel`。

在奖励模型仓库用 [`process.score_native_replay`](../../robometer/process/README.md) 给两个 HDF5 分别生成同名 `*-rewards.h5`。sidecar 按回合 UUID 保存绝对 progress 和 success，记录 split、清单 SHA、奖励权重身份和打标方法。IQL 把 progress 作为奖励，保留录制的回合边界。30,000 步实验的 `keep_steps: [3000]` 会额外保存完整的 `ckpt/step-3000.pt`。

[`train_iql.py`](train_iql.py) 复用官方 HDF5 replay、Transformer、Q/V 和 AWR，在本机从静态数据更新。运行配置参照 [`iql.yaml`](iql.yaml)，用训练集拟合状态与动作归一化，按验证集归一化动作 MAE 留下最佳 actor。MAE 是离线选模指标，真机成功率单独评估。启动正式训练前按 AGENTS 记录步数、GPU、估时与实际生效配置，取得批准后执行。

```bash
CUDA_VISIBLE_DEVICES="$iql_gpu" .venv/bin/python -m process.train_iql \
    --config "$iql_config" --output "$policy_run_dir/outputs"
.venv/bin/python -m process.export_iql \
    --checkpoint "$policy_run_dir/outputs/ckpt/best.pt" \
    --output "$policy_run_dir/outputs/actor/ckpt/iql"
.venv/bin/python -m process.r1lite_policy \
    --policy "$policy_run_dir/outputs/actor/ckpt/iql" \
    --replay "$policy_run_dir/data/val.h5" --episode "$validation_uuid"
```

`iql_config` 指向本轮批准的 `config/iql.yaml`。续训在原命令追加 `--resume`，从 `ckpt/last.pt` 恢复模型、优化器、随机状态和 W&B run ID。部署读取 `actor.pt`、`policy.json`，按 RGB 和左右臂关节原顺序返回 14 维 `action.qpos`，包含反归一化和参考频率。独立的 probe 命令只在录制观测上推理。

## 2. 工作站在线 DSRL

奖励进程从 `../robometer/process/serve_reward.sh` 启动，初始分卡为一张 48 GB 卡跑 Robometer 奖励，另一张给冻结的 VLA 推理和 DSRL 更新。记录实际显存峰值和奖励请求延迟。采集进程将回合转换送入 replay，奖励请求可在回合内异步提交，并在结束时补齐。论文使用 success head 提前结束回合或切换阶段时，需要将该输出接入控制循环。

`serve_frozen_vla.py` 在 EVA-VLA-R1lite 的独立环境里加载冻结的 JAX OpenPI checkpoint，通过 loopback 暴露 VLM 特征和显式噪声采样。`--dino-model` 接受本地 DINOv2-small，并输出左腕特征。R1 Lite checkpoint 的模型 horizon 由服务返回。`r1lite_dsrl.py` 复用官方 SAC 和 Transformer 噪声 actor，将 8×32 的噪声重复到模型 horizon，执行侧只消费前 8 条动作。回合收集的前视帧通过 gRPC 异步重标，单阶段每步奖励为绝对 progress 加步惩罚，成功步取消惩罚。

```bash
cd ../EVA-VLA-R1lite
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="$vla_gpu" \
    .venv/bin/python ../robometer-policy-learning/process/serve_frozen_vla.py \
    --config "$openpi_config" --checkpoint "$frozen_vla_checkpoint" \
    --dino-model "$dino_snapshot" --port 18100
```

在策略仓库用 `process.smoke_dsrl --vla-port 18100` 串联冻结的 VLA、录制回放、奖励服务和 SAC 更新。真实工作站用 EVA 控制、前视帧和回合事件调用同一组接口。上游 [真机入口](../scripts/train_dsrl.py) 的现成配置使用 DROID 协议。R1 Lite 执行进程需在工作站交付并按实际控制频率、关节标定和终止事件调试。
