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

当前候选为 `work_dirs/debug/dsrl-online-261120/`，完整计划、普通配置、预算、测量与审批状态见其 README。IQL 后续评测已取消，原始记录保留在 `iql-eval-261120/`。

工作站使用三个独立进程：本项目根 `.venv` 提供冻结 JAX VLA 与 DSRL learner，同级 `../r1lite_robometer/.venv` 提供冻结第100步奖励，`../DenseRL/.venv` 运行内置 EVA client。不得混装环境。全部原生入口显式传入本轮配置；下面命令只有非运动检查可在未获正式批准时执行。

```bash
# 冻结奖励服务；只监听 localhost，不连接机器人
../r1lite_robometer/.venv/bin/python -m process.serve_reward --config "$RUN/config/reward.json"
# 冻结 VLA 服务；XLA_FLAGS 来自普通配置
env -u LD_LIBRARY_PATH .venv/bin/python -m process.serve_frozen_vla --config "$RUN/config/vla.json"
# 录制回放校验并完成编译热身；不发送动作
.venv/bin/python -m process.probe_dsrl --config "$RUN/config/service_probe.json"
# 两个小规模 SAC 更新及保存恢复；不连接机器人
.venv/bin/python -m process.smoke_dsrl --config "$RUN/config/smoke.json"
# 正式网络形状的两个合成 replay 更新；不初始化正式训练
env -u LD_LIBRARY_PATH .venv/bin/python -m process.benchmark_dsrl --config "$RUN/config/benchmark.json"
# 以下两个入口仅在正式训练与真机运动均获明确批准后启动
env -u LD_LIBRARY_PATH .venv/bin/python -m process.online_dsrl --config "$RUN/config/train.json"
.venv/bin/python -m process.start_dsrl_client --config "$RUN/config/train.json"
```

`RUN` 指本轮实验绝对路径；`.manager/commands/` 仅记录上述命令和 tee 日志。人类与 agent 使用相同入口。恢复 learner 在原命令后加 `--resume` 并指定本轮产生的 checkpoint，继续原 W&B run ID；不能用工程冒烟权重初始化在线学习。

用户确认43条完整 PackTask 轨迹，人工停止、评分与复位；每条轨迹及最终评测保留三路视频和动作。DSRL actor/critic 从头学习8×32扩散噪声，重复到冻结模型H10并执行前8条绝对qpos。输入为state14、左腕DINO384、末token VLM2048；不是关节残差，也不混用Dense的阶段/融合目标。

EVA 同源观测经专用RPC送入 learner；发布返回、后继观测、人工评分及原生保存成功共同决定是否能入 replay。噪声 chunk 末端的冻结 progress 减一步惩罚，人工整任务成功仅在末 chunk 加配置中的成功项；success head单独落盘，不作自动终止。SAC 折扣、warmup和更新比的单位均在本轮方案说明，实际值由 train.json 唯一控制。

同步执行的块内参考为15Hz，块边界包含真实特征/采样耗时；上机前必须先完成回放热身，不能把冷编译塞入运动回合。源时钟和真实发布事实保留，不插补训练credit。Console端的Setup、START、停止和复位由现场操作者执行；不得自动homing。ROS反馈短程通过不等于发布长阻塞已修复。

训练视频/动作在 `outputs/train/trajectories/dsrl_online/episodes/raw/`，人工评分唯一来源为其中 `meta/episodes.jsonl`。精确控制源与chunk/reward在 `outputs/train/replay/<episode_id>/`；checkpoint在 `outputs/train/ckpt/`，不默认清理。最终固定策略评测在同一候选后续配置，预算和权重尚待确认，不把在线训练轨迹作为固定策略最终成功率。

工作站当前训练入口打开RL tab，原生成功/失败保存按钮以0/1标记整任务；EVAL仅用于训练完成后的13条固定策略评测。Setup验证不创建训练episode，不消费轨迹预算。

冻结 VLA 的 `prepare` 返回特征和一次性 context ID，`sample_prepared` 只传该 ID 与 actor 噪声，复用同源前缀/KV，不重复传三路图像。CPU DINO 与 JAX 前缀异步执行重叠；全部冷编译在回放 probe 内完成。诊断报告区分模型内部、回环 RPC 与真机源/发布/录制开销，不能把回环吞吐宣称为真机 RTC。

当前用户指定夹爪来源为冻结VLA：`action_control`声明固定标准高斯噪声种子。同一context的paired解码返回`actions`（DSRL提议）和`baseline_actions`，R1LiteDSRL组合12个机械臂关节与baseline的第6/13通道；Native阈值二值化后仍经实际发布反馈核验。源记录保存两路提议、baseline噪声、组合动作及逐步真实命令；不屏蔽扩散噪声的第6/13维。历史全DSRL replay保留但不加入新动作映射的训练buffer。

当前 DSRL 在线路径使用 DenseRL 原生 H50/prefix5/H8：`serve_frozen_vla --config` 按显式 runtime_python 运行独立冻结服务；`online_dsrl --config ... --resume ...` 在本项目环境运行 learner；`start_dsrl_client --config ...` 复用 DenseExecution 与三路录制。`dsrl_strategy` 仅接原生实际源时钟/发布反馈，慢预取使用独立连接。前256个有效已奖励转移走独立完整高斯 warmup，Setup不消耗其RNG。已承诺prefix纳入rtc_prefix71，折扣单位仍为实际8步。真实Setup/START/人工复位由操作者发起。
