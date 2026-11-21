# Robometer Policy Learning 工作站约定

通用研究流程遵循用户提供的 AI research 工作流。本文件与 CLAUDE.md 记录 2026-11-20 已确认的工作站方案；不把目录确认当作训练、真机运动或最终验收批准。不使用子 agent。

- 本项目负责 R1 Lite 的固定 IQL actor 评测与 Robometer 奖励驱动的 DSRL 在线学习；方法与实验记录独立，IQL/DSRL 执行端按用户确认复用 DenseRL/client 内置 EVA 内核。开发基于用户指定的 `cipual/robometer-policy-learning@13e6725ac38f9f5eb76f4af84aca8769a13cdd0d` fork，上游基准为 `4fe3476230befaea6ec6fd84c5a8a24eae049669`，DSRL OpenPI 子模块为 `b0e3cfbfb1fb46622b4d499fbeeada95d247f4a4`。
- 唯一策略环境为根 `.venv`，Python 3.11，通过 uv 管理；默认 `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`。IQL 评测已安装独立 actor/RPC/录制回放依赖；DSRL 策略与奖励环境已安装并完成冻结服务回放检查；上线状态以 dsrl-online-261120 首页及实际冒烟证据为准。上游 pyproject 的完整依赖涉及 LIBERO、JAX 和 Robometer，安装前应核对实际真机路径，不能直接宣称 `uv sync` 可解。
- 奖励推理使用同级 `../r1lite_robometer/` 自己的 Python 3.10 `.venv`；策略与奖励代码不得借用 DenseRL、EVA-CLIENT 或 rlinf-dev 的环境；复用的 DenseRL/client 执行进程使用 DenseRL 根 .venv，各自环境不混装依赖。官方 `../robometer/` 仅保留作源码参考。
- 本项目根 `work_dirs/` 是共享资产和全部实验记录的唯一工作目录。`data/` 引用明确登记的原始回放和视频；`ckpt/` 存放 ModelScope 冻结交接包及共享冻结 VLA；新候选为 `debug/<实验名>-<YYMMDD>/`，最终材料和结论经人确认后才进入 `run/`。不按 collect/offline/online/eval 再拆分 run，也不新增算法目录层。
- 当前完整目标分别是 `iql-eval-261120` 与 `dsrl-online-261120`。IQL 目标只评测 step-3000，last 仅作为交接来源保留；不重训 IQL。DSRL 目标包含资产核验、奖励服务、R1 Lite 接线和冒烟、在线学习、最终评测及结论。修 bug 在同一候选内 revise。
- 每个候选共享 `config/`、`logs/`、`outputs/<step>/` 和 `.manager/`。不预建未使用的日志、权重和可视化目录。训练 checkpoint 写 `outputs/<step>/ckpt/`，评测写 `outputs/<step>/test/`，可视化写其 `vis/`。用户确认 DSRL 每条训练轨迹与全部最终评测保存三路视频和动作；预算43条，人工标记成功/失败并人工复位。保留全部逐回合 checkpoint，不静默删除。
- 实验管理只在 agent 侧：使用独立 `tool/agent-manager/` 的 SKILL.md 与 scripts/experiment.py。唯一项目声明为 `configs/experiments.json`。普通配置是实际参数唯一来源；尚未实现或未明确预算的步骤配置保留 null，不伪造 SHA 或生成不可执行的启动脚本。
- 人类与 agent 共用 `process/` 的原生入口，显式 `--config`。不向算法代码引入生命周期、配置生成或审批逻辑。复用 fork 既有 `process.r1lite_policy`、`process.r1lite_dsrl`、`process.serve_frozen_vla`、`process.smoke_dsrl` 和 reward_client；新增 `process/00_check_assets.py` 仅做资产完整性校验。工作站原生入口已接线并有无运动回放/模拟证据，真机发布与长期训练仍待批准和现场核验。既有来源入口的 CLI 在需要修改时按单个普通 --config 收敛，不为目录形式增加转发包装。
- IQL 来源是 `cipual/robometer-iqr-1120/20261119-1937_robometer`。必须沿用交接的 runtime、policy.json 和 SHA256SUMS；观测为同一源时刻的 14 维 qpos 与 RGB uint8 三相机 `cam_high/cam_left_wrist/cam_right_wrist`。执行单步绝对 action.qpos，参考 15 Hz，夹爪量纲 0–100，不能套用 Dense 残差或 H8 融合。
- 当前 IQL 真机评测方案为 step-3000 共 13 回合，禁用 VR 干预，统一任务与初始场景，记录整任务成功、foam/phone/lid 三项里程碑、时长、人工中止，以及逐回合视频与动作。接线先回放，再关闭动作下发核对实时观测与延迟，真机运动另行获批。
- DSRL 使用完整 PackTask 指令做整回合学习，三项里程碑用于评测；冻结奖励为 `cipual/robometer-process-1120/robometer-261119` 第 100 步，DSRL noise actor/critic 从头学习，不用 IQL actor 初始化，不训练奖励模型或 VLA。
- 冻结 VLA 明确沿用 `/mnt/data/VLA-RL/rlinf-dev/workdirs/ckpt/base_vla_rtc_d5_all`，通过 `work_dirs/ckpt/base-vla` 软链接引用。同源 JAX 原件已找到：`/mnt/data/VLA-RL/rlinf-dev/workdirs/expo/ckpts/r1lite_pack_phone_all/checkpoints/r1lite_pack_phone_rtc_d5_all/r1lite_pack_phone_rtc_d5_all/24999`，通过 `work_dirs/ckpt/base-vla-jax` 引用；优先沿用 fork 的 JAX NoisePolicy。JAX 与 PyTorch 包的 norm SHA 一致，已有本地转换记录说明该 PyTorch 包来自此 JAX checkpoint；已核验全部 19 个 JAX 文件与来源 Git/LFS 哈希一致。实际冻结服务加载、显式噪声推理及同源回放已通过，见 DSRL 候选证据。必须核对模型、norm、transform 与执行时序，不能用名称代替身份核验。DSRL 的噪声接口、VLM 特征、horizon、控制频率、每次执行步数和折扣单位在实现及冒烟后写入普通配置，不照抄 DROID 默认值，不引入 Dense actor 残差目标。
- 奖励输入使用原始前视 cam_high 与同一任务指令。先在录制数据核验 progress/success、时间对齐、异步回填和延迟；success 终止阈值须用现场数据核验，不能直接沿用论文阈值。训练奖励组合与截断语义在正式配置中明确。
- 默认只做非运动资产核验、回放与 fake 测试。正式训练前完成端到端小规模冒烟、保存恢复、吞吐/GPU 利用率和磁盘核算，再汇报步数、物理 GPU、耗时、数据和配置，等待批准。真实运动必须另外明确批准；不得停止或挤占其他项目进程。
- R1 Lite 硬件与录制复用既有 EVA 客户端接口，先核实当前可用执行端；不复制 Dense 算法，不改外部环境。先前 Dense 真机出现 ROS 发布长阻塞，不能因本项目接线完成而宣称已修复，接真机前需核验当前状态。
- `work_dirs/` 完全不被外层仓库跟踪；内层轻量实验记录自动以 `cipual <1317607404@qq.com>` 提交并推送至私有 `cipual/robometer-policy-learning_work_dirs`。历史服务器 `cipual/robometer-policy-learning_workdirs` 仅作来源，不迁移或改写。数据、权重、日志、输出、下载缓存和凭据不提交。主仓库及工具仓库未经明确 commit 请求不提交或推送。
- GPU 分工以本轮普通配置和实际占用为准；当前准备方案共置物理 GPU1，GPU0 为他人进程占用。数量、索引、预算和执行时长尚须正式批准。

## IQL 评测接线（2026-11-20）

- 用户已确认复用 `/mnt/data/VLA-RL/DenseRL/client` 内置 EVA 的回合管理、foam/phone/lid 里程碑评分、EpisodeLogger 保存与 RESULT 回放，不调用 denserl.evaluation，也不复制评测内核。客户端使用 `/mnt/data/VLA-RL/DenseRL/.venv/bin/python`，工作目录为 DenseRL/client；IQL actor/RPC 保持本项目 Python 3.11 环境，不修改 DenseRL 环境。实验仍为 `work_dirs/debug/iql-eval-261120/`，不在 DenseRL 下创建 IQL run。
- 本项目入口为 `.venv/bin/python -m process.eval_iql --config RUN/config/evaluate.json`；`--serve-only` 只提供 loopback OpenPI 协议服务，不连接机器人。`process.probe_iql --config RUN/config/replay.json` 按显式 parquet 与三路视频的同一帧号检查直接推理/RPC及耗时，不发送机器人动作。
- `configs/client/iql_deploy.py`、`iql_eval.py` 从 `IQL_EVAL_CONFIG` 指定的普通 evaluate.json 读取本轮设置，引用 DenseRL/client 的共享 R1 Lite 与 EVAL 配置，跳过 Dense-H8 业务配置。使用 OpenPI 客户端、BaseInferStrategy 同步 H1、15Hz，无 RTC/H8/EMA/残差；RGB HWC 128×128 直接双线性缩放与交接 loader 一致。夹爪二值化沿用现有评测阈值，在普通配置中显式登记。
- DenseRL 内置 client 含其他工作中的未提交改动，不改写或提交它们；本轮 evaluate.json 显式绑定实际内置客户端源文件 SHA，执行前变更须重新核验和绑定。配置中的 work_dir 与 eval_cfg.output_dir 均显式指向本项目，ROS 接收沿用后台子进程。真实动作发布、速度/关节约束、现场 ROS 与实时观测检查尚待人工评审和运动批准。
- 工程回放数据明确引用 `/mnt/data/VLA-RL/DenseRL/work_dirs/data/base_vr_collection`，通过 `work_dirs/data/iql-probe-base-vr` 链接共享。它只用于接口检查，不作为真机成功率或新增训练数据。实际 parquet、三路视频、采样帧与 SHA 由 replay.json 唯一登记。

## DSRL 在线接线与当前范围（2026-11-20）

- 用户已取消继续 IQL 评测；保留 iql-eval-261120 的一条未评分录制和动作语义证据，不计作完成实验。当前目标为 dsrl-online-261120。
- 在线学习沿用完整 PackTask；人工在内置客户端RL tab标记整任务成功/失败（1/0），人工停止并复位。冻结 Robometer 第100步 progress 用于奖励，success 仅记录，不自动终止。
- process.serve_reward（奖励独立环境）、process.serve_frozen_vla（根环境）、process.probe_dsrl、process.smoke_dsrl、process.benchmark_dsrl 均接受单个 --config。在线原生入口为 process.online_dsrl --config；单独 process.start_dsrl_client --config 启动内置 EVA，Setup/START/复位由人操作。
- 内置 EVA 仅增加通用的回合身份、实际发布后反馈、同源观测与评分通知接口；DSRL 算法和 RPC backend 在本项目，不复制 Dense 方法。只在实际发布、后继观测、人工评分及原生录制 saved 后入 replay。
- 噪声为8×32，重复至冻结 VLA H10，执行前8个绝对 qpos；不是在 VLA 输出上加 IQL/关节 delta。state14、左腕 DINO384、末 token VLM2048。训练配置显式定义 warmup、更新比、gamma单位和回合上限，尚须正式批准。
- 先运行录制回放 probe 使冻结服务完成编译热身；真机中采用同步 chunk，15Hz 是块内参考频率，chunk 边界有真实推理延迟，不宣称端到端恒定15Hz。保留源时间，不补造已执行步。
- 当前 GPU0 为其他项目占用；准备方案在物理 GPU1 共置冻结奖励、VLA 与 learner。用户已要求启动43条训练，随后再次确认人工成功/失败判定；learner/client已启动，真实Setup/START/复位由现场操作者发起，不自动操作。

- 2026-11-20 用户明确训练/评测边界：43条在线采集与DSRL更新在内置RL tab；完成训练后冻结actor，在EVAL评测13条。训练配置不启用eval，Setup验证不创建训练回合，人工RL二元评分通过现有operator event回传。

- 2026-11-20 推理优化：通过显式一次性 context ID 将同源观测的 JAX 前缀/KV 交给后续噪声采样复用；保持 CPU DINO，与异步 GPU 前缀重叠后再取回特征。模型、norm、H10、执行8步和夹爪行为不变。GPU DINO 因 actor 等价检查未通过而不采用；冷编译必须在运动前热身。已有轨迹/replay/优化器/RNG 保留，源码绑定改变时通过显式派生 checkpoint 记录兼容迁移，不放宽原生恢复校验。

- 2026-11-20 用户随后明确改为“机械臂用 DSRL、夹爪用冻结 VLA”。同一准备观测/KV分别按actor噪声和固定标准高斯baseline噪声解码；组合12个DSRL机械臂关节与两个baseline夹爪，再沿用阈值80二值化。baseline噪声种子/分布由当前train.json的action_control明确登记，不占用actor/replay全局RNG。夹爪state仍保留。历史72个全DSRL replay chunk与三路录制完整保留但排除出新映射的学习buffer；原actor/critic从未梯度更新，参数/优化器/RNG保持，策略版本提升。原43条总采集预算已使用1条，剩42条混合策略采集，最终仍13条评测。每chunk保存两路提议、baseline噪声及组合动作，逐步事实保存真实发布后命令。

## Dense base 契约恢复（2026-11-20，替代此前 JAX H10 同步部署）

- 用户明确要求恢复 DenseRL base 的完整推理、图像处理与 RTC，并补独立 warmup。冻结 VLA 服务直接复用 DenseRL `.venv` 中既有 PyTorch/OpenPI runtime 和 `TorchBackend`；根入口按 vla.json 的 `runtime_python` 启动外部服务。DSRL learner 仍用本仓库 `.venv`，奖励仍用 r1lite_robometer 环境；没有安装或跨装依赖。此显式外部服务取代此前优先 JAX 的方案，不能再用旧 H10 同步结果描述当前部署。
- 冻结服务使用同源 PyTorch 权重、完整 H50×32 噪声、10次去噪；客户端沿用 Dense 原生 DenseExecution，prefix5、H8、15Hz。原始三路源图像按原生 360×640、resize_pad=False、HWC、BGR→RGB，再用 FrozenOnlineInputs.flat 的 cam_high 条件裁剪与规范 norm/tokenize/224 resize。预取独立socket，不能阻塞实际发布反馈。源断流/RTC超时停止并保留原始录像，但排除训练。
- warmup 是独立 CPU torch.Generator(seed42) 每次推进的完整 iid 标准高斯 H50×32；前256个已奖励有效转移绕过 actor。Setup与编译热身使用隔离 RNG，不消耗采集噪声流，不创建回合。高斯不裁剪、不重复；replay 存 noise/noise_bound，warmup 样本可超出[-1,1]，保持原base先验。之后机械臂用DSRL actor，夹爪用同观测/前缀的独立高斯冻结VLA分支；warmup相同两路只解码一次。
- RTC 的噪声决策发生在源步0、3、11、19…；观测加入已承诺五步动作的归一化70维及有效位（rtc_prefix71）。训练转移覆盖相邻决策之间真实发布的8条边，状态包含队列前缀；不是把旧5步冒充新噪声已经执行。启动0→3只保留原始事实，不建立错误8步bootstrap。gamma显式为0.995^8，禁用按H50自动指数；终止与时间截断仍分开处理。
- 原actor/critic实际更新为0。H50与prefix71使模型形状改变，按seed42重新初始化；旧轨迹、569次实际发布、三路视频及所有checkpoint完整保留，旧72个不兼容replay仍不进入新buffer。当前版本3，总43条已使用1条、剩42条；训练后EVAL13条。数值回放与假发布检查不代表真机闭环验收。当前状态、耗时、恢复来源见 dsrl-online-261120 首页及 .manager/agent-notes/dense-rtc-*。
