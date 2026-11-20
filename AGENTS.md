# Robometer Policy Learning 项目规则

- 本仓库独立于 `../robometer/`，上游基准为 `robometer/robometer-policy-learning@4fe3476230befaea6ec6fd84c5a8a24eae049669`。通过 `gh` 管理 GitHub 引用。保留官方 IQL、SAC 和 DSRL 实现，R1 Lite 接线放在 [process/](process/README.md)。
- 本仓库使用 `.venv` 和 Python 3.11，安装依赖优先用 uv 清华镜像。奖励服务在 `../robometer/` 的 Python 3.10 `.venv` 独立运行，两边通过 gRPC 或文件交接。
- `work_dirs/` 在主仓库中完全忽略。`work_dirs/data/denserl-c-only-1118` 链接到 `../robometer/work_dirs/data/denserl-c-only-1118`。冒烟输出存 `work_dirs/debug/`，正式 IQL 记录存 `work_dirs/iql/<日期时间>_<实验名>/`，迭代前归档到 `work_dirs/debug/archive/`。私有 `work_dirs` 版本库只同步有效配置、启动命令、代码版本、环境锁和随机种子等轻量记录。
- 用 Robometer 注册的原始回合清单划分训练与验证。训练仅读取 C-only，C-sparse 是独立测试。保持原始 qpos 顺序、帧对齐和接管区间。保存策略时必须同时记录模型构造参数、相机预处理和动作反归一化。
- Robometer 奖励重标读取原始前视视频，用当前转换的因果前缀给 progress 和 success 打分。策略读取自己的三路 HDF5 图像。`reward_method` 和奖励文件 SHA 参与训练恢复校验。
- `process.train_iql` 使用独立 Python 3.11 环境读取已重标 HDF5，`process.export_iql` 导出直接动作 actor，`process.r1lite_policy` 在真实录制观测上检验加载。`process.serve_frozen_vla.py` 由 EVA 仓库的环境承载，策略和奖励进程各用自己的环境。
- 启动正式训练前检查冒烟、保存恢复、吞吐和磁盘。上报实际训练步数、物理 GPU 索引与空闲状态、预计时间、数据路径和完整配置，等待用户明确批准。真机评测使用工作站实际部署配置。
