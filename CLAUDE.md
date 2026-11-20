# Robometer 策略基线

项目规则见 [AGENTS.md](AGENTS.md)，执行顺序和接口见 [process/](process/README.md)。奖励包由同级 `../robometer/` 产出，两条策略线均固定同一份奖励模型。

- 本机离线 IQL 对齐论文 E.3。用冻结的 Robometer 对 C-only 原始前视视频逐帧打 progress 分数，复用官方 Q、V 和 AWR 学独立的直接动作 actor，导出后去工作站做真机评测。
- 工作站在线 DSRL 对齐论文 E.2。冻结基础 VLA，用 SAC 训练独立的噪声 actor。真机回合写入 replay 后由 Robometer gRPC 异步补标。使用论文的自动终止时另接 success head 的回合内查询。
