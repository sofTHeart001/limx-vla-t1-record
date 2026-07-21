# 参赛文档摘要

## 总流程

1. 根目录创建并激活 `troncamp_env`，Python 3.10。
2. 安装 RoboTwin、InterACT-compatible 训练栈、cuRobo 0.8.0。
3. 在根目录还原 `__KIT_ROOT__` 到当前绝对路径。
4. 根目录执行 `collect_data.sh` 采集成功轨迹。
5. 在 `external/robotwin_local/policy/ACT` 下执行数据处理、训练和 `eval.sh`。
6. 根目录执行 `starter/eval_local.py`、`starter/watch_rollout.py` 和 `submit/submit.py`。

## 关键路径

- 官方包根目录：`external/robotwin_local`
- 官方策略目录：`external/robotwin_local/policy/ACT`
- 采集配置：`external/robotwin_local/task_config/<task_config>.yml`
- 处理后数据：`external/robotwin_local/policy/ACT/processed_data/`
- 训练权重：`external/robotwin_local/policy/ACT/act_ckpt/act-<task>/<config>-<num>/`
- 本地评测结果：`eval_result/`

## 提交规则

- T1 只提交 `policy_best.ckpt`。
- T2/T3/T4 提交 `policy_best.ckpt` 加 `external/robotwin_local` 代码包。
- 解锁顺序为 T1 -> T2 -> T3 -> T4。
- 每日正式提交 2 次，本地自评不限次。

## 注意事项

- T2-T4 的采集配置需要按任务单独准备；本项目最终使用 `grab_roller_600ep.yml`、`stack_bowls_two_600ep.yml` 和 `stack_bowls_three_600fast.yml`。
- 采集、训练、评测依赖可用 NVIDIA 驱动和 CUDA 12.x。
- `__KIT_ROOT__` 占位符必须在根目录还原，否则 cuRobo 规划器会失败。
- 官方评测只加载提交的 ckpt、`policy/ACT` 代码和推理配置，不加载 `envs/` 专家。
