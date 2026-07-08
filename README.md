# TronCamp Mani T1：基于 ACT 的机器人模仿学习项目记录

这是我在 TronCamp Mani 机器人操作任务中的 T1 项目记录。目标是在 RoboTwin 双臂仿真环境中完成 `adjust_bottle` 任务：从本地采集演示数据开始，完成数据处理、ACT 策略训练、本地公开 seed 评估和官方提交。

这个仓库面向项目展示和复盘，不直接包含大体积训练产物。采集数据、processed data、checkpoint、评估日志和提交 token 都已排除。

## 项目结果

| 项目 | 结果 |
|---|---|
| 任务 | T1 `adjust_bottle` |
| 方法 | ACT imitation learning |
| 本地演示数据 | 200 episodes |
| 训练 seed | 0 |
| 最佳验证损失 | 0.028757，epoch 5125 |
| 本地公开 100 seed 评估 | `sr = 0.52` |
| 策略部署演示 | seed `20260631`，138 步成功 |
| 官方提交 | 已提交，queue id `#70` |

本地评估结果保存在 [records/t1_public_eval.json](records/t1_public_eval.json)，完整记录见 [records/t1_record.md](records/t1_record.md)。

## 策略部署演示

下面是训练后的 ACT policy 在 T1 环境中的一次成功 rollout。该视频不是专家演示，而是加载训练得到的 `policy_best.ckpt` 后，由策略闭环控制机器人完成任务。

![T1 policy rollout success](media/t1_policy_rollout_success_seed_20260631.gif)

原始 MP4：[`media/t1_policy_rollout_success_seed_20260631.mp4`](media/t1_policy_rollout_success_seed_20260631.mp4)

这次 rollout 使用公开 seed `20260631`，在 138 个策略步内触发成功条件。

## 数据采集样例

下面是一次 T1 专家/数据采集样例，用于对比任务形态和演示数据来源。

![T1 collect demo](media/t1_collect_demo_episode43.gif)

原始 MP4：[`media/t1_collect_demo_episode43.mp4`](media/t1_collect_demo_episode43.mp4)

## 技术路线

整体流程如下：

```text
RoboTwin 仿真任务
        |
        v
本地采集 200 条 T1 演示数据
        |
        v
ACT 数据预处理
        |
        v
训练 ACT policy
        |
        v
公开 100 seed 本地评估
        |
        v
官方提交 T1 checkpoint
```

主要技术栈：

- Python / PyTorch
- ACT policy
- RoboTwin 双臂机器人仿真环境
- CUDA 单卡训练与评估
- 本地评估脚本和官方提交脚本

## 我完成的工作

- 修复本地 GPU / CUDA 运行环境，保证训练和评估能使用显卡。
- 基于 T1 `adjust_bottle` 任务采集 200 条本地演示数据。
- 完成 ACT 数据处理和模型训练。
- 对 `policy_best.ckpt` 做公开 100 seed 本地评估。
- 录制成功的策略部署 rollout，用于展示训练后模型的闭环执行效果。
- 将模型提交到官方评测队列。
- 整理项目记录仓库，排除大体积数据、权重和敏感凭据。

## 关键观察

- 当前 `sr = 0.52` 只比 0.50 略高，说明这是一个能跑通完整流程的 baseline，但不是稳定高分模型。
- `policy_best.ckpt` 是按验证 loss 选出的，不一定等价于 rollout 成功率最高的 checkpoint。
- 机器人闭环任务对早期动作误差比较敏感，轻微偏差可能在后续放大为失败。
- 本地公开 seed 和官方私有 seed 同分布但不相同，公开分数只能作为官方成绩的估计。

## 仓库结构

```text
records/
  t1_record.md             # T1 训练、评估、提交记录
  t1_public_eval.json      # 本地公开 seed 评估结果
media/
  t1_policy_rollout_success_seed_20260631.gif
  t1_policy_rollout_success_seed_20260631.mp4
  t1_collect_demo_episode43.gif
  t1_collect_demo_episode43.mp4
recipes/
  eval/                    # 本地评估相关脚本
  train/                   # ACT 训练相关脚本
scripts/                   # 一键采集、处理、训练、评估、提交入口
starter/                   # 本地评估和可视化入口
  record_policy_rollout.py # 单 seed/搜索成功 seed 的策略部署录制脚本
submit/                    # 官方提交脚本
```

## 不包含的内容

这些内容没有放进公开仓库：

- `external/robotwin_local/`
- 采集得到的 `.hdf5` 演示数据
- ACT `processed_data/`
- `.ckpt` checkpoint
- 本地评估日志
- 官方提交 token 或其他凭据

这样做是为了避免仓库体积过大，也避免把比赛提交凭据或本地生成产物公开。

## 后续验证计划

我后续会把这个项目继续做成一个可复盘的机器人模仿学习实验记录，重点验证：

- checkpoint 选择：对比 `policy_best`、`policy_last` 和不同 epoch checkpoint 的 rollout 成功率。
- 训练随机性：用不同训练 seed 复现实验，观察成功率波动。
- 数据规模：比较 200、更多演示数据下的 ACT 表现。
- 失败分析：继续录制失败 rollout，对比瓶子姿态、夹爪动作和关键时刻偏差。
- 评估稳定性：多次 repeat 和不同公开 seed 子集下的成功率置信区间。
- 任务迁移：尝试从 T1 扩展到 T2/T3，并记录需要改动的策略和工程点。

## 复现实验说明

这个仓库是项目记录版，不是完整的 clone-and-run 发布版。原始实验依赖 TronCamp Mani starter package，并在本地安装了 `external/robotwin_local` 作为 RoboTwin 运行环境。

本地完整流程使用过的入口形式如下：

```bash
make collect TRACK=T1 GPU=0
make process TRACK=T1
make train TRACK=T1 SEED=0 GPU=0
python starter/eval_local.py --track T1 --ckpt-dir <ACT_CKPT_DIR>
python starter/record_policy_rollout.py --track T1 --ckpt-dir <ACT_CKPT_DIR> --seed 20260631
make submit TRACK=T1
```

提交时使用 token 文件或环境变量，不在代码或 README 中保存 token。
