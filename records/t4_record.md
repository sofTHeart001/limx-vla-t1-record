# T4 阶段记录：stack_bowls_three

这是 TronCamp Mani T1-T4 综合项目中的第四阶段记录，任务是 T4 `stack_bowls_three`。这一阶段更接近完整长序列双臂操作：需要处理三只碗的抓取、搬运、对齐和堆叠，任务步数更长，策略对相机扰动和动作连续性的敏感度也更高。

## 任务信息

- 赛道：T4
- 任务：`stack_bowls_three`
- 策略：InterACT-compatible policy
- 本地演示数据：600 episodes
- 训练 seed：0
- 训练配置：`chunk_size=100`、`hidden_dim=512`、`dim_feedforward=3200`、`kl_weight=10`
- 优化配置：`batch_size=24`、`lr=4e-5`、`lr_backbone=4e-6`、`weight_decay=1e-4`
- 训练技巧：`warmup_steps=300`、`grad_clip=0.1`、`val_ratio=0.1`、`ACT_AUG=1`
- 数据处理：`PROCESS_RESUME=1`、`PROCESS_DELETE_SOURCE=1`、`ACT_PROCESS_JPEG_QUALITY=45`
- 训练轮数：1500 epochs
- 最佳验证损失：0.036161，出现在 epoch 1100
- 官方提交：`#695 T4`

## 采集与处理配置

T4 使用 600 条成功轨迹，并保留轻到中等的域随机化：

```yaml
episode_num: 600
domain_randomization:
  random_background: false
  cluttered_table: false
  random_head_camera_dis: 0.015
  random_table_height: 0.003
  random_light: true
  crazy_random_light_rate: 0.08
```

由于 T4 数据量和磁盘压力明显更高，处理阶段采用边处理边删除源数据的方式：

```bash
PROCESS_RESUME=1 PROCESS_DELETE_SOURCE=1 make process TRACK=T4
```

## T4 Policy 自主执行示例

下面是训练出的 InterACT-compatible policy 在 T4 `stack_bowls_three` 上的闭环 rollout 片段。该视频不是专家采集轨迹，而是 checkpoint 在 clean eval 配置下自主执行，公开 seed 为 `20260629`。

![T4 policy rollout](../media/t4_policy_rollout_seed_20260629.gif)

原始 MP4：[`media/t4_policy_rollout_seed_20260629.mp4`](../media/t4_policy_rollout_seed_20260629.mp4)

## 复盘

T4 的主要目标是在截止前完成完整闭环：采集 600 条轨迹、压缩处理数据、完成 InterACT-compatible 训练、生成策略 rollout 并提交官方评测。最终保持官方推理接口兼容，避免线上加载 checkpoint 时出现结构不匹配。

T4 使用更高学习率和较少 epoch 来换取截止时间内的完整产出，同时保留 `chunk_size=100` 和轻量数据增强来适配长序列动作。公开仓库不包含 T4 的 HDF5 数据、processed data、checkpoint 和本地完整日志，只保留配置、脚本、视频和阶段记录。
