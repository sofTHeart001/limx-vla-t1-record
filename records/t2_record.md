# T2 阶段记录：grab_roller

这是 TronCamp Mani T1-T4 综合项目中的第二阶段记录，任务是 T2 `grab_roller`。该阶段开始从单臂任务进入双臂协同操作，重点验证 ACT 在更长动作序列、双腕相机和轻量随机化数据上的泛化能力。

## 任务信息

- 赛道：T2
- 任务：`grab_roller`
- 策略：ACT baseline，后续继续验证 InterACT 主干
- 最终本地演示数据：600 episodes
- 训练 seed：0
- 最终训练配置：`chunk_size=100`、`hidden_dim=512`、`dim_feedforward=3200`、`kl_weight=10`
- 优化配置：`batch_size=24`、`lr=2e-5`、`lr_backbone=3e-6`、`weight_decay=1e-4`
- 训练技巧：`warmup_steps=500`、`grad_clip=0.1`、`val_ratio=0.1`、`ACT_AUG=1`
- 最终训练轮数：4000 epochs
- 最终最佳验证损失：0.015306，出现在 epoch 3666

项目中也保留了一版 400 episodes 的早期实验记录，用于对比数据量、增强训练和 checkpoint 选择对 T2 的影响。

## Baseline 本地公开 seed 评估

评估使用官方本地评估入口，在公开 100 seed 上执行，`repeats=1`。

```json
{
  "sr": 0.53,
  "n_repeats": 1,
  "n_episodes": 100,
  "per_repeat": [0.53],
  "track": "T2"
}
```

## 最新 checkpoint 本地公开 seed 评估

早期增强训练目录下的 `policy_last.ckpt` 在公开 100 seed 上，本地评估结果为：

```json
{
  "sr": 0.64,
  "n_repeats": 1,
  "n_episodes": 100,
  "per_repeat": [0.64],
  "track": "T2"
}
```

后续又扩展到 600 条轨迹，并使用和 T3/T4 一致的 `chunk_size=100` 训练配置。公开仓库只记录结果和复现方式，不提交 `.ckpt` 权重文件。

结论：T2 对 seed 和部署配置比较敏感。扩展轨迹数量、开启轻量视觉增强、保证提交端 deploy 配置与训练结构一致，是这一阶段最关键的工程经验。

## T2 Policy 自主执行示例

下面是 InterACT 复现实验在 T2 `grab_roller` 上的闭环执行展示，使用公开 seed `20260630`，99 步完成任务。

![T2 policy rollout success](../media/t2_policy_rollout_success_seed_20260630.gif)

原始 MP4：[`media/t2_policy_rollout_success_seed_20260630.mp4`](../media/t2_policy_rollout_success_seed_20260630.mp4)

该视频来自本地 policy rollout，用于展示 T2 双臂协同抓举的策略效果，不是专家采集回放。官方提交成绩仍按 ACT checkpoint 和官方推理接口复现。

## 当前优化

当前最终版本使用 600 条轨迹和轻量视觉增强训练，用于提升模型对光照和背景变化的适应能力：

- 只对 train dataset 做亮度、对比度、饱和度扰动。
- validation dataset 保持干净，避免 best checkpoint 选择被随机增强噪声污染。
- 不把 checkpoint、processed data 或训练日志放入 GitHub。

下一步会继续对比更多演示轨迹、temporal aggregation 和 InterACT 结构在更长序列任务上的表现。

## 未公开的本地文件

以下内容没有放入 GitHub：

- 采集到的 `.hdf5` 演示数据
- ACT processed data
- `.ckpt` checkpoint
- 本地训练和评估日志
- 官方提交 token 或其他凭据
