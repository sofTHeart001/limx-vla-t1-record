# InterACT 改造说明

本项目以 `policies/inter-act/` 作为主要算法展示目录，用于验证 InterACT 风格的双臂协同策略结构。为了兼容 TronCamp 官方工程，底层复现实验仍会使用 `policy/ACT` 这样的官方目录名，但项目叙事统一围绕 InterACT 展开。

核心目标：

- 继续复用官方 HDF5 数据格式、三相机 RGB 输入和 16 维动作接口。
- 把左右臂和图像观测拆成不同 segment，显式建模双臂协同。
- 用 hierarchical attention encoder 融合左右臂状态和视觉信息。
- 用 multi-arm decoder 分别生成左右臂动作，再通过同步注意力交换信息。
- 保持独立训练、评估和 checkpoint 目录，方便和官方兼容接口做对照。

## 目录位置

```text
policies/inter-act/
  inter_act_policy.py                  # 训练/部署 wrapper
  imitate_episodes.py                  # 训练入口，兼容官方 HDF5 数据集
  train.sh                             # InterACT 训练脚本
  process_data.sh                      # 数据处理脚本
  deploy_policy.py                     # 部署/评估接口
  deploy_policy.yml                    # 推理配置
  detr/models/interact_vae.py          # InterACT 主干模型
```

主干模型文件：

```text
policies/inter-act/detr/models/interact_vae.py
```

训练 wrapper：

```text
policies/inter-act/inter_act_policy.py
```

## 输入输出约定

当前 Tron2 双臂动作维度是 16：

```text
qpos/action: 16
left arm + left gripper:   0:8
right arm + right gripper: 8:16
```

InterACT 要求状态/动作维度可以均分成左右两臂，因此代码中使用：

```python
arm_dim = state_dim // 2
```

模型输入和官方数据接口保持一致：

```text
qpos:   [B, 16]
image:  [B, 3, 3, 480, 640]
action: [B, T, 16]
```

模型输出：

```text
actions_hat: [B, chunk_size, 16]
```

其中 `chunk_size=50`，表示一次预测未来 50 个动作。

## 为什么使用 InterACT

标准 action chunking 模仿学习训练流程稳定、简单，适合作为官方兼容接口。它通常把图像特征和 qpos 一起送进 Transformer，然后直接输出 action chunk。

但在更复杂的 T3/T4 中，任务会更依赖：

- 左右臂分工
- 左右臂同步
- 图像目标和手部状态的跨模态融合
- 长序列动作中的阶段切换

InterACT 的动机是把这些结构显式拆出来：先分别理解左臂、右臂和图像，再做跨 segment 信息交换，最后分别解码左右臂动作。

## 整体架构

当前实现可以概括为：

```text
qpos[0:8]      -> left arm segment
qpos[8:16]     -> right arm segment
3-camera RGB   -> image segment

left / right / image segments
        |
        v
Hierarchical Attention Encoder
  1. segment-wise attention
  2. cross-segment CLS attention
        |
        v
left context / right context
        |
        v
Multi-Arm Decoder
  1. left/right pre decoder
  2. synchronization self-attention
  3. left/right post decoder
        |
        v
left action head + right action head
        |
        v
[B, chunk_size, 16]
```

## 1. Arm Segment

左右臂状态先被拆开：

```python
left_qpos = qpos[:, :arm_dim]
right_qpos = qpos[:, arm_dim:]
```

每个 arm segment 包含两类 token：

- CLS tokens：表示该臂的全局摘要。
- joint tokens：每个关节/夹爪数值经过 `joint_proj` 投影成 hidden token。

对应代码：

```python
cls_tokens = cls_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
joint_tokens = self.joint_proj(qpos.unsqueeze(-1))
segment = torch.cat([cls_tokens, joint_tokens], dim=1)
```

默认参数：

```text
num_cls_tokens_arm = 3
arm_dim = 8
每个 arm segment 长度 = 3 + 8 = 11
```

## 2. Image Segment

图像输入包含三路相机：

```text
cam_high
cam_right_wrist
cam_left_wrist
```

每路图像经过 ResNet18 backbone，再用 `1x1 Conv` 投影到 `hidden_dim=512`。为了让模型区分不同相机，特征会加上 camera embedding。

```python
features = backbone(image[:, cam_id])
features = image_input_proj(features)
features = features + camera_embed[cam_id]
```

三路相机特征会在空间维度拼接，然后 flatten 成 image tokens。image segment 前面也加入 image CLS tokens。

默认参数：

```text
num_cls_tokens_image = 3
hidden_dim = 512
```

## 3. Hierarchical Attention Encoder

encoder 分两层含义：

### Segment-wise Attention

每个 segment 先在自己内部做 self-attention：

```text
left segment  -> left updated
right segment -> right updated
image segment -> image updated
```

这一步让模型先分别理解：

- 左臂当前状态
- 右臂当前状态
- 图像场景和物体位置

### Cross-segment CLS Attention

然后取出每个 segment 的 CLS tokens，拼起来做跨 segment attention：

```text
[left CLS, right CLS, image CLS] -> cross-segment attention
```

这一步让左右臂和图像摘要交换信息。更新后的 CLS token 再写回各自 segment。

默认重复次数：

```text
num_blocks = 3
```

所以整体是 3 轮：

```text
segment 内部更新 -> CLS 跨 segment 同步 -> 写回 segment
```

## 4. Multi-Arm Decoder

decoder 的目标是分别生成左臂动作和右臂动作，但中间允许两臂同步。

### Left / Right Pre Decoder

先准备长度为 `chunk_size` 的 decoder queries：

```text
decoder_input: [chunk_size, B, hidden_dim]
```

左右臂各自用自己的 decoder 读取 context：

```text
left decoder  reads left_context
right decoder reads right_context
```

其中：

```text
left_context  = left_out + right_cls + image_out
right_context = left_cls + right_out + image_out
```

这样设计的含义是：

- 左臂主要看左臂完整状态，同时看右臂摘要和图像。
- 右臂主要看右臂完整状态，同时看左臂摘要和图像。

### Synchronization Self-Attention

左右臂 pre decoder 输出后拼在一起：

```text
[left_output, right_output]
```

然后通过 sync self-attention 让两臂动作计划交换信息。

默认参数：

```text
n_sync_decoder_layers = 1
```

这一步是 InterACT 分支里最核心的“双臂同步”结构。

### Left / Right Post Decoder

同步后再拆回左右臂，分别经过 post decoder，最终输出：

```text
left_actions:  [B, chunk_size, 8]
right_actions: [B, chunk_size, 8]
```

最后拼成：

```text
actions_hat: [B, chunk_size, 16]
```

## 训练损失

当前 InterACT active path 不使用 CVAE latent，也不计算 KL。训练时使用 masked L1 loss：

```python
valid = ~is_pad.unsqueeze(-1)
all_l1 = F.l1_loss(actions, a_hat, reduction="none")
l1 = (all_l1 * valid).mean()
loss = l1
```

代码中也记录了 `l2` 作为观测指标，但优化目标是 `l1`。

这样做的原因是：当前目标先复现 InterACT 风格主干，并保持和官方数据管线兼容。CVAE 版本可以作为后续独立 ablation，而不是混入第一版 InterACT 主线。

## InterACT 核心差异

| 对比项 | 官方兼容接口 | InterACT 分支 |
|---|---|---|
| 输入数据 | HDF5 + 三相机 RGB + qpos | 相同 |
| 动作维度 | 16 | 16 |
| action chunk | 50 | 50 |
| 主干结构 | 标准 action chunking Transformer | HAE + Multi-Arm Decoder |
| 左右臂建模 | 动作维度整体建模 | 显式拆成 left/right segment |
| 图像建模 | backbone feature 进入 Transformer | 独立 image segment + image CLS |
| 双臂同步 | 依赖 Transformer 自学习 | decoder 中显式 sync self-attention |
| loss | L1 + KL | masked L1 |
| checkpoint 目录 | `act_ckpt/` | `inter_act_ckpt/` |

## 当前默认超参

```text
state_dim = 16
chunk_size = 50
hidden_dim = 512
dim_feedforward = 3200
nheads = 8
num_blocks = 3
num_cls_tokens_arm = 3
num_cls_tokens_image = 3
n_pre_decoder_layers = 2
n_post_decoder_layers = 2
n_sync_decoder_layers = 1
lr = 1e-5
batch_size = 8
num_epochs = 6000
```

这些参数在：

```text
policies/inter-act/train.sh
policies/inter-act/deploy_policy.yml
```

## 如何运行

公开仓库里的 InterACT 代码位于：

```text
policies/inter-act/
```

本地训练前需要复制到 RoboTwin policy 目录：

```bash
rsync -a --delete policies/inter-act/ external/robotwin_local/policy/inter-act/
```

处理数据：

```bash
make process-interact TRACK=T2
```

训练：

```bash
make train-interact TRACK=T2 SEED=0 GPU=0
```

如果不用 Makefile，也可以直接运行：

```bash
cd external/robotwin_local/policy/inter-act
bash process_data.sh grab_roller grab_roller_600ep 600
bash train.sh grab_roller grab_roller_600ep 600 0 0
```

训练输出：

```text
external/robotwin_local/policy/inter-act/inter_act_ckpt/
```

## 当前适配状态

- 已对齐官方 HDF5 数据读取方式。
- 已对齐三相机 RGB 输入顺序。
- 已适配 Tron2 的 16 维双臂动作。
- 已保持独立 checkpoint 目录，避免覆盖官方兼容训练结果。
- 已做过真实 T2 HDF5 小 batch smoke test，确认可以前向和计算 loss。

## 后续实验计划

InterACT 是本项目后续 T3/T4 长序列任务的主要候选结构。后续优先验证：

1. 在 T2 上训练 InterACT，确认 loss 曲线和 rollout 能闭环。
2. 对比 InterACT、视觉增强和不同 action chunking 设置的公开 seed 成功率。
3. 在 T3/T4 长序列任务上观察 InterACT 是否能改善双臂同步和阶段切换。
4. 如果出现明显动作平均化，再单独加入 `InterACT-CVAE` 做 ablation。
5. 点云、SAC/PPO residual 暂不进入当前实现，避免一次引入太多变量。

## 面试讲法

可以把这部分概括为：

> 我把项目主线放在 InterACT 风格的分层解耦策略上。它继续复用官方 HDF5 数据和部署接口，但把 16 维双臂状态拆成左右臂 segment，并把三相机视觉建成 image segment。编码阶段先做 segment 内 attention，再用 CLS token 做跨 segment 信息交换；解码阶段左右臂先分别生成动作计划，中间通过 sync attention 做双臂同步，最后拼回 16 维 action chunk。这样保留了官方接口的工程稳定性，同时为 T3/T4 更复杂的双臂长序列任务准备了结构化策略模型。
