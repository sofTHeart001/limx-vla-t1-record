# 复现指南

这份文档用于从一个干净环境复现本项目的 T1-T4 流程，并说明公开仓库缺少哪些本地模块。公开仓库主要用于项目展示和代码记录，不包含大体积仿真环境、采集数据和训练权重。

## 0. 公开仓库缺少什么

克隆本仓库后，以下内容需要你在本地准备：

| 模块 | 是否在 GitHub | 本地路径 | 说明 |
|---|---:|---|---|
| RoboTwin / TronCamp 官方运行环境 | 否 | `external/robotwin_local/` | 仿真环境、任务、官方 action-chunking 训练栈、cuRobo、资产文件 |
| 采集得到的 HDF5 数据 | 否 | `external/robotwin_local/data/` | 需要自己通过专家脚本采集 |
| InterACT-compatible processed data | 否 | `external/robotwin_local/policy/ACT/processed_data/` | 由 `make process` 生成 |
| checkpoint | 否 | `external/robotwin_local/policy/ACT/act_ckpt/` | 由 `make train` 生成 |
| 提交 token | 否 | 环境变量或 token 文件 | 不要写入 GitHub |

因此，GitHub 仓库不能单独直接运行；需要把官方 starter package 里的 `robotwin_local` 放到 `external/robotwin_local/`。

## 1. 机器要求

建议环境：

- Ubuntu 22.04
- NVIDIA GPU，24GB 显存更稳
- NVIDIA driver 可用，`nvidia-smi` 能正常输出
- conda / miniconda
- Python 3.10
- ffmpeg
- 网络可访问 pip / GitHub；离线环境需要提前准备 wheels

先做基础检查：

```bash
cd hd-act-bimanual-imitation-learning
make check
```

如果 `make check` 提示缺少 `external/robotwin_local`，先完成下一步。

## 2. 放入官方运行环境

从 TronCamp 官方 starter package 中取出 RoboTwin 运行目录，放到：

```text
external/robotwin_local/
```

放好后应能看到：

```text
external/robotwin_local/script/_install.sh
external/robotwin_local/policy/ACT/train.sh
external/robotwin_local/policy/ACT/process_data.sh
external/robotwin_local/task_config/adjust_bottle_200ep.yml
```

再次检查：

```bash
make check
```

## 3. 创建 conda 环境

本项目统一使用 `troncamp_env`：

```bash
make env
```

等价于：

```bash
conda create -y -n troncamp_env python=3.10
conda activate troncamp_env
python -m pip install --upgrade pip
python -m pip install "setuptools<81"
```

后续所有采集、处理、训练、评估命令都在这个环境中执行。

## 4. 安装 RoboTwin / InterACT-compatible / cuRobo 依赖

```bash
make install
```

这个命令会做几件事：

1. 激活 `troncamp_env`
2. 安装 RoboTwin 自带依赖
3. 安装 `setup/requirements.txt` 里的 InterACT-compatible 训练依赖
4. 以 editable 方式安装 `external/robotwin_local/envs/curobo`
5. 还原 `__KIT_ROOT__` 路径占位符
6. 运行 `setup/env_check.py`

安装后单独检查：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate troncamp_env
python setup/env_check.py
```

## 5. 准备任务配置

T1 默认使用：

```text
adjust_bottle_200ep.yml
```

本项目当前 T2 使用：

```text
grab_roller_600ep.yml
```

本项目当前 T3 使用：

```text
stack_bowls_two_600ep.yml
```

本项目当前 T4 使用：

```text
stack_bowls_three_600fast.yml
```

如果官方包里没有 T2-T4 采集配置，可以先生成模板：

```bash
make configs
```

仓库中已经保留了 T2/T3 当前使用过的采集配置：

```text
configs/task_config_grab_roller_600ep.yml
configs/task_config_stack_bowls_two_600ep.yml
configs/task_config_stack_bowls_three_600fast.yml
```

`make configs` 会优先把这些项目配置复制到：

```text
external/robotwin_local/task_config/
```

T2 关键设置：

```yaml
episode_num: 600
domain_randomization:
  random_background: false
  cluttered_table: false
  random_head_camera_dis: 0.01
  random_table_height: 0
  random_light: true
  crazy_random_light_rate: 0.05
camera:
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  depth: false
  pointcloud: false
  qpos: true
```

T3 关键设置：

```yaml
episode_num: 600
domain_randomization:
  random_background: false
  cluttered_table: false
  random_head_camera_dis: 0.015
  random_table_height: 0.003
  random_light: true
  crazy_random_light_rate: 0.08
camera:
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  depth: false
  pointcloud: false
  qpos: true
```

T4 关键设置：

```yaml
episode_num: 600
domain_randomization:
  random_background: false
  cluttered_table: false
  random_head_camera_dis: 0.015
  random_table_height: 0.003
  random_light: true
  crazy_random_light_rate: 0.08
camera:
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  depth: false
  pointcloud: false
  qpos: true
```

## 6. 采集数据

T1：

```bash
make collect TRACK=T1 GPU=0
```

T2：

```bash
make collect TRACK=T2 GPU=0
```

T3：

```bash
make collect TRACK=T3 GPU=0
```

T4：

```bash
make collect TRACK=T4 GPU=0
```

采集结果会写到：

```text
external/robotwin_local/data/<task>/<task_config>/
```

例如 T2：

```text
external/robotwin_local/data/grab_roller/grab_roller_600ep/
```

T3 对应：

```text
external/robotwin_local/data/stack_bowls_two/stack_bowls_two_600ep/
```

T4 对应：

```text
external/robotwin_local/data/stack_bowls_three/stack_bowls_three_600fast/
```

该目录里通常包含：

```text
data/          # episode*.hdf5
video/         # episode*.mp4
seed.txt       # 成功 seed 列表
scene_info.json
```

如果中途断掉，采集脚本会读取已有 `seed.txt` 和 `data/episode*.hdf5`，一般可以继续跑。

## 7. 处理成 InterACT-compatible 数据格式

T1：

```bash
make process TRACK=T1
```

T2：

```bash
make process TRACK=T2
```

T3：

```bash
make process TRACK=T3
```

T4：

```bash
PROCESS_RESUME=1 PROCESS_DELETE_SOURCE=1 make process TRACK=T4
```

T4 数据量较大，建议处理时开启 `PROCESS_DELETE_SOURCE=1`，边生成 InterACT-compatible processed data 边删除 raw HDF5，降低磁盘峰值占用。

输出位置：

```text
external/robotwin_local/policy/ACT/processed_data/
```

T2 对应：

```text
external/robotwin_local/policy/ACT/processed_data/sim-grab_roller/grab_roller_600ep-600/
```

T3 对应：

```text
external/robotwin_local/policy/ACT/processed_data/sim-stack_bowls_two/stack_bowls_two_600ep-600/
```

T4 对应：

```text
external/robotwin_local/policy/ACT/processed_data/sim-stack_bowls_three/stack_bowls_three_600fast-600/
```

处理后会同时更新：

```text
external/robotwin_local/policy/ACT/SIM_TASK_CONFIGS.json
```

## 8. 训练 InterACT-compatible policy

T1：

```bash
make train TRACK=T1 SEED=0 GPU=0
```

T2：

```bash
make train TRACK=T2 SEED=0 GPU=0
```

T3：

```bash
make train TRACK=T3 SEED=0 GPU=0
```

T4：

```bash
make train TRACK=T4 SEED=0 GPU=0
```

checkpoint 输出位置：

```text
external/robotwin_local/policy/ACT/act_ckpt/act-<task>/<task_config>-<num>/
```

T2 对应：

```text
external/robotwin_local/policy/ACT/act_ckpt/act-grab_roller/grab_roller_600ep-600/
```

T3 对应：

```text
external/robotwin_local/policy/ACT/act_ckpt/act-stack_bowls_two/stack_bowls_two_600ep-600/
```

T4 对应：

```text
external/robotwin_local/policy/ACT/act_ckpt/act-stack_bowls_three/stack_bowls_three_600fast-600/
```

训练完成后至少应有：

```text
policy_best.ckpt
policy_last.ckpt
dataset_stats.pkl
```

## 9. T2-T4 视觉增强训练

当前 T2-T4 都使用只增强训练集的 InterACT-compatible 训练。增强由环境变量控制：

```bash
ACT_AUG=1
```

`scripts/30_train_one.sh` 已经为 T2/T3/T4 默认开启 `ACT_AUG=1`，通常直接运行：

```bash
make train TRACK=T2 SEED=0 GPU=0
make train TRACK=T3 SEED=0 GPU=0
make train TRACK=T4 SEED=0 GPU=0
```

注意：增强只应该用于训练，不用于评估或提交。

## 10. T3 训练配置

T3 用的是更长动作 chunk 和更多轨迹。`scripts/30_train_one.sh` 已为 `TRACK=T3` 设置默认超参：

```text
ACT_BATCH_SIZE=24
ACT_CHUNK_SIZE=100
ACT_LR=2e-5
ACT_LR_BACKBONE=3e-6
ACT_WEIGHT_DECAY=1e-4
ACT_WARMUP_STEPS=500
ACT_GRAD_CLIP=0.1
ACT_VAL_RATIO=0.1
ACT_KL_WEIGHT=10
ACT_NUM_EPOCHS=4000
ACT_SAVE_FREQ=1000
ACT_AUG=1
```

本次 T3 训练结果：

```text
Best val loss: 0.017439 @ epoch 3286
Local public eval: 72% on 100 public seeds
```

T3 的 checkpoint 结构依赖 `chunk_size=100`，评估时需要匹配部署配置。仓库里保留了：

```text
configs/deploy_t3.yml
```

`make eval-local TRACK=T3` 会自动把它复制到：

```text
external/robotwin_local/policy/ACT/deploy_t3.yml
```

## 11. T4 快速训练配置

T4 最终使用 600 条轨迹和更短训练轮数，目的是在截止时间内完成完整训练、演示视频和线上提交。`scripts/30_train_one.sh` 已为 `TRACK=T4` 设置默认超参：

```text
ACT_BATCH_SIZE=24
ACT_CHUNK_SIZE=100
ACT_LR=4e-5
ACT_LR_BACKBONE=4e-6
ACT_WEIGHT_DECAY=1e-4
ACT_WARMUP_STEPS=300
ACT_GRAD_CLIP=0.1
ACT_VAL_RATIO=0.1
ACT_KL_WEIGHT=10
ACT_NUM_EPOCHS=1500
ACT_SAVE_FREQ=500
ACT_AUG=1
ACT_PROCESS_JPEG_QUALITY=45
```

本次 T4 训练结果：

```text
Best val loss: 0.036161 @ epoch 1100
Official submission: #695 T4
```

## 12. 本地评估

本地公开 100 seed 评估：

```bash
make eval-local TRACK=T1
make eval-local TRACK=T2
make eval-local TRACK=T3
make eval-local TRACK=T4
```

如果要评估指定 checkpoint 目录：

```bash
python starter/eval_local.py \
  --track T2 \
  --ckpt-dir external/robotwin_local/policy/ACT/act_ckpt/act-grab_roller/grab_roller_600ep-600
```

T3 指定 checkpoint 目录评估：

```bash
python starter/eval_local.py \
  --track T3 \
  --ckpt-dir external/robotwin_local/policy/ACT/act_ckpt/act-stack_bowls_two/stack_bowls_two_600ep-600 \
  --deploy-config policy/ACT/deploy_t3.yml
```

T4 指定 checkpoint 目录评估：

```bash
python starter/eval_local.py \
  --track T4 \
  --ckpt-dir external/robotwin_local/policy/ACT/act_ckpt/act-stack_bowls_three/stack_bowls_three_600fast-600
```

评估目录必须包含：

```text
policy_last.ckpt
dataset_stats.pkl
```

提交脚本会自动把你指定的 `policy_best.ckpt` 打包成评测端读取的 `policy_last.ckpt`。

## 13. 录制展示视频

录制 policy rollout：

```bash
python starter/record_policy_rollout.py \
  --track T1 \
  --ckpt-dir external/robotwin_local/policy/ACT/act_ckpt/act-adjust_bottle/adjust_bottle_200ep-200 \
  --out-dir /tmp/t1_policy_rollout_video \
  --max-seeds 20 \
  --stop-on-success
```

采集视频在：

```text
external/robotwin_local/data/<task>/<task_config>/video/
```

可以把小体积成功视频复制到：

```text
media/
```

再用 ffmpeg 生成 GIF：

```bash
ffmpeg -y -i input.mp4 -vf "fps=12,scale=320:-1:flags=lanczos" output.gif
```

## 14. 使用 InterACT

公开仓库里的 InterACT 代码在：

```text
policies/inter-act/
```

要在本地 RoboTwin 中训练，需要复制到：

```text
external/robotwin_local/policy/inter-act/
```

命令：

```bash
rsync -a --delete policies/inter-act/ external/robotwin_local/policy/inter-act/
```

然后处理数据：

```bash
make process-interact TRACK=T2
```

训练：

```bash
make train-interact TRACK=T2 SEED=0 GPU=0
```

如果当前 Makefile 没有 `process-interact` 或 `train-interact` 目标，可以直接进入目录运行：

```bash
cd external/robotwin_local/policy/inter-act
bash process_data.sh grab_roller grab_roller_600ep 600
bash train.sh grab_roller grab_roller_600ep 600 0 0
```

InterACT 输出目录：

```text
external/robotwin_local/policy/inter-act/inter_act_ckpt/
```

## 15. 提交

准备 token：

```bash
export TRONCAMP_TOKEN="你的 token"
```

或使用 token 文件：

```bash
export TRONCAMP_TOKEN_FILE=/path/to/token.txt
```

提交 T1：

```bash
make submit TRACK=T1
```

提交 T2：

```bash
make submit TRACK=T2
```

提交 T3 前要确认官方赛道已经解锁，并确认代码包内的推理配置与 checkpoint 结构一致。当前 T3 checkpoint 使用 `chunk_size=100`，本地评估使用 `configs/deploy_t3.yml`。

提交 T4：

```bash
make submit TRACK=T4
```

T2/T3/T4 提交时脚本会附带 `external/robotwin_local` 代码包。

## 16. 常见问题

### `nvidia-smi` 不可用

先修 NVIDIA driver。训练和评估都需要 GPU。

### 缺少 `external/robotwin_local`

说明只克隆了展示仓库，还没有放入官方 starter package 的 RoboTwin 运行环境。

### `ModuleNotFoundError: einops`

重新执行：

```bash
make install
```

或：

```bash
pip install -r setup/requirements.txt
```

### cuRobo 版本不对

重新安装内嵌 cuRobo：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate troncamp_env
SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0 \
  python -m pip install -e external/robotwin_local/envs/curobo --no-build-isolation
```

### `__KIT_ROOT__` 路径残留

执行：

```bash
make restore-paths
```

### 评估找不到 `policy_last.ckpt`

本地评估目录需要 `policy_last.ckpt`。如果想评估 best，可以临时建目录：

```bash
mkdir -p /tmp/t2_best_eval
cp <ckpt_dir>/dataset_stats.pkl /tmp/t2_best_eval/
cp <ckpt_dir>/policy_best.ckpt /tmp/t2_best_eval/policy_last.ckpt
python starter/eval_local.py --track T2 --ckpt-dir /tmp/t2_best_eval
```
