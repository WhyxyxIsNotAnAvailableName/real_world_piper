# WallX OpenPI 开发日志

更新日期：2026-06-11

本文档记录本仓库最近围绕 WallX 真机数据、pi0.5 微调和策略服务做的开发。目标是让后续接手的人能快速知道：数据怎么转、模型怎么训、checkpoint 为什么这样保存、server 怎么启动、本地机械臂应该按什么协议请求动作。

## 当前目标

我们在 OpenPI 上为 WallX 单臂真机任务做 pi0.5 微调。模型输入是三路 RGB 图像、当前 7 维机器人状态和一条任务 instruction；模型输出是 10 步 action horizon，每步 7 维动作。

当前 4 类训练 instruction 固定为：

- `Put lemon on the beige plate`
- `Put banana on the blue bowl`
- `Put donut on the pink bowl`
- `Put avocado on the purple plate`

动作和状态约定：

- 前 6 维是机械臂关节角。
- 第 7 维是夹爪，OpenPI/WallX 数据里使用 `0=open, 1=closed`。
- 数据转换后，训练数据里的前 6 维单位是 radian。
- 本地机械臂接口读到的前 6 维是 degree，因此 server 对请求和响应做单位转换。
- 训练使用 delta joint actions 时，只对前 6 个关节做 delta；夹爪维度保持绝对二值。

## 主要文件

- `examples/wallx/convert_wallx_data_to_lerobot.py`
  - 将 WallX 原始 HDF5 episode 转成 LeRobot 数据集。
  - 负责相机字段映射、图像 resize/pad、关节 degree 到 radian、夹爪二值化。
  - 转换完成后写出 `wallx_conversion_report.jsonl`，记录每个 episode 的夹爪阈值推断信息。

- `src/openpi/policies/wallx_policy.py`
  - 定义 WallX 训练和推理共用的数据 transform。
  - 将 LeRobot 字段整理成 OpenPI 模型需要的 `state`、`image`、`image_mask`、`prompt` 和 `actions`。
  - 只保留模型动作里的前 7 维作为 WallX action。

- `src/openpi/training/config.py`
  - 新增 `LeRobotWallXDataConfig`。
  - 新增训练配置 `pi05_wallx`。
  - 默认 repo id 是 `wallx/real_world_960x540`。
  - 默认使用 `prompt_from_task=True` 和 `use_delta_joint_actions=True`。

- `scripts/serve_wallx_policy.py`
  - WallX 专用策略 server。
  - 提供 websocket 推理接口，以及 HTTP `/help`、`/healthz`。
  - 接收本地机械臂更方便发送的数据格式：关节角 degree、三路图像、夹爪 0/1、instruction。
  - 返回本地机械臂可直接使用的 10 步绝对关节目标，前 6 维 degree，第 7 维夹爪 0/1。

- `docx/command.sh`
  - 当前命令备忘，包含数据转换、训练、server 启动和 SSH 端口转发命令。

## 数据转换

原始数据来自 WallX 真机 HDF5。转换脚本默认读取：

```bash
/diff/wallx_workspace/wallx_data_ckp/real_world
```

训练时实际使用的 LeRobot 数据集位置是：

```bash
/diff/wallx_workspace/xyx1/wallx_data_ckp/real_world/lerobot/wallx/real_world_960x540
```

相机映射：

| LeRobot 字段 | 原始 HDF5 相机 |
| --- | --- |
| `base_image` | `camera_l515` |
| `left_wrist_image` | `camera_f` |
| `right_wrist_image` | `camera_r` |

状态和动作映射：

- `observations/qpos` -> `state`
- `action` -> `actions`
- 前 6 维从 degree 转成 radian。
- 夹爪按 episode 推断阈值后转成 `0=open, 1=closed`。
- 每个 episode 的任务文本由物体目录名决定。

完整 960x540 数据转换命令见 `docx/command.sh`。当前使用的关键参数是：

```bash
--repo-id wallx/real_world_960x540
--image-width 960
--image-height 540
--resize-mode pad
--image-writer-processes 8
--image-writer-threads 16
```

## 训练配置

训练入口配置名：

```bash
pi05_wallx
```

模型配置：

- pi0.5：`pi05=True`
- `action_dim=32`
- `action_horizon=10`
- 实际 WallX 动作只使用前 7 维，剩余维度由 OpenPI 模型配置保留。

训练数据 transform：

- `WallXInputs` 将三路图像放到模型 image dict：
  - `base_0_rgb`
  - `left_wrist_0_rgb`
  - `right_wrist_0_rgb`
- `WallXOutputs` 取 `actions[:, :7]`。
- `DeltaActions(make_bool_mask(6, -1))` 表示只把前 6 个关节动作变成 delta。
- `AbsoluteActions(make_bool_mask(6, -1))` 在推理输出阶段把前 6 维恢复为绝对动作。

实际训练命令见 `docx/command.sh`。这次训练的实验名是：

```bash
wallx_960x540_delta_ah10_bs16_lr5e-5_20k
```

关键训练参数：

- dataset：`wallx/real_world_960x540`
- batch size：`16`
- train steps：`20000`
- learning rate：warmup 1000，peak `5e-5`，decay 到 `5e-6`
- EMA：`0.99`
- save interval：`2000`
- keep period：`5000`
- seed：`42`
- base checkpoint：`/diff/wallx_workspace/openpi/checkpoints/pi05_base/params`

## Checkpoint 保留逻辑

本次训练目录：

```bash
/diff/wallx_workspace/openpi/checkpoints/pi05_wallx/wallx_960x540_delta_ah10_bs16_lr5e-5_20k
```

当前保留下来的 checkpoint：

```bash
10000
19999
```

虽然训练命令里 `--save-interval 2000`，但 OpenPI 训练脚本使用 Orbax checkpoint manager，并且配置里有保留策略：

- 每隔 `save_interval` 保存一次。
- 最后一步也保存，最后一步编号是 `num_train_steps - 1`，所以 20000 步训练结束保存的是 `19999`。
- `max_to_keep=1` 会只保留最新 checkpoint。
- `keep_period=5000` 会额外保留能被周期命中的 checkpoint。

因此中间的 `2000/4000/6000/8000/.../18000` 会被清理，`10000` 被 keep period 保留，`19999` 是最终最新 checkpoint。

## 策略 Server

启动示例：

```bash
HF_LEROBOT_HOME=/diff/wallx_workspace/xyx1/wallx_data_ckp/real_world/lerobot \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1 \
uv run --active scripts/serve_wallx_policy.py \
  --host 127.0.0.1 \
  --port 8765 \
  --checkpoint-dir /diff/wallx_workspace/openpi/checkpoints/pi05_wallx/wallx_960x540_delta_ah10_bs16_lr5e-5_20k/19999
```

HTTP 端点：

- `GET /healthz` 返回 `OK`
- `GET /help` 返回纯文本说明

WebSocket 端点：

```text
ws://host:port/
```

注意：`/help` 是 HTTP 端点，不是 websocket 端点。websocket 连接根路径 `/` 成功后，server 会先发一帧 metadata。

SSH 端口转发示例：

```bash
ssh -N -L 8765:127.0.0.1:8765 ubuntu@1.13.198.68
```

转发后，本地 client 连接：

```text
ws://127.0.0.1:8765/
http://127.0.0.1:8765/help
```

## WebSocket 协议

传输格式：

- websocket binary frame
- msgpack 编码
- ndarray 使用 OpenPI `msgpack_numpy` 约定

server 兼容两种 ndarray map key：

- Python/OpenPI 客户端常见的 byte-string key：`b"__ndarray__"`
- 非 Python 客户端常见的 string key：`"__ndarray__"`

ndarray map 格式：

```json
{
  "__ndarray__": true,
  "data": "raw ndarray.tobytes()",
  "dtype": "numpy dtype string, e.g. |u1 or <f4",
  "shape": [10, 7]
}
```

请求 schema：

```python
{
    "instruction": str,
    "base_image": uint8 ndarray [H, W, 3],
    "left_wrist_image": uint8 ndarray [H, W, 3],
    "right_wrist_image": uint8 ndarray [H, W, 3],
    "joint_degrees": float32 ndarray [6],
    "gripper": float,  # 0=open, 1=closed
}
```

响应 schema：

```python
{
    "actions": float32 ndarray [10, 7],
    "server_timing": {
        "infer_ms": float,
        "prev_total_ms": float,  # 第二次请求开始才会出现
    },
}
```

响应动作含义：

- `actions[:, :6]`：10 步绝对关节目标，单位 degree。
- `actions[:, 6]`：10 步夹爪命令，`0=open, 1=closed`。

server 内部处理：

1. 接收本地机械臂当前关节角 `joint_degrees`。
2. 将前 6 维 degree 转成 radian，拼上二值夹爪，形成 OpenPI `state`。
3. 调用 OpenPI trained policy。
4. OpenPI transform 已经把 delta 动作恢复为绝对动作，所以 server 不再手动加当前关节角。
5. server 将输出前 6 维 radian 转成 degree。
6. server 将输出夹爪按阈值转成 0/1。

## 本地机械臂接入约定

本地 robot client 每次请求只需要发当前观测，server 收到后立刻推理并返回动作，不在 server 端做频率控制。

本地侧应保证：

- 三个相机对应关系保持训练数据一致：
  - base -> `base_image`
  - left wrist -> `left_wrist_image`
  - right wrist -> `right_wrist_image`
- 图像最好是 RGB `uint8`，shape `[H, W, 3]`。
- 当前关节角前 6 维用 degree。
- 夹爪发 `0=open, 1=closed`。
- instruction 使用上面 4 条训练 instruction 之一。

本地机械臂控制接口如果接收绝对关节目标，可以直接使用 server 返回的 `actions[:, :6]`。夹爪底层如果需要 `0/80000`，建议在本地 robot client 里从 `0/1` 转换，不放在 server 端。

## 初始位姿参考

训练数据一共有 228 个 episode、15267 帧，fps 为 5。首帧夹爪全部是 open，即 `0.0`。

整体上初始位置大致有两个 home 簇，单位为 degree，最后一维为夹爪：

| 簇 | episode 数 | 首帧 median |
| --- | ---: | --- |
| high-J5 home | 127 | `[6.067, -0.817, -28.683, 6.672, 67.936, -6.344, 0.0]` |
| low-J5 home | 101 | `[3.921, 0.127, -29.573, 4.394, 63.260, -2.502, 0.0]` |

按任务统计的首帧 median：

| 任务 | 首帧 median |
| --- | --- |
| lemon | `[4.54, -0.82, -28.67, 6.97, 67.96, -6.42, 0.0]` |
| banana | `[8.06, -0.82, -28.71, 6.72, 67.93, -7.99, 0.0]` |
| donut | `[4.46, -0.77, -28.74, 5.23, 66.71, -4.68, 0.0]` |
| avocado | `[3.93, 0.13, -29.57, 4.29, 63.23, -2.94, 0.0]` |

真机推理前建议尽量回到对应任务接近的初始位姿，尤其注意第 5 个关节附近存在两个训练分布簇。

## 常见排查

- websocket 报 `Handshake status 200 OK`
  - 通常是连到了 HTTP `/help` 或 server 根路径被 HTTP handler 拦截。
  - 当前 server websocket 端点是 `/`，`/help` 只用于 HTTP。

- `joint_degrees` 解析时报 `float() argument must be a string or a real number, not 'dict'`
  - client 发了 ndarray map，但 server 没递归解码。
  - 当前 `serve_wallx_policy.py` 已兼容 string-key 和 byte-key ndarray map。

- `save-interval=2000` 但只看到 `10000` 和 `19999`
  - 这是 checkpoint retention 逻辑导致的正常现象，见上文 checkpoint 保留逻辑。

- server 输出看起来是 delta
  - 训练数据 transform 使用 `AbsoluteActions` 输出 transform；`create_trained_policy` 推理时会执行该 transform。
  - 当前 server 返回的是绝对关节目标，不需要本地 client 再叠加当前关节角。

## 后续建议

- 为本地 robot client 固化一个最小 msgpack websocket 示例，避免不同客户端重复踩 wire format 的坑。
- 如果后续加入更多任务，先更新转换脚本里的 `TASKS`，再更新 server 的 `WALLX_INSTRUCTIONS` 和本文档。
- 如果夹爪要改回底层 `0/80000`，建议只在本地 robot client 做映射，server 继续保持训练语义 `0/1`。
