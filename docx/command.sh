
  cd /diff/wallx_workspace/openpi
  source .venv-openpi/bin/activate
  HF_LEROBOT_HOME=/diff/wallx_workspace/wallx_data_ckp/real_world/lerobot \
  uv run --active examples/wallx/convert_wallx_data_to_lerobot.py \
    --raw-dir /diff/wallx_workspace/wallx_data_ckp/real_world \
    --repo-id wallx/real_world_test \
    --max-episodes 2

  完整转换：

  HF_LEROBOT_HOME=/diff/wallx_workspace/wallx_data_ckp/real_world/lerobot \
  uv run --active examples/wallx/convert_wallx_data_to_lerobot.py \
    --raw-dir /diff/wallx_workspace/wallx_data_ckp/real_world \
    --repo-id wallx/real_world_960x540 \
    --image-width 960 \
    --image-height 540 \
    --resize-mode pad \
    --image-writer-processes 8 \
    --image-writer-threads 16

HF_LEROBOT_HOME=/diff/wallx_workspace/xyx1/wallx_data_ckp/real_world/lerobot \
WANDB_MODE=offline \
WANDB_DIR=/diff/wallx_workspace/openpi/wandb \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --active scripts/train.py pi05_wallx \
  --exp-name wallx_960x540_delta_ah10_bs16_lr5e-5_20k \
  --checkpoint-base-dir /diff/wallx_workspace/openpi/checkpoints \
  --assets-base-dir /diff/wallx_workspace/openpi/assets \
  --data.repo-id wallx/real_world_960x540 \
  --data.use-delta-joint-actions \
  --model.action-dim 32 \
  --model.action-horizon 10 \
  --model.max-token-len 200 \
  --batch-size 16 \
  --num-train-steps 20000 \
  --num-workers 8 \
  --lr-schedule.warmup-steps 1000 \
  --lr-schedule.peak-lr 5e-5 \
  --lr-schedule.decay-steps 20000 \
  --lr-schedule.decay-lr 5e-6 \
  --optimizer.b1 0.9 \
  --optimizer.b2 0.95 \
  --optimizer.eps 1e-8 \
  --optimizer.weight-decay 1e-10 \
  --optimizer.clip-gradient-norm 1.0 \
  --ema-decay 0.99 \
  --log-interval 100 \
  --save-interval 2000 \
  --keep-period 5000 \
  --seed 42 \
  --fsdp-devices 1 \
  --weight-loader.params-path /diff/wallx_workspace/openpi/checkpoints/pi05_base/params \
  --overwrite

HF_LEROBOT_HOME=/diff/wallx_workspace/xyx1/wallx_data_ckp/real_world/lerobot \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1 \
uv run --active scripts/serve_wallx_policy.py \
  --host 127.0.0.1 \
  --port 8765 \
  --checkpoint-dir /diff/wallx_workspace/openpi/checkpoints/pi05_wallx/wallx_960x540_delta_ah10_bs16_lr5e-5_20k/19999

ssh -N -L 8765:127.0.0.1:8765 ubuntu@1.13.198.68