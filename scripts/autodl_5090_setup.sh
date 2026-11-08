#!/usr/bin/env bash
# AutoDL · RTX 5090（Blackwell / sm_120 / 32GB）环境自举
#
# 用法（在项目根目录执行）：
#   bash scripts/autodl_5090_setup.sh
#   WORK=/root/autodl-tmp bash scripts/autodl_5090_setup.sh   # 自定义数据盘目录
#
# 做四件事：
#   1. 查 GPU 与 torch 的算力是否匹配 sm_120（5090 必须 CUDA 12.8+ 的 torch）
#   2. 装训练/出片依赖（diffusers / accelerate / safetensors / opencv / imageio-ffmpeg …）
#   3. 拉基础模型（SD1.5 + AnimateDiff v1_5_2 运动适配器），走 hf-mirror
#   4. 跑一次出片冒烟测试，打印耗时与峰值显存
set -euo pipefail

WORK="${WORK:-/root/autodl-tmp}"
PY="${PY:-python}"
cd "$(dirname "$0")/.."
mkdir -p "$WORK/models" "$WORK/runs" samples results

echo "== 1) GPU 检查 =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "!! 没有 nvidia-smi：这台不是 GPU 实例，脚本到此为止"; exit 1; }

"$PY" - <<'PY'
import sys
try:
    import torch
except Exception as exc:  # noqa: BLE001
    print(f"torch 导入失败：{exc}")
    sys.exit(2)
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    print("torch 看不到 CUDA —— 需要装 CUDA 版 torch（见下一步）")
    sys.exit(3)
cap = torch.cuda.get_device_capability(0)
print(f"device={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]}")
# 5090 = sm_120（Blackwell）。torch 必须用 cu128+ 的构建才能认这块卡
if cap[0] >= 12 and (torch.version.cuda is None or tuple(map(int, torch.version.cuda.split("."))) < (12, 8)):
    print("!! 5090 需要 CUDA 12.8+ 的 torch，当前构建不匹配")
    sys.exit(4)
PY
TORCH_RC=$?

if [ "$TORCH_RC" -ge 2 ]; then
  echo "== 2) 重装 Blackwell 可用的 torch（cu128 稳定轮子） =="
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch torchvision torchaudio
  "$PY" -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
fi

echo "== 3) 依赖 =="
"$PY" -m pip install --upgrade \
    "diffusers>=0.35" "transformers>=4.44" "accelerate>=1.0" safetensors \
    opencv-python-headless imageio-ffmpeg tensorboard pyyaml tqdm Pillow imageio
# 注意：sageattention 对 sm_120 没有稳定轮子，这里不装；用默认 SDPA 即可

echo "== 4) 基础模型（hf-mirror） =="
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [ ! -d "$WORK/models/sd15" ]; then
  "$PY" -m pip install --upgrade "huggingface_hub[cli]"
  huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
      --local-dir "$WORK/models/sd15" --exclude "*.ckpt" "*.bin"
fi
if [ ! -d "$WORK/models/adapter-v15-2" ]; then
  huggingface-cli download Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
      --local-dir "$WORK/models/adapter-v15-2"
fi

echo "== 5) 冒烟测试（8 帧 / 256 分辨率 / 8 步） =="
/usr/bin/time -v "$PY" animatediff_infer.py \
  --model_id_or_path "$WORK/models/sd15" \
  --motion_adapter_id_or_path "$WORK/models/adapter-v15-2" \
  --prompt "a comic style character walking, 2d cel animation, clean lineart" \
  --num_frames 8 --resolution 256 --num_inference_steps 8 --seed 1234 \
  --output_dir samples/smoke 2>&1 | tail -n 25 || true

echo
echo "== 完成。下一步： =="
cat <<EOF
  # P0 三组对照（判断「接错」还是「没训够」）
  python scripts/diagnose_output.py \\
    --model_path $WORK/models/sd15 \\
    --motion_adapter $WORK/models/adapter-v15-2 \\
    --ckpt <你的 checkpoint.safetensors> \\
    --prompt "a comic style character walking in the rain, 2d cel animation" \\
    --resolutions 512,384 --seed 1234

  # P1 重训（512 / r16 / 2000 步）
  STEPS=2000 RESOLUTION=512 LORA_RANK=16 GRAD_ACCUM=4 \\
    DATA_DIR=$WORK/data/anime_clips VIDEO_FOLDER=$WORK/data/anime_clips_all \\
    bash scripts/train_lora.sh
EOF
