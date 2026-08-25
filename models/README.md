# models/

通过 **git submodule** 引入核心代码，保持本仓库轻量（不把模型/权重入 git）。

## git submodule add 模板

```bash
cd "Comic Video Diffusion"

# 官方 AnimateDiff
git submodule add https://github.com/guoyww/AnimateDiff.git models/animatediff

# 或 LoRA 训练 fork（推荐，面向 LoRA 微调）
git submodule add https://github.com/Shelly-icecream/AnimateDiff-Motion-Module-LoRA.git models/animatediff

# 初始化/更新
git submodule update --init --recursive
```

## `sd_base/`（不入库）

基础 SD 1.5 / 动漫风格 checkpoint（Anything-V5、Realistic_Vision_V5.1）下载后软链到此，
**不提交到 git**（体积大，已被 `.gitignore` 排除）。
