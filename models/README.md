# models/

通过 **git submodule** 引入核心代码，保持本仓库轻量（不把模型/权重入 git）。

- **`animatediff/`** → `git submodule add https://github.com/guoyww/AnimateDiff.git`
  或 LoRA 训练 fork：`git submodule add https://github.com/Shelly-icecream/AnimateDiff-Motion-Module-LoRA.git animatediff`
- **`sd_base/`** → 基础 SD 1.5 / 动漫风格 checkpoint（如 Anything-V5、Realistic_Vision_V5.1）下载后软链到此，
  **不提交到 git**（体积大）。

初始化：
```bash
git submodule update --init --recursive
```
