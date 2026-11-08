# 数据：公开动画片段（不合成 IP）

> 原则：**优先用公开可下、许可清晰的片段**；不合成《冥王神话 LC》《银魂》这类受版权保护的风格，
> 也不把“截图数据”混进时序训练集。数据短缺时宁可缩小实验规模，也不混用不同性质的来源。

## 1. 当前采用（免审批、可直接下）

| 来源 | 内容 | 条数 | 体积 | 用途 |
|---|---|---|---|---|
| `svjack/video-dataset-genshin-impact-anime-organized` | 动漫风 CG 短片，有角色 | **33** | ~7.4 MB | 训练主集（角色演出） |
| `svjack/Anime_Landscape_Videos` | 动画风景/空镜片段 | **65** | ~0.68 GB | 训练副集（画风/运动多样性） |

合计约 **98 条**。短预算实验（600–1000 step）下每条可随机裁窗出多个样本，够用。

下载（A800 上已验证可下）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("svjack/video-dataset-genshin-impact-anime-organized",
                  repo_type="dataset", local_dir="/root/autodl-tmp/data/genshin")
snapshot_download("svjack/Anime_Landscape_Videos",
                  repo_type="dataset", local_dir="/root/autodl-tmp/data/landscape")
PY
```

> 注意：hf-mirror 大文件吞吐波动很大（实测 130 KB/s ~ 12 MB/s 都出现过）；模型权重建议走
> ModelScope（见 `docs/AUTODL_5090.md`），数据集本身很小，走 hf-mirror 没问题。

## 2. 可选补充

| 来源 | 内容 | 体积 | 许可状态 | 用法 |
|---|---|---|---|---|
| `Fitsd/anime_video_and_anime_screenshots` | 按番剧打包的**动画截图**（79 个 zip） | 181 GB（单部 7–12 GB） | 未标注（爬取） | **只做角色/风格参考图**，不进时序训练 |
| `rjdxg/Anime_Videos` | 10K–100K 条动画视频 | 大 | gated（需申请） | 论文级扩量 |
| Sakuga-42M | 动画视频 + 标注 | 大 | gated（需申请） | 论文级扩量 |
| AnimeShooter | 动画视频 + 标注 | 大 | gated（需申请） | 论文级扩量 |
| Manga109 / eBDtheque | 漫画页 | 中 | 学术授权 | 若转向「漫画页→动画」方向 |
| Digital Comic Museum / Comic Book Plus | 公版老漫画扫描件 | 中 | 公有领域 | 同上，无版权风险 |

## 3. 数据组织约定

1. 训练集与评测集**按 clip 划分**，同一 clip 的不同裁窗不能同时出现在两边
2. 每条 clip 的参考帧（Anchor-LoRA 用）= 该 clip 的首帧；**锚帧不能等于监督窗口内的帧**（避免泄漏）
3. `video_metadata.csv` 沿用 `data/prepare_comic_dataset.py` 的 schema：`video, caption, frames, fps, width, height, motion_score`
4. 评测用的 FVD 参考集必须与训练集**不重叠**；样本 < 50 时脚本标 `reliable:false`，如实写进论文

## 4. 已知限制（写进论文 Limitations）

- 98 条片段远低于动画域微调的常规规模（同类工作常用 1k–10k 条）
- 两个来源都是"动漫风格"，但**不是同一部作品/同一角色**，主体一致性结论要按"风格内"口径表述
- 若后续申请到 Sakuga-42M / AnimeShooter，需重跑主实验以保持可比性
