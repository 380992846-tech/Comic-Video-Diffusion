"""鲸鱼娘 Q 版小短片生成器（10 个故事，每个 ~12 秒，全部 < 1 分钟）。

用 AnimateDiff + 动漫向 SD1.5（Counterfeit-V2.5）逐场景生成，再在场景之间做交叉淡入淡出
拼成一个小故事。每个故事 6 个场景、每场景 16 帧 @8fps（2 秒），淡化后约 11-12 秒。

用法：
    python scripts/make_whale_girl.py \
      --model_id_or_path /root/autodl-tmp/models/counterfeit-v2.5 \
      --motion_adapter_id_or_path /root/autodl-tmp/models/adapter-v15-2 \
      --out_dir /root/autodl-tmp/whale_girl --resolution 512 --frames 16 --steps 25

产物：
    <out_dir>/story_01_*.mp4 ... story_10_*.mp4
    <out_dir>/stories.md      中文剧情 + 每场景英文 prompt
    <out_dir>/manifest.json   生成参数（模型/seed/步数/分辨率/帧数）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animatediff_infer import build_pipeline  # noqa: E402

# 角色基底：Q 版鲸鱼娘。放在每个 prompt 最前面，保持跨场景一致。
CHAR = ("chibi, super deformed, chibi style, 1girl, whale girl, cute, kawaii, "
        "big head small body, 2 heads tall, blue hair, big eyes, whale tail, fin ears, "
        "vibrant colors, masterpiece, best quality, anime style")

QUALITY = ""  # CLIP 只有 77 token，多余的风格词会被截断，故不再追加
NEG = ("lowres, bad anatomy, bad hands, bad proportions, extra limbs, extra fingers, "
       "text, error, missing fingers, cropped, worst quality, low quality, normal quality, "
       "jpeg artifacts, signature, watermark, username, blurry, realistic, 3d")

# 10 个故事：中文剧情随意写，场景 prompt 用英文（SD1.5 的 CLIP 只吃英文）
STORIES = [
    {
        "id": "01_morning",
        "title": "小鲸鱼娘的早晨",
        "story": "闹钟响了三次才爬起来，刷牙时把泡沫弄到鼻子上，早餐是鱼形面包，最后被海风吹醒，开心地跑出门。",
        "scenes": [
            "sleeping in a shell-shaped bed, blanket over head, morning sun through the window",
            "sitting up and stretching, yawning, messy hair, tiny bedroom with fish posters",
            "brushing teeth in front of a round mirror, foam on her nose, sleepy face",
            "eating a fish-shaped bread at a small table, cheeks puffed, happy",
            "running out of the door, holding a tiny bag, hair flying, seaside street",
            "standing on a hill looking at the sea, arms up, wind blowing, big smile",
        ],
    },
    {
        "id": "02_shell",
        "title": "捡贝壳的小发现",
        "story": "在沙滩上找到一枚发光贝壳，里面蹦出一只小螃蟹，两人从抢贝壳变成好朋友，最后一起看夕阳。",
        "scenes": [
            "walking on a sunny beach with a small bucket, footprints in the sand",
            "kneeling down and picking up a glowing pink seashell, sparkles",
            "a tiny crab jumps out of the shell, surprised expression, comic shock marks",
            "playing tug of war over the shell with the tiny crab, both determined",
            "laughing together with the crab on her head, holding the shell",
            "sitting on the sand watching the sunset with the crab, warm orange sky",
        ],
    },
    {
        "id": "03_rain",
        "title": "下雨天也有好事",
        "story": "撑着小伞出门，故意踩水坑溅自己一身，雨停后天上出现彩虹，她张开手转圈。",
        "scenes": [
            "holding a transparent umbrella in the rain, raindrops on the ground",
            "jumping into a puddle, water splashing, laughing",
            "sheltering a tiny snail under the umbrella, gentle smile",
            "rain stopping, clouds parting, light breaking through",
            "a rainbow appears over the seaside town, she looks up amazed",
            "spinning around with arms open, skirt fluttering, sparkles in the air",
        ],
    },
    {
        "id": "04_cake",
        "title": "失败的蛋糕也很甜",
        "story": "想做鲸鱼蛋糕，结果奶油糊了一脸，重做一次终于成功，插上蜡烛自己给自己庆祝。",
        "scenes": [
            "wearing a tiny apron and chef hat, mixing batter in a bowl",
            "whipped cream splashing onto her face, flour everywhere, funny mess",
            "sad face looking at a lopsided whale-shaped cake, drooping ahoge",
            "trying again with determined eyes, whisk in hand",
            "a perfectly cute whale-shaped cake on the table, strawberries around",
            "blowing out a single candle on the cake, hands clasped, happy",
        ],
    },
    {
        "id": "05_aquarium",
        "title": "水族馆的一天",
        "story": "去看水母，被发光的水母迷住，举着相机拍照，最后和水母一起比了个心。",
        "scenes": [
            "standing in a dim blue aquarium corridor, big eyes reflecting light",
            "jellyfish glowing above her, hands pressed on the glass, amazed",
            "taking a photo with a small camera, flash, focused expression",
            "sitting on the floor watching fish swim by, chin on knees, calm",
            "making a heart shape with both hands toward a jellyfish, sparkling",
            "walking out of the aquarium, souvenir whale plush in arms, evening sky",
        ],
    },
    {
        "id": "06_snow",
        "title": "雪天的鲸鱼雪人",
        "story": "第一次见到雪，堆了一个鲸鱼形状的雪人，和朋友打雪仗，最后捧着热可可取暖。",
        "scenes": [
            "catching snowflakes with both hands, scarf and mittens, snow falling",
            "rolling a huge snowball, cheeks red, breath visible",
            "proudly standing next to a whale-shaped snowman with a scarf",
            "throwing a snowball, laughing, mid air snow",
            "sitting in the snow with snow on her head, tired but happy",
            "holding a mug of hot cocoa with both hands, steam rising, cozy",
        ],
    },
    {
        "id": "07_dream",
        "title": "书里游出来的梦",
        "story": "看书看睡着了，梦里和小鱼群一起游过珊瑚，醒来时书还摊在脸上，笑了。",
        "scenes": [
            "reading a thick picture book at a desk, eyelids heavy",
            "asleep with the book on her face, small bubble from her mouth",
            "underwater dream, swimming with a school of colorful fish, bubbles",
            "diving through a coral arch, whale tail glowing, dreamy light rays",
            "floating among glowing plankton, eyes closed, peaceful",
            "waking up with the book on her face, smiling, afternoon light",
        ],
    },
    {
        "id": "08_drawing",
        "title": "画一只鲸鱼",
        "story": "想画鲸鱼却画歪了，擦了重画，最后画出一只特别可爱的鲸鱼，举起来给镜头看。",
        "scenes": [
            "sitting at a low table with crayons scattered around",
            "drawing with tongue sticking out, very focused",
            "holding up a scribbled picture with a troubled face",
            "erasing the paper, eraser crumbs everywhere, determined",
            "drawing a cute round whale with a happy face, colorful crayons",
            "holding the finished drawing up to the camera, proud big smile",
        ],
    },
    {
        "id": "09_cat",
        "title": "和猫抢毛线球",
        "story": "毛线球被猫抢走，追着追着把自己缠成团子，被猫解救后一起睡着了。",
        "scenes": [
            "a fluffy cat playing with a red yarn ball on the floor",
            "chasing the cat around the room, arms out, motion blur",
            "wrapped up in red yarn like a cocoon, only head visible, surprised",
            "cat pawing at the yarn on her, curious expression",
            "unwrapped and laughing, yarn everywhere, cat on her lap",
            "both of them asleep on a cushion, yarn still tangled, cozy",
        ],
    },
    {
        "id": "10_birthday",
        "title": "海底生日会",
        "story": "戴上派对帽，气球和小鱼都来庆祝，她对着蛋糕许愿，最后和大家一起比耶。",
        "scenes": [
            "wearing a tiny party hat, confetti in the air, excited",
            "colorful balloons and small fish floating around her",
            "a big strawberry cake with candles in front of her, glowing",
            "closing eyes and making a wish, hands clasped",
            "blowing out the candles, smoke swirls, everyone cheering",
            "jumping with both arms up, tiny fish friends around, sparkles",
        ],
    },
]


def crossfade(clips: list[np.ndarray], fade: int) -> np.ndarray:
    """把若干段 (T,H,W,3) 用交叉淡化拼成一段。"""
    out = clips[0]
    for nxt in clips[1:]:
        fade = min(fade, len(out), len(nxt))
        alpha = np.linspace(0, 1, fade, dtype=np.float32).reshape(-1, 1, 1, 1)
        blended = (out[-fade:].astype(np.float32) * (1 - alpha)
                   + nxt[:fade].astype(np.float32) * alpha).astype(np.uint8)
        out = np.concatenate([out[:-fade], blended, nxt[fade:]], axis=0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id_or_path", default="/root/autodl-tmp/models/counterfeit-v2.5")
    ap.add_argument("--motion_adapter_id_or_path", default="/root/autodl-tmp/models/adapter-v15-2")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/whale_girl")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance_scale", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--fade", type=int, default=4, help="场景间交叉淡化帧数")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--stories", type=int, default=10)
    ap.add_argument("--scenes", type=int, default=0, help="只跑前 N 个场景（调试用，0=全部）")
    ap.add_argument("--decode_chunk", action="store_true", help="额外开启 VAE tiling（更省显存）")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    import imageio.v2 as imageio

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_args = argparse.Namespace(model_id_or_path=args.model_id_or_path,
                                    motion_adapter_id_or_path=args.motion_adapter_id_or_path,
                                    scheduler="ddim")
    pipe, _motion_unet, _adapter = build_pipeline(build_args, device, dtype)
    # AnimateDiff v1.5 必须用 linear beta + linspace + steps_offset=1 的 DDIM。
    # 直接用 SD1.5 仓库里的 scheduler（scaled_linear / leading）会得到发灰、
    # 低对比度的糊图——这是实测 A/B 出来的（std 0.045 -> 0.075）。
    from diffusers import DDIMScheduler
    pipe.scheduler = DDIMScheduler.from_pretrained(
        args.model_id_or_path, subfolder="scheduler",
        clip_sample=False, timestep_spacing="linspace", beta_schedule="linear", steps_offset=1)
    pipe.vae.enable_slicing()   # 逐帧解码，避免与训练抢显存时 VAE decode OOM
    pipe.to(device)
    pipe.unet.eval()
    print(f"[model] {args.model_id_or_path} + {args.motion_adapter_id_or_path} on {device} "
          f"({args.dtype}); scheduler=AnimateDiff-DDIM(linear/linspace/offset1); vae_slicing=on")
    if args.decode_chunk:
        pipe.vae.enable_tiling()

    manifest = {"model": args.model_id_or_path, "motion_adapter": args.motion_adapter_id_or_path,
                "resolution": args.resolution, "frames": args.frames, "steps": args.steps,
                "fps": args.fps, "guidance_scale": args.guidance_scale, "seed": args.seed,
                "fade": args.fade, "character_prompt": CHAR, "negative_prompt": NEG,
                "stories": []}
    lines = ["# 鲸鱼娘 Q 版小短片（10 个）", "",
             f"生成参数：Counterfeit-V2.5 + AnimateDiff v1.5 motion adapter，{args.resolution}×{args.resolution}，"
             f"每场景 {args.frames} 帧 @{args.fps}fps，{args.steps} 步，DDIM，seed {args.seed}。", ""]

    for si, story in enumerate(STORIES[: args.stories], start=1):
        scenes = story["scenes"][: args.scenes] if args.scenes else story["scenes"]
        clips = []
        for ci, scene in enumerate(scenes):
            prompt = f"{CHAR}, {scene}" if not QUALITY else f"{CHAR}, {scene}, {QUALITY}"
            g = torch.Generator(device="cpu").manual_seed(args.seed + si * 100 + ci)
            with torch.no_grad():
                result = pipe(prompt=prompt, negative_prompt=NEG,
                              num_frames=args.frames, guidance_scale=args.guidance_scale,
                              num_inference_steps=args.steps,
                              height=args.resolution, width=args.resolution,
                              generator=g)
            frames = np.stack([np.asarray(f.convert("RGB")) for f in result.frames[0]], axis=0)
            clips.append(frames)
            print(f"  story {si:02d} scene {ci + 1}/{len(scenes)} done ({frames.shape})")
        video = crossfade(clips, args.fade)
        name = f"story_{si:02d}_{story['id'].split('_', 1)[1]}.mp4"
        path = out_dir / name
        imageio.mimwrite(path, list(video), fps=args.fps, quality=8, macro_block_size=8)
        dur = len(video) / args.fps
        print(f"[story {si:02d}] {name}  {len(video)} frames  {dur:.1f}s")
        manifest["stories"].append({"file": name, "title": story["title"], "story": story["story"],
                                    "frames": int(len(video)), "seconds": round(dur, 2),
                                    "scenes": scenes})
        lines += [f"## {si:02d}. {story['title']}（{dur:.1f}s）", "", story["story"], ""]
        for ci, scene in enumerate(scenes, 1):
            lines.append(f"{ci}. `{CHAR}, {scene}, {QUALITY}`")
        lines.append("")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "stories.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] {len(manifest['stories'])} videos -> {out_dir}")
    os._exit(0)


if __name__ == "__main__":
    main()
