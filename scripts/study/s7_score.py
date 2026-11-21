"""Scores the images written by s7_quality.py.

Per arm, on the captions that every arm has finished:
  clip        CLIP ViT-B/32 image-text similarity (prompt alignment)
  pick        PickScore v1 (learned human preference; higher is better)
  cmmd        CLIP-MMD to the real COCO images of the same captions (lower is better)
  cmmd_ref    CLIP-MMD to the 25-step base-model images
  fid, kid    clean-fid against the real COCO images (lower is better; biased at this sample size,
              so only differences between arms mean anything)
  lpips_full  LPIPS to the same adapter's full version on the same seed
Differences in clip and pick against the adapter's full version come with paired bootstrap intervals.

    python scripts/study/s7_score.py [--no-fid] [--trained] [--sdxl] [--ablation] [--transfer]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from s7_quality import CAPTIONS, ROOT, coco_items  # noqa: E402

from wspd import config  # noqa: E402
from wspd.metrics import Evaluator, paired_bootstrap  # noqa: E402
from wspd.model import best_device  # noqa: E402

REAL = CAPTIONS.parent.parent / "val2014"
FEATS = ROOT.parent / "quality_features"
OTHER = {"sdxl": "s11_sdxl_quality.json", "ablation": "s12_ablation_quality.json", "transfer": "s13_transfer_quality.json"}
PICK, PICK_PROCESSOR, CMMD_CLIP = "yuvalkirstain/PickScore_v1", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "openai/clip-vit-large-patch14-336"


def real_image(item: dict) -> Image.Image:
    """The COCO photo, center-cropped to a square and resized to 512."""
    im = Image.open(REAL / item["file_name"]).convert("RGB")
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    return im.crop((left, top, left + side, top + side)).resize((512, 512), Image.BICUBIC)


def cached(name: str, arm: str, n: int, compute) -> np.ndarray:
    """Per-image features, stored so that a later run with more images only computes the new ones."""
    path = FEATS / f"{name}__{arm}.npy"
    have = np.load(path) if path.exists() else None
    if have is not None and len(have) >= n:
        return have[:n]
    start = 0 if have is None else len(have)
    new = compute(start, n)
    out = new if have is None else np.concatenate([have, new])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def feats(x) -> torch.Tensor:
    """Newer transformers return an output object from get_*_features; the projected features are its pooler_output."""
    return x if torch.is_tensor(x) else x.pooler_output


def pick_scorer(device):
    """PickScore v1 as a function of a list of images and their prompts."""
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(PICK).to(device).eval()
    proc = AutoProcessor.from_pretrained(PICK_PROCESSOR)

    def score(images: list[Image.Image], prompts: list[str]) -> np.ndarray:
        inputs = proc(images=images, text=prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        im = F.normalize(feats(model.get_image_features(pixel_values=inputs["pixel_values"])), dim=-1)
        tx = F.normalize(feats(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])), dim=-1)
        return (model.logit_scale.exp() * (im * tx).sum(-1)).float().cpu().numpy()

    return score


def mmd(x: np.ndarray, y: np.ndarray, sigma: float = 10.0) -> float:
    """CMMD (Jayasumana et al., 2024): squared MMD with a Gaussian kernel on unit CLIP embeddings, times 1000."""
    x, y = torch.from_numpy(x).double(), torch.from_numpy(y).double()
    g = 1 / (2 * sigma**2)
    k = lambda a, b: torch.exp(-g * torch.cdist(a, b).pow(2))  # noqa: E731
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    m, n = len(x), len(y)
    return float(1000 * ((kxx.sum() - kxx.trace()) / (m * (m - 1)) + (kyy.sum() - kyy.trace()) / (n * (n - 1)) - 2 * kxy.mean()))


def main() -> None:
    device = best_device()
    # --trained also scores the study 10 arms, on the captions those have (fewer), into its own file.
    trained = "--trained" in sys.argv
    # --sdxl, --ablation and --transfer score the study 11, 12 and 13 arms, each with its own feature cache and results file.
    other = next((k for k in OTHER if f"--{k}" in sys.argv), None)
    root = ROOT.parent / f"quality_{other}" if other else ROOT
    tag = f"{other}_" if other else ""
    folders = {p.name: p for p in root.iterdir() if p.is_dir()}
    if trained:
        folders.update({p.name: p for p in (ROOT.parent / "quality_trained").iterdir() if p.is_dir()})
    arms = sorted(folders)
    hosts = {a[: -len("_full")] for a in arms if a.endswith("_full")}
    n = min(len(list(folders[a].glob("*.png"))) for a in arms)
    n -= n % 50
    items = coco_items(n)
    prompts = [it["caption"] for it in items]
    print(f"{len(arms)} arms, {n} captions each", flush=True)

    def load(arm: str, start: int, stop: int) -> list[Image.Image]:
        if arm == "real":
            return [real_image(it) for it in items[start:stop]]
        return [Image.open(folders[arm] / f"{i:05d}.png").convert("RGB") for i in range(start, stop)]

    evaluator = Evaluator(device)
    from transformers import CLIPModel, CLIPProcessor

    pick = pick_scorer(device)
    big = CLIPModel.from_pretrained(CMMD_CLIP).to(device).eval()
    big_proc = CLIPProcessor.from_pretrained(CMMD_CLIP)

    def batches(start, stop, size=16):
        return [(i, min(i + size, stop)) for i in range(start, stop, size)]

    def clip_score(arm):
        return lambda start, stop: np.concatenate([evaluator.clip_similarity(np.stack([np.asarray(im) for im in load(arm, a, b)]), prompts[a:b]) for a, b in batches(start, stop)])

    def pick_score(arm):
        return lambda start, stop: np.concatenate([pick(load(arm, a, b), prompts[a:b]) for a, b in batches(start, stop)])

    def embedding(arm):
        def compute(start, stop):
            out = []
            for a, b in batches(start, stop):
                pixels = big_proc(images=load(arm, a, b), return_tensors="pt")["pixel_values"].to(device)
                out.append(F.normalize(feats(big.get_image_features(pixel_values=pixels)), dim=-1).float().cpu().numpy())
            return np.concatenate(out)
        return compute

    with torch.no_grad():
        per = {}
        for arm in ["real", *arms]:
            per[arm] = {"clip": cached(tag + "clip", arm, n, clip_score(arm)), "pick": cached(tag + "pick", arm, n, pick_score(arm)), "emb": cached(tag + "emb", arm, n, embedding(arm))}
            print(f"features {arm}", flush=True)
        results = {}
        for arm in ["real", *arms]:
            p = per[arm]
            results[arm] = r = {"clip": float(p["clip"].mean()), "pick": float(p["pick"].mean())}
            if arm != "real":
                r["cmmd"] = mmd(p["emb"], per["real"]["emb"])
                r["cmmd_halves"] = [mmd(p["emb"][h::2], per["real"]["emb"][h::2]) for h in (0, 1)]
            if arm not in ("real", "ref_base25") and "ref_base25" in per:
                r["cmmd_ref"] = mmd(p["emb"], per["ref_base25"]["emb"])
            host = arm.split("_")[0]
            if host in hosts and not arm.endswith("_full"):
                full = f"{host}_full"
                r["clip_minus_full"] = paired_bootstrap(p["clip"] - per[full]["clip"])
                r["pick_minus_full"] = paired_bootstrap(p["pick"] - per[full]["pick"])
                r["pick_win_rate_vs_full"] = float((p["pick"] > per[full]["pick"]).mean())
                r["lpips_full"] = float(cached(tag + "lpips_full", arm, n, lambda s, e: evaluator.lpips(np.stack([np.asarray(im) for im in load(arm, s, e)]), np.stack([np.asarray(im) for im in load(full, s, e)]))).mean())

    if "--no-fid" not in sys.argv:
        from cleanfid import fid

        # clean-fid calls scipy's sqrtm with an argument that newer scipy removed.
        sqrtm = fid.linalg.sqrtm
        fid.linalg = type("linalg", (), {"sqrtm": staticmethod(lambda m, disp=None: sqrtm(m) if disp is None else (sqrtm(m), 0.0))})

        real_dir = ROOT.parent / "quality_real" / str(n)
        if not real_dir.exists():
            real_dir.mkdir(parents=True)
            for i, it in enumerate(items):
                real_image(it).save(real_dir / f"{i:05d}.png")
        cpu = torch.device("cpu")
        for arm in arms:
            # clean-fid reads a whole folder, so score a folder of links to the first n images.
            sub = ROOT.parent / f"quality_subset_{tag}" / arm
            if sub.exists():
                for f in sub.iterdir():
                    f.unlink()
            sub.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (sub / f"{i:05d}.png").symlink_to(folders[arm] / f"{i:05d}.png")
            results[arm]["fid"] = float(fid.compute_fid(str(sub), str(real_dir), device=cpu, num_workers=0, verbose=False))
            results[arm]["kid"] = float(fid.compute_kid(str(sub), str(real_dir), device=cpu, num_workers=0, verbose=False))
            print(f"fid {arm} {results[arm]['fid']:.2f} kid {results[arm]['kid']:.4f}", flush=True)

    out = config.RESULTS / "study" / (OTHER[other] if other else "s10_quality.json" if trained else "s7_quality.json")
    out.write_text(json.dumps({"n_captions": n, "arms": results}, indent=1))
    for arm, r in results.items():
        extra = "".join(f" | {k} {r[k]:.3f}" for k in ("cmmd", "cmmd_ref", "fid", "lpips_full") if k in r)
        delta = f" | pick-full {r['pick_minus_full']['mean']:+.3f} [{r['pick_minus_full']['lo']:+.3f}, {r['pick_minus_full']['hi']:+.3f}]" if "pick_minus_full" in r else ""
        print(f"{arm:18s} clip {r['clip']:.4f} | pick {r['pick']:.3f}{extra}{delta}", flush=True)


if __name__ == "__main__":
    main()
