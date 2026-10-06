"""Writes fixtures for the ControlFoley CLIP conditioner parity tests.

  tiny   : a tiny random open_clip CLIP (released open_clip key layout) plus the outputs of the official ControlFoley
           paths (`patch_clip(...).encode_text(tokens, normalize=True)` and `encode_image(normalize=True)` after the
           `Normalize(mean, std)` of `FeaturesUtils.encode_video_with_clip`) -> clip_tiny.safetensors/.json
  tokens : golden token ids of `open_clip.get_tokenizer('ViT-H-14-378-quickgelu')` for varied prompts -> clip_tokens.json
  real   : the same comparison for real weights; <weights.safetensors> is the fp16 open_clip checkpoint written by
           clip_convert_remote.py (it is loaded into an fp32 model, so python and C# see identical values)
           -> clip_real_reference.safetensors/.json

Usage: python clip_reference.py tiny|tokens <controlfoley checkout> <output dir>
       python clip_reference.py real <controlfoley checkout> <output dir> <weights.safetensors>
(needs torch, open_clip_torch, ftfy, safetensors)
"""
import ast
import json
import sys
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
TOKENIZER_NAME = "ViT-H-14-378-quickgelu"

PROMPTS = [
    "",
    " ",
    "a dog barking in the distance",
    "A Dog BARKING, in the Distance!!",
    "footsteps on gravel... then a door slams.",
    "it's what you'd expect: they're we've I'm she'll isn't",
    "rain   on\ta tin\nroof",
    "1234567890 and 3.14159",
    "café au lait — naïve façade",
    "こんにちは世界 مرحبا привет",
    "emoji \U0001F415\U0001F3B5 sound",
    "AT&amp;T &lt;b&gt; &amp;amp; fish &#39;n&#39; chips",
    "“smart quotes” and ‘single’ ﬁnal ＡＢＣ １２",
    "<start_of_text>hello<end_of_text>",
    "<|startoftext|> literal hf markers <|endoftext|>",
    "!@#$%^&*()_+-=[]{}|;:'\",.<>/?`~",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "word " * 120,
    " ".join(f"token{i}" for i in range(40)),
    "a " * 74 + "b",
    "a " * 75,
    "a " * 76,
    "\u0000\u0001 control \u007f chars ​ zero width",
    "line one\r\nline two line three\u0085",
    "  leading and trailing   ",
]


def load_patch_clip(controlfoley: Path):
    """Extracts the official `patch_clip` from feature_extractor.py without importing its heavy dependencies."""
    source = (controlfoley / "controlfoley" / "feature_extractor.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "patch_clip")
    namespace = {"F": F}
    exec(compile(ast.Module([node], []), "feature_extractor.py", "exec"), namespace)
    return namespace["patch_clip"]


def clip_preprocess(x: torch.Tensor) -> torch.Tensor:
    from torchvision.transforms import Normalize
    return Normalize(mean=MEAN, std=STD)(x)


def randomize(model: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=gen)
            if name.endswith("norm.weight") or ".ln_" in name and name.endswith("weight") or name.startswith("ln_final.weight"):
                p.copy_(1.0 + 0.2 * noise)
            elif p.ndim == 1 and name != "visual.class_embedding":
                p.copy_(0.1 * noise)
            else:
                p.copy_(0.25 * noise)


def clip_encode_text(model, tokens):
    with torch.inference_mode():
        return model.encode_text(tokens, normalize=True)


def clip_encode_images(model, frames):
    with torch.inference_mode():
        return model.encode_image(clip_preprocess(frames), normalize=True)


def write_tiny(controlfoley: Path, out: Path) -> None:
    torch.manual_seed(11)
    vision_cfg = {"image_size": 42, "layers": 3, "width": 32, "head_width": 8, "patch_size": 7}
    text_cfg = {"context_length": 77, "vocab_size": 512, "width": 24, "heads": 3, "layers": 3}
    model = open_clip.CLIP(embed_dim=16, vision_cfg=vision_cfg, text_cfg=text_cfg, quick_gelu=True).eval()
    randomize(model, 5)
    model = load_patch_clip(controlfoley)(model)

    tokens = torch.zeros(3, 77, dtype=torch.long)
    for i, length in enumerate([3, 9, 77]):
        ids = torch.randint(1, 500, (length,))
        ids[0] = 510
        ids[-1 if length < 77 else 76] = 511
        tokens[i, :length] = ids
    frames = torch.rand(2, 3, 44, 44)

    text = clip_encode_text(model, tokens)
    images = clip_encode_images(model, frames)
    tensors = {k: v.contiguous().float() for k, v in model.state_dict().items()}
    tensors["ref.text"] = text.contiguous()
    tensors["ref.image"] = images.contiguous()
    tensors["in.frames"] = frames.contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "clip_tiny.safetensors"))
    (out / "clip_tiny.json").write_text(json.dumps({"tokens": tokens.tolist()}))
    print("wrote tiny", tuple(text.shape), tuple(images.shape))


def write_tokens(out: Path) -> None:
    tokenizer = open_clip.get_tokenizer(TOKENIZER_NAME)
    ids = tokenizer(PROMPTS).tolist()
    out.mkdir(parents=True, exist_ok=True)
    (out / "clip_tokens.json").write_text(
        json.dumps([{"text": t, "ids": i} for t, i in zip(PROMPTS, ids)], ensure_ascii=True), encoding="utf-8")
    print("wrote tokens", len(PROMPTS))


def write_real(controlfoley: Path, out: Path, weights: Path) -> None:
    tokenizer = open_clip.get_tokenizer(TOKENIZER_NAME)
    config = json.loads(Path(open_clip.__file__).parent.joinpath("model_configs", TOKENIZER_NAME + ".json").read_text())
    model = open_clip.CLIP(embed_dim=config["embed_dim"], vision_cfg=config["vision_cfg"], text_cfg=config["text_cfg"],
                           quick_gelu=True).eval()
    state = {k: v.float() for k, v in load_file(str(weights)).items()}
    print(model.load_state_dict(state, strict=False))
    model = load_patch_clip(controlfoley)(model)
    prompts = ["a dog barking in the distance", "footsteps on gravel then a door slams", "", "Rain on a tin roof, thunder!",
               "an orchestra tuning up before the concert begins"]
    text = clip_encode_text(model, tokenizer(prompts))
    gen = torch.Generator().manual_seed(7)
    smooth = torch.nn.functional.interpolate(torch.rand(2, 3, 12, 12, generator=gen), size=(384, 384), mode="bicubic")
    frames = smooth.clamp(0, 1).contiguous()
    images = clip_encode_images(model, frames)
    out.mkdir(parents=True, exist_ok=True)
    save_file({"ref.text": text.contiguous(), "ref.image": images.contiguous(), "in.frames": frames}, str(out / "clip_real_reference.safetensors"))
    (out / "clip_real_reference.json").write_text(json.dumps({"prompts": prompts}))
    print("wrote real", tuple(text.shape), tuple(images.shape))


if __name__ == "__main__":
    mode, cf, outdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if mode == "tiny":
        write_tiny(cf, outdir)
    elif mode == "tokens":
        write_tokens(outdir)
    elif mode == "real":
        write_real(cf, outdir, Path(sys.argv[4]))
