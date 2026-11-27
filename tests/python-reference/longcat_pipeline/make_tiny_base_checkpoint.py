"""Writes a tiny LongCat-Video model folder (tokenizer, text_encoder, vae, scheduler, dit, lora) from the pinned reference's own classes, with
random weights, to try run_base_reference.py without the released files. The output means nothing.

    python make_tiny_base_checkpoint.py --repo /path/LongCat-Video --out /path/tiny
"""

import argparse
import os
import sys

import torch

from run_base_reference import NEGATIVE_PROMPT, PROMPT, install_stubs

LORA_RANK = 4
# The linears the release LoRAs adapt, and the number of up blocks each fused projection is split into.
LORA_MODULES = (("attn.qkv", 3), ("attn.proj", 1), ("cross_attn.q_linear", 1), ("cross_attn.kv_linear", 2), ("ffn.w1", 1), ("ffn.w2", 1),
                ("ffn.w3", 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    install_stubs()

    import sentencepiece
    from safetensors.torch import save_file
    from transformers import T5Tokenizer, UMT5Config, UMT5EncoderModel
    from longcat_video.modules import lora_utils
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    corpus = os.path.join(args.out, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as handle:
        handle.write("\n".join([PROMPT, NEGATIVE_PROMPT] * 20))
    prefix = os.path.join(args.out, "spiece")
    sentencepiece.SentencePieceTrainer.train(input=corpus, model_prefix=prefix, vocab_size=120, hard_vocab_limit=False, pad_id=0, eos_id=1,
                                             unk_id=2, bos_id=-1)
    tokenizer = T5Tokenizer(prefix + ".model", extra_ids=0, legacy=False)
    tokenizer.save_pretrained(os.path.join(args.out, "tokenizer"))

    text = UMT5EncoderModel(UMT5Config(vocab_size=len(tokenizer), d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=2))
    text.save_pretrained(os.path.join(args.out, "text_encoder"))

    vae = AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, temperal_downsample=[False, True, True])
    vae.save_pretrained(os.path.join(args.out, "vae"))

    FlowMatchEulerDiscreteScheduler(shift=12.0).save_pretrained(os.path.join(args.out, "scheduler"))

    dit = LongCatVideoTransformer3DModel(in_channels=16, out_channels=16, hidden_size=64, depth=3, num_heads=4, caption_channels=32,
                                         mlp_ratio=4, adaln_tembed_dim=32, frequency_embedding_size=32, enable_flashattn2=True,
                                         cp_split_hw=[1, 1], text_tokens_zero_pad=True)
    with torch.no_grad():
        for parameter in dit.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    dit.save_pretrained(os.path.join(args.out, "dit"))

    wanted = {}
    for block in range(len(dit.blocks)):
        for name, splits in LORA_MODULES:
            key = "lora___lorahyphen___" + f"blocks.{block}.{name}".replace(".", "___lorahyphen___")
            wanted[key + ".lora_down.weight"] = None
            for index in range(splits if splits > 1 else 0):
                wanted[f"{key}.lora_up.blocks.{index}.weight"] = None
    network = lora_utils.create_lora_network(dit, wanted, multiplier=1.0, network_dim=LORA_RANK, network_alpha=64)
    with torch.no_grad():
        for lora in network.loras:
            for parameter in lora.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.15)
            lora.alpha_scale.fill_(0.5)
    os.makedirs(os.path.join(args.out, "lora"), exist_ok=True)
    save_file({key: value.contiguous() for key, value in network.state_dict().items()},
              os.path.join(args.out, "lora", "cfg_step_lora.safetensors"))
    print(f"tiny model folder written to {args.out}; run the distilled call with --lora-dim {LORA_RANK}")


if __name__ == "__main__":
    main()
