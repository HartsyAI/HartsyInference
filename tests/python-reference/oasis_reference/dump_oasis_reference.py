"""
Oasis-500m parity oracle. Runs the upstream `etched-ai/open-oasis` reference on the real weights and dumps
per-component IO for the C# parity test:
  oasis_ref_io.safetensors:
    vae_rgb_in        [1,3,360,640]   fixed RGB in [-1,1]
    vae_encode_mean   [1,576,16]      VAE encoder posterior.mode() (deterministic mean, UNSCALED)
    vae_dec_latent    [1,576,16]      fixed latent fed to decode
    vae_decode_rgb    [1,3,360,640]   VAE decode output in [-1,1]
    dit_x             [1,T,16,18,32]  fixed latent sequence
    dit_t             [1,T]           per-frame noise indices (int as float)
    dit_cond          [1,T,25]        per-frame action vectors
    dit_v             [1,T,16,18,32]  DiT v-prediction output

Usage:
  python3 tests/python-reference/oasis_reference/dump_oasis_reference.py
Env: OASIS_DIT (oasis500m.pt), OASIS_VAE (vit-l-20.pt), OPEN_OASIS (/tmp/open-oasis)
"""
import os
import sys
import torch
from safetensors.torch import save_file

DIT = os.environ.get("OASIS_DIT", "/tmp/oasis/oasis500m.pt")
VAE = os.environ.get("OASIS_VAE", "/tmp/oasis/vit-l-20.pt")
OPEN = os.environ.get("OPEN_OASIS", "/tmp/open-oasis")
OUT = os.path.dirname(os.path.abspath(__file__))
T = 4

sys.path.insert(0, OPEN)
from dit import DiT_S_2  # noqa: E402
from vae import ViT_L_20_Shallow_Encoder  # noqa: E402


def load_sd(fn):
    sd = torch.load(fn, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and hasattr(sd["model"], "keys"):
        sd = sd["model"]
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    return {k.replace("module.", "").replace("model.", "") if k.startswith(("module.", "model.")) else k: v
            for k, v in sd.items()}


@torch.no_grad()
def main():
    torch.manual_seed(0)
    refs = {}

    # --- VAE ---
    vae = ViT_L_20_Shallow_Encoder().eval()
    miss, unexp = vae.load_state_dict(load_sd(VAE), strict=False)
    print("VAE missing:", [m for m in miss if "rotary" not in m][:8], "unexpected:", [u for u in unexp if "rotary" not in u][:8])
    rgb = (torch.rand(1, 3, 360, 640) * 2 - 1)
    refs["vae_rgb_in"] = rgb.clone()
    post = vae.encode(rgb)
    mean = post.mode()  # [1,576,16]
    refs["vae_encode_mean"] = mean.clone()
    dec_latent = torch.randn(1, 576, 16)
    refs["vae_dec_latent"] = dec_latent.clone()
    refs["vae_decode_rgb"] = vae.decode(dec_latent).clone()

    # --- DiT ---
    dit = DiT_S_2().eval()
    miss, unexp = dit.load_state_dict(load_sd(DIT), strict=False)
    print("DiT missing:", [m for m in miss if "rotary" not in m][:8], "unexpected:", [u for u in unexp if "rotary" not in u][:8])
    x = torch.randn(1, T, 16, 18, 32)
    t = torch.tensor([[100, 300, 500, 15]], dtype=torch.float32)  # per-frame noise idx (last = target)
    cond = torch.rand(1, T, 25)

    # taps: conditioning c [1,T,1024], x after x_embedder [1,T,H,W,D], block0 output [1,T,H,W,D]
    taps = {}
    xe = {}

    def xemb_hook(m, i, o):
        xe["x"] = o.detach().clone()
    h1 = dit.x_embedder.register_forward_hook(xemb_hook)
    h2 = dit.blocks[0].register_forward_hook(lambda m, i, o: taps.__setitem__("block0", o.detach().clone()))
    h3 = dit.blocks[-1].register_forward_hook(lambda m, i, o: taps.__setitem__("blockLast", o.detach().clone()))
    # conditioning c
    tt = t.reshape(-1)
    c = dit.t_embedder(tt)
    c = c.reshape(1, T, -1)
    c = c + dit.external_cond(cond)
    taps["c"] = c.detach().clone()

    v = dit(x, t, cond)  # [1,T,16,18,32]
    h1.remove(); h2.remove(); h3.remove()

    refs["dit_x"] = x.clone()
    refs["dit_t"] = t.clone()
    refs["dit_cond"] = cond.clone()
    refs["dit_v"] = v.clone()
    refs["dit_c"] = taps["c"]                                   # [1,T,1024]
    # x_embedder output: [(B T), H, W, D] -> [1,T,H,W,D]
    refs["dit_xembed"] = xe["x"].reshape(1, T, xe["x"].shape[1], xe["x"].shape[2], xe["x"].shape[3])
    refs["dit_block0"] = taps["block0"]                        # [1,T,H,W,D]
    refs["dit_blockLast"] = taps["blockLast"]                  # [1,T,H,W,D]

    refs = {k: val.float().contiguous() for k, val in refs.items()}
    save_file(refs, os.path.join(OUT, "oasis_ref_io.safetensors"))
    for k, val in refs.items():
        print(f"  {k:18s} {tuple(val.shape)}")
    print("wrote oasis_ref_io.safetensors ->", OUT)


if __name__ == "__main__":
    main()
