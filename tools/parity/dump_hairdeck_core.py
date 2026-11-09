"""Reference dump of the HairDeck core (set DiTs, colour head, card VAE decoder, tokens, Euler sampler, chain DiT).

Runs the official hairdeck code at a pinned commit on the CPU in fp32 and writes every stage's inputs and outputs as raw
little-endian blobs plus manifest.json (name -> dtype, shape, file), which the C# parity tests read.

Run (isolated interpreter; the HairDeck source is a clone at cb20f7073419a33c9ab88f22f686fcb9060a99f0):
  python -I tools/parity/dump_hairdeck_core.py --src <clone> --models <HairDeck models dir> --out <scratch dir>
"""
import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='HairDeck source clone at the pinned commit')
    p.add_argument('--models', required=True, help='folder with ckpt/ and data/ (the released HairDeck files)')
    p.add_argument('--out', required=True, help='output folder')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=2.0)
    return p.parse_args()


ARGS = parse_args()
os.makedirs(ARGS.out, exist_ok=True)
os.environ['HAIRDECK_ROOT'] = ARGS.models
os.environ['HAIRDECK_CKPT'] = os.path.join(ARGS.models, 'ckpt')
os.environ['HAIRDECK_DATA'] = os.path.join(ARGS.models, 'data')
os.environ['HAIRDECK_DEVICE'] = 'cpu'
sys.path.insert(0, ARGS.src)

import torch  # noqa: E402

from hairdeck import ckpt as ckpt_mod  # noqa: E402
from hairdeck.cards.tokens import LatentDecoder, clean_tokens, sign_token_normals, tokens_to_cards  # noqa: E402
from hairdeck.cards.vae import LoadedVAE  # noqa: E402
from hairdeck.models.colormodel import lab_to_srgb, load_color_dit  # noqa: E402
from hairdeck.models.imgmodel import build_img_model  # noqa: E402
from hairdeck.models.tokenizer import build_tokenizer  # noqa: E402
from hairdeck.spring.chaindit import ChainDiT, chain_stats, decode as chain_decode  # noqa: E402
from hairdeck import paths  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
MANIFEST = {}


def save(name, arr):
    arr = np.ascontiguousarray(np.asarray(arr))
    if arr.dtype == np.float64:
        dt = 'float64'
    elif arr.dtype in (np.int64, np.int32):
        arr = arr.astype(np.int64)
        dt = 'int64'
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
        dt = 'uint8'
    else:
        arr = arr.astype(np.float32)
        dt = 'float32'
    fn = f'{name}.bin'
    arr.astype(arr.dtype.newbyteorder('<')).tofile(os.path.join(ARGS.out, fn))
    MANIFEST[name] = dict(dtype=dt, shape=list(arr.shape), file=fn)


def sub_rng(seed):
    return torch.Generator(device='cpu').manual_seed(seed)


def euler_cfg(model, ctx, x, steps, cfg):
    """infer/image.py euler_cfg (copied: the module pulls in the preprocessing stack)."""
    B = ctx.shape[0]
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        tt = torch.full((B,), float(t), device=ctx.device)
        both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([ctx, ctx]),
                     torch.cat([torch.zeros(B, dtype=torch.bool, device=ctx.device),
                                torch.ones(B, dtype=torch.bool, device=ctx.device)])).float()
        vc, vu = both.chunk(2)
        v = vu + cfg * (vc - vu)
        x = x + (tn - t) * v
    return x


@torch.no_grad()
def main():
    m = ckpt_mod.load
    torch.manual_seed(0)
    g = sub_rng(1234)
    # ---- image DiT and tokenizer
    img_ck = m(os.path.join(ARGS.models, 'ckpt', 'image_model', 'model.safetensors'))
    import argparse as _ap
    a = _ap.Namespace(**img_ck['args'])
    tok = build_tokenizer(a.vae, head=True, device='cpu')
    dim_tok = tok.dim
    n_ctx = 1 + 4 + (a.res // 16) ** 2
    model = build_img_model(a, dim_tok).eval()
    model.load_state_dict(img_ck['ema'])
    del img_ck

    dino = torch.randn(1, n_ctx, 1024, generator=g)
    save('dino_tokens', dino)
    ctx = model.encode_cond(dino)
    save('img_ctx', ctx)
    noise = torch.randn(1, a.nmax, dim_tok, generator=g)
    save('noise_cards', noise)

    # one denoiser evaluation, cond and uncond (drop) at a fixed timestep
    t0 = torch.tensor([0.7])
    save('denoise_t', t0)
    save('denoise_cond', model(noise, t0, ctx, torch.zeros(1, dtype=torch.bool)))
    save('denoise_uncond', model(noise, t0, ctx, torch.ones(1, dtype=torch.bool)))

    # ---- full sampling
    x_final = euler_cfg(model, ctx, noise, ARGS.steps, ARGS.cfg)
    save('final_tokens', x_final)
    save('clean_tokens', clean_tokens(tok, x_final))
    save('sign_tokens', sign_token_normals(tok, clean_tokens(tok, x_final)))

    # ---- card VAE: decoder of the sampled cards, decode cards
    vae_dir = os.path.join(ARGS.models, 'ckpt', 'card_vae')
    vae = LoadedVAE(vae_dir, 'cpu')
    dec = LatentDecoder(vae_dir, 'cpu')
    cards = tokens_to_cards(tok, x_final, dec, fix=True)[0]
    seqs = [c[0] for c in cards]
    save('card_count', np.array([len(c[0]) for c in cards], np.int64))
    save('card_shape', np.array([c[1] for c in cards], np.int64))
    save('card_points', np.concatenate(seqs, 0))
    save('card_slot', np.nonzero(tok.decode(x_final)['exist'][0].numpy())[0].astype(np.int64))
    d = tok.decode(x_final)
    save('dec_root', d['root'][0].numpy())
    save('dec_z', d['z'][0].numpy())
    save('dec_nrm', d['nrm'][0].numpy())
    save('dec_exist', d['exist'][0].numpy().astype(np.uint8))

    # fixed latent through the VAE decoder
    zg = sub_rng(77)
    z_fixed = torch.randn(4, 32, generator=zg) * 0.5
    root_fixed = np.array([[0.0, 0.47, 0.01], [0.01, 0.46, 0.0], [-0.02, 0.48, 0.02], [0.0, 0.5, 0.0]], np.float64)
    save('vae_z', z_fixed)
    save('vae_root', root_fixed)
    save('vae_nlogits', vae.model.n_head(z_fixed).numpy())
    out_cards = vae.decode_cards(z_fixed, root_fixed)
    save('vae_count', np.array([len(c) for c in out_cards], np.int64))
    save('vae_points', np.concatenate(out_cards, 0))

    # ---- colour model: one regression pass on the sampled clean tokens
    col, col_a = load_color_dit(paths.ckpt('color_model', 'model.safetensors'), dim_tok, 'cpu')
    img = np.random.default_rng(5).integers(0, 256, size=(a.res, a.res, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:a.res, 0:a.res]
    matte = ((yy - a.res * 0.45) ** 2 + (xx - a.res * 0.5) ** 2) < (a.res * 0.4) ** 2
    img_t = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float()
    matte_t = torch.from_numpy(matte.copy())[None]
    save('color_img', img_t)
    save('color_matte', matte.astype(np.uint8))
    cctx = col.encode_ctx(dino, img_t, matte_t)
    save('color_ctx', cctx)
    xc = sign_token_normals(tok, clean_tokens(tok, x_final))
    lab = col(xc, cctx)
    save('color_lab', lab)
    save('color_srgb', lab_to_srgb(lab[0].float().numpy()))
    save('color_raw', col.raw(xc, cctx))

    # ---- chain DiT: one denoiser evaluation and the full 50-step chain sample
    ch_ck = m(os.path.join(ARGS.models, 'ckpt', 'chain_model', 'model.safetensors'))
    ca = ch_ck['args']
    chain = ChainDiT(ch_ck['dim'], ca['dim'], ca['depth'], ca['heads']).eval()
    chain.load_state_dict(ch_ck['ema'])
    cards_clean = clean_tokens(tok, x_final)
    cctx_chain = chain.encode_cond(cards_clean)
    save('chain_ctx', cctx_chain)
    cg = sub_rng(99)
    chain_noise = torch.randn(1, 48, ch_ck['dim'], generator=cg)
    save('chain_noise', chain_noise)
    save('chain_denoise_cond', chain(chain_noise, t0, cctx_chain, torch.zeros(1, dtype=torch.bool)))
    save('chain_denoise_uncond', chain(chain_noise, t0, cctx_chain, torch.ones(1, dtype=torch.bool)))
    xch = euler_cfg(chain, cctx_chain, chain_noise, ARGS.steps, ARGS.cfg)
    save('chain_final', xch)
    stats = chain_stats()
    chains = chain_decode(xch[0].float().numpy(), stats)
    save('chain_count', np.array([len(c['joints']) for c in chains], np.int64))
    save('chain_stiffness', np.array([c['stiffness'] for c in chains], np.float64))
    save('chain_joints', np.concatenate([np.asarray(c['joints'], np.float64) for c in chains], 0)
         if chains else np.zeros((0, 3)))
    save('chain_path', np.concatenate([np.asarray(c['path'], np.float64) for c in chains], 0)
         if chains else np.zeros((0, 3)))
    with open(os.path.join(ARGS.out, 'manifest.json'), 'w') as f:
        json.dump(dict(steps=ARGS.steps, cfg=ARGS.cfg, tensors=MANIFEST), f, indent=1)
    print('dumped', len(MANIFEST), 'tensors to', ARGS.out)
    print('cards', len(cards), 'chains', len(chains), 'colour lab shape', tuple(lab.shape))


if __name__ == '__main__':
    main()
