"""Reference dump of the Dora-VAE 1.1 shape encoder (the point-cloud condition of the HairDeck point-cloud model).

Runs the official encoder code of HairDeck (hairdeck/encoders/dora.py, Apache-2.0) on the CPU in fp32 for a fixed,
seeded point cloud, and writes the input and the output condition tokens as raw little-endian float32 blobs plus
manifest.json (name -> dtype, shape, file), which the C# parity test reads.

The checkpoint is the released Lightning file (dora_vae_1_1.ckpt). It is read with torch.load(weights_only=True), so
only tensors are unpickled. This script is reference tooling; the engine never calls it.

Run (isolated interpreter; the HairDeck source is a clone at cb20f7073419a33c9ab88f22f686fcb9060a99f0):
  python -I tools/parity/dump_dora_encoder.py --src <clone> --ckpt <dora_vae_1_1.ckpt> --out <scratch dir>
"""
import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='HairDeck source clone at the pinned commit')
    p.add_argument('--ckpt', required=True, help='dora_vae_1_1.ckpt (Seed3D/Dora-VAE-1.1)')
    p.add_argument('--out', required=True, help='output folder')
    p.add_argument('--points', type=int, default=1500, help='points in the seeded test cloud')
    p.add_argument('--tokens', type=int, default=256, help='query points per branch (released: 256)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def load_tensors_only(path):
    """state_dict tensors of the Lightning checkpoint, with the 'shape_model.' prefix stripped. Only the tensor rebuild
    and storage classes are resolved; every other global in the pickle (config objects, the system class) is replaced by
    a placeholder that is never used, so no code from the checkpoint runs."""
    import collections
    import pickle
    import torch

    def placeholder(module, name):
        key = (module, name)
        if key not in placeholders:
            placeholders[key] = type(name, (), {'__init__': lambda self, *a, **k: None,
                                                '__setstate__': lambda self, state: None})
        return placeholders[key]

    placeholders = {}

    class TensorsOnlyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                return torch._utils._rebuild_tensor_v2
            if module == 'torch' and name.endswith('Storage'):
                return getattr(torch, name)
            if module == 'collections' and name in ('OrderedDict', 'defaultdict'):
                return getattr(collections, name)
            return placeholder(module, name)

    class PickleModule:
        Unpickler = TensorsOnlyUnpickler
        load = staticmethod(pickle.load)

    ckpt = torch.load(path, map_location='cpu', weights_only=False, pickle_module=PickleModule)
    sd = ckpt['state_dict']
    prefix = 'shape_model.'
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.src)

    import torch
    from hairdeck.encoders import dora

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rng = np.random.default_rng(args.seed)
    xyz = rng.uniform(-1.0, 1.0, size=(args.points, 3)).astype(np.float32)
    normals = rng.normal(size=(args.points, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    cloud = np.concatenate([xyz, normals], axis=1).astype(np.float32)[None]  # [1, N, 6]

    W = load_tensors_only(args.ckpt)
    model = dora.ShapeVAE('dora', W, 'cpu', qk_norm=any('q_norm' in k for k in W))
    x = torch.from_numpy(cloud)
    with torch.no_grad():
        tokens = model.encode(x, x, args.tokens)  # the released condition feeds the same points to both branches
    tokens_np = tokens.float().cpu().numpy()[0].astype('<f4')

    manifest = {}

    def write(name, array):
        file_name = f'{name}.f32'
        with open(os.path.join(args.out, file_name), 'wb') as f:
            f.write(np.ascontiguousarray(array, dtype='<f4').tobytes())
        manifest[name] = {'dtype': 'float32', 'shape': list(array.shape), 'file': file_name}

    write('points', cloud[0])
    write('dora_tokens', tokens_np)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump({'tokens_per_branch': args.tokens, 'seed': args.seed, 'tensors': manifest}, f, indent=2)
    print('wrote', args.out, 'tokens', tokens_np.shape)


if __name__ == '__main__':
    main()
