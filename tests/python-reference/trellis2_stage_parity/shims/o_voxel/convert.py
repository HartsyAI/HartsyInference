import os
import types

import numpy as np
import torch
from typing import *

_UPSTREAM = os.path.join(os.environ["TRELLIS2_UPSTREAM"], "o-voxel", "o_voxel", "convert", "flexible_dual_grid.py")
_source = open(_UPSTREAM, encoding="utf-8").read()
_body = _source[_source.index("def flexible_dual_grid_to_mesh"):]

_INSERT = """    hashmap = _init_hashmap(grid_size, 2 * N, device=coords.device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1), *grid_size.tolist())
"""
_LOOKUP = """    connected_voxel_hash_key = torch.cat([
        torch.zeros((M * 4, 1), dtype=torch.int, device=coords.device),
        connected_voxel.reshape(-1, 3)
    ], dim=1)
    connected_voxel_indices = _C.hashmap_lookup_3d_cuda(*hashmap, connected_voxel_hash_key, *grid_size.tolist()).reshape(M, 4).int()
"""
_VALID = "quad_indices = connected_voxel_indices[connected_voxel_valid].int()"
for needle in (_INSERT, _LOOKUP, _VALID):
    if needle not in _body:
        raise SystemExit("upstream flexible_dual_grid_to_mesh changed: the hashmap stand-in no longer matches; update shims/o_voxel/convert.py")

_body = _body.replace(_INSERT, """    G = grid_size.tolist()
    ckeys = (coords[:, 0].long() * G[1] + coords[:, 1].long()) * G[2] + coords[:, 2].long()
    corder = torch.argsort(ckeys)
    cskeys = ckeys[corder]
""").replace(_LOOKUP, """    q = connected_voxel.reshape(-1, 3).long()
    inside = ((q >= 0) & (q < torch.tensor(G))).all(dim=1)
    qkeys = (q[:, 0] * G[1] + q[:, 1]) * G[2] + q[:, 2]
    pos = torch.searchsorted(cskeys, qkeys).clamp(max=N - 1)
    hit = inside & (cskeys[pos] == qkeys)
    found = torch.where(hit, corder[pos], torch.full_like(pos, 0xffffffff))
    connected_voxel_indices = found.reshape(M, 4).long()
""").replace(_VALID, "quad_indices = connected_voxel_indices[connected_voxel_valid].long()")

exec(compile(_body, _UPSTREAM, "exec"), globals())
