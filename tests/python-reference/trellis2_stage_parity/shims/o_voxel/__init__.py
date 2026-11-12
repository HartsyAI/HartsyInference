"""o_voxel stand-in. `convert.flexible_dual_grid_to_mesh` is upstream's own function, read from the clone at import time with its CUDA
hashmap (`_C.hashmap_*`) swapped for a sorted-key lookup; nothing else of it changes."""
from . import convert
