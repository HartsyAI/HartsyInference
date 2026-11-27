# LongCat block-sparse attention reference

`dump_longcat_bsa.py` runs the pinned `flash_attn_bsa_3d` on an NVIDIA GPU and writes:

- `--vectors`: `tests/HartsyInference.Core.Tests/Backends/BlockSparseAttentionVectors.cs`, which the CPU lane compares against;
- `--out`: each case's full output (`<case>.output.f32`), selection (`<case>.selected.i32`) and `report.json`. With
  `HARTSY_LONGCAT_BSA_DIR` pointing there, `BlockSparseAttentionReferenceParityTests` compares every output element.

From the repository root, with a LongCat-Video checkout at the commit the script names in `$REPO` and torch 2.6.0 (cu124):

```bash
python tests/python-reference/longcat_bsa/dump_longcat_bsa.py --repo "$REPO" --out "$OUT" \
    --vectors tests/HartsyInference.Core.Tests/Backends/BlockSparseAttentionVectors.cs
HARTSY_LONGCAT_BSA_DIR="$OUT" dotnet test tests/HartsyInference.Core.Tests --filter FullyQualifiedName~BlockSparseAttention
```

The run takes seconds and well under 1 GiB of GPU memory.
