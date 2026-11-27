// Shared by the block-sparse attention shaders (block_sparse_*.comp.glsl).
#ifndef BLOCK_SPARSE_COMMON_GLSL
#define BLOCK_SPARSE_COMMON_GLSL

// The raster row of token `offset` of block `blockIndex` in a grid H tokens high and W wide, cut into bt x bh x bw blocks.
// Blocks and the tokens inside one are numbered in raster order (BlockSparseAttentionOptions.QueryRow / KeyRow).
uint bsaRow(uint blockIndex, uint offset, uint H, uint W, uint bt, uint bh, uint bw) {
    uint blocksHigh = H / bh, blocksWide = W / bw;
    uint frame = blockIndex / (blocksHigh * blocksWide) * bt + offset / (bh * bw);
    uint y = blockIndex / blocksWide % blocksHigh * bh + offset / bw % bh;
    uint x = blockIndex % blocksWide * bw + offset % bw;
    return (frame * H + y) * W + x;
}

// Tests on the bits: without a float-controls mode a driver may assume no NaN reaches a floating-point comparison.
bool bsaIsNaN(float value) { return (floatBitsToUint(value) & 0x7FFFFFFFu) > 0x7F800000u; }
bool bsaIsZero(float value) { return (floatBitsToUint(value) & 0x7FFFFFFFu) == 0u; }

#endif
