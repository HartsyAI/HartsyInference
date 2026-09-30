// Position kernels for the single-latent attention path.
//
// rope_interleaved_offset_f32: rotates the [dimOffset, dimOffset + rotaryDim) slice of each vector as adjacent pairs
//   (view_as_complex): v[2i] = e*c - o*s, v[2i+1] = e*s + o*c, with cos/sin [positions, rotaryDim/2]. Negate sin
//   for the inverse. One thread per (position, head, pair); x is updated in place. IEEE multiplies and adds.
//
// window_indices_i32: the sliding-window ring slots per query, -1 for an empty slot (get_window_topk_idxs).
//   Prefill (startPos == 0): row r sees positions max(r-W+1,0)..r. Decode: the whole ring oldest first, slots past
//   startPos empty. One thread per output element; mirrors WindowIndicesReference.Slot.

extern "C" __global__ void rope_interleaved_offset_f32(
    float* __restrict__ x,
    const float* __restrict__ cosTable,
    const float* __restrict__ sinTable,
    long long positions, int heads, int dim, int half, int dimOffset)
{
    const long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= positions * heads * half) return;
    const long long ph = u / half;                // pos * heads + head
    const int i = (int)(u % half);
    const long long pos = ph / heads;
    const float c = cosTable[pos * half + i], s = sinTable[pos * half + i];
    float* v = x + ph * dim + dimOffset;
    const float even = v[2 * i], odd = v[2 * i + 1];
    v[2 * i] = __fsub_rn(__fmul_rn(even, c), __fmul_rn(odd, s));
    v[2 * i + 1] = __fadd_rn(__fmul_rn(even, s), __fmul_rn(odd, c));
}

extern "C" __global__ void window_indices_i32(
    int* __restrict__ indices,
    int windowSize, int startPos, int rows, int cols)
{
    const long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= (long long)rows * cols) return;
    const int row = (int)(u / cols), col = (int)(u % cols);
    int slot;
    if (startPos == 0)
    {
        const int idx = max(row - windowSize + 1, 0) + col;
        slot = idx > row ? -1 : idx;
    }
    else
    {
        const int oldest = startPos % windowSize + 1;
        const int s = col < windowSize - oldest ? oldest + col : col - (windowSize - oldest);
        slot = s > startPos ? -1 : s;
    }
    indices[u] = slot;
}
