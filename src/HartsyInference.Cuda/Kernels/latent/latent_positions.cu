// Position kernels for the single-latent attention path.
//
// rope_interleaved_offset_f32: rotates the [dimOffset, dimOffset + rotaryDim) slice of each vector as adjacent pairs
//   (view_as_complex): v[2i] = e*c - o*s, v[2i+1] = e*s + o*c, with cos/sin [positions, rotaryDim/2]. Negate sin
//   for the inverse. One thread per (position, head, pair); x is updated in place. IEEE multiplies and adds.
//
// window_indices_i32: the sliding-window ring slots per query, -1 for an empty slot (get_window_topk_idxs).
//   Prefill (startPos == 0): row r sees positions max(r-W+1,0)..r. Decode: the whole ring oldest first, slots past
//   startPos empty. One thread per output element; mirrors WindowIndicesReference.Slot.

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
