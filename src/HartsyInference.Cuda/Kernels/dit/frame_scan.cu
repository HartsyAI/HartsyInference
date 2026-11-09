// Frame-wise bidirectional delta-rule scan (IBackend.FrameDeltaScan, SANA-WM Gated-DeltaNet), F32.
//
//   frame_scan_scale_keys_f32   kb[n, h, :] = beta[h, t, s] * key[n, h, :]  (n = t*spatial + s)
//   frame_scan_recurrence_f32   per head: S <- d_t*(S - G_t*S) + U_t over frames, forward (inclusive) then reverse
//                               (exclusive); writes F_t + R_t per frame
//
// The backend computes G_t = sum_s beta k k^T and U_t = sum_s beta k v^T for every (head, frame) with batched GEMMs
// before the recurrence and reads q*(F_t + R_t) out with a batched GEMM after it, so only the frame-to-frame state
// update is sequential. Contract and host reference: IBackend.FrameDeltaScan and FrameDeltaScanReference.
//
// Build: src/HartsyInference.Cuda/Kernels/dit/build.sh.

#define FRAME_SCAN_COLS 32        // state columns (value dims) per block
#define FRAME_SCAN_MAX_KEY 128    // key dims per head; one thread per state row

extern "C" {

__global__ void frame_scan_scale_keys_f32(
    float* __restrict__ out,          // [N, heads*keyDim]
    const float* __restrict__ key,    // [N, heads*keyDim]
    const float* __restrict__ beta,   // [heads, frames, spatial]
    unsigned int frames,
    unsigned int spatial,
    unsigned int heads,
    unsigned int keyDim)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row = (unsigned long long)heads * keyDim;
    unsigned long long total = (unsigned long long)frames * spatial * row;
    if (idx >= total) return;
    unsigned long long n = idx / row;
    unsigned int h = (unsigned int)((idx - n * row) / keyDim);
    unsigned int t = (unsigned int)(n / spatial);
    unsigned int s = (unsigned int)(n - (unsigned long long)t * spatial);
    out[idx] = beta[((unsigned long long)h * frames + t) * spatial + s] * key[idx];
}

// grid (ceil(valueDim / FRAME_SCAN_COLS), heads), block FRAME_SCAN_MAX_KEY threads; thread r owns state row r.
// gramT[h, t] is G_t transposed ([keyDim, keyDim], gramT[j][r] = G[r][j]) so a warp's row reads coalesce.
__global__ void frame_scan_recurrence_f32(
    float* __restrict__ states,           // [heads, frames, keyDim, valueDim]  F_t + R_t
    const float* __restrict__ gramT,      // [heads, frames, keyDim, keyDim]
    const float* __restrict__ injection,  // [heads, frames, keyDim, valueDim]
    const float* __restrict__ decay,      // [heads, frames]
    unsigned int frames,
    unsigned int keyDim,
    unsigned int valueDim)
{
    __shared__ __align__(16) float tile[2][FRAME_SCAN_MAX_KEY][FRAME_SCAN_COLS];
    const unsigned int h = blockIdx.y;
    const unsigned int c0 = blockIdx.x * FRAME_SCAN_COLS;
    const unsigned int r = threadIdx.x;
    const bool active = r < keyDim;
    const unsigned int cols = min((unsigned int)FRAME_SCAN_COLS, valueDim - c0);
    const unsigned long long gramStride = (unsigned long long)keyDim * keyDim;
    const unsigned long long stateStride = (unsigned long long)keyDim * valueDim;

    for (int pass = 0; pass < 2; pass++)
    {
        const bool reverse = pass == 1;
        int cur = 0;
        for (unsigned int c = 0; c < FRAME_SCAN_COLS; c++)
            tile[0][r][c] = 0.0f;
        __syncthreads();

        for (unsigned int step = 0; step < frames; step++)
        {
            const unsigned int t = reverse ? frames - 1 - step : step;
            const unsigned long long ht = (unsigned long long)h * frames + t;
            float* stateRow = states + ht * stateStride + (unsigned long long)r * valueDim + c0;

            // Reverse: frame t reads the state before its own update (frames after t only).
            if (reverse && active)
            {
                for (unsigned int c = 0; c < cols; c++)
                    stateRow[c] += tile[cur][r][c];
            }

            float acc[FRAME_SCAN_COLS];
#pragma unroll
            for (int c = 0; c < FRAME_SCAN_COLS; c++)
                acc[c] = 0.0f;
            if (active)
            {
                const float* g = gramT + ht * gramStride + r;
                for (unsigned int j = 0; j < keyDim; j++)
                {
                    float gj = g[(unsigned long long)j * keyDim];
                    const float4* srow = reinterpret_cast<const float4*>(&tile[cur][j][0]);
#pragma unroll
                    for (int c4 = 0; c4 < FRAME_SCAN_COLS / 4; c4++)
                    {
                        float4 sv = srow[c4];
                        acc[4 * c4 + 0] += gj * sv.x;
                        acc[4 * c4 + 1] += gj * sv.y;
                        acc[4 * c4 + 2] += gj * sv.z;
                        acc[4 * c4 + 3] += gj * sv.w;
                    }
                }
            }

            const float d = decay[ht];
            const float* u = injection + ht * stateStride + (unsigned long long)r * valueDim + c0;
            const int nxt = cur ^ 1;
#pragma unroll
            for (int c = 0; c < FRAME_SCAN_COLS; c++)
            {
                float next = 0.0f;
                if (active && (unsigned int)c < cols)
                    next = d * (tile[cur][r][c] - acc[c]) + u[c];
                tile[nxt][r][c] = next;
            }

            // Forward: frame t reads the state after its own update.
            if (!reverse && active)
            {
                for (unsigned int c = 0; c < cols; c++)
                    stateRow[c] = tile[nxt][r][c];
            }

            cur = nxt;
            __syncthreads();
        }
    }
}

} // extern "C"
