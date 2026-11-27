// The Mamba-1 selective scan (IBackend.SelectiveScan), F32. Matches SelectiveScanReference.Apply.
// One warp per (row, channel): lane l holds the states n = l + 32*i for i < SEL_MAX_PER_LANE (so up to 256 states) in
// registers, together with their decay rates, and walks the frames in order. Per frame the warp sums c·h with a shuffle
// reduction and lane 0 writes the output. A carried state is read as the initial state and written back after the last
// frame; with hasState == 0 the state starts at zero and nothing is written back.

#define SEL_MAX_PER_LANE 8     // states <= 256
#define SEL_WARPS_PER_BLOCK 4  // channels per block

__device__ __forceinline__ float sel_warp_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

extern "C" {

__global__ void ssm_selective_scan_f32(
    float* __restrict__ output,        // [rows, frames, channels]; distinct from u
    const float* __restrict__ u,       // [rows, frames, channels]
    const float* __restrict__ delta,   // [rows, frames, channels], softplus already applied
    const float* __restrict__ a,       // [channels, states], negative
    const float* __restrict__ b,       // [rows, frames, states]
    const float* __restrict__ c,       // [rows, frames, states]
    const float* __restrict__ d,       // [channels]
    const float* __restrict__ z,       // [rows, frames, channels]; read only when hasZ
    float* __restrict__ state,         // [rows, channels, states], in place; touched only when hasState
    unsigned int hasZ,
    unsigned int hasState,
    unsigned int rows,
    unsigned int frames,
    unsigned int channels,
    unsigned int states)
{
    unsigned int ch = blockIdx.x * SEL_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    unsigned int row = blockIdx.y;
    unsigned int lane = threadIdx.x & 31u;
    if (ch >= channels || row >= rows) return;

    const float* rate = a + (size_t)ch * states;
    float* carried = hasState ? state + ((size_t)row * channels + ch) * states : nullptr;
    float h[SEL_MAX_PER_LANE];
    float r[SEL_MAX_PER_LANE];
#pragma unroll
    for (int i = 0; i < SEL_MAX_PER_LANE; i++)
    {
        unsigned int n = lane + 32u * i;
        bool live = n < states;
        h[i] = (live && hasState) ? carried[n] : 0.0f;
        r[i] = live ? rate[n] : 0.0f;
    }
    float skip = d[ch];

    for (unsigned int t = 0; t < frames; t++)
    {
        size_t at = ((size_t)row * frames + t) * channels + ch;
        const float* bt = b + ((size_t)row * frames + t) * states;
        const float* ct = c + ((size_t)row * frames + t) * states;
        float step = delta[at];
        float x = u[at];
        float drive = step * x;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < SEL_MAX_PER_LANE; i++)
        {
            unsigned int n = lane + 32u * i;
            if (n < states)
            {
                h[i] = expf(step * r[i]) * h[i] + drive * __ldg(bt + n);
                acc += h[i] * __ldg(ct + n);
            }
        }
        acc = sel_warp_sum(acc);
        if (lane == 0)
        {
            float y = acc + x * skip;
            if (hasZ)
            {
                float g = z[at];
                y *= g / (1.0f + expf(-g));
            }
            output[at] = y;
        }
    }

    if (hasState)
    {
#pragma unroll
        for (int i = 0; i < SEL_MAX_PER_LANE; i++)
        {
            unsigned int n = lane + 32u * i;
            if (n < states) carried[n] = h[i];
        }
    }
}

}  // extern "C"
