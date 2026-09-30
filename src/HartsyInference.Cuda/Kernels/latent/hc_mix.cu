// Hyper-connection (mHC) mixing kernels of DeepSeek-V4.1-Flash, matching HcReference.cs.
//
// hc_split_sinkhorn_f32: one thread per token. pre = sigmoid(m0*s0+b)+eps, post = 2*sigmoid(m1*s1+b), comb = row softmax
//   + eps, one column normalization, then iters-1 rounds of row then column normalization, each dividing by sum+eps.
// hc_pre_mix_f32:  out[t,d]   = sum_i pre[t,i] * x[t,i,d]                        (i ascending)
// hc_post_mix_f32: out[t,i,d] = post[t,i] * x[t,d] + sum_j comb[t,j,i] * res[t,j,d]  (j ascending)
// Every multiply and add is IEEE (no contraction) and sums keep the CPU order, so the results are bit-identical to
// the reference except exp. hc <= 8. The post-mix output must not alias the residual.

#define HC_MAX 8

__device__ __forceinline__ float hc_sigmoid(float v)
{
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v)));
}

extern "C" __global__ void hc_split_sinkhorn_f32(
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ comb,
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int tokens, int hc, int iters, float eps)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;
    const int width = (2 + hc) * hc;
    const float* x = mixes + (long long)t * width;
    float c[HC_MAX * HC_MAX];
    for (int i = 0; i < hc; i++)
    {
        pre[(long long)t * hc + i] = __fadd_rn(hc_sigmoid(__fadd_rn(__fmul_rn(x[i], scale[0]), bias[i])), eps);
        post[(long long)t * hc + i] = __fmul_rn(2.0f, hc_sigmoid(__fadd_rn(__fmul_rn(x[hc + i], scale[1]), bias[hc + i])));
    }
    for (int i = 0; i < hc * hc; i++) c[i] = __fadd_rn(__fmul_rn(x[2 * hc + i], scale[2]), bias[2 * hc + i]);
    for (int r = 0; r < hc; r++)
    {
        float mx = c[r * hc];
        for (int k = 1; k < hc; k++) mx = fmaxf(mx, c[r * hc + k]);
        float sum = 0.0f;
        for (int k = 0; k < hc; k++)
        {
            c[r * hc + k] = expf(__fadd_rn(c[r * hc + k], -mx));
            sum = __fadd_rn(sum, c[r * hc + k]);
        }
        for (int k = 0; k < hc; k++) c[r * hc + k] = __fadd_rn(__fdiv_rn(c[r * hc + k], sum), eps);
    }
    for (int it = 0; it < iters; it++)
    {
        if (it > 0)
        {
            for (int r = 0; r < hc; r++)
            {
                float sum = 0.0f;
                for (int k = 0; k < hc; k++) sum = __fadd_rn(sum, c[r * hc + k]);
                for (int k = 0; k < hc; k++) c[r * hc + k] = __fdiv_rn(c[r * hc + k], __fadd_rn(sum, eps));
            }
        }
        for (int k = 0; k < hc; k++)
        {
            float sum = 0.0f;
            for (int r = 0; r < hc; r++) sum = __fadd_rn(sum, c[r * hc + k]);
            for (int r = 0; r < hc; r++) c[r * hc + k] = __fdiv_rn(c[r * hc + k], __fadd_rn(sum, eps));
        }
    }
    for (int i = 0; i < hc * hc; i++) comb[(long long)t * hc * hc + i] = c[i];
}

extern "C" __global__ void hc_pre_mix_f32(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ pre,
    long long tokens, int hc, int dim)
{
    const long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= tokens * dim) return;
    const long long t = u / dim;
    const int d = (int)(u % dim);
    float acc = 0.0f;
    for (int i = 0; i < hc; i++) acc = __fadd_rn(acc, __fmul_rn(pre[t * hc + i], x[(t * hc + i) * dim + d]));
    output[u] = acc;
}

extern "C" __global__ void hc_post_mix_f32(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    long long tokens, int hc, int dim)
{
    const long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= tokens * hc * dim) return;
    const long long ti = u / dim;                 // t * hc + i
    const int d = (int)(u % dim);
    const long long t = ti / hc;
    const int i = (int)(ti % hc);
    float mix = 0.0f;
    for (int j = 0; j < hc; j++)
        mix = __fadd_rn(mix, __fmul_rn(comb[(t * hc + j) * hc + i], residual[(t * hc + j) * dim + d]));
    output[u] = __fadd_rn(__fmul_rn(post[ti], x[t * dim + d]), mix);
}
