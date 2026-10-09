// HartsyInference expert FFN kernels, F32, sm_75 (Turing) and newer.
//
// One expert is y = Down · (act(min(Gate·x, gateMax)) * clamp(Up·x, upMin, upMax)). The GEMM accumulates each output
// element sequentially over k, the same order as ExpertProgramReference, so the only difference from the CPU oracle is
// rounding in the transcendental functions. Plain F32 throughout: no tensor cores, no TF32, no weight casts.

// y[r, i] = sum_k x[r, k] * w[i, k] for r < rows, i < n. w is row-major [n, k], x is row-major [rows, k].
// One thread per output element; blockIdx.y selects the token row.
extern "C" __global__ void expert_gemm_f32(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ w,
                                           int rows, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (i >= n || r >= rows) return;
    const float* xr = x + (long long)r * k;
    const float* wr = w + (long long)i * k;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc += wr[j] * xr[j];
    y[(long long)r * n + i] = acc;
}

// Activation codes match ExpertActivation: 0 Silu, 1 GeluTanh, 2 Relu, 3 ReluSquared.
__device__ __forceinline__ float expert_act(int activation, float v)
{
    switch (activation)
    {
        case 0: return v / (1.0f + expf(-v));
        case 1: return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        case 2: return fmaxf(v, 0.0f);
        default:
        {
            float r = fmaxf(v, 0.0f);
            return r * r;
        }
    }
}

// h[i] = act(min(gate[i], gateMax)) * clamp(up[i], upMin, upMax). Infinite bounds disable a clamp: fminf and fmaxf
// return the other operand for an infinity.
extern "C" __global__ void expert_act_f32(float* __restrict__ h, const float* __restrict__ gate, const float* __restrict__ up,
                                          long long count, int activation, float gateMax, float upMin, float upMax)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float g = fminf(gate[i], gateMax);
    float u = fminf(fmaxf(up[i], upMin), upMax);
    h[i] = expert_act(activation, g) * u;
}
