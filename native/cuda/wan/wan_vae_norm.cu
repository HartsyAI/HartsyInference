// Wan 2.2 VAE channel-wise RMS norm (vae.py `RMS_norm`: F.normalize over dim=1 * gamma * sqrt(C)).
// x, out: [B, C, spatial] with C on dim 1 (stride = spatial). gamma: [C] (or null → 1).
// Per (b, s) position: denom = max(sqrt(sum_c x^2), eps); scale = sqrt(C)/denom; out[c] = x[c]*scale*gamma[c].
// Note sqrt(C)/L2 == 1/rms, i.e. this is an RMS norm over the channel axis. Matches the host reference exactly
// (double accumulation, eps applied to the L2 not the mean).
extern "C" __global__ void wan_vae_rms_norm_channel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    int C, long spatial, float eps, float sqrtC, long numPos)
{
    long pos = blockIdx.x * (long)blockDim.x + threadIdx.x;   // 0 .. B*spatial
    if (pos >= numPos) return;
    long b = pos / spatial;
    long s = pos % spatial;
    long baseB = b * (long)C * spatial;

    double sumSq = 0.0;
    for (int ci = 0; ci < C; ci++)
    {
        float v = x[baseB + (long)ci * spatial + s];
        sumSq += (double)v * (double)v;
    }
    float denom = fmaxf((float)sqrt(sumSq), eps);
    float scale = sqrtC / denom;

    for (int ci = 0; ci < C; ci++)
    {
        long off = baseB + (long)ci * spatial + s;
        float gv = gamma ? gamma[ci] : 1.0f;
        out[off] = x[off] * scale * gv;
    }
}

// Wan2.2 VAE unpatchify (pixel-shuffle): [b, c*p*p, t, h, w] -> [b, c, t, h*p, w*p].
// Channel unpack: for out spatial (hh*p+q, ww*p+r), oc = ci*p*p + r*p + q. One thread per output element.
extern "C" __global__ void wan_vae_unpatchify(
    float* __restrict__ out, const float* __restrict__ x,
    int b, int c, int t, int h, int w, int p, long numOut)
{
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= numOut) return;
    int outW = w * p, outH = h * p;
    int ow = (int)(idx % outW); long tmp = idx / outW;
    int oh = (int)(tmp % outH); tmp /= outH;
    int ti = (int)(tmp % t);    tmp /= t;
    int ci = (int)(tmp % c);    tmp /= c;
    int bi = (int)tmp;
    int hh = oh / p, q = oh % p, ww = ow / p, r = ow % p;
    int packedC = c * p * p;
    int oc = ci * p * p + r * p + q;
    long srcOff = ((((long)bi * packedC + oc) * t + ti) * h + hh) * (long)w + ww;
    out[idx] = x[srcOff];
}

// Wan2.2 VAE attention qkv split: src [bt, 3c, h, w] -> q,k,v each [bt, 1, hw, c] (channel<->token transpose).
extern "C" __global__ void wan_vae_split_qkv(
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    const float* __restrict__ src, int bt, int c, int hw, long numEl)
{
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= numEl) return;               // numEl = bt*c*hw
    int token = (int)(idx % hw); long tmp = idx / hw;
    int ci = (int)(tmp % c);     int i = (int)(tmp / c);
    long frame = hw;
    long srcBase = (long)i * 3 * c * frame + token;
    long dstOff = ((long)i * hw + token) * c + ci;
    q[dstOff] = src[srcBase + (long)ci * frame];
    k[dstOff] = src[srcBase + (long)(c + ci) * frame];
    v[dstOff] = src[srcBase + (long)(2 * c + ci) * frame];
}

// Inverse: attn [bt, 1, hw, c] -> out [bt, c, h, w].
extern "C" __global__ void wan_vae_tokens_to_frame(
    float* __restrict__ out, const float* __restrict__ a, int bt, int c, int hw, long numEl)
{
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= numEl) return;               // numEl = bt*c*hw
    int token = (int)(idx % hw); long tmp = idx / hw;
    int ci = (int)(tmp % c);     int i = (int)(tmp / c);
    out[((long)i * c + ci) * hw + token] = a[((long)i * hw + token) * c + ci];
}
