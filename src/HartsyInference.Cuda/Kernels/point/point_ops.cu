// The per-element and per-point embeddings: uniform fake quantization (IBackend.FakeQuantizeUniform) and the Fourier feature
// embedding (IBackend.FourierEmbed). By-value parameters only, so a launch captures into a graph. F32.
//
//   fake_quantize_uniform_f32  out[i] = low + trunc((clamp(in[i], low, high) − low) / range · steps) / steps · range
//   fourier_embed_f32          per point and coordinate: the coordinate (when included), then sin(x · 2^b) and cos(x · 2^b)
//                              coordinate-major over the bands; one thread per (point, coordinate)
//
// Contracts and host references: FakeQuantizeContract / FakeQuantizeReference, FourierEmbedContract / FourierEmbedReference.
// Build: ./build.sh (nvrtc through ../build_common.sh).

struct FakeQuantizeArgs
{
    float low;
    float high;
    float range;
    float steps;
    unsigned long long count;
};

struct FourierEmbedArgs
{
    unsigned long long count;
    unsigned int dims;
    unsigned int bands;
    unsigned int width;
    unsigned int includeInput;
};

extern "C" __global__ void fake_quantize_uniform_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    FakeQuantizeArgs a)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.count) return;
    float v = input[i];
    v = v > a.high ? a.high : (v < a.low ? a.low : v);
    int b = (int)((v - a.low) / a.range * a.steps);
    output[i] = (float)b / a.steps * a.range + a.low;
}

extern "C" __global__ void fourier_embed_f32(
    float* __restrict__ dst,
    const float* __restrict__ coords,
    FourierEmbedArgs a)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= a.count * a.dims) return;
    unsigned int c = (unsigned int)(gid % a.dims);
    unsigned long long i = gid / a.dims;
    float x = coords[i * a.dims + c];
    unsigned long long o = i * a.width;
    unsigned int sinBase = a.includeInput ? a.dims : 0u, cosBase = sinBase + a.dims * a.bands;
    if (a.includeInput) dst[o + c] = x;
    for (unsigned int band = 0; band < a.bands; band++)
    {
        float v = x * (float)(1u << band);
        dst[o + sinBase + c * a.bands + band] = sinf(v);
        dst[o + cosBase + c * a.bands + band] = cosf(v);
    }
}
