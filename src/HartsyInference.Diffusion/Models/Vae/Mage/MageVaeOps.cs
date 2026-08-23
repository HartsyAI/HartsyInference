using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.Mage;

/// <summary>Host-side helpers shared by the Mage VAE encoder/decoder and their DiCo blocks (one copy, per the shared-primitive rule).</summary>
internal static unsafe class MageVaeOps
{
    /// <summary>Global average pool over the spatial positions: <c>[b, c, h, w] → [b, c, 1, 1]</c>.</summary>
    internal static Tensor GlobalAvgPool(Tensor x, int b, int c, int hw)
    {
        Tensor outp = new(new TensorShape(b, c, 1, 1), DType.F32);
        float* src = (float*)x.DataPointer; float* dst = (float*)outp.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ch = 0; ch < c; ch++)
            {
                double sum = 0; long baseO = (long)(bi * c + ch) * hw;
                for (int s = 0; s < hw; s++) sum += src[baseO + s];
                dst[bi * c + ch] = (float)(sum / hw);
            }
        return outp;
    }

    /// <summary>Scales each channel plane of <paramref name="x"/> in place by the matching <c>[b, c, 1, 1]</c> gate.</summary>
    internal static void ScaleByChannel(Tensor x, Tensor ca, int b, int c, int hw)
    {
        float* p = (float*)x.DataPointer; float* g = (float*)ca.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ch = 0; ch < c; ch++)
            {
                float s = g[bi * c + ch]; long baseO = (long)(bi * c + ch) * hw;
                for (int k = 0; k < hw; k++) p[baseO + k] *= s;
            }
    }

    /// <summary>Affine channel-dim LayerNorm2d: per (batch, spatial) position normalize over C, then <c>y = norm·weight[c] + bias[c]</c>.</summary>
    internal static void ChannelLayerNormAffine(Tensor input, Tensor output, Tensor weight, Tensor bias, int b, int c, int hw)
    {
        float* src = (float*)input.DataPointer; float* dst = (float*)output.DataPointer;
        float* gw = (float*)weight.DataPointer; float* gb = (float*)bias.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int p = 0; p < hw; p++)
            {
                double mean = 0;
                for (int ch = 0; ch < c; ch++) mean += src[((long)(bi * c + ch) * hw) + p];
                mean /= c;
                double var = 0;
                for (int ch = 0; ch < c; ch++) { double d = src[((long)(bi * c + ch) * hw) + p] - mean; var += d * d; }
                float inv = (float)(1.0 / Math.Sqrt(var / c + 1e-6));
                for (int ch = 0; ch < c; ch++)
                {
                    long o = ((long)(bi * c + ch) * hw) + p;
                    dst[o] = (float)((src[o] - mean) * inv) * gw[ch] + gb[ch];
                }
            }
    }

    /// <summary>t_embedder(0) → Linear → SiLU → Linear: the t=0 sinusoidal embedding is <c>[sin(0)=0 (×128) ; cos(0)=1 (×128)]</c>, so the "cos" half is all ones and the "sin" half all zeros.</summary>
    internal static Tensor TimestepEmbedZero(IBackend backend, int b, Tensor w1, Tensor? b1, Tensor w2, Tensor? b2, int hidden)
    {
        Tensor sinEmb = new(new TensorShape(b, 256), DType.F32);
        float* p = (float*)sinEmb.DataPointer;
        for (int i = 0; i < b; i++) for (int j = 0; j < 256; j++) p[i * 256 + j] = j >= 128 ? 1f : 0f;
        Tensor t1 = new(new TensorShape(b, hidden), DType.F32);
        backend.Linear(t1, sinEmb, w1, b1); sinEmb.Dispose();
        Tensor act = new(t1.Shape, DType.F32);
        backend.Silu(act, t1); t1.Dispose();
        Tensor c = new(new TensorShape(b, hidden), DType.F32);
        backend.Linear(c, act, w2, b2); act.Dispose();
        return c;
    }
}
