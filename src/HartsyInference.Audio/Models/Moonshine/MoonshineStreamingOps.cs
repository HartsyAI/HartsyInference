using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Moonshine;

/// <summary>Low-level helpers specific to the 2nd-gen streaming Moonshine encoder (verified against <c>transformers/models/moonshine_streaming/modeling_moonshine_streaming.py</c> 5.14.1): frame-wise CMVN + asinh compression (the raw-waveform front end) and the encoder's "unit-offset gamma" LayerNorm variant.</summary>
internal static unsafe class MoonshineStreamingOps
{
    /// <summary>Per-frame cepstral mean/variance normalization (<c>MoonshineStreamingFrameCMVN</c>, eps=1e-6): for each row of <paramref name="frameLen"/> raw samples, zero-mean then divide by RMS.</summary>
    public static void FrameCmvn(Tensor output, Tensor input, int frameLen)
    {
        long rows = input.ElementCount / frameLen;
        float* x = (float*)input.DataPointer;
        float* y = (float*)output.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            float* xr = x + r * frameLen;
            float* yr = y + r * frameLen;
            float mean = 0f;
            for (int i = 0; i < frameLen; i++) mean += xr[i];
            mean /= frameLen;
            float sq = 0f;
            for (int i = 0; i < frameLen; i++) { float c = xr[i] - mean; sq += c * c; }
            float invRms = 1f / MathF.Sqrt(sq / frameLen + 1e-6f);
            for (int i = 0; i < frameLen; i++) yr[i] = (xr[i] - mean) * invRms;
        }
    }

    /// <summary>Learned-scale asinh compression (<c>MoonshineStreamingAsinhCompression</c>): <c>asinh(exp(logK) * x)</c> in place, where <paramref name="logK"/> is a loaded scalar weight.</summary>
    public static void AsinhCompressInPlace(Tensor t, float logK)
    {
        float k = MathF.Exp(logK);
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) p[i] = MathF.Asinh(k * p[i]);
    }

    /// <summary>SiLU (x * sigmoid(x)), in place.</summary>
    public static void SiluInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float x = p[i];
            p[i] = x / (1f + MathF.Exp(-x));
        }
    }

    /// <summary>The streaming encoder's <c>MoonshineStreamingLayerNorm</c>: parameterless LayerNorm followed by a "unit-offset" learned scale <c>gamma + 1.0</c> (gamma is zero-initialized at training start, so the layer begins as identity) — NOT the same as <see cref="MoonshineOps.LayerNormNoBias"/>, which multiplies by <c>gamma</c> directly.</summary>
    public static void UnitOffsetLayerNorm(Tensor output, Tensor input, Tensor gamma, float eps = 1e-5f)
    {
        int d = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / d;
        float* x = (float*)input.DataPointer;
        float* y = (float*)output.DataPointer;
        float* g = (float*)gamma.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            float* xr = x + r * d;
            float* yr = y + r * d;
            float mean = 0f;
            for (int i = 0; i < d; i++) mean += xr[i];
            mean /= d;
            float sq = 0f;
            for (int i = 0; i < d; i++) { float c = xr[i] - mean; sq += c * c; }
            float invStd = 1f / MathF.Sqrt(sq / d + eps);
            for (int i = 0; i < d; i++) yr[i] = (xr[i] - mean) * invStd * (g[i] + 1f);
        }
    }

    /// <summary>Builds a dense additive <c>[1, 1, seqLen, seqLen]</c> sliding-window mask (0 where allowed, -inf where masked); applied unconditionally, unlike the HF reference which skips the window when no external padding mask is supplied (a reference-code quirk, not the intended architecture).</summary>
    public static Tensor BuildSlidingWindowMask(int seqLen, int left, int right)
    {
        Tensor mask = new(new TensorShape(1, 1, seqLen, seqLen), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int q = 0; q < seqLen; q++)
        {
            int rowOff = q * seqLen;
            for (int kv = 0; kv < seqLen; kv++)
            {
                int dist = q - kv;
                bool allowed = (dist >= 0 && dist < left) || (dist < 0 && -dist < right);
                p[rowOff + kv] = allowed ? 0f : float.NegativeInfinity;
            }
        }
        return mask;
    }
}
