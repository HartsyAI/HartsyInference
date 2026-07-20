using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>BypassModule: a learned per-channel highway/interpolation between two same-shaped tensors,
/// <c>out = orig + (src - orig) * bypass_scale</c>. Used both as an intra-layer highway
/// (<c>bypass_mid</c>/<c>bypass</c>) and as a stage-level "recombine after down/upsample" combiner
/// (<c>out_combiner</c>). At inference the scale is always the raw stored parameter (no clamping/skip —
/// those are training-only).</summary>
internal sealed unsafe class ZipformerBypass
{
    private readonly int _dim;
    private Tensor? _scale;

    public ZipformerBypass(int dim) => _dim = dim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string key)
        => _scale = WhisperOps.EnsureF32(w[key]);

    /// <summary><paramref name="orig"/>/<paramref name="src"/> are <c>[1, T, dim]</c>. Returns a new
    /// <c>[1, T, dim]</c> tensor (does not dispose inputs).</summary>
    public Tensor Forward(Tensor orig, Tensor src, int t)
    {
        Tensor output = new(new TensorShape(1, t, _dim), DType.F32);
        float* op = (float*)output.DataPointer;
        float* origP = (float*)orig.DataPointer;
        float* srcP = (float*)src.DataPointer;
        float* scaleP = (float*)_scale!.DataPointer;
        for (int i = 0; i < t; i++)
        {
            long off = (long)i * _dim;
            for (int d = 0; d < _dim; d++)
                op[off + d] = origP[off + d] + (srcP[off + d] - origP[off + d]) * scaleP[d];
        }
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_scale is not null) yield return _scale;
    }
}

/// <summary>SimpleDownsample: right-pads by repeating the last frame up to a multiple of
/// <paramref name="factor"/>, then takes a fixed (content-independent) learned softmax-weighted average over
/// each group of <c>factor</c> frames. Channel count unchanged; time axis shrinks to
/// <c>ceil(T/factor)</c>.</summary>
internal sealed unsafe class ZipformerSimpleDownsample
{
    private readonly int _dim, _factor;
    private Tensor? _bias;

    public ZipformerSimpleDownsample(int dim, int factor) { _dim = dim; _factor = factor; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        => _bias = WhisperOps.EnsureF32(w[$"{prefix}.bias"]);

    /// <summary><paramref name="x"/> is <c>[1, T, dim]</c>. Returns <c>[1, ceil(T/factor), dim]</c>.</summary>
    public Tensor Forward(Tensor x, int t, out int outT)
    {
        int ds = _factor;
        outT = (t + ds - 1) / ds;
        float* xp = (float*)x.DataPointer;
        float* biasP = (float*)_bias!.DataPointer;

        float[] weights = new float[ds];
        float maxB = float.NegativeInfinity;
        for (int k = 0; k < ds; k++) if (biasP[k] > maxB) maxB = biasP[k];
        float sum = 0f;
        for (int k = 0; k < ds; k++) { weights[k] = MathF.Exp(biasP[k] - maxB); sum += weights[k]; }
        for (int k = 0; k < ds; k++) weights[k] /= sum;

        Tensor output = new(new TensorShape(1, outT, _dim), DType.F32);
        float* op = (float*)output.DataPointer;
        for (int i = 0; i < outT; i++)
        {
            float* outRow = op + (long)i * _dim;
            for (int d = 0; d < _dim; d++) outRow[d] = 0f;
            for (int k = 0; k < ds; k++)
            {
                int srcIdx = i * ds + k;
                if (srcIdx >= t) srcIdx = t - 1;   // right-pad by repeating the last frame
                float* srcRow = xp + (long)srcIdx * _dim;
                float wgt = weights[k];
                for (int d = 0; d < _dim; d++) outRow[d] += wgt * srcRow[d];
            }
        }
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>SimpleUpsample: parameter-free nearest-neighbour repeat, each frame repeated <paramref
/// name="factor"/> times (<c>T → T·factor</c>). Channel count unchanged.</summary>
internal static unsafe class ZipformerSimpleUpsample
{
    public static Tensor Forward(Tensor x, int t, int dim, int factor)
    {
        int outT = t * factor;
        Tensor output = new(new TensorShape(1, outT, dim), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int i = 0; i < t; i++)
        {
            float* srcRow = xp + (long)i * dim;
            for (int k = 0; k < factor; k++)
            {
                float* dstRow = op + (long)(i * factor + k) * dim;
                Buffer.MemoryCopy(srcRow, dstRow, dim * sizeof(float), dim * sizeof(float));
            }
        }
        return output;
    }
}
