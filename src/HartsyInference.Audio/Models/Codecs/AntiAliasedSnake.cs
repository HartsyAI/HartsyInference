using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>BigVGAN anti-aliased SnakeBeta (<c>Activation1d</c>): upsample by the ratio with a Kaiser-sinc filter, SnakeBeta, downsample by the ratio — length-preserving on <c>[1, C, T]</c>.</summary>
/// <remarks>The up pass is always non-causal (replicate pad, crop); the down pass is the causal low-pass (replicate left pad <c>K-1</c>, none on the right) when <c>causalDownPad</c> is set. Alpha and beta are stored log-scaled and exponentiated at load.</remarks>
public sealed unsafe class AntiAliasedSnake : IDisposable
{
    private readonly int _channels;
    private readonly int _ratio;
    private readonly int _kernel;
    private readonly int _upPad;
    private readonly int _upCropLeft;
    private readonly int _upCropRight;
    private readonly int _downPadLeft;
    private readonly int _downPadRight;
    private Tensor? _alpha;
    private Tensor? _beta;
    private Tensor? _upWeight;
    private Tensor? _downWeight;

    public AntiAliasedSnake(int channels, int ratio = 2, int kernel = 12, bool causalDownPad = false)
    {
        if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels));
        if (ratio < 1) throw new ArgumentOutOfRangeException(nameof(ratio));
        if (kernel < ratio) throw new ArgumentOutOfRangeException(nameof(kernel), "kernel must be at least the ratio.");
        _channels = channels;
        _ratio = ratio;
        _kernel = kernel;
        _upPad = kernel / ratio - 1;
        _upCropLeft = _upPad * ratio + (kernel - ratio) / 2;
        _upCropRight = _upPad * ratio + (kernel - ratio + 1) / 2;
        _downPadLeft = causalDownPad ? kernel - 1 : kernel / 2 - (kernel % 2 == 0 ? 1 : 0);
        _downPadRight = causalDownPad ? 0 : kernel / 2;
    }

    /// <summary>Left replicate padding of the down pass (<c>K-1</c> causal, <c>K/2-1</c> for an even symmetric kernel).</summary>
    public int DownPadLeft => _downPadLeft;

    /// <summary>Right replicate padding of the down pass (0 when causal).</summary>
    public int DownPadRight => _downPadRight;

    /// <summary>Kaiser-windowed sinc low-pass summing to 1 (reference <c>kaiser_sinc_filter1d</c>), in double precision.</summary>
    public static float[] KaiserSincFilter(double cutoff, double halfWidth, int kernelSize)
    {
        int halfSize = kernelSize / 2;
        double amplitude = 2.285 * (halfSize - 1) * Math.PI * 4.0 * halfWidth + 7.95;
        double beta = amplitude > 50.0 ? 0.1102 * (amplitude - 8.7)
            : amplitude >= 21.0 ? 0.5842 * Math.Pow(amplitude - 21.0, 0.4) + 0.07886 * (amplitude - 21.0) : 0.0;
        float[] filter = new float[kernelSize];
        if (cutoff == 0.0) return filter;
        double[] raw = new double[kernelSize];
        double sum = 0.0;
        double denom = BesselI0(beta);
        double mid = (kernelSize - 1) / 2.0;
        for (int i = 0; i < kernelSize; i++)
        {
            double r = mid == 0.0 ? 0.0 : (i - mid) / mid;
            double window = BesselI0(beta * Math.Sqrt(Math.Max(0.0, 1.0 - r * r))) / denom;
            double time = (kernelSize % 2 == 0 ? -halfSize + i + 0.5 : i - halfSize) * 2.0 * cutoff;
            double sinc = time == 0.0 ? 1.0 : Math.Sin(Math.PI * time) / (Math.PI * time);
            raw[i] = 2.0 * cutoff * window * sinc;
            sum += raw[i];
        }
        for (int i = 0; i < kernelSize; i++) filter[i] = (float)(raw[i] / sum);
        return filter;
    }

    /// <summary>Loads <c>{prefix}.act.alpha</c>/<c>.act.beta</c> (log-scale), <c>{prefix}.upsample.filter</c> and <c>{prefix}.downsample.lowpass.filter</c> (<c>[1, 1, K]</c> buffers); a missing key throws.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor alpha = ExpLoad(w[$"{prefix}.act.alpha"]);
        Tensor beta = ExpLoad(w[$"{prefix}.act.beta"]);
        Tensor up = BroadcastFilter(w[$"{prefix}.upsample.filter"], _ratio);
        Tensor down = BroadcastFilter(w[$"{prefix}.downsample.lowpass.filter"], 1f);
        Dispose();
        _alpha = alpha;
        _beta = beta;
        _upWeight = up;
        _downWeight = down;
    }

    /// <summary>Applies the activation to <paramref name="x"/> (<c>[1, C, T]</c>); the caller owns the result and <paramref name="x"/> is untouched.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        if (_alpha is null) throw new InvalidOperationException("AntiAliasedSnake weights not loaded.");
        if (x.Shape.Rank != 3 || (int)x.Shape[0] != 1 || (int)x.Shape[1] != _channels)
            throw new ArgumentException($"Expected [1, {_channels}, T]; got {x.Shape}.", nameof(x));
        int t = (int)x.Shape[2];
        using Tensor padded = ReplicatePadTime(backend, x, _upPad, _upPad);
        int upLen = (t + 2 * _upPad - 1) * _ratio + _kernel - _upCropLeft - _upCropRight;
        Tensor up = new(new TensorShape(1, _channels, upLen), DType.F32);
        backend.ConvTranspose1d(up, padded, _upWeight!, null, _ratio, _upCropLeft, _upCropRight, 1, _channels);
        backend.Snake(up, up, _alpha, _beta);
        using Tensor upPadded = ReplicatePadTime(backend, up, _downPadLeft, _downPadRight);
        up.Dispose();
        int outLen = ((int)upPadded.Shape[2] - _kernel) / _ratio + 1;
        Tensor down = new(new TensorShape(1, _channels, outLen), DType.F32);
        backend.Conv1d(down, upPadded, _downWeight!, null, _ratio, 0, 0, 1, _channels);
        return down;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_alpha, _beta, _upWeight, _downWeight];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        _alpha?.Dispose();
        _beta?.Dispose();
        _upWeight?.Dispose();
        _downWeight?.Dispose();
        _alpha = _beta = _upWeight = _downWeight = null;
    }

    /// <summary>Edge-replicate pads <c>[1, C, T]</c> along time on-device: transpose, slice the edge rows, one concat, transpose back.</summary>
    private static Tensor ReplicatePadTime(IBackend backend, Tensor x, int padLeft, int padRight)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        if (padLeft == 0 && padRight == 0)
        {
            Tensor copy = new(x.Shape, DType.F32);
            backend.CopyTo(copy, x);
            return copy;
        }
        using Tensor xt = new(new TensorShape(t, c), DType.F32);
        backend.Transpose2D(xt, x, c, t);
        using Tensor first = new(new TensorShape(1, c), DType.F32);
        backend.SliceRows(first, xt, 0);
        using Tensor last = new(new TensorShape(1, c), DType.F32);
        backend.SliceRows(last, xt, t - 1);
        Tensor[] parts = new Tensor[padLeft + 1 + padRight];
        for (int i = 0; i < padLeft; i++) parts[i] = first;
        parts[padLeft] = xt;
        for (int i = 0; i < padRight; i++) parts[padLeft + 1 + i] = last;
        using Tensor paddedT = new(new TensorShape(t + padLeft + padRight, c), DType.F32);
        backend.Concat(paddedT, parts, 0);
        Tensor padded = new(new TensorShape(1, c, t + padLeft + padRight), DType.F32);
        backend.Transpose2D(padded, paddedT, t + padLeft + padRight, c);
        return padded;
    }

    private Tensor ExpLoad(Tensor raw)
    {
        Tensor f32 = WhisperOps.EnsureF32(raw);
        if (f32.ElementCount != _channels)
            throw new ArgumentException($"Snake parameter has {f32.ElementCount} elements; expected {_channels}.");
        Tensor o = new(new TensorShape(_channels), DType.F32);
        float* src = (float*)f32.DataPointer;
        float* dst = (float*)o.DataPointer;
        for (int i = 0; i < _channels; i++) dst[i] = MathF.Exp(src[i]);
        if (!ReferenceEquals(f32, raw)) f32.Dispose();
        return o;
    }

    private Tensor BroadcastFilter(Tensor raw, float scale)
    {
        Tensor f32 = WhisperOps.EnsureF32(raw);
        if (f32.ElementCount != _kernel)
            throw new ArgumentException($"Filter has {f32.ElementCount} taps; expected {_kernel}.");
        Tensor o = new(new TensorShape(_channels, 1, _kernel), DType.F32);
        float* src = (float*)f32.DataPointer;
        float* dst = (float*)o.DataPointer;
        for (int c = 0; c < _channels; c++)
            for (int k = 0; k < _kernel; k++) dst[c * _kernel + k] = src[k] * scale;
        if (!ReferenceEquals(f32, raw)) f32.Dispose();
        return o;
    }

    private static double BesselI0(double x)
    {
        double sum = 1.0, term = 1.0, half = x / 2.0;
        for (int k = 1; k < 64 && term >= sum * 1e-17; k++)
        {
            term *= half / k * (half / k);
            sum += term;
        }
        return sum;
    }
}
