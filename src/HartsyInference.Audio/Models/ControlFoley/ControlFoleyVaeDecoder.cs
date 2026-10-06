using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Decoder half of the ControlFoley mel VAE (<c>Decoder1D</c> with magnitude-preserving convolutions):
/// latent <c>[1, EmbedDim, T]</c> to the de-normalised mel <c>[1, DataDim, 2T]</c>.</summary>
/// <remarks>The released weights are stored un-normalised; the forced weight normalisation of
/// <c>MPConv1D.remove_weight_norm</c> (and the <c>learnable_gain</c> of the output conv) is folded in at load.</remarks>
public sealed unsafe class ControlFoleyVaeDecoder : IDisposable
{
    private const float PixelNormEps = 1e-4f;
    private const float SiluScale = 1f / 0.596f;
    private const float SumMix = 0.3f;

    private readonly ControlFoleyVaeConfig _cfg;
    private readonly List<Tensor> _owned = [];
    private MpConv? _convIn;
    private ResBlock? _mid1;
    private ResBlock? _mid2;
    private AttnBlock? _midAttn;
    private ResBlock[][] _blocks = [];
    private AttnBlock?[][] _attn = [];
    private MpConv?[] _upsample = [];
    private MpConv? _convOut;
    private Tensor? _mean;
    private Tensor? _std;
    private int _disposed;

    private sealed record MpConv(Tensor Weight, int Kernel);

    private sealed record ResBlock(int InDim, int OutDim, MpConv Conv1, MpConv Conv2, MpConv? Shortcut);

    private sealed record AttnBlock(int Channels, MpConv Qkv, MpConv ProjOut);

    public ControlFoleyVaeDecoder(ControlFoleyVaeConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        if (cfg.ChMult.Length == 0) throw new ArgumentException("ChMult must not be empty.", nameof(cfg));
        _cfg = cfg;
    }

    /// <summary>Loads <c>{prefix}decoder.*</c>, <c>{prefix}data_mean</c> and <c>{prefix}data_std</c>; a missing key throws.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        int levels = _cfg.ChMult.Length;
        int blockIn = _cfg.HiddenDim * _cfg.ChMult[levels - 1];
        _convIn = Conv(w, $"{prefix}decoder.conv_in", _cfg.EmbedDim, blockIn, 3);
        _mid1 = Res(w, $"{prefix}decoder.mid.block_1", blockIn, blockIn);
        _midAttn = Attn(w, $"{prefix}decoder.mid.attn_1", blockIn);
        _mid2 = Res(w, $"{prefix}decoder.mid.block_2", blockIn, blockIn);
        _blocks = new ResBlock[levels][];
        _attn = new AttnBlock?[levels][];
        _upsample = new MpConv?[levels];
        for (int level = levels - 1; level >= 0; level--)
        {
            int blockOut = _cfg.HiddenDim * _cfg.ChMult[level];
            _blocks[level] = new ResBlock[_cfg.NumResBlocks + 1];
            _attn[level] = new AttnBlock?[_cfg.NumResBlocks + 1];
            for (int b = 0; b <= _cfg.NumResBlocks; b++)
            {
                _blocks[level][b] = Res(w, $"{prefix}decoder.up.{level}.block.{b}", blockIn, blockOut);
                blockIn = blockOut;
                if (Array.IndexOf(_cfg.AttnLayers, level) >= 0) _attn[level][b] = Attn(w, $"{prefix}decoder.up.{level}.attn.{b}", blockIn);
            }
            if (Array.IndexOf(_cfg.DownLayers, level - 1) >= 0)
                _upsample[level] = Conv(w, $"{prefix}decoder.up.{level}.upsample.conv", blockIn, blockIn, 3);
        }
        float gain = ReadScalar(w[$"{prefix}decoder.learnable_gain"]) + 1f;
        _convOut = Conv(w, $"{prefix}decoder.conv_out", blockIn, _cfg.DataDim, 3, gain);
        _mean = Own(CopyF32(w[$"{prefix}data_mean"], _cfg.DataDim));
        _std = Own(CopyF32(w[$"{prefix}data_std"], _cfg.DataDim));
    }

    /// <summary>Decodes <paramref name="z"/> <c>[1, EmbedDim, T]</c>; the caller owns the result <c>[1, DataDim, 2T]</c>.
    /// <paramref name="tap"/> receives <c>conv_in</c>, <c>mid1</c>, <c>mid_attn</c>, <c>mid2</c>, <c>level{i}</c>,
    /// <c>up{i}</c> before activation clipping, and the final <c>mel_norm</c>.</summary>
    public Tensor Decode(IBackend backend, Tensor z, Action<string, Tensor>? tap = null)
    {
        if (_convIn is null || _convOut is null) throw new InvalidOperationException("ControlFoleyVaeDecoder weights not loaded.");
        if (z.Shape.Rank != 3 || (int)z.Shape[0] != 1 || (int)z.Shape[1] != _cfg.EmbedDim)
            throw new ArgumentException($"Expected latent [1, {_cfg.EmbedDim}, T]; got {z.Shape}.", nameof(z));
        Tensor h = Convolve(backend, z, _convIn);
        tap?.Invoke("conv_in", h);
        h = Replace(h, ResForward(backend, h, _mid1!));
        tap?.Invoke("mid1", h);
        h = Replace(h, AttnForward(backend, h, _midAttn!));
        tap?.Invoke("mid_attn", h);
        h = Replace(h, ResForward(backend, h, _mid2!));
        tap?.Invoke("mid2", h);
        backend.Clamp(h, h, -_cfg.ClipAct, _cfg.ClipAct);
        for (int level = _cfg.ChMult.Length - 1; level >= 0; level--)
        {
            for (int b = 0; b <= _cfg.NumResBlocks; b++)
            {
                h = Replace(h, ResForward(backend, h, _blocks[level][b]));
                if (b == _cfg.NumResBlocks) tap?.Invoke($"level{level}", h);
                AttnBlock? attn = _attn[level][b];
                if (attn is not null) h = Replace(h, AttnForward(backend, h, attn));
                backend.Clamp(h, h, -_cfg.ClipAct, _cfg.ClipAct);
            }
            MpConv? up = _upsample[level];
            if (up is null) continue;
            h = Replace(h, Convolve(backend, UpsampleNearest(h), up));
            tap?.Invoke($"up{level}", h);
        }
        SiluInPlace(backend, h);
        Tensor mel = Convolve(backend, h, _convOut);
        h.Dispose();
        tap?.Invoke("mel_norm", mel);
        Denormalize(mel);
        return mel;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        GC.SuppressFinalize(this);
    }

    private static Tensor Replace(Tensor old, Tensor next)
    {
        old.Dispose();
        return next;
    }

    private static Tensor Convolve(IBackend backend, Tensor x, MpConv conv)
    {
        int pad = conv.Kernel / 2;
        Tensor y = new(new TensorShape(1, (int)conv.Weight.Shape[0], (int)x.Shape[2]), DType.F32);
        backend.Conv1d(y, x, conv.Weight, null, 1, pad, pad, 1, 1);
        return y;
    }

    private static void SiluInPlace(IBackend backend, Tensor x)
    {
        backend.Silu(x, x);
        backend.Scale(x, x, SiluScale);
    }

    private Tensor ResForward(IBackend backend, Tensor input, ResBlock block)
    {
        Tensor x = PixelNorm(input);
        Tensor h = new(x.Shape, DType.F32);
        backend.CopyTo(h, x);
        SiluInPlace(backend, h);
        Tensor c1 = Convolve(backend, h, block.Conv1);
        h.Dispose();
        SiluInPlace(backend, c1);
        Tensor c2 = Convolve(backend, c1, block.Conv2);
        c1.Dispose();
        if (block.Shortcut is not null) x = Replace(x, Convolve(backend, x, block.Shortcut));
        MpSum(x, c2);
        c2.Dispose();
        return x;
    }

    private Tensor AttnForward(IBackend backend, Tensor x, AttnBlock block)
    {
        int c = block.Channels;
        int t = (int)x.Shape[2];
        using Tensor y = Convolve(backend, x, block.Qkv);
        using Tensor q = new(new TensorShape(1, 1, t, c), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, t, c), DType.F32);
        using Tensor v = new(new TensorShape(1, 1, t, c), DType.F32);
        float* yp = (float*)y.DataPointer;
        Tensor[] parts = [q, k, v];
        float invSqrtC = 1f / MathF.Sqrt(c);
        for (int j = 0; j < 3; j++)
        {
            float* dst = (float*)parts[j].DataPointer;
            for (int ti = 0; ti < t; ti++)
            {
                double sum = 0.0;
                for (int ci = 0; ci < c; ci++)
                {
                    float value = yp[(long)(3 * ci + j) * t + ti];
                    sum += (double)value * value;
                }
                float inv = 1f / (PixelNormEps + (float)Math.Sqrt(sum) * invSqrtC);
                for (int ci = 0; ci < c; ci++) dst[(long)ti * c + ci] = yp[(long)(3 * ci + j) * t + ti] * inv;
            }
        }
        using Tensor attended = new(new TensorShape(1, 1, t, c), DType.F32);
        backend.ScaledDotProductAttention(attended, q, k, v, null, invSqrtC);
        using Tensor channelsFirst = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(channelsFirst, attended, t, c);
        Tensor h = Convolve(backend, channelsFirst, block.ProjOut);
        MpSum(h, x, swap: true);
        return h;
    }

    /// <summary>Channel-wise pixel norm: <c>x / (eps + ||x||_c / sqrt(C))</c> per time step.</summary>
    private static Tensor PixelNorm(Tensor x)
    {
        int c = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        Tensor o = new(x.Shape, DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)o.DataPointer;
        double[] sums = new double[t];
        for (int ci = 0; ci < c; ci++)
        {
            float* row = src + (long)ci * t;
            for (int ti = 0; ti < t; ti++) sums[ti] += (double)row[ti] * row[ti];
        }
        float[] inv = new float[t];
        float invSqrtC = 1f / MathF.Sqrt(c);
        for (int ti = 0; ti < t; ti++) inv[ti] = 1f / (PixelNormEps + (float)Math.Sqrt(sums[ti]) * invSqrtC);
        for (int ci = 0; ci < c; ci++)
        {
            float* row = src + (long)ci * t;
            float* orow = dst + (long)ci * t;
            for (int ti = 0; ti < t; ti++) orow[ti] = row[ti] * inv[ti];
        }
        return o;
    }

    /// <summary>In place <c>a := (0.7 a + 0.3 b) / sqrt(0.58)</c>; with <paramref name="swap"/> the roles of the two
    /// operands are exchanged so the residual <paramref name="b"/> keeps the 0.7 weight.</summary>
    private static void MpSum(Tensor a, Tensor b, bool swap = false)
    {
        float norm = 1f / MathF.Sqrt((1f - SumMix) * (1f - SumMix) + SumMix * SumMix);
        float wa = swap ? SumMix : 1f - SumMix;
        float wb = swap ? 1f - SumMix : SumMix;
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        long n = a.ElementCount;
        for (long i = 0; i < n; i++) ap[i] = (wa * ap[i] + wb * bp[i]) * norm;
    }

    private static Tensor UpsampleNearest(Tensor x)
    {
        int c = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        Tensor o = new(new TensorShape(1, c, 2 * t), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            float* row = src + (long)ci * t;
            float* orow = dst + (long)ci * 2 * t;
            for (int ti = 0; ti < t; ti++) orow[2 * ti] = orow[2 * ti + 1] = row[ti];
        }
        return o;
    }

    private void Denormalize(Tensor mel)
    {
        int c = (int)mel.Shape[1];
        int t = (int)mel.Shape[2];
        float* p = (float*)mel.DataPointer;
        float* mean = (float*)_mean!.DataPointer;
        float* std = (float*)_std!.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            float* row = p + (long)ci * t;
            for (int ti = 0; ti < t; ti++) row[ti] = row[ti] * std[ci] + mean[ci];
        }
    }

    private MpConv Conv(IReadOnlyDictionary<string, Tensor> w, string name, int inDim, int outDim, int kernel, float gain = 1f)
    {
        Tensor raw = w[$"{name}.weight"];
        if (raw.Shape.Rank != 3 || (int)raw.Shape[0] != outDim || (int)raw.Shape[1] != inDim || (int)raw.Shape[2] != kernel)
            throw new ArgumentException($"{name}.weight is {raw.Shape}; expected [{outDim}, {inDim}, {kernel}].");
        return new MpConv(Own(ForcedWeightNorm(raw, gain)), kernel);
    }

    private ResBlock Res(IReadOnlyDictionary<string, Tensor> w, string name, int inDim, int outDim)
    {
        MpConv? shortcut = inDim != outDim ? Conv(w, $"{name}.nin_shortcut", inDim, outDim, 1) : null;
        return new ResBlock(inDim, outDim, Conv(w, $"{name}.conv1", inDim, outDim, 3), Conv(w, $"{name}.conv2", outDim, outDim, 3), shortcut);
    }

    private AttnBlock Attn(IReadOnlyDictionary<string, Tensor> w, string name, int channels)
        => new(channels, Conv(w, $"{name}.qkv", channels, 3 * channels, 1), Conv(w, $"{name}.proj_out", channels, channels, 1));

    /// <summary>Per output channel <c>w / (eps * sqrt(fan_in) + ||w||)</c>, times <paramref name="gain"/>.</summary>
    private static Tensor ForcedWeightNorm(Tensor raw, float gain)
    {
        Tensor f32 = WhisperOps.EnsureF32(raw);
        int outDim = (int)f32.Shape[0];
        long fan = f32.ElementCount / outDim;
        Tensor o = new(f32.Shape, DType.F32);
        float* src = (float*)f32.DataPointer;
        float* dst = (float*)o.DataPointer;
        for (int oc = 0; oc < outDim; oc++)
        {
            double sum = 0.0;
            float* row = src + oc * fan;
            for (long i = 0; i < fan; i++) sum += (double)row[i] * row[i];
            double scale = gain / (PixelNormEps * Math.Sqrt(fan) + Math.Sqrt(sum));
            for (long i = 0; i < fan; i++) dst[oc * fan + i] = (float)(row[i] * scale);
        }
        if (!ReferenceEquals(f32, raw)) f32.Dispose();
        return o;
    }

    private static float ReadScalar(Tensor raw)
    {
        Tensor f32 = WhisperOps.EnsureF32(raw);
        float value = ((float*)f32.DataPointer)[0];
        if (!ReferenceEquals(f32, raw)) f32.Dispose();
        return value;
    }

    private static Tensor CopyF32(Tensor raw, int expected)
    {
        if (raw.ElementCount != expected) throw new ArgumentException($"Expected {expected} elements; got {raw.ElementCount}.");
        Tensor f32 = WhisperOps.EnsureF32(raw);
        Tensor o = new(new TensorShape(expected), DType.F32);
        new ReadOnlySpan<float>((void*)f32.DataPointer, expected).CopyTo(new Span<float>((void*)o.DataPointer, expected));
        if (!ReferenceEquals(f32, raw)) f32.Dispose();
        return o;
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }
}
