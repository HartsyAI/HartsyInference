using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Separation;

/// <summary>The Kim_Vocal_2 spectrogram network (UVR MDX-Net, a TFC-TDF-UNet): <c>[1, 4, F, T]</c> in, same shape out, F32.
/// <code>
///   x = relu(conv1x1(spec)); x = transpose(x)                      # [C, F, T] → [C, T, F]
///   encoder i: x = TFC-TDF(x); skip[i] = x; x = relu(conv2x2 s2(x))
///   bottleneck: x = TFC-TDF(x)
///   decoder i: x = relu(convT2x2 s2(x)) * skip[last-i]; x = TFC-TDF(x)   # multiplicative skip
///   out = conv1x1(transpose(x))
///   TFC-TDF(x): h = 3 × relu(conv3x3(h)); y = relu(affine(Linear2(relu(affine(Linear1(h)))))) ; h + y   # Linear over F
/// </code>
/// Weight names are those <c>KimVocal2OnnxConverter</c> writes. The affines are the inference BatchNorms after each TDF MatMul.
/// <para>The frequency/time transpose writes through reshaped views, which is only safe on the CPU backend (a view as the write target of a
/// GPU op can drop the result); the engine runs this network on the CPU backend and no GPU run is claimed.</para></summary>
public sealed class KimVocal2Network : IDisposable
{
    private const int TfcLayers = 3;

    private readonly Dictionary<string, Tensor> _weights = [];
    private readonly Dictionary<string, Tensor> _affineWeights = [];
    private readonly int _levels;
    private int _disposed;

    /// <summary>Spectrogram channels (real and imaginary of left and right).</summary>
    public int InChannels { get; }

    /// <summary>Frequency bins at the top level (the model's <c>dim_f</c>).</summary>
    public int DimF { get; }

    /// <summary>Encoder levels; the time axis is divided by two at each.</summary>
    public int Levels => _levels;

    /// <summary>Parity probe: called with each TFC-TDF block's output (<c>[1, C, T, F]</c>, borrowed for the call) and its name, in graph order.</summary>
    public Action<string, Tensor>? StageProbe { get; set; }

    /// <summary>Validates <paramref name="weights"/> (every layer present, shapes consistent) and borrows the F32 tensors; the caller keeps them alive.</summary>
    public KimVocal2Network(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        int levels = 0;
        while (weights.ContainsKey($"encoding_blocks.{levels}.tfc.0.weight")) levels++;
        if (levels == 0) throw new InvalidDataException("Kim_Vocal_2 weights contain no encoding_blocks.");
        _levels = levels;

        Tensor first = Get(weights, "first_conv.weight");
        InChannels = (int)first.Shape[1];
        int width0 = (int)first.Shape[0];
        DimF = (int)Get(weights, "encoding_blocks.0.tdf.0.weight").Shape[1];
        Require(first.Shape.Rank == 4 && first.Shape[2] == 1 && first.Shape[3] == 1, "first_conv must be a 1x1 conv");
        Take(weights, "first_conv.weight");
        Take(weights, "first_conv.bias");

        int width = width0;
        int freq = DimF;
        for (int i = 0; i <= levels; i++)
        {
            string prefix = i < levels ? $"encoding_blocks.{i}" : "bottleneck_block";
            width = CheckBlock(weights, prefix, width, freq);
            if (i < levels)
            {
                Tensor down = Get(weights, prefix + ".down.weight");
                Require(down.Shape[1] == width && down.Shape[2] == 2 && down.Shape[3] == 2, $"{prefix}.down must be a 2x2 conv from {width} channels");
                Take(weights, prefix + ".down.weight");
                Take(weights, prefix + ".down.bias");
                width = (int)down.Shape[0];
                freq /= 2;
            }
        }
        for (int i = 0; i < levels; i++)
        {
            string prefix = $"decoding_blocks.{i}";
            Tensor up = Get(weights, prefix + ".up.weight");
            Require(up.Shape[0] == width && up.Shape[2] == 2 && up.Shape[3] == 2, $"{prefix}.up must be a 2x2 transposed conv from {width} channels");
            Take(weights, prefix + ".up.weight");
            Take(weights, prefix + ".up.bias");
            width = (int)up.Shape[1];
            freq *= 2;
            CheckBlock(weights, prefix, width, freq);
        }
        Require(width == width0 && freq == DimF, "the decoder does not return to the first level's width and frequency size");
        Tensor final = Get(weights, "final_conv.weight");
        Require(final.Shape[0] == InChannels && final.Shape[1] == width0, "final_conv must map the first level's width back to the spectrogram channels");
        Take(weights, "final_conv.weight");
        Take(weights, "final_conv.bias");
    }

    /// <summary>Runs the network on <paramref name="spec"/> <c>[1, InChannels, DimF, T]</c>; the result is owned by the caller. T must be divisible by 2^Levels.</summary>
    public Tensor Forward(IBackend backend, Tensor spec)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(spec);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (spec.DType != DType.F32 || spec.Shape.Rank != 4 || spec.Shape[0] != 1 || spec.Shape[1] != InChannels || spec.Shape[2] != DimF)
            throw new ArgumentException($"Expected F32 [1, {InChannels}, {DimF}, T], got {spec.DType} {spec.Shape}.", nameof(spec));
        long time = spec.Shape[3];
        if (time % (1L << _levels) != 0)
            throw new ArgumentException($"T={time} must be divisible by {1 << _levels} (one halving per encoder level).", nameof(spec));

        List<Tensor> skips = [];
        Tensor x = ConvRelu(backend, spec, "first_conv", 1, 0, relu: true);
        try
        {
            x = Replace(x, Transpose(backend, x));
            for (int i = 0; i < _levels; i++)
            {
                string prefix = $"encoding_blocks.{i}";
                x = Replace(x, TfcTdf(backend, x, prefix));
                skips.Add(x);
                x = ConvRelu(backend, x, prefix + ".down", 2, 0, relu: true, stride: 2);
            }
            x = Replace(x, TfcTdf(backend, x, "bottleneck_block"));
            for (int i = 0; i < _levels; i++)
            {
                string prefix = $"decoding_blocks.{i}";
                Tensor up = ConvTransposeRelu(backend, x, prefix + ".up");
                x.Dispose();
                Tensor skip = skips[^1];
                skips.RemoveAt(skips.Count - 1);
                Tensor gated = new(up.Shape, DType.F32);
                backend.Mul(gated, up, skip);
                up.Dispose();
                skip.Dispose();
                x = Replace(gated, TfcTdf(backend, gated, prefix));
            }
            x = Replace(x, Transpose(backend, x));
            Tensor result = ConvRelu(backend, x, "final_conv", 1, 0, relu: false);
            x.Dispose();
            return result;
        }
        catch
        {
            x.Dispose();
            foreach (Tensor s in skips) s.Dispose();
            throw;
        }
    }

    private static Tensor Replace(Tensor old, Tensor replacement)
    {
        old.Dispose();
        return replacement;
    }

    /// <summary>One TFC-TDF block: three 3x3 conv+relu layers, then a residual frequency-dense branch.</summary>
    private Tensor TfcTdf(IBackend backend, Tensor x, string prefix)
    {
        Tensor h = ConvRelu(backend, x, $"{prefix}.tfc.0", 3, 1, relu: true);
        for (int j = 1; j < TfcLayers; j++) h = Replace(h, ConvRelu(backend, h, $"{prefix}.tfc.{j}", 3, 1, relu: true));
        Tensor tdf = Tdf(backend, h, prefix);
        backend.Add(tdf, h, tdf);
        h.Dispose();
        StageProbe?.Invoke(prefix, tdf);
        return tdf;
    }

    /// <summary>Linear over the frequency axis, affine, relu; then back up to the full frequency size, affine, relu.</summary>
    private Tensor Tdf(IBackend backend, Tensor h, string prefix)
    {
        long channels = h.Shape[1], time = h.Shape[2], freq = h.Shape[3];
        Tensor midLinear = _weights[$"{prefix}.tdf.0.weight"];
        long mid = midLinear.Shape[0];
        Tensor a = new(new TensorShape(1, channels, time, mid), DType.F32);
        backend.Linear(a, h, midLinear, null);
        AffineRelu(backend, a, $"{prefix}.tdf.1");
        Tensor b = new(h.Shape, DType.F32);
        backend.Linear(b, a, _weights[$"{prefix}.tdf.2.weight"], null);
        a.Dispose();
        AffineRelu(backend, b, $"{prefix}.tdf.3");
        return b;
    }

    /// <summary>In place: <c>x = relu(scale * x + shift)</c> per channel, as a 1x1 depthwise conv.</summary>
    private void AffineRelu(IBackend backend, Tensor x, string name)
    {
        using Tensor tmp = new(x.Shape, DType.F32);
        backend.Conv2dDepthwise(tmp, x, _affineWeights[name], _weights[name + ".shift"], 1, 1, 0, 0);
        backend.LeakyRelu(x, tmp, 0f);
    }

    private Tensor ConvRelu(IBackend backend, Tensor x, string name, int kernel, int pad, bool relu, int stride = 1)
    {
        Tensor weight = _weights[name + ".weight"];
        long outH = (x.Shape[2] + 2 * pad - kernel) / stride + 1, outW = (x.Shape[3] + 2 * pad - kernel) / stride + 1;
        Tensor y = new(new TensorShape(1, weight.Shape[0], outH, outW), DType.F32);
        backend.Conv2D(y, x, weight, _weights[name + ".bias"], stride, stride, pad, pad);
        if (relu) backend.LeakyRelu(y, y, 0f);
        return y;
    }

    private Tensor ConvTransposeRelu(IBackend backend, Tensor x, string name)
    {
        Tensor weight = _weights[name + ".weight"];
        Tensor y = new(new TensorShape(1, weight.Shape[1], x.Shape[2] * 2, x.Shape[3] * 2), DType.F32);
        backend.ConvTranspose2d(y, x, weight, _weights[name + ".bias"], 2, 2, 0, 0);
        backend.LeakyRelu(y, y, 0f);
        return y;
    }

    /// <summary>Swaps the last two axes of <c>[1, C, A, B]</c>.</summary>
    private static Tensor Transpose(IBackend backend, Tensor x)
    {
        long c = x.Shape[1], a = x.Shape[2], b = x.Shape[3];
        Tensor y = new(new TensorShape(1, c, b, a), DType.F32);
        backend.Transpose2D(y.Reshape(new TensorShape(c, b, a)), x.Reshape(new TensorShape(c, a, b)), (int)a, (int)b);
        return y;
    }

    private int CheckBlock(IReadOnlyDictionary<string, Tensor> weights, string prefix, int width, int freq)
    {
        for (int j = 0; j < TfcLayers; j++)
        {
            Tensor w = Get(weights, $"{prefix}.tfc.{j}.weight");
            Require(w.Shape.Rank == 4 && w.Shape[0] == width && w.Shape[1] == width && w.Shape[2] == 3 && w.Shape[3] == 3,
                $"{prefix}.tfc.{j} must be a 3x3 conv of {width} channels, got {w.Shape}");
            Take(weights, $"{prefix}.tfc.{j}.weight");
            Take(weights, $"{prefix}.tfc.{j}.bias");
        }
        Tensor down = Get(weights, $"{prefix}.tdf.0.weight");
        Tensor up = Get(weights, $"{prefix}.tdf.2.weight");
        Require(down.Shape[1] == freq && up.Shape[0] == freq && up.Shape[1] == down.Shape[0],
            $"{prefix}.tdf must map frequency {freq} down and back, got {down.Shape} and {up.Shape}");
        Take(weights, $"{prefix}.tdf.0.weight");
        Take(weights, $"{prefix}.tdf.2.weight");
        foreach (string bn in new[] { $"{prefix}.tdf.1", $"{prefix}.tdf.3" })
        {
            Tensor scale = Get(weights, bn + ".scale");
            Require(scale.ElementCount == width && Get(weights, bn + ".shift").ElementCount == width, $"{bn} must hold {width} channels");
            _affineWeights[bn] = scale.Reshape(new TensorShape(width, 1, 1, 1));
            Take(weights, bn + ".shift");
        }
        return width;
    }

    private static Tensor Get(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t)) throw new InvalidDataException($"Kim_Vocal_2 weights are missing '{key}'.");
        if (t.DType != DType.F32) throw new InvalidDataException($"Kim_Vocal_2 weight '{key}' is {t.DType}; the network runs F32.");
        return t;
    }

    private void Take(IReadOnlyDictionary<string, Tensor> weights, string key) => _weights[key] = Get(weights, key);

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidDataException($"Kim_Vocal_2 weights: {message}.");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _affineWeights.Values) t.Dispose();
        _affineWeights.Clear();
        _weights.Clear();
    }
}
