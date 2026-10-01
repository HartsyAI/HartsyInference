using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Whisper's GELU is the exact erf form at all four places it runs: both conv-stem activations, the encoder MLP
/// and the decoder MLP. The tanh approximation transcribes clean speech almost the same and only flips near-tie
/// tokens, so nothing else would notice a swap back. A tiny synthetic model — attention zeroed, convolution centre taps
/// and MLP projections set to identity — reduces each network to its GELUs and layer norms, and its outputs are checked
/// against the exact form written out in double precision, which the tanh form misses by far more than the
/// tolerance.</summary>
public sealed class WhisperExactGeluTests
{
    private const int D = 4;
    private const int MelFrames = 6;
    private const int Vocab = 16;
    private const int Positions = 8;
    private const float Tolerance = 1e-5f;

    private static readonly WhisperConfig Config = new()
    {
        NumMelBins = D, HiddenSize = D, NumHeads = 1, IntermediateSize = D, EncoderLayers = 1, DecoderLayers = 1,
        MaxAudioPositions = Positions, MaxTextPositions = Positions, VocabSize = Vocab,
    };

    [Fact]
    public void Encoder_StemAndMlp_UseTheExactGelu()
    {
        double[,] mel = new double[D, MelFrames];
        for (int m = 0; m < D; m++)
            for (int f = 0; f < MelFrames; f++)
                mel[m, f] = -3 + 6.0 * ((m * 7 + f * 3) % 13) / 12;

        Dictionary<string, Tensor> weights = EncoderWeights();
        try
        {
            using CpuBackend backend = new();
            using WhisperEncoder encoder = new(Config);
            encoder.LoadWeights(weights);
            using Tensor input = new(new TensorShape(1, D, MelFrames), DType.F32);
            Span<float> values = input.AsSpan<float>();
            for (int m = 0; m < D; m++)
                for (int f = 0; f < MelFrames; f++)
                    values[m * MelFrames + f] = (float)mel[m, f];

            using Tensor output = encoder.Forward(backend, input);
            AssertExactNotTanh(output.AsSpan<float>(), ExpectedEncoder(mel, GeluExact), ExpectedEncoder(mel, GeluTanh));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Decoder_Mlp_UsesTheExactGelu()
    {
        int[] prompt = [3, 7];
        Dictionary<string, Tensor> weights = DecoderWeights(out double[,] tokens, out double[,] positions);
        try
        {
            using CpuBackend backend = new();
            using WhisperDecoder decoder = new(Config);
            decoder.LoadWeights(weights);
            using Tensor encoded = new(new TensorShape(1, 3, D), DType.F32);
            encoded.AsSpan<float>().Clear();
            using WhisperDecoder.DecodeState state = decoder.StartDecode(backend, encoded);
            using Tensor logits = decoder.DecodeStep(backend, prompt, state);

            int last = prompt.Length - 1;
            AssertExactNotTanh(logits.AsSpan<float>()[..Vocab],
                ExpectedLogits(tokens, positions, prompt[last], last, GeluExact),
                ExpectedLogits(tokens, positions, prompt[last], last, GeluTanh));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    /// <summary>The encoder with identity convolutions, zero attention and identity MLP projections:
    /// <c>LN(h + GELU(LN(h)))</c> with <c>h = GELU(GELU(mel))</c> at every other frame (stride 2).</summary>
    private static double[] ExpectedEncoder(double[,] mel, Func<double, double> gelu)
    {
        int frames = (MelFrames + 2 - 3) / 2 + 1;
        List<double> output = [];
        for (int t = 0; t < frames; t++)
        {
            double[] h = new double[D];
            for (int c = 0; c < D; c++) h[c] = gelu(gelu(mel[c, 2 * t]));
            output.AddRange(LayerNorm(MlpResidual(h, gelu)));
        }
        return [.. output];
    }

    /// <summary>The decoder with zero self- and cross-attention and identity MLP projections at one position: the tied
    /// logits of <c>LN(h + GELU(LN(h)))</c>, <c>h = token embedding + position embedding</c>.</summary>
    private static double[] ExpectedLogits(double[,] tokens, double[,] positions, int token, int position,
        Func<double, double> gelu)
    {
        double[] h = new double[D];
        for (int c = 0; c < D; c++) h[c] = tokens[token, c] + positions[position, c];
        double[] y = LayerNorm(MlpResidual(h, gelu));
        double[] logits = new double[Vocab];
        for (int v = 0; v < Vocab; v++)
            for (int c = 0; c < D; c++)
                logits[v] += y[c] * tokens[v, c];
        return logits;
    }

    private static double[] MlpResidual(double[] h, Func<double, double> gelu)
    {
        double[] normed = LayerNorm(h);
        double[] result = new double[D];
        for (int c = 0; c < D; c++) result[c] = h[c] + gelu(normed[c]);
        return result;
    }

    private static double[] LayerNorm(double[] x)
    {
        double mean = x.Average();
        double variance = x.Sum(v => (v - mean) * (v - mean)) / x.Length;
        return [.. x.Select(v => (v - mean) / Math.Sqrt(variance + Config.LayerNormEps))];
    }

    private static double GeluExact(double x) => 0.5 * x * (1 + Erf(x / Math.Sqrt(2)));

    private static double GeluTanh(double x) => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));

    /// <summary>erf by its Taylor series, exact to double rounding for the |x| ≤ 3 used here.</summary>
    private static double Erf(double x)
    {
        double term = x, sum = x;
        for (int n = 1; n < 200; n++)
        {
            term *= -x * x / n;
            sum += term / (2 * n + 1);
        }
        return 2 / Math.Sqrt(Math.PI) * sum;
    }

    private static void AssertExactNotTanh(ReadOnlySpan<float> actual, double[] exact, double[] tanh)
    {
        Assert.Equal(exact.Length, actual.Length);
        double worst = 0, separation = 0;
        for (int i = 0; i < exact.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(actual[i] - exact[i]));
            separation = Math.Max(separation, Math.Abs(tanh[i] - exact[i]));
        }
        Assert.True(separation > 4 * Tolerance, $"the case cannot tell the forms apart: they differ by at most {separation:E2}");
        Assert.True(worst <= Tolerance, $"max abs {worst:E2} from the exact-GELU reference (the tanh form is {separation:E2} away)");
    }

    private static Dictionary<string, Tensor> EncoderWeights()
    {
        const string p = "model.encoder";
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            [$"{p}.conv1.weight"] = CenterTapIdentity(),
            [$"{p}.conv1.bias"] = Filled([D], 0),
            [$"{p}.conv2.weight"] = CenterTapIdentity(),
            [$"{p}.conv2.bias"] = Filled([D], 0),
            [$"{p}.embed_positions.weight"] = Filled([Positions, D], 0),
            [$"{p}.layer_norm.weight"] = Filled([D], 1),
            [$"{p}.layer_norm.bias"] = Filled([D], 0),
        };
        AddLayer(w, $"{p}.layers.0", cross: false);
        return w;
    }

    private static Dictionary<string, Tensor> DecoderWeights(out double[,] tokens, out double[,] positions)
    {
        const string p = "model.decoder";
        Random rng = new(5);
        tokens = new double[Vocab, D];
        positions = new double[Positions, D];
        // Held at float precision so the reference sees the very values the engine loads.
        for (int v = 0; v < Vocab; v++)
            for (int c = 0; c < D; c++)
                tokens[v, c] = (float)(rng.NextDouble() * 5 - 2.5);
        for (int t = 0; t < Positions; t++)
            for (int c = 0; c < D; c++)
                positions[t, c] = (float)(rng.NextDouble() * 2 - 1);
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            [$"{p}.embed_tokens.weight"] = FromValues(tokens),
            [$"{p}.embed_positions.weight"] = FromValues(positions),
            [$"{p}.layer_norm.weight"] = Filled([D], 1),
            [$"{p}.layer_norm.bias"] = Filled([D], 0),
        };
        AddLayer(w, $"{p}.layers.0", cross: true);
        return w;
    }

    /// <summary>A layer whose attention outputs zero (zero out-projections) and whose MLP projections are identities.</summary>
    private static void AddLayer(Dictionary<string, Tensor> w, string prefix, bool cross)
    {
        foreach (string attention in cross ? (string[])["self_attn", "encoder_attn"] : ["self_attn"])
        {
            w[$"{prefix}.{attention}_layer_norm.weight"] = Filled([D], 1);
            w[$"{prefix}.{attention}_layer_norm.bias"] = Filled([D], 0);
            foreach (string projection in (string[])["q_proj", "k_proj", "v_proj", "out_proj"])
            {
                w[$"{prefix}.{attention}.{projection}.weight"] = Filled([D, D], 0);
                if (projection != "k_proj") w[$"{prefix}.{attention}.{projection}.bias"] = Filled([D], 0);
            }
        }
        w[$"{prefix}.final_layer_norm.weight"] = Filled([D], 1);
        w[$"{prefix}.final_layer_norm.bias"] = Filled([D], 0);
        w[$"{prefix}.fc1.weight"] = Identity();
        w[$"{prefix}.fc1.bias"] = Filled([D], 0);
        w[$"{prefix}.fc2.weight"] = Identity();
        w[$"{prefix}.fc2.bias"] = Filled([D], 0);
    }

    private static Tensor Identity()
    {
        Tensor t = Filled([D, D], 0);
        Span<float> v = t.AsSpan<float>();
        for (int i = 0; i < D; i++) v[i * D + i] = 1;
        return t;
    }

    /// <summary>A <c>[D, D, 3]</c> convolution that copies channel i to channel i through its centre tap.</summary>
    private static Tensor CenterTapIdentity()
    {
        Tensor t = Filled([D, D, 3], 0);
        Span<float> v = t.AsSpan<float>();
        for (int i = 0; i < D; i++) v[(i * D + i) * 3 + 1] = 1;
        return t;
    }

    private static Tensor Filled(long[] shape, float value)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static Tensor FromValues(double[,] values)
    {
        Tensor t = new(new TensorShape(values.GetLength(0), values.GetLength(1)), DType.F32);
        Span<float> v = t.AsSpan<float>();
        for (int r = 0; r < values.GetLength(0); r++)
            for (int c = 0; c < values.GetLength(1); c++)
                v[r * values.GetLength(1) + c] = (float)values[r, c];
        return t;
    }

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values) tensor.Dispose();
    }
}
