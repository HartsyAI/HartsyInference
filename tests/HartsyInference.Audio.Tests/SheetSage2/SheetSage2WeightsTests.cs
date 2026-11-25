using HartsyInference.Audio.Models.Mert2;
using HartsyInference.Audio.Models.SheetSage2;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.SheetSage2;

/// <summary>What SheetSage2 reads from its checkpoint and what it refuses. Tiny synthetic weights, no downloads.</summary>
public sealed class SheetSage2WeightsTests : IDisposable
{
    private const int Dim = 8, Inner = 16, MaxTokens = 12;
    private static readonly SheetSage2Config Config = new() { Dim = Dim, IntermediateSize = Inner, NumHeads = 2, Layers = 1, MaxTokens = MaxTokens };
    private static readonly ScoreTokenizer Tokenizer = new();

    private readonly Dictionary<string, Tensor> _half = Checkpoint();
    private readonly Dictionary<string, Tensor> _full = [];

    public SheetSage2WeightsTests()
    {
        foreach ((string key, Tensor value) in _half) _full[key] = value.CastTo(DType.F32);
    }

    public void Dispose()
    {
        foreach (Tensor t in _half.Values) t.Dispose();
        foreach (Tensor t in _full.Values) t.Dispose();
    }

    [Fact]
    public void A16BitCheckpoint_DecodesLikeItsF32Form()
    {
        Assert.Equal(Logits(_full), Logits(_half));
    }

    [Fact]
    public void ACheckpoint_OutlivesTheModel()
    {
        foreach (Dictionary<string, Tensor> weights in new[] { _half, _full })
        {
            SheetSage2Decoder decoder = new(Config, Tokenizer);
            decoder.LoadWeights(weights);
            decoder.Dispose();
            Assert.All(weights.Values, t => Assert.Null(Record.Exception(() => _ = t.AsReadOnlySpan<byte>().Length)));
        }
    }

    [Theory]
    [InlineData("token_embedding.weight")]
    [InlineData("decoder.layers.0.encoder_attn.k_proj.bias")]
    [InlineData("decoder.layers.0.final_layer_norm.weight")]
    public void AMissingOrMisshapenTensor_IsNamed(string key)
    {
        using SheetSage2Decoder decoder = new(Config, Tokenizer);
        Dictionary<string, Tensor> missing = new(_full);
        missing.Remove(key);
        Assert.Contains($"'{key}'", Assert.Throws<KeyNotFoundException>(() => decoder.LoadWeights(missing)).Message);

        using Tensor wrong = new(new TensorShape(1, 1), DType.F32);
        Dictionary<string, Tensor> misshapen = new(_full) { [key] = wrong };
        Assert.Contains($"'{key}'", Assert.Throws<InvalidDataException>(() => decoder.LoadWeights(misshapen)).Message);
    }

    [Fact]
    public void AQuantizedTensor_IsRefused_InsteadOfLosingItsScales()
    {
        const string Key = "decoder.layers.0.fc1.weight";
        using SheetSage2Decoder decoder = new(Config, Tokenizer);
        using Tensor fp8 = new(new TensorShape(Inner, Dim), DType.F8E4M3) { QuantInfo = new QuantWeightInfo { Format = "fp8_block" } };

        Assert.Contains($"'{Key}'", Assert.Throws<NotSupportedException>(() => decoder.LoadWeights(new Dictionary<string, Tensor>(_full) { [Key] = fp8 })).Message);
    }

    [Fact]
    public void TheMelFrontend_ReadsBothDtypes_AndRefusesATransposedFilterbank()
    {
        Mert2Config config = new() { NFft = 16, HopLength = 4, MelBins = 3, SampleRate = 100, WindowSeconds = 1 };
        const int Bins = 9;
        Dictionary<string, Tensor> half = new()
        {
            ["f.spectrogram.window"] = Filled(1, 16), ["f.mel_scale.fb"] = Filled(2, Bins, 3), ["f.mel_mean"] = Filled(3, 3), ["f.mel_std"] = Filled(4, 3),
        };
        Dictionary<string, Tensor> full = half.ToDictionary(pair => pair.Key, pair => pair.Value.CastTo(DType.F32));
        using Tensor transposed = new(new TensorShape(3, Bins), DType.F32);
        float[] clip = [.. Enumerable.Range(0, 40).Select(i => MathF.Sin(i * 0.7f))];

        Mert2MelFrontend fromHalf = new(config), fromFull = new(config);
        fromHalf.LoadWeights(half, "f");
        fromFull.LoadWeights(full, "f");
        using Tensor a = fromHalf.Compute(clip), b = fromFull.Compute(clip);

        Assert.Equal(b.AsReadOnlySpan<float>().ToArray(), a.AsReadOnlySpan<float>().ToArray());
        Assert.Throws<InvalidDataException>(() => new Mert2MelFrontend(config).LoadWeights(new Dictionary<string, Tensor>(full) { ["f.mel_scale.fb"] = transposed }, "f"));
        foreach (Tensor t in half.Values.Concat(full.Values)) t.Dispose();
    }

    private static float[] Logits(IReadOnlyDictionary<string, Tensor> weights)
    {
        using CpuBackend backend = new();
        using SheetSage2Decoder decoder = new(Config, Tokenizer);
        decoder.LoadWeights(weights);
        using Tensor memory = new(new TensorShape(4, Dim), DType.F32);
        Span<float> values = memory.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = MathF.Sin(i * 0.37f);
        using SheetSage2DecodeState state = decoder.StartDecode(backend, memory, MaxTokens);
        decoder.Decode(backend, [ScoreTokenizer.SosToken, 5, ScoreTokenizer.OutToken], state);
        decoder.Decode(backend, [7], state);
        float[] logits = state.Logits.AsReadOnlySpan<float>().ToArray();
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(logits, v => v != 0f);
        return logits;
    }

    private static Dictionary<string, Tensor> Checkpoint()
    {
        int seed = 10;
        Dictionary<string, Tensor> w = new()
        {
            ["token_embedding.weight"] = Filled(1, Tokenizer.TokenCount, Dim),
            ["decoder.embed_positions.weight"] = Filled(2, Config.PositionRows, Dim),
            ["decoder.layernorm_embedding.weight"] = Filled(3, Dim),
            ["decoder.layernorm_embedding.bias"] = Filled(4, Dim),
            ["output_projection.weight"] = Filled(5, Tokenizer.TokenCount, Dim),
        };
        const string Layer = "decoder.layers.0";
        foreach (string attention in new[] { "self_attn", "encoder_attn" })
        {
            foreach (string projection in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
            {
                w[$"{Layer}.{attention}.{projection}.weight"] = Filled(seed++, Dim, Dim);
                w[$"{Layer}.{attention}.{projection}.bias"] = Filled(seed++, Dim);
            }
            w[$"{Layer}.{attention}_layer_norm.weight"] = Filled(seed++, Dim);
            w[$"{Layer}.{attention}_layer_norm.bias"] = Filled(seed++, Dim);
        }
        w[$"{Layer}.fc1.weight"] = Filled(seed++, Inner, Dim);
        w[$"{Layer}.fc1.bias"] = Filled(seed++, Inner);
        w[$"{Layer}.fc2.weight"] = Filled(seed++, Dim, Inner);
        w[$"{Layer}.fc2.bias"] = Filled(seed++, Dim);
        w[$"{Layer}.final_layer_norm.weight"] = Filled(seed++, Dim);
        w[$"{Layer}.final_layer_norm.bias"] = Filled(seed, Dim);
        return w;
    }

    private static Tensor Filled(int seed, params long[] shape)
    {
        Random rng = new(seed);
        Tensor t = new(new TensorShape(shape), DType.F16);
        foreach (ref Half v in t.AsSpan<Half>()) v = (Half)(rng.NextDouble() - 0.4);
        return t;
    }
}
