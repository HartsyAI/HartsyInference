using HartsyInference.Audio.Models.F5Tts;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.F5Tts;

/// <summary>What the F5-TTS DiT reads from a checkpoint, what it refuses, and which tensors it owns. Tiny synthetic weights, no downloads.</summary>
public sealed class F5DitWeightsTests : IDisposable
{
    private const int Dim = 8, Mel = 4, Text = 6, Inner = Text * 2;

    private static readonly F5TtsConfig Tiny = new()
    {
        Dim = Dim, Depth = 2, Heads = 2, FfMult = 2, MelDim = Mel, TextDim = Text, TextConvLayers = 1, TextConvMult = 2, TextMaxPos = 32,
        TextNumEmbeds = 9, TimeFreqEmbedDim = 8, ConvPosKernel = 3, ConvPosGroups = 2,
    };

    private readonly Dictionary<string, Tensor> _half = Checkpoint();
    private readonly Dictionary<string, Tensor> _full = [];

    public F5DitWeightsTests()
    {
        foreach ((string key, Tensor value) in _half) _full[key] = value.CastTo(DType.F32);
    }

    public void Dispose()
    {
        foreach (Tensor t in _half.Values) t.Dispose();
        foreach (Tensor t in _full.Values) t.Dispose();
    }

    [Fact]
    public void AnF32Checkpoint_IsBorrowed_AndOutlivesTheModel()
    {
        F5Dit dit = new(Tiny);
        dit.LoadWeights(_full, "m");
        List<Tensor> read = [.. dit.EnumerateWeights()];

        dit.Dispose();
        dit.Dispose();

        Assert.Equal(_full.Count, read.Count);
        Assert.All(read, t => Assert.Contains(t, _full.Values));
        Assert.All(_full.Values, t => Assert.True(Alive(t)));
    }

    [Fact]
    public void A16BitCheckpoint_GivesTheSameVelocity_AndItsCopiesDieWithTheModel()
    {
        using F5Dit full = new(Tiny);
        full.LoadWeights(_full, "m");
        F5Dit half = new(Tiny);
        half.LoadWeights(_half, "m");
        List<Tensor> copies = [.. half.EnumerateWeights()];

        Assert.Equal(Velocity(full), Velocity(half));
        Assert.Equal(_half.Count, copies.Count);
        Assert.All(copies, t => Assert.DoesNotContain(t, _half.Values));

        half.Dispose();

        Assert.All(copies, t => Assert.False(Alive(t)));
        Assert.All(_half.Values, t => Assert.True(Alive(t)));
    }

    [Fact]
    public void AReload_FreesTheFirstCopies_AndTheModelStillRuns()
    {
        using F5Dit dit = new(Tiny);
        dit.LoadWeights(_half, "m");
        List<Tensor> first = [.. dit.EnumerateWeights()];
        float[] before = Velocity(dit);

        dit.LoadWeights(_half, "m");

        Assert.All(first, t => Assert.False(Alive(t)));
        Assert.Equal(before, Velocity(dit));
    }

    [Fact]
    public void AFailedReload_LeavesTheModelUnloaded()
    {
        using F5Dit dit = new(Tiny);
        dit.LoadWeights(_half, "m");
        Dictionary<string, Tensor> broken = new(_half);
        broken.Remove("m.proj_out.weight");

        Assert.Throws<KeyNotFoundException>(() => dit.LoadWeights(broken, "m"));

        Assert.Throws<InvalidOperationException>(() => Velocity(dit));
        Assert.Throws<InvalidOperationException>(() => Debug(dit));
        dit.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Debug(dit));
    }

    [Theory]
    [InlineData(-2)]
    [InlineData(9)]
    [InlineData(int.MaxValue)]
    public void ATokenTheEmbeddingLacks_IsRefused(int token)
    {
        using F5Dit dit = new(Tiny);
        dit.LoadWeights(_full, "m");

        Assert.Throws<ArgumentOutOfRangeException>(() => Velocity(dit, token));
        Velocity(dit, 8);
        dit.ResetTextCache();
        Velocity(dit, -1);
    }

    [Fact]
    public void AReload_DropsTheTextEmbeddingOfTheOldWeights()
    {
        using F5Dit dit = new(Tiny);
        dit.LoadWeights(_full, "m");
        float[] first = Velocity(dit);
        using Tensor other = Filled(new Random(9), DType.F32, 10, Text);

        dit.LoadWeights(new Dictionary<string, Tensor>(_full) { ["m.text_embed.text_embed.weight"] = other }, "m");

        Assert.NotEqual(first, Velocity(dit));
    }

    [Theory]
    [InlineData("m.time_embed.time_mlp.2.bias")]
    [InlineData("m.input_embed.conv_pos_embed.conv1d.0.weight")]
    [InlineData("m.text_embed.text_embed.weight")]
    [InlineData("m.text_embed.text_blocks.0.dwconv.weight")]
    [InlineData("m.text_embed.text_blocks.0.grn.gamma")]
    [InlineData("m.text_embed.text_blocks.0.grn.beta")]
    [InlineData("m.transformer_blocks.1.attn.to_out.0.weight")]
    [InlineData("m.proj_out.bias")]
    public void AMissingOrMisshapenTensor_IsNamed(string key)
    {
        using F5Dit dit = new(Tiny);
        Dictionary<string, Tensor> missing = new(_full);
        missing.Remove(key);
        Assert.Contains($"'{key}'", Assert.Throws<KeyNotFoundException>(() => dit.LoadWeights(missing, "m")).Message);

        using Tensor wrong = new(new TensorShape(1, 1), DType.F32);
        Dictionary<string, Tensor> misshapen = new(_full) { [key] = wrong };
        Assert.Contains($"'{key}'", Assert.Throws<InvalidDataException>(() => dit.LoadWeights(misshapen, "m")).Message);
    }

    [Fact]
    public void AQuantizedTensor_IsRefused_InsteadOfLosingItsScales()
    {
        const string Key = "m.transformer_blocks.0.attn.to_q.weight";
        using F5Dit dit = new(Tiny);
        using Tensor fp8 = new(new TensorShape(Dim, Dim), DType.F8E4M3) { QuantInfo = new QuantWeightInfo { Format = "fp8_block" } };

        Assert.Contains($"'{Key}'",
            Assert.Throws<NotSupportedException>(() => dit.LoadWeights(new Dictionary<string, Tensor>(_full) { [Key] = fp8 }, "m")).Message);
    }

    private static void Debug(F5Dit dit)
    {
        using CpuBackend backend = new();
        using Tensor mel = Filled(new Random(5), DType.F32, 1, Mel, 7);
        _ = dit.DebugForward(backend, mel, mel, [1, 4, 2], 0.3f);
    }

    private static float[] Velocity(F5Dit dit, int token = 4)
    {
        const int frames = 7;
        using CpuBackend backend = new();
        using Tensor noisy = Filled(new Random(5), DType.F32, 1, Mel, frames), cond = Filled(new Random(6), DType.F32, 1, Mel, frames);
        using Tensor velocity = dit.Forward(backend, noisy, cond, [1, token, 2], 0.3f, dropAudioCond: false, dropText: false);
        float[] values = velocity.AsReadOnlySpan<float>().ToArray();
        Assert.All(values, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(values, v => v != 0f);
        return values;
    }

    private static bool Alive(Tensor t)
    {
        try
        {
            _ = t.AsReadOnlySpan<byte>().Length;
            return true;
        }
        catch (ObjectDisposedException)
        {
            return false;
        }
    }

    private static Dictionary<string, Tensor> Checkpoint()
    {
        Dictionary<string, Tensor> w = [];
        Random rng = new(1);
        void Add(string key, params long[] shape) => w[$"m.{key}"] = Filled(rng, DType.F16, shape);
        void Linear(string key, int rows, int cols)
        {
            Add($"{key}.weight", rows, cols);
            Add($"{key}.bias", rows);
        }
        Linear("time_embed.time_mlp.0", Dim, 8);
        Linear("time_embed.time_mlp.2", Dim, Dim);
        Add("text_embed.text_embed.weight", 10, Text);
        Add("text_embed.text_blocks.0.dwconv.weight", Text, 1, 7);
        Add("text_embed.text_blocks.0.dwconv.bias", Text);
        Add("text_embed.text_blocks.0.norm.weight", Text);
        Add("text_embed.text_blocks.0.norm.bias", Text);
        Linear("text_embed.text_blocks.0.pwconv1", Inner, Text);
        Linear("text_embed.text_blocks.0.pwconv2", Text, Inner);
        Add("text_embed.text_blocks.0.grn.gamma", 1, 1, Inner);
        Add("text_embed.text_blocks.0.grn.beta", 1, 1, Inner);
        Linear("input_embed.proj", Dim, 2 * Mel + Text);
        foreach (string i in new[] { "0", "2" })
        {
            Add($"input_embed.conv_pos_embed.conv1d.{i}.weight", Dim, Dim / 2, 3);
            Add($"input_embed.conv_pos_embed.conv1d.{i}.bias", Dim);
        }
        for (int block = 0; block < Tiny.Depth; block++)
        {
            string p = $"transformer_blocks.{block}";
            Linear($"{p}.attn_norm.linear", 6 * Dim, Dim);
            foreach (string projection in new[] { "to_q", "to_k", "to_v", "to_out.0" }) Linear($"{p}.attn.{projection}", Dim, Dim);
            Linear($"{p}.ff.ff.0.0", 2 * Dim, Dim);
            Linear($"{p}.ff.ff.2", Dim, 2 * Dim);
        }
        Linear("norm_out.linear", 2 * Dim, Dim);
        Linear("proj_out", Mel, Dim);
        return w;
    }

    private static Tensor Filled(Random rng, DType dtype, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), dtype);
        if (dtype == DType.F16)
            foreach (ref Half v in t.AsSpan<Half>()) v = (Half)(rng.NextDouble() - 0.4);
        else
            foreach (ref float v in t.AsSpan<float>()) v = (float)(rng.NextDouble() - 0.4);
        return t;
    }
}
