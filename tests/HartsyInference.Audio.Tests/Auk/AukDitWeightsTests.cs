using HartsyInference.Audio.Models.Auk;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>What the AuK DiT reads from a checkpoint, what it refuses, and which tensors it owns. Tiny synthetic weights, no downloads.</summary>
public sealed class AukDitWeightsTests : IDisposable
{
    private const int Dim = 8;
    private const int Latent = 4;
    private const int Text = 6;

    private static readonly AukConfig Tiny = new()
    {
        Dim = Dim, Heads = 2, DoubleBlocks = 1, SingleBlocks = 1, FfMult = 2, LatentDim = Latent, TextDim = Text,
        TimeFreqEmbedDim = 8, ConvPosKernel = 3, ConvPosGroups = 2,
    };

    private readonly Dictionary<string, Tensor> _half = Checkpoint();
    private readonly Dictionary<string, Tensor> _full = [];

    public AukDitWeightsTests()
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
        AukDit dit = new(Tiny);
        dit.LoadWeights(_full);
        List<Tensor> read = [.. dit.EnumerateWeights()];

        dit.Dispose();

        Assert.All(read, t => Assert.Contains(t, _full.Values));
        Assert.All(_full.Values, t => Assert.True(Alive(t)));
    }

    [Fact]
    public void A16BitCheckpoint_GivesTheSameVelocity_AndItsCopiesDieWithTheModel()
    {
        using AukDit full = new(Tiny);
        full.LoadWeights(_full);
        AukDit half = new(Tiny);
        half.LoadWeights(_half);
        List<Tensor> copies = [.. half.EnumerateWeights()];

        Assert.Equal(Velocity(full), Velocity(half));
        Assert.All(copies, t => Assert.Equal(DType.F32, t.DType));
        Assert.All(copies, t => Assert.DoesNotContain(t, _half.Values));

        half.Dispose();

        Assert.All(copies, t => Assert.False(Alive(t)));
        Assert.All(_half.Values, t => Assert.True(Alive(t)));
    }

    [Fact]
    public void AReload_FreesTheFirstCopies_AndTheModelStillRuns()
    {
        using AukDit dit = new(Tiny);
        dit.LoadWeights(_half);
        List<Tensor> first = [.. dit.EnumerateWeights()];
        float[] before = Velocity(dit);

        dit.LoadWeights(_half);

        Assert.All(first, t => Assert.False(Alive(t)));
        Assert.Equal(before, Velocity(dit));
    }

    [Fact]
    public void AFailedReload_LeavesTheModelUnloaded()
    {
        using AukDit dit = new(Tiny);
        dit.LoadWeights(_half);
        Dictionary<string, Tensor> broken = new(_half);
        broken.Remove("transformer.proj_out.weight");

        Assert.Throws<KeyNotFoundException>(() => dit.LoadWeights(broken));

        Assert.Throws<InvalidOperationException>(() => Velocity(dit));
    }

    [Theory]
    [InlineData("transformer.time_embed.time_mlp.2.bias")]
    [InlineData("transformer.audio_embed.conv_pos_embed.conv1d.0.weight")]
    [InlineData("transformer.transformer_blocks.0.attn.to_qkv_c.weight")]
    [InlineData("transformer.single_transformer_blocks.0.ff.linear_out.weight")]
    [InlineData("transformer.rotary_embed.inv_freq")]
    public void AMissingOrMisshapenTensor_IsNamed(string key)
    {
        using AukDit dit = new(Tiny);
        Dictionary<string, Tensor> missing = new(_full);
        missing.Remove(key);
        Assert.Contains($"'{key}'", Assert.Throws<KeyNotFoundException>(() => dit.LoadWeights(missing)).Message);

        using Tensor wrong = new(new TensorShape(1, 1), DType.F32);
        Dictionary<string, Tensor> misshapen = new(_full) { [key] = wrong };
        Assert.Contains($"'{key}'", Assert.Throws<InvalidDataException>(() => dit.LoadWeights(misshapen)).Message);
    }

    [Fact]
    public void AQuantizedTensor_IsRefused_InsteadOfLosingItsScales()
    {
        using AukDit dit = new(Tiny);
        const string Key = "transformer.txt_proj.weight";
        using Tensor fp8 = new(new TensorShape(Dim, Text), DType.F8E4M3) { QuantInfo = new QuantWeightInfo { Format = "fp8_block" } };

        Assert.Contains($"'{Key}'", Assert.Throws<NotSupportedException>(() => dit.LoadWeights(new Dictionary<string, Tensor>(_full) { [Key] = fp8 })).Message);
    }

    private static float[] Velocity(AukDit dit)
    {
        using CpuBackend backend = new();
        using Tensor text = Filled(new Random(5), DType.F32, 1, 3, Text);
        using Tensor noisy = Filled(new Random(6), DType.F32, 1, 5, Latent);
        using Tensor reference = Filled(new Random(7), DType.F32, 1, 2, Latent);
        using Tensor condition = dit.ProjectText(backend, text);
        using Tensor velocity = dit.Forward(backend, noisy, reference, condition, 0.3f);
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
        const int HeadDim = Dim / 2;
        const int Inner = Dim * 2;
        void Add(string key, params long[] shape) => w[$"transformer.{key}"] = Filled(rng, DType.F16, shape);
        void Linear(string key, int rows, int cols)
        {
            Add($"{key}.weight", rows, cols);
            Add($"{key}.bias", rows);
        }
        void Attention(string prefix, string qkv, string q, string k, string output)
        {
            Linear($"{prefix}.{qkv}", 3 * Dim, Dim);
            Add($"{prefix}.{q}.weight", HeadDim);
            Add($"{prefix}.{k}.weight", HeadDim);
            Linear($"{prefix}.{output}", Dim, Dim);
        }
        void FeedForward(string prefix)
        {
            Add($"{prefix}.linear_in.weight", 2 * Inner, Dim);
            Add($"{prefix}.linear_out.weight", Dim, Inner);
        }

        Linear("time_embed.time_mlp.0", Dim, 8);
        Linear("time_embed.time_mlp.2", Dim, Dim);
        Linear("audio_embed.linear", Dim, Latent);
        foreach (string i in new[] { "0", "2" })
        {
            Add($"audio_embed.conv_pos_embed.conv1d.{i}.weight", Dim, Dim / 2, 3);
            Add($"audio_embed.conv_pos_embed.conv1d.{i}.bias", Dim);
        }
        Linear("txt_proj", Dim, Text);
        Add("txt_norm.weight", Dim);
        const string Double = "transformer_blocks.0";
        Linear($"{Double}.attn_norm_x.linear", 6 * Dim, Dim);
        Linear($"{Double}.attn_norm_c.linear", 6 * Dim, Dim);
        Attention($"{Double}.attn", "to_qkv", "q_norm", "k_norm", "to_out.0");
        Attention($"{Double}.attn", "to_qkv_c", "c_q_norm", "c_k_norm", "to_out_c");
        FeedForward($"{Double}.ff_x");
        FeedForward($"{Double}.ff_c");
        const string Single = "single_transformer_blocks.0";
        Linear($"{Single}.attn_norm.linear", 6 * Dim, Dim);
        Attention($"{Single}.attn", "to_qkv", "q_norm", "k_norm", "to_out.0");
        FeedForward($"{Single}.ff");
        Linear("norm_out.linear", 2 * Dim, Dim);
        Linear("proj_out", Latent, Dim);
        Add("rotary_embed.inv_freq", HeadDim / 2);
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
