using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Idol;

public sealed class IdolModelTests
{
    private static readonly IBackend Backend = new CpuBackend();

    [Fact]
    public void Config_ReadsTheReleasedArchitectureFromConfigJson()
    {
        IdolConfig c = IdolConfig.FromJson("""
            {"denoiser": {"name": "LatentSeqMMFlowModel", "args": {"q_token_length": 8192, "in_channels": 16, "cam_channels": 5,
              "model_channels": 1024, "cond_channels": 1280, "cond2_channels": 128, "num_blocks": 24, "num_refiner_blocks": 2,
              "num_heads": 16, "mlp_ratio": 4}}}
            """);

        Assert.Equal(new IdolConfig(), c);
        Assert.Equal(4096, c.MlpHidden);
    }

    [Fact]
    public void Config_RejectsAWidthThatIsNotHeadsTimesHeadDim()
    {
        Assert.Throws<FormatException>(() => IdolConfig.FromJson("""{"denoiser": {"args": {"model_channels": 1000}}}"""));
        Assert.Throws<FormatException>(() => IdolConfig.FromJson("{}"));
    }

    [Fact]
    public void PositionFrequencies_AreSixteenOctavesThenFractionalOnesUpToTwoToTheSixteen()
    {
        float[] f = IdolPositionEmbedding.Frequencies(170);

        Assert.Equal(170, f.Length);
        for (int j = 0; j < 16; j++)
            Assert.Equal(MathF.Pow(2f, j), f[j]);
        Assert.Equal(1f, f[16]);                                  // 2^0: the fractional run restarts at exponent 0
        Assert.Equal(Math.Pow(2.0, 16.0 * 153 / 154), f[169], 1);  // and ends just below 2^16
    }

    [Fact]
    public void PositionEmbedding_IsSineThenCosinePerAxisAndPadsTheTail()
    {
        using Tensor positions = new(new TensorShape(1, 1, 3), DType.F32);
        positions.AsSpan<float>()[0] = 0.25f;
        positions.AsSpan<float>()[1] = 0.5f;
        positions.AsSpan<float>()[2] = 0.75f;

        using Tensor e = IdolPositionEmbedding.Build(positions, 1024);

        Assert.Equal(new TensorShape(1, 1, 1024), e.Shape);
        ReadOnlySpan<float> v = e.AsReadOnlySpan<float>();
        // Frequency 2^0 on the first axis: sin(2π·0.25) and cos(2π·0.25).
        Assert.Equal(1f, v[0], 5);
        Assert.Equal(0f, v[170], 5);
        // The second axis starts 340 channels in.
        Assert.Equal(Math.Sin(2 * Math.PI * 0.5), v[340], 5);
        Assert.Equal(Math.Cos(2 * Math.PI * 0.5), v[340 + 170], 5);
        for (int c = 1020; c < 1024; c++)
            Assert.Equal(0f, v[c]);
    }

    [Fact]
    public void Weights_RefuseAMissingTensorAndAWrongShape()
    {
        IdolConfig tiny = TinyConfig();
        Dictionary<string, Tensor> tensors = TinyTensors(tiny, seed: 1);
        try
        {
            tensors["blocks.1.attn.qkv.weight"].Dispose();
            tensors["blocks.1.attn.qkv.weight"] = new Tensor(new TensorShape(5, 5), DType.F32);
            Assert.Contains("blocks.1.attn.qkv.weight", Assert.Throws<Core.Exceptions.HartsyInferenceException>(() => IdolWeights.FromTensors(tiny, tensors)).Message);

            tensors.Remove("blocks.1.attn.qkv.weight");
            Assert.Contains("blocks.1.attn.qkv.weight", Assert.Throws<Core.Exceptions.HartsyInferenceException>(() => IdolWeights.FromTensors(tiny, tensors)).Message);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    [Fact]
    public void Forward_RunsEndToEndOnATinyModelAndIsDeterministic()
    {
        IdolConfig tiny = TinyConfig();
        Dictionary<string, Tensor> tensors = TinyTensors(tiny, seed: 2);
        try
        {
            using IdolWeights weights = IdolWeights.FromTensors(tiny, tensors);
            using IdolDenoiser model = new(weights);
            using Tensor f1 = Random(new TensorShape(1, tiny.CondTokens, tiny.CondChannels), 3, 1f);
            using Tensor f2 = Random(new TensorShape(1, tiny.CondTokens, tiny.Cond2Channels), 4, 1f);
            using Tensor z = Random(new TensorShape(1, tiny.Tokens, tiny.LatentChannels), 5, 1f);
            using Tensor cam = Random(new TensorShape(1, 1, tiny.CameraChannels), 6, 1f);
            using Tensor context = model.EmbedContext(Backend, f1, f2);

            (Tensor v1, Tensor c1) = model.Forward(Backend, z, cam, 731f, context);
            (Tensor v2, Tensor c2) = model.Forward(Backend, z, cam, 731f, context);
            (Tensor v3, Tensor c3) = model.Forward(Backend, z, cam, 100f, context);
            using (v1) using (c1) using (v2) using (c2) using (v3) using (c3)
            {
                Assert.Equal(new TensorShape(1, tiny.Tokens, tiny.LatentChannels), v1.Shape);
                Assert.Equal(new TensorShape(1, 1, tiny.CameraChannels), c1.Shape);
                Assert.All(v1.AsReadOnlySpan<float>().ToArray(), x => Assert.True(float.IsFinite(x)));
                Assert.Equal(v1.AsReadOnlySpan<float>().ToArray(), v2.AsReadOnlySpan<float>().ToArray());
                Assert.Equal(c1.AsReadOnlySpan<float>().ToArray(), c2.AsReadOnlySpan<float>().ToArray());
                // The time step enters through the modulation, so a different time moves the velocity.
                Assert.NotEqual(v1.AsReadOnlySpan<float>().ToArray(), v3.AsReadOnlySpan<float>().ToArray());
            }
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    [Fact]
    public void Forward_RejectsALatentOfTheWrongLength()
    {
        IdolConfig tiny = TinyConfig();
        Dictionary<string, Tensor> tensors = TinyTensors(tiny, seed: 7);
        try
        {
            using IdolWeights weights = IdolWeights.FromTensors(tiny, tensors);
            using IdolDenoiser model = new(weights);
            using Tensor context = Random(new TensorShape(1, 3, tiny.ModelChannels), 1, 1f);
            using Tensor z = Random(new TensorShape(1, tiny.Tokens + 1, tiny.LatentChannels), 5, 1f);
            using Tensor cam = Random(new TensorShape(1, 1, tiny.CameraChannels), 6, 1f);

            Assert.Throws<ArgumentException>(() => model.Forward(Backend, z, cam, 500f, context));
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    /// <summary>Two heads of 8, two joint blocks, one refiner block and 16 tokens: every code path of the full model at toy size.</summary>
    internal static IdolConfig TinyConfig() => new()
    {
        Tokens = 16, LatentChannels = 4, CameraChannels = 5, ModelChannels = 16, CondChannels = 6, Cond2Channels = 3,
        Blocks = 2, RefinerBlocks = 1, Heads = 2, HeadDim = 8, MlpRatio = 2, CondTokens = 5,
    };

    internal static Dictionary<string, Tensor> TinyTensors(IdolConfig c, int seed)
    {
        Dictionary<string, Tensor> t = new(StringComparer.Ordinal);
        int s = seed * 1000;
        foreach ((string key, long[] shape) in IdolWeights.ExpectedShapes(c))
        {
            // Norm gains start near one, the rest small: a stable forward whose every term matters.
            float scale = key.EndsWith(".gamma", StringComparison.Ordinal) || key.EndsWith("norm.weight", StringComparison.Ordinal) || key.EndsWith("norm1.weight", StringComparison.Ordinal) || key.EndsWith("norm2.weight", StringComparison.Ordinal) ? 0.1f : 0.2f;
            Tensor w = Random(new TensorShape(shape), s++, scale);
            if (scale == 0.1f)
            {
                Span<float> span = w.AsSpan<float>();
                for (int i = 0; i < span.Length; i++) span[i] += 1f;
            }

            t[key] = w;
        }

        Tensor pos = Random(new TensorShape(1, c.Tokens, 3), s++, 0.5f);
        Span<float> p = pos.AsSpan<float>();
        for (int i = 0; i < p.Length; i++) p[i] += 0.5f;
        t["pos_pe"] = pos;
        return t;
    }

    internal static Tensor Random(TensorShape shape, int seed, float scale)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }
}
