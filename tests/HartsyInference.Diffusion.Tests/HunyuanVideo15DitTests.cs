using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The HunyuanVideo-1.5 / MovieRoll DiT forward on a tiny random checkpoint in the reference's own key names, against the saved
/// outputs of <c>tests/python-reference/movieroll_reference/dump_hy15_tiny.py</c>. The reference side imports MovieRoll's own
/// modulation, MLP, RMSNorm, timestep, ByT5 and final-layer modules (checkout 0b5f73e) and replaces FlashAttention with torch SDPA.
/// This pins the converter's renames and splits, the tokenisation of the latent, both text streams, the block maths, the final layer
/// and the unpatchify. It is not parity with real weights: no HunyuanVideo-1.5 or MovieRoll checkpoint has run yet.</summary>
public sealed class HunyuanVideo15DitTests
{
    private const float Tolerance = 1e-5f;
    private const int Hidden = 32;
    private const int Blocks = 2;

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = 4, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    [Fact]
    public void TextToVideoForwardMatchesTheReference()
    {
        using Fixture fixture = new();
        using Tensor velocity = fixture.Run(withByT5: true, conditioned: false);

        AssertClose(fixture.Reference("ref.velocity"), velocity, "text-to-video velocity");
    }

    [Fact]
    public void ConditioningBlockIsAddedToTheInput()
    {
        using Fixture fixture = new();
        using Tensor velocity = fixture.Run(withByT5: true, conditioned: true);

        AssertClose(fixture.Reference("ref.velocity_conditioned"), velocity, "conditioned velocity");
    }

    [Fact]
    public void ForwardWithoutByT5TokensMatchesTheReference()
    {
        using Fixture fixture = new();
        using Tensor velocity = fixture.Run(withByT5: false, conditioned: false);

        AssertClose(fixture.Reference("ref.velocity_no_byt5"), velocity, "velocity without byT5");
    }

    [Fact]
    public void ConverterSplitsTheFusedProjectionIntoThirds()
    {
        using Fixture fixture = new();
        Tensor fused = fixture.Weights["double_blocks.0.img_attn_qkv.weight"];
        Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(fixture.Weights, Blocks, Hidden);

        Tensor query = converted["transformer_blocks.0.attn.to_q.weight"];
        Tensor value = converted["transformer_blocks.0.attn.to_v.weight"];
        Assert.Equal(new TensorShape(Hidden, Hidden), query.Shape);
        Assert.Equal(fused.AsReadOnlySpan<float>()[0], query.AsReadOnlySpan<float>()[0]);
        Assert.Equal(fused.AsReadOnlySpan<float>()[2 * Hidden * Hidden + 5], value.AsReadOnlySpan<float>()[5]);
        Assert.Equal(new TensorShape(Hidden, 9), converted["img_in.proj.weight"].Shape);
    }

    [Fact]
    public void ConverterNamesTheMissingTensor()
    {
        using Fixture fixture = new();
        Dictionary<string, Tensor> broken = new(fixture.Weights);
        broken.Remove("final_layer.linear.weight");

        KeyNotFoundException error = Assert.Throws<KeyNotFoundException>(() => HunyuanVideo15CheckpointConverter.Convert(broken, Blocks, Hidden));
        Assert.Contains("final_layer.linear.weight", error.Message);
    }

    [Fact]
    public void ForwardRejectsAWrongChannelCount()
    {
        using Fixture fixture = new();
        using Tensor wrong = new(new TensorShape([1L, 3, 3, 2, 3]), DType.F32);

        Assert.Throws<ArgumentException>(() => fixture.Dit.Forward(fixture.Backend, wrong, 1f, fixture.Loader.GetTensor("in.qwen")));
    }

    [Fact]
    public void ForwardRejectsA16BitLatent()
    {
        using Fixture fixture = new();
        using Tensor half = new(new TensorShape([1L, 4, 3, 2, 3]), DType.F16);

        ArgumentException error = Assert.Throws<ArgumentException>(() => fixture.Dit.Forward(fixture.Backend, half, 1f, fixture.Loader.GetTensor("in.qwen")));
        Assert.Equal("latent", error.ParamName);
    }

    [Fact]
    public void ForwardRejectsConditioningWithTheWrongLayout()
    {
        using Fixture fixture = new();
        // Same element count as the real conditioning block, wrong axis order.
        using Tensor misplaced = new(new TensorShape([1L, 3, 5, 2, 3]), DType.F32);

        ArgumentException error = Assert.Throws<ArgumentException>(() => fixture.Dit.Forward(
            fixture.Backend, fixture.Loader.GetTensor("in.latent"), 1f, fixture.Loader.GetTensor("in.qwen"), null, misplaced));
        Assert.Equal("conditioning", error.ParamName);
    }

    [Fact]
    public void ForwardRejectsEmptyOrMisshapenText()
    {
        using Fixture fixture = new();
        Tensor latent = fixture.Loader.GetTensor("in.latent");
        using Tensor empty = new(new TensorShape(1, 0, 24), DType.F32);
        using Tensor batched = new(new TensorShape(2, 3, 24), DType.F32);
        using Tensor wrongWidth = new(new TensorShape(1, 3, 99), DType.F32);

        Assert.Equal("qwen", Assert.Throws<ArgumentException>(() => fixture.Dit.Forward(fixture.Backend, latent, 1f, empty)).ParamName);
        Assert.Equal("qwen", Assert.Throws<ArgumentException>(() => fixture.Dit.Forward(fixture.Backend, latent, 1f, batched)).ParamName);
        Assert.Equal("byt5", Assert.Throws<ArgumentException>(() => fixture.Dit.Forward(fixture.Backend, latent, 1f, fixture.Loader.GetTensor("in.qwen"), wrongWidth)).ParamName);
    }

    [Fact]
    public void VelocityIsAnOwnedFiveDimensionalTensor()
    {
        using Fixture fixture = new();
        using Tensor velocity = fixture.Run(withByT5: true, conditioned: false);

        Assert.Equal(new TensorShape([1L, 4, 3, 2, 3]), velocity.Shape);
    }

    [Fact]
    public void ConstructorRejectsRopeAxesThatDoNotSumToTheHeadDim()
    {
        HunyuanVideo15Config bad = TinyConfig() with { RopeAxesDim = [4, 6, 8] };

        Assert.Throws<ArgumentException>(() => new HunyuanVideo15Dit(bad));
    }

    [Theory]
    [InlineData("guidance_in.mlp.0.weight")]
    [InlineData("vision_in.proj.0.weight")]
    [InlineData("txt_in.individual_token_refiner.blocks.0.self_attn_q_norm.weight")]
    public void ConverterRefusesTensorsItDoesNotConsume(string key)
    {
        using Fixture fixture = new();
        Dictionary<string, Tensor> extra = new(fixture.Weights) { [key] = new Tensor(new TensorShape(2), DType.F32) };

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => HunyuanVideo15CheckpointConverter.Convert(extra, Blocks, Hidden));
        Assert.Contains(key, error.Message);
    }

    [Fact]
    public void ConverterConsumesEverythingInTheFixture()
    {
        using Fixture fixture = new();

        Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(fixture.Weights, Blocks, Hidden);

        Assert.Contains("transformer_blocks.1.attn.add_v_proj.bias", converted.Keys);
        Assert.Contains("txt_in.individual_token_refiner.blocks.1.self_attn.qkv.weight", converted.Keys);
    }

    [Fact]
    public void PerFrameTimestepsMatchTheReference()
    {
        using Fixture fixture = new();
        float[] frameTimesteps = fixture.Reference("in.frame_timesteps").AsReadOnlySpan<float>().ToArray();
        float window = fixture.Reference("in.timestep").AsReadOnlySpan<float>()[0];
        using Tensor velocity = fixture.Dit.ForwardPerFrame(fixture.Backend, fixture.Reference("in.latent"), frameTimesteps, window,
            fixture.Reference("in.qwen"), fixture.Reference("in.byt5"));
        AssertClose(fixture.Reference("ref.velocity_per_frame"), velocity, "per-frame velocity");
    }

    [Fact]
    public void EqualPerFrameTimestepsAreTheClipWideForward()
    {
        using Fixture fixture = new();
        float window = fixture.Reference("in.timestep").AsReadOnlySpan<float>()[0];
        Tensor latent = fixture.Reference("in.latent");
        float[] same = Enumerable.Repeat(window, (int)latent.Shape[2]).ToArray();
        Tensor qwen = fixture.Reference("in.qwen");
        Tensor byt5 = fixture.Reference("in.byt5");
        using Tensor perFrame = fixture.Dit.ForwardPerFrame(fixture.Backend, latent, same, window, qwen, byt5);
        using Tensor clip = fixture.Dit.Forward(fixture.Backend, latent, window, qwen, byt5);
        AssertClose(clip, perFrame, "equal per-frame timesteps");
    }

    [Fact]
    public void EqualPerFrameTimestepsWithoutByT5AreTheClipWideForward()
    {
        using Fixture fixture = new();
        float window = fixture.Reference("in.timestep").AsReadOnlySpan<float>()[0];
        Tensor latent = fixture.Reference("in.latent");
        float[] same = Enumerable.Repeat(window, (int)latent.Shape[2]).ToArray();
        Tensor qwen = fixture.Reference("in.qwen");
        using Tensor perFrame = fixture.Dit.ForwardPerFrame(fixture.Backend, latent, same, window, qwen, null);
        using Tensor clip = fixture.Dit.Forward(fixture.Backend, latent, window, qwen, null);
        AssertClose(clip, perFrame, "equal per-frame timesteps without byT5");
    }

    [Fact]
    public void EqualPerFrameTimestepsWithConditioningAreTheClipWideForward()
    {
        using Fixture fixture = new();
        float window = fixture.Reference("in.timestep").AsReadOnlySpan<float>()[0];
        Tensor latent = fixture.Reference("in.latent");
        Tensor conditioning = fixture.Reference("in.conditioning");
        float[] same = Enumerable.Repeat(window, (int)latent.Shape[2]).ToArray();
        Tensor qwen = fixture.Reference("in.qwen");
        Tensor byt5 = fixture.Reference("in.byt5");
        using Tensor perFrame = fixture.Dit.ForwardPerFrame(fixture.Backend, latent, same, window, qwen, byt5, conditioning);
        using Tensor clip = fixture.Dit.Forward(fixture.Backend, latent, window, qwen, byt5, conditioning);
        AssertClose(clip, perFrame, "equal per-frame timesteps with conditioning");
    }

    [Fact]
    public void PerFrameForwardRejectsAFrameTimestepCountThatDoesNotMatchTheLatent()
    {
        using Fixture fixture = new();
        Tensor latent = fixture.Reference("in.latent");
        Tensor qwen = fixture.Reference("in.qwen");
        float[] tooFew = [500f, 500f];
        Assert.Throws<ArgumentException>(() => fixture.Dit.ForwardPerFrame(fixture.Backend, latent, tooFew, 500f, qwen, null));
    }

    private static void AssertClose(Tensor expected, Tensor actual, string label)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float max = 0f;
        for (int i = 0; i < e.Length; i++) max = Math.Max(max, Math.Abs(e[i] - a[i]));
        Assert.True(max <= Tolerance, $"{label}: max abs error {max:E3} exceeds {Tolerance:E0}.");
    }

    private sealed class Fixture : IDisposable
    {
        public Fixture()
        {
            Loader = new SafeTensorsLoader();
            Loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
            Weights = Loader.GetAllTensors()
                .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                    && !pair.Key.StartsWith("win.", StringComparison.Ordinal))
                .ToDictionary(pair => pair.Key, pair => pair.Value);
            Backend = new CpuBackend();
            Dit = new HunyuanVideo15Dit(TinyConfig());
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(Weights, Blocks, Hidden));
        }

        public SafeTensorsLoader Loader { get; }
        public Dictionary<string, Tensor> Weights { get; }
        public IBackend Backend { get; }
        public HunyuanVideo15Dit Dit { get; }

        public Tensor Reference(string name) => Loader.GetTensor(name);

        public Tensor Run(bool withByT5, bool conditioned) =>
            Dit.Forward(Backend, Loader.GetTensor("in.latent"), Loader.GetTensor("in.timestep").AsReadOnlySpan<float>()[0],
                Loader.GetTensor("in.qwen"), withByT5 ? Loader.GetTensor("in.byt5") : null,
                conditioned ? Loader.GetTensor("in.conditioning") : null);

        public void Dispose()
        {
            Dit.Dispose();
            Backend.Dispose();
            Loader.Dispose();
        }
    }
}
