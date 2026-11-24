using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
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
                .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal))
                .ToDictionary(pair => pair.Key, pair => pair.Value);
            Backend = new CpuBackend();
            // The block loop here runs F32; the 16-bit hot path has no CPU kernels.
            _knobs = KnobProfile.Create("hy15-dit-test").With(EngineKnobs.DitF16, false).Push();
            Dit = new HunyuanVideo15Dit(TinyConfig());
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(Weights, Blocks, Hidden));
        }

        private readonly IDisposable _knobs;

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
            _knobs.Dispose();
            Loader.Dispose();
        }
    }
}
