using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The HunyuanVideo-1.5 causal VAE on a tiny random checkpoint (channels 4·8·16·32·32, latent 2, 16× spatial, 4× temporal, two resnets per
/// level) against <c>tests/python-reference/movieroll_reference/dump_hy15_vae_tiny.py</c>, which runs MovieRoll's own Encoder and Decoder. The
/// per-stage tests check each DCAE resampler (main path plus shortcut) and each mid attention in isolation, so a failure names its branch.
/// This pins the DCAE index maps, the RMS norm, the frame-causal mid attention and the strict loader. It is not parity with real weights.</summary>
public sealed class HunyuanVideo15VaeTests
{
    private const float Tolerance = 1e-4f;
    private const float StageTolerance = 1e-5f;
    private static readonly int[] BlockOut = [4, 8, 16, 32, 32];

    private static HunyuanVideo15VaeConfig TinyConfig() => new()
    {
        LatentChannels = 2,
        BlockOutChannels = BlockOut,
        LayersPerBlock = 2,
        SpatialCompression = 16,
        TemporalCompression = 4,
        ScalingFactor = 1f,
    };

    [Fact]
    public void EncodeMomentsMatchTheReference()
    {
        using Fixture fixture = new();
        using Tensor moments = fixture.Vae.EncodeMoments(fixture.Backend, fixture.Reference("input"));
        AssertClose(fixture.Reference("moments"), moments, "video moments", Tolerance);
    }

    [Theory]
    [InlineData(4, 16, 32)]
    [InlineData(5, 20, 32)]
    [InlineData(5, 16, 24)]
    [InlineData(5, 16, 33)]
    public void EncodeRejectsAClipThatIsNotATemporalAndSpatialMultiple(int frames, int height, int width)
    {
        using Fixture fixture = new();
        Tensor video = HunyuanVideo15VaeShuffle.Alloc(3, frames, height, width);
        Assert.Throws<ArgumentException>(() => fixture.Vae.EncodeMoments(fixture.Backend, video));
    }

    [Theory]
    [InlineData(true, 3, 4)]
    [InlineData(true, 4, 3)]
    [InlineData(false, 3, 4)]
    [InlineData(false, 4, 3)]
    public void ShuffleRefusesOddSpatialSizes(bool temporal, int height, int width)
    {
        using Fixture fixture = new();
        Tensor h = HunyuanVideo15VaeShuffle.Alloc(temporal ? 2 : 1, 3, height, width);
        Tensor x = HunyuanVideo15VaeShuffle.Alloc(4, 3, height, width);
        Assert.Throws<ArgumentException>(() => HunyuanVideo15VaeShuffle.PackDownsample(fixture.Backend, h, temporal ? 16 : 4, temporal));
        Assert.Throws<ArgumentException>(() => HunyuanVideo15VaeShuffle.DownsampleShortcut(fixture.Backend, x, 4, temporal));
    }

    [Fact]
    public void AllocRefusesAnElementCountThatCannotBeIndexed()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => HunyuanVideo15VaeShuffle.Alloc(256, 2048, 1024, 1024));
    }

    [Fact]
    public void EncodeMeanIsTheFirstLatentHalfOfTheMoments()
    {
        using Fixture fixture = new();
        using Tensor mean = fixture.Vae.EncodeMean(fixture.Backend, fixture.Reference("input"));
        AssertClose(fixture.Reference("mean"), mean, "video mean", Tolerance);
    }

    [Fact]
    public void DecodeMatchesTheReference()
    {
        using Fixture fixture = new();
        using Tensor decoded = fixture.Vae.Decode(fixture.Backend, fixture.Reference("mean"));
        AssertClose(fixture.Reference("decoded"), decoded, "video decode", Tolerance);
    }

    [Fact]
    public void SingleFrameImageEncodeAndDecodeMatchTheReference()
    {
        using Fixture fixture = new();
        using Tensor moments = fixture.Vae.EncodeMoments(fixture.Backend, fixture.Reference("image_input"));
        AssertClose(fixture.Reference("image_moments"), moments, "image moments", Tolerance);
        using Tensor mean = fixture.Vae.EncodeMean(fixture.Backend, fixture.Reference("image_input"));
        using Tensor decoded = fixture.Vae.Decode(fixture.Backend, mean);
        AssertClose(fixture.Reference("image_decoded"), decoded, "image decode", Tolerance);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void EncoderDownsampleStageMatchesTheReference(int level)
    {
        using Fixture fixture = new();
        HunyuanVideo15VaeConfig config = TinyConfig();
        int cout = BlockOut[level + 1];
        bool temporal = config.EncoderTemporalDownsample(level);
        string stage = $"stage.enc.down.{level}.downsample";
        using Tensor main = HunyuanVideo15VaeShuffle.PackDownsample(fixture.Backend, fixture.Reference($"{stage}.h"), cout, temporal);
        using Tensor shortcut = HunyuanVideo15VaeShuffle.DownsampleShortcut(fixture.Backend, fixture.Reference($"{stage}.x"), cout, temporal);
        using Tensor sum = fixture.Add(main, shortcut);
        AssertClose(fixture.Reference($"{stage}.out"), sum, $"encoder downsample level {level} (temporal={temporal})", StageTolerance);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void DecoderUpsampleStageMatchesTheReference(int level)
    {
        using Fixture fixture = new();
        HunyuanVideo15VaeConfig config = TinyConfig();
        int[] reversed = Enumerable.Reverse(BlockOut).ToArray();
        int cout = reversed[level + 1];
        bool temporal = config.DecoderTemporalUpsample(level);
        string stage = $"stage.dec.up.{level}.upsample";
        using Tensor main = HunyuanVideo15VaeShuffle.UnpackUpsample(fixture.Backend, fixture.Reference($"{stage}.h"), cout, temporal);
        using Tensor shortcut = HunyuanVideo15VaeShuffle.UpsampleShortcut(fixture.Backend, fixture.Reference($"{stage}.x"), cout, temporal);
        using Tensor sum = fixture.Add(main, shortcut);
        AssertClose(fixture.Reference($"{stage}.out"), sum, $"decoder upsample level {level} (temporal={temporal})", StageTolerance);
    }

    [Fact]
    public void EncoderMidAttentionMatchesTheReference()
    {
        using Fixture fixture = new();
        HunyuanVideo15VaeWeights weights = new(fixture.Weights);
        HunyuanVideo15VaeMidAttention attention = new(weights, "encoder.mid.attn_1", 1e-12f);
        using Tensor output = attention.Forward(fixture.Backend, fixture.Reference("stage.enc.mid.attn_1.in"));
        AssertClose(fixture.Reference("stage.enc.mid.attn_1.out"), output, "encoder mid attention", StageTolerance);
    }

    [Fact]
    public void DecoderMidAttentionMatchesTheReference()
    {
        using Fixture fixture = new();
        HunyuanVideo15VaeWeights weights = new(fixture.Weights);
        HunyuanVideo15VaeMidAttention attention = new(weights, "decoder.mid.attn_1", 1e-12f);
        using Tensor output = attention.Forward(fixture.Backend, fixture.Reference("stage.dec.mid.attn_1.in"));
        AssertClose(fixture.Reference("stage.dec.mid.attn_1.out"), output, "decoder mid attention", StageTolerance);
    }

    [Fact]
    public void DefaultConfigPlacesTemporalResamplersOnTheLastTwoSpatialLevels()
    {
        HunyuanVideo15VaeConfig config = HunyuanVideo15VaeConfig.Default;
        Assert.Equal([false, false, true, true, false], Enumerable.Range(0, 5).Select(config.EncoderTemporalDownsample).ToArray());
        Assert.Equal([true, true, false, false, false], Enumerable.Range(0, 5).Select(config.DecoderTemporalUpsample).ToArray());
        Assert.Equal([true, true, true, true, false], Enumerable.Range(0, 5).Select(config.EncoderSpatialDownsample).ToArray());
    }

    [Fact]
    public void ConfigRejectsASpatialFactorThatDisagreesWithTheLevelCount()
    {
        HunyuanVideo15VaeConfig config = TinyConfig() with { SpatialCompression = 8 };
        Assert.Throws<ArgumentException>(() => new HunyuanVideo15Vae(config));
    }

    [Fact]
    public void LoadRejectsAMissingTensor()
    {
        using Fixture fixture = new();
        Dictionary<string, Tensor> weights = new(fixture.Weights);
        weights.Remove("decoder.conv_out.conv.bias");
        HunyuanVideo15Vae vae = new(TinyConfig());
        Assert.Throws<KeyNotFoundException>(() => vae.LoadWeights(weights));
    }

    [Fact]
    public void LoadRejectsATensorTheLayersDoNotRead()
    {
        using Fixture fixture = new();
        Dictionary<string, Tensor> weights = new(fixture.Weights)
        {
            ["encoder.unused.weight"] = new Tensor(new TensorShape(1), DType.F32),
        };
        HunyuanVideo15Vae vae = new(TinyConfig());
        Assert.Throws<InvalidDataException>(() => vae.LoadWeights(weights));
    }

    private static void AssertClose(Tensor expected, Tensor actual, string label, float tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float max = 0f;
        for (int i = 0; i < e.Length; i++) max = Math.Max(max, Math.Abs(e[i] - a[i]));
        Assert.NotEqual(0, e.Length);
        Assert.True(max <= tolerance, $"{label}: max abs error {max:E3} exceeds {tolerance:E0}.");
    }

    private sealed class Fixture : IDisposable
    {
        public Fixture()
        {
            Loader = new SafeTensorsLoader();
            Loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_vae_tiny.safetensors"));
            Weights = Loader.GetAllTensors()
                .Where(pair => pair.Key.StartsWith("encoder.", StringComparison.Ordinal) || pair.Key.StartsWith("decoder.", StringComparison.Ordinal))
                .ToDictionary(pair => pair.Key, pair => pair.Value);
            Backend = new CpuBackend();
            Vae = new HunyuanVideo15Vae(TinyConfig());
            Vae.LoadWeights(Weights);
        }

        public SafeTensorsLoader Loader { get; }
        public Dictionary<string, Tensor> Weights { get; }
        public IBackend Backend { get; }
        public HunyuanVideo15Vae Vae { get; }

        public Tensor Reference(string name) => Loader.GetTensor(name);

        public Tensor Add(Tensor a, Tensor b)
        {
            Tensor sum = new(a.Shape, DType.F32);
            Backend.Add(sum, a, b);
            return sum;
        }

        public void Dispose() => Loader.Dispose();
    }
}
