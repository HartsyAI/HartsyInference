using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FLUX.2 multi-grid forward Iris-Z runs on, on a tiny random checkpoint in BFL key names. The reference
/// implementation was not run here, so these pin the forward against the existing single-image FLUX.2 forward and
/// against the properties the reference's layout implies; they are not parity with speridlabs/iris-z-sdk.</summary>
public sealed class IrisZTransformerTests
{
    private const int Hidden = IrisZTestData.Hidden;
    private const int Channels = IrisZTestData.Channels;
    private const int Context = IrisZTestData.Context;
    private const int GridHeight = 2;
    private const int GridWidth = 3;
    private const int TextLength = 5;

    private static Flux2Config TinyConfig(int modalities) => IrisZTestData.Config(modalities);

    [Fact]
    public void ConverterKeepsTheModalityEmbedding()
    {
        Dictionary<string, Tensor> converted = Converted(zeroModality: false);

        Assert.True(converted.ContainsKey(Flux2CheckpointConverter.ModalityEmbeddingKey));
        Assert.Equal(new TensorShape(4, Hidden), converted[Flux2CheckpointConverter.ModalityEmbeddingKey].Shape);
    }

    [Fact]
    public void LoaderRequiresTheModalityEmbeddingWhenConfigured()
    {
        Dictionary<string, Tensor> converted = Converted(zeroModality: false);
        converted.Remove(Flux2CheckpointConverter.ModalityEmbeddingKey);
        using Flux2Transformer transformer = new(TinyConfig(modalities: 4));

        Assert.Throws<KeyNotFoundException>(() => transformer.LoadWeights(converted));
    }

    [Fact]
    public void MultiGridForwardNeedsAModalityEmbedding()
    {
        using CpuBackend backend = new();
        using Flux2Transformer plain = new(TinyConfig(modalities: 0));
        plain.LoadWeights(Converted(zeroModality: true));
        using Tensor tokens = Random(new Random(1), 1, GridHeight * GridWidth, Channels);
        using Tensor text = Random(new Random(2), 1, TextLength, Context);

        Assert.Throws<InvalidOperationException>(() =>
            plain.ForwardGrids(backend, tokens, text, 0.5f, [new Flux2Grid(0.5f, 0, 0f)], GridHeight, GridWidth));
    }

    /// <summary>One grid at T = 0 with a zero modality row and the text's own sigma is exactly the single-image forward,
    /// so the per-segment modulation, the grid position ids and the per-grid final layer must reproduce it.</summary>
    [Fact]
    public void OneGridAtTheTextSigmaMatchesTheSingleImageForward()
    {
        // The single-image forward's F16 block loop has no CPU kernels; both sides run the F32 stream.
        using IDisposable f32 = KnobProfile.Create("iris-z-test").With(EngineKnobs.DitF16, false).Push();
        using CpuBackend backend = new();
        Dictionary<string, Tensor> converted = Converted(zeroModality: true);
        using Flux2Transformer transformer = new(TinyConfig(modalities: 4));
        transformer.LoadWeights(converted);
        using Tensor tokens = Random(new Random(1), 1, GridHeight * GridWidth, Channels);
        using Tensor text = Random(new Random(2), 1, TextLength, Context);

        using Tensor expected = transformer.Forward(backend, tokens, text, 0.6f, 0f, GridHeight, GridWidth);
        using Tensor actual = transformer.ForwardGrids(backend, tokens, text, 0.6f, [new Flux2Grid(0.6f, 0, 0f)], GridHeight, GridWidth);

        AssertClose(expected, actual, 1e-4f);
    }

    /// <summary>Loaded under <c>numerics.ditF16</c> a backend may ride the residual stream at a damped scale that every LayerNorm cancels;
    /// the modality rows must enter at that scale too, or the two loads disagree.</summary>
    [Fact]
    public void ResidualDampLeavesTheMultiGridForwardUnchanged()
    {
        using CpuBackend backend = new();
        using Tensor tokens = Random(new Random(5), 1, 2 * GridHeight * GridWidth, Channels);
        using Tensor text = Random(new Random(6), 1, TextLength, Context);
        Flux2Grid[] grids = [new Flux2Grid(0f, 2, 0f), new Flux2Grid(0.7f, 1, 1f)];

        float[] plain = RunLoaded(backend, tokens, text, grids, ditF16: false);
        float[] damped = RunLoaded(backend, tokens, text, grids, ditF16: true);

        float difference = MaxDifference(plain, damped);
        Assert.True(difference <= 1e-3f, $"max abs difference {difference} between the damped and undamped loads");
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void TwoGridsRespondToTheirOwnSigmaModalityAndPosition()
    {
        using CpuBackend backend = new();
        using Flux2Transformer transformer = new(TinyConfig(modalities: 4));
        transformer.LoadWeights(Converted(zeroModality: false));
        int gridTokens = GridHeight * GridWidth;
        using Tensor tokens = Random(new Random(3), 1, 2 * gridTokens, Channels);
        using Tensor text = Random(new Random(4), 1, TextLength, Context);

        float[] joint = Run(transformer, backend, tokens, text, [new Flux2Grid(0.5f, 0, 0f), new Flux2Grid(0.5f, 1, 1f)]);
        float[] depthAtNoise = Run(transformer, backend, tokens, text, [new Flux2Grid(0.5f, 0, 0f), new Flux2Grid(1f, 1, 1f)]);
        float[] givenDepth = Run(transformer, backend, tokens, text, [new Flux2Grid(0.5f, 0, 0f), new Flux2Grid(0.5f, 3, 1f)]);
        float[] samePosition = Run(transformer, backend, tokens, text, [new Flux2Grid(0.5f, 0, 0f), new Flux2Grid(0.5f, 1, 0f)]);

        Assert.Equal(2 * gridTokens * Channels, joint.Length);
        Assert.All(joint, value => Assert.True(float.IsFinite(value)));
        Assert.True(MaxDifference(joint, depthAtNoise) > 1e-4f, "the depth grid's sigma reaches the output");
        Assert.True(MaxDifference(joint, givenDepth) > 1e-4f, "the modality row reaches the output");
        Assert.True(MaxDifference(joint, samePosition) > 1e-4f, "the grid's T coordinate reaches the output");
    }

    [Fact]
    public void GridPositionIdsPlaceEachGridAtItsOwnTime()
    {
        using Tensor ids = Flux2PosEmbed.BuildGridPositionIds(2, 1, 2, [0f, 1f]);
        using Tensor single = Flux2PosEmbed.BuildPositionIds(2, 1, 2);

        float[] expected =
        [
            0, 0, 0, 0,   0, 0, 0, 1,
            0, 0, 0, 0,   0, 0, 1, 0,
            1, 0, 0, 0,   1, 0, 1, 0,
        ];
        Assert.Equal(expected, ids.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(expected[..16], single.AsReadOnlySpan<float>().ToArray());
    }

    private static float[] RunLoaded(CpuBackend backend, Tensor tokens, Tensor text, Flux2Grid[] grids, bool ditF16)
    {
        using IDisposable knob = KnobProfile.Create("iris-z-test").With(EngineKnobs.DitF16, ditF16).Push();
        using Flux2Transformer transformer = new(TinyConfig(modalities: 4));
        transformer.LoadWeights(Converted(zeroModality: false));
        return Run(transformer, backend, tokens, text, grids);
    }

    private static float[] Run(Flux2Transformer transformer, CpuBackend backend, Tensor tokens, Tensor text, Flux2Grid[] grids)
    {
        using Tensor velocity = transformer.ForwardGrids(backend, tokens, text, 0.5f, grids, GridHeight, GridWidth);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        float difference = MaxDifference(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray());
        Assert.True(difference <= tolerance, $"max abs difference {difference} exceeds {tolerance}");
    }

    private static float MaxDifference(float[] a, float[] b)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
    }

    private static Dictionary<string, Tensor> Converted(bool zeroModality) => IrisZTestData.Converted(zeroModality);

    private static Tensor Random(Random rng, params long[] dims) => IrisZTestData.Random(rng, dims);
}
