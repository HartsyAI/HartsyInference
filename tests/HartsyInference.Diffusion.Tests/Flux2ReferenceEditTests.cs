using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FLUX.2 Klein reference-edit forward and pipeline Open Vision Banana runs on, on a tiny random checkpoint.
/// No torch reference was run here, so these pin the forward against the existing single-image forward and against the
/// layout diffusers' <c>Flux2KleinPipeline</c> implies (T = 10 + 10·i for references, one shared timestep, only the target
/// stepped); they are not parity with diffusers.</summary>
public sealed class Flux2ReferenceEditTests
{
    private const int Channels = IrisZTestData.Channels;
    private const int Context = IrisZTestData.Context;
    private const int GridHeight = 2;
    private const int GridWidth = 3;
    private const int TextLength = 5;
    private const int Size = 32;

    private static Flux2Config Tiny => IrisZTestData.Config(modalities: 0);

    [Fact]
    public void ReferenceTimesAreTenPerReferenceAfterTheTarget()
    {
        Assert.Equal([0f], Flux2EditPipeline.GridTimes(0));
        Assert.Equal([0f, 10f, 20f, 30f], Flux2EditPipeline.GridTimes(3));
    }

    /// <summary>The target alone is the single-image forward, so the shared-timestep path, the grid position ids and the
    /// final layer must reproduce it.</summary>
    [Fact]
    public void TheTargetAloneMatchesTheSingleImageForward()
    {
        using IDisposable f32 = KnobProfile.Create("ovb-test").With(EngineKnobs.DitF16, false).Push();
        using CpuBackend backend = new();
        using Flux2Transformer transformer = new(Tiny);
        transformer.LoadWeights(IrisZTestData.Converted(zeroModality: true));
        using Tensor tokens = IrisZTestData.Random(new Random(1), 1, GridHeight * GridWidth, Channels);
        using Tensor text = IrisZTestData.Random(new Random(2), 1, TextLength, Context);

        using Tensor expected = transformer.Forward(backend, tokens, text, 0.6f, 0f, GridHeight, GridWidth);
        using Tensor actual = transformer.ForwardWithReferences(backend, tokens, text, 0.6f, GridHeight, GridWidth, [0f]);

        Assert.Equal(expected.Shape, actual.Shape);
        Assert.True(MaxDifference(expected.AsReadOnlySpan<float>(), actual.AsReadOnlySpan<float>()) <= 1e-4f);
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void TheTargetVelocityRespondsToTheReferenceTokensAndTheirTimeCoordinate()
    {
        using IDisposable f32 = KnobProfile.Create("ovb-test").With(EngineKnobs.DitF16, false).Push();
        using CpuBackend backend = new();
        using Flux2Transformer transformer = new(Tiny);
        transformer.LoadWeights(IrisZTestData.Converted(zeroModality: true));
        int gridTokens = GridHeight * GridWidth;
        using Tensor tokens = IrisZTestData.Random(new Random(3), 1, 2 * gridTokens, Channels);
        using Tensor otherReference = IrisZTestData.Random(new Random(4), 1, 2 * gridTokens, Channels);
        using Tensor text = IrisZTestData.Random(new Random(5), 1, TextLength, Context);
        // Same target rows, different reference rows.
        float[] changed = tokens.AsReadOnlySpan<float>().ToArray();
        otherReference.AsReadOnlySpan<float>()[(gridTokens * Channels)..].CopyTo(changed.AsSpan(gridTokens * Channels));
        using Tensor swapped = new(tokens.Shape, DType.F32);
        changed.CopyTo(swapped.AsSpan<float>());

        float[] baseline = TargetVelocity(transformer, backend, tokens, text, [0f, 10f]);
        float[] otherRef = TargetVelocity(transformer, backend, swapped, text, [0f, 10f]);
        float[] otherTime = TargetVelocity(transformer, backend, tokens, text, [0f, 20f]);

        Assert.All(baseline, value => Assert.True(float.IsFinite(value)));
        Assert.Equal(gridTokens * Channels, baseline.Length);
        Assert.True(MaxDifference(baseline, otherRef) > 1e-4f, "the reference tokens reach the target velocity");
        Assert.True(MaxDifference(baseline, otherTime) > 1e-4f, "the reference's T coordinate reaches the target velocity");
    }

    /// <summary>The reference rope tables are keyed exactly and survive other forwards in between: the same layout gives the same
    /// answer before and after a plain forward and a differently-timed reference layout rewrote the shared tables.</summary>
    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void ReferenceRopeTablesAreRebuiltWhenAnotherLayoutHasUsedThem()
    {
        using IDisposable f32 = KnobProfile.Create("ovb-test").With(EngineKnobs.DitF16, false).Push();
        using CpuBackend backend = new();
        using Flux2Transformer transformer = new(Tiny);
        transformer.LoadWeights(IrisZTestData.Converted(zeroModality: true));
        int gridTokens = GridHeight * GridWidth;
        using Tensor two = IrisZTestData.Random(new Random(3), 1, 2 * gridTokens, Channels);
        using Tensor one = IrisZTestData.Random(new Random(4), 1, gridTokens, Channels);
        using Tensor text = IrisZTestData.Random(new Random(5), 1, TextLength, Context);

        float[] first = TargetVelocity(transformer, backend, two, text, [0f, 10f]);
        using (transformer.Forward(backend, one, text, 0.5f, 0f, GridHeight, GridWidth)) { }
        float[] afterPlain = TargetVelocity(transformer, backend, two, text, [0f, 10f]);
        float[] otherTimes = TargetVelocity(transformer, backend, two, text, [0f, 11f]);
        float[] afterOther = TargetVelocity(transformer, backend, two, text, [0f, 10f]);

        Assert.Equal(first, afterPlain);
        Assert.Equal(first, afterOther);
        Assert.True(MaxDifference(first, otherTimes) > 1e-4f, "a different T coordinate is a different layout");
    }

    [Fact]
    public void ALoneTargetSharesThePlainTablesAndMatchesAfterAReferenceLayoutRan()
    {
        using IDisposable f32 = KnobProfile.Create("ovb-test").With(EngineKnobs.DitF16, false).Push();
        using CpuBackend backend = new();
        using Flux2Transformer transformer = new(Tiny);
        transformer.LoadWeights(IrisZTestData.Converted(zeroModality: true));
        int gridTokens = GridHeight * GridWidth;
        using Tensor two = IrisZTestData.Random(new Random(3), 1, 2 * gridTokens, Channels);
        using Tensor one = IrisZTestData.Random(new Random(4), 1, gridTokens, Channels);
        using Tensor text = IrisZTestData.Random(new Random(5), 1, TextLength, Context);

        using Tensor plain = transformer.Forward(backend, one, text, 0.5f, 0f, GridHeight, GridWidth);
        _ = TargetVelocity(transformer, backend, two, text, [0f, 10f]);
        using Tensor lone = transformer.ForwardWithReferences(backend, one, text, 0.5f, GridHeight, GridWidth, [0f]);

        Assert.True(MaxDifference(plain.AsReadOnlySpan<float>(), lone.AsReadOnlySpan<float>()) <= 1e-4f);
    }

    [Fact]
    public void BadTextShapesAndNullReferencesAreRefusedByName()
    {
        using CpuBackend backend = new();
        using Flux2Transformer transformer = new(Tiny);
        transformer.LoadWeights(IrisZTestData.Converted(zeroModality: true));
        using Tensor tokens = IrisZTestData.Random(new Random(1), 1, GridHeight * GridWidth, Channels);
        using Tensor flatText = IrisZTestData.Random(new Random(2), TextLength, Context);
        using Tensor batchedText = IrisZTestData.Random(new Random(2), 2, TextLength, Context);

        Assert.Throws<ArgumentException>(() => transformer.ForwardWithReferences(backend, tokens, flatText, 0.5f, GridHeight, GridWidth, [0f]));
        Assert.Throws<ArgumentException>(() => transformer.ForwardWithReferences(backend, tokens, batchedText, 0.5f, GridHeight, GridWidth, [0f]));

        using Flux2EditFixture fixture = new();
        Assert.Throws<ArgumentNullException>(() => fixture.Pipeline.EditEncoded(fixture.Cond, null, [null!], Request(cfg: 1f)));
    }

    [Fact]
    public void AMismatchedTokenCountIsRefusedByName()
    {
        using CpuBackend backend = new();
        using Flux2Transformer transformer = new(Tiny);
        transformer.LoadWeights(IrisZTestData.Converted(zeroModality: true));
        using Tensor tokens = IrisZTestData.Random(new Random(1), 1, GridHeight * GridWidth, Channels);
        using Tensor text = IrisZTestData.Random(new Random(2), 1, TextLength, Context);

        Assert.Throws<ArgumentException>(() =>
            transformer.ForwardWithReferences(backend, tokens, text, 0.5f, GridHeight, GridWidth, [0f, 10f]));
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void ThePipelineEditsDeterministicallyAndTheReferenceMatters()
    {
        using Flux2EditFixture fixture = new();
        using Tensor reference = IrisZTestData.Random(new Random(9), 1, 3, Size, Size);
        using Tensor otherReference = IrisZTestData.Random(new Random(10), 1, 3, Size, Size);

        (byte[] first, int w, int h, int seed) = fixture.Pipeline.EditEncoded(fixture.Cond, null, [reference], Request(cfg: 1f));
        (byte[] second, _, _, _) = fixture.Pipeline.EditEncoded(fixture.Cond, null, [reference], Request(cfg: 1f));
        (byte[] other, _, _, _) = fixture.Pipeline.EditEncoded(fixture.Cond, null, [otherReference], Request(cfg: 1f));
        (byte[] none, _, _, _) = fixture.Pipeline.EditEncoded(fixture.Cond, null, [], Request(cfg: 1f));

        Assert.Equal((Size, Size, 4), (w, h, seed));
        Assert.Equal(Size * Size * 3, first.Length);
        Assert.Equal(first, second);
        Assert.NotEqual(first, other);
        Assert.NotEqual(first, none);
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void GuidanceNeedsTheUnconditionalBranchAndChangesTheImage()
    {
        using Flux2EditFixture fixture = new();
        using Tensor reference = IrisZTestData.Random(new Random(9), 1, 3, Size, Size);

        Assert.Throws<ArgumentException>(() => fixture.Pipeline.EditEncoded(fixture.Cond, null, [reference], Request(cfg: 3.5f)));
        (byte[] unguided, _, _, _) = fixture.Pipeline.EditEncoded(fixture.Cond, null, [reference], Request(cfg: 1f));
        (byte[] guided, _, _, _) = fixture.Pipeline.EditEncoded(fixture.Cond, fixture.Uncond, [reference], Request(cfg: 3.5f));

        Assert.NotEqual(unguided, guided);
    }

    [Fact]
    public void AReferenceOfAnotherSizeIsRefusedByName()
    {
        using Flux2EditFixture fixture = new();
        using Tensor wrong = IrisZTestData.Random(new Random(9), 1, 3, Size, Size / 2);

        Assert.Throws<ArgumentException>(() => fixture.Pipeline.EditEncoded(fixture.Cond, null, [wrong], Request(cfg: 1f)));
    }

    private static TextToImageRequest Request(float cfg) => new()
    {
        Prompt = "",
        Width = Size,
        Height = Size,
        Steps = 3,
        CfgScale = cfg,
        Seed = 4,
    };

    private static float[] TargetVelocity(Flux2Transformer transformer, CpuBackend backend, Tensor tokens, Tensor text, float[] times)
    {
        using Tensor velocity = transformer.ForwardWithReferences(backend, tokens, text, 0.5f, GridHeight, GridWidth, times);
        return velocity.AsReadOnlySpan<float>()[..(GridHeight * GridWidth * Channels)].ToArray();
    }

    private static float MaxDifference(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
    }

    /// <summary>A <see cref="Flux2EditPipeline"/> on the CPU over the tiny random transformer and VAE; the text encoder
    /// carries no weights, so the tests hand the pipeline encoded conditioning.</summary>
    private sealed class Flux2EditFixture : IDisposable
    {
        // The F16 block loop has no CPU kernels; the loop reads the knob per forward, so it stays pushed for the fixture's life.
        private readonly IDisposable _f32 = KnobProfile.Create("ovb-test").With(EngineKnobs.DitF16, false).Push();
        private readonly CpuBackend _backend = new();
        private readonly Flux2Transformer _transformer = new(Tiny);
        private readonly Tensor _bnMean = IrisZTestData.Filled(0f, Channels);
        private readonly Tensor _bnVar = IrisZTestData.Filled(1f, Channels);

        public Flux2EditFixture()
        {
            _transformer.LoadWeights(IrisZTestData.Converted(zeroModality: true));
            IrisZTestData.VaeWeights vaeWeights = new();
            VaeDecoder decoder = new(IrisZTestData.Vae);
            decoder.LoadWeights(vaeWeights);
            VaeEncoder encoder = new(IrisZTestData.Vae);
            encoder.LoadWeights(vaeWeights);
            Cond = IrisZTestData.Random(new Random(11), 1, TextLength, Context);
            Uncond = IrisZTestData.Random(new Random(12), 1, TextLength, Context);
            Pipeline = new Flux2EditPipeline(_backend, new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B), _transformer, decoder,
                encoder, _bnMean, _bnVar, Tiny, [9, 18, 27]);
        }

        public Flux2EditPipeline Pipeline { get; }

        public Tensor Cond { get; }

        public Tensor Uncond { get; }

        public void Dispose()
        {
            Pipeline.Dispose();
            _transformer.Dispose();
            _bnMean.Dispose();
            _bnVar.Dispose();
            Cond.Dispose();
            Uncond.Dispose();
            _backend.Dispose();
            _f32.Dispose();
        }
    }
}
