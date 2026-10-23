using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Tiny Fibo 2.0 denoiser against the pure-torch reference (tests/python-reference/fibo2_reference.py, which
/// follows ComfyUI's fibo2 forward). The fixture holds seeded F32 weights under ComfyUI-native names with fused shapes,
/// the latent, sigma and the text bundles, and the reference velocity. Tolerance is the full-denoiser F32 scale from
/// docs/Agents/ADD_MODEL.md (1e-3); the measured error is written to the test output.</summary>
public sealed unsafe class Fibo2ParityTests(ITestOutputHelper output)
{
    private const float VelocityTolerance = 1e-3f;
    private static readonly string FixturePath = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Fibo2", "fibo2_tiny.safetensors");

    [Fact]
    public void Forward_TinyConfig_MatchesReferenceVelocity() => RunAgainstReference(0, "expected.velocity");

    /// <summary>Edit conditioning with one reference image. The Python reference shares our reading of ComfyUI's
    /// <c>ref_latents</c> handling, so this catches porting slips (token order, clean group, RoPE frame, refiner span), not a
    /// misreading both sides share.</summary>
    [Fact]
    public void Forward_OneReference_MatchesReferenceVelocity() => RunAgainstReference(1, "expected.velocity_ref1");

    /// <summary>Two references of different sizes, each on its own RoPE frame.</summary>
    [Fact]
    public void Forward_TwoReferences_MatchesReferenceVelocity() => RunAgainstReference(2, "expected.velocity_ref2");

    private void RunAgainstReference(int referenceCount, string expectedKey)
    {
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(FixturePath);
        Dictionary<string, Tensor> all = loader.GetAllTensors();
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach ((string key, Tensor value) in all)
        {
            if (key.StartsWith("weights.", StringComparison.Ordinal))
                weights[key["weights.".Length..]] = value;
        }

        Fibo2Config config = Fibo2Config.FromWeights(weights, TinyTemplate());
        Assert.Equal(64, config.Dim);
        Assert.Equal(2, config.NumHeads);
        Assert.Equal(2, config.NumLayers);
        Assert.Equal(new[] { 1 }, config.InjectionLayers);
        Assert.Equal(96, config.CapFeatDim);
        Assert.Equal(6, config.GistCount(40));

        using Fibo2Transformer transformer = new(config);
        transformer.LoadWeights(weights);
        float sigma = ((float*)all["inputs.sigma"].DataPointer)[0];
        List<Tensor> references = [];
        if (referenceCount > 0) references.Add(all["inputs.ref0"]);
        if (referenceCount > 1) references.Add(all["inputs.ref1"]);
        using Tensor velocity = transformer.Forward(new CpuBackend(), all["inputs.latent"], sigma, all["inputs.context"],
            referenceCount == 0 ? null : references);

        AssertClose(all[expectedKey], velocity, VelocityTolerance, expectedKey);
    }

    [Fact]
    public void Forward_TextMask_IsRejected()
    {
        Fibo2Config config = TinyTemplate() with
        {
            Dim = 64,
            NumHeads = 2,
            HeadDim = 32,
            NumLayers = 1,
            NumRefinerLayers = 0,
            PerceiverQueries = 8,
            PerceiverLayers = 1,
            FfnDim = 96,
            CapFeatDim = 96,
            InjectionLayers = [],
        };
        using Fibo2Transformer transformer = new(config);
        using Tensor latent = new(new TensorShape(1, 32, 2, 2), DType.F32);
        using Tensor context = new(new TensorShape(1, 4, 96), DType.F32);
        using Tensor mask = new(new TensorShape(1, 4), DType.F32);
        Assert.Throws<NotSupportedException>(() => transformer.Forward(new CpuBackend(), latent, 0.5f, context, null, mask));
    }

    [Fact]
    public void Forward_MisshapenReference_IsRejected()
    {
        Fibo2Config config = TinyTemplate() with
        {
            Dim = 64, NumHeads = 2, HeadDim = 32, NumLayers = 1, NumRefinerLayers = 0, PerceiverQueries = 8, PerceiverLayers = 1,
            FfnDim = 96, CapFeatDim = 96, InjectionLayers = [],
        };
        using Fibo2Transformer transformer = new(config);
        using Tensor latent = new(new TensorShape(1, 32, 2, 2), DType.F32);
        using Tensor context = new(new TensorShape(1, 4, 96), DType.F32);
        using Tensor odd = new(new TensorShape(1, 32, 3, 2), DType.F32);
        using Tensor wrongChannels = new(new TensorShape(1, 16, 2, 2), DType.F32);
        Assert.Throws<ArgumentException>(() => transformer.Forward(new CpuBackend(), latent, 0.5f, context, [odd]));
        Assert.Throws<ArgumentException>(() => transformer.Forward(new CpuBackend(), latent, 0.5f, context, [wrongChannels]));
    }

    /// <summary>The fixture's dimensions. Gist and RoPE settings match tests/python-reference/fibo2_reference.py TINY.</summary>
    private static Fibo2Config TinyTemplate() => Fibo2Config.Checkpoint with
    {
        InChannels = 32,
        PatchSize = 2,
        AxesDims = [8, 12, 12],
        RopeTheta = 256f,
        NormEps = 1e-5f,
        TimestepScale = 1000f,
        MaxGist = 8,
        MinGist = 4,
        GistStep = 2,
        GistMinTextLen = 16,
        GistMaxTextLen = 64,
    };

    private void AssertClose(Tensor expected, Tensor actual, float tolerance, string what)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = new((void*)expected.DataPointer, (int)expected.ElementCount);
        ReadOnlySpan<float> a = new((void*)actual.DataPointer, (int)actual.ElementCount);
        float worst = 0f;
        for (int i = 0; i < e.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(e[i] - a[i]));
        output.WriteLine($"{what}: max |Δ| = {worst:E3} (tolerance {tolerance:E0})");
        Assert.True(worst <= tolerance, $"{what}: max |Δ| = {worst} exceeds {tolerance}");
    }
}
