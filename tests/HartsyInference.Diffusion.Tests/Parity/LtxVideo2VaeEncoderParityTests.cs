using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Diffusion.Tests.LtxVideo2VaeEncoderFixtures;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Tiny-config encoder against diffusers 0.37.0 <c>AutoencoderKLLTX2Video.encode(x).latent_dist.mode()</c> on CPU.</summary>
public sealed class LtxVideo2VaeEncoderParityTests
{
    private const float AbsTolerance = 1e-4f;
    private readonly ITestOutputHelper _output;

    public LtxVideo2VaeEncoderParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Encode_MatchesDiffusersPosteriorModeOnTinyConfig()
    {
        using SafeTensorsLoader weights = Open(WeightsFile);
        using SafeTensorsLoader io = Open(IoFile);
        IReadOnlyDictionary<string, TensorShape> expectedShapes = LtxVideo2VaeEncoder.ExpectedWeightShapes(TinyConfig());
        HashSet<string> fixtureKeys = weights.Descriptors.Keys.ToHashSet(StringComparer.Ordinal);
        Assert.True(fixtureKeys.SetEquals(expectedShapes.Keys), "fixture encoder keys differ from the config-derived set");

        LtxVideo2VaeEncoder encoder = LoadEncoder(TinyConfig(), weights.GetAllTensors());
        using IBackend backend = new CpuBackend();
        Tensor input = io.GetTensor("input");
        Tensor expected = io.GetTensor("expected_mode");
        using Tensor actual = encoder.Encode(backend, input);

        Assert.Equal(expected.Shape, actual.Shape);
        float maxAbs = MaxAbsDiff(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        float scale = MaxAbs(expected.AsReadOnlySpan<float>());
        _output.WriteLine(
            $"max abs {maxAbs:E3}, max rel (vs max |expected| {scale:F4}) {maxAbs / scale:E3}, " +
            $"latent elements {expected.ElementCount}");
        Assert.True(maxAbs <= AbsTolerance, $"max abs error {maxAbs:E3} exceeds {AbsTolerance:E0}");
    }
}
