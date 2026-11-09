using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>CoLa3D decomposer (<c>cola3d_free_q20</c>, fp32) against the Python CPU reference dump. Integration lane: needs
/// <c>HARTSY_PARITY_DIR</c> and the checkpoints under Models.</summary>
[Trait("Category", "Integration")]
public sealed class CoLa3DDecomposerParityTests
{
    private const int Voxels = 6;
    private readonly ITestOutputHelper _output;

    public CoLa3DDecomposerParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void FreeQ20Fp32_MatchesPythonCpuReferenceStageByStage()
    {
        if (!ParityFixtures.TryResolve([ParityFixtures.DecomposerFixture], out string parityDir, out string modelRoot, out string reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        CoLa3DConfig config = CoLa3DConfig.FreeQ20;
        NpzArchive fixture = NpzArchive.Open(Path.Combine(parityDir, ParityFixtures.DecomposerFixture));
        // The fixture stores the exact inputs, so the test does not regenerate them from the seed.
        int[] coords = fixture.Get("coords").RequireShape(Voxels, 4).ToInt32();
        float[] latentFeats = fixture.Get("latent_feats").RequireShape(Voxels, config.LatentChannels).ToFloat32();
        float[] utoniaFeats = fixture.Get("utonia_feats").RequireShape(Voxels, config.UtoniaFeatureDim).ToFloat32();
        float[] refFused = fixture.Get("fused_input_feats").RequireShape(Voxels, config.HiddenDim).ToFloat32();
        float[] refQueryInit = fixture.Get("query_init_positional").RequireShape(1, config.MaxParts, config.HiddenDim).ToFloat32();
        float[] refPEnriched = fixture.Get("p_enriched_feats").RequireShape(Voxels, config.HiddenDim).ToFloat32();
        float[] refQueryFeats = fixture.Get("query_feats").RequireShape(1, config.MaxParts, config.HiddenDim).ToFloat32();
        float[] refMaskLogits = fixture.Get("mask_logits").RequireShape(1, Voxels, config.MaxParts).ToFloat32();
        float[] refTemperature = fixture.Get("temperature").RequireShape(1).ToFloat32();

        using CoLa3DWeights weights = CoLa3DCheckpointConverter.Load(modelRoot);
        using CpuBackend backend = new();
        using CoLa3DBackbone backbone = new(weights.Decomposer, config);
        CoLa3DFixedQueries queries = new(weights.Decomposer, config);
        using CoLa3DDecomposer decomposer = new(config, weights.Decomposer);

        Tensor latentFeatTensor = ToTensor(latentFeats, [Voxels, config.LatentChannels]);
        Tensor utoniaTensor = ToTensor(utoniaFeats, [Voxels, config.UtoniaFeatureDim]);
        Tensor fused = backbone.FuseInput(
            backend,
            latentFeatTensor.Reshape(new TensorShape(1, Voxels, config.LatentChannels)),
            utoniaTensor.Reshape(new TensorShape(1, Voxels, config.UtoniaFeatureDim)));
        Tensor queryInit = queries.Compute(backend);
        try
        {
            SparseTensor latent = new(latentFeatTensor, coords, config.LatentResolution);
            using CoLa3DOutput output = decomposer.Forward(backend, latent, utoniaTensor);

            ReadOnlySpan<float> temperature = weights.Decomposer["head.temperature"].AsReadOnlySpan<float>();
            List<ParityResult> results =
            [
                ParityCompare.Compare("fused_input", fused.AsReadOnlySpan<float>(), refFused),
                ParityCompare.Compare("query_init", queryInit.AsReadOnlySpan<float>(), refQueryInit),
                ParityCompare.Compare("p_enriched", output.PointFeatures.Feats.AsReadOnlySpan<float>(), refPEnriched),
                ParityCompare.Compare("query_feats", output.QueryFeatures.AsReadOnlySpan<float>(), refQueryFeats),
                ParityCompare.Compare("mask_logits", output.MaskLogits.AsReadOnlySpan<float>(), refMaskLogits),
                ParityCompare.Compare("temperature", temperature, refTemperature),
            ];
            ParityCompare.Finish(results, _output);
        }
        finally
        {
            fused.Dispose();
            queryInit.Dispose();
            latentFeatTensor.Dispose();
            utoniaTensor.Dispose();
        }
    }

    private static Tensor ToTensor(float[] values, long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }
}
