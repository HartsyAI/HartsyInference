using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class Trellis2PipelineConfigTests
{
    // The shape-relevant nodes of the shipped pipeline.json (TRELLIS.2-4B), with the texture entries left out.
    private const string Json = """
        {"name": "Trellis2ImageTo3DPipeline", "args": {
          "sparse_structure_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "guidance_interval": [0.6, 1.0], "rescale_t": 5.0}},
          "shape_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}},
          "shape_slat_normalization": {
            "mean": [0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
              0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
              -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665],
            "std": [5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
              5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
              5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691]},
          "default_pipeline_type": "1024_cascade"}}
        """;

    [Fact]
    public void ShippedSamplersAndNormalization_ParseToThePipelineJsonValues()
    {
        Trellis2PipelineConfig config = Trellis2PipelineConfig.FromJson(Json);

        Assert.Equal(12, config.SparseStructureSampler.Steps);
        Assert.Equal(7.5f, config.SparseStructureSampler.GuidanceStrength);
        Assert.Equal(0.7f, config.SparseStructureSampler.GuidanceRescale);
        Assert.Equal((0.6, 1.0), (config.SparseStructureSampler.IntervalLow, config.SparseStructureSampler.IntervalHigh));
        Assert.Equal(5.0, config.SparseStructureSampler.RescaleT);
        Assert.Equal(1e-5, config.SparseStructureSampler.SigmaMin);
        Assert.Equal(0.5f, config.ShapeSampler.GuidanceRescale);
        Assert.Equal(3.0, config.ShapeSampler.RescaleT);
        Assert.Equal(32, config.ShapeLatentMean.Length);
        Assert.Equal(0.781296f, config.ShapeLatentMean[0]);
        Assert.Equal(1.578665f, config.ShapeLatentMean[31]);
        Assert.Equal(5.972266f, config.ShapeLatentStd[0]);
        Assert.Equal(5.405691f, config.ShapeLatentStd[31]);
        Assert.Equal("1024_cascade", config.DefaultPipelineType);
    }

    [Fact]
    public void ANormalizationWithTheWrongChannelCountIsRejected()
    {
        string json = Json.Replace("1.578665]", "1.578665, 0.0]", StringComparison.Ordinal);

        Assert.Throws<ArgumentException>(() => Trellis2PipelineConfig.FromJson(json));
    }

    [Fact]
    public void AnotherSamplerIsRefusedByName()
    {
        string json = Json.Replace("FlowEulerGuidanceIntervalSampler", "FlowEulerCfgSampler", StringComparison.Ordinal);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Trellis2PipelineConfig.FromJson(json));
        Assert.Contains("FlowEulerCfgSampler", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AConfigOfAnotherPipelineIsRefused()
    {
        string json = Json.Replace("Trellis2ImageTo3DPipeline", "TrellisImageTo3DPipeline", StringComparison.Ordinal);

        Assert.Throws<NotSupportedException>(() => Trellis2PipelineConfig.FromJson(json));
    }

    [Fact]
    public void ANonPositiveNormalizationStdIsRejected()
    {
        string json = Json.Replace("5.972266", "0.0", StringComparison.Ordinal);

        Assert.Throws<ArgumentException>(() => Trellis2PipelineConfig.FromJson(json));
    }
}
