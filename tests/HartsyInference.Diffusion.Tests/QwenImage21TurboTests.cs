using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Qwen-Image-2.1-Turbo: variant resolution from the file name, its fixed 8-step schedule, the diffusers
/// split-MLP fuse, and finding the whole shard set from one shard. None of this needs weights on disk.</summary>
public sealed unsafe class QwenImage21TurboTests : IDisposable
{
    private readonly List<Tensor> _owned = new();
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "qwen21turbo-" + Guid.NewGuid().ToString("N"));

    public QwenImage21TurboTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        foreach (Tensor tensor in _owned)
        {
            tensor.Dispose();
        }
        Directory.Delete(_dir, recursive: true);
    }

    [Theory]
    [InlineData("Qwen-Image-2.1-Turbo-00001-of-00002.safetensors", "turbo")]
    [InlineData("qwen_image_2.1_turbo.safetensors", "turbo")]
    [InlineData("Qwen-Image-2.1-00001-of-00002.safetensors", "base")]
    [InlineData("qwen_image_2.1_bf16.safetensors", "base")]
    public void Variant_ResolvesFromFileName(string fileName, string expected)
    {
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(QwenImage21Variants.Catalog, Probe(fileName), []);
        Assert.Equal(expected, resolved.Id);
    }

    [Fact]
    public void TurboDefaults_AreEightStepsGuidanceFree()
    {
        ResolvedModelVariant turbo = ModelVariantResolver.Classify(QwenImage21Variants.Catalog, Probe("Qwen-Image-2.1-Turbo-00001-of-00002.safetensors"), []);
        ResolvedModelVariant bas = ModelVariantResolver.Classify(QwenImage21Variants.Catalog, Probe("qwen_image_2.1_bf16.safetensors"), []);

        QwenImage21Recipe recipe = new();
        Assert.Equal(8, recipe.DefaultsFor(turbo).Steps);
        Assert.Equal(1.0f, recipe.DefaultsFor(turbo).CfgScale);
        Assert.Equal(25, recipe.DefaultsFor(bas).Steps);
    }

    [Fact]
    public void TurboCatalogId_SharesTheBaseSamplerTable()
    {
        Assert.True(SamplingCapabilities.HasImageEntry("qwen-image-2.1-turbo"));
        Assert.Equal(SamplingCapabilities.ForImage("qwen-image-2.1").Samplers, SamplingCapabilities.ForImage("qwen-image-2.1-turbo").Samplers);
    }

    [Fact]
    public void TurboSigmas_HaveEightDescendingEntriesEndingAboveZero()
    {
        IReadOnlyList<float> sigmas = QwenImage21Variants.TurboSigmas;
        Assert.Equal(8, sigmas.Count);
        Assert.Equal(1.0f, sigmas[0]);
        for (int i = 1; i < sigmas.Count; i++)
        {
            Assert.True(sigmas[i] < sigmas[i - 1], $"sigma {i} is not below sigma {i - 1}");
        }
        Assert.True(sigmas[^1] > 0f);
    }

    [Fact]
    public void SetSigmas_UsesTheGivenScheduleAndAppendsTerminalZero()
    {
        float[] sigmas = [1.0f, 0.5f, 0.25f];
        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 1.0f);
        scheduler.SetSigmas(sigmas);

        Assert.Equal(3, scheduler.NumInferenceSteps);
        Assert.Equal(new float[] { 1.0f, 0.5f, 0.25f, 0.0f }, scheduler.Sigmas());
        Assert.Equal(1000.0f, scheduler.Timesteps[0]);
        Assert.Equal(-0.25f, scheduler.Dt(2));
        Assert.True(scheduler.HasExplicitSigmas);

        scheduler.SetTimesteps(4);
        Assert.False(scheduler.HasExplicitSigmas);
    }

    [Fact]
    public void SetSigmas_RejectsEmptySchedule()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 1.0f);
        Assert.Throws<ArgumentException>(() => scheduler.SetSigmas(ReadOnlySpan<float>.Empty));
    }

    [Fact]
    public void SplitMlp_FusesToGateFirstGateUp()
    {
        // The diffusers block is out(silu(gate_layer(x)) * proj(x)); the engine's fused gate_up takes the gate first.
        Tensor gate = Make(2, 3, 1f);
        Tensor proj = Make(2, 3, 10f);
        Dictionary<string, Tensor> source = new()
        {
            ["transformer_blocks.0.img_mlp.gate_layer.weight"] = gate,
            ["transformer_blocks.0.img_mlp.proj.weight"] = proj,
            ["transformer_blocks.0.img_mlp.out.weight"] = Make(3, 2, 5f),
        };

        QwenImage21CheckpointConverter.ConvertedWeights converted = QwenImage21CheckpointConverter.Convert(source);

        Assert.False(converted.Transformer.ContainsKey("transformer_blocks.0.img_mlp.gate_layer.weight"));
        Assert.False(converted.Transformer.ContainsKey("transformer_blocks.0.img_mlp.proj.weight"));
        Tensor fused = converted.Transformer["transformer_blocks.0.img_mlp.gate_up.weight"];
        Assert.Equal(4L, fused.Shape[0]);
        Assert.Equal(3L, fused.Shape[1]);
        float* p = (float*)fused.DataPointer;
        Assert.Equal(1f, p[0]);          // gate row 0
        Assert.Equal(1f, p[5]);          // gate row 1
        Assert.Equal(10f, p[6]);         // proj row 0 starts at mlpDim
        Assert.Equal(10f, p[11]);        // proj row 1
        Assert.True(converted.Transformer.ContainsKey("transformer_blocks.0.img_mlp.out.weight"));
        _owned.Add(fused);
    }

    [Fact]
    public void SplitMlp_LeftoverSplitTensorIsAnError()
    {
        Dictionary<string, Tensor> source = new()
        {
            ["transformer_blocks.0.img_mlp.gate_layer.weight"] = Make(2, 3, 1f),
            ["transformer_blocks.0.img_mlp.proj.weight"] = Make(2, 3, 2f),
            ["transformer_blocks.0.img_mlp.gate_layer.bias"] = Make(1, 2, 0f),
        };
        Assert.Throws<InvalidOperationException>(() => QwenImage21CheckpointConverter.Convert(source));
    }

    [Fact]
    public void SetSigmas_RejectsNonDescendingOrNonPositive()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 1.0f);
        Assert.Throws<ArgumentException>(() => scheduler.SetSigmas([0.5f, 0.75f]));
        Assert.Throws<ArgumentException>(() => scheduler.SetSigmas([1.0f, 0.0f]));
    }

    [Theory]
    [InlineData("euler")]
    [InlineData("dpmpp_2m")]
    [InlineData("dpm_2")]
    [InlineData("dpm_2_ancestral")]
    [InlineData("uni_pc")]
    [InlineData("uni_pc_bh2")]
    public void ExplicitSchedule_IsKeptByEverySampler(string sampler)
    {
        // The shipped sigmas (terminal 0 appended) must come back unchanged for every sampler. The discard-penultimate
        // samplers (dpm_2, uni_pc, ...) would otherwise interpolate them onto a different grid.
        float[] shipped = [.. QwenImage21Variants.TurboSigmas, 0.0f];
        float[] built = SamplerRegistry.BuildSigmas(sampler, null, shipped, false, null, explicitSchedule: true);
        Assert.Equal(shipped, built);
    }

    [Fact]
    public void ExplicitSchedule_ControlShowsTheDiscardRuleOtherwiseChangesTheGrid()
    {
        // The negative control for the test above: without the explicit flag, a discard-penultimate sampler does
        // rebuild the grid, so the flag is what protects Turbo's sigmas and the test has teeth.
        float[] shipped = [.. QwenImage21Variants.TurboSigmas, 0.0f];
        float[] rebuilt = SamplerRegistry.BuildSigmas("dpm_2", null, shipped, false, null, explicitSchedule: false);
        Assert.NotEqual(shipped, rebuilt);
    }

    [Fact]
    public void SplitMlp_MissingProjIsAnError()
    {
        Dictionary<string, Tensor> source = new()
        {
            ["transformer_blocks.0.img_mlp.gate_layer.weight"] = Make(2, 3, 1f),
        };
        Assert.Throws<InvalidOperationException>(() => QwenImage21CheckpointConverter.Convert(source));
    }

    [Fact]
    public void ShardDiscovery_FindsSiblingsFromAnyShard()
    {
        string first = Touch("Qwen-Image-2.1-Turbo-00001-of-00002.safetensors");
        string second = Touch("Qwen-Image-2.1-Turbo-00002-of-00002.safetensors");

        Assert.Equal(new[] { first, second }, ShardSetDiscovery.Resolve(second));
        Assert.Equal(new[] { first, second }, ShardSetDiscovery.Resolve(first));
    }

    [Fact]
    public void ShardDiscovery_SingleFileIsReturnedAsIs()
    {
        string single = Touch("qwen_image_2.1_bf16.safetensors");
        Assert.Equal(new[] { single }, ShardSetDiscovery.Resolve(single));
    }

    [Fact]
    public void ShardDiscovery_MissingSiblingIsAnError()
    {
        string first = Touch("Qwen-Image-2.1-Turbo-00001-of-00002.safetensors");
        Assert.Throws<FileNotFoundException>(() => ShardSetDiscovery.Resolve(first));
    }

    private string Touch(string name)
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllBytes(path, []);
        return path;
    }

    private Tensor Make(int rows, int cols, float value)
    {
        Tensor tensor = new(new TensorShape(rows, cols), DType.F32);
        _owned.Add(tensor);
        float* p = (float*)tensor.DataPointer;
        for (int i = 0; i < rows * cols; i++)
        {
            p[i] = value;
        }
        return tensor;
    }

    private static CheckpointProbe Probe(string fileName) => CheckpointProbe.Empty with { FileNames = [fileName] };
}
