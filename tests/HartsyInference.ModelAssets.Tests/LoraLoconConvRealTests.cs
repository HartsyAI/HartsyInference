using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The conv half of the LoRA merge, against a real LoCon. Rank-4 convolution modules have been supported
/// since the conv-LoRA work landed, but every test until now built its deltas by hand — so what was pinned was the
/// arithmetic, not that a shipped adapter's conv modules resolve to weights this engine actually holds. Those are
/// different claims, and only the second one fails when a key rule is wrong.</summary>
/// <remarks>The adapter is Pony-trained. That is irrelevant here: it shares SDXL's UNet and kohya's key grammar,
/// and what these assert is that the conv path resolves and fits, not that the output looks like anything.</remarks>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LoraLoconConvRealTests
{
    private readonly ITestOutputHelper _output;
    public LoraLoconConvRealTests(ITestOutputHelper output) => _output = output;

    /// <summary>A LoCon carries rank-4 modules at all. If this drops to zero the file was swapped for a plain LoRA
    /// and every other assertion here becomes vacuous while still passing.</summary>
    [Fact]
    public void TheAdapterActuallyCarriesConvModules()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.Lora.LoconSdxl)) return;

        using LoraFile file = LoraFile.Load(TestPaths.Lora.LoconSdxl);
        int conv = CountConvModules(file);
        _output.WriteLine($"{file.Layers.Count} modules, {conv} of them convolutions.");
        Assert.True(conv > 0, "This file has no rank-4 modules — it is not a LoCon, and the conv path is untested.");
    }

    /// <summary>Every module must name a weight the converted checkpoint holds, convolutions included. A conv key
    /// that lands nowhere merges silently: the linear modules still apply, the image still changes, and the conv
    /// contribution is simply absent — which is exactly the failure a "does the output differ" check cannot see.</summary>
    [Fact]
    public void EveryConvModuleNamesAKeyTheCheckpointHas()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.Lora.LoconSdxl, TestPaths.Sdxl.SingleFile)) return;

        using LoraFile file = LoraFile.Load(TestPaths.Lora.LoconSdxl);
        using SafeTensorsLoader checkpoint = new();
        checkpoint.Load(TestPaths.Sdxl.SingleFile);
        const string UnetPrefix = "model.diffusion_model.";
        Dictionary<string, SafeTensorDescriptor> unet = [];
        foreach ((string key, SafeTensorDescriptor descriptor) in checkpoint.Descriptors)
        {
            if (key.StartsWith(UnetPrefix, StringComparison.Ordinal)
                && SdxlCheckpointConverter.ConvertUNetKey(key[UnetPrefix.Length..]) is string converted)
            {
                unet[converted] = descriptor;
            }
        }

        List<string> unmatched = [];
        int convChecked = 0, shapeMismatched = 0;
        foreach (LoraLayer layer in file.Layers)
        {
            if (layer.Target != LoraTarget.UNet || !IsConv(layer))
            {
                continue;
            }
            convChecked++;
            if (!unet.TryGetValue(layer.TargetKey, out SafeTensorDescriptor? descriptor))
            {
                unmatched.Add(layer.TargetKey);
                continue;
            }
            // The base weight is rank 4; the delta folds the kernel axes into its columns, so the two agree only
            // if the fold matches the base's own in-channel x kernel product.
            long folded = 1;
            for (int i = 1; i < descriptor.Shape.Rank; i++)
            {
                folded *= descriptor.Shape[i];
            }
            if (layer.Delta.OutFeatures != descriptor.Shape[0] || layer.Delta.InFeatures != folded)
            {
                shapeMismatched++;
                _output.WriteLine($"shape mismatch {layer.TargetKey}: delta {layer.Delta.OutFeatures}x{layer.Delta.InFeatures} "
                    + $"vs base rank {descriptor.Shape.Rank}");
            }
        }

        _output.WriteLine($"{convChecked} conv modules checked, {unmatched.Count} unmatched, {shapeMismatched} mis-shaped.");
        Assert.True(convChecked > 0, "No UNet conv modules were checked — the filter or the file changed.");
        Assert.Empty(unmatched);
        Assert.Equal(0, shapeMismatched);
    }

    /// <summary>Rank-4 modules are the ones whose down weight keeps its kernel axes.</summary>
    private static bool IsConv(LoraLayer layer) =>
        layer.Variant == LoraVariant.StandardLora && layer.LoraDown.Shape.Rank == 4;

    private static int CountConvModules(LoraFile file)
    {
        int conv = 0;
        foreach (LoraLayer layer in file.Layers)
        {
            if (IsConv(layer))
            {
                conv++;
            }
        }
        return conv;
    }
}
