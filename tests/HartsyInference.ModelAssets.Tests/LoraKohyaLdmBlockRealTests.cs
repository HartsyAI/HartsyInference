using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>A stock CivitAI SDXL LoRA against the real SDXL base checkpoint. The synthetic tests next door pin the
/// mapping rules on representative keys; this pins that every module of a file the world actually ships resolves to a
/// key the loaded model has, which is the property a rule written from hand-built keys can satisfy and still miss.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class LoraKohyaLdmBlockRealTests
{
    private readonly ITestOutputHelper _output;
    public LoraKohyaLdmBlockRealTests(ITestOutputHelper output) => _output = output;

    /// <summary>Every one of the file's modules must name a weight the converted checkpoint holds — a key that lands
    /// nowhere merges silently, and only the all-components-zero case is refused at generation time.</summary>
    [Fact]
    public void RealSdxlLora_EveryModuleNamesAKeyTheCheckpointHas()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.Lora.HarrlogosSdxl, TestPaths.Sdxl.SingleFile)) return;

        using LoraFile file = LoraFile.Load(TestPaths.Lora.HarrlogosSdxl);
        Assert.Equal(LoraFormat.KohyaSdxl, file.Format);

        using SafeTensorsLoader checkpoint = new();
        checkpoint.Load(TestPaths.Sdxl.SingleFile);
        HashSet<string> unetKeys = [];
        HashSet<string> clipKeys = [];
        const string UnetPrefix = "model.diffusion_model.";
        const string ClipLPrefix = "conditioner.embedders.0.transformer.";
        foreach (string key in checkpoint.Descriptors.Keys)
        {
            if (key.StartsWith(UnetPrefix, StringComparison.Ordinal)
                && SdxlCheckpointConverter.ConvertUNetKey(key[UnetPrefix.Length..]) is string converted)
            {
                unetKeys.Add(converted);
            }
            else if (key.StartsWith(ClipLPrefix, StringComparison.Ordinal))
            {
                clipKeys.Add(key[ClipLPrefix.Length..]);
            }
        }

        List<string> unmatched = [];
        int unet = 0, clipL = 0, clipG = 0;
        foreach (LoraLayer layer in file.Layers)
        {
            switch (layer.Target)
            {
                case LoraTarget.UNet:
                    unet++;
                    if (!unetKeys.Contains(layer.TargetKey)) unmatched.Add(layer.TargetKey);
                    break;
                // CLIP-G is OpenCLIP in the checkpoint (fused in_proj, ln_1/c_fc naming), so it has no pre-conversion
                // key set to compare against; CLIP-L shares the layer path, and the two halves are named identically.
                case LoraTarget.ClipL:
                    clipL++;
                    if (!clipKeys.Contains(layer.TargetKey)) unmatched.Add(layer.TargetKey);
                    break;
                case LoraTarget.ClipG:
                    clipG++;
                    break;
            }
        }

        _output.WriteLine($"{file.Layers.Count} modules: {unet} UNet, {clipL} CLIP-L, {clipG} CLIP-G.");
        Assert.Empty(unmatched);
        Assert.Equal(722, unet);
        Assert.Equal(72, clipL);
        Assert.Equal(192, clipG);
        Assert.Equal(986, file.Layers.Count);
    }
}
