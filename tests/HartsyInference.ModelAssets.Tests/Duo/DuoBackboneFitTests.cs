using HartsyInference.ModelAssets.Duo;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Duo;

/// <summary>Every realization LoRA module of the released Duo checkpoint fits the backbone weight its key maps to: A reads the weight's input
/// dimension, B's rows are one slot of the weight's rows, and A and B share the rank. The check reads only the shard headers of the backbone, so
/// no tensor data is loaded. It needs the real LoRA folder and the backbone folder: <c>HARTSY_DUO_DCP_DIR</c> and
/// <c>HARTSY_DUO_BACKBONE_DIR</c>. Without them the test is reported as skipped.</summary>
public sealed class DuoBackboneFitTests
{
    [DuoRealWeightsFact("HARTSY_DUO_BACKBONE_DIR")]
    [Trait("Category", "Integration")]
    public void RealBackbone_EveryMappedLoraModuleFitsItsWeight()
    {
        string backbone = Environment.GetEnvironmentVariable("HARTSY_DUO_BACKBONE_DIR")!;
        string lora = Environment.GetEnvironmentVariable("HARTSY_DUO_DCP_DIR")
            ?? throw new InvalidOperationException("HARTSY_DUO_DCP_DIR must name the LoRA folder.");

        Dictionary<string, long[]> weights = [];
        foreach (string file in Directory.EnumerateFiles(backbone, "*.safetensors").OrderBy(f => f, StringComparer.Ordinal))
        {
            SafeTensorHeader header = SafeTensorHeaderReader.Read(file);
            foreach ((string key, SafeTensorDescriptor descriptor) in header.Tensors)
            {
                weights[key] = Enumerable.Range(0, descriptor.Shape.Rank).Select(i => descriptor.Shape[i]).ToArray();
            }
        }
        Assert.NotEmpty(weights);

        using DuoLoraSource source = DuoLoraSource.FromDcp(lora);
        Dictionary<string, long[]> shapes = source.Describe().ToDictionary(d => d.Key, d => d.Shape);
        Assert.Equal(317, source.Modules.Count);

        foreach (string module in source.Modules)
        {
            Assert.True(DuoLoraKeyMap.TryMap(module, out DuoLoraTarget target), $"'{module}' has no backbone target.");
            long[] a = shapes[module + DuoLoraKeyMap.LoraASuffix], b = shapes[module + DuoLoraKeyMap.LoraBSuffix];
            Assert.True(weights.TryGetValue(target.BackboneKey, out long[]? w), $"'{module}' maps to '{target.BackboneKey}', which is not in the backbone.");
            Assert.Equal(2, w!.Length);
            Assert.Equal(2, a.Length);
            Assert.Equal(2, b.Length);
            Assert.Equal(w[1], a[1]);
            Assert.Equal(a[0], b[1]);
            Assert.Equal(0, w[0] % target.Slots);
            Assert.Equal(w[0] / target.Slots, b[0]);
        }
    }
}
