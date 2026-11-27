using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FP8 FFN quantization on released BF16 weights: <c>ffn.w1</c> and <c>ffn.w3</c> of the two blocks that bound the selection's
/// first run, 6 and 9. Runs on the CPU and only when HARTSY_LONGCAT_FP8_BF16_INDEX names an index whose shards hold them.</summary>
public sealed class LongCatFp8FfnReleaseTests(ITestOutputHelper output)
{
    [Trait("Category", "Integration")]
    [RealWeightsFact("HARTSY_LONGCAT_FP8_BF16_INDEX")]
    public void TheReleasedFfnWeightsAreWithinTheirFp8BoundOfTheirBf16Form()
    {
        string index = Environment.GetEnvironmentVariable("HARTSY_LONGCAT_FP8_BF16_INDEX")!;
        using ShardedSafeTensorSet release = ShardedSafeTensorSet.OpenIndex(Path.GetDirectoryName(index)!, null, Path.GetFileName(index));
        int[] blocks = [6, 9];
        Dictionary<string, Tensor> weights = LongCatFp8FfnConverter.Keys(blocks).ToDictionary(key => key, release.GetTensor, StringComparer.Ordinal);
        List<Tensor> owned = [];
        try
        {
            Assert.Equal(4, LongCatFp8FfnConverter.Convert(weights, owned, blocks));
            List<string> failures = [];
            foreach ((string key, Tensor packed) in weights)
            {
                using Tensor source = release.GetTensor(key).CastTo(DType.F32);
                LongCatFp8FfnCheck.Report report = LongCatFp8FfnCheck.Compare(source, packed);
                output.WriteLine($"{key,-28} {packed.Shape}  scale {packed.Fp8ScaleFactor:E4}  relRMS {report.RelRms:E3}  "
                    + $"worst {report.Worst:F4} of the bound  over {report.Violations}  rounded down at the subnormal edge {report.EdgeRoundedDown}");
                if (report.Violations != 0 || !report.ScaleIsAbsMaxOver448)
                    failures.Add($"{key}: {report}");
            }
            Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }
}
