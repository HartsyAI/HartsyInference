using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests;

/// <summary>The converted S generator equals the engine's own read of the Causal Forcing++ pickle, tensor for tensor. Both sides
/// go through the engine's reader (<c>PytorchEnvelopeReader</c> for the pickle, safetensors for the converted file), so this shows
/// the converter wrote what the reader read; it does not check the reader against upstream's <c>torch.load</c>, which the runbook
/// does not allow. Gated: skipped unless <c>FF_REAL_CONVERSION_PT</c> and <c>FF_REAL_CONVERSION_ST</c> name the two files.</summary>
public sealed class FastFramesConversionTests(ITestOutputHelper output)
{
    private const string Prefix = "model._fsdp_wrapped_module.";

    [RealWeightsParityFact]
    public void TheConvertedSGeneratorEqualsTheEngineReadOfThePickle()
    {
        string pt = Environment.GetEnvironmentVariable("FF_REAL_CONVERSION_PT")!;
        string st = Environment.GetEnvironmentVariable("FF_REAL_CONVERSION_ST")!;

        List<IDisposable> loaders = [];
        try
        {
            // The engine's read of the pickle, as the converter reads it: F32 copies under generator_ema, the wrapper prefix stripped.
            Dictionary<string, Tensor> fromPickle = FastFramesRecipe.ReadGenerator(pt, DType.F32, castForCpu: true, loaders);
            // The converted file as the engine reads it: passthrough (no cast), prefix kept.
            Dictionary<string, Tensor> fromFile = FastFramesRecipe.ReadGenerator(st, DType.F32, castForCpu: true, loaders);

            Assert.Equal(fromPickle.Count, fromFile.Count);
            int compared = 0;
            foreach ((string key, Tensor pickleTensor) in fromPickle)
            {
                string fileKey = Prefix + key;
                Assert.True(fromFile.TryGetValue(fileKey, out Tensor? fileTensor), $"converted file has no '{fileKey}'.");
                Assert.Equal(pickleTensor.DType, fileTensor!.DType);
                Assert.Equal(pickleTensor.ElementCount, fileTensor.ElementCount);
                ReadOnlySpan<float> a = pickleTensor.AsReadOnlySpan<float>();
                ReadOnlySpan<float> b = fileTensor.AsReadOnlySpan<float>();
                Assert.True(a.SequenceEqual(b), $"'{fileKey}' differs between the pickle read and the converted file.");
                compared++;
            }
            output.WriteLine($"compared {compared} tensors: the converted file equals the engine's read of the pickle, bit for bit.");
        }
        finally
        {
            foreach (IDisposable loader in loaders) loader.Dispose();
        }
    }
}
