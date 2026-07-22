using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Face;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Real-weight parity for the ArcFace IR-50 face-embedding backbone (buffalo_l w600k_r50) against an
/// onnxruntime oracle. Reference IO comes from <c>tests/python-reference/convert_arcface_onnx.py</c>, which
/// also produces the converted safetensors. Gated on <c>ARCFACE_WEIGHTS</c> + <c>ARCFACE_REF</c>.</summary>
public sealed class ArcFaceParityTests
{
    private readonly ITestOutputHelper _output;

    public ArcFaceParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public unsafe void Forward_MatchesOnnxRuntime_CosineAbove0999()
    {
        string? weightsPath = Environment.GetEnvironmentVariable("ARCFACE_WEIGHTS");
        string? refPath = Environment.GetEnvironmentVariable("ARCFACE_REF");
        if (string.IsNullOrEmpty(weightsPath) || !File.Exists(weightsPath)
            || string.IsNullOrEmpty(refPath) || !File.Exists(refPath))
        {
            _output.WriteLine("SKIPPED: set ARCFACE_WEIGHTS + ARCFACE_REF (run tests/python-reference/convert_arcface_onnx.py).");
            return;
        }

        using SafeTensorsLoader weightLoader = new();
        weightLoader.Load(weightsPath);
        ArcFaceModel model = new();
        model.LoadWeights(weightLoader.GetAllTensors());

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(refPath);
        IReadOnlyDictionary<string, Tensor> reference = refLoader.GetAllTensors();

        using IBackend backend = new CpuBackend();
        string[] cases = reference.Keys.Where(k => k.StartsWith("input", StringComparison.Ordinal)).Order().ToArray();
        Assert.NotEmpty(cases);
        foreach (string inputKey in cases)
        {
            Tensor input = reference[inputKey];
            Tensor expected = reference[inputKey.Replace("input", "output")];
            Tensor actual = model.Forward(backend, input);
            try
            {
                float* ap = (float*)actual.DataPointer;
                float* ep = (float*)expected.DataPointer;
                double dot = 0, na = 0, ne = 0, maxAbs = 0;
                for (int i = 0; i < ArcFaceModel.EmbeddingDim; i++)
                {
                    dot += (double)ap[i] * ep[i];
                    na += (double)ap[i] * ap[i];
                    ne += (double)ep[i] * ep[i];
                    maxAbs = Math.Max(maxAbs, Math.Abs(ap[i] - ep[i]));
                }
                double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(ne));
                _output.WriteLine($"{inputKey}: cosine={cosine:F6}, maxAbs={maxAbs:E2}, |a|={Math.Sqrt(na):F4}, |e|={Math.Sqrt(ne):F4}");
                Assert.True(cosine > 0.999, $"{inputKey}: cosine {cosine:F6} <= 0.999 vs onnxruntime.");
            }
            finally
            {
                actual.Dispose();
            }
        }
    }
}
