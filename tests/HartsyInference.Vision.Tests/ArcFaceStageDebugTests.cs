using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Face;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Per-stage ArcFace parity debugging against the numpy stage dump (see the stage-dump snippet in the
/// worklog / convert_arcface_onnx.py). Gated on <c>ARCFACE_WEIGHTS</c> + <c>ARCFACE_STAGES</c> +
/// <c>ARCFACE_REF</c>. Reports the first diverging stage rather than a single end-to-end number.</summary>
public sealed class ArcFaceStageDebugTests
{
    private readonly ITestOutputHelper _output;

    public ArcFaceStageDebugTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public unsafe void StageOutputs_MatchNumpyReference()
    {
        string? weightsPath = Environment.GetEnvironmentVariable("ARCFACE_WEIGHTS");
        string? stagesPath = Environment.GetEnvironmentVariable("ARCFACE_STAGES");
        string? refPath = Environment.GetEnvironmentVariable("ARCFACE_REF");
        if (string.IsNullOrEmpty(weightsPath) || !File.Exists(weightsPath)
            || string.IsNullOrEmpty(stagesPath) || !File.Exists(stagesPath)
            || string.IsNullOrEmpty(refPath) || !File.Exists(refPath))
        {
            _output.WriteLine("SKIPPED: set ARCFACE_WEIGHTS + ARCFACE_STAGES + ARCFACE_REF.");
            return;
        }

        using SafeTensorsLoader weightLoader = new();
        weightLoader.Load(weightsPath);
        ArcFaceModel model = new();
        model.LoadWeights(weightLoader.GetAllTensors());

        using SafeTensorsLoader stageLoader = new();
        stageLoader.Load(stagesPath);
        IReadOnlyDictionary<string, Tensor> stages = stageLoader.GetAllTensors();

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(refPath);
        Tensor input = refLoader.GetAllTensors()["input_random"];

        using IBackend backend = new CpuBackend();
        List<string> failures = [];
        Tensor emb = model.Forward(backend, input, (name, t) =>
        {
            if (!stages.TryGetValue(name, out Tensor? expected)) return;
            (double maxAbs, double corr) = Compare(t, expected);
            _output.WriteLine($"{name}: shape={t.Shape} maxAbs={maxAbs:E2} corr={corr:F6}");
            if (maxAbs > 5e-3) failures.Add($"{name} maxAbs={maxAbs:E2}");
        });
        (double embMaxAbs, double embCorr) = Compare(emb, stages["emb"]);
        _output.WriteLine($"emb: maxAbs={embMaxAbs:E2} corr={embCorr:F6}");
        emb.Dispose();
        Assert.True(failures.Count == 0 && embMaxAbs < 5e-2,
            $"Diverging stages: {string.Join(", ", failures)}; emb maxAbs={embMaxAbs:E2}");
    }

    private static unsafe (double maxAbs, double corr) Compare(Tensor a, Tensor b)
    {
        if (a.ElementCount != b.ElementCount)
            return (double.PositiveInfinity, 0);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < a.ElementCount; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(ap[i] - bp[i]));
            dot += (double)ap[i] * bp[i];
            na += (double)ap[i] * ap[i];
            nb += (double)bp[i] * bp[i];
        }
        return (maxAbs, dot / Math.Max(Math.Sqrt(na) * Math.Sqrt(nb), 1e-12));
    }
}
