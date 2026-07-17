using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight parity for the IP-Adapter FaceID-Plus / Plus-v2 image projection
/// (<see cref="IpAdapterFaceIdPlusProjection"/>, h94's <c>ProjPlusModel</c>) against the official tencent-ailab
/// module definitions run in torch float32. Golden refs come from
/// <c>tests/python-reference/ipadapter_faceid_plus_ref.py</c> (fixed seeded ArcFace + CLIP inputs; outputs for
/// shortcut off, shortcut scale 1.0, shortcut scale 0.6). Skips cleanly when the checkpoint or ref file is
/// missing; override locations with <c>IPA_*_FACEID_PLUS*_PATH</c> / <c>FACEID_PLUS_REF_DIR</c>.</summary>
public sealed class IpAdapterFaceIdPlusParityTests
{
    private const double MinCorrelation = 0.9999;

    private readonly ITestOutputHelper _output;

    public IpAdapterFaceIdPlusParityTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [Trait("Category", "Integration")]
    [InlineData("sd15-plus")]
    [InlineData("sd15-plusv2")]
    [InlineData("sdxl-plusv2")]
    public void ProjPlusModel_MatchesTorchReference(string which)
    {
        (string binPath, bool expectV2, IpAdapterBaseModel expectedBase) = which switch
        {
            "sd15-plus" => (TestPaths.IpAdapter.Sd15FaceIdPlus, false, IpAdapterBaseModel.Sd15),
            "sd15-plusv2" => (TestPaths.IpAdapter.Sd15FaceIdPlusV2, true, IpAdapterBaseModel.Sd15),
            "sdxl-plusv2" => (TestPaths.IpAdapter.SdxlFaceIdPlusV2, true, IpAdapterBaseModel.Sdxl),
            _ => throw new ArgumentOutOfRangeException(nameof(which)),
        };
        string refPath = Path.Combine(TestPaths.IpAdapter.FaceIdPlusRefDir,
            Path.GetFileNameWithoutExtension(binPath) + "_projref.safetensors");
        if (!File.Exists(binPath)) { _output.WriteLine($"SKIPPED: checkpoint not found: {binPath}"); return; }
        if (!File.Exists(refPath)) { _output.WriteLine($"SKIPPED: golden ref not found: {refPath} (run ipadapter_faceid_plus_ref.py)"); return; }

        using IpAdapterFile file = IpAdapterLoader.Load(binPath);
        Assert.Equal(expectedBase, file.BaseModel);
        Assert.True(file.Config.IsFaceId);
        Assert.True(file.Config.IsPlus);
        Assert.Equal(expectV2, file.Config.IsFaceIdV2);
        Assert.Equal(512, file.Config.ImageEmbeddingDim);
        Assert.Equal(1280, file.Config.ClipEmbeddingDim);
        Assert.Equal(4, file.Config.NumImageTokens);

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(refPath);
        IReadOnlyDictionary<string, Tensor> reference = refLoader.GetAllTensors();

        // The projection module is exercised directly (both shortcut modes) so one checkpoint validates the
        // v1 AND v2 math; the IpAdapter-level dispatch is covered at the end via the loaded config.
        IpAdapterFaceIdPlusProjection plain = new(file.Config.CrossAttentionDim, file.Config.NumImageTokens,
            file.Config.ClipEmbeddingDim, useShortcut: false);
        plain.LoadWeights(file.Weights);
        IpAdapterFaceIdPlusProjection shortcut = new(file.Config.CrossAttentionDim, file.Config.NumImageTokens,
            file.Config.ClipEmbeddingDim, useShortcut: true);
        shortcut.LoadWeights(file.Weights);

        using IBackend backend = new CpuBackend();
        for (int i = 0; ; i++)
        {
            if (!reference.TryGetValue($"input_id_{i}", out Tensor? idEmbeds)) { Assert.True(i > 0, "ref file has no cases"); break; }
            Tensor clipEmbeds = reference[$"input_clip_{i}"];

            using (Tensor actual = plain.Forward(backend, idEmbeds, clipEmbeds))
            {
                AssertMatches($"case{i} plain", actual, reference[$"output_plain_{i}"]);
            }
            using (Tensor actual = shortcut.Forward(backend, idEmbeds, clipEmbeds, shortcutScale: 1.0f))
            {
                AssertMatches($"case{i} shortcut s=1.0", actual, reference[$"output_shortcut10_{i}"]);
            }
            using (Tensor actual = shortcut.Forward(backend, idEmbeds, clipEmbeds, shortcutScale: 0.6f))
            {
                AssertMatches($"case{i} shortcut s=0.6", actual, reference[$"output_shortcut06_{i}"]);
            }

            // IpAdapter-level dispatch: config-selected shortcut mode + the two-input overload.
            using IpAdapter adapter = new(file.Config);
            adapter.LoadWeights(file.Weights);
            Assert.IsType<IpAdapterFaceIdPlusProjection>(adapter.ImageProjection);
            using (Tensor viaAdapter = adapter.ProjectImage(backend, idEmbeds, clipEmbeds, shortcutScale: 1.0f))
            {
                string expectedKey = expectV2 ? $"output_shortcut10_{i}" : $"output_plain_{i}";
                AssertMatches($"case{i} via IpAdapter ({(expectV2 ? "v2" : "v1")})", viaAdapter, reference[expectedKey]);
            }
            Assert.Throws<InvalidOperationException>(() => adapter.ProjectImage(backend, idEmbeds));
        }
    }

    private unsafe void AssertMatches(string tag, Tensor actual, Tensor expected)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        long count = actual.ElementCount;
        float* ap = (float*)actual.DataPointer;
        float* ep = (float*)expected.DataPointer;
        double sumA = 0, sumE = 0;
        for (long i = 0; i < count; i++) { sumA += ap[i]; sumE += ep[i]; }
        double meanA = sumA / count, meanE = sumE / count;
        double cov = 0, varA = 0, varE = 0, maxAbs = 0;
        for (long i = 0; i < count; i++)
        {
            double da = ap[i] - meanA, de = ep[i] - meanE;
            cov += da * de;
            varA += da * da;
            varE += de * de;
            maxAbs = Math.Max(maxAbs, Math.Abs(ap[i] - ep[i]));
        }
        double corr = cov / Math.Sqrt(varA * varE);
        _output.WriteLine($"{tag}: corr={corr:F6}, maxAbs={maxAbs:E2}, n={count}");
        Assert.True(corr >= MinCorrelation, $"{tag}: corr {corr:F6} < {MinCorrelation} (maxAbs={maxAbs:E2}).");
    }
}
