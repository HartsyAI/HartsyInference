using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Siglip;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight numeric A/B of the FLUX.1 Redux conditioning path (SigLIP-so400m/14-384
/// <c>last_hidden_state</c> → <c>redux_down(silu(redux_up(x)))</c>) against the diffusers/BFL reference.
/// <c>tests/python-reference/dump_redux_ab.py</c> dumps the shared preprocessed pixels plus the reference
/// hidden states and projected tokens from the SAME local checkpoints; this test replays the pixels through
/// <see cref="SiglipVisionEncoder.EncodeHiddenStates"/> + <see cref="ReduxImageEncoder.Project"/> on the CPU
/// backend and compares element-wise. Skips cleanly when the checkpoints or the Python dump are absent.</summary>
public sealed unsafe class ReduxParityTests
{
    private readonly ITestOutputHelper _output;
    public ReduxParityTests(ITestOutputHelper output) => _output = output;

    private static string ReferenceDir =>
        Path.Combine(RepoRoot.Path, "tests", "python-reference", "redux_reference_tensors");

    [Fact]
    [Trait("Category", "Integration")]
    public void ReduxTokens_MatchDiffusersReference()
    {
        string sigclipPath = TestPaths.Redux.SigclipVision384;
        string reduxPath = TestPaths.Redux.StyleModel;
        string pixelsPath = Path.Combine(ReferenceDir, "pixel_values.bin");
        if (!File.Exists(sigclipPath) || !File.Exists(reduxPath))
        {
            _output.WriteLine($"SKIPPED: checkpoints not found ({sigclipPath}, {reduxPath}).");
            return;
        }
        if (!File.Exists(pixelsPath))
        {
            _output.WriteLine($"SKIPPED: Python reference not found in {ReferenceDir} — run dump_redux_ab.py first.");
            return;
        }

        SiglipPreset preset = SiglipPreset.So400m14_384;
        using SafeTensorsLoader visionLoader = new();
        visionLoader.Load(sigclipPath);
        SiglipVisionEncoder siglip = new(preset);
        siglip.LoadWeights(visionLoader.GetAllTensors());

        using SafeTensorsLoader styleLoader = new();
        styleLoader.Load(reduxPath);
        using ReduxImageEncoder projector = new();
        projector.LoadWeights(styleLoader.GetAllTensors());

        using Tensor pixels = LoadF32(pixelsPath, new TensorShape(1, 3, preset.ImageSize, preset.ImageSize));
        using IBackend backend = new CpuBackend();

        using Tensor hidden = siglip.EncodeHiddenStates(backend, pixels);
        Assert.Equal(new TensorShape(1, preset.NumPatches, preset.HiddenSize), hidden.Shape);
        (double hidCorr, double hidMaxAbs, double hidAvg) = Compare(hidden, Path.Combine(ReferenceDir, "ref_hidden.bin"));
        _output.WriteLine($"hidden states: corr {hidCorr:F6}, max abs diff {hidMaxAbs:E3}, avg abs diff {hidAvg:E3}");

        using Tensor tokens = projector.Project(backend, hidden);
        Assert.Equal(new TensorShape(1, preset.NumPatches, ReduxImageEncoder.TxtInFeatures), tokens.Shape);
        (double tokCorr, double tokMaxAbs, double tokAvg) = Compare(tokens, Path.Combine(ReferenceDir, "ref_tokens.bin"));
        _output.WriteLine($"redux tokens: corr {tokCorr:F6}, max abs diff {tokMaxAbs:E3}, avg abs diff {tokAvg:E3}");

        string dumpDir = Path.Combine(TestPaths.OutputDir, "fluxdepth_redux_ab", "redux_csharp");
        Directory.CreateDirectory(dumpDir);
        WriteF32(Path.Combine(dumpDir, "engine_tokens.bin"), tokens);

        Assert.True(hidCorr > 0.999, $"SigLIP hidden states diverge from reference: corr {hidCorr:F6} < 0.999.");
        Assert.True(tokCorr > 0.999, $"Redux tokens diverge from reference: corr {tokCorr:F6} < 0.999.");
    }

    /// <summary>End-to-end preprocessing sensitivity: same source image through the ENGINE bilinear
    /// stretch preprocessor instead of the shared HF-preprocessed pixels. Documents how much the resize
    /// kernel difference (bilinear vs PIL bicubic-antialias) moves the final conditioning tokens; the
    /// meta.json written next to the reference records the corr for the ledger. Informational — the
    /// hard ≥0.999 gate lives in <see cref="ReduxTokens_MatchDiffusersReference"/>.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void ReduxTokens_EnginePreprocess_CloseToReference()
    {
        string sigclipPath = TestPaths.Redux.SigclipVision384;
        string reduxPath = TestPaths.Redux.StyleModel;
        string metaPath = Path.Combine(ReferenceDir, "meta.json");
        if (!File.Exists(sigclipPath) || !File.Exists(reduxPath) || !File.Exists(metaPath))
        {
            _output.WriteLine("SKIPPED: checkpoints or Python reference missing.");
            return;
        }
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(metaPath));
        string imagePath = meta.RootElement.GetProperty("image").GetString()!;
        if (!File.Exists(imagePath))
        {
            _output.WriteLine($"SKIPPED: source image not found: {imagePath}");
            return;
        }

        SiglipPreset preset = SiglipPreset.So400m14_384;
        using SafeTensorsLoader visionLoader = new();
        visionLoader.Load(sigclipPath);
        SiglipVisionEncoder siglip = new(preset);
        siglip.LoadWeights(visionLoader.GetAllTensors());
        using SafeTensorsLoader styleLoader = new();
        styleLoader.Load(reduxPath);
        using ReduxImageEncoder projector = new();
        projector.LoadWeights(styleLoader.GetAllTensors());

        (byte[] rgb, int w, int h) = Vision.Codec.PngDecoder.DecodeFromFile(imagePath);
        SiglipImagePreprocessor pre = new(preset.ImageSize);
        using Tensor pixels = pre.Preprocess(rgb, w, h);

        using IBackend backend = new CpuBackend();
        using Tensor hidden = siglip.EncodeHiddenStates(backend, pixels);
        using Tensor tokens = projector.Project(backend, hidden);
        (double corr, double maxAbs, double avg) = Compare(tokens, Path.Combine(ReferenceDir, "ref_tokens.bin"));
        _output.WriteLine($"engine-preprocessed tokens vs reference: corr {corr:F6}, max abs {maxAbs:E3}, avg abs {avg:E3}");

        string dumpDir = Path.Combine(TestPaths.OutputDir, "fluxdepth_redux_ab", "redux_csharp");
        Directory.CreateDirectory(dumpDir);
        WriteF32(Path.Combine(dumpDir, "engine_tokens_enginepre.bin"), tokens);

        // The engine preprocessor is HF-parity antialiased bicubic (a = −0.5); the residual difference is
        // float rounding. The plain-bilinear kernel it replaced scored corr 0.9404 here — a regression to
        // that level means the resize kernel (or normalize/channel order) broke.
        Assert.True(corr > 0.999, $"Engine-preprocessed Redux tokens too far from reference: corr {corr:F6} <= 0.999.");
    }

    private static Tensor LoadF32(string path, TensorShape shape)
    {
        byte[] raw = File.ReadAllBytes(path);
        Assert.Equal(shape.ElementCount * 4, raw.Length);
        Tensor t = new(shape, DType.F32);
        raw.AsSpan().CopyTo(new Span<byte>((void*)t.DataPointer, raw.Length));
        return t;
    }

    private static void WriteF32(string path, Tensor t)
    {
        ReadOnlySpan<byte> bytes = new((void*)t.DataPointer, (int)(t.ElementCount * 4));
        File.WriteAllBytes(path, bytes.ToArray());
    }

    private static (double Corr, double MaxAbs, double AvgAbs) Compare(Tensor actual, string refPath)
    {
        byte[] raw = File.ReadAllBytes(refPath);
        Assert.Equal(actual.ElementCount * 4, raw.Length);
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        double sumA = 0, sumR = 0, sumAA = 0, sumRR = 0, sumAR = 0, sumAbs = 0, maxAbs = 0;
        int n = a.Length;
        for (int i = 0; i < n; i++)
        {
            float r = BitConverter.ToSingle(raw, i * 4);
            double d = Math.Abs(a[i] - r);
            sumAbs += d;
            if (d > maxAbs) maxAbs = d;
            sumA += a[i]; sumR += r;
            sumAA += (double)a[i] * a[i]; sumRR += (double)r * r; sumAR += (double)a[i] * r;
        }
        double cov = sumAR / n - sumA / n * (sumR / n);
        double varA = sumAA / n - sumA / n * (sumA / n);
        double varR = sumRR / n - sumR / n * (sumR / n);
        return (cov / Math.Sqrt(Math.Max(varA * varR, 1e-30)), maxAbs, sumAbs / n);
    }
}
