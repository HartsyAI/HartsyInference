using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Siglip;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>SigLIP vision-encoder checkpoint test against the canonical Python reference.
/// Loads <c>google/siglip-base-patch16-224</c> (812 MB), preprocesses bus.png, runs the C#
/// vision encoder, and compares element-wise with a dumped Python embedding for the same image.
/// <para>Cosine similarity > 0.999 against the Python reference is the gate — the absolute
/// magnitudes of SigLIP embeddings vary across runs due to float-precision accumulation in
/// transformer attention, but the embedding direction should be essentially identical.</para></summary>
public sealed class SiglipCheckpointTest
{
    private readonly ITestOutputHelper _output;
    public SiglipCheckpointTest(ITestOutputHelper output) => _output = output;

    private static string TestImagePath => Path.Combine(AppContext.BaseDirectory, "TestData", "bus.png");

    [Fact]
    public void SiglipBase_BusImage_EmbeddingMatchesPython()
    {
        string checkpoint = TestPaths.Siglip.Base16_224;
        string refFile = "/tmp/siglip_bus_image_emb.bin";
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: SigLIP checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(TestImagePath))
        {
            _output.WriteLine($"SKIPPED: bus.png test image not found.");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        _output.WriteLine($"[load] {sw.ElapsedMilliseconds} ms — {weights.Count} tensors");

        SiglipPreset preset = SiglipPreset.Base16_224;
        SiglipVisionEncoder encoder = new(preset);
        encoder.LoadWeights(weights, prefix: "vision_model");
        SiglipImagePreprocessor pre = new(preset.ImageSize);

        (byte[] rgb, int w, int h) = PngDecoder.DecodeFromFile(TestImagePath);
        _output.WriteLine($"[preprocess] input {w}x{h} → {preset.ImageSize}x{preset.ImageSize}");

        using IBackend backend = new CpuBackend();
        sw.Restart();
        using Tensor pixels = pre.Preprocess(rgb, w, h);
        using Tensor embedding = encoder.Encode(backend, pixels);
        _output.WriteLine($"[encode] {sw.ElapsedMilliseconds} ms — output shape {embedding.Shape}");

        Assert.Equal(new TensorShape(1, preset.EmbeddingDim), embedding.Shape);
        ReadOnlySpan<float> cs = embedding.AsReadOnlySpan<float>();

        // Stats sanity.
        double sum = 0, sumSq = 0;
        for (int i = 0; i < cs.Length; i++) { sum += cs[i]; sumSq += cs[i] * cs[i]; }
        double mean = sum / cs.Length;
        double std = Math.Sqrt(Math.Max(0, sumSq / cs.Length - mean * mean));
        double l2 = Math.Sqrt(sumSq);
        _output.WriteLine($"C# embedding: mean={mean:F4}, std={std:F4}, L2 norm={l2:F4}, dim={cs.Length}");

        // Compare against Python reference if available.
        if (!File.Exists(refFile))
        {
            _output.WriteLine($"NOTE: Python reference {refFile} missing — skipping element-wise comparison.");
            return;
        }
        byte[] raw = File.ReadAllBytes(refFile);
        Assert.Equal(cs.Length, raw.Length / 4);

        double dot = 0, refSumSq = 0;
        double maxAbsDiff = 0;
        for (int i = 0; i < cs.Length; i++)
        {
            float refVal = BitConverter.ToSingle(raw, i * 4);
            dot += cs[i] * refVal;
            refSumSq += refVal * refVal;
            double d = Math.Abs(cs[i] - refVal);
            if (d > maxAbsDiff) maxAbsDiff = d;
        }
        double refL2 = Math.Sqrt(refSumSq);
        double cosine = dot / (l2 * refL2);
        _output.WriteLine($"vs Python: cosine sim = {cosine:F6}, max abs diff = {maxAbsDiff:F6}, ref L2 = {refL2:F4}");

        // Cosine similarity ~ 1.0 → embedding direction matches. Tolerance 1e-3 accommodates
        // float-precision accumulation across 12 transformer layers + attention pooling.
        Assert.True(cosine > 0.999, $"SigLIP image embedding diverges from Python reference: cosine sim {cosine:F6} < 0.999.");
    }
}
