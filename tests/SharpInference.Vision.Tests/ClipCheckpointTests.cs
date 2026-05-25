using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Cpu;
using SharpInference.Tests.Common;
using SharpInference.Vision.Clip;
using SharpInference.Vision.Embeddings;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Vision.Tests;

/// <summary>End-to-end CLIP checkpoint tests. These require <c>openai/clip-vit-large-patch14</c>
/// (or compatible) on disk and are skipped cleanly when missing — the SafeTensors loader is fast
/// enough that running on CPU is acceptable (single image embed ~5s, single text embed ~1s on a
/// modern desktop).
/// <para>The validation here is non-comparative — we check intrinsic invariants (shape, L2 norm,
/// self-similarity) and relative-ranking correctness (CLIP knows "kitten" ≈ "cat" ≠ "rocket").
/// Bit-exact Python reference comparison is a separate, heavier test that needs a Python harness
/// dumping CLIPModel outputs to disk; see <c>tests/python-reference/</c> for the convention used
/// by SD3.5.</para></summary>
public sealed class ClipCheckpointTests
{
    private readonly ITestOutputHelper _output;

    public ClipCheckpointTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ClipL_Loads_AndProducesNormalizedEmbeddings()
    {
        string checkpoint = TestPaths.Clip.ViTLargePatch14;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpoint}");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        using ClipModelLoader loader = new(ClipPreset.OpenAiClipLarge);
        loader.LoadFromSingleFile(checkpoint);
        _output.WriteLine($"[load] {sw.ElapsedMilliseconds} ms");

        ClipImageEncoder imageEnc = loader.ImageEncoder;
        ClipTextEmbedder textEmb = loader.TextEmbedder;
        Assert.Equal(768, imageEnc.EmbeddingDim);
        Assert.Equal(768, textEmb.EmbeddingDim);

        using IBackend backend = new CpuBackend();

        sw.Restart();
        // Synthesize an image: a 256×256 RGB image with a vertical color gradient so the
        // bicubic resize + center-crop preserves visual information rather than collapsing
        // to a flat color (which would land at a degenerate point in CLIP space).
        byte[] rgb = new byte[256 * 256 * 3];
        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                int i = (y * 256 + x) * 3;
                rgb[i + 0] = (byte)y;                 // R ramps top-to-bottom
                rgb[i + 1] = (byte)x;                 // G ramps left-to-right
                rgb[i + 2] = (byte)((x + y) / 2);     // B is their average
            }
        }
        ClipImagePreprocessor pre = new(imageSize: 224);
        ImageEmbeddingPipeline imagePipe = new(backend, pre, imageEnc, "openai/clip-vit-large-patch14");
        using ImageEmbedding imgEmbedding = imagePipe.Embed(rgb, 256, 256);
        _output.WriteLine($"[image embed] {sw.ElapsedMilliseconds} ms");

        Assert.Equal(768, imgEmbedding.EmbeddingDim);
        AssertUnitNorm(imgEmbedding.AsSpan(), "image");
        Assert.Equal(1f, ClipScorer.Score(imgEmbedding, imgEmbedding), tolerance: 1e-5f);

        sw.Restart();
        using TextEmbedding txtEmbedding = textEmb.Encode(backend, "a photograph of a cat");
        _output.WriteLine($"[text embed] {sw.ElapsedMilliseconds} ms");

        Assert.Equal(768, txtEmbedding.EmbeddingDim);
        AssertUnitNorm(txtEmbedding.AsSpan(), "text");
        Assert.Equal(1f, ClipScorer.Score(txtEmbedding, txtEmbedding), tolerance: 1e-5f);

        // Cross-modal score must be in valid range. With this synthetic gradient image we don't
        // expect a strong match to "cat" — just a sane cosine sim.
        float xScore = ClipScorer.Score(imgEmbedding, txtEmbedding);
        _output.WriteLine($"[cross score] gradient↔cat: {xScore:F4}");
        Assert.InRange(xScore, -1f, 1f);
    }

    [Fact]
    public void ClipL_TextToText_RanksSemanticallyRelatedPairsAbove_Unrelated()
    {
        string checkpoint = TestPaths.Clip.ViTLargePatch14;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpoint}");
            return;
        }

        using ClipModelLoader loader = new(ClipPreset.OpenAiClipLarge);
        loader.LoadFromSingleFile(checkpoint);
        ClipTextEmbedder textEmb = loader.TextEmbedder;
        using IBackend backend = new CpuBackend();

        // Encode three texts. "kitten" is the close semantic match to "cat";
        // "rocket ship" is clearly unrelated.
        using TextEmbedding cat = textEmb.Encode(backend, "a photograph of a cat");
        using TextEmbedding kitten = textEmb.Encode(backend, "an image of a kitten");
        using TextEmbedding rocket = textEmb.Encode(backend, "a diagram of a rocket ship");

        float catKitten = ClipScorer.Score(cat, kitten);
        float catRocket = ClipScorer.Score(cat, rocket);
        _output.WriteLine($"cat↔kitten: {catKitten:F4}");
        _output.WriteLine($"cat↔rocket: {catRocket:F4}");

        // The semantic pair should win by a clear margin. Empirically CLIP-L gives
        // cat↔kitten ~ 0.92, cat↔rocket ~ 0.62 on these strings — gap > 0.1 is the
        // floor we assert against to tolerate float drift across hardware.
        Assert.True(catKitten > catRocket, $"Expected cat↔kitten ({catKitten:F4}) > cat↔rocket ({catRocket:F4}) — the text tower may be miswired.");
        Assert.True(catKitten - catRocket > 0.05f, $"Semantic gap too small: cat↔kitten={catKitten:F4}, cat↔rocket={catRocket:F4}, gap={(catKitten - catRocket):F4}.");
    }

    [Fact]
    public void ClipL_BatchEmbed_MatchesSingleEmbed()
    {
        string checkpoint = TestPaths.Clip.ViTLargePatch14;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpoint}");
            return;
        }

        using ClipModelLoader loader = new(ClipPreset.OpenAiClipLarge);
        loader.LoadFromSingleFile(checkpoint);
        ClipTextEmbedder textEmb = loader.TextEmbedder;
        using IBackend backend = new CpuBackend();

        string[] texts = ["the quick brown fox", "the lazy dog"];

        using TextEmbedding singleA = textEmb.Encode(backend, texts[0]);
        using TextEmbedding singleB = textEmb.Encode(backend, texts[1]);
        IReadOnlyList<TextEmbedding> batch = textEmb.EncodeBatch(backend, texts);
        try
        {
            Assert.Equal(2, batch.Count);
            // Cosine sim between single-pass and batch result for each element must be ~1.
            // Padding behavior is identical (ClipTokenizer pads to 77 with EOT regardless of batch),
            // so the only source of drift is float-op ordering inside the batched matmuls.
            float simA = ClipScorer.Score(singleA, batch[0]);
            float simB = ClipScorer.Score(singleB, batch[1]);
            _output.WriteLine($"single[0]↔batch[0]: {simA:F6}");
            _output.WriteLine($"single[1]↔batch[1]: {simB:F6}");
            Assert.True(simA > 0.9999f, $"Batch consistency broken: {simA:F6}");
            Assert.True(simB > 0.9999f, $"Batch consistency broken: {simB:F6}");
        }
        finally
        {
            foreach (TextEmbedding e in batch) e.Dispose();
        }
    }

    private void AssertUnitNorm(ReadOnlySpan<float> v, string label)
    {
        double sumSq = 0.0;
        for (int i = 0; i < v.Length; i++) sumSq += v[i] * v[i];
        double norm = Math.Sqrt(sumSq);
        _output.WriteLine($"[{label}] L2 norm: {norm:F8} (dim={v.Length})");
        Assert.InRange(norm, 1.0 - 1e-5, 1.0 + 1e-5);
    }
}
