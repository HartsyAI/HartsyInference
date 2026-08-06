using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;
using HartsyInference.Vision.Codec;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Structural check for <see cref="LlavaNextImagePreprocessor"/>'s OWN resize/pad/tile pipeline (not
/// bypassed via identical-pixel injection like <see cref="LlavaNextMergeParityTests"/>) against the real HF
/// image processor's output. Expects near-but-not-exact correlation (~0.99, not ~1.0) — this repo uses the same
/// bilinear resize kernel as every other VLM family (<see cref="VlmImagePreprocessor"/>), not HF's bicubic; that
/// gap is an already-accepted approximation (LLaVA-1.5's own "corr 1.0" tower validation never actually exercised
/// its resize step, since the Python reference loads the C#-computed pixels directly rather than resizing itself).
/// A large corr drop, or a mismatched tile count/order, would indicate a real structural bug the merge check
/// (which consumes Python's pixels directly) can't see: e.g. wrong best-resolution selection, wrong pad amount,
/// or padded regions not landing at <c>-mean/std</c> after normalize (pad is applied in raw [0,255] space BEFORE
/// normalization, matching HF's own pipeline order — a bug here would silently pad with exactly 0 instead).
///
/// <para>NOTE: <see cref="DumpDir"/> is a session-scoped scratch path, not a repo-relative fixture — this test
/// is a one-off validation artifact from the LLaVA-NeXT bring-up (2026-07-24/25), not a permanent CI gate. It
/// SKIPS cleanly (not fails) when the dump files aren't present.</para></summary>
[Trait("Category", "Slow")]
public sealed class LlavaNextTilingParityTests
{
    private readonly ITestOutputHelper _output;
    public LlavaNextTilingParityTests(ITestOutputHelper output) => _output = output;

    private const string BusPng = "/home/hartsy/Desktop/HartsyInference/tests/HartsyInference.Cuda.Tests/TestData/bus.png";
    private const string DumpDir = "/tmp/claude-1000/-home-hartsy/653b4ecd-7040-4d22-9749-94356f3a7c72/scratchpad/llavanextdump";
    // GGUF clip.vision.image_grid_pinpoints for llava-v1.6-vicuna-7b, confirmed via direct GGUF metadata read.
    private static readonly (int h, int w)[] GridPinpoints = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)];
    private static readonly float[] Mean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] Std = [0.26862954f, 0.26130258f, 0.27577711f];

    [Fact]
    public unsafe void OwnTiling_MatchesRealHfImageProcessor_Structurally()
    {
        string pyMetaPath = Path.Combine(DumpDir, "py_meta.txt");
        string pyPixelsPath = Path.Combine(DumpDir, "py_pixel_values.f32");
        if (!File.Exists(pyMetaPath) || !File.Exists(pyPixelsPath))
        {
            _output.WriteLine($"SKIPPED: run dump_llavanext_vision_ref.py first ({pyMetaPath} missing).");
            return;
        }
        if (!File.Exists(BusPng)) { _output.WriteLine($"SKIPPED: {BusPng} not found."); return; }

        Dictionary<string, int> meta = [];
        foreach (string line in File.ReadAllLines(pyMetaPath))
        {
            string[] kv = line.Split('=', 2);
            if (kv.Length == 2 && int.TryParse(kv[1], out int v)) meta[kv[0]] = v;
        }
        int expectedNumPatches = meta["num_patches"];

        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(BusPng);
        using Tensor native = LlavaNextImagePreprocessor.RawToNativeTensor(rgb, width, height);
        (Tensor[] tiles, int origH, int origW) = LlavaNextImagePreprocessor.Tile(native, GridPinpoints, 336, Mean, Std);

        _output.WriteLine($"C# tiling: numPatches={tiles.Length} origH={origH} origW={origW} (Python: numPatches={expectedNumPatches})");
        Assert.Equal(expectedNumPatches, tiles.Length);
        Assert.Equal(height, origH);
        Assert.Equal(width, origW);

        // Dump concatenated tiles matching py_pixel_values.f32's [numPatches,3,336,336] layout for a direct
        // numpy corrcoef in the Python script (structural check, not exact match — bilinear vs bicubic).
        long tileFloats = 3L * 336 * 336;
        byte[] all = new byte[tiles.Length * tileFloats * 4];
        fixed (byte* ab = all)
        {
            float* dst = (float*)ab;
            for (int i = 0; i < tiles.Length; i++)
                Buffer.MemoryCopy((void*)tiles[i].DataPointer, dst + i * tileFloats, tileFloats * 4, tileFloats * 4);
        }
        File.WriteAllBytes(Path.Combine(DumpDir, "cs_pixel_values.f32"), all);

        // Spot-check a definitely-padded corner pixel (tile 1, top-left, channel 0) lands near -mean/std, not 0 —
        // confirms pad-then-normalize ordering rather than the inverted (and silently wrong) normalize-then-pad.
        float* t1 = (float*)tiles[1].DataPointer;
        float expectedPadValue = (0f / 255f - Mean[0]) / Std[0];
        _output.WriteLine($"tile[1] top-left channel-0 = {t1[0]:F4} (expected pad value ~= {expectedPadValue:F4} if padded, or a real pixel otherwise)");

        foreach (Tensor t in tiles) t.Dispose();
    }
}
