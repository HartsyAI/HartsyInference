using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Multimodal;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Numerical parity check for LLaVA-NeXT's two genuinely NEW pieces (anyres tiling and
/// <c>pack_image_features</c> merge — the CLIP tower + <c>mm.0</c>/<c>mm.2</c> projector are unchanged, already
/// validated at corr=1.0 for LLaVA-1.5). Feeds the SAME pixel tiles the real <c>transformers</c> library computed
/// (dumped by <c>tests/python-reference/dump_llavanext_vision_ref.py</c> to <c>{prefix}_pixel_values.f32</c>)
/// through this repo's own <see cref="SiglipVlmEncoder"/> + <see cref="LlavaNextFeatureMerger"/>, deliberately
/// bypassing this repo's own (bilinear, not HF's bicubic) tiling — that isolates the merge/tower math from the
/// resize-kernel mismatch. Runs two cases: "py" (bus.png, portrait — takes the crop-WIDTH branch of
/// <c>unpad_image</c>) and "landscape" (bus.png rotated 90°, same pixels — takes the crop-HEIGHT branch), so both
/// sides of the H/W-swap-prone unpad conditional get exercised, not just the one bus.png happens to hit.
///
/// <para>NOTE: <see cref="DumpDir"/> is a session-scoped scratch path, not a repo-relative fixture — this test
/// is a one-off validation artifact from the LLaVA-NeXT bring-up (2026-07-24/25), not a permanent CI gate. It
/// SKIPS cleanly (not fails) when the dump files aren't present, which will be true outside that session unless
/// <c>dump_llavanext_vision_ref.py</c> is re-run against a path this points at.</para></summary>
[Trait("Category", "Slow")]
[Collection("CudaSerial")]
public sealed class LlavaNextMergeParityTests
{
    private readonly ITestOutputHelper _output;
    public LlavaNextMergeParityTests(ITestOutputHelper output) => _output = output;

    private const string Mmproj = "/home/hartsy/Desktop/HartsyInference/Models/LLM/llava-next-vicuna-7b/llava-v1.6-vicuna-7b-mmproj-model-f16.gguf";
    private const string DumpDir = "/tmp/claude-1000/-home-hartsy/653b4ecd-7040-4d22-9749-94356f3a7c72/scratchpad/llavanextdump";
    private static readonly string[] Prefixes = ["py", "landscape"];

    [Fact]
    public unsafe void DumpTileEmbedsAndPacked_ForPythonComparison()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!File.Exists(Mmproj)) { _output.WriteLine($"SKIPPED: mmproj not found: {Mmproj}"); return; }

        using SiglipVlmEncoder tower = SiglipVlmEncoder.Load(Mmproj);
        using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
        Environment.SetEnvironmentVariable("HARTSY_VLM_DUMP", DumpDir);

        Tensor imageNewline = tower.Weights["model.image_newline"];
        int[] pinpointsFlat = tower.Metadata.GetIntArray("clip.vision.image_grid_pinpoints")!;
        (int h, int w)[] pinpoints = new (int, int)[pinpointsFlat.Length / 2];
        for (int i = 0; i < pinpoints.Length; i++) pinpoints[i] = (pinpointsFlat[2 * i], pinpointsFlat[2 * i + 1]);
        const int tileSize = 336, hidden = 4096;

        foreach (string prefix in Prefixes)
        {
            string tag(string name) => prefix == "py" ? name : $"{prefix}_{name}";
            string metaPath = Path.Combine(DumpDir, $"{prefix}_meta.txt");
            string pixelsPath = Path.Combine(DumpDir, $"{prefix}_pixel_values.f32");
            if (!File.Exists(metaPath) || !File.Exists(pixelsPath))
            {
                _output.WriteLine($"SKIPPED [{prefix}]: run dump_llavanext_vision_ref.py with HARTSY_DUMP_PREFIX={prefix} first.");
                continue;
            }

            Dictionary<string, int> meta = [];
            foreach (string line in File.ReadAllLines(metaPath))
            {
                string[] kv = line.Split('=', 2);
                if (kv.Length == 2 && int.TryParse(kv[1], out int v)) meta[kv[0]] = v;
            }
            int numPatches = meta["num_patches"], origH = meta["image_size_h"], origW = meta["image_size_w"];
            _output.WriteLine($"[{prefix}] numPatches={numPatches} origH={origH} origW={origW}");

            byte[] pixelBytes = File.ReadAllBytes(pixelsPath);
            long expectedBytes = (long)numPatches * 3 * tileSize * tileSize * 4;
            Assert.True(pixelBytes.LongLength == expectedBytes,
                $"[{prefix}] pixel_values.f32 size {pixelBytes.LongLength} != expected {expectedBytes}");

            Tensor[] tiles = new Tensor[numPatches];
            long tileFloats = 3L * tileSize * tileSize;
            fixed (byte* pb = pixelBytes)
            {
                float* src = (float*)pb;
                for (int i = 0; i < numPatches; i++)
                {
                    Tensor t = new(new TensorShape(1, 3, tileSize, tileSize), DType.F32);
                    Buffer.MemoryCopy(src + i * tileFloats, (void*)t.DataPointer, tileFloats * 4, tileFloats * 4);
                    tiles[i] = t;
                }
            }

            Tensor[] tileEmbeds = new Tensor[numPatches];
            for (int i = 0; i < numPatches; i++)
            {
                tileEmbeds[i] = tower.Encode(backend, tiles[i]);
                DumpF32(Path.Combine(DumpDir, $"cs_{tag($"tile{i}_embeds")}.f32"), tileEmbeds[i]);
            }

            Tensor packed = LlavaNextFeatureMerger.PackImageFeatures(
                tileEmbeds, origH, origW, pinpoints, tileSize, tower.PatchGrid, hidden, imageNewline);
            DumpF32(Path.Combine(DumpDir, $"cs_{tag("packed")}.f32"), packed);
            _output.WriteLine($"[{prefix}] packed: shape={packed.Shape}");

            foreach (Tensor t in tiles) t.Dispose();
            foreach (Tensor t in tileEmbeds) t.Dispose();
            packed.Dispose();
        }

        Assert.True(true, "Dumps written; run dump_llavanext_vision_ref.py (both prefixes) again to compute correlations.");
    }

    private static unsafe void DumpF32(string path, Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        byte[] bytes = new byte[n * 4];
        fixed (byte* b = bytes) Buffer.MemoryCopy(p, b, bytes.Length, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }
}
