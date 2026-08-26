using HartsyInference.Core.Configuration;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>LLaVA-NeXT/1.6 anyres vision pipeline: tiles the image (<see cref="LlavaNextImagePreprocessor"/>), runs each tile through the SAME CLIP tower + projector validated for LLaVA-1.5 (<see cref="SiglipVlmEncoder"/>, wrapped by composition), then merges the per-tile embeddings (<see cref="LlavaNextFeatureMerger"/>) into one flat sequence.</summary>
/// <remarks><see cref="Encode"/> takes the RAW (un-resized, un-normalized, <c>[0,255]</c>) image as <c>[1, 3, origH, origW]</c> — unlike every other family's fixed-square contract via <see cref="VlmImagePreprocessor"/> — because the merge's unpad step needs the original aspect ratio, which a pre-resized square would destroy; callers must route LLaVA-NeXT through <see cref="LlavaNextImagePreprocessor.RawToNativeTensor"/> instead.</remarks>
public sealed unsafe class LlavaNextEncoder : IVlmImageEncoder
{
    private readonly SiglipVlmEncoder _tower;
    private readonly (int h, int w)[] _gridPinpoints;
    private readonly Tensor _imageNewline;
    private int _disposed;

    public int TokensPerImage { get; private set; }
    public int ImageSize => _tower.ImageSize;
    public float[] ImageMean => _tower.ImageMean;
    public float[] ImageStd => _tower.ImageStd;

    /// <summary>Vicuna-v1 prompt format, identical to LLaVA-1.5 — reuses <c>MultimodalGenerator</c>'s existing "llava" branch unchanged.</summary>
    public string Family => "llava";

    private LlavaNextEncoder(SiglipVlmEncoder tower, (int h, int w)[] gridPinpoints, Tensor imageNewline)
    {
        _tower = tower; _gridPinpoints = gridPinpoints; _imageNewline = imageNewline;
        TokensPerImage = tower.NumPatches + 1;   // nominal single-tile estimate; Encode() overwrites with the real per-call count.
    }

    public static LlavaNextEncoder Load(string mmprojPath)
    {
        SiglipVlmEncoder tower = SiglipVlmEncoder.Load(mmprojPath);
        try
        {
            int[]? flat = tower.Metadata.GetIntArray("clip.vision.image_grid_pinpoints");
            if (flat is null || flat.Length == 0 || flat.Length % 2 != 0)
                throw new InvalidOperationException("mmproj is missing a valid clip.vision.image_grid_pinpoints (LLaVA-NeXT requires anyres pinpoints).");
            (int h, int w)[] pinpoints = new (int, int)[flat.Length / 2];
            for (int i = 0; i < pinpoints.Length; i++) pinpoints[i] = (flat[2 * i], flat[2 * i + 1]);
            if (!tower.Weights.TryGetValue("model.image_newline", out Tensor? newline))
                throw new InvalidOperationException("mmproj is missing the model.image_newline tensor (LLaVA-NeXT merge separator).");
            return new LlavaNextEncoder(tower, pinpoints, newline);
        }
        catch { tower.Dispose(); throw; }
    }

    /// <summary>Encodes a native-resolution image (see <see cref="LlavaNextImagePreprocessor.RawToNativeTensor"/>) into <c>[1, totalTokens, projectionDim]</c> — token count varies per image (tile count × aspect ratio).</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        (Tensor[] tiles, int origH, int origW) = LlavaNextImagePreprocessor.Tile(
            pixelValues, _gridPinpoints, _tower.ImageSize, _tower.ImageMean, _tower.ImageStd);
        Dbg(backend, "pixel_values", pixelValues);

        Tensor[] tileEmbeds = new Tensor[tiles.Length];
        try
        {
            for (int i = 0; i < tiles.Length; i++)
            {
                tileEmbeds[i] = _tower.Encode(backend, tiles[i]);
                Dbg(backend, $"tile{i}_embeds", tileEmbeds[i]);
            }
            Tensor packed = LlavaNextFeatureMerger.PackImageFeatures(
                tileEmbeds, origH, origW, _gridPinpoints, _tower.ImageSize, _tower.PatchGrid, _tower.ProjectionDim, _imageNewline);
            TokensPerImage = (int)packed.Shape[1];
            Dbg(backend, "packed", packed);
            return packed;
        }
        finally
        {
            foreach (Tensor t in tiles) t.Dispose();
            foreach (Tensor? t in tileEmbeds) t?.Dispose();
        }
    }

    /// <summary>Same <c>HARTSY_VLM_DUMP</c> capture-hook convention as <see cref="SiglipVlmEncoder"/>'s internal <c>Dbg</c> — raw little-endian f32, file <c>cs_{tag}.f32</c> — so the same Python reference harness (<c>dump_llavanext_vision_ref.py</c>) can load and compare them.</summary>
    private static void Dbg(IBackend backend, string tag, Tensor t)
    {
        if (!EngineKnobs.VlmDebug.Value && EngineKnobs.VlmDump.Value is null) return;
        backend.Sync();
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount; double sum = 0, max = 0;
        for (long i = 0; i < n; i++) { float v = p[i]; sum += v; max = Math.Max(max, Math.Abs(v)); }
        if (EngineKnobs.VlmDebug.Value)
            Console.Error.WriteLine($"[vis-next:{tag}] mean={sum / n:F4} maxabs={max:F4} shape={t.Shape}");
        string? dir = EngineKnobs.VlmDump.Value;
        if (dir is not null)
        {
            Directory.CreateDirectory(dir);
            byte[] bytes = new byte[n * 4];
            fixed (byte* b = bytes) Buffer.MemoryCopy(p, b, bytes.Length, bytes.Length);
            File.WriteAllBytes(Path.Combine(dir, $"cs_{tag}.f32"), bytes);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _tower.Dispose();
    }
}
