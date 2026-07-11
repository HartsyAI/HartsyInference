using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Zeta-Chroma transformer (<c>lodestones/Zeta-Chroma</c> pixel-proto) — an UNMODIFIED Z-Image S3-DiT
/// backbone configured for pixel IO (<c>InChannels = 3</c>, patch inferred from the <c>x_embedder</c> in-dim,
/// reported 3072 → 32) plus the DeCo-style <see cref="ZetaChromaDecoderHead"/> replacing <c>final_layer</c>.
/// Predicts <b>x0</b> [B, 3, H, W]: the decoder output is unpatchified and negated (the NextDiT lineage negates the
/// raw network output — ComfyUI's <c>NextDiTPixelSpace._forward</c> returns <c>-img_out</c> and converts via
/// <c>(x − out) / t</c>, which equals <c>v = (x_t − x0)/t</c> with <c>x0 = −img_out</c>). The pipeline does the
/// velocity conversion via <c>HartsyInference.Diffusion.Utilities.X0Prediction.ToVelocity</c>.
/// Everything is validation-gated — the model is mid-pretraining and no checkpoint has been diffed locally.</summary>
public sealed class ZetaChromaTransformer : IDisposable
{
    private readonly ZetaChromaConfig _config;
    private readonly ZImageTransformer _backbone;
    private readonly ZetaChromaDecoderHead _decoder;
    private int _disposed;

    // Binary stage dumps for the torch oracle (HARTSY_ZETA_DUMP=<dir>): raw F32 .bin per stage on the
    // FIRST forward only (step-0 cond pass — the pipeline runs cond before uncond), plus a shapes manifest.
    private static readonly string? ZetaDumpDir = Environment.GetEnvironmentVariable("HARTSY_ZETA_DUMP");
    private static int _dumpForwardIndex = -1;

    /// <summary>True while the step-0 cond forward is being dumped (HARTSY_ZETA_DUMP set).</summary>
    internal static bool DumpActive => ZetaDumpDir is not null && _dumpForwardIndex == 0;

    /// <summary>Writes <paramref name="t"/> as raw F32 to <c>{HARTSY_ZETA_DUMP}/{name}.bin</c> and appends the shape to the manifest. No-op unless <see cref="DumpActive"/>.</summary>
    internal static unsafe void DumpBin(string name, Tensor t)
    {
        if (!DumpActive) return;
        Directory.CreateDirectory(ZetaDumpDir!);
        Tensor src = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        long bytes = src.Shape.ElementCount * sizeof(float);
        byte[] buf = new byte[bytes];
        fixed (byte* dst = buf) Buffer.MemoryCopy((void*)src.DataPointer, dst, bytes, bytes);
        File.WriteAllBytes(Path.Combine(ZetaDumpDir!, name + ".bin"), buf);
        System.Text.StringBuilder shape = new System.Text.StringBuilder();
        for (int i = 0; i < t.Shape.Rank; i++)
        {
            if (i > 0) shape.Append('x');
            shape.Append(t.Shape[i]);
        }
        File.AppendAllText(Path.Combine(ZetaDumpDir!, "shapes.txt"), $"{name} {shape}\n");
        if (!ReferenceEquals(src, t)) src.Dispose();
    }

    /// <summary>Appends a scalar record to the dump manifest. No-op unless <see cref="DumpActive"/>.</summary>
    internal static void DumpScalar(string name, float value)
    {
        if (!DumpActive) return;
        Directory.CreateDirectory(ZetaDumpDir!);
        File.AppendAllText(Path.Combine(ZetaDumpDir!, "shapes.txt"), $"{name} {value:R}\n");
    }

    /// <summary>Creates a Zeta-Chroma transformer from configuration (use <see cref="ZetaChromaConfig.FromWeights"/>).</summary>
    public ZetaChromaTransformer(ZetaChromaConfig config)
    {
        _config = config;
        _backbone = new ZImageTransformer(config.Backbone);
        _decoder = new ZetaChromaDecoderHead(
            patchDim: config.PatchSize * config.PatchSize * 3,
            maxFreqs: config.DecoderMaxFreqs,
            resBlockCount: config.DecoderResBlocks);
    }

    /// <summary>Pixel patch size (32 reported). Generation resolution must be divisible by this.</summary>
    public int PatchSize => _config.PatchSize;

    /// <summary>Loads all weights from a partitioned Zeta-Chroma dict (Z-Image single-file naming plus
    /// <c>dec_net.*</c>). Refuses non-Zeta dicts (no decoder head) and unknown decoder layouts —
    /// <see cref="ZetaChromaDecoderHead.LoadWeights"/> throws <see cref="UnsupportedModelException"/>
    /// listing every missing/unmatched <c>dec_net.*</c> key.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        if (!ZetaChromaConfig.IsZetaChroma(weights))
        {
            throw new UnsupportedModelException(
                "No dec_net.* keys found — this is a plain Z-Image checkpoint, not Zeta-Chroma. Use ZImageTransformer.",
                architecture: "zeta-chroma", format: null);
        }
        _backbone.LoadWeightsInternal(weights, requireFinalLayer: false);
        _decoder.LoadWeights(weights);
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _backbone.EnumerateWeights()) yield return w;
        foreach (Tensor w in _decoder.EnumerateWeights()) yield return w;
    }

    /// <summary>Forward pass: predicts the clean image (x0) for one denoising step.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="pixels">Noisy pixel sample [B, 3, H, W] in [-1, 1]; H/W must be multiples of <see cref="PatchSize"/>.</param>
    /// <param name="captionEmbeddings">Qwen3-4B caption embeddings [B, capLen, 2560] (same encoder path as Z-Image).</param>
    /// <param name="sigma">Timestep conditioning value. The pipeline passes <c>1 − sigma</c> following Z-Image's
    /// inversion convention (validation-gated, research doc item 7).</param>
    /// <returns>x0 prediction [B, 3, H, W].</returns>
    public unsafe Tensor Forward(IBackend backend, Tensor pixels, Tensor captionEmbeddings, float sigma)
    {
        int batch = (int)pixels.Shape[0];
        int height = (int)pixels.Shape[2];
        int width = (int)pixels.Shape[3];
        int patch = _config.PatchSize;
        if (height % patch != 0 || width % patch != 0)
            throw new ArgumentException($"Pixel H/W ({height}x{width}) must be divisible by patch size {patch}.");

        _dumpForwardIndex++;
        DumpBin("pixels_in", pixels);
        DumpBin("caption_embeddings", captionEmbeddings);
        DumpScalar("sigma_arg", sigma);

        // Raw noisy patches captured BEFORE the backbone embeds them — the decoder refines these directly.
        Tensor pixelPatches = ZImageTransformer.Patchify(pixels, batch, 3, height, width, patch);
        DumpBin("pixel_patches", pixelPatches);

        (Tensor imgTokens, Tensor tEmb, int hPacked, int wPacked) =
            _backbone.ForwardCore(backend, pixels, captionEmbeddings, sigma);
        DumpBin("img_tokens", imgTokens);
        // The decoder conditions on per-patch hidden states only; the timestep embedding goes unused.
        tEmb.Dispose();

        Tensor decoded = _decoder.Forward(backend, pixelPatches, imgTokens);
        pixelPatches.Dispose();
        imgTokens.Dispose();
        DumpBin("decoder_out", decoded);

        Tensor x0 = ZImageTransformer.Unpatchify(decoded, batch, 3, hPacked, wPacked, patch);
        decoded.Dispose();

        // NextDiT negation convention: raw output = −x0 (validation-gated, see class doc).
        float* p = (float*)x0.DataPointer;
        long count = x0.Shape.ElementCount;
        for (long i = 0; i < count; i++)
            p[i] = -p[i];

        DumpBin("x0_out", x0);
        return x0;
    }

    /// <summary>Releases component weight references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _backbone.Dispose();
            _decoder.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}
