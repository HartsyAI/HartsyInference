using System.Collections.Concurrent;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Engine.Features;

/// <summary>Lazy-loads and process-wide-caches the per-architecture TAESD preview decoders. The first call for an
/// architecture downloads the weights and mmaps them; later calls are dictionary lookups. The mmap-backed tensors are
/// read-only, so one decoder instance is safe to share across concurrent generations. Returns null for architectures
/// with no published TAESD checkpoint (Flux.2's 32-channel latent, AuraFlow, F-Lite) — callers fall back to latent2rgb.</summary>
public static class TaesdResolver
{
    /// <summary>One slot per arch; lazy so toggling previews on doesn't download every checkpoint, only the one in use.</summary>
    private static readonly ConcurrentDictionary<LatentArchitecture, TaesdDecoder?> _cache = new ConcurrentDictionary<LatentArchitecture, TaesdDecoder?>();

    /// <summary>Serializes the download+load so two concurrent generations on the same arch don't both fetch the file.</summary>
    private static readonly SemaphoreSlim _gate = new SemaphoreSlim(1, 1);

    /// <summary>Returns a TAESD decoder for <paramref name="arch"/>, downloading + loading on first call, or null when no
    /// published checkpoint exists (caller falls back to <see cref="LatentPreview"/>). A failed download or corrupt file
    /// also yields null rather than breaking generation.</summary>
    public static async Task<TaesdDecoder?> ResolveAsync(LatentArchitecture arch, Action<string> log, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(log);
        if (_cache.TryGetValue(arch, out TaesdDecoder? cached))
        {
            return cached;
        }
        ModelAsset? asset = AssetFor(arch);
        if (asset is null)
        {
            // No published TAESD for this architecture — caller falls back to latent2rgb.
            _cache[arch] = null;
            return null;
        }
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_cache.TryGetValue(arch, out cached))
            {
                return cached;
            }
            TaesdDecoder? decoder = null;
            try
            {
                string path = await ModelDownloader.EnsureSideModelAsync(asset, null, ct).ConfigureAwait(false);
                if (!File.Exists(path))
                {
                    log($"[TAESD] {asset.FileName}: resolved model has no usable file path. Falling back to latent2rgb.");
                }
                else
                {
                    log($"[TAESD] Loading {asset.FileName} from {path}");
                    decoder = TaesdDecoder.LoadFromSafetensors(arch, path);
                    log($"[TAESD] {asset.FileName} ready ({decoder.LatentChannels}-channel latent).");
                }
            }
            catch (Exception ex)
            {
                // A failed download / corrupt file shouldn't break generation — fall back.
                Logs.Warning($"[Features][TAESD] Failed to load decoder for {arch}: {ex.GetType().Name}: {ex.Message}. Falling back to latent2rgb.");
                decoder = null;
            }
            _cache[arch] = decoder;
            return decoder;
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Maps a latent architecture to its TAESD weight asset; Chroma / Z-Image reuse the Flux.1 weights since they share Flux's VAE.</summary>
    private static ModelAsset? AssetFor(LatentArchitecture arch) => arch switch
    {
        LatentArchitecture.Sd15 => SideModels.TaesdSd15,
        LatentArchitecture.Sdxl => SideModels.TaesdSdxl,
        LatentArchitecture.Sd3 => SideModels.TaesdSd3,
        // Flux.1 weights work for any 16-channel Flux-family VAE.
        LatentArchitecture.Flux or LatentArchitecture.Chroma or LatentArchitecture.ZImage => SideModels.TaesdFlux,
        // Flux.2 (32-ch) needs a different weight set; AuraFlow / F-Lite haven't been distilled upstream.
        _ => null,
    };
}
