using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Splits a Hunyuan3D-2 checkpoint into the three component weight dicts the pipeline loads:
/// the shape <c>Dit</c>, the <c>ShapeVae</c> decoder, and the <c>Dinov2</c> image conditioner. Follows the
/// project's converter pattern (detect layout → route keys by prefix → strip component prefix).
/// <para><b>Numerics validation-pending</b> — the per-key rename tables and prefix set are
/// <b>validation-gated</b>; they're finalized against the reference checkpoint during the layer-diff pass
/// (see <c>docs/Research/HUNYUAN3D_2_ARCHITECTURE.md</c>). This router classifies by coarse prefix so the
/// structural pipeline can load a converted checkpoint end-to-end.</para></summary>
public static class Hunyuan3DCheckpointConverter
{
    /// <summary>Per-component weight dictionaries, keyed by the names each component's <c>LoadWeights</c> expects.</summary>
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Dit { get; init; }
        public required Dictionary<string, Tensor> ShapeVae { get; init; }
        public required Dictionary<string, Tensor> Dinov2 { get; init; }
    }

    // Real tencent/Hunyuan3D-2 layout: DiT = `model.*`, ShapeVAE = `vae.*`, DINOv2-giant conditioner =
    // `conditioner.main_image_encoder.model.*` (strip the whole prefix → HF ViT `embeddings.*`/`encoder.layer.*`).
    private static readonly string[] VaePrefixes = ["vae.", "shapevae.", "shape_model."];
    private static readonly string[] DinoPrefixes = ["conditioner.main_image_encoder.model.", "conditioner.", "image_encoder.", "dinov2.", "dino."];
    private static readonly string[] DitPrefixes = ["model.", "denoiser.", "dit.", "transformer."];

    /// <summary>Routes every weight to its component, stripping the matched component prefix. Keys that don't
    /// match a VAE/DINO prefix default to the DiT bucket.</summary>
    public static ConvertedWeights Convert(IReadOnlyDictionary<string, Tensor> all)
    {
        ArgumentNullException.ThrowIfNull(all);
        Dictionary<string, Tensor> dit = new(), vae = new(), dino = new();

        foreach ((string key, Tensor t) in all)
        {
            if (TryRoute(key, t, VaePrefixes, vae)) continue;
            if (TryRoute(key, t, DinoPrefixes, dino)) continue;
            // Default → DiT (strip a DiT prefix if present, else keep as-is).
            dit[StripFirst(key, DitPrefixes)] = t;
        }

        return new ConvertedWeights { Dit = dit, ShapeVae = vae, Dinov2 = dino };
    }

    private static bool TryRoute(string key, Tensor t, string[] prefixes, Dictionary<string, Tensor> dst)
    {
        foreach (string p in prefixes)
            if (key.StartsWith(p, StringComparison.Ordinal))
            {
                dst[key[p.Length..]] = t;
                return true;
            }
        return false;
    }

    private static string StripFirst(string key, string[] prefixes)
    {
        foreach (string p in prefixes)
            if (key.StartsWith(p, StringComparison.Ordinal)) return key[p.Length..];
        return key;
    }
}
