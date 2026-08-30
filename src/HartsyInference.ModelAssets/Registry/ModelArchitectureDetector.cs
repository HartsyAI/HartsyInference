using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Registry;

/// <summary>Detects the <see cref="ModelArchitecture"/> of a checkpoint by inspecting its tensor-key layout. The per-model <c>*CheckpointConverter</c> classes each key off a distinctive prefix set; this consolidates those signatures into one ordered dispatch table so a factory / loader can route to the right pipeline without each caller re-deriving the heuristic.
///
/// <para>Rules are evaluated <b>most-specific first</b> — e.g. SDXL base (which has <c>conditioner.embedders.1</c>) must be tested before the SDXL refiner rule (which only requires embedder 0), and SD3 (MMDiT) before the generic LDM rules. A checkpoint that matches nothing returns <see cref="ModelArchitecture.Unknown"/>.</para></summary>
public static class ModelArchitectureDetector
{
    private delegate bool Signature(KeySet keys);

    /// <summary>Ordered (architecture, predicate) rules. First match wins.</summary>
    private static readonly (ModelArchitecture Arch, Signature Match)[] Rules =
    [
        // ── MiniMax-H3 joint audio/video DiT ─────────────────────────────
        // Published component files are bare; bundled community repacks may add either diffusion wrapper.
        (ModelArchitecture.MiniMaxH3, k =>
            k.HasPair("video_patch_proj.", "audio_patch_proj.")
            || k.HasPair("diffusion_model.video_patch_proj.", "diffusion_model.audio_patch_proj.")
            || k.HasPair("model.diffusion_model.video_patch_proj.",
                "model.diffusion_model.audio_patch_proj.")),

        // ── SDXL family (LDM single-file) ────────────────────────────────
        // Base has both CLIP embedders; refiner has only embedder 0 + ADM label_emb.
        (ModelArchitecture.Sdxl, k =>
            k.HasPrefix("model.diffusion_model.") &&
            k.HasPrefix("conditioner.embedders.1.")),
        (ModelArchitecture.SdxlRefiner, k =>
            k.HasPrefix("model.diffusion_model.") &&
            k.HasPrefix("conditioner.embedders.0.") &&
            k.HasPrefix("model.diffusion_model.label_emb.") &&
            !k.HasPrefix("conditioner.embedders.1.")),

        // ── SD3 / 3.5 (MMDiT) ────────────────────────────────────────────
        (ModelArchitecture.StableDiffusion3, k =>
            k.HasPrefix("text_encoders.clip_g.") ||
            k.HasPrefix("model.diffusion_model.joint_blocks.") ||
            k.HasPrefix("model.diffusion_model.x_embedder.proj.")),

        // ── SD1.5 (LDM single-file) ──────────────────────────────────────
        (ModelArchitecture.StableDiffusion15, k =>
            k.HasPrefix("model.diffusion_model.") &&
            k.HasPrefix("cond_stage_model.transformer.")),

        // ── Flux.2 before Flux.1 (Flux.2 modulation keys are distinctive) ─
        (ModelArchitecture.Flux2, k =>
            k.HasPrefixAfterOptional("model.diffusion_model.", "double_stream_modulation_img.")),

        // ── Flux.1 (double + single streams) ─────────────────────────────
        (ModelArchitecture.Flux1, k =>
            k.HasPrefixAfterOptional("model.diffusion_model.", "double_blocks.") &&
            k.HasPrefixAfterOptional("model.diffusion_model.", "single_blocks.")),

        // ── Chroma (Flux-lineage with radiance/nerf heads) ───────────────
        (ModelArchitecture.Chroma, k =>
            k.HasPrefixAfterOptional("model.diffusion_model.", "nerf_") ||
            k.HasPrefixAfterOptional("model.diffusion_model.", "img_in_patch.")),

        // ── AuraFlow ─────────────────────────────────────────────────────
        (ModelArchitecture.AuraFlow, k =>
            k.HasPrefixAfterOptional("model.", "double_layers.") &&
            k.HasPrefixAfterOptional("model.", "single_layers.")),
    ];

    /// <summary>Detects the architecture from a set of tensor keys.</summary>
    public static ModelArchitecture Detect(IEnumerable<string> tensorKeys)
    {
        KeySet keys = new KeySet(tensorKeys);
        foreach ((ModelArchitecture arch, Signature match) in Rules)
        {
            if (match(keys))
            {
                Logs.Debug($"ModelArchitectureDetector: matched {arch}.");
                return arch;
            }
        }

        Logs.Debug("ModelArchitectureDetector: no rule matched; returning Unknown.");
        return ModelArchitecture.Unknown;
    }

    /// <summary>Detects the architecture from already-loaded weights.</summary>
    public static ModelArchitecture Detect(IReadOnlyDictionary<string, Tensor> weights) =>
        Detect(weights.Keys);

    /// <summary>Detects the architecture from a safetensors file by reading only its header (no tensor data is materialized). For a diffusers-layout directory or multi-shard checkpoint, pass the representative file resolved by <see cref="ModelLayoutResolver"/>.</summary>
    public static ModelArchitecture DetectFromFile(string safetensorsPath)
    {
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(safetensorsPath);
        return Detect(loader.Descriptors.Keys);
    }

    /// <summary>Lightweight prefix-membership probe over a checkpoint's key set. Sorted once so prefix hits are a single binary search rather than a full scan per query.</summary>
    private sealed class KeySet
    {
        private readonly string[] _sorted;

        public KeySet(IEnumerable<string> keys)
        {
            _sorted = keys.ToArray();
            Array.Sort(_sorted, StringComparer.Ordinal);
        }

        /// <summary>True if any key starts with <paramref name="prefix"/>.</summary>
        public bool HasPrefix(string prefix)
        {
            int idx = Array.BinarySearch(_sorted, prefix, StringComparer.Ordinal);
            if (idx >= 0) return true;
            int insert = ~idx;
            return insert < _sorted.Length && _sorted[insert].StartsWith(prefix, StringComparison.Ordinal);
        }

        /// <summary>True if any key starts with <paramref name="inner"/>, with or without the <paramref name="optionalOuter"/> wrapper (handles BFL <c>model.diffusion_model.</c> prefixing).</summary>
        public bool HasPrefixAfterOptional(string optionalOuter, string inner) =>
            HasPrefix(inner) || HasPrefix(optionalOuter + inner);

        public bool HasPair(string first, string second) => HasPrefix(first) && HasPrefix(second);
    }
}
