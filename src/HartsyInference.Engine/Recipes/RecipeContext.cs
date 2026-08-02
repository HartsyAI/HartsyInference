using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>Everything a recipe needs to construct its pipeline: the checkpoint path, the compute backend, and any
/// per-request component overrides (VAE / text encoders). Side models beyond the checkpoint are fetched by the recipe
/// via <see cref="ModelDownloader"/> against the <see cref="SideModels"/> registry.</summary>
public sealed record RecipeContext
{
    /// <summary>Local path to the primary checkpoint (the transformer/UNet file or its directory).</summary>
    public required string CheckpointPath { get; init; }

    /// <summary>Compute backend the constructed pipeline runs on.</summary>
    public required IBackend Backend { get; init; }

    /// <summary>Backend for prompt/text encoders (CLIP/T5/umT5/LLM-style); null = <see cref="Backend"/>. Safe
    /// without any peer-copy machinery because every encoder→denoiser handoff host-materializes the embeddings —
    /// an invariant the pipelines assert with their pre-loop <c>DataPointer</c> sweeps.</summary>
    public IBackend? TextEncoderBackend { get; init; }

    /// <summary>Backend for VAE encode/decode; null = <see cref="Backend"/>. The latent handoff is host-side
    /// (UnpackLatent and friends), so this is placement-safe like the text encoder.</summary>
    public IBackend? VaeBackend { get; init; }

    /// <summary>The text-encoder backend with the primary fallback applied.</summary>
    public IBackend TextEncoderBackendOrDefault => TextEncoderBackend ?? Backend;

    /// <summary>The VAE backend with the primary fallback applied.</summary>
    public IBackend VaeBackendOrDefault => VaeBackend ?? Backend;

    /// <summary>Every distinct backend this recipe will touch — recipes that set per-backend flags
    /// (CacheWeightCasts, fp8 toggles) must apply them to ALL of these, not just <see cref="Backend"/>.</summary>
    public IEnumerable<IBackend> AllBackends
    {
        get
        {
            yield return Backend;
            if (TextEncoderBackend is not null && !ReferenceEquals(TextEncoderBackend, Backend))
            {
                yield return TextEncoderBackend;
            }
            if (VaeBackend is not null && !ReferenceEquals(VaeBackend, Backend)
                && !ReferenceEquals(VaeBackend, TextEncoderBackend))
            {
                yield return VaeBackend;
            }
        }
    }

    /// <summary>Optional swappable-component overrides; null keeps the recipe's defaults.</summary>
    public ComponentOverrides? Components { get; init; }

    /// <summary>LoRA stack to merge into the loaded weights at construction; null for none. LoRA is baked into the
    /// weights, so the constructed pipeline is cached under a key that includes this stack.</summary>
    public LoraStack? Loras { get; init; }
}
