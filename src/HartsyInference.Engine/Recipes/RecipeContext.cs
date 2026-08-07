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

    /// <summary>Second backend to run the CFG uncond branch on, concurrent with cond on <see cref="Backend"/>;
    /// null (the default) means no CFG-branch parallelism — cond/uncond run sequentially on <see cref="Backend"/>
    /// as before. Unlike <see cref="TextEncoderBackend"/>/<see cref="VaeBackend"/> there is no "OrDefault" —
    /// null here is a meaningful "not configured", not a fallback to resolve. The denoiser's weights must be
    /// replicated on this backend too (VRAM cost, not split — see <c>ROADMAP.md</c> §1), so recipes only opt in
    /// when they can preload onto it.</summary>
    public IBackend? CfgParallelBackend { get; init; }

    /// <summary>Second backend to run the DiT's tail block range on for VRAM-pooling sharding (Phase 8); null
    /// (the default) means no DiT sharding. Unlike <see cref="CfgParallelBackend"/> (replicated weights, a latency
    /// win) this SPLITS the block range — the win is pooled VRAM, not latency (sequential pipeline split). Like
    /// <see cref="CfgParallelBackend"/> there is no "OrDefault": null is meaningful "not configured". Recipes compute
    /// the actual split point once they know the DiT's block count (see <c>Krea2Recipe</c> /
    /// <c>PlacementPlanner.DitSplitPlan</c>).</summary>
    public IBackend? DitShardBackend { get; init; }

    /// <summary>Ordered backends for stages <c>1..N-1</c> of Phase 8+ N-way DiT block-range sharding (stage 0 is
    /// always <see cref="Backend"/>); null/empty = no N-way sharding. Currently consumed only by
    /// <c>QwenImageRecipe</c> — the other five sharded families stay on <see cref="DitShardBackend"/>'s 2-way shape
    /// (<c>ROADMAP.md</c> item 7 tracks generalizing them too). Populated alongside <see cref="DitShardBackend"/>
    /// from the same <c>PlacementConfig.ShardDevices</c> list, so a 2-device config gives both the same information.</summary>
    public IReadOnlyList<IBackend>? DitShardBackends { get; init; }

    /// <summary>Ordered rank backends for context parallelism (the video DiT's token-sequence split); rank 0 is
    /// always <see cref="Backend"/>. Null/empty = no context parallelism. Like <see cref="CfgParallelBackend"/> the
    /// weights are REPLICATED on every rank (latency, not VRAM pooling) and null is a meaningful "not configured".
    /// Currently consumed only by the Wan video recipe.</summary>
    public IReadOnlyList<IBackend>? CpBackends { get; init; }

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
            if (CfgParallelBackend is not null && !ReferenceEquals(CfgParallelBackend, Backend)
                && !ReferenceEquals(CfgParallelBackend, TextEncoderBackend) && !ReferenceEquals(CfgParallelBackend, VaeBackend))
            {
                yield return CfgParallelBackend;
            }
            if (DitShardBackend is not null && !ReferenceEquals(DitShardBackend, Backend)
                && !ReferenceEquals(DitShardBackend, TextEncoderBackend) && !ReferenceEquals(DitShardBackend, VaeBackend)
                && !ReferenceEquals(DitShardBackend, CfgParallelBackend))
            {
                yield return DitShardBackend;
            }
            if (DitShardBackends is not null)
            {
                foreach (IBackend stageBackend in DitShardBackends)
                {
                    if (!ReferenceEquals(stageBackend, Backend) && !ReferenceEquals(stageBackend, TextEncoderBackend)
                        && !ReferenceEquals(stageBackend, VaeBackend) && !ReferenceEquals(stageBackend, CfgParallelBackend)
                        && !ReferenceEquals(stageBackend, DitShardBackend))
                    {
                        yield return stageBackend;
                    }
                }
            }
            if (CpBackends is not null)
            {
                foreach (IBackend rankBackend in CpBackends)
                {
                    if (!ReferenceEquals(rankBackend, Backend) && !ReferenceEquals(rankBackend, TextEncoderBackend)
                        && !ReferenceEquals(rankBackend, VaeBackend) && !ReferenceEquals(rankBackend, CfgParallelBackend)
                        && !ReferenceEquals(rankBackend, DitShardBackend))
                    {
                        yield return rankBackend;
                    }
                }
            }
        }
    }

    /// <summary>Optional swappable-component overrides; null keeps the recipe's defaults.</summary>
    public ComponentOverrides? Components { get; init; }

    /// <summary>LoRA stack to merge into the loaded weights at construction; null for none. LoRA is baked into the
    /// weights, so the constructed pipeline is cached under a key that includes this stack.</summary>
    public LoraStack? Loras { get; init; }
}
