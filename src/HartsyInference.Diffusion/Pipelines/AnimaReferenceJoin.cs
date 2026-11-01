using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>A reference joined into one Anima generation: its encoded latent and that latent's identity, the gate that switches the reference
/// adapter's LoRA, the adapter's strength on that gate, the reference's RoPE placement, and how its guidance combines. The latent is
/// borrowed: the caller owns it.</summary>
public sealed record AnimaReferenceJoin(Tensor Latent, long Identity, AdjunctGate Gate, float Strength, ReferencePlacement Placement,
    ReferenceGuidanceSettings Guidance);
