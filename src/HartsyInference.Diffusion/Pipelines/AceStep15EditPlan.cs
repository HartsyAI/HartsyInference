using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Audio-conditioned edit conditioning for <see cref="AceStepPipeline15"/> — the (source latent, per-frame chunk mask, start sigma) triple that turns plain text-to-music into continuation / repaint / cover. The caller owns <see cref="SrcLatent"/>; the pipeline never disposes it. <b>Parity-pending:</b> nothing on this path has been run against real weights, and the mask polarity is inferred from upstream's boolean <c>ones()</c> full-generation mask rather than measured.</summary>
public sealed record AceStep15EditPlan
{
    /// <summary>Source latents <c>[1, frames, latentChannels]</c> occupying the DiT's <c>src_latents</c> slot — the audio being preserved (repaint/continuation) or re-rendered (cover).</summary>
    public required Tensor SrcLatent { get; init; }

    /// <summary>Per-frame chunk mask, length = frames: <b>1 = generate, 0 = preserve</b> (values below 0.5 count as preserve). Polarity follows upstream's boolean full-generation <c>ones()</c> mask; unverified on real weights.</summary>
    public required float[] ChunkMask { get; init; }

    /// <summary>Schedule entry point in sigma, 0..1: 1 starts from full noise (the source only conditions via the context), lower values start further down the table so the result stays closer to the source (cover strength).</summary>
    public float StartSigmaFraction { get; init; } = 1f;

    /// <summary>Substitute the silence latent for the src row of every generate-masked frame in the context tensor — upstream's repaint rule (<c>src = where(chunk_mask &gt; 0.5, silence_tiled, src)</c>). True for repaint and continuation; false for cover, where the whole source stays as conditioning.</summary>
    public bool SilenceMaskedFrames { get; init; }

    /// <summary>Throws when the plan does not line up with the generation the pipeline is about to run.</summary>
    public void Validate(int frames, int latentChannels)
    {
        if (SrcLatent is null || SrcLatent.Shape.Rank != 3 || SrcLatent.Shape[0] != 1 || SrcLatent.Shape[1] != frames
            || (int)SrcLatent.Shape[2] != latentChannels)
        {
            throw new ArgumentException(
                $"Edit-plan src latent must be [1, {frames}, {latentChannels}]; got {(SrcLatent is null ? "null" : SrcLatent.Shape.ToString())}.");
        }
        if (ChunkMask is null || ChunkMask.Length != frames)
        {
            throw new ArgumentException(
                $"Edit-plan chunk mask must have one entry per latent frame ({frames}); got {ChunkMask?.Length ?? 0}.");
        }
        if (!float.IsFinite(StartSigmaFraction) || StartSigmaFraction < 0f || StartSigmaFraction > 1f)
        {
            throw new ArgumentException($"Edit-plan start sigma fraction must be within 0..1; got {StartSigmaFraction}.");
        }
    }
}
