using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Auto-detects the <see cref="WanVideoConfig"/> for a Wan-Video DiT directly from its converted
/// (diffusers-named) weight dictionary — inner dim, patch size, in/out channels, layer count, FFN dim, head count,
/// and I2V CLIP image conditioning are all read from tensor shapes/keys. This lets a single entry point load any Wan
/// variant (T2V 1.3B/14B, I2V-14B, TI2V-5B, A14B experts, VACE) without the caller hand-picking a preset.
///
/// <para>Two things are NOT recoverable from a single DiT file and are left at their config defaults for the caller to
/// override: the <b>MoE boundary ratio</b> (Wan2.2 A14B ships as two architecturally-identical expert files, so
/// "is this MoE" is a packaging fact, not a weight fact) and the resolution-specific <b>flow shift</b> (3.0 for 480p /
/// 5.0 for 720p). VACE control layers are also not in the base DiT (they ship in a separate control checkpoint).</para></summary>
public static class WanConfigDetector
{
    /// <summary>Builds a <see cref="WanVideoConfig"/> from converted transformer weights
    /// (the <c>WanVideoCheckpointConverter.Convert</c> output). Throws if the dict is not a Wan DiT.</summary>
    public static WanVideoConfig Detect(IReadOnlyDictionary<string, Tensor> w)
    {
        ArgumentNullException.ThrowIfNull(w);
        if (!w.TryGetValue("patch_embedding.weight", out Tensor? patch))
            throw new ArgumentException("Not a Wan DiT: missing patch_embedding.weight.", nameof(w));
        if (patch.Shape.Rank != 5)
            throw new ArgumentException($"patch_embedding.weight rank {patch.Shape.Rank} != 5 [inner,in,pt,ph,pw].", nameof(w));

        int inner = (int)patch.Shape[0];
        int inChannels = (int)patch.Shape[1];
        (int T, int H, int W) patchSize = ((int)patch.Shape[2], (int)patch.Shape[3], (int)patch.Shape[4]);
        int patchVol = patchSize.T * patchSize.H * patchSize.W;

        if (!w.TryGetValue("proj_out.weight", out Tensor? projOut))
            throw new ArgumentException("Not a Wan DiT: missing proj_out.weight.", nameof(w));
        int outChannels = (int)(projOut.Shape[0] / patchVol);

        int numLayers = 0;
        while (w.ContainsKey($"blocks.{numLayers}.ffn.net.0.proj.weight")) numLayers++;
        if (numLayers == 0)
            throw new ArgumentException("Not a Wan DiT: no blocks.{i}.ffn.net.0.proj.weight found.", nameof(w));
        int ffnDim = (int)w["blocks.0.ffn.net.0.proj.weight"].Shape[0];

        const int headDim = 128;   // Wan is fixed at 128 across every variant
        int numHeads = inner / headDim;

        // I2V CLIP image conditioning: the image_embedder MLP + per-block image KV projections.
        bool i2vClip = w.ContainsKey("condition_embedder.image_embedder.norm1.weight")
                       || w.ContainsKey("blocks.0.attn2.add_k_proj.weight");
        int imageDim = 0, addedKv = 0, posLen = 0;
        if (i2vClip)
        {
            imageDim = w.TryGetValue("condition_embedder.image_embedder.norm1.weight", out Tensor? in1)
                ? (int)in1.Shape[in1.Shape.Rank - 1] : 1280;                     // LayerNorm over the CLIP feature dim
            addedKv = imageDim;
            posLen = w.TryGetValue("condition_embedder.image_embedder.pos_embed", out Tensor? pe)
                ? (int)pe.Shape[pe.Shape.Rank - 2] : 257;                        // [.,tokens,dim]
        }

        int z = outChannels;                        // predicted VAE latent width
        int spatialCompression = z >= 48 ? 16 : 8;  // Wan2.2 z=48 VAE is 16×; Wan2.1 z=16 VAE is 8×

        return new WanVideoConfig
        {
            PatchSize = patchSize,
            NumHeads = numHeads,
            HeadDim = headDim,
            InChannels = inChannels,
            OutChannels = outChannels,
            FfnDim = ffnDim,
            NumLayers = numLayers,
            VaeLatentChannels = z,
            VaeSpatialCompression = spatialCompression,
            ImageDim = imageDim,
            AddedKvProjDim = addedKv,
            PosEmbedSeqLen = posLen,
        };
    }

    /// <summary>Human-readable one-line summary of a detected/loaded config (for logs + status reporting).</summary>
    public static string Describe(WanVideoConfig c) =>
        $"Wan DiT: inner={c.InnerDim} ({c.NumHeads}h×{c.HeadDim}) layers={c.NumLayers} ffn={c.FfnDim} " +
        $"in={c.InChannels} out={c.OutChannels} z={c.VaeLatentChannels}/{c.VaeSpatialCompression}× " +
        (c.HasImageConditioning ? $"I2V-CLIP(img={c.ImageDim},pos={c.PosEmbedSeqLen}) " : "") +
        (c.InChannels > c.OutChannels ? "concat-I2V " : "") +
        (c.IsMixtureOfExperts ? $"MoE(boundary={c.BoundaryRatio}) " : "");
}
