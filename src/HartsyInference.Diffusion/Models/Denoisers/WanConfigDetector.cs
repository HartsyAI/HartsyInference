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
/// 5.0 for 720p). VACE checkpoints (merged base + control branch) are detected via <c>vace_patch_embedding</c>.</para></summary>
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

        // S2V: the causal audio encoder + audio injector (post-converter names are the raw checkpoint names — the
        // converter's rule table doesn't touch them). The inject-layer INDICES are a config fact, not a weight fact;
        // only the count is recoverable, so map known (numLayers, count) combos to the reference lists.
        int[] audioInjectLayers = [];
        int audioDim = 1024, audioTokens = 4, audioLayers = 25;
        if (w.TryGetValue("casual_audio_encoder.weights", out Tensor? audioW))
        {
            audioLayers = (int)audioW.Shape[audioW.Shape.Rank >= 2 ? 1 : 0];   // [1, numLayers, 1, 1]
            Tensor conv1Local = w["casual_audio_encoder.encoder.conv1_local.conv.weight"];
            Tensor conv2 = w["casual_audio_encoder.encoder.conv2.conv.weight"];
            audioDim = (int)conv1Local.Shape[1];
            audioTokens = (int)(conv1Local.Shape[0] / conv2.Shape[1]);         // conv1_local out = (dim/4)·tokens; conv2 in = dim/4
            int injectorCount = 0;
            while (w.ContainsKey($"audio_injector.injector.{injectorCount}.q.weight")) injectorCount++;
            audioInjectLayers = (numLayers, injectorCount) switch
            {
                (40, 12) => [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],     // Wan2.2-S2V-14B
                (_, 2) => [0, 27],                                             // reference AudioInjector_WAN default
                _ => throw new ArgumentException(
                    $"Unknown S2V audio-inject layout: {numLayers} blocks with {injectorCount} injectors — set WanVideoConfig.AudioInjectLayers manually.", nameof(w)),
            };
        }

        // fp8 checkpoints carry a small velocity DC bias that CFG≥5 amplifies into a dark trajectory — enable CFG
        // renormalization for them (fp16/bf16 stay at 0 = plain CFG, byte-identical). Detect fp8 by the matmul dtype.
        float cfgRescale = w.TryGetValue("blocks.0.ffn.net.0.proj.weight", out Tensor? ffn0) && ffn0.DType.IsFp8 ? 0.7f : 0f;

        // VACE control branch: vace_patch_embedding [inner, vaceIn, pt, ph, pw] + evenly-spread vace_blocks
        // (block i attaches to main layer i·(numLayers/count) — comfy's vace_layers_mapping).
        int vaceInChannels = 0;
        int[] vaceLayers = [];
        if (w.TryGetValue("vace_patch_embedding.weight", out Tensor? vacePatch))
        {
            vaceInChannels = (int)vacePatch.Shape[1];
            int vaceCount = 0;
            while (w.ContainsKey($"vace_blocks.{vaceCount}.proj_out.weight")) vaceCount++;
            if (vaceCount == 0 || numLayers % vaceCount != 0)
                throw new ArgumentException($"VACE DiT: {vaceCount} vace_blocks don't evenly divide {numLayers} layers.", nameof(w));
            int step = numLayers / vaceCount;
            vaceLayers = new int[vaceCount];
            for (int i = 0; i < vaceCount; i++) vaceLayers[i] = i * step;
        }

        // Wan2.2-Animate: pose patch embed + face adapter (post-converter these keep their original names).
        bool isAnimate = w.ContainsKey("pose_patch_embedding.weight")
                         && w.ContainsKey("face_adapter.fuser_blocks.0.linear1_kv.weight");

        // S2V CFG defaults (empirical, 2026-07-02): the reference-faithful silence-audio uncond darkens the output
        // ~linearly in (cfg−1) — cfg 5:0.7 collapses to black, cfg 2 with full renorm is the usable ceiling until the
        // uncond path's numeric parity is settled. Text-only CFG (HARTSY_S2V_TEXT_CFG=1) is the interim for higher cfg.
        float guidance = audioInjectLayers.Length > 0 ? 2.0f : 5.0f;
        if (audioInjectLayers.Length > 0) cfgRescale = 1.0f;

        return new WanVideoConfig
        {
            GuidanceScale = guidance,
            IsAnimate = isAnimate,
            VaceLayers = vaceLayers,
            VaceInChannels = vaceInChannels,
            CfgRescale = cfgRescale,
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
            AudioInjectLayers = audioInjectLayers,
            AudioDim = audioDim,
            AudioTokens = audioTokens,
            AudioLayers = audioLayers,
        };
    }

    /// <summary>Human-readable one-line summary of a detected/loaded config (for logs + status reporting).</summary>
    public static string Describe(WanVideoConfig c) =>
        $"Wan DiT: inner={c.InnerDim} ({c.NumHeads}h×{c.HeadDim}) layers={c.NumLayers} ffn={c.FfnDim} " +
        $"in={c.InChannels} out={c.OutChannels} z={c.VaeLatentChannels}/{c.VaeSpatialCompression}× " +
        (c.HasImageConditioning ? $"I2V-CLIP(img={c.ImageDim},pos={c.PosEmbedSeqLen}) " : "") +
        (c.InChannels > c.OutChannels ? "concat-I2V " : "") +
        (c.VaceLayers.Length > 0 ? $"VACE(in={c.VaceInChannels},blocks={c.VaceLayers.Length}) " : "") +
        (c.HasAudioConditioning ? $"S2V(audio={c.AudioDim}×{c.AudioLayers},inject={c.AudioInjectLayers.Length}) " : "") +
        (c.IsAnimate ? "Animate(pose+face) " : "") +
        (c.IsMixtureOfExperts ? $"MoE(boundary={c.BoundaryRatio}) " : "");
}
