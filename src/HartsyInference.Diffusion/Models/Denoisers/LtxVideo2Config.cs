using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for LTX-2.3 (Lightricks, 22B) — a dual-stream audio+video DiT. Verified against the
/// vendored diffusers <c>LTX2VideoTransformer3DModel</c> and the checkpoint header (original Lightricks key names
/// under <c>model.diffusion_model.*</c>: <c>patchify_proj</c>, <c>adaln_single</c>, <c>av_ca_*</c>, the per-modality
/// <c>{video,audio}_embeddings_connector</c>). Two interleaved streams — video (inner 4096 = 32×128) and audio
/// (inner 2048 = 32×64) — run through 48 <see cref="LtxVideo2Block"/>s with self-attn (gated, QK-RMSNorm, RoPE),
/// text cross-attn, and bidirectional audio↔video cross-attn, all AdaLN-Single modulated. The LTX-2.3 variant
/// enables cross-attention modulation (<c>cross_attn_mod</c>/<c>audio_cross_attn_mod</c> ⇒ 9 self-attn AdaLN params
/// and prompt/av-ca modulation tables).</summary>
public sealed record LtxVideo2Config
{
    // ── Video stream ──
    public int InChannels { get; init; } = 128;
    public int OutChannels { get; init; } = 128;
    public int NumHeads { get; init; } = 32;
    public int HeadDim { get; init; } = 128;
    public int InnerDim => NumHeads * HeadDim;          // 4096
    /// <summary>Text cross-attention KV width (video attn2 keys/values; = caption-projected dim).</summary>
    public int CrossAttentionDim { get; init; } = 4096;

    // ── Audio stream ──
    public int AudioInChannels { get; init; } = 128;
    public int AudioOutChannels { get; init; } = 128;
    public int AudioNumHeads { get; init; } = 32;
    public int AudioHeadDim { get; init; } = 64;
    public int AudioInnerDim => AudioNumHeads * AudioHeadDim;   // 2048
    public int AudioCrossAttentionDim { get; init; } = 2048;

    // ── Shared ──
    public int NumLayers { get; init; } = 48;
    /// <summary>Gemma-3-12B caption feature width fed to the per-modality connectors.</summary>
    public int CaptionChannels { get; init; } = 3840;
    /// <summary>FFN expansion factor (plain gelu-approximate MLP: proj ↑4×, then back down). Video 4096→16384,
    /// audio 2048→8192.</summary>
    public int FfnMultiplier { get; init; } = 4;

    /// <summary>Self-attention AdaLN params per block when cross-attn modulation is on (shift/scale/gate ×3 stages:
    /// self-attn, cross-attn, FFN). The per-block <c>scale_shift_table</c> is <c>[9, inner]</c>.</summary>
    public int SelfAttnModParams { get; init; } = 9;
    /// <summary>Output-layer AdaLN params (<c>scale_shift_table [2, inner]</c>: shift, scale).</summary>
    public int OutputModParams { get; init; } = 2;
    /// <summary>Cross-attn modulation is enabled for LTX-2.3 (prompt + av-ca modulation tables present).</summary>
    public bool CrossAttnMod { get; init; } = true;

    // ── Norms ──
    public float NormEps { get; init; } = 1e-6f;
    /// <summary>QK-RMSNorm "rms_norm_across_heads" epsilon (diffusers <c>norm_eps</c> = 1e-6).</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    // ── RoPE ──
    /// <summary>Rotary apply convention. LTX-2.3 (22B) ships <c>rope_type=split</c> in the checkpoint metadata; base
    /// LTX / the diffusers default is interleaved. Applying the wrong flavor scrambles spatial positions (32px grid).</summary>
    public LtxVideo2Rope.RopeType RopeType { get; init; } = LtxVideo2Rope.RopeType.Split;
    public float RopeTheta { get; init; } = 10000.0f;
    public int RopeBaseNumFrames { get; init; } = 20;
    public int RopeBaseHeight { get; init; } = 2048;
    public int RopeBaseWidth { get; init; } = 2048;
    public int CausalOffset { get; init; } = 1;
    /// <summary>Audio RoPE positions derive from sample timing (audio is 1-D along time).</summary>
    public int AudioSamplingRate { get; init; } = 16000;
    public int AudioHopLength { get; init; } = 160;
    public int AudioScaleFactor { get; init; } = 4;
    public int AudioPosEmbedMaxPos { get; init; } = 20;

    // ── VAE compression (pipeline grid math) ──
    public int VaeSpatialCompression { get; init; } = 32;
    public int VaeTemporalCompression { get; init; } = 8;

    // ── Sampling defaults ──
    public int TimestepScaleMultiplier { get; init; } = 1000;
    /// <summary>Scale multiplier for the a2v/v2a cross-attn gate timestep. The gate AdaLN sees
    /// <c>timestep · (CrossAttnTimestepScaleMultiplier / TimestepScaleMultiplier)</c>.</summary>
    public int CrossAttnTimestepScaleMultiplier { get; init; } = 1000;
    public float TimestepShift { get; init; } = 1.0f;
    public int NumInferenceSteps { get; init; } = 50;
    public float GuidanceScale { get; init; } = 3.0f;

    /// <summary>The audio stream's own CFG scale; null follows the video scale, matching the reference default.
    /// The authors' 7.0-for-audio recommendation assumes the guidance rescale/STG this port does not implement —
    /// measured alone it makes the soundtrack quieter, not louder.</summary>
    public float? AudioGuidanceScale { get; init; }

    /// <summary>Blend factor for rescaling the guided audio velocity back to the conditional's statistics
    /// (diffusers' <c>rescale_noise_cfg</c>); 0 disables it. Full rescale is the default because CFG otherwise
    /// leaves the audio latent ~2.5x over-dispersed and the soundtrack ~30 dB down.</summary>
    public float AudioGuidanceRescale { get; init; } = 1.0f;

    // ── Generation deltas (2.5) ──

    /// <summary>Whether the video-branch FFN Linears carry biases. LTX-2.5 ships <c>ff_bias=false</c>; the audio and
    /// connector FFNs keep theirs in every released checkpoint. Nothing branches on this — a bias-free FFN already
    /// works by key absence — it exists so loading can fail loudly when the checkpoint disagrees with the variant.</summary>
    public bool FfBias { get; init; } = true;

    /// <summary>Whether the checkpoint carries <c>keyframes_abs_pos_embedding</c>, a learned marker added to the
    /// tokens whose temporal position starts at 0. LTX-2.5 sets this; it affects ordinary text-to-video, not only
    /// multi-shot generation.</summary>
    public bool UseKeyframesAbsPosEmbedding { get; init; }

    /// <summary>Sigma schedule the distilled checkpoints were trained against, replacing the dynamic flow-match
    /// shift when set. Distillation baked these values in, so a different step count is not a valid schedule.</summary>
    public float[]? FixedSigmas { get; init; }

    /// <summary>Distilled 2.5 schedule (8 steps: 9 sigmas ending at 0), from the reference pipeline's
    /// <c>DISTILLED_SIGMA_VALUES</c>. A fresh array per call — a shared instance would let one in-place edit
    /// anywhere corrupt every config built afterwards.</summary>
    public static float[] Ltx25DistilledSigmas =>
        [1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f];

    public static LtxVideo2Config V23 => new();

    /// <summary>LTX-2.5 dev. Differs from <see cref="V23"/> in exactly the two keys the shipped checkpoints'
    /// metadata differ in — every other dimension, rope and modulation setting is identical.</summary>
    public static LtxVideo2Config V25 => new() { FfBias = false, UseKeyframesAbsPosEmbedding = true };

    /// <summary>LTX-2.5 distilled. Architecturally indistinguishable from <see cref="V25"/> — the checkpoints share
    /// a model version, config and tensor keys — so this is selected by the caller's intent, never by inspection.</summary>
    public static LtxVideo2Config V25Distilled => V25 with
    {
        FixedSigmas = Ltx25DistilledSigmas,
        NumInferenceSteps = 8,
        GuidanceScale = 1.0f,
    };
}
