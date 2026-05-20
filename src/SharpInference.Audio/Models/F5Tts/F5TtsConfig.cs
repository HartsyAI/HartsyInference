namespace SharpInference.Audio.Models.F5Tts;

/// <summary>Configuration for F5-TTS (SWivid, 2024) — flow-matching DiT for zero-shot
/// voice cloning. Architectural fields verified against the <c>SWivid/F5-TTS</c>
/// safetensors layout (<c>F5TTS_v1_Base/model_1250000.safetensors</c>).
///
/// <para>Forward consumes (noisy_target_mel, ref_audio_mel, text_token_ids,
/// timestep) → predicts a vector field over the target portion of the mel. The
/// pipeline runs flow-match Euler with Sway-Sampling timesteps. The final
/// generated mel is vocoded to 24 kHz audio via <see cref="Vocoders.Vocos"/>.</para></summary>
public sealed record F5TtsConfig
{
    /// <summary>DiT hidden dim. 1024 for F5-TTS v1 base.</summary>
    public int Dim { get; init; } = 1_024;

    /// <summary>Number of DiT blocks. 22 for v1 base.</summary>
    public int Depth { get; init; } = 22;

    /// <summary>Attention heads. 16 → head_dim = 64 (full RoPE on all 64 dims).</summary>
    public int Heads { get; init; } = 16;

    /// <summary>Convenience.</summary>
    public int HeadDim => Dim / Heads;

    /// <summary>Feed-forward multiplier. 2 for v1 base (unusually small — most DiTs
    /// use 4x).</summary>
    public int FfMult { get; init; } = 2;

    /// <summary>FFN inner dim, used directly.</summary>
    public int FfInner => Dim * FfMult;

    /// <summary>Mel channels. 100 — matches Vocos 24 kHz preset.</summary>
    public int MelDim { get; init; } = 100;

    /// <summary>Text encoder hidden dim. 512 for v1 base — smaller than DiT dim.</summary>
    public int TextDim { get; init; } = 512;

    /// <summary>Text encoder ConvNeXtV2 block count.</summary>
    public int TextConvLayers { get; init; } = 4;

    /// <summary>ConvNeXtV2 intermediate dim ratio (inside the text encoder).</summary>
    public int TextConvMult { get; init; } = 2;

    /// <summary>Maximum text positions cached in the sinusoidal text-pos table.
    /// 8192 at 24 kHz audio with hop 256 = ~87 s — generous upper bound.</summary>
    public int TextMaxPos { get; init; } = 8_192;

    /// <summary>Character vocab size. F5-TTS v1 base uses 2545 chars (+1 filler =
    /// 2546 embedding rows).</summary>
    public int TextNumEmbeds { get; init; } = 2_545;

    /// <summary>Sinusoidal timestep embedding pre-MLP dim.</summary>
    public int TimeFreqEmbedDim { get; init; } = 256;

    /// <summary>ConvPositionEmbedding kernel size — 31 (must be odd).</summary>
    public int ConvPosKernel { get; init; } = 31;

    /// <summary>ConvPositionEmbedding groups — 16 (NOT depthwise; 1024-ch split into
    /// 16 groups of 64 channels each).</summary>
    public int ConvPosGroups { get; init; } = 16;

    /// <summary>RoPE base frequency. 10000.</summary>
    public float RopeTheta { get; init; } = 10_000f;

    /// <summary>Default F5-TTS v1 Base preset (SWivid/F5-TTS, F5TTS_v1_Base).</summary>
    public static F5TtsConfig V1Base => new();
}
