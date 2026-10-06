namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>Config for <see cref="IndexTtsConformerEncoder"/> — IndexTTS-1.5's <c>gpt.condition_module</c> (ESPnet Conformer, non-macaron, relative-position attention, conv module, <c>conv2d2</c> subsampling).</summary>
public sealed record IndexTtsConformerConfig
{
    public required int InputSize { get; init; }     // mel bands (100)
    public required int OutputSize { get; init; }     // 512
    public required int AttentionHeads { get; init; } // 8
    public required int LinearUnits { get; init; }    // 2048
    public required int NumBlocks { get; init; }      // 6
    public int ConvKernel { get; init; } = 15;
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>IndexTTS-1.5's <c>condition_module</c> preset from <c>config.yaml</c>.</summary>
    public static IndexTtsConformerConfig V1_5 => new()
    {
        InputSize = 100,
        OutputSize = 512,
        AttentionHeads = 8,
        LinearUnits = 2048,
        NumBlocks = 6,
    };

    /// <summary>IndexTTS-2's <c>gpt.condition_module</c> preset — the SAME shape for both 2.0 and 2.5 (confirmed:
    /// the config.yaml block is byte-identical between the two real repos). Input is the w2v-bert-2.0 feature
    /// (1024-dim), not a mel spectrogram — used by IndexTTS-2.0's speaker-conditioning Conformer (reusing
    /// <see cref="IndexTtsSpeakerEncoder"/> unchanged) since 2.0's real checkpoint still uses
    /// <c>condition_type: conformer_perceiver</c> for its own speaker path (confirmed: 2.0's gpt.pth has
    /// conditioning_encoder/perceiver_encoder/speed_emb tensors; 2.5's has spk_emb_proj/lang_embedding instead —
    /// NOT used for 2.5's speaker conditioning at all).</summary>
    public static IndexTtsConformerConfig IndexTts2Condition => new()
    {
        InputSize = 1024,
        OutputSize = 512,
        AttentionHeads = 8,
        LinearUnits = 2048,
        NumBlocks = 6,
    };

    /// <summary>IndexTTS-2's <c>gpt.emo_condition_module</c> preset — used by BOTH 2.0 and 2.5 (confirmed
    /// byte-identical emo_conditioning_encoder/emo_perceiver_encoder tensors in both real checkpoints; only the
    /// speaker-conditioning path differs between versions, not the emotion path).</summary>
    public static IndexTtsConformerConfig IndexTts2EmoCondition => new()
    {
        InputSize = 1024,
        OutputSize = 512,
        AttentionHeads = 4,
        LinearUnits = 1024,
        NumBlocks = 4,
    };
}
