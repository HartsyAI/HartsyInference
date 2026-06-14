namespace HartsyInference.Audio.Models.Moonshine;

/// <summary>Configuration for the Moonshine encoder-decoder STT model
/// (Useful Sensors, 2024). All values are verified from the upstream HuggingFace
/// <c>config.json</c> for the official tiny / base releases. Moonshine's defining
/// differences from Whisper:
/// <list type="bullet">
///   <item>Raw waveform input — a 3-layer Conv1D stem at the front, NO mel spectrogram.</item>
///   <item>RoPE on encoder AND decoder, with a partial rotary factor (only ~62% of head dims rotated).</item>
///   <item>Decoder MLP is gated SwiGLU (fc1 outputs 2× intermediate); encoder MLP is standard GELU.</item>
///   <item>Variable-length input — no fixed 30s zero-padding.</item>
/// </list>
/// </summary>
public sealed record MoonshineConfig
{
    /// <summary>Hidden size (d_model).</summary>
    public int HiddenSize { get; init; } = 416;

    /// <summary>Encoder layer count.</summary>
    public int EncoderLayers { get; init; } = 8;

    /// <summary>Decoder layer count.</summary>
    public int DecoderLayers { get; init; } = 8;

    /// <summary>Encoder + decoder attention head count. Head dim = HiddenSize / NumHeads;
    /// for base this is 416/8 = 52, which is NOT a multiple of 8 (this is fine — the
    /// <c>pad_head_dim_to_multiple_of</c> config field is a kernel-optimization hint
    /// that doesn't change the stored weights).</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>Convenience: head dimension.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>FFN inner dim. Standard Llama-style 4 * HiddenSize.</summary>
    public int IntermediateSize { get; init; } = 1664;

    /// <summary>Token vocabulary size including the 768 reserved special tokens
    /// (32000 BPE + 768 &lt;&lt;ST_x&gt;&gt; pseudo-timestamps).</summary>
    public int VocabSize { get; init; } = 32_768;

    /// <summary>Maximum text positions. Moonshine uses 194; long-form audio is handled
    /// by chunked streaming rather than extended decoder context.</summary>
    public int MaxTextPositions { get; init; } = 194;

    /// <summary>RoPE theta (base frequency). Standard 10000.</summary>
    public float RopeTheta { get; init; } = 10_000f;

    /// <summary>Fraction of each head's dim that gets RoPE applied. The remaining
    /// <c>head_dim - rotary_dim</c> trailing dims pass through unchanged. base/tiny
    /// both use 0.62. The actual rotary dim is computed as
    /// <c>int(HeadDim * PartialRotaryFactor) &amp; ~1</c> (round down to an even number).</summary>
    public float PartialRotaryFactor { get; init; } = 0.62f;

    /// <summary>Resolved rotary dim — even-length prefix of each head that rotates.</summary>
    public int RotaryDim => (int)(HeadDim * PartialRotaryFactor) & ~1;

    /// <summary>LayerNorm epsilon. Moonshine uses the default 1e-5.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Encoder activation. Always GELU for stock Moonshine.</summary>
    public bool EncoderUseSiluGated { get; init; } = false;

    /// <summary>Decoder activation. Always SwiGLU (gated SiLU) for stock Moonshine.</summary>
    public bool DecoderUseSiluGated { get; init; } = true;

    /// <summary>Beginning-of-sequence token. 1.</summary>
    public int BosTokenId { get; init; } = 1;

    /// <summary>End-of-sequence token. 2.</summary>
    public int EosTokenId { get; init; } = 2;

    /// <summary>Padding token. Also 2 (Moonshine reuses EOS as pad).</summary>
    public int PadTokenId { get; init; } = 2;

    // ── Audio front-end (Conv1D stem) ──────────────────────────────────────────

    /// <summary>Conv1 output channels. Always HiddenSize.</summary>
    public int Conv1OutChannels => HiddenSize;
    /// <summary>Conv1 kernel size.</summary>
    public int Conv1Kernel { get; init; } = 127;
    /// <summary>Conv1 stride.</summary>
    public int Conv1Stride { get; init; } = 64;

    /// <summary>Conv2 output channels. 2 × HiddenSize.</summary>
    public int Conv2OutChannels => HiddenSize * 2;
    public int Conv2Kernel { get; init; } = 7;
    public int Conv2Stride { get; init; } = 3;

    /// <summary>Conv3 output channels. Back to HiddenSize.</summary>
    public int Conv3OutChannels => HiddenSize;
    public int Conv3Kernel { get; init; } = 3;
    public int Conv3Stride { get; init; } = 2;

    /// <summary>Total downsample factor: <c>Conv1Stride × Conv2Stride × Conv3Stride</c>.
    /// 64 × 3 × 2 = 384 for the stock base, giving a 16kHz/384 ≈ 41.67 Hz encoder rate.</summary>
    public int TotalDownsample => Conv1Stride * Conv2Stride * Conv3Stride;

    // ── Presets ────────────────────────────────────────────────────────────────

    /// <summary>tiny — 27M params, hidden=288, 6 layers, head_dim=36.</summary>
    public static MoonshineConfig Tiny => new()
    {
        HiddenSize = 288,
        EncoderLayers = 6,
        DecoderLayers = 6,
        NumHeads = 8,             // tiny: head_dim = 288/8 = 36
        IntermediateSize = 1152,  // 4 * 288
        VocabSize = 32_768,
        PartialRotaryFactor = 0.9f,  // tiny uses 0.9 (per HF config)
    };

    /// <summary>base — 61M params, hidden=416, 8 layers, head_dim=52.</summary>
    public static MoonshineConfig Base => new()
    {
        HiddenSize = 416,
        EncoderLayers = 8,
        DecoderLayers = 8,
        NumHeads = 8,
        IntermediateSize = 1664,
        VocabSize = 32_768,
        PartialRotaryFactor = 0.62f,
    };
}
