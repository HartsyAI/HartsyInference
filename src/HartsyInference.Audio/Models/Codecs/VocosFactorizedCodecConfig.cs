namespace HartsyInference.Audio.Models.Codecs;

/// <summary>Config for <see cref="VocosFactorizedCodec"/> — the Amphion/MaskGCT "Vocos encoder/decoder +
/// factorized single-codebook VQ, optional 2x down/up resample" codec shape. Real upstream ships this under two
/// names for the exact same architecture: <c>RepCodec</c> (<c>amphion/MaskGCT</c>'s semantic codec, IndexTTS-2.0's
/// dependency, no resample) and <c>EnhancedCodec</c> (IndexTTS-2.5's own bundled <c>codec.pth</c>, 2x resample
/// active) — both <c>indextts/codec/*.py</c> sources are otherwise byte-identical. Not IndexTTS-scoped: any future
/// MaskGCT-family model can reuse this class and config directly.</summary>
public sealed record VocosFactorizedCodecConfig
{
    public required int CodebookSize { get; init; }
    public required int HiddenSize { get; init; }
    public required int CodebookDim { get; init; }
    public required int VocosDim { get; init; }
    public required int VocosIntermediateDim { get; init; }
    public required int VocosNumLayers { get; init; }

    /// <summary>0 or 1 = no resample (<c>RepCodec</c>'s real default); 2 = the real <c>EnhancedCodec</c>'s active
    /// 2x down/up (confirmed: IndexTTS-2.5's <c>config.yaml</c> has no <c>downsample_scale</c> key, so the real
    /// Python falls back to each class's own different default — 1 for <c>RepCodec</c>, 2 for
    /// <c>EnhancedCodec</c> — rather than both sharing one config value).</summary>
    public int DownsampleScale { get; init; } = 1;

    /// <summary>IndexTTS-2.0's dependency (<c>amphion/MaskGCT</c>'s <c>semantic_codec/model.safetensors</c>): no
    /// resample. Confirmed only <c>.Quantize</c> is ever called in the real <c>infer_v2.py</c> — the decoder half
    /// is trained but dead weight for this inference path, so skip loading it (<c>loadDecoder: false</c>).</summary>
    public static VocosFactorizedCodecConfig IndexTts2V0 => new()
    {
        CodebookSize = 8192,
        HiddenSize = 1024,
        CodebookDim = 8,
        VocosDim = 384,
        VocosIntermediateDim = 2048,
        VocosNumLayers = 12,
        DownsampleScale = 1,
    };

    /// <summary>IndexTTS-2.5's own bundled <c>codec.pth</c>: 2x resample active. Confirmed both <c>.Quantize</c>
    /// (reference encoding) AND <c>.Decode</c> (AR-generated codes → S2Mel content, with the 2x upsample) are
    /// used in the real <c>infer_v2_5.py</c> — load the decoder.</summary>
    public static VocosFactorizedCodecConfig IndexTts2V5 => IndexTts2V0 with { DownsampleScale = 2 };
}
