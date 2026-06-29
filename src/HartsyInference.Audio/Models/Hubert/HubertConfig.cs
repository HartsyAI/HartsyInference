namespace HartsyInference.Audio.Models.Hubert;

/// <summary>Configuration for a HuBERT / Wav2Vec2-base content encoder (e.g. <c>chinese-hubert-base</c>). A
/// 7-layer strided conv feature extractor (320× downsample → 50 Hz) feeds a 12-layer post-LayerNorm
/// transformer; GPT-SoVITS and RVC tap the final <c>last_hidden_state</c> as the content features. 16 kHz
/// mono input. See <c>docs/Research/GPT_SOVITS_ARCHITECTURE.md</c>.</summary>
public sealed record HubertConfig
{
    public int ConvDim { get; init; } = 512;
    public IReadOnlyList<int> ConvKernels { get; init; } = [10, 3, 3, 3, 3, 2, 2];
    public IReadOnlyList<int> ConvStrides { get; init; } = [5, 2, 2, 2, 2, 2, 2];

    public int Hidden { get; init; } = 768;
    public int NumLayers { get; init; } = 12;
    public int NumHeads { get; init; } = 12;
    public int HeadDim => Hidden / NumHeads;     // 64
    public int FfnDim { get; init; } = 3_072;

    public int PosConvKernel { get; init; } = 128;
    public int PosConvGroups { get; init; } = 16;
    public float NormEps { get; init; } = 1e-5f;
    public int SampleRate { get; init; } = 16_000;

    /// <summary>Whether the feature-extractor conv layers carry a bias term (base = false, large = true).</summary>
    public bool ConvBias { get; init; } = false;
    /// <summary>Feature-extractor normalization mode: "group" (GroupNorm on layer 0 only) or "layer" (LayerNorm on every conv layer).</summary>
    public string FeatExtractNorm { get; init; } = "group";
    /// <summary>Use the pre-LayerNorm (stable) transformer variant (true on the large model). The forward path for this is Phase 4.</summary>
    public bool DoStableLayerNorm { get; init; } = false;
    /// <summary>Optional final projection dim (e.g. ContentVec projects to 256); null = no projection.</summary>
    public int? FinalProjDim { get; init; } = null;

    /// <summary>Total conv downsample factor (320 → 50 Hz at 16 kHz).</summary>
    public int Downsample
    {
        get { int p = 1; for (int i = 0; i < ConvStrides.Count; i++) p *= ConvStrides[i]; return p; }
    }

    public static HubertConfig ChineseHubertBase => new();

    /// <summary>HuBERT-Base trained on LibriSpeech 960h (same base dims as <see cref="ChineseHubertBase"/>).</summary>
    public static HubertConfig HubertBaseLs960 => new();

    /// <summary>HuBERT-Large trained on Libri-Light 60k: wider/deeper stack with conv bias, per-layer
    /// feature-extractor LayerNorm, and the pre-LN (stable) transformer variant. Stable-LN forward path is Phase 4.</summary>
    public static HubertConfig HubertLargeLl60k => new()
    {
        Hidden = 1024,
        NumLayers = 24,
        NumHeads = 16,
        FfnDim = 4_096,
        ConvBias = true,
        FeatExtractNorm = "layer",
        DoStableLayerNorm = true,
    };

    /// <summary>ContentVec content encoder: HuBERT-Base with a final projection to 256 dims.</summary>
    public static HubertConfig ContentVec => new()
    {
        FinalProjDim = 256,
    };
}
