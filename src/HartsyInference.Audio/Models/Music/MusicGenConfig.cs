namespace HartsyInference.Audio.Models.Music;

/// <summary>Configuration for MusicGen / AudioGen (Meta AudioCraft) — an autoregressive Transformer that
/// generates EnCodec tokens conditioned on T5-base text (cross-attention). K residual codebooks are
/// predicted per step via the <b>delay pattern</b> (each codebook offset by k timesteps). The shared
/// recipe: frozen T5-base encoder + single-stage causal decoder with cross-attn + frozen EnCodec; music
/// (32 kHz / 4 codebooks) vs sound (16 kHz / 4 codebooks) differ only in codec config.
/// See <c>docs/Research/MUSICGEN_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> codec decode reuses the built EnCodec; sampling reuses `NucleusSampler`. The
/// decoder self-attention + GELU MLP are the GPT-2 family (shared patterns) plus a cross-attention
/// sublayer to the (caller-supplied) T5 states — so the pipeline takes precomputed T5 cross-attn states
/// and carries no T5 dependency in the Audio package.</para></summary>
public sealed record MusicGenConfig
{
    public required int Hidden { get; init; }
    public required int NumLayers { get; init; }
    public required int NumHeads { get; init; }

    public int HeadDim => Hidden / NumHeads;
    public int FfnDim => 4 * Hidden;
    public int MaxPositions { get; init; } = 2_048;

    /// <summary>T5-base encoder dim (cross-attn K/V source). 768.</summary>
    public int TextDim { get; init; } = 768;

    public int NumCodebooks { get; init; } = 4;
    public int CodebookSize { get; init; } = 2_048;
    /// <summary>Special token id reserved as BOS / "no token yet" in the delay pattern (== codebook size).</summary>
    public int SpecialToken { get; init; } = 2_048;

    public int CodecSampleRate { get; init; } = 32_000;
    public int CodecFrameRate { get; init; } = 50;

    /// <summary>Delay offsets per codebook. Mono default <c>[0,1,2,3]</c>; stereo doubles to
    /// <c>[0,0,1,1,2,2,3,3]</c>.</summary>
    public int[] DelayPattern { get; init; } = [0, 1, 2, 3];

    /// <summary>Output audio channels. Mono is 1; stereo variants use 2 (NumCodebooks doubles to 8).</summary>
    public int AudioChannels { get; init; } = 1;

    /// <summary>Number of chroma bins used by the melody conditioner (12 pitch classes).</summary>
    public int NumChroma { get; init; } = 12;

    /// <summary>Chroma sequence length (frames) for the melody conditioner.</summary>
    public int ChromaLength { get; init; } = 235;

    public float Temperature { get; init; } = 1.0f;
    public int TopK { get; init; } = 250;
    public float TopP { get; init; } = 0.0f;
    public float GuidanceScale { get; init; } = 3.0f;

    public static MusicGenConfig Small => new() { Hidden = 1_024, NumLayers = 24, NumHeads = 16 };
    public static MusicGenConfig Medium => new() { Hidden = 1_536, NumLayers = 48, NumHeads = 24 };
    public static MusicGenConfig Large => new() { Hidden = 2_048, NumLayers = 48, NumHeads = 32 };
    /// <summary>AudioGen-medium: t5-large text conditioner (dim 1024, not t5-base 768) + 16 kHz EnCodec.</summary>
    public static MusicGenConfig AudioGen => new() { Hidden = 1_536, NumLayers = 48, NumHeads = 24, TextDim = 1_024, CodecSampleRate = 16_000 };

    /// <summary>Melody-conditioned medium variant (text + chroma conditioning).</summary>
    public static MusicGenConfig Melody => new() { Hidden = 1_536, NumLayers = 48, NumHeads = 24 };

    /// <summary>Stereo small: 8 codebooks (2 channels x 4) with the doubled delay pattern.</summary>
    public static MusicGenConfig StereoSmall => new() { Hidden = 1_024, NumLayers = 24, NumHeads = 16, AudioChannels = 2, NumCodebooks = 8, DelayPattern = [0, 0, 1, 1, 2, 2, 3, 3] };

    /// <summary>Stereo medium: 8 codebooks (2 channels x 4) with the doubled delay pattern.</summary>
    public static MusicGenConfig StereoMedium => new() { Hidden = 1_536, NumLayers = 48, NumHeads = 24, AudioChannels = 2, NumCodebooks = 8, DelayPattern = [0, 0, 1, 1, 2, 2, 3, 3] };

    /// <summary>Stereo large: 8 codebooks (2 channels x 4) with the doubled delay pattern.</summary>
    public static MusicGenConfig StereoLarge => new() { Hidden = 2_048, NumLayers = 48, NumHeads = 32, AudioChannels = 2, NumCodebooks = 8, DelayPattern = [0, 0, 1, 1, 2, 2, 3, 3] };
}
