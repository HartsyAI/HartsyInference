namespace HartsyInference.Audio.Models.LanguageModels.Gpt;

/// <summary>Config for a <see cref="GptBackbone"/> — width / depth / head count parameterize the shared body; token vocab and output heads are owned by each model.</summary>
public sealed record GptConfig
{
    public required int Hidden { get; init; }
    public required int NumLayers { get; init; }
    public required int NumHeads { get; init; }

    /// <summary>Max positions the learned positional embedding addresses (1024 for Bark/GPT-2).</summary>
    public int BlockSize { get; init; } = 1_024;

    public int HeadDim => Hidden / NumHeads;
    public int MlpDim => 4 * Hidden;

    /// <summary>Whether the attention/MLP linear projections carry a bias. False (Bark/GPT-2-small) skips the checkpoint lookup entirely and zero-fills; true (IndexTTS, a standard HF GPT-2 export) loads the real bias, falling back to zero only if a specific key is unexpectedly absent. LayerNorm biases are unaffected — those load bias-or-zero unconditionally, since their presence is a separate convention from the linears'.</summary>
    public bool Bias { get; init; } = false;

    /// <summary>Bark full preset (hidden 1024 / 24 layers / 16 heads).</summary>
    public static GptConfig BarkFull => new() { Hidden = 1_024, NumLayers = 24, NumHeads = 16 };

    /// <summary>Bark-Small preset (hidden 768 / 12 layers / 12 heads — GPT-2-small footprint).</summary>
    public static GptConfig BarkSmall => new() { Hidden = 768, NumLayers = 12, NumHeads = 12 };

    /// <summary>IndexTTS-1.5 T2S preset (hidden 1280 / 24 layers / 20 heads); checkpoint is a standard biased HF GPT-2.</summary>
    public static GptConfig IndexTts15 => new() { Hidden = 1_280, NumLayers = 24, NumHeads = 20, BlockSize = 1_402, Bias = true };
}
