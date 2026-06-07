namespace SharpInference.Audio.Models.LanguageModels.Gpt;

/// <summary>Config for a <see cref="GptBackbone"/> — a vanilla GPT-2 pre-norm decoder. Width / depth /
/// head count parameterize the shared body; token vocab + output heads are owned by each model.</summary>
public sealed record GptConfig
{
    public required int Hidden { get; init; }
    public required int NumLayers { get; init; }
    public required int NumHeads { get; init; }

    /// <summary>Max positions the learned positional embedding addresses (1024 for Bark/GPT-2).</summary>
    public int BlockSize { get; init; } = 1_024;

    public int HeadDim => Hidden / NumHeads;
    public int MlpDim => 4 * Hidden;

    /// <summary>Bark full preset (hidden 1024 / 24 layers / 16 heads).</summary>
    public static GptConfig BarkFull => new() { Hidden = 1_024, NumLayers = 24, NumHeads = 16 };

    /// <summary>Bark-Small preset (hidden 768 / 12 layers / 12 heads — GPT-2-small footprint).</summary>
    public static GptConfig BarkSmall => new() { Hidden = 768, NumLayers = 12, NumHeads = 12 };
}
