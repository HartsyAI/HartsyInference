using HartsyInference.ModelAssets.TextEncoders.Bert;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Faithful configuration for <c>IDEA-Research/grounding-dino-tiny</c> (HF <c>GroundingDinoForObjectDetection</c>).
/// Values mirror the shipped <c>config.json</c>: Swin-T backbone (depths [2,2,6,2], heads [3,6,12,24], embed 96),
/// d_model 256, 6 encoder + 6 decoder layers, 4 feature levels, 900 queries, BERT-base text tower, two-stage query
/// selection.</summary>
public sealed record GroundingDinoConfig
{
    // ── Transformer core ──
    public int DModel { get; init; } = 256;
    public int EncoderLayers { get; init; } = 6;
    public int DecoderLayers { get; init; } = 6;
    public int EncoderAttentionHeads { get; init; } = 8;
    public int DecoderAttentionHeads { get; init; } = 8;
    public int EncoderFfnDim { get; init; } = 2048;
    public int DecoderFfnDim { get; init; } = 2048;
    public int NumFeatureLevels { get; init; } = 4;
    public int EncoderNPoints { get; init; } = 4;
    public int DecoderNPoints { get; init; } = 4;
    public int NumQueries { get; init; } = 900;
    public int QueryDim { get; init; } = 4;
    public int MaxTextLen { get; init; } = 256;
    public float LayerNormEps { get; init; } = 1e-5f;
    public float PositionalEmbeddingTemperature { get; init; } = 20f;
    public bool TwoStage { get; init; } = true;

    // ── Swin-T backbone ──
    public int SwinEmbedDim { get; init; } = 96;
    public int[] SwinDepths { get; init; } = [2, 2, 6, 2];
    public int[] SwinNumHeads { get; init; } = [3, 6, 12, 24];
    public int SwinWindowSize { get; init; } = 7;
    public int SwinPatchSize { get; init; } = 4;
    public float SwinMlpRatio { get; init; } = 4f;
    public float SwinLayerNormEps { get; init; } = 1e-5f;
    /// <summary>Backbone stages exported to the neck: stage2/3/4 → channels 192/384/768.</summary>
    public int[] BackboneOutChannels { get; init; } = [192, 384, 768];

    // ── Text tower ──
    public BertConfig Text { get; init; } = BertConfig.GroundingDino;

    /// <summary>Per-coordinate half-width for the DETR sinusoidal position embedding (d_model/2 = 128).</summary>
    public int SinePosDim => DModel / 2;

    public static GroundingDinoConfig Tiny => new();
}
