namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Hyperparameters of the SAM 3 image path (text-prompted detection, masks and scores). Defaults match
/// the upstream image builder at facebookresearch/sam3 commit 0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc.</summary>
public sealed record Sam3Config
{
    /// <summary>Square model input resolution; images are stretched to this size.</summary>
    public int ImageSize { get; init; } = 1008;

    /// <summary>ViT patch size; the patch grid is <c>ImageSize / PatchSize</c> (72).</summary>
    public int PatchSize { get; init; } = 14;

    /// <summary>ViT trunk width.</summary>
    public int VitEmbedDim { get; init; } = 1024;

    /// <summary>Number of ViT blocks.</summary>
    public int VitDepth { get; init; } = 32;

    /// <summary>ViT attention heads (head dim = <c>VitEmbedDim / VitNumHeads</c>).</summary>
    public int VitNumHeads { get; init; } = 16;

    /// <summary>ViT MLP hidden width (<c>int(VitEmbedDim * 4.625)</c>).</summary>
    public int VitMlpHidden { get; init; } = 4736;

    /// <summary>Window edge (in patches) for windowed attention blocks.</summary>
    public int VitWindowSize { get; init; } = 24;

    /// <summary>Block indices that attend globally over the whole patch grid instead of windows.</summary>
    public int[] VitGlobalAttentionBlocks { get; init; } = [7, 15, 23, 31];

    /// <summary>Side of the pretrained absolute position grid that is tiled to the input grid.</summary>
    public int VitPretrainGrid { get; init; } = 24;

    /// <summary>RoPE base frequency.</summary>
    public float RopeTheta { get; init; } = 10000f;

    /// <summary>Epsilon of every LayerNorm on the image path.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Feature width shared by the FPN neck, fusion encoder, decoder, scorer and mask head.</summary>
    public int DModel { get; init; } = 256;

    /// <summary>FPN scale factors, coarse to fine; the last entry is computed upstream and then dropped by scalp.</summary>
    public float[] NeckScaleFactors { get; init; } = [4f, 2f, 1f, 0.5f];

    /// <summary>Number of FPN levels kept after scalp (the finest <c>ScalpLevels</c> levels are removed).</summary>
    public int ScalpLevels { get; init; } = 1;

    /// <summary>Text tower width.</summary>
    public int TextWidth { get; init; } = 1024;

    /// <summary>Text tower depth.</summary>
    public int TextLayers { get; init; } = 24;

    /// <summary>Text tower attention heads.</summary>
    public int TextHeads { get; init; } = 16;

    /// <summary>Text tower MLP hidden width (4x width).</summary>
    public int TextMlpHidden { get; init; } = 4096;

    /// <summary>Token count per prompt, including padding.</summary>
    public int TextContextLength { get; init; } = 32;

    /// <summary>CLIP BPE vocabulary size.</summary>
    public int TextVocabSize { get; init; } = 49408;

    /// <summary>Fusion encoder depth.</summary>
    public int EncoderLayers { get; init; } = 6;

    /// <summary>Fusion encoder attention heads (self and text cross-attention).</summary>
    public int EncoderHeads { get; init; } = 8;

    /// <summary>Fusion encoder FFN hidden width.</summary>
    public int EncoderFfn { get; init; } = 2048;

    /// <summary>Decoder depth.</summary>
    public int DecoderLayers { get; init; } = 6;

    /// <summary>Decoder attention heads.</summary>
    public int DecoderHeads { get; init; } = 8;

    /// <summary>Decoder FFN hidden width.</summary>
    public int DecoderFfn { get; init; } = 2048;

    /// <summary>Object queries per prompt (the presence token is one extra query).</summary>
    public int NumQueries { get; init; } = 200;

    /// <summary>Geometry encoder depth (only the CLS path runs for text-only prompts).</summary>
    public int GeometryLayers { get; init; } = 3;

    /// <summary>Hidden width of the prompt MLP in the dot-product scorer.</summary>
    public int ScorerHidden { get; init; } = 2048;

    /// <summary>Symmetric clamp applied to scorer logits before the sigmoid.</summary>
    public float ScorerClamp { get; init; } = 12f;

    /// <summary>Pixel-decoder upsampling stages declared upstream; the image path runs the first two (see the spec).</summary>
    public int PixelDecoderStages { get; init; } = 3;

    /// <summary>Mask-embedding MLP depth.</summary>
    public int MaskEmbedLayers { get; init; } = 3;

    /// <summary>Side of the mask logit grid produced by the segmentation head (before the processor upsample).</summary>
    public int MaskResolution { get; init; } = 288;

    /// <summary>Minimum score (sigmoid(class) * sigmoid(presence)) for an instance to be reported.</summary>
    public float ScoreThreshold { get; init; } = 0.5f;

    /// <summary>Mask-logit threshold applied after the bilinear upsample to image resolution.</summary>
    public float MaskLogitThreshold { get; init; } = 0f;

    /// <summary>Patch-grid side (<c>ImageSize / PatchSize</c>).</summary>
    public int PatchGrid => ImageSize / PatchSize;
}
