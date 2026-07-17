using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>End-to-end faithful Grounding DINO (<c>IDEA-Research/grounding-dino-tiny</c>): Swin-T backbone → neck
/// (input_proj + sine positions) → BERT text tower → cross-modality enhancer encoder → two-stage query selection →
/// decoder → contrastive class + box heads. Produces the raw <c>logits</c> and <c>pred_boxes</c>; use
/// <see cref="GroundingDinoPipeline.PostProcess"/> to turn them into thresholded, labeled detections.</summary>
public sealed unsafe class GroundingDinoModel : IDisposable
{
    private readonly GroundingDinoConfig _cfg;
    private readonly SwinBackbone _backbone;
    private readonly GroundingDinoNeck _neck;
    private readonly GroundingDinoTextTower _textTower;
    private readonly GroundingDinoEncoder _encoder;
    private readonly GroundingDinoDetector _detector;
    private int _disposed;

    public GroundingDinoConfig Config => _cfg;

    public GroundingDinoModel(GroundingDinoConfig cfg)
    {
        _cfg = cfg;
        _backbone = new SwinBackbone(cfg);
        _neck = new GroundingDinoNeck(cfg);
        _textTower = new GroundingDinoTextTower(cfg);
        _encoder = new GroundingDinoEncoder(cfg);
        _detector = new GroundingDinoDetector(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _backbone.LoadWeights(weights);
        _neck.LoadWeights(weights);
        _textTower.LoadWeights(weights);
        _encoder.LoadWeights(weights);
        _detector.LoadWeights(weights);
    }

    /// <summary>Runs the full detector. <paramref name="pixelValues"/> is <c>[1, 3, H, W]</c> (already preprocessed),
    /// <paramref name="inputIds"/> the tokenized caption. Returns the raw <c>logits [1, Nq, max_text_len]</c> and
    /// <c>pred_boxes [1, Nq, 4]</c>. Caller owns both.</summary>
    public GroundingDinoDetector.Output Forward(IBackend backend, Tensor pixelValues, int[] inputIds)
    {
        Tensor[] feats = _backbone.Forward(backend, pixelValues);
        GroundingDinoNeck.Result neck = _neck.Forward(backend, feats);
        foreach (Tensor f in feats) f.Dispose();

        Tensor text = _textTower.Encode(backend, inputIds);
        (bool[] attend, int[] positionIds) = GroundingDinoTextPrompt.Build(inputIds);

        (Tensor encVision, Tensor encText) = _encoder.Forward(backend, neck.SourceFlatten, neck.PositionFlatten,
            text, attend, positionIds, neck.SpatialShapes, neck.LevelStartIndex);
        neck.SourceFlatten.Dispose();
        neck.PositionFlatten.Dispose();
        text.Dispose();

        GroundingDinoDetector.Output outp = _detector.Forward(backend, encVision, encText, inputIds.Length,
            neck.SpatialShapes, neck.LevelStartIndex);
        encVision.Dispose();
        encText.Dispose();
        return outp;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        _neck.Dispose();
        _textTower.Dispose();
        _encoder.Dispose();
        _detector.Dispose();
        GC.SuppressFinalize(this);
    }
}
