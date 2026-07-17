using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end RT-DETR model: CNN backbone → hybrid encoder (AIFI + CCFM) → deformable
/// transformer decoder → class + box heads. NMS-free — the decoder emits exactly <c>NumQueries</c>
/// candidate detections that the pipeline thresholds and top-k's directly.
/// <para>Input is <c>[1, 3, H, W]</c> (H, W multiples of 32). Output is class logits
/// <c>[1, NumQueries, NumClasses]</c> (pre-sigmoid) and boxes <c>[1, NumQueries, 4]</c> as
/// cxcywh normalized to [0, 1].</para></summary>
public sealed class RtDetrModel
{
    private readonly RtDetrConfig _config;
    private readonly RtDetrBackbone _backbone;
    private readonly RtDetrHybridEncoder _encoder;
    private readonly RtDetrDecoder _decoder;

    /// <summary>The config this model was constructed with.</summary>
    public RtDetrConfig Config => _config;

    /// <summary>Number of detection classes.</summary>
    public int NumClasses => _config.NumClasses;

    /// <summary>Number of object queries (= max detections emitted per image).</summary>
    public int NumQueries => _config.NumQueries;

    /// <summary>The deformable-attention decoder — exposed for structural tests + diagnostics.</summary>
    public RtDetrDecoder Decoder => _decoder;

    /// <summary>Builds the model from a config (validated here).</summary>
    public RtDetrModel(RtDetrConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        _config = config;
        _backbone = new RtDetrBackbone(config);
        _encoder = new RtDetrHybridEncoder(config);
        _decoder = new RtDetrDecoder(config);
    }

    /// <summary>Loads all weights. Sub-module prefixes are <c>{prefix}.backbone</c>, <c>{prefix}.encoder</c>,
    /// <c>{prefix}.decoder</c> (default top-level <paramref name="prefix"/> is empty, so keys start at the module name).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        string root = string.IsNullOrEmpty(prefix) ? "" : $"{prefix}.";
        _backbone.LoadWeights(weights, $"{root}backbone");
        _encoder.LoadWeights(weights, $"{root}encoder");
        _decoder.LoadWeights(weights, $"{root}decoder");
    }

    /// <summary>Yields every weight tensor for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the full forward pass. Returns class logits <c>[1, NumQueries, NumClasses]</c>
    /// (pre-sigmoid) and boxes <c>[1, NumQueries, 4]</c> (cxcywh in [0,1]). Caller owns both.</summary>
    public (Tensor ClassLogits, Tensor Boxes) Forward(IBackend backend, Tensor pixelValues)
    {
        (Tensor cls, Tensor boxes, Tensor hidden) = ForwardInternal(backend, pixelValues);
        hidden.Dispose();
        return (cls, boxes);
    }

    /// <summary>Structural-test hook: runs the full forward and returns only the final decoder hidden
    /// state <c>[1, NumQueries, hidden]</c> (class + box heads discarded). Caller owns the tensor.</summary>
    internal Tensor ForwardHiddenState(IBackend backend, Tensor pixelValues)
    {
        (Tensor cls, Tensor boxes, Tensor hidden) = ForwardInternal(backend, pixelValues);
        cls.Dispose();
        boxes.Dispose();
        return hidden;
    }

    private (Tensor ClassLogits, Tensor Boxes, Tensor Hidden) ForwardInternal(IBackend backend, Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(pixelValues);
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"RT-DETR input must be [1, 3, H, W]; got {pixelValues.Shape}.", nameof(pixelValues));

        (Tensor c3, Tensor c4, Tensor c5) = _backbone.Forward(backend, pixelValues);
        (Tensor p3, Tensor n4, Tensor n5) = _encoder.Forward(backend, c3, c4, c5);
        c3.Dispose(); c4.Dispose(); c5.Dispose();

        (Tensor Map, int H, int W)[] maps =
        [
            (p3, (int)p3.Shape[2], (int)p3.Shape[3]),
            (n4, (int)n4.Shape[2], (int)n4.Shape[3]),
            (n5, (int)n5.Shape[2], (int)n5.Shape[3]),
        ];
        (Tensor cls, Tensor boxes, Tensor hidden) = _decoder.Forward(backend, maps);
        p3.Dispose(); n4.Dispose(); n5.Dispose();
        return (cls, boxes, hidden);
    }
}
