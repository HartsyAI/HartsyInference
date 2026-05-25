using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>YOLOv8 Bottleneck block: two 3×3 Conv-BN-SiLU stages, optionally with a residual
/// shortcut. The expansion ratio is fixed at 1.0 inside the bottleneck (so the hidden channel
/// count equals the output channel count), and the residual is added only when the channel
/// counts match AND <see cref="Shortcut"/> is true — Ultralytics' standard contract.</summary>
public sealed unsafe class Bottleneck
{
    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;
    private readonly bool _shortcut;
    private readonly int _inChannels;
    private readonly int _outChannels;

    /// <summary>Whether the shortcut path is enabled (only takes effect when in/out channels match).</summary>
    public bool Shortcut => _shortcut;

    /// <summary>Creates a Bottleneck. <paramref name="shortcut"/> is the YOLO standard contract — backbone bottlenecks use shortcut=true, neck bottlenecks use shortcut=false.</summary>
    public Bottleneck(int inChannels, int outChannels, bool shortcut)
    {
        _cv1 = new ConvBnSilu(outChannels, strideH: 1, strideW: 1, padH: 1, padW: 1, useSilu: true);
        _cv2 = new ConvBnSilu(outChannels, strideH: 1, strideW: 1, padH: 1, padW: 1, useSilu: true);
        _shortcut = shortcut;
        _inChannels = inChannels;
        _outChannels = outChannels;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _cv1.LoadWeights(weights, $"{prefix}.cv1.conv");
        _cv2.LoadWeights(weights, $"{prefix}.cv2.conv");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _cv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv2.EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        Tensor mid = _cv1.Forward(backend, input);
        Tensor twoConvs = _cv2.Forward(backend, mid);
        mid.Dispose();

        if (!_shortcut || _inChannels != _outChannels)
            return twoConvs;

        // Residual add. Both tensors are [N, C_out, H, W] of the same shape — Conv with stride=1,
        // padding=1, k=3 preserves spatial dimensions, and we just confirmed channels match.
        Tensor residual = new Tensor(twoConvs.Shape, DType.F32);
        backend.Add(residual, twoConvs, input);
        twoConvs.Dispose();
        return residual;
    }
}
