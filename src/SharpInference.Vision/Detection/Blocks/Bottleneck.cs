using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>YOLOv8 Bottleneck block: two 3×3 Conv-BN-SiLU stages, optionally with a residual
/// shortcut. The hidden channel count is <c>int(outChannels * expansion)</c>:
/// <list type="bullet">
///   <item><c>expansion=1.0</c> (no compression): used by C2f's inner Bottlenecks and C3k's inner Bottlenecks (Ultralytics overrides the default).</item>
///   <item><c>expansion=0.5</c> (half compression, default): used by C3k2's inner Bottlenecks when c3k=False (Ultralytics uses the Bottleneck default).</item>
/// </list>
/// The residual adds only when shortcut=true AND in/out channels match.</summary>
public sealed unsafe class Bottleneck
{
    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;
    private readonly bool _shortcut;
    private readonly int _inChannels;
    private readonly int _outChannels;

    /// <summary>Whether the shortcut path is enabled (only takes effect when in/out channels match).</summary>
    public bool Shortcut => _shortcut;

    /// <summary>Creates a Bottleneck with explicit expansion ratio. <paramref name="expansion"/>
    /// controls the hidden width: <c>cv1</c> projects to <c>int(outChannels * expansion)</c>,
    /// <c>cv2</c> projects back to <c>outChannels</c>.</summary>
    public Bottleneck(int inChannels, int outChannels, bool shortcut, float expansion = 1.0f)
    {
        int hidden = (int)(outChannels * expansion);
        _cv1 = new ConvBnSilu(hidden, strideH: 1, strideW: 1, padH: 1, padW: 1, useSilu: true);
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
