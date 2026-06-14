using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.Blocks;

/// <summary>YOLOv8 C2f block — "Cross Stage Partial with 2 convolutions, Faster". Evolves the
/// older C3 block by concatenating <i>every</i> intermediate bottleneck output (not just the
/// last) for richer gradient flow.
/// <list type="number">
///   <item>1×1 <c>cv1</c>: expand to <c>2 * c</c> channels.</item>
///   <item>Split into two halves <c>y[0], y[1]</c> of <c>c</c> channels each.</item>
///   <item>Run <c>n</c> bottlenecks starting from <c>y[1]</c>: <c>y[2] = b0(y[1]), y[3] = b1(y[2]), ...</c>.</item>
///   <item>Concatenate every accumulated tensor <c>[y[0], y[1], y[2], ..., y[n+1]]</c> → <c>(2 + n) * c</c> channels.</item>
///   <item>1×1 <c>cv2</c>: compress to the requested output channels.</item>
/// </list>
/// <para>Expansion ratio is fixed at 0.5 by Ultralytics — i.e. <c>c = int(outChannels * 0.5)</c>.</para></summary>
public sealed class C2f
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _hiddenChannels;
    private readonly int _numBottlenecks;
    private readonly bool _shortcut;

    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;
    private readonly Bottleneck[] _bottlenecks;

    /// <summary>Creates a C2f. <paramref name="n"/> is the bottleneck repeat count (post-depth-multiplier).</summary>
    public C2f(int inChannels, int outChannels, int n, bool shortcut, float expansion = 0.5f)
    {
        if (n < 1)
            throw new ArgumentOutOfRangeException(nameof(n), n, "C2f requires at least one bottleneck.");
        _inChannels = inChannels;
        _outChannels = outChannels;
        _hiddenChannels = (int)(outChannels * expansion);
        _numBottlenecks = n;
        _shortcut = shortcut;

        _cv1 = new ConvBnSilu(2 * _hiddenChannels, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: true);
        _cv2 = new ConvBnSilu(outChannels, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: true);
        _bottlenecks = new Bottleneck[n];
        for (int i = 0; i < n; i++)
            _bottlenecks[i] = new Bottleneck(_hiddenChannels, _hiddenChannels, shortcut);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _cv1.LoadWeights(weights, $"{prefix}.cv1.conv");
        _cv2.LoadWeights(weights, $"{prefix}.cv2.conv");
        for (int i = 0; i < _bottlenecks.Length; i++)
            _bottlenecks[i].LoadWeights(weights, $"{prefix}.m.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _cv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv2.EnumerateWeights()) yield return t;
        for (int i = 0; i < _bottlenecks.Length; i++)
            foreach (Tensor t in _bottlenecks[i].EnumerateWeights()) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];

        // 1×1 expand to 2c channels.
        Tensor expanded = _cv1.Forward(backend, input);

        // Split into two halves along the channel dim. Each half has _hiddenChannels channels.
        // After the split we'll have (2 + n) tensors in flight before the final concat: the two
        // halves plus n bottleneck outputs. Each is an owning Tensor and must be disposed.
        Tensor[] accumulated = new Tensor[2 + _numBottlenecks];
        TensorShape halfShape = new TensorShape(batch, _hiddenChannels, height, width);
        accumulated[0] = new Tensor(halfShape, DType.F32);
        accumulated[1] = new Tensor(halfShape, DType.F32);
        backend.Split([accumulated[0], accumulated[1]], expanded, dim: 1);
        expanded.Dispose();

        // Chain bottlenecks on the second half. accumulated[1] is the input to b0, accumulated[2]
        // is the output of b0 / input to b1, and so on.
        for (int i = 0; i < _numBottlenecks; i++)
            accumulated[2 + i] = _bottlenecks[i].Forward(backend, accumulated[1 + i]);

        // Concat all (2 + n) accumulated tensors along the channel dim.
        TensorShape concatShape = new TensorShape(batch, (2 + _numBottlenecks) * _hiddenChannels, height, width);
        Tensor concatenated = new Tensor(concatShape, DType.F32);
        backend.Concat(concatenated, accumulated, dim: 1);
        for (int i = 0; i < accumulated.Length; i++)
            accumulated[i].Dispose();

        // 1×1 compress to outChannels.
        Tensor output = _cv2.Forward(backend, concatenated);
        concatenated.Dispose();
        return output;
    }
}
